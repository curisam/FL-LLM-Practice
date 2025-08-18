# federatedscope/core/monitors/monitor.py
import copy
import json
import logging
import os
import gzip
import shutil
import datetime
from collections import defaultdict
from importlib import import_module
import time

import numpy as np

from federatedscope.core.auxiliaries.logging import logline_2_wandb_dict
from federatedscope.core.monitors.metric_calculator import MetricCalculator

try:
    import torch
except ImportError:
    torch = None

import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global_all_monitors = []  # used in standalone mode, to merge sys metric results for all workers


# ============================================================================
# [ADD] main/rank0 판단 + outdir 정규화 + 파일로거 설치 유틸
# ============================================================================

_FILE_LOGGER_INSTALLED = False  # 중복 FileHandler 방지


def _is_main_process():
    # torch.distributed → ENV
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
    except Exception:
        pass
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0") or 0)) == 0
    except Exception:
        return True


def _normalize_outdir(path: str) -> str:
    """sub_exp/.. 하위로 내려가면 sub_exp 앞까지만 남겨 단일 폴더만 사용"""
    if not path:
        return path
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if "sub_exp" in parts:
        idx = parts.index("sub_exp")
        return os.sep.join(parts[:idx])
    return norm


def _install_rank0_file_logger_once(outdir: str):
    """rank0에서만 exp_print.log 파일핸들러를 1회 설치"""
    global _FILE_LOGGER_INSTALLED
    if _FILE_LOGGER_INSTALLED:
        return
    if not _is_main_process():
        return
    if not outdir:
        return
    
    #루트 로거를 가져와서 INFO 레벨 이상만 출력하도록 설정

    os.makedirs(outdir, exist_ok=True) #로그 파일을 저장할 디렉토리를 생성 (이미 있으면 무시)
    root = logging.getLogger()      # 루트 로거에 달아서 전체 로그 수집. 방송국 역할
    root.setLevel(logging.INFO)

    # 이미 설치된 파일 핸들러가 있는지 검사->핸들러가 여러 개 붙을 수 있으므로, 중복 설치 방지용 검사
    for h in list(root.handlers):#root.handlers: 루트 로거에 붙은 핸들러 목록
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.basename(getattr(h, 'baseFilename', '')) == "exp_print.log": #로그 파일 이름이 정확히 "exp_print.log"인지 확인
                    _FILE_LOGGER_INSTALLED = True
                    return #이미 설치되어 있으니 함수 종료
            except Exception:
                pass
    #새 파일 핸들러(FileHandler) 를 생성합니다. 녹음기 역할. 터미널도 파일핸들러와 유사한 stream handler. 다만 이것은 default로 설정됨.
    fh = logging.FileHandler(os.path.join(outdir, "exp_print.log"),
                             mode="a", encoding="utf-8", delay=True)#로그를 파일로 저장하겠다는 뜻. 저장할 파일 이름:"exp_print.log". mode="a":기존 파일에 덧붙이기. encoding="utf-8": 한글 포함. delay=True: 실제 로그가 처음 찍힐 때 파일을 열겠다는 최적화.
    
    #로그 메시지의 출력 형식을 지정합니다:
    """
    키워드	설명
    %(asctime)s	로그 시간
    %(name)s	로거 이름
    %(lineno)d	로그가 발생한 소스 코드 라인 번호
    %(levelname)s	로그 레벨(INFO, WARNING, ...)
    %(message)s	실제 로그 내용
    
    예시:
    2025-08-07 20:51:11 (monitor.py:300) INFO: 학습이 종료되었습니다.       
    """
    fmt = logging.Formatter("%(asctime)s (%(name)s:%(lineno)d) %(levelname)s: %(message)s")

    """
    방금 만든 로그 포맷을 핸들러에 지정합니다.

    핸들러도 INFO 레벨 이상만 기록하게 설정합니다.  
    """

    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)


    """
    이 파일 핸들러를 루트 로거에 등록합니다.

    이후 모든 logging.info(...) 등의 로그는 exp_print.log로 저장됩니다.

    """
    root.addHandler(fh)#방송국에 녹음기 연결
    _FILE_LOGGER_INSTALLED = True


class Monitor(object):
    """
    Provide the monitoring functionalities such as formatting the \
    evaluation results into diverse metrics. \
    Besides the prediction related performance, the monitor also can \
    track efficiency related metrics for a worker

    Args:
        cfg: a cfg node object
        monitored_object: object to be monitored

    Attributes:
        log_res_best: best ever seen results
        outdir: output directory
        use_wandb: whether use ``wandb``
        wandb_online_track: whether use ``wandb`` to track online
        monitored_object: object to be monitored
        metric_calculator: metric calculator, /
            see ``core.monitors.metric_calculator``
        round_wise_update_key: key to decide which result of evaluation \
            round is better
    """
    SUPPORTED_FORMS = ['weighted_avg', 'avg', 'fairness', 'raw']

    def __init__(self, cfg, monitored_object=None):
        self.cfg = cfg
        self.log_res_best = {}

        self.outdir = cfg.outdir

        self.use_wandb = cfg.wandb.use
        self.wandb_online_track = cfg.wandb.online_track if cfg.wandb.use \
            else False
        # self.use_tensorboard = cfg.use_tensorboard

        self.monitored_object = monitored_object
        self.metric_calculator = MetricCalculator(cfg.eval.metrics) 
        
        #self.metric_calculator.eval_metric-> loss, acc, avg_loss, total를 key로 갖는 dict -> #{'acc': (<function eval_acc at 0x7f2edf2eb310>, True), 'avg_loss': (<function eval_avg_loss at 0x7f2edf2eb790>, False), 'loss': (<function eval_loss at 0x7f2edf2eb700>, False), 'total': (<function eval_total at 0x7f2edf2eb820>, False)}

        # Obtain the whether the larger the better
        self.round_wise_update_key = cfg.eval.best_res_update_round_wise_key  #test_loss
        update_key = None
        for mode in ['train', 'val', 'test']:
            if mode in self.round_wise_update_key: #test, test_loss만 대응 됨.
                update_key = self.round_wise_update_key.split(f'{mode}_')[1] #'test_loss'를 'test_'로 나눴더니: 앞에는 아무 것도 없어서 '', 뒤에는 'loss' -> loss
        if update_key is None:
            # 안전장치: metrics에 있는 임의의 첫 키를 사용
            # (실전에서는 cfg 설정을 권장)
            update_key = list(self.metric_calculator.eval_metric.keys())[0]

        assert update_key in self.metric_calculator.eval_metric, \
            f'{update_key} not found in metrics.' #update_key는 loss인 상황
        self.the_larger_the_better = self.metric_calculator.eval_metric[update_key][1] #False로 나옴. loss는 크면 안좋은거기때문에

        # =======  efficiency indicators of the worker to be monitored =======
        self.total_model_size = 0 # model size used in the worker, in terms, 모델의 총 파라미터 수 (정수값)
       
        self.flops_per_sample = 0 # average flops for forwarding each data, 샘플 1개를 처리할 때의 평균 FLOPs
        self.flop_count = 0 # used to calculated the running mean for, 누적 FLOPs 측정을 위한 카운트
        self.total_flops = 0 # total computation flops to convergence until, 지금까지 총 연산 FLOPs
        
        self.total_upload_bytes = 0 # total upload space cost in bytes, 지금까지 업로드한 데이터 총량 (bytes 단위), 클라이언트 → 서버로 전송한 모델 업데이트의 크기
        self.total_download_bytes = 0 # total download space cost in bytes, 지금까지 다운로드한 데이터 총량 (bytes 단위), 서버 → 클라이언트로 전송한 모델의 크기


        self.fl_begin_wall_time = datetime.datetime.now() #학습 시작 시간
        
        self.fl_end_wall_time = 0 #학습 종료 시간
        
        self.global_convergence_round = 0 # total fl rounds to convergence, 전체 학습 라운드 중 글로벌 수렴 시점
        self.global_convergence_wall_time = 0 #위 시점까지 경과 시간 (wall time)
        
        self.local_convergence_round = 0 # total fl rounds to convergence, 클라이언트 로컬 수렴 시점
        self.local_convergence_wall_time = 0 #위 시점까지 경과 시간

        if self.wandb_online_track: #False
            global_all_monitors.append(self)
        if self.use_wandb: #False
            try:
                import wandb  # noqa: F401
            except ImportError:
                logger.error("cfg.wandb.use=True but not install the wandb package")
                exit()

        # [추가] rank0만 exp_print.log 파일핸들러 설치 (1회)
        _install_rank0_file_logger_once(self.outdir)

    def eval(self, ctx): #eval_metric의 key들에 있는 것에 대해 측정한 값들 바탕으로 dictionary return. #주요 호출위치: eval 시점
        """
        Evaluates the given context with ``metric_calculator``.
        """
        results = self.metric_calculator.eval(ctx)
        return results

    def global_converged(self): #🔹 목적: 학습 도중 글로벌 수렴(Global convergence)이 이루어진 시점을 기록 #주요 호출위치: early stop시
        """Calculate wall time and round when global convergence has been reached."""
        self.global_convergence_wall_time = datetime.datetime.now() - self.fl_begin_wall_time #FL 과정 전체가 시작된 시점(self.fl_begin_wall_time)으로부터 글로벌 조기 종료가 결정되기까지 경과한 실제 벽시계 시간(wall-clock time)을 계산해서 저장
        #마지막 GFL 라운드 저장.
        self.global_convergence_round = self.monitored_object.state

    def local_converged(self): # 🔹 목적: 클라이언트 로컬 수렴 시점 기록 (서버와는 별개) #주요 호출위치: 클라이언트 종료시
        """Calculate wall time and round when local convergence has been reached."""
        self.local_convergence_wall_time = datetime.datetime.now() - self.fl_begin_wall_time #FL 과정 전체가 시작된 시점(self.fl_begin_wall_time)으로부터 로컬 조기 종료가 결정되기까지 경과한 실제 벽시계 시간(wall-clock time)을 계산해서 저장
        #클라이언트가 마지막으로 수행한 FL 라운드 번호(self.monitored_object.state)를 기록. monitored_object 는 이 경우 BaseClient 혹은 LLMMultiLoRAClient 인스턴스로, 그 .state 가 “현재 학습이 진행된 라운드 수”를 의미. 몇 번째 라운드까지 로컬 학습이 진행되었을 때 멈췄는지
        self.local_convergence_round = self.monitored_object.state

    def finish_fl(self): #🔹 목적: 학습 종료 시, 시스템 지표들을 system_metrics.log 파일에 기록 (단, rank 0에서만 실행) #주요 호출위치: 종료 직전
        """
        When FL finished, write system metrics to file.
        """

        #학습 종료 시간, 총 업로드/다운로드 바이트, 모델 크기 등 기록
        
        self.fl_end_wall_time = datetime.datetime.now() - self.fl_begin_wall_time
        if not _is_main_process():  # ✅ rank0만 파일 기록
            return

        system_metrics = self.get_sys_metrics()
        sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
        os.makedirs(self.outdir, exist_ok=True)
        with open(sys_metric_f_name, "a") as f:
            f.write(json.dumps(system_metrics) + "\n")#한 줄 JSON으로 기록되며, 이후 merge_system_metrics_simulation_mode()에서 이 파일을 줄 단위로 읽어와 평균 계산에 활용한다.

    def get_sys_metrics(self, verbose=True): #🔹 목적: 현재까지 모아진 시스템 메트릭들을 딕셔너리로 반환. 📦 사용처: finish_fl(), merge_system_metrics_simulation_mode()
        system_metrics = {
            "id": self.monitored_object.ID,                                           # Monitor.__init__() 호출 시 계산됨
            "fl_end_time_minutes": self.fl_end_wall_time.total_seconds() / 60         # Monitor.finish_fl() 호출 시 계산됨
            if isinstance(self.fl_end_wall_time, datetime.timedelta) else 0,
            "total_model_size": self.total_model_size, #track_model_size() 호출 시.
            "total_flops": self.total_flops, #학습 중 track_avg_flops() 등 통해 누적 계산
            "total_upload_bytes": self.total_upload_bytes, # CommManager.send() → monitor.track_upload_bytes() 호출 시
            "total_download_bytes": self.total_download_bytes, # 서버가 클라이언트에게 모델을 내려줄 때 추적
            "global_convergence_round": self.global_convergence_round, #global_converged() 호출 시 기록
            "local_convergence_round": self.local_convergence_round, # local_converged() 호출 시 기록
            "global_convergence_time_minutes": self.global_convergence_wall_time.total_seconds() / 60 # global_converged() 호출 시 계산
            if isinstance(self.global_convergence_wall_time, datetime.timedelta) else 0,
            "local_convergence_time_minutes": self.local_convergence_wall_time.total_seconds() / 60 # local_converged() 호출 시 계산
            if isinstance(self.local_convergence_wall_time, datetime.timedelta) else 0,
        }
        if verbose:
            logger.info(
                f"In worker #{self.monitored_object.ID}, the system-related "
                f"metrics are: {str(system_metrics)}")
        return system_metrics

    def merge_system_metrics_simulation_mode(self,
                                             file_io=True,
                                             from_global_monitors=False): # Standalone 모드에서 여러 클라이언트의 시스템 메트릭을 모아 평균과 표준편차를 계산함. 결과를 system_metrics.log에 평균/표준편차 행 추가
        """
        Average the system metrics recorded in ``system_metrics.json`` by all workers
        """

        """
        Standalone 시뮬레이션 모드에서 각 클라이언트가 학습 종료 시 기록한 system_metrics.log를 읽어서

        모든 클라이언트 값의 **평균(sys_avg)**과 **표준편차(sys_std)**를 계산

        다시 로그 파일에 append        
        
        """


        all_sys_metrics = defaultdict(list)#value들은 list. 모든 client의 값을 list로 저장.
        avg_sys_metrics = defaultdict() #일반 dict와 다르게 존재하지 않는 키 접근 시 자동으로 기본값 생성 후 반환.
        std_sys_metrics = defaultdict()

        if file_io: #파일에서 읽기 모드
            if not _is_main_process():   # ✅ rank0만 병합/쓰기. 다른 프로세스가 동시에 같은 파일 쓰는 걸 방지.
                return
            sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
            if not os.path.exists(sys_metric_f_name): #로그 파일 없으면 경고 후 종료
                logger.warning(
                    "You have not tracked the workers' system metrics in "
                    "$outdir$/system_metrics.log, we will skip the merging. "
                    "Plz check whether you do not want to call monitor.finish_fl()")
                return
            with open(sys_metric_f_name, "r") as f: #각 라인 읽어서 누적
                for line in f: #"system_metrics.log"를 라인별로 읽음.
                    """
                    f의 예시.
                    {"id": 1, "total_upload_bytes": 1234, "total_flops": 111}
                    {"id": 2, "total_upload_bytes": 2345, "total_flops": 222}
                    {"id": 3, "total_upload_bytes": 3456, "total_flops": 333}                  
                    """
                    res = json.loads(line)# {"id": 1, "total_upload_bytes": 1234, "total_flops": 111}

                    for k, v in res.items():
                        all_sys_metrics[k].append(v)

                """
                all_sys_metrics = {
                    "id": [1, 2, 3],
                    "total_upload_bytes": [1234, 2345, 3456],
                    "total_flops": [111, 222, 333]
                }
                """
            
            #중복 id 체크
            id_to_be_merged = all_sys_metrics["id"]
            if len(id_to_be_merged) != len(set(id_to_be_merged)):
                logger.warning(
                    f"The sys_metric_file ({sys_metric_f_name}) contains "
                    f"duplicated tracked sys-results with these ids: f{id_to_be_merged} "
                    f"We will skip the merging as the merge is invalid. "
                    f"Plz check whether you specify the 'outdir' "
                    f"as the same as the one of another older experiment.")
                return
        elif from_global_monitors: #메모리에서 읽기. 파일 대신 메모리에 있는  global_all_monitors 리스트에서 시스템 메트릭을 수집.
            for monitor in global_all_monitors:
                res = monitor.get_sys_metrics(verbose=False)

                for k, v in res.items():
                    all_sys_metrics[k].append(v)
        else:
            raise ValueError("file_io or from_monitors should be True: "
                             f"but got file_io={file_io}, from_monitors={from_global_monitors}")

        for k, v in all_sys_metrics.items():
            if k == "id": #"id" 키는 숫자가 아니라 워커 식별자이므로 → "sys_avg" / "sys_std" 라벨로 변경
                avg_sys_metrics[k] = "sys_avg"
                std_sys_metrics[k] = "sys_std"
            else:
                v = np.array(v).astype("float")
                mean_res = np.mean(v)
                std_res = np.std(v)
                if "flops" in k or "bytes" in k or "size" in k: #"flops", "bytes", "size" 포함 시 사람이 읽기 좋은 단위로 변환 (예: 1048576 → "1.0M")
                    mean_res = self.convert_size(mean_res)
                    std_res = self.convert_size(std_res)
                avg_sys_metrics[f"sys_avg/{k}"] = mean_res
                std_sys_metrics[f"sys_std/{k}"] = std_res

        logger.info(f"After merging the system metrics from all works, we got avg: {avg_sys_metrics}")
        logger.info(f"After merging the system metrics from all works, we got std: {std_sys_metrics}")
        if file_io:
            with open(sys_metric_f_name, "a") as f: #평균/표준편차 결과를 같은 로그 파일에 뒤에 두 줄로 추가
                f.write(json.dumps(avg_sys_metrics) + "\n")
                f.write(json.dumps(std_sys_metrics) + "\n")

        if self.use_wandb and self.wandb_online_track: #False
            try:
                import wandb
                for k, v in avg_sys_metrics.items():
                    wandb.summary[k] = v
                for k, v in std_sys_metrics.items():
                    wandb.summary[k] = v
            except ImportError:
                logger.error("cfg.wandb.use=True but not install the wandb package")
                exit()

    def save_formatted_results(self,
                               formatted_res,
                               save_file_name="eval_results.log"): #🔹 목적: 각 라운드에서 포맷팅된 평가 결과를 로그 파일에 저장. 단, rank 0에서만 수행됨. 📦 저장 위치: eval_results.log
        """
        Save formatted results to a file.
        """
        line = str(formatted_res) + "\n" #formatted_res (예: {'Role': 'Client #1', 'Round': 3, 'Results': {...}})를 문자열로 변환해서 한 줄 형태로 만듦.
        if save_file_name != "":
            if _is_main_process():        # ✅ rank0만
                os.makedirs(self.outdir, exist_ok=True)
                with open(os.path.join(self.outdir, save_file_name), "a") as outfile: #eval_results.log (기본) 파일을 append 모드로 열어서 한 줄 추가
                    outfile.write(line)
        if self.use_wandb and self.wandb_online_track: #FALSE.
            try:
                import wandb
                exp_stop_normal = False
                exp_stop_normal, log_res = logline_2_wandb_dict(
                    exp_stop_normal, line, self.log_res_best, raw_out=False)
                wandb.log(log_res)
            except ImportError:
                logger.error("cfg.wandb.use=True but not install the wandb package")
                exit()

    def finish_fed_runner(self, fl_mode=None): #🔹 목적: 전체 실험 종료 시 후처리 수행. 📦 동작: eval_results.raw 압축.  standalone인 경우 system metric merge. 
        """
        Finish the Fed runner.
        """
        self.compress_raw_res_file() #eval_results.raw 압축
        if fl_mode == "standalone":
            self.merge_system_metrics_simulation_mode() #클라이언트들의 평균/std 집계하여 "system_metrics.log"에 추가.

        if self.use_wandb and not self.wandb_online_track: #False
            try:
                import wandb
            except ImportError:
                logger.error("cfg.wandb.use=True but not install the wandb package")
                exit()

            from federatedscope.core.auxiliaries.logging import logfile_2_wandb_dict
            with open(os.path.join(self.outdir, "eval_results.log"), "r") as exp_log_f:
                # track the prediction related performance
                all_log_res, exp_stop_normal, last_line, log_res_best = \
                    logfile_2_wandb_dict(exp_log_f, raw_out=False)
                for log_res in all_log_res:
                    wandb.log(log_res)
                wandb.log(log_res_best)

                # track the system related performance
                sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
                with open(sys_metric_f_name, "r") as f:
                    for line in f:
                        res = json.loads(line)
                        if res["id"] in ["sys_avg", "sys_std"]:
                            for k, v in res.items():
                                wandb.summary[k] = v

    def compress_raw_res_file(self): #🔹 목적: eval_results.raw 파일을 .gz 압축하여 디스크 공간 절약. 🧩 동작: eval_results.raw → eval_results.raw.gz 변환,  기존 파일 삭제
        """
        Compress the raw res file to be written to disk.
        """
        if not _is_main_process():  # ✅ rank0만
            return
        old_f_name = os.path.join(self.outdir, "eval_results.raw")
        if os.path.exists(old_f_name):
            logger.info(
                "We will compress the file eval_results.raw into a .gz file, "
                "and delete the old one")
            with open(old_f_name, 'rb') as f_in:
                with gzip.open(old_f_name + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(old_f_name)


    def format_eval_res(self,
                        results,
                        rnd,
                        role=-1,
                        forms=None,
                        return_raw=False): #🔹 목적: 각 라운드 결과(results)를 여러 형태(avg, weighted_avg, fairness, raw)로 포맷하여 반환 #평가 로그 상황에서 서버, 클라이언트에서 호출.
        #클라이언트: federatedscope/core/workers/client.py (raw 포맷 만들 떄)
        #####  callback_funcs_for_evaluate 혹은 유사한 evaluate 핸들러 내에서 monitor.format_eval_res(results_client, rnd, role=f"Client #{cid}", forms=['raw'], return_raw=True)→ Raw 라인(JSONL)으로 기록하는 용도, round_formatted_results_raw 반환
        # 서버: federatedscope/core/workers/server.py (avg/weighted_avg/fairness 집계 만들 떄). merge_eval_results_from_all_clients method에서 쓰임.
        #####  formatted = monitor.format_eval_res(results_aggregated, rnd, role="Server #", forms=['weighted_avg','avg','fairness','raw']), round_formatted_results 반환
        ##### monitor.save_formatted_results(formatted)->eval_results.log에 한 줄 append
        """
        Format the evaluation results from trainer.ctx.eval_results

        Args:
            results (dict): a dict to store the evaluation results {metric: value}
            rnd (int|string): FL round
            role (int|string): the output role
            forms (list): format type
            return_raw (bool): return either raw results, or other results

        Returns:
            dict: round_formatted_results / round_formatted_results_raw
        """




        # 공통 헤더 구성.
        #forms가 지정 안되면 4가지 모두 계산 대상으로 잡음. 반환 딕셔너리의 공통 헤더(누가, 몇 라운드인지) 세팅.
        if forms is None:
            forms = ['weighted_avg', 'avg', 'fairness', 'raw']
        round_formatted_results = {'Role': role, 'Round': rnd}
        round_formatted_results_raw = {'Role': role, 'Round': rnd}

        if 'group_avg' in forms: #특수 분기
            new_results = {}
            num_of_client_for_data = self.cfg.data.num_of_client_for_data #[]
            client_start_id = 1
            for group_id, num_clients in enumerate(num_of_client_for_data):
                if client_start_id > len(results):
                    break
                group_res = copy.deepcopy(results[client_start_id])
                num_div = num_clients - max(
                    0, client_start_id + num_clients - len(results) - 1)
                for client_id in range(client_start_id,
                                    client_start_id + num_clients):
                    if client_id > len(results):
                        break
                    for k, v in group_res.items():
                        if isinstance(v, dict):
                            for kk in v:
                                if client_id == client_start_id:
                                    group_res[k][kk] /= num_div
                                else:
                                    group_res[k][kk] += results[client_id][k][kk] / num_div
                        else:
                            if client_id == client_start_id:
                                group_res[k] /= num_div
                            else:
                                group_res[k] += results[client_id][k] / num_div
                new_results[group_id + 1] = group_res
                client_start_id += num_clients
                round_formatted_results['Results_group_avg'] = new_results
        else: #일반 케이스: forms 루프
            for form in forms:
                new_results = copy.deepcopy(results)
                if not str(role).lower().startswith('server') or form == 'raw': #클라이언트 역할이거나 raw 포맷이면: 그냥 원본(results)을 Results_raw로 붙이고 끝.
                    round_formatted_results_raw['Results_raw'] = new_results
                elif form not in Monitor.SUPPORTED_FORMS:
                    continue
                else: #서버 역할 + SUPPORTED_FORMS(= ['weighted_avg','avg','fairness','raw'])일 때만 아래 집계 계산을 수행. 여기서 쓰는 results는 **서버가 클라이언트로부터 모아온 각 metric들의 "클라이언트별 결과 리스트"**입니다.
                    for key in results.keys():
                        dataset_name = key.split("_")[0] # 예: "val_acc" → "val"
                        if f'{dataset_name}_total' not in results:
                            raise ValueError(
                                "Results to be formatted should include the dataset_num in the dict, "
                                f"with key = {dataset_name}_total")


                        dataset_num = np.array(results[f'{dataset_name}_total'])

                        # === total / correct는 평균만 계산하고 건너뛰기 ===
                        if key in [f'{dataset_name}_total', f'{dataset_name}_correct']:
                            new_results[key] = np.mean(new_results[key])
                            continue  # 아래 weighted_avg / avg / fairness 계산 건너뜀

                        # === 나머지 metric은 집계 방식에 따라 처리 ===
                        all_res = np.array(copy.copy(results[key])) #서버가 모은 클라이언트별 metric 값의 리스트 (예: 모든 클라의 val_acc 목록).
                        if form == 'weighted_avg': #임의의 key에 대해서 weight avg. train/val/test인지는 앞선 dataset_name에서 결정됨. dataset_num도 이를 바탕으로 결정됐었음. 
                            new_results[key] = np.sum(
                                np.array(new_results[key]) * dataset_num) / np.sum(dataset_num)
                        elif form == "avg":
                            new_results[key] = np.mean(new_results[key])
                        elif form == "fairness" and all_res.size > 1:
                            new_results.pop(key, None)
                            all_res.sort()
                            new_results[f"{key}_std"] = np.std(np.array(all_res)) #표준편차: 클라 간 값의 흩어짐 정도. 값이 클수록 불균형/격차가 큼.
                            new_results[f"{key}_bottom_decile"] = all_res[all_res.size // 10] #하위 10% 지점의 값(10번째 분위값)
                            new_results[f"{key}_top_decile"] = all_res[all_res.size * 9 // 10] #상위 10% 지점의 값(90번째 분위값)
                            new_results[f"{key}_min"] = all_res[0] #분포의 최솟값
                            new_results[f"{key}_max"] = all_res[-1] #분포의 최댓값
                            new_results[f"{key}_bottom10%"] = np.mean(all_res[:all_res.size // 10]) #하위 10% 구간의 평균
                            new_results[f"{key}_top10%"] = np.mean(all_res[all_res.size * 9 // 10:]) #상위 10% 구간의 평균
                            new_results[f"{key}_cos1"] = np.mean(all_res) / (np.sqrt(np.mean(all_res**2))) #벡터 all_res와 모두 같은 값인 벡터(예: 1,1,1,...) 의 “유사도”.
                            all_res_preprocessed = all_res + 1e-9 #안정화를 위한 전처리.
                            new_results[f"{key}_entropy"] = np.sum(
                                -all_res_preprocessed / np.sum(all_res_preprocessed) *
                                (np.log(all_res_preprocessed / np.sum(all_res_preprocessed)))) #분포를 정규화해서 확률처럼 만들고(각 엔트로피 값 / 합) 를 계산. 값이 높을수록 분포가 퍼져 있음(균일에 가까움), 낮을수록 한쪽에 몰림.



                    round_formatted_results[f'Results_{form}'] = new_results

        # ⛔️ 더 이상 여기서 파일을 쓰지 않음 (rank0가 client.py에서 기록)
        return round_formatted_results_raw if return_raw else round_formatted_results




    def calc_model_metric(self, last_model, local_updated_models, rnd): #🔹 목적: 모델 수준의 메트릭(cos_sim, l2_distance, 등)을 계산하여 로깅 #서버 라운드 끝에서 호출. self.cfg.eval.monitoring=[]라 의미 없는 듯.
        """
        Arguments:
            last_model (dict): the state of last round.
            local_updated_models (list): each element is (data_size, model).

        Returns:
            dict: model_metric_dict
        """

        """
        📦 인자:

            last_model: 전 라운드 서버 모델

            local_updated_models: 각 클라이언트의 모델


        📦 결과:
            {'train_cos_sim': 0.97, 'train_l2_dist': 1.23}

        """

        model_metric_dict = {}
        for metric in self.cfg.eval.monitoring: #self.cfg.eval.monitoring=[]
            func_name = f'calc_{metric}'
            calc_metric = getattr(
                import_module('federatedscope.core.monitors.metric_calculator'),
                func_name)
            metric_value = calc_metric(last_model, local_updated_models)
            model_metric_dict[f'train_{metric}'] = metric_value
        formatted_log = {
            'Role': 'Server #',
            'Round': rnd,
            'Results_model_metric': model_metric_dict
        }
        if len(model_metric_dict.keys()):
            logger.info(formatted_log)

        return model_metric_dict

    def convert_size(self, size_bytes): #🔹 목적: 바이트 단위 값을 사람이 보기 쉬운 단위(예: MB, GB)로 변환. 📦 예시: convert_size(1048576) → '1.0M'
        """
        Convert bytes to human-readable size.
        """
        import math
        if size_bytes <= 0:
            return str(size_bytes)
        size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s}{size_name[i]}"

    def track_model_size(self, models): #🔹 목적: 모델의 전체 파라미터 수를 계산 (self.total_model_size에 저장).  📦 호출 시점: 학습 시작 전 (보통 client.py 초기화에서)
        """
        calculate the total model size given the models hold by the worker/trainer
        """
        if self.total_model_size != 0:
            logger.warning(
                "the total_model_size is not zero. You may have been "
                "calculated the total_model_size before")

        if not hasattr(models, '__iter__'):
            models = [models]
        for model in models:
            assert isinstance(model, torch.nn.Module), \
                f"the `model` should be type torch.nn.Module when " \
                f"calculating its size, but got {type(model)}"
            for name, para in model.named_parameters():
                self.total_model_size += para.numel()

    def track_avg_flops(self, flops, sample_num=1): #🔹 목적: 샘플당 평균 FLOPs를 추적. 누적 방식: moving average #학습 중 호출
        """
        update the average flops for forwarding each data sample,
        for most models and tasks,
        the averaging is not needed as the input shape is fixed
        """
        self.flops_per_sample = (self.flops_per_sample * self.flop_count + flops) / (self.flop_count + sample_num)
        self.flop_count += 1

    def track_upload_bytes(self, bytes): #목적: 클라이언트가 서버로 업로드한 모델 크기 누적. # CommManager.send() 내부에서 호출
        """
        Track the number of bytes uploaded.
        """
        self.total_upload_bytes += bytes

    def track_download_bytes(self, bytes): #목적: 서버가 클라이언트로 내려보낸 모델 크기 누적 # CommManager에서 호출
        """
        Track the number of bytes downloaded.
        """
        self.total_download_bytes += bytes

    def update_best_result(self, best_results, new_results, results_type): #🔹 목적: 현재 평가 결과가 기존 best보다 더 낫다면 갱신. 📦 로직 요약: round_wise_update_key에 따라 더 좋은 성능인지 판단 (loss는 작을수록, acc는 클수록) #평가 후에 호출.

        """
        best_results: 지금까지 저장된 best 결과들의 딕셔너리
        예: {"server_best": {"val_loss": 0.52, "val_acc": 0.88}, ...}

        new_results: 현재 라운드에서 나온 평가 결과
        예: {"val_loss": 0.48, "val_acc": 0.86}

        results_type: 현재 업데이트 대상의 키
        예: "server_best", "client_best_individual"

        self.round_wise_update_key: 어떤 지표를 기준으로 best인지 결정 (예: "val_loss")

        self.the_larger_the_better: 해당 지표가 클수록 좋은지 여부 (예: "acc"는 True, "loss"는 False
        
        
        """

        """
        Update best evaluation results.
        by default, the update is based on validation loss with
        ``round_wise_update_key="val_loss" ``
        """
        update_best_this_round = False
        if not isinstance(new_results, dict):
            raise ValueError(
                f"update best results require `results` a dict, but got"
                f" {type(new_results)}")
        else:
            #초기화.
            if results_type not in best_results:
                best_results[results_type] = dict()
            best_result = best_results[results_type]

            # update different keys separately: the best values can be in different rounds
            if self.round_wise_update_key is None: #test_loss라 보통 여기에 안걸림!!
                for key in new_results:
                    cur_result = new_results[key]
                    if 'loss' in key or 'std' in key:  # the smaller, the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = min(cur_result) # 클라이언트  개별 결과면 최소
                        if key not in best_result or cur_result < best_result[key]:
                            best_result[key] = cur_result
                            update_best_this_round = True

                    elif 'acc' in key:  # the larger, the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = max(cur_result) # 클라이언트  개별 결과면 최대
                        if key not in best_result or cur_result > best_result[key]:
                            best_result[key] = cur_result
                            update_best_this_round = True
                    else:
                        # unconcerned metric
                        pass
            # update different keys round-wise: if find better round_wise_update_key,
            # update others at the same time
            else:
                found_round_wise_update_key = False
                sorted_keys = [] #new_results의 key를 집어넣을 예정. 다만 self.round_wise_update_key를 맨 앞으로 할 것!!
                for key in new_results:
                    if self.round_wise_update_key in key: #self.round_wise_update_key: test_loss
                        sorted_keys.insert(0, key) #리스트의 맨 앞에 key를 삽입.
                        found_round_wise_update_key = key
                    else:
                        sorted_keys.append(key)
                if not found_round_wise_update_key:
                    raise ValueError(
                        "Your specified eval.best_res_update_round_wise_key "
                        "is not in target results, use another key or check the name. \n"
                        f"Got eval.best_res_update_round_wise_key={self.round_wise_update_key}, "
                        f"the keys of results are {list(new_results.keys())}")

                # the first key must be the `round_wise_update_key`
                cur_result = new_results[found_round_wise_update_key] #self.round_wise_update_key 관점의 metric 가져옴.

                if self.the_larger_the_better:
                    # The larger, the better
                    if results_type in [
                            "client_best_individual",
                            "unseen_client_best_individual"
                    ]:
                        cur_result = max(cur_result)#client들 결과중 max인걸로 가져옴.
                    if found_round_wise_update_key not in best_result or cur_result > best_result[found_round_wise_update_key]:
                        best_result[found_round_wise_update_key] = cur_result
                        update_best_this_round = True
                else:
                    # The smaller, the better
                    if results_type in [
                            "client_best_individual",
                            "unseen_client_best_individual"
                    ]:
                        cur_result = min(cur_result) #client들 결과중 min인걸로 가져옴.
                    if found_round_wise_update_key not in best_result or cur_result < best_result[found_round_wise_update_key]:
                        best_result[found_round_wise_update_key] = cur_result
                        update_best_this_round = True

                # update other metrics only if update_best_this_round is True
                if update_best_this_round: #이번 라운드때 best가 업데이트 되는 경우.
                    for key in sorted_keys[1:]: #self.round_wise_update_key 제외한 것들에 대해 최신 것으로 업데이트
                        cur_result = new_results[key]
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            # Obtain whether the larger the better
                            for mode in ['train', 'val', 'test']:
                                if mode in key:
                                    _key = key.split(f'{mode}_')[1]
                                    if self.metric_calculator.eval_metric[_key][1]:#lager이면 좋은 것.
                                        cur_result = max(cur_result)
                                    else:#smaller이면 좋은 것.
                                        cur_result = min(cur_result)
                        best_result[key] = cur_result #비교 없이 덮어씀.

        if update_best_this_round:
            line = f"Find new best result: {best_results}"
            logging.info(line) #로그로 남김.
            if self.use_wandb and self.wandb_online_track: #False
                try:
                    import wandb
                    exp_stop_normal = False
                    exp_stop_normal, log_res = logline_2_wandb_dict(
                        exp_stop_normal,
                        line,
                        self.log_res_best,
                        raw_out=False)
                    for k, v in self.log_res_best.items():
                        wandb.summary[k] = v
                except ImportError:
                    logger.error(
                        "cfg.wandb.use=True but not install the wandb package")
                    exit()
        return update_best_this_round #best가 경신되었는지의 여부를 return.

    def add_items_to_best_result(self, best_results, new_results, results_type): #🔹 목적: 특정 평가결과를 best result dict에 단순 추가. 📦 차이점: update_best_result는 비교 후 업데이트. add_items_to_best_result는 비교 없이 그냥 삽입
        #PFL 등 후처리에 호출
        """
        Add a new key: value item (results-type: new_results) to best_result
        """
        best_results[results_type] = new_results
