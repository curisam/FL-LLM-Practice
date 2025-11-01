# FedBiscuit/federatedscope/core/workers/client.py
# -*- coding: utf-8 -*-

import copy
import logging
import sys
import pickle

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    calculate_time_cost, add_prefix_to_path, get_ds_rank
from federatedscope.core.workers.base_client import BaseClient

import os
from uuid import uuid4


"""
각 모듈(client.py 등): 자기 모듈 전용 로거 객체 생성 → main.py 설정을 자동으로 따라감.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
→ 덕분에 로그를 찍을 때 어느 모듈에서 나왔는지 구분 가능.

"""

logger = logging.getLogger(__name__)
# if get_ds_rank() == 0:
#     logger.setLevel(logging.INFO)

logger.setLevel(logging.DEBUG)

import os
import json

from collections import Counter
import itertools
import inspect
import time

def _summarize_content(msg):
    c = msg.content
    try:
        if msg.msg_type == 'model_para':
            if isinstance(c, tuple) and len(c) == 2:
                n, mp = c
                keys = list(mp.keys()) if hasattr(mp, 'keys') else []
                return f"(sample={n}, param_keys={len(keys)})"
        elif msg.msg_type == 'metrics':
            if isinstance(c, dict):
                ks = sorted(list(c.keys()))
                # 키 전부 출력(개수 적음). 길면 잘려도 무방.
                return f"keys={ks}"
        return type(c).__name__
    except Exception as e:
        return f"<summ err: {e}>"

def _dump_queue_snapshot(comm_mgr, tag:str, drain:bool=False, max_tail:int=10, filter_state=None):
    q = comm_mgr.comm_queue  # deque
    items = list(q)
    vis = [m for m in items if (filter_state is None or m.state == filter_state)]
    print(f"[queue@{tag}] len={len(items)} (vis={len(vis)} filter_state={filter_state})")
    kinds = Counter((m.msg_type, m.state, tuple(m.receiver)) for m in vis)
    print("  summary:", dict(kinds))
    tail = list(itertools.islice(vis, max(0, len(vis)-max_tail), len(vis)))
    for i, m in enumerate(tail, 1):
        print(f"  tail[-{len(tail)-i+1}]: "
              f"type={m.msg_type} state={m.state} sender={m.sender} recv={m.receiver} "
              f"ts={getattr(m,'timestamp',None)} content={_summarize_content(m)}")

    if drain:
        print("  [drain] dumping ALL (arrival order):")
        for idx, m in enumerate(items):
            print(f"    {idx:03d}: type={m.msg_type} state={m.state} sender={m.sender} recv={m.receiver} "
                  f"content={_summarize_content(m)}")
        print("  [drain] (not removing; snapshot only)")


def _wrap_comm_send_debug(comm_mgr):
    if getattr(comm_mgr, "_debug_wrapped", False):
        return
    orig = comm_mgr.send
    def wrapped(msg):
        caller = inspect.stack()[1]
        where = f"{caller.function}@{caller.filename.split('/')[-1]}:{caller.lineno}"
        print(f"[SEND] t={time.time():.3f} from={where} "
              f"type={msg.msg_type} state={msg.state} sender={msg.sender} -> recv={msg.receiver} "
              f"content={_summarize_content(msg)} (before_len={len(comm_mgr.comm_queue)})")
        return orig(msg)
    comm_mgr.send = wrapped
    comm_mgr._debug_wrapped = True












class Client(BaseClient):
    """
    The Client class, which describes the behaviors of client in an FL \
    course. The behaviors are described by the handling functions (named as \
    ``callback_funcs_for_xxx``)

    Arguments:
        ID: The unique ID of the client, which is assigned by the server
        when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: The configuration
        data: The data owned by the client
        model: The model maintained locally
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        ss_manager: secret sharing manager
        msg_buffer: dict buffer for storing message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``

    ---------------------------------------------------------------
    🔧 이 버전에서의 주요 커스터마이징(한글 주석 유지/추가)
    - 메인 프로세스에서만 파일 기록 / 서버 전송 → 중복 방지
    - eval_results.raw: test_*와 val_*가 모두 있을 때만 1회 기록
    - eval.best_res_update_round_wise_key 가 없을 경우 안전한 대체 키 자동 선택
    - evaluate()는 항상 metrics를 정의(초기화)하고 예외 보호
    - train()/evaluate() 시작 전 해당 split 캐시를 리셋(라운드 간 누적 방지)
    - 로깅: 로컬(per-rank), per-proc 리스트(rank0), 집계본(rank0) 3단 출력
    ---------------------------------------------------------------
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None, # ClientData 클래스
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(Client, self).__init__(ID, state, config, model, strategy)

        self.data = data  # ClientData 클래스

        # Register message handlers
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        # the unseen_client indicates that whether this client contributes to
        # FL process by training on its local data and uploading the local
        # model update, which is useful for check the participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.is_unseen_client = is_unseen_client

        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):  # True, -1
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attacker_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        # Attack only support the stand alone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'   # False

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker, #False
                                   monitor=self._monitor)  # federatedscope.llm.trainer.reward_choice_trainer.RewardChoiceTrainer
        

        self.trainer.ctx.client_ID = int(self.ID)   # ← 클라이언트 ID를 컨텍스트에 심기
        self.trainer.ctx.role = "client"

        self.device = device

        # 🔒 outdir 강제: sub_exp/rank-* 제거하고 상위 exp 폴더 하나만 쓰기


        self._force_single_outdir()



        # in local or global training mode, we do use the early stopper.
        # Otherwise, we set patience=0 to deactivate the local early-stopper
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in [
                "local", "global"
            ] else 0   # 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)  # self._cfg.early_stop.improve_indicator_mode='best'.  #self._monitor.the_larger_the_better: False ->loss의 경우 클수록 안좋은 거라서.

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None  # None

        self.msg_buffer = {'train': dict(), 'eval': dict()} #내부 비밀 셰어링(ss) 조각 수집이나, (이 예시에서는) 거의 쓰이지 않는 train/eval 키만 가집니다. 의미 없음.

        # Communication and communication ability
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None:  # PASS
            self.comp_speed = float(
                kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(
                kwargs['resource_info']['communication'])  # (kbit/s)
        else:  # 여기 걸림
            self.comp_speed = None
            self.comm_bandwidth = None

        if self._cfg.backend == 'torch':  # 여기 걸림
            try:
                self.model_size = sys.getsizeof(pickle.dumps(
                    self.model)) / 1024.0 * 8.  # kbits
            except Exception as error:
                self.model_size = 1.0
                logger.warning(f'{error} in calculate model size.')
        else:
            # TODO: calculate model size for TF Model
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:'
                           f'{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id
        

        # 큐를 이용해 서버·클라이언트 간 메시지 송수신을 담당할 CommManager 인스턴스를 생성
        comm_queue = kwargs.get('shared_comm_queue')  # deque([]). # Runner가 주입해 주는 deque.


        # self.comm_manager:  send() / receive() API를 제공하면서, 내부적으로는 comm_queue.append(msg)와 comm_queue.popleft()를 호출해 메시지를 주고받게 합니다.
        if self._cfg.federate.process_num <= 1:  # 단일 프로세스 시뮬레이션용
            self.comm_manager = StandaloneCommManager(
                comm_queue=comm_queue, monitor=self._monitor)
        else: # DDP 기반 병렬 시뮬레이션용
            self.comm_manager = StandaloneDDPCommManager(
                comm_queue=comm_queue, monitor=self._monitor)
        self.local_address = None #gRPC로 넘어갈 때 클라이언트 자신의 네트워크 주소(host: port)를 담는 필드. standalone에서는 쓰이지 않음.

        self.logger = logger


        ###############################################################################################################################################################

        # === 중복 기록 방지 마커들 ===. 평가(eval) 또는 학습(train) 결과가 동일 라운드에 중복 기록되는 것 방지. 멀티프로세스/재시도/콜백 중복 호출 같은 상황에서 같은 (클라ID, 라운드) 결과를 두 번 파일에 쓰지 않도록 가드.
        self._eval_written_marker = set()   # {(client_id, round)}
        self._train_written_marker = set()  # 필요시 train에도 사용

        # 이미 있는 초기화들 뒤에 추가.  메시지 핸들러(idempotency) 용. 동일 라운드의 같은 타입 메시지가 두 번 들어와도 한 번만 처리하게끔. 
        # callback_funcs_for_train/evaluate 같은 핸들러 진입 시, (round in handled_set) 체크 → 이미 처리했으면 바로 리턴.
        self._handled_rounds = {
            'model_para': set(),   # 학습(훈련) 지시 메시지 처리한 라운드 기록
            'evaluate': set(),     # 평가 지시 메시지 처리한 라운드 기록
        }

        ###############################################################################################################################################################


    def _force_single_outdir(self):
        """
        accelerate가 있으면 rank-별 하위 폴더(sub_exp/rank-*)에 로그가 쌓이는데, 이를 상위 공용 폴더(exp/<expname>/<run>/)로 모읍니다.
        즉, 여러 rank가 동시에 쓸 때 결과 파일이 흩어지지 않고 한 폴더에만 1벌 생성되도록 정리하는 동작입니다.
        
        """


        try:
            outdir = getattr(self._monitor, "outdir", None)  #'exp/tldr/choice_qwen/fedbis_test'
            if not outdir:
                return
            norm = os.path.normpath(outdir)
            parts = norm.split(os.sep)
            if "sub_exp" in parts: #NO
                idx = parts.index("sub_exp")
                fixed = os.sep.join(parts[:idx])  # sub_exp 앞까지만
                if fixed:
                    self._monitor.outdir = fixed
                    os.makedirs(self._monitor.outdir, exist_ok=True)
                    self.logger.info(f"[outdir-fixed] monitor.outdir -> {self._monitor.outdir}")
        except Exception as e:
            self.logger.debug(f"[outdir-fixed] skip due to: {e}")


    # ------------------------------------------------------------------
    # 파일 아웃디렉토리 보장 / JSONL append 유틸 / (rank별) 로깅 유틸
    # ------------------------------------------------------------------

    def _ensure_outdir(self):
        if hasattr(self._monitor, "outdir") and self._monitor.outdir:
            # ✅ outdir 생성 직전 한 번 더 상위 폴더로 고정
            self._force_single_outdir()
            os.makedirs(self._monitor.outdir, exist_ok=True)

    # def _is_main_process(self) -> bool: # accelerate가 있으면 그걸로, 없으면 단일 프로세스 가정 
    #     trainer = getattr(self, 'trainer', None) 
    #     if trainer is not None and hasattr(trainer, 'accelerator') and trainer.accelerator is not None: 
    #         return trainer.accelerator.is_main_process 
        
    #     return True # accelerator 미사용 시 단일 프로세스이므로 True

    def _is_main_process(self) -> bool:
        """
        랭크0만 True.
        우선순위:
        1) accelerate 살아있으면 acc.is_main_process
        2) 환경변수 RANK/PMI_RANK/SLURM_PROCID == 0
        3) torch.distributed 초기화돼 있으면 dist.get_rank() == 0
        4) 그 외(진짜 단일 프로세스)만 True
        """
        # 1) accelerator가 살아있으면 그 값 사용
        trainer = getattr(self, 'trainer', None)
        acc = getattr(trainer, 'accelerator', None)
        if acc is not None:
            try:
                return bool(acc.is_main_process)
            except Exception:
                pass  # 비정상 상태면 아래 fallback으로

        # 2) 환경변수 기반( torchrun / SLURM / Intel MPI 등 공통 )
        import os
        for k in ("RANK", "PMI_RANK", "SLURM_PROCID"):
            v = os.environ.get(k)
            if v is not None:
                try:
                    return int(v) == 0  # 글로벌 랭크가 0이면 메인
                except ValueError:
                    break  # 형식 이상 시 다음 단계로

        # 3) torch.distributed가 살아있다면 그 랭크 사용
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass

        # 4) 진짜로 분산 힌트가 전혀 없으면 단일 프로세스로 간주
        return True


    def _append_raw_line(self, role_str: str, round_idx, results_dict: dict, filename: str):

        """
        함수 정의:

            role_str: "Client #1", "Server" 등 역할 문자열

            round_idx: 현재 라운드 번호

            results_dict: 기록하려는 결과(예: train_loss, val_acc 등)

            filename: 기록할 파일명 (예: eval_results.raw)        
        """

        """
        목표:

        outdir (self._monitor.outdir)/filename 에 JSON 라인 한 줄 append
        results_dict: {'train_*'...} 또는 {'test_*','val_*'...} 같은 집계 키만 넘기세요.
        """

        try:
            outdir = getattr(self._monitor, "outdir", None)
            os.makedirs(outdir, exist_ok=True) #해당 디렉토리가 없으면 생성 (exist_ok=True → 이미 있으면 무시).
            outpath = os.path.join(outdir, filename) # 최종 파일 경루(outpath) 생성.

            line = {
                "Role": role_str,
                "Round": round_idx,
                "Results_raw": results_dict
            }#기록할 JSON line 딕셔너리 구성.

            #파일을 append 모드("a")로 열고 UTF-8로 인코딩. json.dumps로 문자열 변환 후 \n 붙여서 한 줄 단위로 기록. ensure_ascii=False → 한글 같은 비ASCII 문자도 그대로 기록
            with open(outpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            # 파일 문제로 학습이 죽지 않도록 방어
            self.logger.warning(f"[append_raw_line] failed to write {filename}: {e}")

    def _log_split_metrics(self, role_str, round_idx, split, trainer_ctx):

        """
        함수는 각 라운드 학습/평가 결과를 로그로 남기는 유틸리티.

        role_str: "Client #1", "Server" 같은 역할 이름

        round_idx: 현재 라운드 번호

        split: 'train' | 'val' | 'test' 중 하나

        trainer_ctx: trainer.ctx 객체, 내부에 로컬/집계 metric들이 들어 있음

        """

        """
        ① 각 rank 로컬 스냅샷
        ② rank0에서 per-proc 리스트 (모든 rank의 local 스냅샷 모음)
        ③ rank0에서 해당 split의 집계 결과(접두사 'split_')
        """

        #1) accelerator 찾기 (client, trainer.ctx, trainer 순으로 시도), self.trainer.accelerator에 보통 지정되어 있음.
        accel = getattr(self, "accelerator", None)
        if accel is None:
            accel = getattr(trainer_ctx, "accelerator", None)
        if accel is None and hasattr(self, "trainer") and hasattr(self.trainer, "accelerator"):
            accel = self.trainer.accelerator


        #2) 랭크/월드 추정 (accelerator 없을 때 env로 보완). world와 rank의 기본값은 1,0.
        if accel is not None:
            world = getattr(accel, "num_processes", 1)
            rank  = getattr(accel, "process_index", 0)
        else:
            world = int(os.getenv("WORLD_SIZE", "1"))
            rank  = int(os.getenv("LOCAL_RANK", "0"))



        #3) 각 rank 로컬 스냅샷.  터미널에는 모든 rank에 대해서 다 뜬다. exp_print.log에는 rank 0에 대해서만 기록.
        local = getattr(trainer_ctx, "local_results_for_log", {}) or {} #local_results_for_log: 각 rank(프로세스)가 개별적으로 측정한 결과.
        self.logger.info({
            'Role': role_str, 'Round': round_idx, 'Split': split,
            'Rank': f'{rank}/{world}', 'Local': True,
            'Results': local
        })#INFO: {'Role': 'Client #44', 'Round': 0, 'Split': 'train', 'Rank': '0/4', 'Local': True, 'Results': {'train_total': 120, 'train_loss': 84.59094136953354, 'train_avg_loss': 0.7049245114127795, 'train_seen': 120, 'train_correct': 67, 'train_acc': 0.5583333333333333}}
        # ③ 집계 결과(해당 split 접두사 4키만; alias 금지)
        agg = getattr(trainer_ctx, "eval_metrics", {}) or {}
        sp  = f"{split}_"
        agg_clean = {k: v for k, v in agg.items() if k.startswith(sp)} #sp에 해당하는 결과만 추출

        # 집계 결과는 오직 rank0에서만 출력. subprocess에서는 agg가 빈 {} 이기에/
        if agg_clean and rank == 0:
            self.logger.info({
                'Role': role_str, 'Round': round_idx, 'Split': split,
                'Aggregated': True, 'Results_raw': agg_clean
            })#INFO: {'Role': 'Client #44', 'Round': 0, 'Split': 'train', 'Aggregated': True, 'Results_raw': {'train_total': 480, 'train_loss': 348.3512268066406, 'train_avg_loss': 0.7257317225138347, 'train_seen': 480, 'train_correct': 234, 'train_acc': 0.4875}}
 

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth) #self.comp_speed=self.comp_bandwidth=None인 상황. 따라서  comp_cost=comm_cost=0으로 나옴.
        return init_timestamp + comp_cost + comm_cost #init_timestamp으로 나옴.

    def _reset_ctx_split_metrics(self, split: str):
        """
        ✅ (중요) 라운드 시작 전에 해당 split의 캐시를 리셋
        - trainer.ctx.eval_metrics, trainer.ctx.local_results_for_log 등에서 split 접두사의 키를 제거.
        - DDP 상황에서 라운드 간 누적되어 train_total 이 960처럼 불어나는 현상 방지.
        """
        try:
            ctx = getattr(self.trainer, "ctx", None)
            if ctx is None:
                return

            # 1) 집계본에서 split 접두사 키 삭제
            em = getattr(ctx, "eval_metrics", None)
            if isinstance(em, dict):
                for k in list(em.keys()):
                    if k.startswith(f"{split}_"):
                        del em[k]

            # 2) 로컬 스냅샷에서 split 접두사 키 삭제
            lr = getattr(ctx, "local_results_for_log", None)
            if isinstance(lr, dict):
                for k in list(lr.keys()):
                    if k.startswith(f"{split}_"):
                        del lr[k]

        except Exception as e:
            self.logger.debug(f"[reset split={split}] skip due to: {e}")

    # ------------------------------------------------------------------
    # 러닝 루프
    # ------------------------------------------------------------------

    def join_in(self): #client:"join_in" 메시지를 서버에게 보냄.
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self): #안쓰임.
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def run_standalone(self): #안쓰임.
        """
        Run in standalone mode
        """
        self.join_in()
        self.run()

    # ------------------------------------------------------------------
    # 서버로부터 받은 글로벌 모델 파라미터를 처리하고,
    # 로컬 학습을 트리거한 뒤 결과를 서버에 다시 보냅니다.
    # ------------------------------------------------------------------
    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """

        # ---- 여기부터 일반(비-SS) 케이스 ----
        round = message.state  # 서버가 보낸 라운드 번호
        sender = message.sender  # 메시지를 보낸 주체(서버) ID
        timestamp = message.timestamp  # 서버 시각
        content = message.content  # 실제 모델 파라미터 (또는 파라미터 델타)

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)


        is_main = self._is_main_process()

        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)  # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다.

        self.state = round






        # 이미 로컬 수렴한 클라이언트라면 훈련을 건너뛰기도 하고…
        if self.early_stopper.early_stopped and \
                self._monitor.local_convergence_round == 0: #self.early_stopper.early_stopped가  False라 Pass!!
            logger.info(
                f"[Normal FL Mode] Client #{self.ID} has been locally "
                f"early stopped. "
                f"The next FL update may result in negative effect")
            self._monitor.local_converged()  # self._monitor의 self.local_convergence_wall_time, self.local_convergence_round 지정.

        # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
        self._reset_ctx_split_metrics('train')

        sample_size, model_para_all, results = self.trainer.train(round_num=self.state)  # 전체 train data 갯수, model_para_all (adapter 1개인 상황이라 이것은 active adapter의 state_dict), split routine 돌며 집계된 results (num_total, loss, acc 등) 리턴

        if self._cfg.federate.share_local_model and not self._cfg.federate.online_aggr:
            model_para_all = copy.deepcopy(model_para_all)  # 안전하게 복사

        # ✅ 랭크별 train result 로그 띄움 2)모들 랭크 종합한 train result 로그 띄움.
        self._log_split_metrics(
            role_str=f'Client #{self.ID}',
            round_idx=self.state,
            split='train',
            trainer_ctx=self.trainer.ctx
        )

        # train 끝나고 _log_split_metrics(...) 뒤:     
 
        ctx = getattr(self.trainer, "ctx", None)
        agg = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}
        train_agg = {k: v for k, v in (agg or {}).items() if k.startswith("train_")}#한 클라이언트 기준 집계된 train split 결과.
        if is_main: # ✅ rank0(=main process)에서만 집계본을 파일로 기록
            self._append_raw_line(
                role_str=f"Client #{self.ID}",
                round_idx=self.state,
                results_dict=train_agg,
                filename="train_results.raw"
            )

        

        train_agg = {
            k: v for k, v in getattr(self.trainer.ctx, "eval_metrics", {}).items()
            if k.startswith("train_")
        }
        if is_main: # ✅ exp_print용: train 집계본만 한 줄 (rank0에서만)
            self.logger.info({
                'Role': f'Client #{self.ID}',
                'Round': self.state,
                'Results_raw': train_agg
            }) #INFO: {'Role': 'Client #44', 'Round': 0, 'Results_raw': {'train_total': 480, 'train_loss': 348.3512268066406, 'train_avg_loss': 0.7257317225138347, 'train_seen': 480, 'train_correct': 234, 'train_acc': 0.4875}}
 
        # Return the feedbacks to the server after local update

        shared_model_para = model_para_all

        # ✅ 모든 rank에 대해서 동일하게 메시지를 보냄. 서버는 각 process마다 동일하게 self.comm_manager.comm_queue를 관리해야함.
        self.comm_manager.send(
            Message(msg_type='model_para',  # ↔ 서버가 “train” 단계로 인식
                    sender=self.ID,  # 이 클라이언트 ID
                    receiver=[sender],  # 앞서 저장한 서버 ID
                    state=self.state,  # (같은) 라운드 번호
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),  # → 서버의 time-based staleness 제어용
                    content=(sample_size, shared_model_para)))  # 데이터 갯수 및 로컬 모델을 content로 담아서 보낸다.    




 
    # ------------------------------------------------------------------
    # 서버가 “평가 요청(evaluate)” 메시지를 보낼 때,
    # 로컬 데이터에 대해 평가를 수행하고 결과(metrics)를 서버에 보냅니다.
    # ------------------------------------------------------------------






    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating
        """

        # 항상 먼저 초기화해서 NameError 방지
 

        metrics = {}
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state


        is_main = self._is_main_process()

        # 1) 서버 파라미터로 동기화
        if message.content is not None:
            self.trainer.update(
                message.content,
                strict=self._cfg.federate.share_local_model
            )

        # 2) 평가 실행
        try:
            # ✅ (중요) 이번 라운드 평가 시작 전, 요청된 split 캐시를 선제 리셋
            #    (val → test 순서로 돌더라도 라운드 간/스플릿 간 누적 방지)
            for sp in set(self._cfg.eval.split): #['val', 'test']
                self._reset_ctx_split_metrics(sp)

            if self._cfg.finetune.before_eval: #False. PFL에서는 True로 해도 될듯.
                self.trainer.finetune()

            for split in self._cfg.eval.split:  #['val', 'test']
                # logger.info(f"[DEBUG] after evaluate(split={split}): eval_metrics={eval_metrics}")
                # eval_metrics = self.trainer.evaluate(target_data_split_name=split)
                eval_metrics = self.trainer.evaluate(target_data_split_name=split)
                """
                    eval_metrics = {
                    f'{split}_total':    int(total_all),
                    f'{split}_loss':     float(loss_all),
                    f'{split}_avg_loss': float(loss_all / total_all),
                    f'{split}_seen':     int(seen_all),
                    f'{split}_correct':  int(correct_all),
                    f'{split}_acc':      float(correct_all / max(1, seen_all))
                    } 

                    의 형태.    
                """

                if is_main:
                    logger.info(f"[DEBUG][after {split}] eval_metrics            = {eval_metrics}")
                    logger.info(f"[DEBUG][after {split}] metrics (merged so far) = {metrics}")
                    logger.info(f"[DEBUG][after {split}] ctx.eval_metrics        = {self.trainer.ctx.eval_metrics}")



                metrics.update(**eval_metrics)


                # ✅ 원하는 포맷의 로그 3종(로컬 / per-proc / 집계) 출력
                self._log_split_metrics(
                    role_str=f'Client #{self.ID}',
                    round_idx=self.state,
                    split=split,
                    trainer_ctx=self.trainer.ctx
                )

        except Exception as e:
            # 평가 중 문제가 나도 서버로는 빈 dict라도 보내도록
            self.logger.warning(f"[evaluate] exception during evaluation: {e}", exc_info=True)



          

        # 우선순위: trainer.ctx.eval_metrics(집계본) -> metrics(병합본)
        ctx = getattr(self.trainer, "ctx", None)
        agg_all = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}

        logger.info(f"[DEBUG][before write] agg_all={agg_all}, metrics={metrics}") 

        base = {**agg_all, **metrics}

        # test_/val_만 추출
        combined = {k: v for k, v in (base or {}).items()
                    if k.startswith("test_") or k.startswith("val_")}
        has_test = any(k.startswith("test_") for k in combined) #Boolean
        has_val  = any(k.startswith("val_") for k in combined) #Boolean

        logger.info(f"[DEBUG] combined keys={list(combined.keys())}, has_val={has_val}, has_test={has_test}")


        write_key = (self.ID, int(self.state))

        # 3) rank0만 파일 기록/서버 전송
        if is_main:

            self._ensure_outdir()


            # ✅ 파일 기록: test & val 둘 다 있고, 아직 안 쓴 경우 1회만
            if has_test and has_val:
                if write_key not in self._eval_written_marker:
                    self._append_raw_line(
                        role_str=f"Client #{self.ID}",
                        round_idx=self.state,
                        results_dict=combined,
                        filename="eval_results.raw"
                    )
                    self._eval_written_marker.add(write_key)
                else:
                    self.logger.debug(f"[skip duplicate eval write] {write_key}")
            else:
                self.logger.debug(
                    f"[skip write eval_results.raw] only one split present. "
                    f"available={list(combined.keys())}"
                )


            # 3-2) 모니터에 기록. round_formatted_results_raw 반환하는 것.
            try:
                self._monitor.format_eval_res(
                    metrics, rnd=self.state, role=f'Client #{self.ID}', forms=['log']
                )
            except Exception as e:
                # metrics가 비어도 로깅 실패하지 않도록
                self.logger.debug(f"[format_eval_res] skip log due to: {e}")

        #4) 모든 RANK에서 서버로  메시지 전송
        if metrics:    # ← metrics 가 비어있지 않을 때만 전송

            # _seen, _correct 접미사 키 제거한 사본 만들기
            pruned_metrics = {
                k: v for k, v in metrics.items()
                if not (k.endswith('_seen') or k.endswith('_correct'))
            }

            # 비워졌으면 굳이 보내지 않음
            if pruned_metrics:
                self.comm_manager.send(
                        Message(msg_type='metrics',
                                sender=self.ID,
                                receiver=[sender],
                                state=self.state,
                                timestamp=timestamp,
                                content=pruned_metrics)
                    )
        else:
            logger.debug(f"[skip send metrics] empty metrics for Client #{self.ID}, round={self.state}")







    # FL 과정이 “완료(finish)” 신호를 받을 때, 최종 모델을 로드하고 로컬 모니터에 종료를 알립니다.
    def callback_funcs_for_finish(self, message: Message):
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")
        
        self._monitor.finish_fl()
        return

        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
            
        # ✅ rank0만 종료 파일 기록(시스템 메트릭 등)
        if self._is_main_process():
            self._monitor.finish_fl() #self.fl_end_wall_time 계산. system_metrics을 얻어내서 "system_metrics.log"에 기록.



    def _calculate_model_delta(self, init_model, updated_model): # 쓸 일 없음.
        if not isinstance(init_model, list):
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            model_delta = copy.deepcopy(init_model[model_index])
            for key in init_model[model_index].keys():
                model_delta[key] = updated_model[model_index][
                    key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        else:
            return model_deltas[0]



    # 볼 필요 없을 듯. 분산 모드에서 서버가 부여한 클라이언트 ID (assign_client_id)를 받아 self.ID 에 설정합니다.
    def callback_funcs_for_assign_id(self, message: Message):
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.

        Arguments:
            message: The received message
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    # 볼 필요 없을 듯. 서버가 “참가 정보”를 요청할 때(batch size, 샘플 개수, 리소스 등), 로컬 설정을 읽어 채워서 응답합니다.
    def callback_funcs_for_join_in_info(self, message: Message):
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    num_sample = self._cfg.train.local_update_steps * \
                                 self._cfg.dataloader.batch_size
                else:
                    num_sample = self._cfg.train.local_update_steps * \
                                 len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = \
                        self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                assert self.comm_bandwidth is not None and self.comp_speed \
                       is not None, "The requirement join_in_info " \
                                    "'client_resource' does not exist."
                join_in_info['client_resource'] = self.model_size / \
                    self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    # 볼 필요 없을 듯. (비밀 셰어링 등) 복잡한 토폴로지를 위해 서버가 다른 클라이언트 주소 목록을 보낼 때 처리합니다. 볼 필요 없을 듯.
    def callback_funcs_for_address(self, message: Message):
        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology

        Arguments:
            message: The received message
        """
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)


    # 볼 필요 없을 듯. 서버가 “조기 수렴(converged)” 신호를 보냈을 때, 로컬 모니터를 통해 더 이상의 업데이트를 멈춥니다.
    def callback_funcs_for_converged(self, message: Message):
        """
        The handling function for receiving the signal that the FL course \
        converged

        Arguments:
            message: The received message
        """
        self._monitor.global_converged() #self.global_convergence_wall_time , self.local_convergence_round 기록.

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
