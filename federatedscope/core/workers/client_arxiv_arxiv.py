# federatedscope/core/workers/client.py
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

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)

import os
import json


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
    🔧 이 버전에서의 주요 커스터마이징
    - 메인 프로세스에서만 파일 기록 / 서버 전송 → 중복 방지
    - eval_results.raw: test_*와 val_*가 모두 있을 때만 1회 기록
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
                 data=None,  # ClientData 클래스
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

        # unseen client 여부
        self.is_unseen_client = is_unseen_client

        # 공격자 여부 파싱
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attacker_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'

        # Build Trainer
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)
        self.device = device

        # 🔒 outdir 강제: sub_exp/rank-* 제거하고 상위 exp 폴더 하나만 쓰기
        self._force_single_outdir()

        # For client-side evaluation
        self.best_results = dict()
        self.history_results = dict()

        # EarlyStopper
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in ["local", "global"] else 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None

        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Communication and communication ability
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None:
            self.comp_speed = float(kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(kwargs['resource_info']['communication'])  # (kbit/s)
        else:
            self.comp_speed = None
            self.comm_bandwidth = None

        if self._cfg.backend == 'torch':
            try:
                self.model_size = sys.getsizeof(pickle.dumps(self.model)) / 1024.0 * 8.  # kbits
            except Exception as error:
                self.model_size = 1.0
                logger.warning(f'{error} in calculate model size.')
        else:
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id

        comm_queue = kwargs.get('shared_comm_queue')
        if self._cfg.federate.process_num <= 1:
            self.comm_manager = StandaloneCommManager(
                comm_queue=comm_queue, monitor=self._monitor)
        else:
            self.comm_manager = StandaloneDDPCommManager(
                comm_queue=comm_queue, monitor=self._monitor)
        self.local_address = None

        self.logger = logger

        # === 중복 기록 방지 마커들 ===
        self._eval_written_marker = set()   # {(client_id, round)}
        self._train_written_marker = set()

        # 이미 있는 초기화들 뒤에 추가
        self._handled_rounds = {
            'model_para': set(),
            'evaluate': set(),
        }

        # --- 비-메인(rank>0) 파일 기록 비활성화 + outdir 통일 ---
        is_main = True
        try:
            if hasattr(self, "trainer") and hasattr(self.trainer, "accelerator") and self.trainer.accelerator is not None:
                is_main = self.trainer.accelerator.is_main_process
            else:
                _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0") or 0))
                is_main = (_rank == 0)
        except Exception:
            is_main = True

        if not is_main:
            try:
                if hasattr(self._monitor, "exp_print_output_stream") and self._monitor.exp_print_output_stream:
                    try:
                        self._monitor.exp_print_output_stream.close()
                    except Exception:
                        pass
                    self._monitor.exp_print_output_stream = None
                if hasattr(self._monitor, "to_file"):
                    self._monitor.to_file = False
            except Exception as e:
                self.logger.debug(f"[monitor-disable-nonmain] skip due to: {e}")

        # outdir이 sub_exp/... 라면 상위로 강제 고정 (최종 한 번 더)
        self._force_single_outdir()

    def _force_single_outdir(self):
        """
        self._monitor.outdir가 sub_exp/rank-*/... 로 내려가면 상위 exp 폴더로 끌어올립니다.
        이렇게 하면 결과 파일은 항상 exp/<expname>/<run>/ 에만 1벌 생깁니다.
        """
        try:
            outdir = getattr(self._monitor, "outdir", None)
            if not outdir:
                return
            norm = os.path.normpath(outdir)
            parts = norm.split(os.sep)
            if "sub_exp" in parts:
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

    def _append_raw_line(self, role_str: str, round_idx, results_dict: dict, filename: str):
        """
        outdir (self._monitor.outdir)/filename 에 JSON 라인 한 줄 append
        results_dict: {'train_*'...} 또는 {'test_*','val_*'...} 같은 집계 키만 넘기세요.
        """
        if not self._is_main_process():
            return

        try:
            outdir = getattr(self._monitor, "outdir", None)
            if not outdir:
                outdir = os.path.join("exp", "default")
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, filename)

            line = {
                "Role": role_str,
                "Round": round_idx,
                "Results_raw": results_dict
            }
            with open(outpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            # 파일 문제로 학습이 죽지 않도록 방어
            self.logger.warning(f"[append_raw_line] failed to write {filename}: {e}")

    def _log_split_metrics(self, role_str, round_idx, split, trainer_ctx):
        """
        split: 'train' | 'val' | 'test'
        trainer_ctx: self.trainer.ctx

        ① 각 rank 로컬 스냅샷
        ② rank0에서 per-proc 리스트 (모든 rank의 local 스냅샷 모음)
        ③ rank0에서 해당 split의 집계 결과(접두사 'split_')
        """
        # accelerator 찾기 (client, trainer.ctx, trainer 순으로 시도)
        accel = getattr(self, "accelerator", None)
        if accel is None:
            accel = getattr(trainer_ctx, "accelerator", None)
        if accel is None and hasattr(self, "trainer") and hasattr(self.trainer, "accelerator"):
            accel = self.trainer.accelerator

        # 랭크/월드 추정 (accelerator 없을 때 env로 보완)
        if accel is not None:
            world = getattr(accel, "num_processes", 1)
            rank = getattr(accel, "process_index", 0)
        else:
            world = int(os.getenv("WORLD_SIZE", "1"))
            rank = int(os.getenv("LOCAL_RANK", "0"))

        # ① 각 rank 로컬 스냅샷
        local = getattr(trainer_ctx, "local_results_for_log", {}) or {}
        self.logger.info({
            'Role': role_str, 'Round': round_idx, 'Split': split,
            'Rank': f'{rank}/{world}', 'Local': True,
            'Results': local
        })

        # ② rank0: per-proc 리스트
        if world > 1 and rank == 0:
            per_procs = getattr(trainer_ctx, "per_proc_local_results", None)
            if per_procs:
                self.logger.info({
                    'Role': role_str, 'Round': round_idx, 'Split': split,
                    'Local': False, 'Results_procs': per_procs
                })

        # ③ 집계 결과(해당 split 접두사 4키만; alias 금지)
        agg = getattr(trainer_ctx, "eval_metrics", {}) or {}
        sp = f"{split}_"
        agg_clean = {k: v for k, v in agg.items() if k.startswith(sp)}

        # 집계 결과는 오직 rank0에서만 출력
        if agg_clean and rank == 0:
            self.logger.info({
                'Role': role_str, 'Round': round_idx, 'Split': split,
                'Aggregated': True, 'Results_raw': agg_clean
            })

    def _is_main_process(self) -> bool:
        # accelerate가 있으면 그걸로, 없으면 단일 프로세스 가정
        trainer = getattr(self, 'trainer', None)
        if trainer is not None and hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            return trainer.accelerator.is_main_process
        return True  # accelerator 미사용 시 단일 프로세스이므로 True

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def _calculate_model_delta(self, init_model, updated_model):
        if not isinstance(init_model, list):
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            model_delta = copy.deepcopy(init_model[model_index])
            for key in init_model[model_index].keys():
                model_delta[key] = updated_model[model_index][key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        else:
            return model_deltas[0]

    def _reset_ctx_split_metrics(self, split: str):
        """
        ✅ (중요) 라운드 시작 전에 해당 split의 캐시를 리셋
        - trainer.ctx.eval_metrics, trainer.ctx.local_results_for_log,
          trainer.ctx.per_proc_local_results 등에서 split 접두사의 키를 제거.
        - DDP 상황에서 라운드 간 누적되어 train_total 이 불어나는 현상 방지.
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

            # 3) per-proc 캐시 정리
            if hasattr(ctx, "per_proc_local_results"):
                pplr = getattr(ctx, "per_proc_local_results")
                if isinstance(pplr, dict):
                    for k in list(pplr.keys()):
                        if k.startswith(f"{split}_") or k == "rank":
                            del pplr[k]
                else:
                    ctx.per_proc_local_results = None

            # 4) 임시 텐서 캐시 제거
            if hasattr(ctx, "ddp_tmp_tensors"):
                ctx.ddp_tmp_tensors = {}

        except Exception as e:
            self.logger.debug(f"[reset split={split}] skip due to: {e}")

    # ------------------------------------------------------------------
    # 러닝 루프
    # ------------------------------------------------------------------

    def join_in(self):
        """To send ``join_in`` message to the server for joining in the FL course."""
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self):
        """To listen to the message and handle them accordingly (used for distributed mode)"""
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def run_standalone(self):
        """Run in standalone mode"""
        self.join_in()
        self.run()

    # ------------------------------------------------------------------
    # 서버로부터 받은 글로벌 모델 파라미터(혹은 시크릿 셰어 조각)를 처리하고,
    # 로컬 학습을 트리거한 뒤 결과를 서버에 다시 보냅니다.
    # ------------------------------------------------------------------
    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers the local training process.
        """
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content, timestamp = message.state, message.content, message.timestamp
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train'][state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else first_aggregate_model_para)))
            return

        # ---- 여기부터 일반(비-SS) 케이스 ----
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import symmetric_uniform_dequantization
            if isinstance(content, list):  # multiple model
                content = [symmetric_uniform_dequantization(x) for x in content]
            else:
                content = symmetric_uniform_dequantization(content)

        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)

        self.trainer.update(content, strict=self._cfg.federate.share_local_model)

        self.state = round
        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and self._cfg.federate.method in ["local", "global"]

        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            sample_size, model_para_all, results = 0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(
                    f"[Local/Global mode] Client #{self.ID} has been early stopped, "
                    f"we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()

            # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
            self._reset_ctx_split_metrics('train')

            sample_size, model_para_all, results = self.trainer.train()

            if self._cfg.federate.share_local_model and not self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)

            rank, world = 0, 1
            if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
                rank = self.trainer.accelerator.process_index
                world = self.trainer.accelerator.num_processes

            # ✅ 원하는 포맷의 로그 3종(로컬 / per-proc / 집계) 출력
            self._log_split_metrics(
                role_str=f'Client #{self.ID}',
                round_idx=self.state,
                split='train',
                trainer_ctx=self.trainer.ctx
            )

            # ✅ rank0에서만 집계본을 파일로 기록
            if self._is_main_process():
                ctx = getattr(self.trainer, "ctx", None)
                agg = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}
                train_agg = {k: v for k, v in (agg or {}).items() if k.startswith("train_")}
                if train_agg:
                    self._append_raw_line(
                        role_str=f"Client #{self.ID}",
                        round_idx=self.state,
                        results_dict=train_agg,
                        filename="train_results.raw"
                    )

            # ✅ exp_print용: train 집계본만 한 줄 (rank0에서만)
            if self._is_main_process():
                train_agg = {
                    k: v for k, v in getattr(self.trainer.ctx, "eval_metrics", {}).items()
                    if k.startswith("train_")
                }
                if train_agg:
                    self.logger.info({
                        'Role': f'Client #{self.ID}',
                        'Round': self.state,
                        'Results_raw': train_agg
                    })

        # Return the feedbacks to the server after local update
        if self._cfg.federate.use_ss:
            assert not self.is_unseen_client, \
                "Un-support using secret sharing for unseen clients." \
                "i.e., you set cfg.federate.use_ss=True and " \
                "cfg.federate.unseen_clients_rate in (0, 1)"
            single_model_case = True
            if isinstance(model_para_all, list):
                assert isinstance(model_para_all[0], dict), \
                    "model_para should a list of multiple state_dict for multiple models"
                single_model_case = False
            else:
                assert isinstance(model_para_all, dict), \
                    "model_para should a state_dict for single model case"
                model_para_all = [model_para_all]
            model_para_list_all = []
            for model_para in model_para_all:
                for key in model_para:
                    model_para[key] = model_para[key] * sample_size
                model_para_list = self.ss_manager.secret_split(model_para)
                model_para_list_all.append(model_para_list)
            frame_idx = 0
            for neighbor in self.comm_manager.neighbors:
                if neighbor != self.server_id:
                    content_frame = model_para_list_all[0][frame_idx] if \
                        single_model_case else \
                        [model_para_list[frame_idx] for model_para_list in model_para_list_all]
                    self.comm_manager.send(
                        Message(msg_type='ss_model_para',
                                sender=self.ID,
                                receiver=[neighbor],
                                state=self.state,
                                timestamp=self._gen_timestamp(
                                    init_timestamp=timestamp,
                                    instance_number=sample_size),
                                content=content_frame))
                    frame_idx += 1
            content_frame = model_para_list_all[0][frame_idx] if \
                single_model_case else \
                [model_para_list[frame_idx] for model_para_list in model_para_list_all]
            self.msg_buffer['train'][self.state] = [(sample_size, content_frame)]
        else:
            if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                    ['krum', 'normbounding', 'median', 'trimmedmean', 'bulyan']:
                shared_model_para = self._calculate_model_delta(
                    init_model=content, updated_model=model_para_all)
            else:
                shared_model_para = model_para_all

            # quantization
            if self._cfg.quantization.method == 'uniform':
                from federatedscope.core.compression import symmetric_uniform_quantization
                nbits = self._cfg.quantization.nbits
                if isinstance(shared_model_para, list):
                    shared_model_para = [symmetric_uniform_quantization(x, nbits) for x in shared_model_para]
                else:
                    shared_model_para = symmetric_uniform_quantization(shared_model_para, nbits)

            # ✅ rank0만 서버로 업로드 (중복 방지)
            if self._is_main_process():
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self._gen_timestamp(
                                init_timestamp=timestamp,
                                instance_number=sample_size),
                            content=(sample_size, shared_model_para)))

    def callback_funcs_for_assign_id(self, message: Message):
        """
        The handling function for receiving the client_ID assigned by the server (distributed mode).
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    def callback_funcs_for_join_in_info(self, message: Message):
        """
        The handling function for receiving the request of join in information.
        """
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    num_sample = self._cfg.train.local_update_steps * self._cfg.dataloader.batch_size
                else:
                    num_sample = self._cfg.train.local_update_steps * len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                assert self.comm_bandwidth is not None and self.comp_speed is not None, \
                    "The requirement join_in_info 'client_resource' does not exist."
                join_in_info['client_resource'] = self.model_size / self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError('Fail to get the join in information with type {}'.format(requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    def callback_funcs_for_address(self, message: Message):
        """
        The handling function for receiving other clients' IP addresses,
        which is used for constructing a complex topology
        """
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    # ------------------------------------------------------------------
    # 서버가 “평가 요청(evaluate)” 메시지를 보낼 때
    # ------------------------------------------------------------------
    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        # 항상 먼저 초기화해서 NameError 방지
        metrics = {}

        # 1) 서버 파라미터로 동기화
        if message.content is not None:
            self.trainer.update(
                message.content,
                strict=self._cfg.federate.share_local_model
            )

        # 2) 평가 실행
        try:
            if self.early_stopper.early_stopped and self._cfg.federate.method in ["local", "global"]:
                if self.best_results:
                    metrics = list(self.best_results.values())[-1]
                else:
                    metrics = {}
            else:
                # ✅ (중요) 이번 라운드 평가 시작 전, 요청된 split 캐시를 선제 리셋
                for sp in set(self._cfg.eval.split):
                    self._reset_ctx_split_metrics(sp)

                if self._cfg.finetune.before_eval:
                    self.trainer.finetune()

                # 예: ['test','val'] 또는 ['val'] …
                for split in self._cfg.eval.split:
                    eval_metrics = self.trainer.evaluate(target_data_split_name=split)

                    # ✅ 원하는 포맷의 로그 3종(로컬 / per-proc / 집계) 출력
                    self._log_split_metrics(
                        role_str=f'Client #{self.ID}',
                        round_idx=self.state,
                        split=split,
                        trainer_ctx=self.trainer.ctx
                    )

                    # 결과 병합
                    if eval_metrics:
                        metrics.update(**eval_metrics)

        except Exception as e:
            # 평가 중 문제가 나도 서버로는 빈 dict라도 보내도록
            self.logger.warning(f"[evaluate] exception during evaluation: {e}", exc_info=True)

        # 3) rank0만 파일 기록/서버 전송
        is_main = self._is_main_process()
        if is_main:
            self._ensure_outdir()

            # 우선순위: trainer.ctx.eval_metrics(집계본) -> metrics(병합본)
            ctx = getattr(self.trainer, "ctx", None)
            agg_all = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}
            base = agg_all if agg_all else metrics

            # test_/val_만 추출
            combined = {k: v for k, v in (base or {}).items()
                        if k.startswith("test_") or k.startswith("val_")}
            has_test = any(k.startswith("test_") for k in combined)
            has_val = any(k.startswith("val_") for k in combined)

            write_key = (self.ID, int(self.state))

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

            # 3-2) 터미널 요약 로그(보기 좋게) - Monitor는 이제 raw 파일을 쓰지 않음
            try:
                self._monitor.format_eval_res(
                    metrics, rnd=self.state, role=f'Client #{self.ID}', forms=['log']
                )
            except Exception as e:
                self.logger.debug(f"[format_eval_res] skip log due to: {e}")

            # 3-3) best/early-stop 갱신 (키 자동 선택은 client 측에서 수행)
            if combined:
                want_key = self._cfg.eval.best_res_update_round_wise_key  # 예: 'test_loss'
                use_key = want_key if want_key in combined else None
                if use_key is None:
                    # 선호 순서: val_loss → val_avg_loss → val_acc → test_loss → test_avg_loss → test_acc
                    fallback_order = [
                        'val_loss', 'val_avg_loss', 'val_acc',
                        'test_loss', 'test_avg_loss', 'test_acc'
                    ]
                    use_key = next((k for k in fallback_order if k in combined), None)

                if use_key is None:
                    self.logger.warning(
                        f"[best-update] No suitable key in eval results. "
                        f"wanted='{want_key}', available={list(combined.keys())}"
                    )
                else:
                    # monitor는 cfg에 지정된 키 이름(want_key)만 찾을 수 있어 alias 처리
                    eval_for_best = dict(combined)
                    if use_key != want_key:
                        eval_for_best[want_key] = eval_for_best[use_key]
                        self.logger.warning(
                            f"[best-update] '{want_key}' not found; fallback to '{use_key}'."
                        )

                    try:
                        updated = self._monitor.update_best_result(
                            self.best_results, eval_for_best, results_type=f"client #{self.ID}"
                        )
                    except Exception as e:
                        self.logger.warning(f"[best-update] failed: {e}", exc_info=True)
                        updated = False

                    if updated and self._cfg.federate.save_client_model:
                        path = add_prefix_to_path(f'client_{self.ID}_', self._cfg.federate.save_to)
                        try:
                            self.trainer.save_model(path, self.state)
                        except Exception as e:
                            self.logger.warning(f"[save_model] failed: {e}", exc_info=True)

                    # 히스토리 누적 및 early-stop 트래킹
                    try:
                        self.history_results = merge_dict_of_results(self.history_results, eval_for_best)
                        track_key = want_key if want_key in self.history_results else use_key
                        if track_key and track_key in self.history_results:
                            self.early_stopper.track_and_check(self.history_results[track_key])
                    except Exception as e:
                        self.logger.debug(f"[early-stopper] skip tracking due to: {e}")

        # 4) 서버로는 rank0만 전송 (중복 방지)
        if is_main:
            self.comm_manager.send(
                Message(msg_type='metrics',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        timestamp=timestamp,
                        content=metrics)
            )

    def callback_funcs_for_finish(self, message: Message):
        """
        The handling function for receiving the signal of finishing the FL course.
        """
        logger.info(
            f"================= client {self.ID} received finish message =================")

        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        # ✅ rank0만 종료 파일 기록(시스템 메트릭 등)
        if self._is_main_process():
            self._monitor.finish_fl()

    def callback_funcs_for_converged(self, message: Message):
        """
        The handling function for receiving the signal that the FL course converged
        """
        self._monitor.global_converged()

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
