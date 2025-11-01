import logging
import math
import torch  # 누락된 임포트
import gc  # 누락된 임포트
from typing import Dict, List, Optional, Any
from federatedscope.core.message import Message
from federatedscope.llm.llm_local.server import LLMMultiLoRAServer
from federatedscope.llm.llm_local.client import LLMMultiLoRAClient

logger = logging.getLogger(__name__)


class _BaseMoEServer(LLMMultiLoRAServer):
    """LLMMultiLoRAServer를 상속하여 그루핑 주기에 따라 E‑step을 수행할 수 있게 수정한 기본 서버입니다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mapping from client id -> weight vector [w_0,m, ..., w_{K-1,m}]
        self.client_weights: Dict[int, List[float]] = {} #client_weights: 클라이언트별 w_vec 저장 딕셔너리이며, w_{u,m} 초기값은 클라이언트 수에 따라 균등분포로 설정됩니다.
        # EMA smoothing factor (0 <= beta < 1).  A higher beta means
        # slower adaptation to new accuracies.
        self.beta = float(getattr(self._cfg.llm.adapter, 'ema_beta', 0.9))
        # Prepare a buffer for per‑round per‑client accuracies
        self.msg_buffer['adapter_acc'] = {} #각 라운드에 클라이언트들이 전송한 어댑터별 정확도를 임시 저장하는 버퍼입니다.

    # ------------------------------------------------------------------
    # Override grouping logic
    def _start_new_training_round(self, aggregated_num: int = 0,
                                  skip_grouping: bool = False): #재정의해 loss 대신 분류 정확도를 계산합니다.
        """
        Trigger a new round.  For MoE variants we replace the grouping
        trigger with a soft assignment step.  When the grouping period
        arrives, broadcast an ``adapter_eval`` message to all clients
        and wait for responses.  Otherwise, proceed with normal FL
        behaviour.
        """
        # Only trigger soft grouping at specified intervals
        if self._cfg.llm.adapter.grouping.use and not skip_grouping:
            total_warmup_round = 0
            if self._cfg.llm.adapter.warmup.use:
                total_warmup_round = (self._cfg.llm.adapter.warmup.round
                                      * self._cfg.llm.adapter.count) #warm up round 갯수를 총 lora adapter 갯수배 만큼 증강.
            regroup_interval = self._cfg.llm.adapter.grouping.round #re-grouping할 주기
            if (self.state >= total_warmup_round and
                    (self.state - total_warmup_round) % regroup_interval == 0):
                logger.info('Server: performing MoE E‑step (soft grouping)')
                #모든 클라이언트들에게 ''adapter_eval'' 메시지와 함께 서버의 파라미터를 content로 넣어서 보냄.
                self.broadcast_model_para(msg_type='adapter_eval',
                                          filter_unseen_clients=False)
                return
                
        # ───── 라운드 종료 직후 메모리/버퍼 정리 ─────
        prev_round = self.state            # self.state는 이미 1 증가하기 전 값
        if prev_round in self.msg_buffer['train']:
            self.msg_buffer['train'][prev_round].clear()
        torch.cuda.empty_cache()
        gc.collect()

        # #다음 FL 라운드에 참여할 클라이언트를 정한 후 'model_para' 메시지로 Server model para로 컨텐츠 넣어서 보냄.
        self.broadcast_model_para(msg_type='model_para',
                                    sample_client_num=self.sample_client_num)            
        

    def callback_funcs_for_grouping(self, message: Message) -> bool: #클라가 어댑터별 평가 결과를 들고 서버에 msg_type='grouping' 으로 응답한 상황에서 발동.

        """
        Callback invoked when a client sends its per‑adapter metrics.

        Each message is expected to have ``content`` of the form
        ``{'adapter_0_acc': float, 'adapter_1_acc': float, ...}``.
        We accumulate these per client and, once all clients have
        responded, compute new weight vectors and dispatch them.
        """
        if self._grouping_is_fixed:
            return False

        rnd = message.state
        cid = message.sender
        content: Dict[str, Any] = message.content

        # Initialise storage for this round
        if rnd not in self.msg_buffer['adapter_acc']:
            self.msg_buffer['adapter_acc'][rnd] = {}

        # Extract accuracies into a list sorted by adapter index
        K = int(self._cfg.llm.adapter.count)
        acc_list: List[float] = []
        for u in range(K):
            key = f'adapter_{u}_acc'
            acc_list.append(float(content.get(key, 0.0)))
        self.msg_buffer['adapter_acc'][rnd][cid] = acc_list

        # self.check_and_grouping() 대체 ─────
        if len(self.msg_buffer['adapter_acc'][rnd]) < self.client_num:
            return False
        # Compute new weight vectors for each client
        client_accs = self.msg_buffer['adapter_acc'][rnd]
        for cid, accs in client_accs.items():
            total_acc = sum(accs)
            # If no accuracy reported (unlikely), use uniform
            if total_acc <= 0:
                accs = [1.0 for _ in accs]
                total_acc = sum(accs)
            accs_normalised = [a / total_acc for a in accs]
            if cid not in self.client_weights:
                # Initialise with uniform distribution
                w_old = [1.0 / len(accs) for _ in accs]
            else:
                w_old = self.client_weights[cid]
            # EMA update
            w_new = [self.beta * w_old[u] + (1.0 - self.beta) * accs_normalised[u]
                     for u in range(len(accs))]
            # Renormalise to ensure sum(w)=1
            s = sum(w_new)

            w_new = [w / s for w in w_new]


            self.client_weights[cid] = w_new
            logger.info(f'Client {self.ID}: original weight vector = {w_old}')
            logger.info(f'Client {self.ID}: recent weight vector = {accs_normalised[u]}')
            logger.info(f'Client {self.ID}: updated weight vector = {w_new}')            

        # Dispatch weight vectors to clients and proceed
        self._dispatch_weights(self.client_weights)
        # Clear buffer for this round
        del self.msg_buffer['adapter_acc'][rnd]

        
        # Start new training round (skip grouping)
        self._start_new_training_round(skip_grouping=True)
        return True

    def _dispatch_weights(self, w_map: Dict[int, List[float]]):
        """Dispatch updated weights to clients.

        Subclasses must override this method to send weight vectors
        appropriately.  It is responsible for constructing and sending
        ``set_adapter_weights`` messages to clients.
        """
        raise NotImplementedError


class FullMoEServer(_BaseMoEServer):
    """Server implementing soft assignment and dispatch for Full‑MoE."""

    def _dispatch_weights(self, w_map: Dict[int, List[float]]):
        for cid, w_vec in w_map.items():
            # send weight vector to each client; receiver must be a list
            self.comm_manager.send(Message(
                msg_type='set_adapter_weights',
                sender=self.ID,
                receiver=[cid],
                state=self.state,
                timestamp=self.cur_timestamp,
                content={'w_vec': w_vec}
            ))


class FusionMoEServer(_BaseMoEServer):
    """Server implementing soft assignment and dispatch for Fusion‑MoE."""

    def _dispatch_weights(self, w_map: Dict[int, List[float]]):
        for cid, w_vec in w_map.items():
            self.comm_manager.send(Message(
                msg_type='set_adapter_weights',
                sender=self.ID,
                receiver=[cid],
                state=self.state,
                timestamp=self.cur_timestamp,
                content={'w_vec': w_vec}
            ))


class _BaseMoEClient(LLMMultiLoRAClient):
    """Client base class supporting weight vector handling.

    Subclasses must override methods for evaluation and training to use
    ``self.w_vec`` as appropriate.  The base class adds a handler for
    ``set_adapter_weights`` messages and stores the received vector.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight vector initialised on server dispatch
        self.w_vec: Optional[List[float]] = None
        # Register handler for receiving weight vectors
        self.register_handlers('set_adapter_weights',
                               self.callback_funcs_for_set_adapter_weights,
                               [])

    def callback_funcs_for_set_adapter_weights(self, message: Message) -> bool:
        """Store the weight vector sent by the server."""
        w_vec = message.content.get('w_vec', None)
        if w_vec is None:
            logger.warning(f'Client {self.ID}: received empty w_vec')
        else:
            self.w_vec = list(map(float, w_vec))
            logger.info(f'Client {self.ID}: updated weight vector = {self.w_vec}')
        return True

    def callback_funcs_for_adapter_eval(self, message: Message) -> bool:


        """
        모든 어댑터에 대해:
        - 서버 전송: val_acc 
        - 로컬 파일(adapter_eval_result.raw) 기록: val_* 및 test_* 메트릭만 기록 (train 제거)
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        # 0) 서버 파라미터 동기화 (기존 유지)
        if message.content is not None:
            self.trainer.update(
                message.content,
                strict=self._cfg.federate.share_local_model
            )

        # 1) 서버로 보낼 요약 (각 어댑터의 val_acc만)
        metrics = {}

        # 2) 파일 기록은 rank0만 수행 (중복 방지용 마커)
        is_main = getattr(self, "_is_main_process", lambda: True)()
        if is_main and not hasattr(self, "_adapter_eval_written_marker"):
            self._adapter_eval_written_marker = set()  # (client_id, round, adapter_idx)

        results_to_write = []  # 어댑터별 val/test 결과를 모아 한 번에 기록
    
        with torch.no_grad():
            for i in range(self._cfg.llm.adapter.count):
                # 어댑터 전환
                self.model.set_active_adapter(f'Adapter_{i}')
                self.model.eval()
                self._sync_after_adapter_swap()

                # -------- (A) 서버 전송용: val --------
                try:
                    self._reset_ctx_split_metrics('val')
                except Exception:
                    pass
                val_metrics = self.trainer.evaluate(target_data_split_name='val')
                logger.info(f'Client {self.ID} Adapter {i} with val results: {val_metrics}')
                # ★ 서버로는 val_avg_loss만 유지
                metrics[f'adapter_{i}_acc'] = val_metrics['val_acc']


                # -------- (B) 파일 기록용: val/test --------
                rec = {
                    "client_id": int(self.ID),
                    "round": int(self.state),
                    "adapter_idx": int(i),
                }
                # val_* 메트릭 추가
                rec.update({k: v for k, v in val_metrics.items() if k.startswith("val_")})

                # test_* 메트릭 추가
                try:
                    self._reset_ctx_split_metrics('test')
                except Exception:
                    pass
                try:
                    test_metrics = self.trainer.evaluate(target_data_split_name='test')
                    rec.update({k: v for k, v in test_metrics.items() if k.startswith("test_")})
                    logger.info(f'Client {self.ID} Adapter {i} with test results: {test_metrics}')
                except Exception as e:
                    logger.debug(f"[adapter_eval] skip test eval for adapter {i}: {e}")

                results_to_write.append(rec)

        # 3) 서버로 메시지 전송(기존 포맷 유지: 각 어댑터의 val_avg_loss만 포함)
        self.comm_manager.send(
            Message(
                msg_type='grouping',
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=timestamp,
                content=metrics
            )
        )

        # 4) 로컬 파일 기록: adapter_eval_result.raw (rank0만)
        if is_main:
            try:
                self._ensure_outdir()
            except Exception:
                pass

            for rec in results_to_write:
                key = (rec["client_id"], rec["round"], rec["adapter_idx"])
                if key in self._adapter_eval_written_marker:
                    continue
                try:
                    self._append_raw_line(
                        role_str=f"Client #{self.ID}",
                        round_idx=self.state,
                        results_dict=rec,
                        filename="adapter_eval_result.raw",
                        adapter_idx=rec["adapter_idx"]
                    )
                    self._adapter_eval_written_marker.add(key)
                except Exception as e:
                    logger.debug(f"[adapter_eval] write failed for {key}: {e}")

        return True

class FullMoEClient(_BaseMoEClient):
    """Client implementation for Full‑MoE.

    In Full‑MoE, each client trains all experts simultaneously.  The
    trainer must support the ``w_vec`` attribute on the context; this
    implementation attaches the current weight vector to the training
    context before each batch.
    """

    def callback_funcs_for_model_para(self, message: Message) -> bool:
        # ── 0) 기본 전처리: 서버 파라미터 동기화 ──
        round_idx = message.state  # 서버가 보낸 라운드 번호
        sender = message.sender  # 메시지를 보낸 주체(서버) ID
        timestamp = message.timestamp  # 서버 시각
        content = message.content  # 실제 모델 파라미터


        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device) 

        self.model.train()

        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model) # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다. (일단 모든 adapter 다 update 됨.)
        self.state = round_idx
        is_main = self._is_main_process()

        # ── 1) w_vec 준비 ──
        setattr(self.trainer.ctx, 'w_vec', self.w_vec)
        
        # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
        self._reset_ctx_split_metrics('train')


        #Full MOE 학습. 모든 어댑터를 순차적으로 활성화하고 학습.

        """
        # sample_size: 전체 train data 갯수
        # model_para_all: requires_grad=True인 파라미터 포함인 것 혹은 self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함. 즉 활성/비활성 모든 adapter들만 반환.
        # results: split routine 돌며 집계된 results (num_total, loss, acc 등) 리턴
        """

        sample_size, model_para_all, results = self.trainer.train(round_num=round_idx)  #여기서 model_para_all은 active 뿐만 아니라 모든 adpater 다 받아온 것. 그리고 DDP여도 .module이 다  제거된 상태.


        # 공유 파라미터 정리: Active adapter만 필터링
        shared_model_para = model_para_all
        if self._cfg.federate.share_local_model and not self._cfg.federate.online_aggr:
            import copy
            mp = copy.deepcopy(model_para_all)
            shared_model_para = {k: v for k, v in mp.items() if 'Adapter_' in k}




        # ✅ 1)랭크별 train result 로그 띄움 2)모들 랭크 종합한 train result 로그 띄움.
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
                filename="train_results.raw",
                adapter_idx=-1

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

        payload = {'sample_size': sample_size, 'model_para': shared_model_para, 'w': self.w_vec}

        # ✅ 모든 rank에 대해서 동일하게 메시지를 보냄. 서버는 각 process마다 동일하게 self.comm_manager.comm_queue를 관리해야함.
        self.comm_manager.send(
            Message(msg_type='model_para', # ↔ 서버가 “train” 단계로 인식
                    sender=self.ID, # 이 클라이언트 ID
                    receiver=[sender], # 앞서 저장한 서버 ID
                    state=self.state, # (같은) 라운드 번호
                    timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                instance_number=sample_size), # → 서버의 time-based staleness 제어용
                    content=payload)) # 데이터 갯수 및 로컬 active adapter 모델만을 content로 담아서 보낸다.

        return True

class FusionMoEClient(_BaseMoEClient):
    def _fuse_experts(self):
        # 전문가 어댑터 가중합 → default 어댑터로 주입 (텐서형 0으로 누적)
        state = self.model.state_dict()
        default_keys = [k for k in state.keys() if 'default' in k]
        K = int(self._cfg.llm.adapter.count)

        fused = {k: None for k in default_keys}
        for u in range(K):
            w = self.w_vec[u] 
            prefix = f'Adapter_{u}'
            for k in default_keys:
                expert_key = k.replace('default', prefix)
                if expert_key in state:
                    if fused[k] is None:
                        fused[k] = state[expert_key].detach().clone()
                        fused[k].mul_(0.0)  # zeros_like와 동일
                    fused[k].add_(w * state[expert_key])

        for k in default_keys:
            if fused[k] is not None:
                state[k] = fused[k]
        self.model.load_state_dict(state, strict=False)

    def callback_funcs_for_model_para(self, message: Message) -> bool:
        # ── 0) 기본 전처리: 서버 파라미터 동기화 ──
        round_idx = message.state  # 서버가 보낸 라운드 번호
        sender = message.sender  # 메시지를 보낸 주체(서버) ID
        timestamp = message.timestamp  # 서버 시각
        content = message.content  # 실제 모델 파라미터


        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device) 

        self.model.train()
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model) # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다. (일단 모든 adapter 다 update 됨.)
        self.state = round_idx
        is_main = self._is_main_process()

        # ── 1) w_vec 준비 ──
        setattr(self.trainer.ctx, 'w_vec', self.w_vec)
        
        # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
        self._reset_ctx_split_metrics('train')

        # 전문가 → default로 퓨전 후 default만 활성화
        self._fuse_experts()
        try:
            self.model.set_active_adapter('default')
            self.model.train()
            self._sync_after_adapter_swap()
        except Exception:
            pass


        #Fusion MOE 학습. 

        """
        # sample_size: 전체 train data 갯수
        # model_para_all: requires_grad=True인 파라미터 포함인 것 혹은 self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함. 즉 활성/비활성 모든 adapter들만 반환.
        # results: split routine 돌며 집계된 results (num_total, loss, acc 등) 리턴
        """

        sample_size, model_para_all, results = self.trainer.train(round_num=round_idx)  #여기서 model_para_all은 active 뿐만 아니라 모든 adpater 다 받아온 것. 그리고 DDP여도 .module이 다  제거된 상태.


        # default 파라미터만 추출
        shared_model_para = {k: v for k, v in model_para_all.items() if 'default' in k}




        # ✅ 1)랭크별 train result 로그 띄움 2)모들 랭크 종합한 train result 로그 띄움.
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
                filename="train_results.raw",
                adapter_idx="default"

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

        payload = {'sample_size': sample_size, 'model_para': shared_model_para, 'w': self.w_vec}

        # ✅ 모든 rank에 대해서 동일하게 메시지를 보냄. 서버는 각 process마다 동일하게 self.comm_manager.comm_queue를 관리해야함.
        self.comm_manager.send(
            Message(msg_type='model_para', # ↔ 서버가 “train” 단계로 인식
                    sender=self.ID, # 이 클라이언트 ID
                    receiver=[sender], # 앞서 저장한 서버 ID
                    state=self.state, # (같은) 라운드 번호
                    timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                instance_number=sample_size), # → 서버의 time-based staleness 제어용
                    content=payload)) # 데이터 갯수 및 로컬 active adapter 모델만을 content로 담아서 보낸다.

        return True