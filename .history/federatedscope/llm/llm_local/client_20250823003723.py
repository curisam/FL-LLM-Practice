import copy
import logging

import torch
from torch.utils.data import random_split, ConcatDataset, Subset

import random
import copy

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.data import ClientData


import os
import pickle

logger = logging.getLogger(__name__)



class LLMMultiLoRAClient(Client):
    """
    Client implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):

        super(LLMMultiLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('adapter_eval',
                               self.callback_funcs_for_adapter_eval,
                               ['grouping'])
        self.register_handlers('set_active_adapter_idx',
                               self.callback_funcs_for_setting_adapter_idx,
                               [None])

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state  # 서버가 보낸 라운드 번호
        sender = message.sender  # 메시지를 보낸 주체(서버) ID
        timestamp = message.timestamp  # 서버 시각
        content = message.content  # 실제 모델 파라미터

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)

        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)  # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다.
        
        self.state = round

        # 2) 어댑터 선택/활성화
        m = self.trainer.model
        adapter_idx = None

        #학습 시작 이전에 adapter 할당

        if self._cfg.llm.adapter.count > 1: #3. “군집/선택” 모드:
            # This is clustering
            warmup_round = self._cfg.llm.adapter.warmup.round #25
            total_warmup_round = warmup_round * self._cfg.llm.adapter.count #75
            if self._cfg.llm.adapter.warmup.use and self.state < total_warmup_round: #워밍업 라운드: adapter_idx = state // warmup_round로 0..(count-1) 순환 활성화.
                # Initialization for all adapters
                adapter_idx = self.state // warmup_round

            elif self._cfg.llm.adapter.grouping.use: #True. 서버/외부에서 정해준 self.adapter_idx를 그대로 사용.
                adapter_idx = getattr(self, "adapter_idx", 0)

            else: #val셋으로 각 어댑터를 평가해서 val_avg_loss가 가장 작은 어댑터 후보를 뽑고(동률이면 리스트에 모음), 그 중 랜덤으로 하나 선택.
                # select the adapter with min val loss

                # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
                self._reset_ctx_split_metrics('val')
                with torch.no_grad():

                    best_loss = None
                    candidates = []
                    for i in range(self._cfg.llm.adapter.count):
                        m.set_active_adapter(f'Adapter_{i}')
                        m.eval()
                        metrics = self.trainer.evaluate(target_data_split_name='val')#global model에서 validation 기준 결과들.
                        logger.info(f"[Adapter Select] {i} -> {metrics}")
                        loss_i = metrics.get('val_avg_loss', None)
                        if loss_i is None: # val이 없거나 키가 없으면 일단 후보에 포함
                            candidates.append(i)
                            continue

                        if (best_loss is None) or (loss_i < best_loss): #i==0은 초기값으로 임시 지정. 그 이후에는 가장 min loss로 나오는 것으로 업데이트
                            best_loss = loss_i
                            candidates = [i]
                        elif best_loss == loss_i: #동률일 경우에는 추가.
                            candidates.append(i)
                    logger.info(candidates)
                    adapter_idx = random.choice(candidates) if candidates else 0

            # activate the selected adapter for further training
            #최종적으로 self.model.set_active_adapter(f'Adapter_{adapter_idx}') 하고 self.model.train() 으로 학습 모드 세팅.
            logger.info(
                f'Activate the adapter {adapter_idx} for training...')
            self.model.set_active_adapter(f'Adapter_{adapter_idx}')
            self.model.train()
        else:
            raise ValueError(
                'You should set llm.adapter.local_only to True '
                'or llm.adapter.count > 1')
        

        #할당된 adapter로부터 학습.
        # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
        self._reset_ctx_split_metrics('train')


        sample_size, model_para_all, results = self.trainer.train() # 전체 train data 갯수, model_para_all (업데이트된 active adapter의 state_dict), split routine 돌며 집계된 results (num_total, loss, acc 등) 리턴
 
        
        if self._cfg.federate.share_local_model and not self._cfg.federate.online_aggr:
            model_para_all = copy.deepcopy(model_para_all)  # 안전하게 복사

            model_para_all = {
                key: value
                for key, value in model_para_all.items()
                if f'Adapter_{adapter_idx}.' in key
            }

        rank, world = 0, 1
        if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
            rank  = self.trainer.accelerator.process_index
            world = self.trainer.accelerator.num_processes

        # ✅ 1)랭크별 train result 로그 띄움 2)모들 랭크 종합한 train result 로그 띄움.
        self._log_split_metrics(
            role_str=f'Client #{self.ID}',
            round_idx=self.state,
            split='train',
            trainer_ctx=self.trainer.ctx
        )

        # ✅ rank0(=main process)에서만 집계본을 파일로 기록
        if self._is_main_process():
            ctx = getattr(self.trainer, "ctx", None)
            agg = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}
            train_agg = {k: v for k, v in (agg or {}).items() if k.startswith("train_")}#한 클라이언트 기준 집계된 train split 결과.
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
                }) #INFO: {'Role': 'Client #44', 'Round': 0, 'Results_raw': {'train_total': 480, 'train_loss': 348.3512268066406, 'train_avg_loss': 0.7257317225138347, 'train_seen': 480, 'train_correct': 234, 'train_acc': 0.4875}}
 
        # Return the feedbacks to the server after local update 
        shared_model_para = model_para_all

        # ✅ rank0만 서버로 업로드 (중복 방지)
        if self._is_main_process():
            self.comm_manager.send(
                Message(msg_type='model_para', # ↔ 서버가 “train” 단계로 인식
                        sender=self.ID, # 이 클라이언트 ID
                        receiver=[sender], # 앞서 저장한 서버 ID
                        state=self.state, # (같은) 라운드 번호
                        timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                    instance_number=sample_size), # → 서버의 time-based staleness 제어용
                        content=(sample_size, model_para_all))) # 데이터 갯수 및 로컬 모델을 content로 담아서 보낸다.

    def callback_funcs_for_adapter_eval(self, message: Message):
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        metrics = {}
        with torch.no_grad():
            for i in range(self._cfg.llm.adapter.count):
                if len(self.data.val_data) == 0:
                    metrics[f'adapter_{i}_avg_loss'] = random.random()
                    continue
                self.model.set_active_adapter(f'Adapter_{i}')
                self.model.eval()
                adap_metrics = self.trainer.evaluate(
                    target_data_split_name='val')
                logger.info(f'Client {self.ID} Adapter {i} with '
                            f'the results: {adap_metrics}')

                metrics[f'adapter_{i}_avg_loss'] = adap_metrics['val_avg_loss']

        self.comm_manager.send(
            Message(msg_type='grouping',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_setting_adapter_idx(self, message: Message):
        self.adapter_idx = message.content
