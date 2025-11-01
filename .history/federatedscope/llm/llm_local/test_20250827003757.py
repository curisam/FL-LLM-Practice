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


    def callback_funcs_for_model_para(self, message: Message):
            round = message.state  # 서버가 보낸 라운드 번호
            sender = message.sender  # 메시지를 보낸 주체(서버) ID
            timestamp = message.timestamp  # 서버 시각
            content = message.content  # 실제 모델 파라미터


            if self._cfg.federate.process_num > 1:
                for k, v in content.items():
                    content[k] = v.to(self.device) 

            is_main = self._is_main_process()


            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model) # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다.
            self.state = round


            #학습 시작 이전에 adapter 할당
            # Two mode of multilora training: Client-wise and clustering
            if self._cfg.llm.adapter.local_only:
                # This is client-wise
                self.model.set_active_adapter(f'Adapter_{self.ID}')
                adapter_idx = self.ID
            elif self._cfg.llm.adapter.count > 1: #3. “군집/선택” 모드:
                # This is clustering
                warmup_round = self._cfg.llm.adapter.warmup.round #25
                total_warmup_round = warmup_round * self._cfg.llm.adapter.count #75

                if self._cfg.llm.adapter.warmup.use and \
                        self.state < total_warmup_round:
                    # Initialization for all adapters
                    adapter_idx = self.state // warmup_round #워밍업 라운드: adapter_idx = state // warmup_round로 0..(count-1) 순환 활성화.
                elif self._cfg.llm.adapter.grouping.use: #True. 서버/외부에서 정해준 self.adapter_idx를 그대로 사용. GFL 과정 중에는  Warm-up 이후에는 일반적으로 이것.
                    adapter_idx = self.adapter_idx
                else: #서버에서 정해주지 않아서 val셋으로 각 어댑터를 평가해서 val_avg_loss가 가장 작은 어댑터 후보를 뽑고(동률이면 리스트에 모음), 그 중 랜덤으로 하나 선택.
                    # select the adapter with min val loss

                    with torch.no_grad():
                        min_loss, adapter_indices = 10.0, []
                        for i in range(self._cfg.llm.adapter.count):
                            if len(self.data.val_data) == 0:
                                adapter_indices.append(i)
                                continue

                            #self.trainer.ctx.model에서도 동시에 적용됨.
                            self.model.set_active_adapter(f'Adapter_{i}') 
                            self.model.eval()

                            # ✅ (중요) 이번 라운드 val 전에 split 캐시 리셋 — 누적 방지
                            self._reset_ctx_split_metrics('val)

                            metrics = self.trainer.evaluate(
                                target_data_split_name='val')
                            
                            logger.info(
                                f'Adapter {i} with the results: {metrics}')
                            
                            if i == 0 or min_loss > metrics['val_avg_loss']:
                                min_loss, adapter_indices = metrics[
                                    'val_avg_loss'], [i]
                                
                            elif min_loss == metrics['val_avg_loss']:
                                adapter_indices.append(i)

                        logger.info(adapter_indices)
                        adapter_idx = random.choice(adapter_indices)
                # activate the selected adapter for further training
                logger.info(
                    f'Activate the adapter {adapter_idx} for training...')
                self.model.set_active_adapter(f'Adapter_{adapter_idx}')
                self.model.train()
            else:
                raise ValueError(
                    'You should set llm.adapter.local_only to True '
                    'or llm.adapter.count > 1')

            
            # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
            self._reset_ctx_split_metrics('train')


            #할당된 adapter로부터 학습.

            """
            # sample_size: 전체 train data 갯수
            # model_para_all: requires_grad=True인 파라미터 포함인 것 혹은 self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함. 즉 활성/비활성 모든 adapter들만 반환.
            # results: split routine 돌며 집계된 results (num_total, loss, acc 등) 리턴
            """

            sample_size, model_para_all, results = self.trainer.train() 
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all) # 안전하게 복사
                model_para_all = {
                    key: value
                    for key, value in model_para_all.items()
                    if f'Adapter_{adapter_idx}.' in key
                } #Active adapter 정보인 Adapter_{adapter_idx}에 한한 것만 필터링.

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
                Message(msg_type='model_para', # ↔ 서버가 “train” 단계로 인식
                        sender=self.ID, # 이 클라이언트 ID
                        receiver=[sender], # 앞서 저장한 서버 ID
                        state=self.state, # (같은) 라운드 번호
                        timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                    instance_number=sample_size), # → 서버의 time-based staleness 제어용
                        content=(sample_size, shared_model_para))) # 데이터 갯수 및 로컬 모델을 content로 담아서 보낸다.


    def callback_funcs_for_adapter_eval(self, message: Message): #클라이언트가 모든 adapter의 성능을 validation data로 측정해서 서버로 전달.
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model) # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다.

        metrics = {}
        with torch.no_grad():
            for i in range(self._cfg.llm.adapter.count): #각 adapter 마다 validation loss 측정.
                #self.trainer.ctx.model에서도 동시에 적용됨.
                self.model.set_active_adapter(f'Adapter_{i}')#i번쨰 adapter를 activate
                self.model.eval() 
                # ✅ (중요) 이번 라운드 val 전에 split 캐시 리셋 — 누적 방지
                self._reset_ctx_split_metrics('val)
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

    def callback_funcs_for_setting_adapter_idx(self, message: Message): #서버에서 지정해 준 adapter idx로 지정하여 학습.
        self.adapter_idx = message.content
