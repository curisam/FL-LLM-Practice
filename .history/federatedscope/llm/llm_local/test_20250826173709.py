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
                elif self._cfg.llm.adapter.grouping.use: #True. 서버/외부에서 정해준 self.adapter_idx를 그대로 사용.
                    adapter_idx = self.adapter_idx
                else: #val셋으로 각 어댑터를 평가해서 val_avg_loss가 가장 작은 어댑터 후보를 뽑고(동률이면 리스트에 모음), 그 중 랜덤으로 하나 선택.
                    # select the adapter with min val loss
                    with torch.no_grad():
                        min_loss, adapter_indices = 10.0, []
                        for i in range(self._cfg.llm.adapter.count):
                            if len(self.data.val_data) == 0:
                                adapter_indices.append(i)
                                continue
                            self.model.set_active_adapter(f'Adapter_{i}')
                            self.model.eval()
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

            sample_size, model_para_all, results = self.trainer.train()
            train_data_size = len(self.data.train_data)
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
                model_para_all = {
                    key: value
                    for key, value in model_para_all.items()
                    if f'Adapter_{adapter_idx}.' in key
                }
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                        save_file_name="")

            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                    instance_number=sample_size),
                        content=(sample_size, model_para_all)))