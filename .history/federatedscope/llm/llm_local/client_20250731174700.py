import copy
import logging

import torch
from torch.utils.data import random_split, ConcatDataset, Subset

import random

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.data import ClientData


import os
import pickle

logger = logging.getLogger(__name__)


def reorg_client_data(cdata: ClientData): #기존 train_data를 학습용 + 검증용으로 나누고, 기존의 val_data와 test_data는 합쳐서 새로운 test set으로 만든다. cdata라는 인자는 ClientData 타입의 객체여야 함을 명시적으로 표현
    data_length = len(cdata.train_data)
    val_data_length = max(min(int(data_length * 0.05), 200), 1) #최대 200개, 최소 1개
    train_data_length = data_length - val_data_length
    train_data, val_data = random_split(cdata.train_data,
                                        [train_data_length, val_data_length])
    test_data = ConcatDataset([cdata.val_data, cdata.test_data])
    return ClientData(cdata.client_cfg, train_data, val_data, test_data)


# 기존 reorg_client_data 함수를 아래와 같이 수정합니다.
def reorg_client_data_with_caching(cdata: ClientData, client_id: int, config):
    """
    기존 train_data를 train/val로 나누고, val/test는 합쳐서 test로 만듭니다.
    이 과정에서 생성된 데이터 분할을 저장하거나 로드하는 캐싱 기능을 추가합니다.
    또한, 최종 test set을 40개로 랜덤 샘플링하고, 각 데이터셋의 크기를 출력합니다.
    """
    splits_path = getattr(config.data, 'splits_path', './final_data_splits')
    pickle_file = os.path.join(splits_path, f"client_{client_id}_final.pkl")

    # 1. 데이터 로드 시도
    if getattr(config.data, 'load_splits', False):
        logger.info(f"Client {client_id}: Attempting to load final data split from {pickle_file}")
        if not os.path.exists(pickle_file):
            raise FileNotFoundError(
                f"Final split file for client {client_id} not found at {pickle_file}. "
                "You must run with 'data.save_splits: True' at least once."
            )
        with open(pickle_file, 'rb') as f:
            reorganized_data = pickle.load(f)
        logger.info(f"Client {client_id}: Final data split loaded successfully.")
        
        # ======================= 로드 후 사이즈 출력 추가 =======================
        logger.info(f"Client {client_id} - Loaded dataset sizes: "
                    f"Train={len(reorganized_data.train_data)}, "
                    f"Val={len(reorganized_data.val_data)}, "
                    f"Test={len(reorganized_data.test_data)}")
        # ====================================================================
        
        return reorganized_data

    # 2. 데이터 로드 안 함 -> 새로 생성
    # 2.1. Train/Validation 분할
    logger.info(f"Client {client_id}: Reorganizing train/val split...")
    data_length = len(cdata.train_data)
    val_data_length = max(min(int(data_length * 0.05), 200), 1)
    train_data_length = data_length - val_data_length
    
    train_val_generator = torch.Generator().manual_seed(config.seed + client_id)
    train_data, val_data = random_split(cdata.train_data,
                                        [train_data_length, val_data_length],
                                        generator=train_val_generator)
    
    # 2.2. Test 데이터 통합 및 샘플링
    logger.info(f"Client {client_id}: Reorganizing and sampling test set...")
    combined_test_data = ConcatDataset([cdata.val_data, cdata.test_data])
    
    target_test_size = 40
    original_test_size = len(combined_test_data)
    final_test_size = min(target_test_size, original_test_size)

    if original_test_size < target_test_size:
        logger.warning(f"Client {client_id}: Combined test set size ({original_test_size}) "
                       f"is smaller than target size ({target_test_size}). Using all {original_test_size} samples.")

    test_sampler_generator = torch.Generator().manual_seed(config.seed + client_id + 1)
    shuffled_indices = torch.randperm(original_test_size, generator=test_sampler_generator).tolist()
    sampled_indices = shuffled_indices[:final_test_size]
    final_test_data = Subset(combined_test_data, sampled_indices)
    
    # 최종적으로 재구성된 데이터를 ClientData 객체로 만듭니다.
    reorganized_data = ClientData(cdata.client_cfg, train_data, val_data, final_test_data)
    
    # ======================= 생성 후 사이즈 출력 추가 =======================
    logger.info(f"Client {client_id} - Created dataset sizes: "
                f"Train={len(reorganized_data.train_data)}, "
                f"Val={len(reorganized_data.val_data)}, "
                f"Test={len(reorganized_data.test_data)}")
    # ====================================================================

    # 3. 데이터 저장
    if getattr(config.data, 'save_splits', False):
        logger.info(f"Client {client_id}: Saving final data split to {pickle_file}")
        os.makedirs(splits_path, exist_ok=True)
        with open(pickle_file, 'wb') as f:
            pickle.dump(reorganized_data, f)
        logger.info(f"Client {client_id}: Final data split saved.")

    return reorganized_data


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
        
        import ipdb; ipdb.set_trace(context=15)

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
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        self.state = round
        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            # for these cases (1) unseen client (2) isolated_global_mode,
            # we do not local train and upload local model
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(f"[Local/Global mode] Client #{self.ID} has been "
                            f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()

            # Two mode of multilora training: Client-wise and clustering
            if self._cfg.llm.adapter.local_only:
                # This is client-wise
                self.model.set_active_adapter(f'Adapter_{self.ID}')
                adapter_idx = self.ID
            elif self._cfg.llm.adapter.count > 1:
                # This is clustering
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count
                if self._cfg.llm.adapter.warmup.use and \
                        self.state < total_warmup_round:
                    # Initialization for all adapters
                    adapter_idx = self.state // warmup_round
                elif self._cfg.llm.adapter.grouping.use:
                    adapter_idx = self.adapter_idx
                else:
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
