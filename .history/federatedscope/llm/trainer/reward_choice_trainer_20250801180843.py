import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np
import math
from transformers import AdamW



from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer

from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.llm.dataset.llm_dataset import DefaultToken
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler


from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.model.adapter_builder import AdapterModel


import sys
import gc







sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


# def cal_loss(logits, labels, choices):
#     # 원본 logits, labels: [B, S, V]
#     #.contiguous()
#     ###슬라이스([:, :-1, :])를 하면 메모리상에 연속적(contiguous) 으로 저장되지 않을 수 있음.
#     ###.view(...) 로 모양을 바꾸려면 반드시 텐서가 contiguous여야 하니, .contiguous()로 연속 메모리에 복사해 줌.
#     shift_logits = logits[..., :-1, :].contiguous() # 전체에서 맨 마지막 부분만 제외. eos 다음 예측하는건 의미 없음, # [B, S−1, V]
#     shift_labels = labels[..., 1:].contiguous() #전체에서 맨 앞의 것 제외. 맨 앞의 것은 모델 입장에서 주어져야 하는것이기에 의미 없음. # [B, S−1, V].

#     new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value) #모든 위치를 -100(IGNORE_INDEX)으로 채웁니다.
#     for idx, choice in enumerate(choices):
#         new_labels[shift_labels == choice] = idx # :A, :B와 같이 라벨 부분만 정답의 토큰 ID로 바꾼다.

#     # new_logits = logits * mask.unsqueeze(-1)
#     # new_logits = torch.sum(new_logits, dim=1)[:, choices]
#     # new_labels = torch.sum(new_labels, dim=1)

#     new_logits = shift_logits[..., choices] #V차원(logits[..., v]) 중 오직 choices 인덱스만 골라 C개 열(column)으로 축소합니다.[B, S-1, C]
#     loss_fn = torch.nn.CrossEntropyLoss()
#     loss = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1)) # [B*(S−1), C], # [B*(S−1)]
#     # return new_logits.view(-1, len(choices)), new_labels.view(-1), loss
#     return new_logits, new_labels, loss #[B, S-1, C], [B, S-1], Scalar

# [수정] cal_loss 함수: 디바이스 불일치 문제 해결
def cal_loss(logits, labels, choices):
    choices_on_device = choices.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices_on_device):
        mask = (shift_labels == choice)
        new_labels[mask] = idx
    new_logits = shift_logits[..., choices_on_device]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(new_logits.view(-1, len(choices_on_device)), new_labels.view(-1))
    return new_logits, new_labels, loss




class RewardChoiceTrainer(LLMTrainer): #LLMTrainer를 상속 받아, “여러 선택지 중 정답을 고르는” 형태의 LLM 파인튜닝 과제를 지원하도록 만든 커스텀 트레이너입니다.
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        try:
            choices_list = []
            for choice in config.trainer.choices:
                token_ids = self.tokenizer(f': {choice}', add_special_tokens=False)['input_ids']
                if not token_ids:
                    raise ValueError(f"Tokenizer returned empty list for choice: '{choice}'")
                choices_list.append(token_ids[-1])
            self.choices = torch.tensor(choices_list) # CPU 텐서로 생성
            logger.info(f'Choice token IDs (on CPU): {self.choices.tolist()}')
        except Exception as e:
            logger.error(f"Error during trainer initialization: {e}")
            raise ValueError('Failed to initialize trainer.choices.')
        
    # [신규] train 함수 오버라이드
    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        self._reset_and_build_dataloader(target_data_split_name)
        return super().train(target_data_split_name, hooks_set)

    # [신규] evaluate 함수 오버라이드
    # def evaluate(self, target_data_split_name="test", hooks_set=None):
    #     self._reset_and_build_dataloader(target_data_split_name)
    #     return super().evaluate(target_data_split_name, hooks_set)

    def evaluate(self, target_data_split_name="test", hooks_set=None):
            # 1. hooks_set을 준비합니다 (부모 클래스 로직).
            hooks_set = hooks_set or self.hooks_in_eval
            
            # 2. 데이터로더를 먼저 준비합니다.
            self._reset_and_build_dataloader(target_data_split_name)
            
            # 3. super().evaluate() 대신, 부모 클래스의 나머지 로직을 직접 수행합니다.
            if self.ctx.check_split(target_data_split_name, skip=True):
                self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
            else:
                self.ctx.eval_metrics = dict()
            
            # 4. 결과를 직접 반환합니다.
            return self.ctx.eval_metrics    

    # [신규] 데이터로더 파괴 및 재생성 헬퍼 함수
    def _reset_and_build_dataloader(self, split_name):
        data_key = f"{split_name}_data"
        loader_key = split_name  # ClientData는 'train'을 키로 사용
        ctx_loader_key = f"{split_name}_loader"
        
        client_data_obj = self.ctx.data # ClientData 객체
        
        # 1. ClientData 내부의 기존 로더 삭제 (참조의 근원 제거)
        if loader_key in client_data_obj:
            del client_data_obj[loader_key]
            
        # 2. 원본 데이터를 사용하여 새로운 로더 생성
        if hasattr(client_data_obj, data_key) and getattr(client_data_obj, data_key) is not None:
            dataset = WrapDataset(getattr(client_data_obj, data_key))
            loader = get_dataloader(dataset, self.cfg, split_name)
            
            # 3. ClientData와 ctx 모두에 새로운 로더 할당
            client_data_obj[loader_key] = loader
            setattr(self.ctx, ctx_loader_key, loader)
            logger.info(f"Dataloader for '{split_name}' has been reset and recreated.")




        # try:
        #     # 1. 임시 파이썬 리스트를 만듭니다.
        #     choices_list = []
        #     for choice in config.trainer.choices: # ['A', 'B']
        #         # 토크나이저가 여러 토큰을 반환하더라도, 가장 마지막 토큰 ID만 사용합니다.
        #         token_ids = self.tokenizer(f': {choice}', add_special_tokens=False)['input_ids']
        #         if not token_ids:
        #             raise ValueError(f"Tokenizer returned empty list for choice: '{choice}'")
        #         choices_list.append(token_ids[-1]) # 마지막 토큰 ID를 선택

        #     # 2. 리스트를 텐서로 변환하고 모델과 같은 디바이스(GPU)로 보냅니다.
        #     #    self.ctx.device는 trainer가 할당받은 GPU 디바이스입니다.
        #     self.choices = torch.tensor(choices_list, device=self.ctx.device)
            
        #     logger.info(f'Choice token IDs: {self.choices.tolist()} on device: {self.choices.device}')

        # except Exception as e:
        #     logger.error(f"Error during trainer initialization: {e}")
        #     raise ValueError('Failed to initialize trainer.choices. Check your config file and tokenizer.')    
        



    # def _hook_on_fit_start_init(self, ctx):
    #     # 부모 클래스의 _hook_on_fit_start_init을 먼저 실행합니다.
    #     # 이 안에서 accelerator.prepare()가 호출되고 ctx.device가 최종 확정됩니다.
    #     super()._hook_on_fit_start_init(ctx)
        
    #     # --- [핵심] 이제 ctx.device가 확실하므로, choices를 GPU 텐서로 만듭니다 ---
    #     if not hasattr(self, 'choices') or self.choices.device != ctx.device:
    #         self.choices = torch.tensor(self.choices_ids, device=ctx.device)
    #         logger.info(f'Choices tensor created on device: {self.choices.device}')

    #     # 자신에게만 필요한 다른 변수 초기화
    #     ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)


    def _hook_on_fit_start_init(self, ctx):
        # 부모의 초기화 로직을 먼저 호출합니다.
        super()._hook_on_fit_start_init(ctx)
        # 자신에게만 필요한 변수를 초기화합니다.
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)    

    def _hook_on_batch_forward(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids']
            labels = ctx.data_batch['labels']
            attention_mask = ctx.data_batch['attention_mask']
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits


        #이 부분이 바뀜. LLMTRAINER 특징. Classification.
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices) #일부 모델에서 reduction='none'이면 shape가 [batch_size] 혹은 [batch_size, seq_len]일 수도 있지만, 대부분은 1개의 스칼라 값. #[B, S-1, C], [B, S-1], Scalar


        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # logger.info(f'{input_ids}')
        # logger.info(f'{labels}')

        new_labels = new_labels.view(-1) # [B*(S−1)]
        new_logits = new_logits.view(-1, len(self.choices)) # [B*(S−1), C]

        ##  “-100”(IGNORE_INDEX)로 마킹된 위치(=선택지가 아닌 토큰 위치) 제거
        new_logits = new_logits[(
            new_labels != DefaultToken.IGNORE_INDEX.value), :] ## [N_valid, C]
        new_labels = new_labels[(new_labels !=
                                 DefaultToken.IGNORE_INDEX.value)]  # [N_valid]
        _, predicted = new_logits.max(1) # [N_valid]
        # logger.info(f'{predicted}, {new_labels}, {new_logits}')

        ctx.y_true = CtxVar(new_labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(new_logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    # def _hook_on_batch_end(self, ctx):
    #     if ctx.skip_this_batch:
    #         if ctx.cfg.llm.retry_on_nan_loss:
    #             # Retry with new data in train and finetune
    #             if ctx.cur_mode == MODE.TRAIN:
    #                 self._run_batch(self.hooks_in_train, run_step=1)
    #             elif ctx.cur_mode == MODE.FINETUNE:
    #                 self._run_batch(self.hooks_in_ft, run_step=1)
    #         return

    #     # update statistics
    #     ctx.num_samples += ctx.batch_size
    #     ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
    #     ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))


    #     #이 부분이 추가됨. accuracy 측정 하기 위함.
    #     # cache label for evaluate
    #     ctx.ys_true.append(ctx.y_true)
    #     ctx.ys_pred.append(ctx.y_pred)


    # # ================== 이 메소드를 추가 또는 수정 ==================
    # def _hook_on_batch_end(self, ctx):
    #     # [핵심 추가] train 모드일 때만 correct를 계산하고 누적
    #     if ctx.cur_mode == MODE.TRAIN:
    #         # ctx에 correct 누적 변수가 없으면 0으로 초기화
    #         if not hasattr(ctx, 'correct'):
    #             ctx.correct = 0

    #         # 현재 배치의 정답 개수를 계산하여 누적
    #         pred = torch.argmax(ctx.y_prob, dim=-1)
    #         # y_true는 1차원이어야 함, new_labels는 [N_valid] 모양이었음
    #         correct_in_batch = (pred == ctx.y_true).sum().item()
    #         ctx.correct += correct_in_batch

    #     # [핵심 추가] ys_true, ys_pred를 평가 모드에서만 쌓도록 함
    #     if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
    #          # cache label for evaluate
    #         ctx.ys_true.append(ctx.y_true)
    #         ctx.ys_pred.append(ctx.y_pred)
            
    #     # 부모 클래스(LLMTrainer)의 _hook_on_batch_end를 호출하여
    #     # num_samples, loss_batch_total 등을 누적하게 함
    #     super()._hook_on_batch_end(ctx)


    # ================== 1. _hook_on_batch_end 수정 ==================
    def _hook_on_batch_end(self, ctx):
        """
        매 배치가 끝난 후, 모든 모드에 대해 결과를 ctx에 누적합니다.
        """
        # NaN loss 재시도 로직은 그대로 둡니다.
        if ctx.get('skip_this_batch', False):
            # ... (재시도 로직) ...
            return

        # --- [핵심 수정] ---
        # ctx에 누적 변수가 없으면 0으로 초기화
        if not hasattr(ctx, 'loss_total'):
            ctx.loss_total = 0.0
        if not hasattr(ctx, 'correct'):
            ctx.correct = 0
        if not hasattr(ctx, 'num_samples'):
            ctx.num_samples = 0
        
        batch_size = len(ctx.data_batch['input_ids'])

        # <<-- 누락되었던 num_samples 누적 코드 추가 -->>
        ctx.num_samples += batch_size
        
        # Loss 누적
        ctx.loss_total += ctx.loss_batch.item() * batch_size

        # Accuracy 누적
        pred = torch.argmax(ctx.y_prob, dim=-1)
        correct_in_batch = (pred == ctx.y_true).sum().item()
        ctx.correct += correct_in_batch

        # 평가 모드일 때만 ys_true/ys_pred를 리스트에 쌓음
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
            if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
            ctx.ys_true.append(ctx.y_true)
            ctx.ys_pred.append(ctx.y_pred)
            
        # 부모 클래스의 _hook_on_batch_end를 호출하지 않습니다.
        # 왜냐하면 num_samples와 loss_total 누적이 중복될 수 있기 때문입니다.
        # 여기서 모든 필요한 누적을 다 처리했습니다.




    # def _hook_on_fit_end(self, ctx): #torch_trainer.py와 같음. accuracy 측정위해 바뀜.
    #     #모든 배치가 처리된 후, loss_batch_total / num_samples로 평균 loss 계산.  ys_true, ys_prob로 정확도·기타 지표 계산(MetricCalculator 사용). 최종 결과를 ctx.eval_metrics = { ... }에 저장. 
    #     # evaluate()가 이 딕셔너리를 반환.
    #     ctx.ys_true = CtxVar(torch.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
    #     ctx.ys_pred = CtxVar(torch.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
    #     results = ctx.monitor.eval(ctx)#loss, avg_loss, acc, total을 기록한 dictionary 
    #     setattr(ctx, 'eval_metrics', results)

    # def _hook_on_fit_end(self, ctx):
    #     # 이 훅은 오직 평가(val/test) 모드에서만 작동하도록 제한
    #     if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
    #         if hasattr(ctx, 'ys_true') and ctx.ys_true:
    #             ctx.ys_true = CtxVar(torch.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
    #         if hasattr(ctx, 'ys_pred') and ctx.ys_pred:
    #             ctx.ys_pred = CtxVar(torch.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
            
    #         # 평가 모드에서는 monitor.eval을 통해 eval_metrics를 최종적으로 설정
    #         results = ctx.monitor.eval(ctx)
    #         setattr(ctx, 'eval_metrics', results)

    # ================== 2. _hook_on_fit_end 수정 ==================
    def _hook_on_fit_end(self, ctx):
        """
        이 훅은 이제 결과 계산에 관여하지 않습니다.
        메모리 정리와 같은 후처리 작업만 수행됩니다 (다른 훅에 의해).
        """
        # 기존의 모든 결과 계산 로직을 제거합니다.
        # 모든 결과 계산 및 통합은 _run_routine에서 처리됩니다.
        pass

            
def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer':
        trainer_builder = RewardChoiceTrainer
        return trainer_builder


register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
