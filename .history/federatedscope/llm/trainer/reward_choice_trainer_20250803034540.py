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


from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist


import sys
import gc

import os







sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)



def _get_dist_info():
    # torch.distributed 우선, 아니면 ENV(RANK, WORLD_SIZE) fallback
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    try:
        ws = int(os.environ.get('WORLD_SIZE', '1'))
        rk = int(os.environ.get('RANK', '0'))
        return (ws > 1), ws, rk
    except Exception:
        return False, 1, 0

class EvalShardSampler(Sampler):
    """평가 전용 샘플러: 중복/패딩 없이 rank마다 고유 인덱스만 반환"""
    def __init__(self, dataset_len: int, rank: int, world_size: int):
        self.indices = list(range(rank, int(dataset_len), int(world_size)))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)


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

            with torch.no_grad():  # 추가            
                # 3. super().evaluate() 대신, 부모 클래스의 나머지 로직을 직접 수행합니다.
                if self.ctx.check_split(target_data_split_name, skip=True):
                    self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
                else:
                    self.ctx.eval_metrics = dict()
            
            # 4. 결과를 직접 반환합니다.
            return self.ctx.eval_metrics

    def _reset_and_build_dataloader(self, split_name):
        data_key = f"{split_name}_data"
        loader_key = split_name            # ClientData dict 키 ('train'/'val'/'test')
        ctx_loader_key = f"{split_name}_loader"
        client_data_obj = self.ctx.data    # ClientData 객체

        # 1) 기존 로더 제거(참조 루트 끊기)
        if loader_key in client_data_obj:
            del client_data_obj[loader_key]

        # 2) 원본 데이터로 새 로더 생성
        if hasattr(client_data_obj, data_key) and getattr(client_data_obj, data_key) is not None:
            dataset = WrapDataset(getattr(client_data_obj, data_key))

            # 기본 로더(배치/콜레이트 재사용 목적)
            base_loader = get_dataloader(dataset, self.cfg, split_name)

            # 3) 분산 정보 (ENV fallback 포함)
            dist_ready, world_size, rank = _get_dist_info()

            # 4) world_size>1 이면 모든 split 샤딩
            if world_size > 1:
                if split_name == 'train':
                    # train: 분산 학습 표준 샘플러 (에폭별 셔플)
                    sampler = DistributedSampler(
                        base_loader.dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True
                    )
                    loader = DataLoader(
                        dataset=base_loader.dataset,
                        batch_size=base_loader.batch_size,
                        sampler=sampler,
                        shuffle=False,          # sampler 사용 시 False
                        num_workers=getattr(base_loader, 'num_workers', 0),
                        pin_memory=getattr(base_loader, 'pin_memory', False),
                        drop_last=getattr(base_loader, 'drop_last', False),
                        collate_fn=getattr(base_loader, 'collate_fn', None)
                    )
                else:
                    # val/test: 중복 없는 샘플러
                    sampler = EvalShardSampler(len(base_loader.dataset), rank, world_size)
                    loader = DataLoader(
                        dataset=base_loader.dataset,
                        batch_size=base_loader.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                        drop_last=False,
                        collate_fn=getattr(base_loader, 'collate_fn', None)
                    )
                sharded = True
                local_count = len(sampler)
            else:
                loader = base_loader
                sharded = False
                local_count = len(base_loader.dataset)


    #         # 4) test/val일 때만 "중복/패딩 없는" 샤딩 적용
    #         if split_name in ['test', 'val'] and world_size > 1:
    #             sampler = EvalShardSampler(len(base_loader.dataset), rank, world_size)
    #             # loader = DataLoader(
    #             #     dataset=base_loader.dataset,
    #             #     batch_size=base_loader.batch_size,
    #             #     sampler=sampler,         # sampler 지정
    #             #     shuffle=False,           # sampler와 shuffle 동시 사용 금지
    #             #     num_workers=getattr(base_loader, 'num_workers', 0),
    #             #     pin_memory=getattr(base_loader, 'pin_memory', False),
    #             #     collate_fn=getattr(base_loader, 'collate_fn', None)
    #             # )
    #             loader = DataLoader(
    #                 dataset=base_loader.dataset,
    #                 batch_size=base_loader.batch_size,
    #                 sampler=sampler,
    #                 shuffle=False,
    #                 num_workers=0,          # ← 고정
    #                 pin_memory=False,       # 보수적으로 OFF 권장
    #                 drop_last=False,
    #                 collate_fn=getattr(base_loader, 'collate_fn', None)
    # )

    #             sharded = True
    #             local_count = len(sampler)
    #         else:
    #             loader = base_loader
    #             sharded = False
    #             local_count = len(base_loader.dataset)

    

            # 5) ClientData와 ctx에 반영
            client_data_obj[loader_key] = loader
            setattr(self.ctx, ctx_loader_key, loader)
            logger.info(
                f"Dataloader for '{split_name}' has been reset and recreated. "
                f"(sharded={sharded}, world_size={world_size}, rank={rank}, "
                f"local_count={local_count}, total={len(base_loader.dataset)})"
            )
    

    # def _reset_and_build_dataloader(self, split_name):
    #     data_key = f"{split_name}_data"
    #     loader_key = split_name            # ClientData dict 키 ('train'/'val'/'test')
    #     ctx_loader_key = f"{split_name}_loader"
    #     client_data_obj = self.ctx.data    # ClientData 객체

    #     # 1) 기존 로더 제거(참조 루트 끊기)
    #     if loader_key in client_data_obj:
    #         del client_data_obj[loader_key]

    #     # 2) 원본 데이터로 새 로더 생성
    #     if hasattr(client_data_obj, data_key) and getattr(client_data_obj, data_key) is not None:
    #         dataset = WrapDataset(getattr(client_data_obj, data_key))

    #         # 기본 로더(배치/콜레이트 재사용 목적)
    #         base_loader = get_dataloader(dataset, self.cfg, split_name)

    #         # 3) 분산 초기화 상태 확인 (Accelerate 생성 전이므로 torch.distributed로 판단)
    #         dist_ready = dist.is_available() and dist.is_initialized()
    #         world_size = dist.get_world_size() if dist_ready else 1
    #         rank = dist.get_rank() if dist_ready else 0

    #         # 4) test/val일 때만 "중복/패딩 없는" 샤딩 적용
    #         if split_name in ['test', 'val'] and world_size > 1:
    #             sampler = EvalShardSampler(len(base_loader.dataset), rank, world_size)
    #             loader = DataLoader(
    #                 dataset=base_loader.dataset,
    #                 batch_size=base_loader.batch_size,
    #                 sampler=sampler,          # sampler 지정
    #                 shuffle=False,            # sampler와 shuffle는 같이 쓰지 않음
    #                 num_workers=getattr(base_loader, 'num_workers', 0),
    #                 pin_memory=getattr(base_loader, 'pin_memory', False),
    #                 collate_fn=getattr(base_loader, 'collate_fn', None)
    #             )
    #             sharded = True
    #             local_count = len(sampler)
    #         else:
    #             loader = base_loader
    #             sharded = False
    #             local_count = len(base_loader.dataset)

    #         # 5) ClientData와 ctx에 반영
    #         client_data_obj[loader_key] = loader
    #         setattr(self.ctx, ctx_loader_key, loader)
    #         logger.info(
    #             f"Dataloader for '{split_name}' has been reset and recreated. "
    #             f"(sharded={sharded}, world_size={world_size}, rank={rank}, "
    #             f"local_count={local_count}, total={len(base_loader.dataset)})"
    #         )



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

        # [라운드-로컬 집계 카운터] 초기화
        ctx.round_seen_local = 0          # 이 rank가 이 라운드에서 '본' 샘플 수
        ctx.round_sum_loss_local = 0.0    # 이 rank의 loss 합(샘플 기준 합)
        ctx.round_correct_local = 0       # (선택) 정답 개수, 쓰면 acc도 계산 가능


        # 자신에게만 필요한 변수를 초기화합니다.
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)    

    def _hook_on_batch_forward(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
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


    def _hook_on_batch_end(self, ctx):
        if ctx.get('skip_this_batch', False):
            return

        split = ctx.cur_split  # 'train' / 'val' / 'test'

        # ---- 배치 샘플 수(=시퀀스 개수) ----
        # 위에서 _hook_on_batch_forward 에서: ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        bs = int(ctx.batch_size)

        # ---- 선택지(pair) 수: 예) ['A','B'] -> 2 ----
        pair_factor = max(1, len(getattr(self, 'choices', [])))  # choices 텐서 길이

        # ---- 샘플 기준 누적 (한 번만!) ----
        incr = bs * pair_factor

        # 손실은 '샘플 기준 합'으로 누적 (loss가 배치 평균이면 loss * incr)
        ctx.round_sum_loss_local += float(ctx.loss_batch.item()) * incr
        ctx.round_seen_local     += incr

        # 정확도 계산을 쓰고 싶다면 아래 두 줄 켜기 (현재 ctx.y_true/ctx.y_prob 는 IGNORE_INDEX 제거됨)
        pred    = torch.argmax(ctx.y_prob, dim=-1)
        correct = int((pred == ctx.y_true).sum().item())
        ctx.round_correct_local  += correct

        # ---- split별 누적(옵션: 모니터/메트릭 호환) ----
        loss_total_key  = f'loss_total_{split}'
        num_samples_key = f'num_samples_{split}'
        correct_key     = f'correct_{split}'
        if not hasattr(ctx, loss_total_key):   setattr(ctx, loss_total_key, 0.0)
        if not hasattr(ctx, num_samples_key):  setattr(ctx, num_samples_key, 0)
        if not hasattr(ctx, correct_key):      setattr(ctx, correct_key, 0)
        setattr(ctx, loss_total_key,  getattr(ctx, loss_total_key)  + float(ctx.loss_batch.item()) * incr)
        setattr(ctx, num_samples_key, getattr(ctx, num_samples_key) + incr)
        setattr(ctx, correct_key,     getattr(ctx, correct_key)     + correct)

        # 평가 모드일 때만 raw 예측을 저장(옵션)
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
            if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
            ctx.ys_true.append(ctx.y_true)
            ctx.ys_pred.append(pred)


    # def _hook_on_batch_end(self, ctx):
    #     if ctx.get('skip_this_batch', False):
    #         return

    #     split = ctx.cur_split
    #     loss_total_key  = f'loss_total_{split}'
    #     correct_key     = f'correct_{split}'
    #     num_samples_key = f'num_samples_{split}'

    #     # 안전 초기화 (Context는 dict 인덱싱이 아니라 attr 사용)
    #     if not hasattr(ctx, loss_total_key):   setattr(ctx, loss_total_key, 0.0)
    #     if not hasattr(ctx, correct_key):      setattr(ctx, correct_key, 0)
    #     if not hasattr(ctx, num_samples_key):  setattr(ctx, num_samples_key, 0)

    #     # 이 트레이너에선 ctx.y_true/ctx.y_prob가 이미 IGNORE_INDEX 제거됨
    #     n_valid = ctx.y_true.numel()  # 유효 토큰 수

    #     # 손실은 유효 토큰 수로 가중합
    #     setattr(ctx, loss_total_key, getattr(ctx, loss_total_key) + ctx.loss_batch.item() * n_valid)

    #     # 정확도 (유효 토큰 단위)
    #     pred = torch.argmax(ctx.y_prob, dim=-1)       # [N_valid]
    #     correct = (pred == ctx.y_true).sum().item()
    #     setattr(ctx, correct_key, getattr(ctx, correct_key) + correct)

    #     # 분모도 유효 토큰 수로 누적
    #     setattr(ctx, num_samples_key, getattr(ctx, num_samples_key) + n_valid)

    #     # 평가 모드일 때만 원시 예측값을 보관(옵션)
    #     if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
    #         if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
    #         if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
    #         ctx.ys_true.append(ctx.y_true)
    #         ctx.ys_pred.append(pred)





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

    # # ================== 2. _hook_on_fit_end 는 비워둠 ==================
    # def _hook_on_fit_end(self, ctx):
    #     # 이 훅은 아무것도 하지 않습니다. _run_routine이 모든 것을 처리합니다.
    #     pass    

    def _hook_on_fit_end(self, ctx):
        # split별 누적치를 generic 키에도 매핑해서 client의 Results_raw가 0이 되지 않도록 함
        split = getattr(ctx, "cur_split", None)
        if split in ["train", "test", "val"]:
            n = getattr(ctx, f"num_samples_{split}", 0)
            loss_total = getattr(ctx, f"loss_total_{split}", 0.0)
            correct = getattr(ctx, f"correct_{split}", 0)

            # generic 키 (client가 Results_raw 만들 때 참조)
            ctx.total = int(n)
            ctx.loss = float(loss_total)
            ctx.avg_loss = float(loss_total / max(n, 1))
            ctx.acc = float(correct / max(n, 1))

            # split별 키는 기존대로 유지됨 (예: val_total, val_loss, val_avg_loss, val_acc)
        # super()는 호출하지 말 것: 상위가 평균/키를 다시 덮어씀


            
def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer':
        trainer_builder = RewardChoiceTrainer
        return trainer_builder


register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
