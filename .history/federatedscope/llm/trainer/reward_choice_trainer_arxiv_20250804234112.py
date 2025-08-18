import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np
import math
from transformers import AdamW



from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer_arxivarxiv import LLMTrainer

from federatedscope.core.trainers.context import CtxVar, lifecycle

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.llm.dataset.llm_dataset import DefaultToken
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler


from federatedscope.core.monitors.monitor_arxiv import Monitor
from federatedscope.llm.model.adapter_builder import AdapterModel

from torch.optim.lr_scheduler import MultiStepLR



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
                    self._train_dist_sampler = sampler
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
        super()._hook_on_fit_start_init(ctx)

        # 라운드 집계(샘플 단위)
        ctx.round_seen_local     = 0
        ctx.round_sum_loss_local = 0.0
        ctx.round_correct_local  = 0

        # 모니터/상위 루틴 호환용 누적치도 초기화
        ctx.num_samples = 0
        ctx.loss_batch_total = 0.0
        ctx.loss_regular_total = 0.0

        ctx._seen_batch_sigs = set()
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

        # train sampler 있으면 epoch 고정
        sampler = getattr(self, "_train_dist_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            epoch = int(getattr(ctx, "cur_state", 0))
            sampler.set_epoch(epoch)

    def _hook_on_batch_forward(self, ctx):
        # 1) 입력/forward
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

        # 2) 선택지 분류용 loss/로짓
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)

        # 3) NaN 가드
        if torch.isnan(loss):
            ctx.skip_this_batch = True
            logger.warning('Skip the batch due to NaN loss.')
            return
        else:
            ctx.skip_this_batch = False

        # 4) ---- "샘플 단위" 정확도/카운트 계산 ----
        IGN = DefaultToken.IGNORE_INDEX.value
        valid_mask = (new_labels != IGN)           # [B, S-1]
        has_valid  = valid_mask.any(dim=1)         # [B]
        B = new_labels.size(0)
        arangeB = torch.arange(B, device=new_labels.device)
        first_idx = torch.argmax(valid_mask.int(), dim=1)  # [B]
        sel_b     = arangeB[has_valid]
        sel_t     = first_idx[has_valid]

        if sel_b.numel() > 0:
            logits_1st = new_logits[sel_b, sel_t, :]   # [Bv, C]
            labels_1st = new_labels[sel_b, sel_t]      # [Bv]
            pred_1st   = torch.argmax(logits_1st, dim=-1)
            sample_correct = (pred_1st == labels_1st).sum().item()
            sample_count   = int(has_valid.sum().item())
        else:
            sample_correct, sample_count = 0, 0

        ctx.sample_correct = sample_correct
        ctx.sample_count   = sample_count

        # 5) (옵션) 토큰 단위 정보도 유지
        new_labels = new_labels.view(-1)
        new_logits = new_logits.view(-1, len(self.choices))
        keep = (new_labels != IGN)
        new_logits = new_logits[keep, :]
        new_labels = new_labels[keep]
        _, predicted = new_logits.max(1)

        ctx.y_true = CtxVar(new_labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(new_logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)






    # def _hook_on_batch_forward(self, ctx):
    #     if ctx.cfg.llm.accelerator.use:
    #         input_ids = ctx.data_batch['input_ids'].to(ctx.device)
    #         labels = ctx.data_batch['labels'].to(ctx.device)
    #         attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
    #         outputs = ctx.model(input_ids=input_ids,
    #                             labels=labels,
    #                             attention_mask=attention_mask)

    #     elif ctx.cfg.llm.deepspeed.use:
    #         input_ids = ctx.data_batch['input_ids'].to(ctx.device)
    #         labels = ctx.data_batch['labels'].to(ctx.device)
    #         attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
    #         outputs = ctx.model_engine(input_ids=input_ids,
    #                                    labels=labels,
    #                                    attention_mask=attention_mask)

    #     else:
    #         input_ids = ctx.data_batch['input_ids'].to(ctx.device)
    #         labels = ctx.data_batch['labels'].to(ctx.device)
    #         attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
    #         outputs = ctx.model(input_ids=input_ids,
    #                             labels=labels,
    #                             attention_mask=attention_mask)

    #     logits = outputs.logits


    #     #이 부분이 바뀜. LLMTRAINER 특징. Classification.
    #     new_logits, new_labels, loss = cal_loss(logits, labels, self.choices) #일부 모델에서 reduction='none'이면 shape가 [batch_size] 혹은 [batch_size, seq_len]일 수도 있지만, 대부분은 1개의 스칼라 값. #[B, S-1, C], [B, S-1], Scalar


    #     if torch.isnan(loss):
    #         ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
    #         logger.warning('Skip the batch due to the loss is NaN, '
    #                        'it may be caused by exceeding the precision or '
    #                        'invalid labels.')
    #     else:
    #         ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

    #     # logger.info(f'{input_ids}')
    #     # logger.info(f'{labels}')

    #     new_labels = new_labels.view(-1) # [B*(S−1)]
    #     new_logits = new_logits.view(-1, len(self.choices)) # [B*(S−1), C]

    #     ##  “-100”(IGNORE_INDEX)로 마킹된 위치(=선택지가 아닌 토큰 위치) 제거
    #     new_logits = new_logits[(
    #         new_labels != DefaultToken.IGNORE_INDEX.value), :] ## [N_valid, C]
    #     new_labels = new_labels[(new_labels !=
    #                              DefaultToken.IGNORE_INDEX.value)]  # [N_valid]
    #     _, predicted = new_logits.max(1) # [N_valid]
    #     # logger.info(f'{predicted}, {new_labels}, {new_logits}')

    #     ctx.y_true = CtxVar(new_labels, LIFECYCLE.BATCH)
    #     ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
    #     ctx.y_prob = CtxVar(new_logits, LIFECYCLE.BATCH)

    #     ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
    #     ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)





    # def _hook_on_batch_end(self, ctx):
    #     if ctx.get('skip_this_batch', False):
    #         return

    #     split = ctx.cur_split  # 'train' / 'val' / 'test'

    #     # ---- 배치 샘플 수(=시퀀스 개수) ----
    #     # 위에서 _hook_on_batch_forward 에서: ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
    #     bs = int(ctx.batch_size)

    #     # ---- 선택지(pair) 수: 예) ['A','B'] -> 2 ----
    #     pair_factor = max(1, len(getattr(self, 'choices', [])))  # choices 텐서 길이

    #     # ---- 샘플 기준 누적 (한 번만!) ----
    #     incr = bs * pair_factor

    #     # 손실은 '샘플 기준 합'으로 누적 (loss가 배치 평균이면 loss * incr)
    #     ctx.round_sum_loss_local += float(ctx.loss_batch.item()) * incr
    #     ctx.round_seen_local     += incr

    #     # 정확도 계산을 쓰고 싶다면 아래 두 줄 켜기 (현재 ctx.y_true/ctx.y_prob 는 IGNORE_INDEX 제거됨)
    #     pred    = torch.argmax(ctx.y_prob, dim=-1)
    #     correct = int((pred == ctx.y_true).sum().item())
    #     ctx.round_correct_local  += correct

    #     # ---- split별 누적(옵션: 모니터/메트릭 호환) ----
    #     loss_total_key  = f'loss_total_{split}'
    #     num_samples_key = f'num_samples_{split}'
    #     correct_key     = f'correct_{split}'
    #     if not hasattr(ctx, loss_total_key):   setattr(ctx, loss_total_key, 0.0)
    #     if not hasattr(ctx, num_samples_key):  setattr(ctx, num_samples_key, 0)
    #     if not hasattr(ctx, correct_key):      setattr(ctx, correct_key, 0)
    #     setattr(ctx, loss_total_key,  getattr(ctx, loss_total_key)  + float(ctx.loss_batch.item()) * incr)
    #     setattr(ctx, num_samples_key, getattr(ctx, num_samples_key) + incr)
    #     setattr(ctx, correct_key,     getattr(ctx, correct_key)     + correct)

    #     # 평가 모드일 때만 raw 예측을 저장(옵션)
    #     if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
    #         if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
    #         if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
    #         ctx.ys_true.append(ctx.y_true)
    #         ctx.ys_pred.append(pred)


    def _hook_on_batch_end(self, ctx):
        skip = getattr(ctx, "skip_this_batch", False)
        if isinstance(skip, CtxVar):
            try:
                skip = bool(skip)
            except Exception:
                skip = False
        if skip:
            return

        split = ctx.cur_split
        loss_total_key  = f'loss_total_{split}'
        correct_key     = f'correct_{split}'
        num_samples_key = f'num_samples_{split}'
        if not hasattr(ctx, loss_total_key):   setattr(ctx, loss_total_key, 0.0)
        if not hasattr(ctx, correct_key):      setattr(ctx, correct_key, 0)
        if not hasattr(ctx, num_samples_key):  setattr(ctx, num_samples_key, 0)

        batch_samples = int(getattr(ctx, "sample_count", 0))
        batch_correct = int(getattr(ctx, "sample_correct", 0))

        # split별 누적
        setattr(ctx, loss_total_key,
                getattr(ctx, loss_total_key) + ctx.loss_batch.item() * batch_samples)
        setattr(ctx, correct_key,
                getattr(ctx, correct_key) + batch_correct)
        setattr(ctx, num_samples_key,
                getattr(ctx, num_samples_key) + batch_samples)

        # 상위 루틴/서버 가중치 용(중요!)
        ctx.num_samples       = int(getattr(ctx, "num_samples", 0)) + batch_samples
        ctx.loss_batch_total  = float(getattr(ctx, "loss_batch_total", 0.0)) + ctx.loss_batch.item() * batch_samples
        ctx.loss_regular_total = float(getattr(ctx, "loss_regular_total", 0.0)) + float(ctx.get("loss_regular", 0.0))

        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
            if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
            pred = torch.argmax(ctx.y_prob, dim=-1)
            ctx.ys_true.append(ctx.y_true)
            ctx.ys_pred.append(pred)



    def _ddp_reduce_split(self, ctx, split: str):
        """
        각 rank가 누적한 split별 통계를 all-reduce(sum) 해서 rank0만 최종값을 사용.
        반환: (n_total, loss_total, correct_total)
        """
        acc = getattr(self, "accelerator", None)
        device = ctx.device

        n         = torch.tensor([getattr(ctx, f"num_samples_{split}", 0)], device=device, dtype=torch.long)
        loss_sum  = torch.tensor([getattr(ctx, f"loss_total_{split}", 0.0)], device=device, dtype=torch.float32)
        correct   = torch.tensor([getattr(ctx, f"correct_{split}", 0)], device=device, dtype=torch.long)

        if acc is not None and acc.num_processes > 1:
            n        = acc.reduce(n, reduction="sum")
            loss_sum = acc.reduce(loss_sum, reduction="sum")
            correct  = acc.reduce(correct, reduction="sum")

        return int(n.item()), float(loss_sum.item()), int(correct.item()) 



    # def _hook_on_fit_end(self, ctx):
    #     split = getattr(ctx, "cur_split", None)
    #     if split not in ["train", "val", "test"]:
    #         return

    #     using_accel = (self.accelerator is not None)
    #     if using_accel:
    #         device = self.accelerator.device
    #         rank   = self.accelerator.process_index
    #         world  = self.accelerator.num_processes
    #     else:
    #         device = ctx.device
    #         rank, world = 0, 1

    #     # ---- 로컬 텐서화 ----
    #     t_seen = torch.tensor(int(ctx.round_seen_local),     device=device, dtype=torch.long)
    #     t_loss = torch.tensor(float(ctx.round_sum_loss_local), device=device, dtype=torch.float)
    #     t_corr = torch.tensor(int(ctx.round_correct_local),  device=device, dtype=torch.long)

    #     # ---- 전 랭크 합산(sum-only) ----
    #     if using_accel and world > 1:
    #         all_seen = self.accelerator.gather_for_metrics(t_seen)  # [world]
    #         all_loss = self.accelerator.gather_for_metrics(t_loss)  # [world]
    #         all_corr = self.accelerator.gather_for_metrics(t_corr)  # [world]
    #         global_seen = int(all_seen.sum().item())
    #         global_loss = float(all_loss.sum().item())
    #         global_corr = int(all_corr.sum().item())
    #         # (옵션) rank0에서 per-proc 스냅샷
    #         if rank == 0:
    #             per_list = []
    #             for i in range(world):
    #                 s = int(all_seen[i].item())
    #                 l = float(all_loss[i].item())
    #                 per_list.append({
    #                     f'{split}_total':   s,
    #                     f'{split}_loss':    l,
    #                     f'{split}_avg_loss': (l / max(s, 1)),
    #                 })
    #             ctx.per_proc_local_results = per_list
    #     else:
    #         global_seen = int(t_seen.item())
    #         global_loss = float(t_loss.item())
    #         global_corr = int(t_corr.item())

    #     global_avg = float(global_loss / max(global_seen, 1))
    #     global_acc = float(global_corr / max(global_seen, 1))  # 선택: 정답/샘플 기준

    #     # ---- 각 rank 로컬 스냅샷 (클라이언트 Local 로그용) ----
    #     ctx.local_results_for_log = {
    #         f'{split}_total':    int(ctx.round_seen_local),
    #         f'{split}_loss':     float(ctx.round_sum_loss_local),
    #         f'{split}_avg_loss': float(ctx.round_sum_loss_local / max(int(ctx.round_seen_local), 1)),
    #         # f'{split}_acc':   float(ctx.round_correct_local / max(int(ctx.round_seen_local), 1)),  # 원하면 포함
    #     }

    #     # ---- rank0: 최종 집계본을 ctx.eval_metrics에 기록 ----
    #     if (not using_accel) or self.accelerator.is_main_process:
    #         ctx.eval_metrics = getattr(ctx, 'eval_metrics', {}) or {}
    #         ctx.eval_metrics.update({
    #             f'{split}_total':    global_seen,
    #             f'{split}_loss':     global_loss,
    #             f'{split}_avg_loss': global_avg,
    #             f'{split}_acc':      global_acc,   # acc 쓰지 않으면 지워도 됨
    #         })



    def _hook_on_fit_end(self, ctx):
        split = getattr(ctx, "cur_split", None)
        if split not in ("train", "val", "test"):
            return

        n, loss_total, correct = self._ddp_reduce_split(ctx, split)

        acc = getattr(self, "accelerator", None)
        is_main = (acc.is_main_process if acc is not None else True)

        # generic 키: 상위/클라이언트 로깅 호환
        ctx.total    = int(n)
        ctx.loss     = float(loss_total)
        ctx.avg_loss = float(loss_total / max(n, 1))
        ctx.acc      = float(correct / max(n, 1))

        # (선택) 각 rank 로컬 스냅샷도 남기고 싶으면:
        ctx.local_results_for_log = {
            f'{split}_total':    int(getattr(ctx, f'num_samples_{split}', 0)),
            f'{split}_loss':     float(getattr(ctx, f'loss_total_{split}', 0.0)),
            f'{split}_avg_loss': float(getattr(ctx, f'loss_total_{split}', 0.0) / max(int(getattr(ctx, f'num_samples_{split}', 0)), 1)),
            # f'{split}_acc':   float(getattr(ctx, f'correct_{split}', 0) / max(int(getattr(ctx, f'num_samples_{split}', 0)), 1)),
        }

        if is_main:
            if not hasattr(ctx, "eval_metrics") or ctx.eval_metrics is None:
                ctx.eval_metrics = {}
            ctx.eval_metrics[f"{split}_total"]     = n
            ctx.eval_metrics[f"{split}_loss"]      = loss_total
            ctx.eval_metrics[f"{split}_avg_loss"]  = loss_total / max(n, 1)
            ctx.eval_metrics[f"{split}_acc"]       = correct / max(n, 1)
        else:
            ctx.eval_metrics = {}



            
def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer':
        trainer_builder = RewardChoiceTrainer
        return trainer_builder


register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
