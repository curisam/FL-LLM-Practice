# import os
# import gc
# import sys
# import math
# import copy
# import logging
# import numpy as np

# import torch
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.utils.data import DataLoader, Sampler
# from torch.utils.data.distributed import DistributedSampler
# from transformers import AdamW

# from federatedscope.register import register_trainer
# from federatedscope.llm.trainer.trainer_arxivarxiv import LLMTrainer
# from federatedscope.core.trainers.context import CtxVar, lifecycle
# from federatedscope.core.trainers.enums import MODE, LIFECYCLE
# from federatedscope.core.auxiliaries.decorators import use_diff
# from federatedscope.core.data.wrap_dataset import WrapDataset
# from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
# from federatedscope.core.auxiliaries.ReIterator import ReIterator
# from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
# from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
# from federatedscope.core.monitors.monitor_arxiv import Monitor

# from federatedscope.llm.dataset.llm_dataset import DefaultToken
# from federatedscope.llm.model.adapter_builder import AdapterModel

# from torch.optim.lr_scheduler import MultiStepLR

# logger = logging.getLogger(__name__)
# sys.setrecursionlimit(100000)

# # ------------------------------
# # 분산 상태 헬퍼
# # ------------------------------
# def _get_dist_info():
#     if dist.is_available() and dist.is_initialized():
#         return True, dist.get_world_size(), dist.get_rank()
#     try:
#         ws = int(os.environ.get('WORLD_SIZE', '1'))
#         rk = int(os.environ.get('RANK', '0'))
#         return (ws > 1), ws, rk
#     except Exception:
#         return False, 1, 0

# class EvalShardSampler(Sampler):
#     """평가 전용 샘플러: 중복/패딩 없이 rank마다 고유 인덱스만 반환"""
#     def __init__(self, dataset_len: int, rank: int, world_size: int):
#         self.indices = list(range(rank, int(dataset_len), int(world_size)))
#     def __iter__(self):
#         return iter(self.indices)
#     def __len__(self):
#         return len(self.indices)

# # ------------------------------
# # 선택지 기반 로스
# # ------------------------------
# def cal_loss(logits, labels, choices):
#     """
#     logits: [B, S, V], labels: [B, S], choices: LongTensor(C,) (디바이스 일치)
#     """
#     choices = choices.to(logits.device)
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()

#     new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
#     for idx, choice in enumerate(choices):
#         mask = (shift_labels == choice)
#         new_labels[mask] = idx

#     new_logits = shift_logits[..., choices]
#     loss_fn = torch.nn.CrossEntropyLoss()
#     loss = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1))
#     return new_logits, new_labels, loss

# # ------------------------------
# # 트레이너
# # ------------------------------
# class RewardChoiceTrainer(LLMTrainer):
#     def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
#         super().__init__(model, data, device, config, only_for_eval, monitor)

#         # choices 토큰 ID 준비 (CPU 텐서로 보관, 실제 사용 시 디바이스에 올림)
#         try:
#             choices_list = []
#             for choice in config.trainer.choices:
#                 token_ids = self.tokenizer(f': {choice}', add_special_tokens=False)['input_ids']
#                 if not token_ids:
#                     raise ValueError(f"Tokenizer returned empty list for choice: '{choice}'")
#                 choices_list.append(token_ids[-1])
#             self.choices_cpu = torch.tensor(choices_list, dtype=torch.long)  # CPU 보관본
#             logger.info(f'Choice token IDs: {self.choices_cpu.tolist()}')
#         except Exception as e:
#             logger.error(f"Error during trainer initialization: {e}")
#             raise ValueError('Failed to initialize trainer.choices.')

#     # --- dataloader를 라운드마다 재빌드 (train/eval 공용) ---
#     def _reset_and_build_dataloader(self, split_name):
#         data_key = f"{split_name}_data"
#         loader_key = split_name
#         ctx_loader_key = f"{split_name}_loader"
#         client_data_obj = self.ctx.data

#         if loader_key in client_data_obj:
#             del client_data_obj[loader_key]

#         if hasattr(client_data_obj, data_key) and getattr(client_data_obj, data_key) is not None:
#             dataset = WrapDataset(getattr(client_data_obj, data_key))
#             base_loader = get_dataloader(dataset, self.cfg, split_name)

#             dist_ready, world_size, rank = _get_dist_info()

#             if world_size > 1:
#                 if split_name == 'train':
#                     sampler = DistributedSampler(
#                         base_loader.dataset,
#                         num_replicas=world_size,
#                         rank=rank,
#                         shuffle=True
#                     )
#                     self._train_dist_sampler = sampler
#                     loader = DataLoader(
#                         dataset=base_loader.dataset,
#                         batch_size=base_loader.batch_size,
#                         sampler=sampler,
#                         shuffle=False,
#                         num_workers=getattr(base_loader, 'num_workers', 0),
#                         pin_memory=getattr(base_loader, 'pin_memory', False),
#                         drop_last=getattr(base_loader, 'drop_last', False),
#                         collate_fn=getattr(base_loader, 'collate_fn', None)
#                     )
#                 else:
#                     sampler = EvalShardSampler(len(base_loader.dataset), rank, world_size)
#                     loader = DataLoader(
#                         dataset=base_loader.dataset,
#                         batch_size=base_loader.batch_size,
#                         sampler=sampler,
#                         shuffle=False,
#                         num_workers=0,
#                         pin_memory=False,
#                         drop_last=False,
#                         collate_fn=getattr(base_loader, 'collate_fn', None)
#                     )
#                 sharded = True
#                 local_count = len(sampler)
#             else:
#                 loader = base_loader
#                 sharded = False
#                 local_count = len(base_loader.dataset)

#             client_data_obj[loader_key] = loader
#             setattr(self.ctx, ctx_loader_key, loader)
#             logger.info(
#                 f"Dataloader for '{split_name}' has been reset and recreated. "
#                 f"(sharded={sharded}, world_size={1 if not sharded else world_size}, "
#                 f"rank={0 if not sharded else rank}, local_count={local_count}, "
#                 f"total={len(base_loader.dataset)})"
#             )

#     # --- train/eval 진입 시 로더 리빌드 ---
#     @use_diff
#     def train(self, target_data_split_name="train", hooks_set=None):
#         # FL 라운드 기준으로만 스케줄러 step
#         if hasattr(self, "scheduler"):
#             # ctx에 저장된 round index를 가져오되, 없으면 0
#             round_idx = getattr(self.ctx, "cur_state")
#             print(round_idx)
#             self.scheduler.step(round_idx)
#             lr = self.scheduler.get_last_lr()[0]
#             logger.info(f"[LRScheduler] Round {round_idx} → lr={lr:.2e}")


#         self._reset_and_build_dataloader(target_data_split_name)
#         return super().train(target_data_split_name, hooks_set)

#     def evaluate(self, target_data_split_name="test", hooks_set=None):
#         hooks_set = hooks_set or self.hooks_in_eval
#         self._reset_and_build_dataloader(target_data_split_name)
#         with torch.no_grad():
#             if self.ctx.check_split(target_data_split_name, skip=True):
#                 self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
#             else:
#                 self.ctx.eval_metrics = dict()
#         return self.ctx.eval_metrics

#     # --- 라운드 시작 초기화 ---
#     def _hook_on_fit_start_init(self, ctx):
#         super()._hook_on_fit_start_init(ctx)

#         # choices를 디바이스로
#         self.choices = self.choices_cpu.to(ctx.device, non_blocking=True)

#         # split별 누적치
#         for sp in ["train", "val", "test"]:
#             setattr(ctx, f"num_samples_{sp}", 0)
#             setattr(ctx, f"loss_total_{sp}", 0.0)
#             setattr(ctx, f"correct_{sp}", 0)

#         # 부모 루틴 호환용 누적치
#         ctx.num_samples = 0
#         ctx.loss_batch_total = 0.0
#         ctx.loss_regular_total = 0.0

#         # 정확도(논리 샘플 기준) 누적치
#         ctx.sample_seen = 0
#         ctx.sample_correct_accum = 0

#         # DDP sampler 에폭 고정
#         sampler = getattr(self, "_train_dist_sampler", None)
#         if sampler is not None and hasattr(sampler, "set_epoch"):
#             epoch = int(getattr(ctx, "cur_state", 0))
#             sampler.set_epoch(epoch)

#         # ── ctx.optimizer(또는 ctx.get('optimizer'))에서 옵티마이저 가져와 스케줄러 초기화 ──
#         sched_cfg = getattr(self.cfg.train, "scheduler", None)
#         if sched_cfg and getattr(sched_cfg, "type", "") == "MultiStepLR":
#             # context에 붙어있는 optimizer를 먼저 시도
#             opt = getattr(ctx, "optimizer", None)
#             if opt is None:
#                 # fallback: 혹시 self.optimizer에 있는지
#                 opt = getattr(self, "optimizer", None)
#             if opt is None:
#                 logger.warning("[LRScheduler] 옵티마이저를 찾을 수 없어 스케줄러를 생성하지 않습니다.")
#             else:
#                 self.scheduler = MultiStepLR(
#                     opt,
#                     milestones=sched_cfg.milestones,
#                     gamma=sched_cfg.gamma
#                 )




#     # --- 배치 forward ---
#     def _hook_on_batch_forward(self, ctx):
#         # 입력 준비
#         input_ids = ctx.data_batch['input_ids'].to(ctx.device)
#         labels = ctx.data_batch['labels'].to(ctx.device)
#         attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

#         if ctx.cfg.llm.accelerator.use:
#             outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
#         elif ctx.cfg.llm.deepspeed.use:
#             outputs = ctx.model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
#         else:
#             outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

#         logits = outputs.logits
#         new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)

#         if torch.isnan(loss):
#             ctx.skip_this_batch = True
#             logger.warning('Skip the batch due to NaN loss.')
#             return
#         else:
#             ctx.skip_this_batch = False

#         # ---- 논리 샘플 기준 정확도 계산 ----
#         IGN = DefaultToken.IGNORE_INDEX.value
#         valid_mask = (new_labels != IGN)     # [B, S-1]
#         has_valid  = valid_mask.any(dim=1)   # [B] 시퀀스 단위 유효여부

#         B = new_labels.size(0)
#         arangeB = torch.arange(B, device=new_labels.device)
#         first_idx = torch.argmax(valid_mask.int(), dim=1)
#         sel_b = arangeB[has_valid]
#         sel_t = first_idx[has_valid]

#         if sel_b.numel() > 0:
#             logits_1st = new_logits[sel_b, sel_t, :]  # [Bv, C]
#             labels_1st = new_labels[sel_b, sel_t]     # [Bv]
#             pred_1st   = torch.argmax(logits_1st, dim=-1)
#             sample_correct = int((pred_1st == labels_1st).sum().item())
#             sample_count   = int(has_valid.sum().item())  # 논리 샘플 수
#         else:
#             sample_correct, sample_count = 0, 0

#         ctx.sample_correct_batch = sample_correct
#         ctx.sample_count_batch   = sample_count

#         # ---- 토큰 단위(옵션) 저장 ----
#         flat_labels = new_labels.view(-1)
#         flat_logits = new_logits.view(-1, new_logits.size(-1))
#         keep = (flat_labels != IGN)
#         flat_logits = flat_logits[keep, :]
#         flat_labels = flat_labels[keep]
#         _, predicted = flat_logits.max(1)

#         ctx.y_true = CtxVar(flat_labels, LIFECYCLE.BATCH)
#         ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
#         ctx.y_prob = CtxVar(flat_logits, LIFECYCLE.BATCH)

#         ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
#         ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)  # == raw micro-batch size

#     # --- 배치 종료: 누적/집계 ---
#     def _hook_on_batch_end(self, ctx):
#         skip = getattr(ctx, "skip_this_batch", False)
#         if isinstance(skip, CtxVar):
#             try:
#                 skip = bool(skip)
#             except Exception:
#                 skip = False
#         if skip:
#             return

        
#         # ── LR 디버그 로그: 안전하게 옵티마이저 찾기 ──
#         opt = None
#         # 1) Trainer에 직접 붙어 있을 경우
#         if hasattr(self, "optimizer"):
#             opt = self.optimizer
#         # 2) Context에 붙어 있을 경우
#         elif hasattr(ctx, "optimizer"):
#             opt = ctx.optimizer
#         # 3) Accelerator wrapping을 쓴다면 Accelerator에 있을 수도
#         elif hasattr(self, "accelerator"):
#             # accelerate v0.18+ 기준: accelerator._optimizer_groups
#             try:
#                 # 첫 번째 optimizer group의 optimizer
#                 opt = self.accelerator._optimizer_groups[0]["optimizer"]
#             except Exception:
#                 opt = None

#         if opt is not None:
#             current_lr = opt.param_groups[0]["lr"]
#             # round_idx가 없을 수도 있으니 getattr + 기본값 처리
#             round_idx = getattr(ctx, "cur_state",
#                          getattr(ctx, "round_idx", "?"))
#             batch_idx = getattr(ctx, "batch_idx", "?")
#             logger.debug(
#                 f"[LR DEBUG] round={round_idx} split={ctx.cur_split} "
#                 f"batch_idx={batch_idx} lr={current_lr:.2e}"
#             )
#         else:
#             logger.debug("[LR DEBUG] optimizer 객체를 찾을 수 없어 LR을 출력하지 않았습니다.")


#         split = ctx.cur_split
#         loss_total_key  = f'loss_total_{split}'
#         correct_key     = f'correct_{split}'
#         num_samples_key = f'num_samples_{split}'
#         if not hasattr(ctx, loss_total_key):   setattr(ctx, loss_total_key, 0.0)
#         if not hasattr(ctx, correct_key):      setattr(ctx, correct_key, 0)
#         if not hasattr(ctx, num_samples_key):  setattr(ctx, num_samples_key, 0)

#         # --- 중요 ---
#         # 집계 표본 수는 "raw 시퀀스 수"로 유지해야 과거와 동일한 480 집계가 나온다.
#         batch_raw_samples = int(ctx.batch_size)                 # e.g., 2
#         batch_logic_seen  = int(getattr(ctx, "sample_count_batch", 0))   # e.g., 1 (논리)
#         batch_logic_corr  = int(getattr(ctx, "sample_correct_batch", 0))

#         # split별 누적(로스는 'raw 시퀀스 수' 가중합으로)
#         setattr(ctx, loss_total_key,
#                 getattr(ctx, loss_total_key) + ctx.loss_batch.item() * batch_raw_samples)
#         setattr(ctx, num_samples_key,
#                 getattr(ctx, num_samples_key) + batch_raw_samples)

#         # 논리 샘플 기준 정확도 누적 (별도)
#         ctx.sample_seen             = int(getattr(ctx, "sample_seen", 0)) + batch_logic_seen
#         ctx.sample_correct_accum    = int(getattr(ctx, "sample_correct_accum", 0)) + batch_logic_corr

#         # 부모 루틴 호환(서버 가중치 등)
#         ctx.num_samples        = int(getattr(ctx, "num_samples", 0)) + batch_raw_samples
#         ctx.loss_batch_total   = float(getattr(ctx, "loss_batch_total", 0.0)) + ctx.loss_batch.item() * batch_raw_samples
#         ctx.loss_regular_total = float(getattr(ctx, "loss_regular_total", 0.0)) + float(ctx.get("loss_regular", 0.0))

#         # 평가일 때만 원하면 추적
#         if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
#             if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
#             if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
#             pred = torch.argmax(ctx.y_prob, dim=-1)
#             ctx.ys_true.append(ctx.y_true)
#             ctx.ys_pred.append(pred)

#     # --- 전 프로세스 reduce 헬퍼 (split별) ---
#     def _ddp_reduce_split(self, ctx, split: str):
#         acc = getattr(self, "accelerator", None)
#         device = ctx.device

#         n        = torch.tensor([getattr(ctx, f"num_samples_{split}", 0)], device=device, dtype=torch.long)
#         loss_sum = torch.tensor([getattr(ctx, f"loss_total_{split}", 0.0)], device=device, dtype=torch.float32)

#         if acc is not None and acc.num_processes > 1:
#             n        = acc.reduce(n, reduction="sum")
#             loss_sum = acc.reduce(loss_sum, reduction="sum")

#         return int(n.item()), float(loss_sum.item())

#     # --- 라운드 종료: 메트릭 작성 ---
#     def _hook_on_fit_end(self, ctx):
#         split = getattr(ctx, "cur_split", None)
#         if split not in ("train", "val", "test"):
#             return

#         acc = getattr(self, "accelerator", None)
#         is_main = (acc.is_main_process if acc is not None else True)
#         device  = ctx.device

#         # 1) raw 시퀀스 기준 표본/로스 합산 (과거 480 집계 재현)
#         n_global, loss_global = self._ddp_reduce_split(ctx, split)
#         avg_global = float(loss_global / max(n_global, 1))

#         # 2) 정확도는 논리 샘플 기준 합산
#         t_seen = torch.tensor([int(getattr(ctx, "sample_seen", 0))], device=device, dtype=torch.long)
#         t_corr = torch.tensor([int(getattr(ctx, "sample_correct_accum", 0))], device=device, dtype=torch.long)
#         if acc is not None and acc.num_processes > 1:
#             t_seen = acc.reduce(t_seen, reduction="sum")
#             t_corr = acc.reduce(t_corr, reduction="sum")
#         seen_global = int(t_seen.item())
#         corr_global = int(t_corr.item())
#         acc_global  = float(corr_global / max(seen_global, 1)) if seen_global > 0 else 0.0

#         # 3) 결과 기록 (main만)
#         if is_main:
#             if not hasattr(ctx, "eval_metrics") or ctx.eval_metrics is None:
#                 ctx.eval_metrics = {}
#             ctx.eval_metrics.update({
#                 f"{split}_total":    n_global,
#                 f"{split}_loss":     loss_global,
#                 f"{split}_avg_loss": avg_global,
#                 f"{split}_acc":      acc_global,
#             })
#             logger.info(
#                 f"[{split}|reduce] total={n_global}, loss_sum={loss_global:.6f}, "
#                 f"avg_loss={avg_global:.6f}, acc={acc_global:.6f} "
#                 f"(local_seen={getattr(ctx,'sample_seen',0)}, "
#                 f"local_corr={getattr(ctx,'sample_correct_accum',0)})"
#             )
#         else:
#             ctx.eval_metrics = {}

# # ---- registry ----
# def call_reward_choice_trainer(trainer_type):
#     if trainer_type == 'llmrewardchoicetrainer':
#         return RewardChoiceTrainer

# register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)



import os
import gc
import sys
import math
import copy
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer_arxivarxiv import LLMTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.monitors.monitor_arxiv import Monitor

from federatedscope.llm.dataset.llm_dataset import DefaultToken
from federatedscope.llm.model.adapter_builder import AdapterModel

from torch.optim.lr_scheduler import MultiStepLR

logger = logging.getLogger(__name__)
sys.setrecursionlimit(100000)

# ------------------------------
# (이하 헬퍼 함수들은 변경 없음)
# ------------------------------
def _get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    try:
        ws = int(os.environ.get('WORLD_SIZE', '1'))
        rk = int(os.environ.get('RANK', '0'))
        return (ws > 1), ws, rk
    except Exception:
        return False, 1, 0

class EvalShardSampler(Sampler):
    def __init__(self, dataset_len: int, rank: int, world_size: int):
        self.indices = list(range(rank, int(dataset_len), int(world_size)))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def cal_loss(logits, labels, choices):
    choices = choices.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        mask = (shift_labels == choice)
        new_labels[mask] = idx
    new_logits = shift_logits[..., choices]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1))
    return new_logits, new_labels, loss

# ------------------------------
# 트레이너
# ------------------------------
class RewardChoiceTrainer(LLMTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        try:
            choices_list = []
            for choice in config.trainer.choices:
                token_ids = self.tokenizer(f': {choice}', add_special_tokens=False)['input_ids']
                if not token_ids:
                    raise ValueError(f"Tokenizer returned empty list for choice: '{choice}'")
                choices_list.append(token_ids[-1])
            self.choices_cpu = torch.tensor(choices_list, dtype=torch.long)
            logger.info(f'Choice token IDs: {self.choices_cpu.tolist()}')
        except Exception as e:
            logger.error(f"Error during trainer initialization: {e}")
            raise ValueError('Failed to initialize trainer.choices.')

    def _reset_and_build_dataloader(self, split_name):
        data_key = f"{split_name}_data"
        loader_key = split_name
        ctx_loader_key = f"{split_name}_loader"
        client_data_obj = self.ctx.data
        if loader_key in client_data_obj:
            del client_data_obj[loader_key]
        if hasattr(client_data_obj, data_key) and getattr(client_data_obj, data_key) is not None:
            dataset = WrapDataset(getattr(client_data_obj, data_key))
            base_loader = get_dataloader(dataset, self.cfg, split_name)
            dist_ready, world_size, rank = _get_dist_info()
            if world_size > 1:
                if split_name == 'train':
                    sampler = DistributedSampler(base_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
                    self._train_dist_sampler = sampler
                    loader = DataLoader(dataset=base_loader.dataset, batch_size=base_loader.batch_size, sampler=sampler, shuffle=False, num_workers=getattr(base_loader, 'num_workers', 0), pin_memory=getattr(base_loader, 'pin_memory', False), drop_last=getattr(base_loader, 'drop_last', False), collate_fn=getattr(base_loader, 'collate_fn', None))
                else:
                    sampler = EvalShardSampler(len(base_loader.dataset), rank, world_size)
                    loader = DataLoader(dataset=base_loader.dataset, batch_size=base_loader.batch_size, sampler=sampler, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=getattr(base_loader, 'collate_fn', None))
                sharded = True
                local_count = len(sampler)
            else:
                loader = base_loader
                sharded = False
                local_count = len(base_loader.dataset)
            client_data_obj[loader_key] = loader
            setattr(self.ctx, ctx_loader_key, loader)
            logger.info(f"Dataloader for '{split_name}' has been reset and recreated. (sharded={sharded}, world_size={1 if not sharded else world_size}, rank={0 if not sharded else rank}, local_count={local_count}, total={len(base_loader.dataset)})")

    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        self._reset_and_build_dataloader(target_data_split_name)
        return super().train(target_data_split_name, hooks_set)

    def evaluate(self, target_data_split_name="test", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_eval
        self._reset_and_build_dataloader(target_data_split_name)
        with torch.no_grad():
            if self.ctx.check_split(target_data_split_name, skip=True):
                self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
            else:
                self.ctx.eval_metrics = dict()
        return self.ctx.eval_metrics

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)

        # ==================== Stateless LR 제어 로직 (진짜 최종 수정본) ====================

        # 1. 현재 FL 라운드 번호를 `self.cfg`에서 직접 가져옵니다.
        current_round = self.cfg.federate.round_num

        scheduler_cfg = getattr(self.cfg.train, "scheduler", None)
        initial_lr = self.cfg.train.optimizer.lr
        new_lr = initial_lr

        if scheduler_cfg:
            milestones = getattr(scheduler_cfg, "milestones", [])
            gamma = getattr(scheduler_cfg, "gamma", 1.0)
            
            # 2. 현재 라운드(0부터 시작)가 몇 개의 milestone을 지났는지 계산합니다.
            num_decays = 0
            for milestone in sorted(milestones):
                if current_round >= milestone:
                    num_decays += 1
            
            new_lr = initial_lr * (gamma ** num_decays)
            
            opt = getattr(ctx, "optimizer", None) or getattr(self, "optimizer", None)
            if opt:
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
                logger.info(
                    f"[Stateless LR Controller] Round #{current_round}: Milestones={milestones}, "
                    f"Gamma={gamma}. Num decays={num_decays}. LR is set to {new_lr:.2e}"
                )
            else:
                logger.warning("[Stateless LR Controller] Optimizer not found. Could not set LR.")

        # =================================================================================

        self.choices = self.choices_cpu.to(ctx.device, non_blocking=True)
        for sp in ["train", "val", "test"]:
            setattr(ctx, f"num_samples_{sp}", 0)
            setattr(ctx, f"loss_total_{sp}", 0.0)
            setattr(ctx, f"correct_{sp}", 0)
        ctx.num_samples = 0
        ctx.loss_batch_total = 0.0
        ctx.loss_regular_total = 0.0
        ctx.sample_seen = 0
        ctx.sample_correct_accum = 0

        sampler = getattr(self, "_train_dist_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(current_round)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        if ctx.cfg.llm.accelerator.use:
            outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        elif ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        logits = outputs.logits
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)
        if torch.isnan(loss):
            ctx.skip_this_batch = True
            logger.warning('Skip the batch due to NaN loss.')
            return
        else:
            ctx.skip_this_batch = False
        IGN = DefaultToken.IGNORE_INDEX.value
        valid_mask = (new_labels != IGN)
        has_valid  = valid_mask.any(dim=1)
        B = new_labels.size(0)
        arangeB = torch.arange(B, device=new_labels.device)
        first_idx = torch.argmax(valid_mask.int(), dim=1)
        sel_b = arangeB[has_valid]
        sel_t = first_idx[has_valid]
        if sel_b.numel() > 0:
            logits_1st = new_logits[sel_b, sel_t, :]
            labels_1st = new_labels[sel_b, sel_t]
            pred_1st   = torch.argmax(logits_1st, dim=-1)
            sample_correct = int((pred_1st == labels_1st).sum().item())
            sample_count   = int(has_valid.sum().item())
        else:
            sample_correct, sample_count = 0, 0
        ctx.sample_correct_batch = sample_correct
        ctx.sample_count_batch   = sample_count
        flat_labels = new_labels.view(-1)
        flat_logits = new_logits.view(-1, new_logits.size(-1))
        keep = (flat_labels != IGN)
        flat_logits = flat_logits[keep, :]
        flat_labels = flat_labels[keep]
        _, predicted = flat_logits.max(1)
        ctx.y_true = CtxVar(flat_labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(flat_logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_end(self, ctx):
        skip = getattr(ctx, "skip_this_batch", False)
        if isinstance(skip, CtxVar):
            try: skip = bool(skip)
            except Exception: skip = False
        if skip: return
        opt = None
        if hasattr(self, "optimizer"): opt = self.optimizer
        elif hasattr(ctx, "optimizer"): opt = ctx.optimizer
        elif hasattr(self, "accelerator"):
            try: opt = self.accelerator._optimizer_groups[0]["optimizer"]
            except Exception: opt = None
        if opt is not None:
            current_lr = opt.param_groups[0]["lr"]
            round_idx = self.cfg.federate.round_num
            batch_idx = getattr(ctx, "batch_idx", "?")
            logger.debug(f"[LR DEBUG] round={round_idx} split={ctx.cur_split} batch_idx={batch_idx} lr={current_lr:.2e}")
        else:
            logger.debug("[LR DEBUG] optimizer 객체를 찾을 수 없어 LR을 출력하지 않았습니다.")
        split = ctx.cur_split
        loss_total_key  = f'loss_total_{split}'
        correct_key     = f'correct_{split}'
        num_samples_key = f'num_samples_{split}'
        if not hasattr(ctx, loss_total_key): setattr(ctx, loss_total_key, 0.0)
        if not hasattr(ctx, correct_key): setattr(ctx, correct_key, 0)
        if not hasattr(ctx, num_samples_key): setattr(ctx, num_samples_key, 0)
        batch_raw_samples = int(ctx.batch_size)
        batch_logic_seen  = int(getattr(ctx, "sample_count_batch", 0))
        batch_logic_corr  = int(getattr(ctx, "sample_correct_batch", 0))
        setattr(ctx, loss_total_key, getattr(ctx, loss_total_key) + ctx.loss_batch.item() * batch_raw_samples)
        setattr(ctx, num_samples_key, getattr(ctx, num_samples_key) + batch_raw_samples)
        ctx.sample_seen = int(getattr(ctx, "sample_seen", 0)) + batch_logic_seen
        ctx.sample_correct_accum = int(getattr(ctx, "sample_correct_accum", 0)) + batch_logic_corr
        ctx.num_samples = int(getattr(ctx, "num_samples", 0)) + batch_raw_samples
        ctx.loss_batch_total = float(getattr(ctx, "loss_batch_total", 0.0)) + ctx.loss_batch.item() * batch_raw_samples
        ctx.loss_regular_total = float(getattr(ctx, "loss_regular_total", 0.0)) + float(ctx.get("loss_regular", 0.0))
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            if not hasattr(ctx, 'ys_true'): ctx.ys_true = []
            if not hasattr(ctx, 'ys_pred'): ctx.ys_pred = []
            pred = torch.argmax(ctx.y_prob, dim=-1)
            ctx.ys_true.append(ctx.y_true)
            ctx.ys_pred.append(pred)

    def _ddp_reduce_split(self, ctx, split: str):
        acc = getattr(self, "accelerator", None)
        device = ctx.device
        n = torch.tensor([getattr(ctx, f"num_samples_{split}", 0)], device=device, dtype=torch.long)
        loss_sum = torch.tensor([getattr(ctx, f"loss_total_{split}", 0.0)], device=device, dtype=torch.float32)
        if acc is not None and acc.num_processes > 1:
            n = acc.reduce(n, reduction="sum")
            loss_sum = acc.reduce(loss_sum, reduction="sum")
        return int(n.item()), float(loss_sum.item())

    def _hook_on_fit_end(self, ctx):
        split = getattr(ctx, "cur_split", None)
        if split not in ("train", "val", "test"): return
        acc = getattr(self, "accelerator", None)
        is_main = (acc.is_main_process if acc is not None else True)
        device  = ctx.device
        n_global, loss_global = self._ddp_reduce_split(ctx, split)
        avg_global = float(loss_global / max(n_global, 1))
        t_seen = torch.tensor([int(getattr(ctx, "sample_seen", 0))], device=device, dtype=torch.long)
        t_corr = torch.tensor([int(getattr(ctx, "sample_correct_accum", 0))], device=device, dtype=torch.long)
        if acc is not None and acc.num_processes > 1:
            t_seen = acc.reduce(t_seen, reduction="sum")
            t_corr = acc.reduce(t_corr, reduction="sum")
        seen_global = int(t_seen.item())
        corr_global = int(t_corr.item())
        acc_global  = float(corr_global / max(seen_global, 1)) if seen_global > 0 else 0.0
        if is_main:
            if not hasattr(ctx, "eval_metrics") or ctx.eval_metrics is None:
                ctx.eval_metrics = {}
            ctx.eval_metrics.update({f"{split}_total": n_global, f"{split}_loss": loss_global, f"{split}_avg_loss": avg_global, f"{split}_acc": acc_global})
            logger.info(f"[{split}|reduce] total={n_global}, loss_sum={loss_global:.6f}, avg_loss={avg_global:.6f}, acc={acc_global:.6f} (local_seen={getattr(ctx,'sample_seen',0)}, local_corr={getattr(ctx,'sample_correct_accum',0)})")
        else:
            ctx.eval_metrics = {}

def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer':
        return RewardChoiceTrainer

register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)