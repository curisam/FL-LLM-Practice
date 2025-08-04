import os
import gc
import sys
import math
import copy
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from transformers import AdamW

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.dataset.llm_dataset import DefaultToken
from federatedscope.llm.model.adapter_builder import AdapterModel

logger = logging.getLogger(__name__)
sys.setrecursionlimit(100000)


# ------------------------- Utilities -------------------------
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


def cal_loss(logits, labels, choices):
    """
    Reward-choice용 CE:
      - logits: [B, S, V]
      - labels: [B, S]
      - choices: [C] (vocab ids)
    """
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


# ------------------------- Trainer -------------------------
class RewardChoiceTrainer(LLMTrainer):
    """
    - forward: reward-choice CE만 계산
    - 배치/샘플 집계는 LLMTrainer(부모)가 수행 (논리 배치 = raw_bs // n_choices)
    - 여기서는 '정확도'만 별도 누적 후 라운드 말에 reduce & 기록
    """

    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        # choices 토큰 id 준비 (CPU에 보관; 라운드 시작 시 디바이스로 옮김)
        try:
            choices_list = []
            for choice in getattr(config.trainer, "choices", []):
                token_ids = self.tokenizer(f': {choice}', add_special_tokens=False)['input_ids']
                if not token_ids:
                    raise ValueError(f"Tokenizer returned empty list for choice: '{choice}'")
                choices_list.append(token_ids[-1])
            if not choices_list:
                raise ValueError("config.trainer.choices is empty or invalid.")
            self._choices_cpu = torch.tensor(choices_list, dtype=torch.long)  # CPU 텐서로 보관
            logger.info(f"[reward-choice] choice token ids={self._choices_cpu.tolist()}")
        except Exception as e:
            logger.error(f"Failed to initialize trainer.choices: {e}")
            raise

    # ---------------- Train/Eval entry: rebuild loaders to avoid dup eval ----------------
    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        self._reset_and_build_dataloader(target_data_split_name)
        return super().train(target_data_split_name, hooks_set)

    def evaluate(self, target_data_split_name="test", hooks_set=None):
        self._reset_and_build_dataloader(target_data_split_name)
        return super().evaluate(target_data_split_name, hooks_set)

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
                    sampler = DistributedSampler(
                        base_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
                    )
                    self._train_dist_sampler = sampler
                    loader = DataLoader(
                        dataset=base_loader.dataset,
                        batch_size=base_loader.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=getattr(base_loader, 'num_workers', 0),
                        pin_memory=getattr(base_loader, 'pin_memory', False),
                        drop_last=getattr(base_loader, 'drop_last', False),
                        collate_fn=getattr(base_loader, 'collate_fn', None)
                    )
                else:
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

            client_data_obj[loader_key] = loader
            setattr(self.ctx, ctx_loader_key, loader)
            logger.info(
                f"Dataloader for '{split_name}' has been reset and recreated. "
                f"(sharded={sharded}, world_size={world_size}, rank={rank}, "
                f"local_count={local_count}, total={len(base_loader.dataset)})"
            )

    # ---------------- Fit lifecycle overrides ----------------
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        # choices → device
        try:
            if not hasattr(self, "_choices_cpu"):
                raise RuntimeError("choices CPU tensor missing.")
            self.choices = self._choices_cpu.to(ctx.device, non_blocking=True)
        except Exception as e:
            logger.error(f"[reward-choice] move choices to device failed: {e}")
            raise

        # 정확도 누적 카운터 (샘플 기준; 분모는 '논리 샘플 수'와 동일)
        ctx.sample_seen = 0
        ctx.sample_correct = 0

        # train sampler 에폭 고정
        sampler = getattr(self, "_train_dist_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            epoch = int(getattr(ctx, "cur_state", 0))
            sampler.set_epoch(epoch)

    def _hook_on_batch_forward(self, ctx):
        # 1) 모델 forward
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

        # 2) reward-choice CE
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)

        # 3) NaN 가드
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to NaN loss.')
            return
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # 4) 논리 배치 크기 산정(부모 로직 사용; 선택지 힌트가 있으므로 정확)
        raw_bs = int(labels.size(0))
        inferred = self._infer_n_choices_from_batch(ctx.data_batch, raw_bs)
        if inferred < 1:
            inferred = 1
        if raw_bs % inferred == 0:
            logical_bs = raw_bs // inferred
        else:
            logical_bs = raw_bs
            logger.warning(f"[logical-batch warn] raw_bs={raw_bs}, inferred_n_choices={inferred} → use raw_bs.")

        # 5) 정확도(샘플 기준, '시퀀스당 첫 유효 토큰'으로 판단)
        IGN = DefaultToken.IGNORE_INDEX.value
        valid_mask = (new_labels != IGN)            # [B, S-1]
        has_valid  = valid_mask.any(dim=1)          # [B]
        B = new_labels.size(0)
        arangeB = torch.arange(B, device=new_labels.device)
        first_idx = torch.argmax(valid_mask.int(), dim=1)  # [B]
        sel_b = arangeB[has_valid]
        sel_t = first_idx[has_valid]
        if sel_b.numel() > 0:
            logits_1st = new_logits[sel_b, sel_t, :]   # [Bv, C]
            labels_1st = new_labels[sel_b, sel_t]      # [Bv]
            pred_1st   = torch.argmax(logits_1st, dim=-1)
            batch_correct = int((pred_1st == labels_1st).sum().item())
            batch_seen    = int(has_valid.sum().item())
        else:
            batch_correct, batch_seen = 0, 0

        # 6) 부모 집계용 필수 컨텍스트 설정
        ctx.y_true = CtxVar(new_labels.view(-1)[new_labels.view(-1) != IGN], LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(new_logits.view(-1, new_logits.size(-1))[new_labels.view(-1) != IGN], LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.loss_task  = CtxVar(loss, LIFECYCLE.BATCH)

        # ✅ 논리 배치 크기만 설정(부모가 이 값을 사용해 샘플 수 집계)
        ctx.batch_size = CtxVar(int(logical_bs), LIFECYCLE.BATCH)

        # 디버깅 정보(첫 배치에서 부모가 로그 찍음)
        ctx.debug_raw_bs = raw_bs
        ctx.debug_inferred_choices = inferred

        # 임시 정확도 결과(우리만 사용) 저장
        ctx._tmp_batch_seen = batch_seen if logical_bs > 0 else 0
        ctx._tmp_batch_correct = batch_correct if logical_bs > 0 else 0

    def _hook_on_batch_backward(self, ctx):
        # 부모의 옵티마이저/스케줄러/accumulate 로직 사용
        return super()._hook_on_batch_backward(ctx)

    def _hook_on_batch_end(self, ctx):
        """
        - 정확도 카운트만 누적
        - 샘플 수/로스 합산은 부모(LBMTrainer)가 논리 배치 기준으로 수행함
        """
        skip = bool(getattr(ctx, "skip_this_batch", False))
        if skip:
            return

        # 정확도 카운트만 누적 (누락/중복 방지)
        add_seen = int(getattr(ctx, "_tmp_batch_seen", 0))
        add_corr = int(getattr(ctx, "_tmp_batch_correct", 0))
        ctx.sample_seen = int(getattr(ctx, "sample_seen", 0)) + add_seen
        ctx.sample_correct = int(getattr(ctx, "sample_correct", 0)) + add_corr

        # 부모의 배치 종료 훅(샘플 수/로스 합산) 호출
        super()._hook_on_batch_end(ctx)

    def _ddp_reduce_pair(self, ctx, v_seen: int, v_corr: int):
        """
        (seen, correct)를 all-reduce(sum)하여 rank0에서 최종값 반환
        """
        acc = getattr(self, "accelerator", None)
        device = ctx.device
        t_seen = torch.tensor([int(v_seen)], device=device, dtype=torch.long)
        t_corr = torch.tensor([int(v_corr)], device=device, dtype=torch.long)
        if acc is not None and acc.num_processes > 1:
            t_seen = acc.reduce(t_seen, reduction="sum")
            t_corr = acc.reduce(t_corr, reduction="sum")
        return int(t_seen.item()), int(t_corr.item())

    def _hook_on_fit_end(self, ctx):
        """
        - 부모: 샘플 수/로스 예상치 검증 로그 출력 (expected vs actual)
        - 여기: 정확도(샘플 기준)를 reduce하여 eval_metrics에 기록
        """
        super()._hook_on_fit_end(ctx)

        split = getattr(ctx, "cur_split", None)
        if split not in ("train", "val", "test"):
            return

        # 정확도 집계 (전 프로세스 합)
        total_seen, total_corr = self._ddp_reduce_pair(
            ctx,
            int(getattr(ctx, "sample_seen", 0)),
            int(getattr(ctx, "sample_correct", 0))
        )
        acc_val = (float(total_corr) / max(1, int(total_seen))) if total_seen > 0 else 0.0

        # 메트릭 기록 (부모는 loss/total을 기록; 여기에 acc 추가)
        if not hasattr(ctx, "eval_metrics") or ctx.eval_metrics is None:
            ctx.eval_metrics = {}
        ctx.eval_metrics[f"{split}_acc"] = acc_val
        logger.info(f"[reward-choice|{split}] sample_acc={acc_val:.6f} "
                    f"(seen={total_seen}, correct={total_corr})")


# ------------------------- Register -------------------------
def call_reward_choice_trainer(trainer_type):
    if trainer_type in ['llmrewardchoicetrainer', 'rewardchoicetrainer', 'reward_choice_trainer']:
        return RewardChoiceTrainer

register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
