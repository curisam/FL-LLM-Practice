"""
Full‑MoE trainer that:
- Activates each adapter one by one and computes classification loss/metrics.
- Sums per-adapter losses weighted by w_{u,m} to form ctx.loss_task.
- Aggregates per-adapter statistics using the same weights.
- Uses the original LLMTrainer-style backward logic (accelerator/deepspeed aware).
"""

import logging
from typing import List, Optional
import torch

from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.llm.dataset.llm_dataset import DefaultToken
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

logger = logging.getLogger(__name__)

def cal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    choices: torch.Tensor
):
    """RewardChoiceTrainer와 동일한 분류 손실 함수."""
    choices = choices.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    new_labels = torch.full_like(
        shift_labels, DefaultToken.IGNORE_INDEX.value
    )
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx
    restricted_logits = shift_logits[..., choices]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(
        restricted_logits.view(-1, len(choices)),
        new_labels.view(-1)
    )
    return restricted_logits, new_labels, loss

class FullMoETrainer(LLMTrainer):
    """
    Full‑MoE용 트레이너:
      - 각 LoRA 어댑터를 순차적으로 활성화
      - 분류 손실 및 통계를 계산하고 클러스터 가중치로 합산
      - backward 단계는 가중합 손실을 사용하여 한 번만 수행
    """

    def __init__(self, model, data, device, config,
                 only_for_eval: bool = False,
                 monitor: Optional[object] = None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        # cfg.llm.adapter.count 값으로 어댑터 이름 생성
        count = int(getattr(config.llm.adapter, "count"))

        self.adapter_names: List[str] = [
            f"Adapter_{i}" for i in range(count)
            if f"Adapter_{i}".lower() != "default"
        ]


    def _hook_on_batch_forward(self, ctx) -> None:
        """어댑터별 forward와 통계를 계산하고, 가중합 손실을 ctx.loss_task에 저장."""
        input_ids = ctx.data_batch["input_ids"].to(ctx.device)
        labels = ctx.data_batch["labels"].to(ctx.device)
        attention_mask = ctx.data_batch["attention_mask"].to(ctx.device)

        try:
            original_adapter = ctx.model.get_active_adapter()
        except Exception:
            original_adapter = None

        # 가중치 벡터 w_{u,m}
        w_vec = getattr(ctx, "w_vec", None)
        total_weight = sum(w_vec) if sum(w_vec) > 0 else 1.0

        weighted_loss = 0.0
        weighted_correct = 0.0
        weighted_y_prob = None

        for idx, adapter_name in enumerate(self.adapter_names):
            ctx.model.set_active_adapter(adapter_name)

            if ctx.cfg.llm.accelerator.use: #참고, #ctx.model 기반으로 outputs 뽑아낸다.
                outputs = ctx.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            elif ctx.cfg.llm.deepspeed.use: #ctx.model_engine 기반으로 outputs 뽑아낸다.
                outputs = ctx.model_engine(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            else: #참고, #ctx.model 기반으로 outputs 뽑아낸다.
                outputs = ctx.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )


            new_logits, new_labels, loss = cal_loss(
                outputs.logits, labels, self.choices)

            # 비정상 손실 방어
            if not torch.isfinite(loss):
                ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
                logger.warning(f"Skip batch: non-finite loss={loss.item()}")
                return
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

            # weight 계산
            weight = float(w_vec[idx]) / total_weight
            weighted_loss = weighted_loss + weight * loss

            # 통계 계산 및 가중합
            IGN = DefaultToken.IGNORE_INDEX.value# 보통 -100
            valid_mask = (new_labels != IGN)# 유효 토큰 위치만 True
            has_valid = valid_mask.any(dim=1)# 각 샘플에 유효 토큰이 있는지 여부

            B = new_labels.size(0)
            arangeB = torch.arange(B, device=new_labels.device)#[0,1, ..., B-1]
            first_idx = torch.argmax(valid_mask.int(), dim=1)#샘플별 처음으로 유효 토큰에 등장하는 위치
            
            sel_b = arangeB[has_valid]#유효 토큰 있는 Index들만
            sel_t = first_idx[has_valid]#유효 토큰 있는 샘프들 기준 첫 유효 토큰 위치 집합.
            if sel_b.numel() > 0:
                logits_1st = new_logits[sel_b, sel_t, :] #new logit 중 유효 토큰 있는 것들 한해서 전체 vocab 중 classifier에 해당하는 축소된 logit 반환. 즉  (|sel_b|, 1, C) 차원.
                labels_1st = new_labels[sel_b, sel_t] #|sel_b|, 1 Classification true label만 뽑음.
                pred_1st = torch.argmax(logits_1st, dim=-1) #|sel_b| 개에 대해 예측 label 뽑음.
                
                sample_correct = int((pred_1st == labels_1st).sum().item()) #맞춘 갯수
                sample_count = int(has_valid.sum().item()) #전체 sample 갯수.
            else:
                sample_correct, sample_count = 0, 0

            weighted_correct += weight * sample_correct #배치 내에서 유효한 것 중 맞춘 것 갯수


            # 예측 확률 가중합
            flat_labels = new_labels.view(-1)
            flat_logits = new_logits.view(-1, new_logits.size(-1))

            keep = (flat_labels != IGN) #무시 하지말아야할 토큰 찾기.
            flat_logits = flat_logits[keep, :] #무시해야할 토큰들만 제거 #[N_valid, V]
            flat_labels = flat_labels[keep] #무시해야할 토큰들만 제거 #[N_valid]

            if weighted_y_prob is None:
                weighted_y_prob = weight * flat_logits
            else:
                weighted_y_prob = weighted_y_prob + weight * flat_logits

        # 원래 어댑터 복구
        if original_adapter is not None:
            ctx.model.set_active_adapter(original_adapter)

        # Save per-adapter losses for backward
        ctx.adapter_losses = adapter_losses

        ctx.sample_correct_batch = int(weighted_correct) #배치 내에서 유효한 것 중 맞춘 것 갯수
        ctx.sample_count_batch = float(sample_count) #배치 내의 유효한 샘플 수.

        predicted = torch.argmax(weighted_y_prob, dim=1)
        ctx.y_true = CtxVar(flat_labels.cpu(), LIFECYCLE.BATCH) #[N_valid]
        ctx.y_pred = CtxVar(predicted.cpu(), LIFECYCLE.BATCH) #[N_valid]
        ctx.y_prob = CtxVar(weighted_y_prob.cpu(), LIFECYCLE.BATCH) #[N_valid, V]

        ctx.loss_batch = CtxVar(weighted_loss.item(), LIFECYCLE.BATCH) #Scalar.
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH) #B. 일반적으로 B=N_valid.

   def _hook_on_batch_backward(self, ctx) -> None:
        """
        저장해 둔 adapter_losses를 사용해 각 어댑터를 개별적으로 업데이트합니다.
        - w_vec가 제공되면 loss * (w_vec[idx] / sum(w_vec))로 가중치를 적용합니다.
        - Accelerate/DeepSpeed 환경에 따라 분기합니다.
        - 마지막에 training data를 CPU로 이동합니다.
        """
        if bool(getattr(ctx, "skip_this_batch", False)):
            return

        # w_vec 로드: 길이가 어댑터 수와 다르면 균등 가중치
        w_vec = getattr(ctx, "w_vec", None)
        if not w_vec or len(w_vec) != len(self.adapter_names):
            w_vec = [1.0 for _ in self.adapter_names]
        total_weight = sum(w_vec) if sum(w_vec) > 0 else 1.0

        # 원래 활성 어댑터 저장
        try:
            original_adapter = ctx.model.get_active_adapter()
        except Exception:
            original_adapter = None

        for idx, (adapter_name, loss) in enumerate(
            zip(self.adapter_names, ctx.adapter_losses)
        ):
            if loss is None:
                continue

            # 해당 adapter 활성화
            ctx.model.set_active_adapter(adapter_name)

            # gradient 초기화
            if hasattr(ctx.model, "zero_grad"):
                ctx.model.zero_grad()
            elif hasattr(self, "optimizer"):
                self.optimizer.zero_grad()

            # 가중치 적용
            weight = float(w_vec[idx]) / total_weight
            weighted_loss = weight * loss

            # Accelerate 또는 DeepSpeed 환경 여부에 따른 backward/step
            if ctx.cfg.llm.accelerator.use:
                # accelerate가 grad accumulation 및 DDP 통신을 관리
                self.accelerator.backward(weighted_loss)
                if getattr(self.accelerator, "sync_gradients", True):
                    # gradient clipping (선택)
                    if getattr(ctx, "grad_clip", 0) and ctx.grad_clip > 0:
                        try:
                            self.accelerator.clip_grad_norm_(
                                ctx.model.parameters(), ctx.grad_clip
                            )
                        except Exception:
                            torch.nn.utils.clip_grad_norm_(
                                ctx.model.parameters(), ctx.grad_clip
                            )
                    # optimizer step & scheduler
                    ctx.optimizer.step()
                    if ctx.scheduler is not None:
                        ctx.scheduler.step()
                    ctx.optimizer.zero_grad()
                    # 글로벌 업데이트 카운트
                    self._global_updates += 1
                    # 중간 평가 트리거 (옵션)
                    if self._ct_ft and self._mid_eval_every > 0 \
                            and (self._global_updates % self._mid_eval_every) == 0:
                        self._mid_eval_pending = True
            else:
                # 기본 PyTorch backward with grad_accum_step
                (weighted_loss / self.grad_accum_step).backward()
                # grad_accum_step 경계에서만 update
                if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                    if ctx.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            ctx.model.parameters(), ctx.grad_clip
                        )
                    ctx.optimizer.step()
                    if ctx.scheduler is not None:
                        ctx.scheduler.step()
                    ctx.optimizer.zero_grad()
                    # 글로벌 업데이트 카운트
                    self._global_updates += 1
                    if self._ct_ft and self._mid_eval_every > 0 \
                            and (self._global_updates % self._mid_eval_every) == 0:
                        self._mid_eval_pending = True

        # 원래 어댑터로 복구
        if original_adapter is not None:
            try:
                ctx.model.set_active_adapter(original_adapter)
            except Exception:
                pass

        # 마지막으로 training data를 CPU로 이동하여 메모리 관리
        db = getattr(ctx, "data_batch", None)
        if isinstance(db, dict):
            for k in ('input_ids', 'labels', 'attention_mask'):
                t = db.get(k, None)
                if torch.is_tensor(t):
                    db[k] = t.detach().to('cpu', non_blocking=True)
