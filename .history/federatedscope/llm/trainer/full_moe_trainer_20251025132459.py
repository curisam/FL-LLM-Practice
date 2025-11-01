"""
Full‑MoE trainer with per‑adapter classification loss and independent updates.

이 트레이너는 각 어댑터를 순차적으로 활성화하여 분류 손실을 계산하고,
즉시 backward()와 optimizer.step()을 호출하여 해당 어댑터만 업데이트합니다.
RewardChoiceTrainer의 cal_loss와 동일한 로직을 cal_classification_loss()로 구현했습니다.
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

from federatedscope.llm.trainer.trainer import LLMTrainer

IGNORE_INDEX: int = -100
logger = logging.getLogger(__name__)


def cal_loss(logits: torch.Tensor,
                            labels: torch.Tensor,
                            choices: torch.Tensor) -> torch.Tensor:
    """
    input_ids, labels가 아래와 같이 주어졌다 가정.
    [<BOS>, ▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>]   (T = 7)

    logits

    [▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>, -100]
    
    """


    # 1) next-token 셋업: (t-1, t) 정렬
    choices = choices.to(logits.device) 
    shift_logits = logits[..., :-1, :].contiguous() # [B, T-1, V]. EOS 에 대한 예측하는 POSITION만 제외. V중 Max [▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>]
    shift_labels = labels[..., 1:].contiguous() # [B, T-1]. BOS 에 대한 예측하는 POSITION만 제외. [▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>]


    # 2) 라벨 맵핑: A/B 토큰이 등장한 자리만 클래스 인덱스(0/1)로 바꾸고 나머지는 IGN(-100
    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value) #[-100, -100, -100, -100, -100, -100]으로 초기화
    for idx, choice in enumerate(choices): # 예: [ID('▁A'), ID('▁B')]
        mask = (shift_labels == choice); new_labels[mask] = idx
    #mask->[False, False, False, False,  True, False]
    #new_labels = [-100, -100, -100, -100, 0, -100]


    # 예: [ID('▁A'), ID('▁B')] 관한 걸로만 logit space 축소.
    new_logits = shift_logits[..., choices]
    loss_fn = torch.nn.CrossEntropyLoss() #torch.nn.CrossEntropyLoss()의 기본 ignore_index가 -100이라서, 타깃에 -100이 들어 있으면 그 위치들은 손실/그라디언트 계산에서 제외돼요. 평균도 유효한 항만으로 나눕니다.

    # 4) (B×(T-1), K) vs (B×(T-1))로 펼쳐 CrossEntropy
    loss = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1)) #실질적으로 t=4인 지점에 대해서만 ce 계산됨.
    
    return new_logits, new_labels, loss

class FullMoETrainer(LLMTrainer):
    """Full‑MoE용 트레이너(분류 손실 + per‑adapter 업데이트)."""

    def __init__(self, model, data, device, config,
                 only_for_eval: bool = False,
                 monitor: Optional[object] = None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        # LoRA adapter 이름 구성 (Active_0 ~ Active_{count-1})
        try:
            count = int(getattr(config.llm.adapter, "count"))
        except Exception:
            count = 0
        self.adapter_names: List[str] = [
            f"Active_{i}" for i in range(count)
            if f"Active_{i}".lower() != "default"
        ]
        if not self.adapter_names:
            logger.warning("FullMoETrainer initialised with no adapters.")

        # 선택지 토큰 ID 로드 (없으면 분류 손실 대신 모델 기본 loss 사용)
        choices_attr = None
        try:
            if hasattr(config.llm, "choices"):
                choices_attr = getattr(config.llm, "choices")
            elif hasattr(config.llm.adapter, "choices"):
                choices_attr = getattr(config.llm.adapter, "choices")
        except Exception:
            choices_attr = None
        if choices_attr is not None:
            try:
                self.choices = torch.tensor(list(choices_attr),
                                            dtype=torch.long)
            except Exception:
                logger.warning("Cannot parse llm.choices; fallback to default loss.")
                self.choices = None
        else:
            self.choices = None

    def _hook_on_batch_forward(self, ctx) -> None:
        """
        모든 어댑터에 대해 forward 수행 후 loss를 저장.
        분류 손실(choices 사용)을 계산하며, backward/step은 여기서 하지 않습니다.
        """
        input_ids = ctx.data_batch["input_ids"].to(ctx.device)
        labels = ctx.data_batch["labels"].to(ctx.device)
        attention_mask = ctx.data_batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(ctx.device)

        # 현재 활성 어댑터를 저장
        try:
            original_adapter = ctx.model.get_active_adapter()
        except Exception:
            original_adapter = None

        adapter_losses = []
        total_loss_val = 0.0

        # 각 어댑터마다 forward 및 loss 계산(분류 혹은 기본 loss)
        for adapter_name in self.adapter_names:
            try:
                ctx.model.set_active_adapter(adapter_name)
            except Exception as e:
                logger.warning(f"Cannot activate adapter {adapter_name}: {e}")
                adapter_losses.append(None)
                continue

            # accelerator/deepspeed 여부에 따라 forward
            if ctx.cfg.llm.accelerator.use:
                outputs = ctx.model(
                    input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask
                )
            elif ctx.cfg.llm.deepspeed.use:
                outputs = ctx.model_engine(
                    input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask
                )
            else:
                outputs = ctx.model(
                    input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask
                )

            # classification loss 또는 default loss
            if self.choices is not None:
                logits = outputs.logits
                loss = cal_classification_loss(logits, labels, self.choices)
            else:
                loss = outputs.loss

            adapter_losses.append(loss)
            total_loss_val += float(loss.item())

        # 원래 어댑터로 복구
        if original_adapter is not None:
            try:
                ctx.model.set_active_adapter(original_adapter)
            except Exception:
                pass

        # 다음 backward hook에서 사용하도록 저장
        ctx.adapter_losses = adapter_losses
        # 로그 기록용 평균 loss
        ctx.loss = (total_loss_val / len(self.adapter_names)
                    if self.adapter_names else total_loss_val)

    def _hook_on_batch_backward(self, ctx) -> None:
        """
        저장된 각 어댑터의 loss에 대해 순차적으로 backward 및 optimizer.step 실행.
        클러스터 가중치 `ctx.w_vec`가 있으면 loss에 곱합니다.
        """
        # 가중치 벡터 w_{u,m} (없으면 균등 가중치)
        w_vec = getattr(ctx, "w_vec", None)
        if not w_vec or len(w_vec) != len(self.adapter_names):
            w_vec = [1.0 for _ in self.adapter_names]
        total_weight = sum(w_vec) if sum(w_vec) > 0 else 1.0

        # 현재 어댑터 저장
        try:
            original_adapter = ctx.model.get_active_adapter()
        except Exception:
            original_adapter = None

        for idx, (adapter_name, loss) in enumerate(
            zip(self.adapter_names, ctx.adapter_losses)
        ):
            if loss is None:
                continue
            try:
                ctx.model.set_active_adapter(adapter_name)
            except Exception as e:
                logger.warning(f"Cannot re-activate adapter {adapter_name}: {e}")
                continue

            # gradient 초기화
            if hasattr(ctx.model, "zero_grad"):
                ctx.model.zero_grad()
            elif hasattr(self, "optimizer"):
                self.optimizer.zero_grad()

            # w_{u,m} 적용
            weight = float(w_vec[idx]) / total_weight
            (weight * loss).backward()

            # optimizer 스텝
            if hasattr(self, "optimizer"):
                self.optimizer.step()
            else:
                if ctx.cfg.llm.deepspeed.use:
                    ctx.model_engine.step()

        # 원래 어댑터 복구
        if original_adapter is not None:
            try:
                ctx.model.set_active_adapter(original_adapter)
            except Exception:
                pass

        # 마지막에 gradients 초기화(선택)
        if hasattr(ctx.model, "zero_grad"):
            ctx.model.zero_grad()
        elif hasattr(self, "optimizer"):
            self.optimizer.zero_grad()
