"""
Trainer implementation for Fusion‑MoE style federated learning.

The Fusion‑MoE regime maintains the same per‑client training behaviour as the
vanilla ``LLMTrainer``—each client activates a single LoRA adapter for local
updates.  The distinction lies on the server side, where updates across
clients are linearly combined (fused) according to weights from the E‑step.
"""

from federatedscope.llm.trainer.reward_choice_trainer import RewardChoiceTrainer


class FusionMoETrainer(RewardChoiceTrainer):
    """
    Fusion‑MoE does not require client‑side customisations.
    Clients still train one active adapter; fusion happens on the server.
    """
    pass
