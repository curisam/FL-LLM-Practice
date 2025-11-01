"""
Aggregators for LoRA‑based federated learning under Full‑MoE and Fusion‑MoE regimes.

In Full‑MoE the server averages each adapter independently across clients,
while in Fusion‑MoE the server only aggregates the fused default adapter.
"""

from .full_moe_aggregator import FullMoEAggregator
from .fusion_moe_aggregator import FusionMoEAggregator
