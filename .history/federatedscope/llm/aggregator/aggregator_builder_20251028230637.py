"""
Builder for LLM‑specific aggregators.

This module defines a registry that maps string identifiers used in YAML
configuration to actual aggregator classes.  It mirrors the generic
aggregator_builder but registers Full‑MoE and Fusion‑MoE aggregators.
"""

import importlib
from typing import Tuple, Dict, Any

AGGREGATOR_CLASS_DICT: Dict[str, Tuple[str, str]] = {
    "llmfullmoeaggregator": (
        "federatedscope.llm.aggregator.full_moe_aggregator",
        "FullMoEAggregator",
    ),
    "llmfusionmoeaggregator": (
        "federatedscope.llm.aggregator.fusion_moe_aggregator",
        "FusionMoEAggregator",
    ),
}

def get_llm_aggregator(type_name: str) -> Any:
    """Return the aggregator class for the given type."""
    key = type_name.lower()
    if key not in AGGREGATOR_CLASS_DICT:
        raise KeyError(f"Unknown LLM aggregator type: {type_name}")
    module_path, class_name = AGGREGATOR_CLASS_DICT[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
