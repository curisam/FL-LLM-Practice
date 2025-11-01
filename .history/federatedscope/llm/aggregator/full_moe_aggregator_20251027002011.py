"""
Server‑side aggregator for Full‑MoE training.

Each client sends its state_dict for all adapters.  The aggregator
performs a weighted average for each adapter separately, using client
weights ``w`` (if provided) or ``sample_num`` otherwise.
"""

import logging
from typing import Dict, List, Any, Tuple
from federatedscope.core.aggregators.clients_avg_aggregator import (
    ClientsAvgAggregator,
)

logger = logging.getLogger(__name__)


class FullMoEAggregator(ClientsAvgAggregator):
    """Aggregate LoRA adapters separately with optional per‑adapter weights."""

    def _aggregate_param(self, params_list: List[Tuple[float, Dict[str, Any]]]):
        """Helper: weighted average over dictionaries."""
        if not params_list:
            return {}
        aggregated = {}
        total_weight = 0.0
        for weight, param_dict in params_list:
            if weight is None:
                weight = 1.0
            total_weight += weight
            for k, v in param_dict.items():
                if aggregated.get(k) is None:
                    aggregated[k] = v.clone() * weight
                else:
                    aggregated[k] += v * weight
        for k in aggregated:
            aggregated[k] /= max(total_weight, 1e-12)
        return aggregated

    def aggregate(self, agg_info: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Aggregate adapter parameters across clients."""
        clients_data: List[Dict[str, Any]] = agg_info.get("client_feedback", []) #content= (sample_size, model)들을 담은 리스트
        if not clients_data:
            logger.warning("No client feedback provided.")
            return {}

        # Determine adapter keys (Active_u or default) from first client
        first_params = clients_data[0]["model_para"]
        adapter_keys = {}
        for k in first_params.keys():
            if "Active_" in k:
                # Parse index after "Active_"
                parts = k.split("Active_")
                if len(parts) > 1:
                    prefix = parts[1].split(".")[0]
                    adapter_keys.setdefault(prefix, []).append(k)
            elif "default" in k:
                adapter_keys.setdefault("default", []).append(k)

        aggregated_params: Dict[str, Any] = {}
        # Aggregate each adapter separately
        for adapter_idx, keys in adapter_keys.items():
            params_list = []
            for fb in clients_data:
                model_para = fb["model_para"]
                weight = None
                if fb.get("w") is not None:
                    w_vec = fb["w"]
                    # Choose weight for this adapter u
                    try:
                        u = int(adapter_idx)
                        weight = float(w_vec[u])
                    except Exception:
                        # Default: use sum of w
                        weight = float(sum(w_vec)) if isinstance(w_vec, (list, tuple)) else 1.0
                elif fb.get("sample_num") is not None:
                    weight = float(fb["sample_num"])
                params_subset = {k: model_para[k] for k in keys if k in model_para}
                params_list.append((weight, params_subset))
            aggregated_subset = self._aggregate_param(params_list)
            aggregated_params.update(aggregated_subset)
        return aggregated_params
