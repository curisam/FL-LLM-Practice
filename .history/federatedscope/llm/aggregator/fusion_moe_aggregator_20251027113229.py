"""
Server‑side aggregator for Fusion‑MoE training.

Clients upload a fused default adapter.  The aggregator collects and
averages only the ``default`` adapter across clients, weighted by
``w`` (sum of client weights) or ``sample_num`` if provided.
"""

import logging
from typing import Dict, Any, List, Tuple
from federatedscope.core.aggregators.clients_avg_aggregator import (
    ClientsAvgAggregator,
)

logger = logging.getLogger(__name__)


class FusionMoEAggregator(ClientsAvgAggregator):
    """Aggregate only the fused default adapter across clients."""

    def _aggregate_param(self, params_list: List[Tuple[float, Dict[str, Any]]]):
        """Helper for weighted averaging."""
        aggregated = {}
        total_weight = 0.0
        for weight, param_dict in params_list:
            total_weight += weight
            for k, v in param_dict.items():
                if aggregated.get(k) is None:
                    aggregated[k] = v.clone() * weight
                else:
                    aggregated[k] += v * weight
        for k in aggregated:
            aggregated[k] /= total_weight
        return aggregated

    def aggregate(self, agg_info: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Aggregate the fused default adapter."""

        """
        clients_data= [{
            "model_para": state_dict_m,  # 모델 파라미터. 클라이언트가 로컬에서 이미 fusion해서 default만 보냄
            "sample_num": n_m,           # (선택) 샘플 수  
            "w": [w_0m, w_1m, ...],      # (선택) 전문가 가중치 벡터
                        },...
        ]
        """

        clients_data: List[Dict[str, Any]] = agg_info.get("client_feedback", [])
        if not clients_data:
            logger.warning("No client feedback provided.")
            return {}

        # Identify default adapter keys (containing "default")
        first_params = clients_data[0]["model_para"]
        default_keys = [k for k in first_params if "default" in k]
        if not default_keys:
            default_keys = [k for k in first_params if "Active_0" in k]

        params_list: List[Tuple[float, Dict[str, Any]]] = []
        for fb in clients_data:
            model_para = fb["model_para"]
            weight = None
            if fb.get("w") is not None:
                w_vec = fb["w"]
                # Use sum of w as weight for fused model
                weight = float(sum(w_vec)) if isinstance(w_vec, (list, tuple)) else float(w_vec)
            elif fb.get("sample_num") is not None:
                weight = float(fb["sample_num"])
            params_subset = {
                k: model_para[k]
                for k in default_keys if k in model_para
            }
            params_list.append((weight, params_subset))
        return self._aggregate_param(params_list)
