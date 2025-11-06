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
        clients_data = agg_info.get("client_feedback", [])


        # ① 추출: default 파라미터 키
        first_para = clients_data[0]["model_para"]
        default_keys = [k for k in first_para if "default" in k]

        # ② 어댑터 개수 U는 w 길이나 config에서 가져옴
        U = len(clients_data[0].get("w", []))

        aggregated_params = {}
        for u in range(U):
            params_list = []
            for fb in clients_data:
                model_para = fb["model_para"]
                w_vec = fb["w"]
                weight = float(w_vec[u])  # w_{u,m}
                # default -> Adapter_u 로 이름 변환
                mapped = {}
                for k in default_keys:
                    new_k = k.replace("default", f"Adapter_{u}")
                    mapped[new_k] = model_para[k]
                params_list.append((weight, mapped))
            # ③ 각 u에 대해 가중평균
            aggregated_subset = self._aggregate_param(params_list)
            aggregated_params.update(aggregated_subset)

            
        return aggregated_params