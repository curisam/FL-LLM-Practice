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

        """
        clients_data= [{
            "model_para": state_dict_m,  # 모델 파라미터. 클라이언트가 모든 어댑터를 보냄.
            "sample_num": n_m,           # (선택) 샘플 수
            "w": [w_0,m, w_1,m, ... w_U-1,m],      # (선택) 전문가 가중치 벡터
                        },...
        ]
        """


        clients_data: List[Dict[str, Any]] = agg_info.get("client_feedback", []) 

        # 1단계: 어댑터 키 그룹핑
        first_params = clients_data[0]["model_para"] #0번쨰 client의 state_dict
        adapter_keys = {} #adapter_keys는 { "0":[Active_0.*], "1":[Active_1.*], ..., "default":[default.*] }.
        for k in first_params.keys():
            if "Active_" in k:
                # Parse index after "Active_"
                if len(parts) > 1:
                    u_idx = k.split("Active_")[1].split(".")[0]  # 예: "Active_3.lora_A...." → "3"
                    adapter_keys.setdefault(u_idx, []).append(k)  
            elif "default" in k:
                adapter_keys.setdefault("default", []).append(k)


        #2단계: 어댑터 u에 대한 클라이언트별 (weight, params_subset) 생성

        aggregated_params: Dict[str, Any] = {}
        # Aggregate each adapter separately
        for adapter_idx, keys in adapter_keys.items():
            params_list = []
            for fb in clients_data:
                model_para = fb["model_para"] #단일 client local model
                weight = None
                # 가중치 결정
                w_vec = fb["w"] # [w_0,m, w_1,m, ... w_U-1,m]
                try:
                    u = int(adapter_idx)
                    weight = float(w_vec[u])
                except Exception:
                    #default 등 정수가 아닐 떄는 sum(w)를 사용("전체" 바중)
                    weight = float(sum(w_vec)) if isinstance(w_vec, (list, tuple)) else 1.0

                # 이 어댑터의 Parameter subset만 뽑기.
                params_subset = {k: model_para[k] for k in keys if k in model_para}
                params_list.append((weight, params_subset))

            #3단계: 어댑터 u의 파라미터 평균 → 전체 결과에 병합  
            aggregated_subset = self._aggregate_param(params_list)
            aggregated_params.update(aggregated_subset)
        return aggregated_params
