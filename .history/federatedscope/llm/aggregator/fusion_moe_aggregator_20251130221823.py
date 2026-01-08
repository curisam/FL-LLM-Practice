"""
Server‑side aggregator for Fusion‑MoE training.

Clients upload a fused default adapter.  The aggregator collects and
averages only the ``default`` adapter across clients, weighted by
``w`` (sum of client weights) or ``sample_size`` if provided.
"""

import logging
from typing import Dict, Any, List, Tuple
from federatedscope.core.aggregators.clients_avg_aggregator import (
    ClientsAvgAggregator,
)

import torch

from federatedscope.core.auxiliaries.utils import param2tensor



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
            "sample_size": n_m,           # (선택) 샘플 수  
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


        logger.info(f"[DEBUG] first_para keys = {list(first_para.keys())}")

        logger.info(f"[DEBUG] Default keys: {default_keys}")

        logger.info(f"[DEBUG] Received {len(clients_data)} clients for aggregation")
        logger.info(f"[DEBUG] U (len w) = {U}")


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
    

    def _ema_merge(self,
                   aggregated_params: Dict[str, torch.Tensor],
                   alpha: float) -> Dict[str, torch.Tensor]:
        """
        theta <- (1 - alpha) * theta + alpha * aggregated_params
        aggregated_params: self.aggregate(...) 가 만든 Adapter_u.* 키만 담긴 부분 state_dict
        """

        cur = {k: v.to(self.device) for k, v in self.model.state_dict().items()}
        out: Dict[str, torch.Tensor] = {}
        for k, v_new in aggregated_params.items():
            v_new = param2tensor(v_new).to(self.device)
            v_old = cur.get(k, v_new)
            out[k] = (1.0 - alpha) * v_old + alpha * v_new
        return out

    def _ema_merge_with_averaged(self,
                                aggregated_params: Dict[str, torch.Tensor],
                                alpha: float,
                                agg_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        1) cur := 현재 서버 모델 파라미터
        2) 각 default 키에 대해 Adapter_0..Adapter_{U-1} 값을 simple average로 통일
        3) 그 cur을 기준으로 EMA:
            theta <- (1 - alpha) * theta + alpha * aggregated_params
        aggregated_params: self.aggregate(...) 가 만든 Adapter_u.* 키만 담긴 부분 state_dict
        agg_info: client_feedback 안에서 default_keys, U 계산용
        """

        clients_data = agg_info.get("client_feedback", [])


        # ① default_keys, U 추출
        first_para = clients_data[0]["model_para"]
        # 필요하면 ".default." 등으로 더 구체화해도 됨
        default_keys = [k for k in first_para.keys() if "default" in k]

        U = len(clients_data[0].get("w", []))

        # ② 현재 서버 파라미터 로드
        cur = {k: v.to(self.device) for k, v in self.model.state_dict().items()}

        # ③ 각 default 키에 대해 Adapter_0..Adapter_{U-1} simple average
        for k_default in default_keys:
            adapter_keys = []
            tensors = []
            for u in range(U):
                k_u = k_default.replace("default", f"Adapter_{u}")
                if k_u in cur:
                    adapter_keys.append(k_u)
                    tensors.append(cur[k_u])

            mean_tensor = sum(tensors) / float(len(tensors))   
            for k_u in adapter_keys:
                cur[k_u] = mean_tensor.clone()

        # ④ EMA 적용
        out: Dict[str, torch.Tensor] = {}
        for k, v_new in aggregated_params.items():
            v_new = param2tensor(v_new).to(self.device)
            v_old = cur.get(k, v_new)
            out[k] = (1.0 - alpha) * v_old + alpha * v_new
        return out


    # ② 데이터 비율 기반 모멘텀
    def aggregate_with_data_momentum(self,
                                     agg_info: Dict[str, Any],
                                     *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        alpha = sum(sample_size of selected) / total_train_size
        agg_info:
          - client_feedback: [{model_para, sample_size, w}, ...]
          - (optional) total_train_size
        """
        # 기존 집계(전문가별 가중평균) 그대로 활용
        aggregated_params = self.aggregate(agg_info, *args, **kwargs)

        clients = agg_info.get("client_feedback", [])
        sel_sum = float(sum(fb.get("sample_size", 0) for fb in clients))

        total = self.total_train_size
        

        alpha = sel_sum / float(total)



        # sample_list = [fb.get("sample_size", 0) for fb in clients]
        # logger.info(f"[DEBUG] sample_size list = {sample_list}")
        # logger.info(f"[DEBUG] sel_sum = {sel_sum}")
        # logger.info(f"[DEBUG] total_train_size = {total}")
        # logger.info(f"[DEBUG] alpha (sel_sum/total) = {alpha}")


        return self._ema_merge(aggregated_params, alpha)

    # ③ 클라이언트 수 비율 기반 모멘텀
    def aggregate_with_count_momentum(self,
                                      agg_info: Dict[str, Any],
                                      *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        alpha = |S_r| / M
        agg_info:
          - client_feedback: [...]
          - (optional) num_total_clients
        """
        # 기존 집계(전문가별 가중평균) 그대로 활용
        aggregated_params = self.aggregate(agg_info, *args, **kwargs)


        alpha = float(self.sample_client_num) / float(self.num_clients)



        return self._ema_merge(aggregated_params, alpha)
    


    # ② 데이터 비율 기반 모멘텀
    def aggregate_with_data_momentum_with_averaged(self,
                                     agg_info: Dict[str, Any],
                                     *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        alpha = sum(sample_size of selected) / total_train_size
        agg_info:
          - client_feedback: [{model_para, sample_size, w}, ...]
          - (optional) total_train_size
        """
        # 기존 집계(전문가별 가중평균) 그대로 활용
        aggregated_params = self.aggregate(agg_info, *args, **kwargs)

        clients = agg_info.get("client_feedback", [])
        sel_sum = float(sum(fb.get("sample_size", 0) for fb in clients))

        total = self.total_train_size
        

        alpha = sel_sum / float(total)

        # sample_list = [fb.get("sample_size", 0) for fb in clients]
        # logger.info(f"[DEBUG] sample_size list = {sample_list}")
        # logger.info(f"[DEBUG] sel_sum = {sel_sum}")
        # logger.info(f"[DEBUG] total_train_size = {total}")
        # logger.info(f"[DEBUG] alpha (sel_sum/total) = {alpha}")        



        return self._ema_merge_with_averaged(aggregated_params, alpha, agg_info)

    # ③ 클라이언트 수 비율 기반 모멘텀
    def aggregate_with_count_momentum_with_averaged(self,
                                      agg_info: Dict[str, Any],
                                      *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        alpha = |S_r| / M
        agg_info:
          - client_feedback: [...]
          - (optional) num_total_clients
        """
        # 기존 집계(전문가별 가중평균) 그대로 활용
        aggregated_params = self.aggregate(agg_info, *args, **kwargs)


        alpha = float(self.sample_client_num) / float(self.num_clients)

        return self._ema_merge_with_averaged(aggregated_params, alpha, agg_info)








