import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor


class MultiLoRAAvgAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info): #warmup 라운드 때 적용
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"] #content= (sample_size, model)들을 담은 리스트
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None #None
        scaler = agg_info['scaler'] if ('scaler' in agg_info) else 1.0   #1.0
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun) #ClientsAvgAggregator와 사실상 동일하게 작동.

        return avg_model

    def aggregate_on_model(self, agg_info): #warmup 라운드 아닐 때 적용
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None #None
        scaler = agg_info['scaler'] if ('scaler' in agg_info) else 1.0 #1.0
        avg_model = self._para_weighted_avg_on_model(models,
                                                     recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters): #self.model을 업데이트
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1): #당시 (fl라운드, 모든 adapter 모델)을 path에 저장.
        assert self.model is not None


        # 1️⃣ 상위 디렉터리 자동 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 2️⃣ 파일명에 라운드 번호 붙이기
        base, ext = os.path.splitext(path)                 # ⇒ "checkpoints/tldr_choice_qwen_fedbis", ".ckpt"
        if cur_round >= 0: #이대로 적용.
            path = f"{base}_round_{cur_round}{ext}"         # ⇒ checkpoints/tldr_choice_qwen_fedbis_round50.ckpt


        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _para_weighted_avg(self, models, recover_fun=None, scaler=1.0): #ClientsAvgAggregator와 사실상 동일하게 작동._para_weighted_avg_on_model 와  유사하게 짜려고 코드 구조만 변경한 것.
        """
        Calculates the weighted average of models.
        """
        keywise_training_set_size = dict()
        for i in range(len(models)):
            sample_size, model = models[i]
            for key in model.keys():
                if key not in keywise_training_set_size:
                    keywise_training_set_size[key] = [sample_size, 1]
                else:
                    keywise_training_set_size[key][0] += sample_size #aggregate할 개체 내의 train data size
                    keywise_training_set_size[key][1] += 1 #aggregate할 개체 갯수

        avg_model = dict()
        for i in range(len(models)):
            sample_size, model = models[i]

            for key, param in model.items():
                if self.cfg.federate.ignore_weight: #False
                    weight = 1.0 / keywise_training_set_size[key][1]
                else: #여기에 걸림.
                    weight = sample_size / keywise_training_set_size[key][0]

                weight = weight * scaler #scaler=1.0
                param = param2tensor(param)

                if key not in avg_model:
                    avg_model[key] = param * weight
                else:
                    avg_model[key] += param * weight

        return avg_model #모델의 state_dict 형태

    def _para_weighted_avg_on_model(self,
                                    models,
                                    recover_fun=None,
                                    scaler=1.0): 
        keywise_training_size = dict()
        for i in range(len(models)):
            train_size, model = models[i]
            for key in model.keys():
                if key not in keywise_training_size:
                    keywise_training_size[key] = [train_size, 1]
                else:
                    keywise_training_size[key][0] += train_size #aggregate할 개체 내의 train data size
                    keywise_training_size[key][1] += 1 #aggregate할 개체 갯수

        avg_model, raw_model_scaler = dict(), dict()

        for i in range(len(models)):
            train_size, model = models[i]

            for key, param in model.items():
                if self.cfg.federate.ignore_weight: #False
                    if hasattr(self, 'num_clients'): #이거에 해당.
                        weight = 1.0 / self.num_clients #전체 클라이언트 수로 나눔.
                    else:
                        weight = 1.0 / keywise_training_size[key][1]
                else: #이거에 해당.
                    if hasattr(self, 'total_train_size'): #이거에 해당.
                        weight = train_size / self.total_train_size ##모든 client들의 train data 갯수 총합으로 나눔.
                    else:
                        weight = train_size / keywise_training_size[key][0]

                weight = weight * scaler #scaler=1.0
                param = param2tensor(param)

                #adapter가 여러개 인 상황 고려해야 함. 동일 adapter인 것에만 aggregate 반영하는 것 유의. key의 이름에 adapter_idx가 있으므로. 
                if key not in avg_model:
                    avg_model[key] = param * weight
                    raw_model_scaler[key] = 1 - weight
                else:
                    avg_model[key] += param * weight #계속 더해나가서 \sigma p_m param_m을 만듬.
                    raw_model_scaler[key] -= weight #계속 빼 나가서 1-\sigma p_m을 만듬.

        # merge with the original model
        # model_state_dict = self.model.state_dict() #모든 adapter에 대한 정보 있음.
        # for key in avg_model.keys(): #momentum 스럽게 더해나감.
        #     avg_model[key] += raw_model_scaler[key] * model_state_dict[key]
        model_state_dict = {k: v.to(self.device) for k, v in self.model.state_dict().items()}
        for key in avg_model.keys():
            avg_model[key] = avg_model[key].to(self.device)
            avg_model[key] += raw_model_scaler[key] * model_state_dict[key]
        return avg_model
