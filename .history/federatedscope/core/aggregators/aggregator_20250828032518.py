import os
import torch
from abc import ABC, abstractmethod


class Aggregator(ABC):
    """
    Abstract class of Aggregator.
    """
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            agg_info: information to be aggregated.
        """
        pass


class NoCommunicationAggregator(Aggregator):
    """Clients do not communicate. Each client work locally
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def update(self, model_parameters): #self.model을 업데이트
        '''
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1): #당시 (fl라운드, self.model)을 path에 저장.
        assert self.model is not None

        # 1️⃣ 상위 디렉터리 자동 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 2️⃣ 파일명에 라운드 번호 붙이기
        base, ext = os.path.splitext(path)                 # ⇒ "checkpoints/tldr_choice_qwen_fedbis", ".ckpt"
        if cur_round >= 0: #이대로 적용.
            path = f"{base}_round_{cur_round}{ext}"         # ⇒ checkpoints/tldr_choice_qwen_fedbis_round50.ckpt


        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()} #당시 (fl라운드, 모든 adapter 모델)을 path에 저장.
        torch.save(ckpt, path)

    def load_model(self, path): #당시 FL라운드 값을 return하고 aggregator의 self.model을 업데이트
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'], strict=False)
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            agg_info: information to be aggregated.
        """
        # do nothing
        return {}
