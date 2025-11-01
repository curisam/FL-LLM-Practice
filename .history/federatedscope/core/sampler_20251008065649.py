from abc import ABC, abstractmethod

import numpy as np
from bisect import bisect_right

import logging
logger = logging.getLogger(__name__)

class Sampler(ABC):
    """
    The strategies of sampling clients for each training round

    Arguments:
        client_state: a dict to manager the state of clients (idle or busy)
    """
    def __init__(self, client_num):

        # client_state[i] = 1: i번 클라이언트는 ‘idle(대기)’ 상태,
        # 0: ‘working(이미 선택되어 훈련 중)’ 또는 ‘unseen(제외된)’ 상태
        self.client_state = np.asarray([1] * (client_num + 1))
        # Set the state of server (index=0) to 'working'
        self.client_state[0] = 0

    @abstractmethod
    def sample(self, size):
        raise NotImplementedError

    def change_state(self, indices, state):#전체 중 indices에 대해 state 반영
        """
        To modify the state of clients (idle or working)
        """

        """
        선택/해제된 클라이언트의 상태를 바꿔주는 유틸
         - idle/seen → client_state=1, i번 클라이언트가 다음 샘플링에 “뽑을 수 있는” 상태(idle)
         - working/unseen → client_state=0, 0이면 이미 한 라운드에서 선택됐거나(working), 평가나 제외(unseen) 등으로 뽑지 말아야 할 상태입니다.
        """

        if isinstance(indices, list) or isinstance(indices, np.ndarray):
            all_idx = indices
        else:
            all_idx = [indices]
        for idx in all_idx:
            if state in ['idle', 'seen']:
                self.client_state[idx] = 1
            elif state in ['working', 'unseen']:
                self.client_state[idx] = 0
            else:
                raise ValueError(
                    f"The state of client should be one of "
                    f"['idle', 'working', 'unseen], but got {state}")


class UniformSampler(Sampler):
    """
    To uniformly sample the clients from all the idle clients
    """
    def __init__(self, client_num):
        super(UniformSampler, self).__init__(client_num)

    def sample(self, size):
        """
        To sample clients
        """
        idle_clients = np.nonzero(self.client_state)[0] ## np.nonzero(self.client_state)는 (array([1, 2, 4, 5]),) 이런 느낌으로 나옴

        sampled_clients = np.random.choice(idle_clients,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working') 
        return sampled_clients


class GroupSampler(Sampler):
    """
    To grouply sample the clients based on their responsiveness (or other
    client information of the clients)
    """
    def __init__(self, client_num, client_info, bins=10):
        super(GroupSampler, self).__init__(client_num)
        self.bins = bins
        self.update_client_info(client_info)
        self.candidate_iterator = self.partition()

    def update_client_info(self, client_info):
        """
        To update the client information
        """
        self.client_info = np.asarray(
            [1.0] + [x for x in client_info
                     ])  # client_info[0] is preversed for the server
        assert len(self.client_info) == len(
            self.client_state
        ), "The first dimension of client_info is mismatched with client_num"

    def partition(self):
        """
        To partition the clients into groups according to the client
        information

        Arguments:
        :returns: a iteration of candidates
        """
        sorted_index = np.argsort(self.client_info)
        num_per_bins = np.int(len(sorted_index) / self.bins)

        # grouped clients
        self.grouped_clients = np.split(
            sorted_index, np.cumsum([num_per_bins] * (self.bins - 1)))

        return self.permutation()

    def permutation(self):
        candidates = list()
        permutation = np.random.permutation(np.arange(self.bins))
        for i in permutation:
            np.random.shuffle(self.grouped_clients[i])
            candidates.extend(self.grouped_clients[i])

        return iter(candidates)

    def sample(self, size, shuffle=False):
        """
        To sample clients
        """
        if shuffle:
            self.candidate_iterator = self.permutation()

        sampled_clients = list()
        for i in range(size):
            # To find an idle client
            while True:
                try:
                    item = next(self.candidate_iterator)
                except StopIteration:
                    self.candidate_iterator = self.permutation()
                    item = next(self.candidate_iterator)

                if self.client_state[item] == 1:
                    break

            sampled_clients.append(item)
            self.change_state(item, 'working')

        return sampled_clients


class ResponsivenessRealtedSampler(Sampler):
    """
    To sample the clients based on their responsiveness (or other information
    of clients)
    """
    def __init__(self, client_num, client_info):
        super(ResponsivenessRealtedSampler, self).__init__(client_num)
        self.update_client_info(client_info)

    def update_client_info(self, client_info):
        """
        To update the client information
        """
        self.client_info = np.asarray(
            [1.0] + [np.sqrt(x) for x in client_info
                     ])  # client_info[0] is preversed for the server
        assert len(self.client_info) == len(
            self.client_state
        ), "The first dimension of client_info is mismatched with client_num"

    def sample(self, size):
        """
        To sample clients
        """
        idle_clients = np.nonzero(self.client_state)[0]
        client_info = self.client_info[idle_clients]
        client_info = client_info / np.sum(client_info, keepdims=True)
        sampled_clients = np.random.choice(idle_clients,
                                           p=client_info,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working')
        return sampled_clients



# federatedscope/core/sampler.py (파일 끝)
import numpy as np
from bisect import bisect_right

class ClusterUniformSampler(Sampler):
    """
    boundaries(누적 물리라운드)로 현재 라운드의 어댑터 인덱스를 구하고,
    해당 어댑터의 (1-based) 클러스터 내부에서만 균등추출.
    클러스터<=5면 sample_num_per_adapter에 전원참여 수가 설정되어 있음.
    """
    def __init__(self, client_num, clusters_1b, boundaries, sample_num_per_adapter):
        super().__init__(client_num)          # Sampler 내부 상태 벡터 크기 등 초기화(기존과 동일)
        self.clusters = [set(g) for g in clusters_1b]   # 1-based ID 집합
        self.boundaries = list(boundaries)
        self.s_per = [int(x) for x in sample_num_per_adapter]
        assert len(self.clusters) == len(self.s_per) == len(self.boundaries)
        self._allowed = set()
        self._cur_s = None

    def _adapter_idx(self, round_idx: int) -> int:
        return bisect_right(self.boundaries, int(round_idx))

    def set_allowed_for_round(self, round_idx: int):
        aidx = self._adapter_idx(round_idx)
        self._allowed = self.clusters[aidx]
        self._cur_s = self.s_per[aidx]

    def sample(self, size: int):
        # cluster에서는 'size'(=5)가 넘어와도 s[i]를 우선 사용
        size = int(self._cur_s) if self._cur_s is not None else int(size)
        idle = np.nonzero(self.client_state)[0].tolist()   # Sampler 표준: 1-based 인덱싱 지원
        candidates = [i for i in idle if i in self._allowed]
        if len(candidates) < size:
            raise ValueError(f"[ClusterSampler] 후보 {len(candidates)} < 요청 {size}")
        picked = np.random.choice(candidates, size=size, replace=False).tolist()
        self.change_state(picked, 'working')
        return picked
