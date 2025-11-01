import numpy as np
from federatedscope.core.splitters import BaseSplitter


class IIDSplitter(BaseSplitter):
    """
    This splitter splits dataset following the independent and identically \
    distribution.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num):
        super(IIDSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None):
        from torch.utils.data import Dataset, Subset

        #전체 길이와 인덱스 생성
        length = len(dataset)
        index = [x for x in range(length)]
        #인덱스 섞기 (랜덤 셔플)
        np.random.shuffle(index)
        # 클라이언트 개수만큼 분할. 섞인 인덱스 배열을 client_num개로 균등하게 나눕니다. 나누어떨어지지 않을 때도 최대한 균등하게 분배합니다. 예: 103개, 5 클라이언트 → [21, 21, 21, 20, 20].
        idx_slice = np.array_split(np.array(index), self.client_num)
        
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
