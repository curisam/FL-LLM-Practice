import random
import numpy as np
import logging

from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.generic import IIDSplitter

logger = logging.getLogger(__name__)


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num, **kwargs):
        super(MetaSplitter, self).__init__(client_num)
        # Create an IID spliter in case that num_client < categories
        self.iid_spliter = IIDSplitter(client_num)

    def __call__(self, dataset, prior=None, **kwargs):

        
        from torch.utils.data import Dataset, Subset

        # 1) 데이터 형태를 리스트로 변환
        tmp_dataset = [ds for ds in dataset]
        # 2) 각 샘플의 “레이블” 혹은 “카테고리” 벡터 추출
        if isinstance(tmp_dataset[0], tuple):
            # (feature, label) 튜플일 때
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict): #이거에 해당. tmp_dataset[0]의 key는 dict_keys(['input_ids', 'labels', 'categories']).
            # 사전형 샘플일 때, 미리 categories 필드에 저장되어 있다고 가정
            label = np.array([x['categories'] for x in tmp_dataset])#annotator 정보. client id가 될 예정.
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        


 

        # Split by categories
        # categories = set(label) 
        categories = sorted(list(set(label))) # 전체 카테고리 집합

  
        idx_slice = [] # 카테고리별 샘플 인덱스 리스트
        for cat in categories:
            # label == cat 인 모든 위치(인덱스)를 모아서 하나의 리스트로
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())
        # idx_slice[i] 는 i번째 카테고리에 속하는 샘플들의 인덱스 목록

        ######################################################################################################
        # random.shuffle(idx_slice) # 순서를 섞어서 클라이언트 할당 순서를 무작위화.  이거 문제 될 수 있을 거 같음.

        

        # print the size of each categories, 각 카테고리에 속한 샘플이 몇 개씩 있는지 로그로 찍어줌.
        tot_size = 0
        for i, cat in enumerate(categories):
            logger.info(f'Index: {i}\t'
                        f'Category: {cat}\t'
                        f'Size: {len(idx_slice[i])}')
            tot_size += len(idx_slice[i])
        logger.info(f'Total size: {tot_size}')



        if len(categories) < self.client_num: # 카테고리 수가 클라이언트 수보다 적으면 IID로 전환
            logger.warning(
                f'The number of clients is {self.client_num}, which is '
                f'smaller than a total of {len(categories)} catagories, '
                'use iid splitter instead.')
            return self.iid_spliter(dataset)
        
        # 카테고리 수가 클라이언트 수보다 많거나 같을 때
        # self.client_num 개수만큼의 카테고리만 사용하고 나머지는 버림.
        if len(categories) > self.client_num:
            logger.warning(
                f'The number of categories ({len(categories)}) is greater than '
                f'the number of clients ({self.client_num}). '
                f'Only the first {self.client_num} categories will be used. '
                f'{len(categories) - self.client_num} categories will be discarded.'
            )


        # idx_slice를 앞에서부터 client_num 개수만큼만 슬라이싱
        final_idx_slice = idx_slice[:self.client_num]


        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in final_idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in final_idx_slice]
        
        # 최종 data_list의 길이는 self.client_num과 같아야 함
        assert len(data_list) == self.client_num
        logger.info(
            f'Data successfully split into {len(data_list)} clients, '
            f'each with a unique category.'
        )


        # if isinstance(dataset, Dataset):
        #     data_list = [Subset(dataset, idxs) for idxs in idx_slice]

        
        
        # else:
        #     data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]

        
        return data_list
