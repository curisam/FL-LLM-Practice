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

 

        

        # print the size of each categories, 각 카테고리에 속한 샘플이 몇 개씩 있는지 로그로 찍어줌.
        tot_size = 0
        for i, cat in enumerate(categories):
            logger.info(f'Index: {i}\t'
                        f'Category: {cat}\t'
                        f'Size: {len(idx_slice[i])}')
            tot_size += len(idx_slice[i])
        logger.info(f'Total size: {tot_size}')



        # ---- 여기부터 새로 추가 ----
        assert len(categories) == self.client_num, (
            f'[#MetaSplitter] Number of categories ({len(categories)}) '
            f'must equal number of clients ({self.client_num}).'
        )
        # ---- 여기까지 ----

        # 카테고리 수 == 클라이언트 수인 경우에만 여기 도달
        final_idx_slice = idx_slice  # 전체 사용


        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in final_idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in final_idx_slice]


        from collections import Counter
        import torch
        def count_token_frequency_in_subset(subset, id_A, id_B):
            """
            Subset 객체에서 레이블의 `id(▁A)`와 `id(▁B)` 빈도를 계산하는 함수.
            """
            # Subset에서 레이블 추출
            all_labels = [sample['labels'] for sample in subset]  # 각 샘플의 레이블들
            flat_labels = [label.item() for sublist in all_labels for label in sublist]  # 2D 리스트를 1D로 평탄화하고 item()으로 Tensor에서 값 추출
            
            # Counter로 id_A, id_B의 빈도 수 세기
            label_counter = Counter(flat_labels)
            freq_A = label_counter.get(id_A, 0)  # id_A의 빈도 수
            freq_B = label_counter.get(id_B, 0)  # id_B의 빈도 수
            
            return freq_A, freq_B
        
        for idx, subset in enumerate(data_list):  # data_list는 각 Subset 객체들이 들어있는 리스트
            freq_A, freq_B = count_token_frequency_in_subset(subset, 362, 425) 
            print(f"Subset {idx}: id(▁A): {freq_A}, id(▁B): {freq_B}")
        
        # 최종 data_list의 길이는 self.client_num과 같아야 함
        assert len(data_list) == self.client_num
        logger.info(
            f'Data successfully split into {len(data_list)} clients, '
            f'each with a unique category.'
        )


        
        return data_list
