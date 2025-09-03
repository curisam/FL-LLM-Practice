import os
import json
import copy
import pickle
import datasets

from federatedscope.core.splitters.generic.lda_splitter import LDASplitter
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset, \
    LLMDataset

# --- 환경 변수 기반 헬퍼 함수 ---
def is_main_process_env():
    return os.environ.get("LOCAL_RANK", "0") == "0"
# --------------------------------

SHP_PROMPT_DICT = {
    "shp": ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:"),
    "shp_cmp": ("Below is a query followed by two responses. Pick a "
                "helpful response that is precise, concise, and casual. "
                "State your choice with a single capital letter, "
                "i.e., \"A\" if RESPONSE A is better, "
                "\"B\" if RESPONSE B is better.\n\n"
                "### QUERY: {instruction}\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:"),
    "mix_cmp": ("Below is an instruction that describes a task. "
                "There are two responses that complete the request. "
                "Pick an appropriate response and state your choice with "
                "a single capital letter, i.e., "
                "\"A\" if RESPONSE A is better and more appropriate, "
                "\"B\" if RESPONSE B is better and more appropriate.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:")
}


def _download_shp_cmpr(data_root):
    """
    (전체 개요)
    입력: data_root (데이터를 저장할 경로)

    출력: list_train_dict, list_val_dict, list_test_dict (세 가지 split의 샘플 리스트)

    샘플 구조: 각 샘플은 딕셔너리로,

    {
        "instruction": ...,  # 프롬프트 (질문/지시문)
        "output_A": ...,     # 후보 응답 A
        "output_B": ...,     # 후보 응답 B
        "choice": ...,       # 사람이 선택한 정답 (0/1)
        "category": ...      # 원래 도메인 (reddit 등)
    }
        
    """


    #파일 경로 설정.
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'shp_cmpr_train.jsonl'),
        os.path.join(data_root, 'shp_cmpr_val.jsonl'),
        os.path.join(data_root, 'shp_cmpr_test.jsonl')
    ]

    dataloader_kwargs = {
        'instruction': 'instruction', # 프롬프트 (질문/지시문)
        'output_A': 'output_A', # 후보 응답 A
        'output_B': 'output_B', # 후보 응답 B
        'choice': 'choice', # 사람이 선택한 정답 (0/1)
        'category': 'category' # 원래 도메인 (reddit 등)
    }
    if os.path.exists(train_fp) and os.path.exists(val_fp) and \
            os.path.exists(test_fp):
        list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
        list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
        list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    else:
        dataset = datasets.load_dataset("stanfordnlp/SHP")
        list_train_dict, list_val_dict, list_test_dict = [], [], []
        tag_fp = {
            'train': (train_fp, list_train_dict),
            'validation': (val_fp, list_val_dict),
            'test': (test_fp, list_test_dict)
        }
        for tag, (fp, list_data_dict) in tag_fp.items():
            file = open(fp, 'w')
            for hist, ref_A, ref_B, choice, domain in \
                zip(dataset[tag]['history'],
                    dataset[tag]['human_ref_A'],
                    dataset[tag]['human_ref_B'],
                    dataset[tag]['labels'],
                    dataset[tag]['domain']):
                record = {
                    'instruction': hist,
                    'output_A': ref_A,
                    'output_B': ref_B,
                    'choice': choice,
                    'category': domain.split('_')[0]
                }
                file.write(f'{json.dumps(record)}\n')
                list_data_dict.append(record)
            file.close()

    return list_train_dict, list_val_dict, list_test_dict


def _download_shp(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'shp_rlhf_train.jsonl'),
        os.path.join(data_root, 'shp_rlhf_val.jsonl'),
        os.path.join(data_root, 'shp_rlhf_test.jsonl')
    ]

    dataloader_kwargs = {'instruction': 'instruction', 'category': 'category'}
    if os.path.exists(train_fp) and os.path.exists(val_fp) and \
            os.path.exists(test_fp):
        list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
        list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
        list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    else:
        dataset = datasets.load_dataset("stanfordnlp/SHP")
        instructions = []
        list_train_dict, list_val_dict, list_test_dict = [], [], []
        tag_fp = {
            'train': (train_fp, list_train_dict),
            'validation': (val_fp, list_val_dict),
            'test': (test_fp, list_test_dict)
        }
        for tag, (fp, list_data_dict) in tag_fp.items():
            file = open(fp, 'w')
            for hist, domain in zip(dataset[tag]['history'],
                                    dataset[tag]['domain']):
                if hist not in instructions:
                    instructions.append(hist)
                    record = {
                        'instruction': hist,
                        'category': domain.split('_')[0]
                    }
                    file.write(f'{json.dumps(record)}\n')
                    list_data_dict.append(record)
            file.close()

    return list_train_dict, list_val_dict, list_test_dict


def shp_dataset(data_root, num_clients, tokenizer):
    #데이터 불러오기


    list_train_dict, list_val_dict, list_test_dict = \
        _download_shp_cmpr(data_root)

    # First, disjoint by post instructions
    list_train_instructions, _, _ = _download_shp(data_root)


   #원래 카테고리 → 숫자 인덱스로 변환
    """
    각 instruction 샘플의 원래 category(예: Reddit 서브레딧 이름)를 정수 인덱스로 매핑.

    sample['categories']라는 새 필드에 그 인덱스를 기록.
    👉 디리클레 분할기(LDASplitter)가 카테고리 분포를 쓸 수 있게 준비.
    """
    cat_idx_map = {}
    for sample in list_train_instructions:
        if sample['category'] not in cat_idx_map:
            cat_idx_map[sample['category']] = len(cat_idx_map)
        sample['categories'] = cat_idx_map[sample['category']]


    # 디리클레 분할로 instruction을 클라이언트에 할당

    """
    LDASplitter: instruction 샘플들을 num_clients개의 클라이언트에 비IID(불균일)하게 분배. alpha=0.3은 한 클라이언트가 특정 주제에 더 치우치도록 만듦.

    결과물 inst_split_list는 [[클라이언트0 샘플들], [클라이언트1 샘플들], ...].

    이걸 풀어서 inst_client_map이라는 dict(“어떤 instruction → 어떤 client_id”)를 만듦.
    
    """
    splitter = LDASplitter(num_clients, alpha=0.3)
    inst_split_list = splitter(list_train_instructions)
    inst_client_map = {}
    for idx, sublist in enumerate(inst_split_list):
        for sample in sublist:
            inst_client_map[sample['instruction']] = idx


    #비교 데이터셋에 클라이언트 라벨 입히기
    """
    비교 데이터(train)의 각 샘플은 원래 category를 domain에 따로 저장.

    대신 category에는 Client_0, Client_1 같은 클라이언트 ID를 기록.
    👉 즉, 같은 instruction을 공유하는 비교 샘플들은 한 클라이언트 소유로 묶임.

    """

    # Update their categories and force the data splitter as meta
    for sample in list_train_dict:
        sample['domain'] = sample['category']
        sample['category'] = \
            f"Client_{inst_client_map[sample['instruction']]}"

    # 토큰 길이 제한 필터링
    """
    instruction + output_A + output_B 세 문장을 토큰화해서 길이 합산.

    512 토큰 이하인 샘플만 train 데이터에 남김.
    👉 학습 시 입력 길이 초과로 생기는 메모리 문제 방지.

    """
    new_list_train_dict = []
    for sample in list_train_dict:
        len_inst = len(tokenizer(sample['instruction'])['input_ids'])
        len_resA = len(tokenizer(sample['output_A'])['input_ids'])
        len_resB = len(tokenizer(sample['output_B'])['input_ids'])
        if len_inst + len_resA + len_resB <= 512:
            new_list_train_dict.append(sample)
    list_train_dict = new_list_train_dict

    # 클라이언트별 도메인 분포 출력

    """
    각 클라이언트(Client_0, Client_1, …)에 어떤 domain(원래 카테고리)이 얼마나 분포했는지 출력.

    결과를 보면 클라이언트별 데이터 분포가 비IID(불균등)하게 잘 나뉘었는지 확인 가능.

    주의: range(num_clients + 1)라 마지막에 빈 client 하나 더 찍힐 수도 있음.    
    
    """
    for client_id in range(num_clients + 1):
        print(f'Client {client_id}:')
        num_sample_by_domains = dict()
        for sample in new_list_train_dict:
            if sample['category'] == f'Client_{client_id}':
                if sample['domain'] not in num_sample_by_domains:
                    num_sample_by_domains[sample['domain']] = 0
                num_sample_by_domains[sample['domain']] += 1
        print(num_sample_by_domains)


    # 최종 train은 클라이언트 라벨+토큰 필터링 적용됨. val/test는 그대로 원본 분할을 유지.
    return list_train_dict, list_val_dict, list_test_dict


def load_rlhf_dataset(data_root,
                      tokenizer,
                      max_num_test=-1,
                      raw_no_prompt=False):
    _, list_val_dict, list_test_dict = \
        _download_shp(data_root)

    # reorganize the training data for RLHF
    list_train_dict = list_val_dict + list_test_dict
    list_val_dict = list_test_dict[:len(list_test_dict) // 2]
    list_test_dict = list_test_dict[len(list_test_dict) // 2:]

    if max_num_test > 0:
        return (list_train_dict, list_val_dict[:max_num_test],
                list_test_dict[:max_num_test])
    else:
        return list_train_dict, list_val_dict, list_test_dict


def load_safe_dataset():
    ds = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF-prompt")
    list_train_dict = [{
        'instruction': prompt
    } for prompt in ds['train']['prompt']]

    return list_train_dict, None, None


def load_comparison_dataset(data_root, tokenizer, config, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    num_clients = config.federate.client_num
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, f'{token_name}_train_{num_clients}.pickle'),
        os.path.join(data_root, f'{token_name}_val.pickle'),
        os.path.join(data_root, f'{token_name}_test.pickle')
    ]

    if os.path.exists(train_fp) and os.path.exists(val_fp) and os.path.exists(
            test_fp):
        with open(train_fp, 'rb') as f_train, open(val_fp, 'rb') as f_val, \
                open(test_fp, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            shp_dataset(data_root, num_clients, tokenizer)

        # load dataset, which should be tuple
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        val_dataset = LLMComparisonDataset(
            list_val_dict,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        test_dataset = LLMComparisonDataset(
            list_test_dict,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')

        # Store these three lists to a pickle file
        with open(train_fp, 'wb') as f_train, \
                open(val_fp, 'wb') as f_val, \
                open(test_fp, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.win_dataset.input_ids = \
            val_dataset.win_dataset.input_ids[:max_num_test]
        val_dataset.lose_dataset.input_ids = \
            val_dataset.lose_dataset.input_ids[:max_num_test]
        test_dataset.win_dataset.input_ids = \
            test_dataset.win_dataset.input_ids[:max_num_test]
        test_dataset.lose_dataset.input_ids = \
            test_dataset.lose_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_shp_best_dataset(data_root, tokenizer, config, max_num_test=-1):
    train_dataset, val_dataset, test_dataset = \
        load_comparison_dataset(data_root, tokenizer, config, max_num_test)
    # Use the win_dataset only
    dataset = (train_dataset.win_dataset, val_dataset.win_dataset,
               test_dataset.win_dataset)
    return dataset


def load_shp_cmp_dataset_by_choice(data_root,
                                   tokenizer,
                                   config,
                                   max_num_test=-1): #이거에 해당.
    token_name = os.path.basename(tokenizer.name_or_path)
    num_clients = config.federate.client_num

    train_fp, val_fp, test_fp = [
        os.path.join(data_root,
                     f'{token_name}_train_choice_{num_clients}.pickle'),
        os.path.join(data_root, f'{token_name}_val_choice.pickle'),
        os.path.join(data_root, f'{token_name}_test_choice.pickle')
    ]

    # 동기화를 위한 완료 파일(completion file) 경로
    completion_file_path = os.path.join(data_root, f'{token_name}_shp.complete')

    if is_main_process_env():

        # 메인 프로세스는 캐시가 유효한지 확인하고, 유효하지 않으면 재생성
        if not os.path.exists(completion_file_path):
            logger.info("Main process: Completion file not found. Generating data...")

        else:
            # ... (데이터 생성 로직은 기존과 동일: shp_dataset, 레이블 변환, LLMDataset 생성)
            list_train_dict, list_val_dict, list_test_dict = shp_dataset(data_root, num_clients, tokenizer)

            # ... (레이블 변환) ...
            # map the choice to "A" and "B" instead of 0 and 1. 

            #레이블(Choice) 변환. 동작: LLM이 답변을 생성하기 쉽도록, 숫자 레이블 0, 1을 문자열 " A", " B"로 변환합니다.
            #예시:
            #### choice가 0이었던 샘플은 chr(0 + ord("A")) -> chr(65) -> "A"가 되고, 앞에 공백이 붙어 최종적으로 " A"가 됩니다.
            #### choice가 1이었던 샘플은 chr(1 + ord("A")) -> chr(66) -> "B"가 되고, 최종적으로 " B"가 됩니다.  

            for list_dict in [list_train_dict, list_test_dict, list_val_dict]:
                for sample in list_dict:
                    sample['choice'] = " " + chr(sample['choice'] + ord("A"))

            # ... (LLMDataset 객체 3개 생성) ...

            #전처리된 딕셔너리 리스트(list_train_dict 등)를 LLMDataset 클래스에 전달하여 최종 데이터셋 객체를 생성합니다.
            ####prompt_input (프롬프트 템플릿)을 가져옵니다.
            """Below is a query followed by two responses. Pick a helpful response that is precise, concise, and casual. 
            State your choice with a single capital letter, i.e., \"A\" if RESPONSE A is better, \"B\" if RESPONSE B is better.\n\n ### QUERY: {instruction}\n 
            ### RESPONSE A: {output_A}\n ### RESPONSE B: {output_B}\n ### YOUR CHOICE: """
            #### 각 샘플 딕셔너리의 내용(instruction, output_A, output_B)을 이 프롬프트 템플릿에 채워 넣어 완전한 입력 텍스트를 만듭니다.
            #### tokenizer를 사용하여 이 입력 텍스트와 타겟 텍스트(" A" 또는 " B")를 토큰화(숫자 시퀀스로 변환)하여 input_ids, attention_mask, labels 등을 생성합니다.
            ####이 모든 정보를 담고 있는 데이터셋 객체를 반환합니다.
            train_dataset = LLMDataset(list_train_dict,
                                    tokenizer,
                                    prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                    prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                    output_tag='choice')
            val_dataset = LLMDataset(list_val_dict,
                                    tokenizer,
                                    prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                    prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                    output_tag='choice')
            test_dataset = LLMDataset(list_test_dict,
                                    tokenizer,
                                    prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                    prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                    output_tag='choice')

            # Store these three lists to a pickle file  최종 LLMDataset 객체들을 pickle을 사용해 파일로 저장
            # 데이터 파일 저장.
            with open(train_fp, 'wb') as f_train, \
                    open(val_fp, 'wb') as f_val, \
                    open(test_fp, 'wb') as f_test:
                pickle.dump(train_dataset, f_train)
                pickle.dump(val_dataset, f_val)
                pickle.dump(test_dataset, f_test)

            logger.info("Main process: Caching completed in shp.")
            # 모든 작업이 성공적으로 끝나면 완료 파일 생성
            with open(completion_file_path, 'w') as f:
                f.write('done')

    # 다른 프로세스들은 완료 파일이 생성될 때까지 대기
    else:
        local_rank = os.environ.get("LOCAL_RANK", "?")
        logger.info(f"Process {local_rank}: Waiting for completion file...")
        while not os.path.exists(completion_file_path):
            time.sleep(2)
        logger.info(f"Process {local_rank}: Completion file found.")



    # 이제 모든 프로세스는 메인 프로세스가 모든 작업을 완료했음을 확신하고
    # 안전하게 파일을 로드할 수 있습니다.
    with open(train_fp, 'rb') as f_train, open(val_fp, 'rb') as f_val, \
            open(test_fp, 'rb') as f_test:
        train_dataset = pickle.load(f_train)
        val_dataset = pickle.load(f_val)
        test_dataset = pickle.load(f_test)

    # shrink val and test dataset
    if max_num_test > 0: #데이터 사이즈 줄여서 반환. LLMDataset에 접근하여 적용.
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_alpacafarm_human_for_eval(data_root, tokenizer):
    token_name = os.path.basename(tokenizer.name_or_path)
    path = os.path.join(data_root,
                        f'{token_name}_alpacafarm_human_choice.pickle')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        ds = datasets.load_dataset("tatsu-lab/alpaca_farm",
                                   "alpaca_human_preference")["preference"]
        list_data_dict = []
        for row in ds.iter(batch_size=1):
            record = {
                "instruction": row["instruction"][0],
                "output_A": row["output_1"][0],
                "output_B": row["output_2"][0],
                "choice": {
                    1: 'A',
                    2: 'B'
                }[row["preference"][0]],
            }
            if row["input"][0]:
                record["instruction"] += f'\n\n{row["input"][0]}'
            list_data_dict.append(record)

        test_dataset = LLMDataset(list_data_dict,
                                  tokenizer,
                                  prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                  prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                  output_tag='choice')

        with open(path, 'wb') as f:
            pickle.dump(test_dataset, f)

    return test_dataset
