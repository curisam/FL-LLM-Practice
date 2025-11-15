import os
import json
import copy
import pickle

import logging
import time # time 모듈 추가

from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, LLMComparisonDataset


logger = logging.getLogger(__name__)


# --- 환경 변수 기반 헬퍼 함수 ---
def is_main_process_env():
    return os.environ.get("LOCAL_RANK", "0") == "0"
# --------------------------------

TLDR_PROMPT_DICT = {
    # "summary": ("Below is a forum post. Write a precise and concise summary "
    #             "that includes the most important points of the post.\n\n"
    #             "### SUBREDDIT: r/{subreddit}\n"
    #             "### TITLE: {title}\n"
    #             "### POST: {post}\n"
    #             "### TL;DR:"),
    "summary": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize the following Reddit post in "
        "a paragraph of 50 words or less.\n\n"
        "### Input:\n"
        "SUBREDDIT: r/{subreddit}\n"
        "TITLE: {title}\n"
        "POST: {post}\n\n"
        "### Response:"),
    "summary_cmp": (
        "Below is a forum post followed by two summaries. "
        "Pick a more precise and concise one that summarizes the most "
        "important points in the given forum post, without including "
        "unimportant or irrelevant details. State your choice with a "
        "single capital letter, i.e., \"A\" if SUMMARY A is better, "
        "\"B\" if SUMMARY B is better.\n\n"
        "### SUBREDDIT: r/{subreddit}\n"
        "### TITLE: {title}\n"
        "### POST: {post}\n"
        "### SUMMARY A:{output_A}\n"
        "### SUMMARY B:{output_B}\n"
        "### YOUR CHOICE:"),
    "mix_cmp": ("Below is an instruction that describes a task, "
                "paired with an input that provides further context. "
                "There are two responses that complete the request. "
                "Pick an appropriate response and state your choice with "
                "a single capital letter, i.e., "
                "\"A\" if RESPONSE A is better and more appropriate, "
                "\"B\" if RESPONSE B is better and more appropriate.\n\n"
                "### Instruction:\nSummarize the following Reddit post.\n\n"
                "### Input:\n"
                "SUBREDDIT: r/{subreddit}\n"
                "TITLE: {title}\n"
                "POST: {post}\n\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:")
}


def _download_tldr_cmpr(data_root):
    all_files = [f'batch{i}' for i in range(3, 21)] + ['batch22'] #batch 3-20, 22 를 가져올 것.
    # all_files = ['batch8']

    for cmp_file in all_files:#원본파일 다운로드
        download_url(
            'https://openaipublic.blob.core.windows.net/'
            'summarize-from-feedback/dataset/comparisons/'
            f'{cmp_file}.json', data_root)

    # Preprocess the above data. 파싱할 파일 경로 리스트 생성.
    file_paths = [
        os.path.join(data_root, f'{cmp_file}.json') for cmp_file in all_files
    ]


    #모든 .json 파일의 내용을 읽어와 하나의 거대한 딕셔너리 리스트로 만듭니다. 단순히 파일을 읽는 것뿐만 아니라, 복잡한 JSON 구조를 **단순화하고 필요한 정보만 추출(파싱)**하는 중요한 역할을 함.
    list_data_dict = load_jsonls(file_paths,
                                 subreddit='info.subreddit',
                                 title='info.title', #info.title을 title 키로 지정.
                                 post='info.post',
                                 output_A='summaries.0.text',
                                 output_B='summaries.1.text',
                                 category='worker', #worker를 category 키로 지정.
                                 split='split', 
                                 choice='choice') #모든 파일의 모든 샘플에 대해 위 변환을 적용한 딕셔너리들의 거대한 리스트, 즉 list_data_dict를 반환
    # Split the dataset into ['train', 'val', 'test'] 
    list_train_dict, list_val_dict, list_test_dict = [], [], []

    
    for data_dict in list_data_dict:
        if data_dict['split'] == 'train':
            list_train_dict.append(data_dict)
        elif data_dict['split'] == 'valid1':
            list_val_dict.append(data_dict)
        elif data_dict['split'] == 'valid2':
            list_test_dict.append(data_dict)

    # # merge the worker with less than 10 samples
    # dict_cat = {}
    # for idx, sample in enumerate(list_train_dict):
    #     if sample['category'] not in dict_cat:
    #         dict_cat[sample['category']] = [idx]
    #     else:
    #         dict_cat[sample['category']].append(idx)
    # for values in dict_cat.values():
    #     if len(values) < 50:
    #         for idx in values:
    #             list_train_dict[idx]['category'] = 'merged_client'

    return list_train_dict, list_val_dict, list_test_dict


def _download_tldr_human(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'reddit-tldr_train.jsonl'),
        os.path.join(data_root, 'reddit-tldr_val.jsonl'),
        os.path.join(data_root, 'reddit-tldr_test.jsonl')
    ]

    for name, fp in [('train', train_fp), ('valid', val_fp),
                     ('test', test_fp)]:
        if not os.path.exists(fp):
            download_url(
                'https://openaipublic.blob.core.windows.net/'
                'summarize-from-feedback/datasets/'
                f'tldr_3_filtered/{name}.jsonl', data_root)
            os.rename(os.path.join(data_root, f'{name}.jsonl'), fp)

    dataloader_kwargs = {
        'subreddit': 'subreddit',
        'title': 'title',
        'post': 'post',
        'summary': 'summary'
    }
    list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
    list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
    list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    return list_train_dict, list_val_dict, list_test_dict


def _tldr_human_for_prtraining(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'reddit-tldr_train_finetune.jsonl'),
        os.path.join(data_root, 'reddit-tldr_val_finetune.jsonl'),
        os.path.join(data_root, 'reddit-tldr_test_finetune.jsonl')
    ]

    dataloader_kwargs = {
        'subreddit': 'subreddit',
        'title': 'title',
        'post': 'post',
        'summary': 'summary'
    }
    if os.path.exists(train_fp) and os.path.exists(val_fp) and \
            os.path.exists(test_fp):
        list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
        list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
        list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    else:
        h_train, h_val, h_test = _download_tldr_human(data_root)
        c_train, c_val, c_test = _download_tldr_cmpr(data_root)

        # get a full list of comparison data
        c_posts = []
        for list_dict in [c_train, c_val, c_test]:
            for sample in list_dict:
                if sample['post'] not in c_posts:
                    c_posts.append(sample['post'])

        # remove the comparison data in human dataset
        list_train_dict = [s for s in h_train if s['post'] not in c_posts]
        list_val_dict = [s for s in h_val if s['post'] not in c_posts]
        list_test_dict = [s for s in h_test if s['post'] not in c_posts]

        # Add a space to the start of a summary, and save to file
        for fp, list_dict in [(train_fp, list_train_dict),
                              (val_fp, list_val_dict),
                              (test_fp, list_test_dict)]:
            with open(fp, "w") as file:
                for sample in list_dict:
                    sample["summary"] = " " + sample["summary"]
                    file.write(json.dumps(sample) + "\n")

    return list_train_dict, list_val_dict, list_test_dict


def get_tldr_dataset(list_data_dict,
                     tokenizer,
                     prompt=TLDR_PROMPT_DICT['summary']):
    return LLMDataset(list_data_dict,
                      tokenizer,
                      prompt_input=prompt,
                      prompt_no_input=prompt,
                      output_tag='summary')


def load_human_annotated_dataset(data_root, tokenizer):
    list_train_dict, list_val_dict, list_test_dict = \
        _download_tldr_human(data_root)

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=TLDR_PROMPT_DICT['summary'],
                               prompt_no_input=TLDR_PROMPT_DICT['summary'],
                               output_tag='summary')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=TLDR_PROMPT_DICT['summary'],
                             prompt_no_input=TLDR_PROMPT_DICT['summary'],
                             output_tag='summary')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=TLDR_PROMPT_DICT['summary'],
                              prompt_no_input=TLDR_PROMPT_DICT['summary'],
                              output_tag='summary')

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_human_finetuning_dataset(data_root,
                                  tokenizer,
                                  rlhf=False,
                                  max_num_test=-1,
                                  raw_no_prompt=False):
    list_train_dict, list_val_dict, list_test_dict = \
        _tldr_human_for_prtraining(data_root)

    # First 60% for fine-tuning, last 40% for rlhf
    idx = int(len(list_train_dict) * 0.6)
    list_train_dict = list_train_dict[:idx] if not rlhf else \
        list_train_dict[idx:]
    if raw_no_prompt:
        if max_num_test > 0:
            return (list_train_dict, list_val_dict[:max_num_test],
                    list_test_dict[:max_num_test])
        else:
            return list_train_dict, list_val_dict, list_test_dict

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=TLDR_PROMPT_DICT['summary'],
                               prompt_no_input=TLDR_PROMPT_DICT['summary'],
                               output_tag='summary')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=TLDR_PROMPT_DICT['summary'],
                             prompt_no_input=TLDR_PROMPT_DICT['summary'],
                             output_tag='summary')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=TLDR_PROMPT_DICT['summary'],
                              prompt_no_input=TLDR_PROMPT_DICT['summary'],
                              output_tag='summary')

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_comparison_dataset(data_root, tokenizer, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    train_set_path = os.path.join(data_root, f'{token_name}_train.pickle')
    val_set_path = os.path.join(data_root, f'{token_name}_val.pickle')
    test_set_path = os.path.join(data_root, f'{token_name}_test.pickle')
    if os.path.exists(train_set_path) and os.path.exists(val_set_path) \
            and os.path.exists(test_set_path):
        with open(train_set_path, 'rb') as f_train, \
                open(val_set_path, 'rb') as f_val, \
                open(test_set_path, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            _download_tldr_cmpr(data_root)

        # load dataset, which should be tuple
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        val_dataset = LLMComparisonDataset(
            list_val_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        test_dataset = LLMComparisonDataset(
            list_test_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')

        # Store these three lists to a pickle file
        with open(train_set_path, 'wb') as f_train, \
                open(val_set_path, 'wb') as f_val, \
                open(test_set_path, 'wb') as f_test:
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


def load_best_dataset(data_root, tokenizer, max_num_test=-1):
    train_dataset, val_dataset, test_dataset = \
        load_comparison_dataset(data_root, tokenizer, max_num_test)
    # Use the win_dataset only
    dataset = (train_dataset.win_dataset, val_dataset.win_dataset,
               test_dataset.win_dataset)
    return dataset


def load_comparison_dataset_by_choice(data_root, tokenizer, max_num_test=-1): #이거에 해당.
    token_name = os.path.basename(tokenizer.name_or_path)
    # 데이터 파일 경로들
    paths = tuple(os.path.join(data_root, f'{token_name}_{split}_choice.pickle') 
                  for split in ['train', 'val', 'test'])
    # 동기화를 위한 완료 파일(completion file) 경로
    completion_file_path = os.path.join(data_root, f'{token_name}_tldr.complete')
    if is_main_process_env():
        # 메인 프로세스는 캐시가 유효한지 확인하고, 유효하지 않으면 재생성
        if not os.path.exists(completion_file_path):
            logger.info("Main process: Completion file not found. Generating data...")
            
            # 이전 캐시 파일이 불완전할 수 있으므로 모두 삭제
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)


            # # ... (데이터 생성 로직은 기존과 동일: _download_tldr_cmpr, 레이블 변환, LLMDataset 생성)
            # list_train_dict, list_val_dict, list_test_dict = _download_tldr_cmpr(data_root)
            # # ... (레이블 변환) ...
            # # map the choice to "A" and "B" instead of 0 and 1. 

            # #레이블(Choice) 변환. 동작: LLM이 답변을 생성하기 쉽도록, 숫자 레이블 0, 1을 문자열 " A", " B"로 변환합니다.
            # #예시:
            # #### choice가 0이었던 샘플은 chr(0 + ord("A")) -> chr(65) -> "A"가 되고, 앞에 공백이 붙어 최종적으로 " A"가 됩니다.
            # #### choice가 1이었던 샘플은 chr(1 + ord("A")) -> chr(66) -> "B"가 되고, 최종적으로 " B"가 됩니다.
            # for list_dict in [list_train_dict, list_test_dict, list_val_dict]:
            #     for sample in list_dict:
            #         sample['choice'] = " " + chr(sample['choice'] + ord("A"))

            list_train_dict, list_val_dict, list_test_dict = _download_tldr_cmpr(data_root)

            # 각 분할(train/val/test)에서 등장하는 라벨러(annotator) 집합을 추출합니다.
            train_cats = {s['category'] for s in list_train_dict}
            val_cats   = {s['category'] for s in list_val_dict}
            test_cats  = {s['category'] for s in list_test_dict}
            # 세 집합의 교집합만 남깁니다.
            common_cats = sorted(train_cats & val_cats & test_cats)



            # 교집합에 속하지 않는 카테고리의 샘플을 모두 제거합니다.
            list_train_dict = [s for s in list_train_dict if s['category'] in common_cats]
            list_val_dict   = [s for s in list_val_dict   if s['category'] in common_cats]
            list_test_dict  = [s for s in list_test_dict  if s['category'] in common_cats]

            # 선호 그룹을 만듭니다. 교집합 크기가 홀수면 앞쪽 그룹이 하나 더 많습니다.
            num_common = len(common_cats)
            num_long_pref = num_common // 2 + (num_common % 2)
            long_prefer_clients  = set(common_cats[:num_long_pref])
            short_prefer_clients = set(common_cats[num_long_pref:])

            # 요약 길이 기준으로 choice 레이블을 재정의합니다.
            # 긴 요약 선호 그룹은 더 긴 요약을, 짧은 요약 선호 그룹은 더 짧은 요약을 선택합니다.
            for list_dict in (list_train_dict, list_val_dict, list_test_dict):
                for sample in list_dict:
                    len_a = len(sample['output_A'])
                    len_b = len(sample['output_B'])
                    prefer_long = sample['category'] in long_prefer_clients
                    if prefer_long:
                        choice_idx = 0 if len_a >= len_b else 1
                    else:
                        choice_idx = 0 if len_a <= len_b else 1
                    sample['choice'] = " " + chr(choice_idx + ord("A"))



            # ... (LLMDataset 객체 3개 생성) ...

            #전처리된 딕셔너리 리스트(list_train_dict 등)를 LLMDataset 클래스에 전달하여 최종 데이터셋 객체를 생성합니다.
            ####prompt_input (프롬프트 템플릿)을 가져옵니다.
            """Review the following post and the two summaries, then choose the better summary.\n\n### Post:\n{post}\n\n### Summary A:\n{output_A}\n\n### Summary B:\n{output_B}\n\n### Choice:"""
            #### 각 샘플 딕셔너리의 내용(post, output_A, output_B)을 이 프롬프트 템플릿에 채워 넣어 완전한 입력 텍스트를 만듭니다.
            #### tokenizer를 사용하여 이 입력 텍스트와 타겟 텍스트(" A" 또는 " B")를 토큰화(숫자 시퀀스로 변환)하여 input_ids, attention_mask, labels 등을 생성합니다.
            ####이 모든 정보를 담고 있는 데이터셋 객체를 반환합니다.

            import ipdb; ipdb.set_trace(context=15)
            train_dataset = LLMDataset(
                list_train_dict,
                tokenizer,
                prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
                prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
                output_tag='choice')
            val_dataset = LLMDataset(
                list_val_dict,
                tokenizer,
                prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
                prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
                output_tag='choice')
            test_dataset = LLMDataset(
                list_test_dict,
                tokenizer,
                prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
                prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
                output_tag='choice')

            # Store these three lists to a pickle file  최종 LLMDataset 객체들을 pickle을 사용해 파일로 저장
            # 데이터 파일 저장
            # train_set_path 대신 paths[0] 사용
            with open(paths[0], 'wb') as f: pickle.dump(train_dataset, f)
            # val_set_path 대신 paths[1] 사용
            with open(paths[1], 'wb') as f: pickle.dump(val_dataset, f)
            # test_set_path 대신 paths[2] 사용
            with open(paths[2], 'wb') as f: pickle.dump(test_dataset, f)

            logger.info("Main process: Caching completed in reddit_tldr.")
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

    with open(paths[0], 'rb') as f_train, \
         open(paths[1], 'rb') as f_val, \
         open(paths[2], 'rb') as f_test:
        train_dataset = pickle.load(f_train)
        val_dataset = pickle.load(f_val)
        test_dataset = pickle.load(f_test)

    # shrink val and test dataset
    if max_num_test > 0: #데이터 사이즈 줄여서 반환. LLMDataset에 접근하여 적용.
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def check_sim(data_root):
    cmpr_list_train_dict, cmpr_list_val_dict, cmpr_list_test_dict = \
        _download_tldr_cmpr(data_root)

    human_list_train_dict, human_list_val_dict, human_list_test_dict = \
        _download_tldr_human(data_root)

    # show if human-annotated overlaps cmpr in terms of train_dict
    cmpr_train = [sample['post'] for sample in cmpr_list_val_dict]
    human_train = [sample['post'] for sample in human_list_train_dict]

    print(len(cmpr_train))  # 92858
    print(len(human_train))  # 116722

    total_overlapping = 0

    for data in cmpr_train:
        if data in human_train:
            total_overlapping += 1
            human_train.pop(human_train.index(data))

    print(len(human_train))
    print(total_overlapping)  # 59685/282/475


if __name__ == "__main__":
    data_root = os.path.join('/local/scratch/d/wu1977/dataset/',
                             'reddit-tldr-comparison')
    check_sim(data_root)
