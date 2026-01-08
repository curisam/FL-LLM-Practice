"""
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

import copy
import logging
import pandas as pd

import torch

from enum import Enum
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


# TODO: support LDA when 'category' in keys
class LLMDataset(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 output_tag='output'):
        super(LLMDataset, self).__init__()

        # Print prompt info


        """
        (
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
            "### YOUR CHOICE:")


        """

        logger.info(f'prompt_input: {prompt_input}')
        logger.info(f'prompt_no_input: {prompt_no_input}')

        """
        example dict 의 키들을 {…}에 채워 넣어서

        source_text = prompt.format_map(example)
        이렇게 만들어진 문자열 목록이 self.sources 에 저장됩니다.

        "Below is a forum post …\n### SUBREDDIT: r/tldr\n### TITLE: Example title\n### POST: Here is the full post …\n### SUMMARY A:Summary A text\n### SUMMARY B:Summary B text\n### YOUR CHOICE:"
        
        """
        self.sources = []
        for example in list_data_dict:
            input = example.get("input", None)
            if input is not None and input != "":
                self.sources.append(prompt_input.format_map(example))
            else: #이거에 해당
                self.sources.append(prompt_no_input.format_map(example))
                



        """
        example['choice'] 가 " A" 또는 " B"인 상황.

        tokenizer.eos_token (예: "</s>") 를 붙여서

        " A</s>" / " B</s>" 형태의 리스트가 targets 에 저장됩니다.
        """



        targets = [
            f"{example[output_tag]}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        prompt_no_input.format_map(example)


        #sources + targets 전체를 토크나이즈 → input_ids 생성, sources 부분 길이는 -100(IGNORE_INDEX) 마스킹 → labels 생성

        """
        각 source + target 문자열을 토크나이즈해 input_ids 로 만들고,
        소스(prompt) 부분의 토큰 위치에는 labels = -100 (IGNORE_INDEX),

        타깃(choice+eos) 부분의 토큰 위치에는 실제 토큰 ID를 labels 에 채워 넣습니다.
        """

        data_dict = self.preprocess(self.sources, targets, tokenizer) #input_ids, labels를 key로 갖는 dict 반환.

        self.input_ids = data_dict["input_ids"] #input_ids: source + target 문자열을 토크나이즈
        """
        input_ids =[<BOS>, ▁Review, ▁the, ▁following, ▁post, ▁..., ▁###, ▁Choice, :, ▁A, <EOS>]

        """
        self.labels = data_dict["labels"] #labels: input_ids 중 source 부분만 -100 적용.
        """
        labels =[-100, -100, -100, -100, -100, -100, -100, -100, -100,  id(▁A),  id(<EOS>)]

        """
 
        self.tokenizer = tokenizer


        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]#데이터 라벨러에 대한 정보.
        
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)


    def _tokenize_fn(self, strings, tokenizer):
        import torch  # 보너스 안전장치: 여기서 확실히 임포트

        tokenized_list = []
        for text in strings:
            enc = tokenizer(
                text,
                return_tensors=None,     # 바로 텐서를 안 만들고, 파이썬 리스트 형태로 반환.
                padding=False,           # 한 문장씩 처리라면 불필요
                max_length=tokenizer.model_max_length, #모델 최대 길이로 잘라줌
                truncation=True, #길면 잘라라
            )

            """
            enc = {
                "input_ids": [[101, 123, 456, ...]],   # 혹은 [101, 123, 456, ...]
                "attention_mask": [[1, 1, 1, ...]], # 어떤 토큰은 “실제 내용”이고, 어떤 토큰은 “패딩”인지 모델에게 알려주는 마스크. batch 단위로 처리할때 생성.
            }

            """


            # enc["input_ids"]는 [[...]] 혹은 [...] 형태일 수 있음 → 1D로 정규화
            ids = enc["input_ids"]
            if isinstance(ids[0], list):   # [[...]] 형태면
                ids = ids[0]
            input_ids = torch.tensor(ids, dtype=torch.long)

            tokenized_list.append(
                type("Obj", (), {"input_ids": input_ids})  # 간단한 네임스페이스 객체.  tokenized_list.append(input_ids) 해도 되는데 편의상 이렇게 함
            )

        input_ids = labels = [tok.input_ids for tok in tokenized_list]

        # 길이 계산: pad 토큰과 다른 토큰 개수
        pad_id = tokenizer.pad_token_id
        def _length(t):
            if pad_id is None:
                return t.numel()
            return int((t != pad_id).sum().item())

        input_ids_lens = labels_lens = [_length(t) for t in input_ids]

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer):
        """
        Tokenize sources and targets while ensuring that the final label
        tokens (" A"/" B") are not truncated.  For each sample we first
        tokenize the target to determine its length, then truncate the
        source so that `len(source_ids) + len(target_ids) <= model_max_length`.
        If an example would exceed the maximum length even after truncation,
        the example is skipped.
        """
        max_len = tokenizer.model_max_length - 2  # Reserve space for the labels
        input_ids_list = []
        labels_list = []
        input_ids_lens = []  # List to store the length of input_ids
        labels_lens = []  # List to store the length of labels
        
        for src, tgt in zip(sources, targets):
            # create the example as source + target
            example = src + tgt
            # tokenize example (source + target)
            example_ids = tokenizer(
                example,
                return_tensors=None,
                padding=False,
                truncation=False,  # Do not truncate yet, just check length
                add_special_tokens=False,
            )["input_ids"]
            
            # Skip example if it exceeds max_len
            if len(example_ids) > max_len:
                logger.warning(f"Skipping example with length {len(example_ids)} > max_len {max_len}")
                continue
            
            # Tokenize target (e.g. " A</s>") without special tokens
            tgt_ids = tokenizer(
                tgt,
                return_tensors=None,
                padding=False,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )["input_ids"]
            
            max_src_len = max_len - len(tgt_ids)
            src_ids = tokenizer(
                src,
                return_tensors=None,
                padding=False,
                truncation=True,
                max_length=max_src_len,
                add_special_tokens=False,
            )["input_ids"]
            # combine ids and create labels (mask source part with IGNORE_INDEX)
            combined_ids = src_ids + tgt_ids
            combined_labels = [DefaultToken.IGNORE_INDEX.value] * len(src_ids) + tgt_ids
            
            # Calculate the lengths
            input_ids_lens.append(len(combined_ids))
            labels_lens.append(len(combined_labels))
            
            input_ids_list.append(torch.tensor(combined_ids, dtype=torch.long))
            labels_list.append(torch.tensor(combined_labels, dtype=torch.long))

        #input_ids: source + target 문자열을 토크나이즈, labels: input_ids 중 source 부분만 -100 적용.
        return {"input_ids": input_ids_list, "labels": labels_list, "input_ids_lens": input_ids_lens, "labels_lens": labels_lens}

    # def preprocess(self, sources, targets, tokenizer):
    #     examples = [s + t for s, t in zip(sources, targets)]
    #     examples_tokenized, sources_tokenized = [
    #         self._tokenize_fn(strings, tokenizer)
    #         for strings in (examples, sources)
    #     ]
    #     input_ids = examples_tokenized["input_ids"] 
    #     labels = copy.deepcopy(input_ids)
    #     for label, source_len in zip(labels,
    #                                  sources_tokenized["input_ids_lens"]):
    #         label[:source_len] = DefaultToken.IGNORE_INDEX.value
    #         # TODO: remove the data which is longer than the max input length
    #     return dict(input_ids=input_ids, labels=labels) #input_ids: source + target 문자열을 토크나이즈, labels: input_ids 중 source 부분만 -100 적용.

    def __len__(self): #데이터의 갯수 반환.
        return len(self.input_ids)

    def __getitem__(self, i): #input_ids, lables, categories를 key로 갖는 dict 반환
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])



class LLMComparisonDataset(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 output_A='output_A',
                 output_B='output_B',
                 choice='choice'):
        new_list_data_dict = []
        for example in list_data_dict:
            if choice in example and int(example[choice]) == 1:
                # output_B is better than output_A
                example[output_A], example[output_B] = \
                    example[output_B], example[output_A]
                new_list_data_dict.append(example)
        # remove the data without choice
        list_data_dict = new_list_data_dict

        # After switching, output_A > output_B
        self.win_dataset = LLMDataset(list_data_dict=list_data_dict,
                                      tokenizer=tokenizer,
                                      prompt_input=prompt_input,
                                      prompt_no_input=prompt_no_input,
                                      output_tag=output_A)
        self.lose_dataset = LLMDataset(list_data_dict=list_data_dict,
                                       tokenizer=tokenizer,
                                       prompt_input=prompt_input,
                                       prompt_no_input=prompt_no_input,
                                       output_tag=output_B)

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

        # super(LLMComparisonDataset, self).__init__(
        #     list_data_dict, tokenizer, prompt_input,
        #     prompt_no_input, output_A)

        # self.win_labels = self.labels

        # targets_B = [
        #     f"{example[output_B]}{tokenizer.eos_token}"
        #     for example in list_data_dict
        # ]
        # data_dict_B = self.preprocess(self.sources, targets_B, tokenizer)
        # self.lose_labels = data_dict_B["labels"]

    def __len__(self):
        return len(self.win_dataset)

    def __getitem__(self, i):
        return dict(win_data=self.win_dataset[i],
                    lose_data=self.lose_dataset[i],
                    categories=self.categories[i])