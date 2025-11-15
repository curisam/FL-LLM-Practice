"""
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

import copy
import logging
import pandas as pd

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
            else:
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

        # self.llm_generated_targets = [
        #     f"{example['llm_output']}"
        #     if example.get("llm_output", "") != "" else None
        #     for example in list_data_dict
        # ]


        #sources + targets 전체를 토크나이즈 → input_ids 생성, sources 부분 길이는 -100(IGNORE_INDEX) 마스킹 → labels 생성

        """
        각 source + target 문자열을 토크나이즈해 input_ids 로 만들고,
        소스(prompt) 부분의 토큰 위치에는 labels = -100 (IGNORE_INDEX),

        타깃(choice+eos) 부분의 토큰 위치에는 실제 토큰 ID를 labels 에 채워 넣습니다.
        """

        data_dict = self.preprocess(self.sources, targets, tokenizer) 

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

    def _tokenize_fn(self, strings, tokenizer):#strings=(examples, sources)
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer):
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self._tokenize_fn(strings, tokenizer)
            for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"] 
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels,
                                     sources_tokenized["input_ids_lens"]):
            label[:source_len] = DefaultToken.IGNORE_INDEX.value
            # TODO: remove the data which is longer than the max input length
        return dict(input_ids=input_ids, labels=labels) #input_ids: source + target 문자열을 토크나이즈, labels: input_ids 중 source 부분만 -100 적용.

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])

    # def overwrite_by_llm(self, i):
    #     source = self.sources[i]
    #     llm_answer = self.llm_generated_targets[i]

    #     if llm_answer is None or llm_answer == "":
    #         return

    #     llm_result_tknz = \
    #         self.preprocess([source], [llm_answer], self.tokenizer)
    #     self.input_ids[i], self.labels[i] = \
    #         llm_result_tknz['input_ids'][0], llm_result_tknz['labels'][0]


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