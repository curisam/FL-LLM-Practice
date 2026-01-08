import os
import gzip
import json
import pickle
import random
import logging
import torch
import datasets
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, PROMPT_DICT
from federatedscope.core.data.utils import download_url
from federatedscope.llm.model.model_builder import get_llm

logger = logging.getLogger(__name__)


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class LLMRewardCollator():
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        win_data, lose_data = \
            tuple([instance[key] for instance in instances]
                  for key in ("win_data", "lose_data"))

        # Form a concatenated dataset
        concat_data = win_data + lose_data
        concat_data_collator = LLMDataCollator(tokenizer=self.tokenizer)
        concat_data_dict = concat_data_collator(concat_data)

        return dict(
            win_input_ids=concat_data_dict["input_ids"][:len(win_data)],
            win_labels=concat_data_dict["labels"][:len(win_data)],
            win_attention_mask=concat_data_dict["attention_mask"]
            [:len(win_data)],
            lose_input_ids=concat_data_dict["input_ids"][len(win_data):],
            lose_labels=concat_data_dict["labels"][len(win_data):],
            lose_attention_mask=concat_data_dict["attention_mask"]
            [len(win_data):])


class Generator:
    """Generate the output from the original LLM model"""
    def __init__(self, config, tokenizer, generate_kwargs=None):
        self.device = f'cuda:{config.device}'

        # self.model = get_llm(config).to(self.device)
        self.add_special_tokens = True
        self.tokenizer = tokenizer

        if generate_kwargs is not None:
            self.generate_kwargs = generate_kwargs
        else:
            self.generate_kwargs = {
                'temperature': 0.0,
                'top_p': 1.0,
                'max_new_tokens': config.llm.chat.max_len,
            }
            self.generate_kwargs = {
                'max_new_tokens': config.llm.chat.max_len,
                'num_beams': 4,
                'no_repeat_ngram_size': 2,
                'early_stopping': True,
                'temperature': 0.0
            }

    def __call__(self, input_text, model):
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        response = model.generate(input_ids=input_ids, **self.generate_kwargs)
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        if response_tokens == "":
            print('INPUT:', input_text)
            print(len(input_text))
            print('===============================\n\n')
        return response_tokens


def get_tokenizer(model_name, cache_dir, tok_len=128, padding_side="right"):
    from transformers import AutoTokenizer, GPT2Tokenizer
    if model_name == 'CarperAI/openai_summarize_tldr_sft':
        tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2',
            cache_dir=cache_dir,
            model_max_length=tok_len,
            padding_side=padding_side,
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            model_max_length=tok_len,
            padding_side=padding_side,
            use_fast=False,
        )

    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens) #실제 vocab이 증가한 갯수.

    return tokenizer, num_new_tokens


class new_dict(dict):
    """
    Create a new_dict to ensure we can access the dictionary with
    one bracket only
    e.g., dict[key1][key2][key3] --> dict[key1.key2.key3]
    """
    def __init__(self, init_dict: dict):
        self.dict = init_dict
        for key in self.dict.keys():
            if type(self.dict[key]) is dict:
                self.dict[key] = new_dict(self.dict[key])
            if type(self.dict[key]) is list:
                self.dict[key] = new_dict({
                    str(idx): value
                    for idx, value in enumerate(self.dict[key])
                })

    def __getitem__(self, __key):
        try:
            if '.' not in __key:
                return self.dict[__key]
            else:
                prefix, suffix = __key.split('.', 1)
                return self.dict[prefix][suffix]
        except:
            return None

    def __setitem__(self, __key, __value):
        if type(__value) is dict:
            self.dict[__key] = new_dict(__value)
        else:
            if '.' not in __key:
                self.dict[__key] = __value
            else:
                prefix, suffix = __key.split('.', 1)
                if prefix not in self:
                    self.dict[prefix] = new_dict({})
                self.dict[prefix][suffix] = __value


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category',
              **kwargs):
    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        for key, value in kwargs.items():
            new_item[key] = item[value]
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def load_jsonl(file_path,
               is_gzip=False,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               **kwargs):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = new_dict(json.loads(line))
            new_item = dict(instruction=item[instruction],
                            input=item[input],
                            output=item[output],
                            category=item[category])
            for key, value in kwargs.items():
                new_item[key] = item[value]
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_jsonls(file_paths,
                is_gzip=False,
                instruction='instruction',
                input='input',
                output='output',
                category='category',
                **kwargs):
    list_data_dict = []
    for path in file_paths:
        list_data_dict.extend(
            load_jsonl(path, is_gzip, instruction, input, output, category,
                       **kwargs))
    return list_data_dict


def load_llm_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)

    dataset_name, _ = config.data.type.split('@')


    if dataset_name.lower() == 'reddit-tldr-comparison-choice': #TLDR
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_comparison_dataset_by_choice
        data_root = os.path.join(config.data.root, 'reddit-tldr-comparison')
        dataset = load_comparison_dataset_by_choice(data_root,
                                                    tokenizer,
                                                    max_num_test=-1)#(train_dataset, val_dataset, test_dataset)의 형태. 각각은 LLMDataset 클래스

 


    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    return dataset, config


if __name__ == '__main__':
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.cmd_args import parse_args, parse_client_cfg
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.logging import update_logger

    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    load_llm_dataset(init_cfg)
