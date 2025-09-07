import os
import gc
import sys
import math
import copy
import logging
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AdamW

from federatedscope.register import register_trainer
# from federatedscope.llm.trainer.trainer_arxivarxiv import LLMTrainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.monitors.monitor import Monitor

from federatedscope.llm.dataset.llm_dataset import DefaultToken
from federatedscope.llm.model.adapter_builder import AdapterModel

from torch.optim.lr_scheduler import MultiStepLR


logger = logging.getLogger(__name__)
sys.setrecursionlimit(100000)

    
def cal_loss(logits, labels, choices):

    """
    input_ids, labels가 아래와 같이 주어졌다 가정.
    [<BOS>, ▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>]   (T = 7)

    logits

    [▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>, -100]
    
    """


    # 1) next-token 셋업: (t-1, t) 정렬
    choices = choices.to(logits.device) 
    shift_logits = logits[..., :-1, :].contiguous() # [B, T-1, V]. EOS 에 대한 예측하는 POSITION만 제외. V중 Max [▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>]
    shift_labels = labels[..., 1:].contiguous() # [B, T-1]. BOS 에 대한 예측하는 POSITION만 제외. [▁Question,  :,  ▁2+2,  ?,  ▁A,  <EOS>]


    # 2) 라벨 맵핑: A/B 토큰이 등장한 자리만 클래스 인덱스(0/1)로 바꾸고 나머지는 IGN(-100
    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value) #[-100, -100, -100, -100, -100, -100]으로 초기화
    for idx, choice in enumerate(choices): # 예: [ID('▁A'), ID('▁B')]
        mask = (shift_labels == choice); new_labels[mask] = idx
    #mask->[False, False, False, False,  True, False]
    #new_labels = [-100, -100, -100, -100, 0, -100]


    # 예: [ID('▁A'), ID('▁B')] 관한 걸로만 logit space 축소.
    new_logits = shift_logits[..., choices]
    loss_fn = torch.nn.CrossEntropyLoss() #torch.nn.CrossEntropyLoss()의 기본 ignore_index가 -100이라서, 타깃에 -100이 들어 있으면 그 위치들은 손실/그라디언트 계산에서 제외돼요. 평균도 유효한 항만으로 나눕니다.

    # 4) (B×(T-1), K) vs (B×(T-1))로 펼쳐 CrossEntropy
    loss = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1)) #실질적으로 t=4인 지점에 대해서만 ce 계산됨.
    return new_logits, new_labels, loss

class RewardChoiceTrainer(LLMTrainer): #LLM이 뱉는 logits[B, T, V](V=전체 vocab)에 대해 “다음 토큰이 A냐 B냐만” 보도록 축소해서 **CrossEntropyLoss(B×(T-1) vs 2)**로 학습/평가한다. 
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        #실제 라벨은 " A</s>"= "▁A"+"</s>" or " B</s>"="▁B"+"</s>"인 상황. 최종 목표: "▁A", "▁B" 이렇게 2차원으로 Classifier를 축소.
        try:
            # (중요) "choices" 문자열 리스트를 토크나이즈하여
            # 마지막 토큰 ID만 추출 → 해당 토큰 등장 시 그 위치를 선택지 클래스 라벨로 사용
            choices_list = [] #"▁A", "▁B"의 token id 추출이 목표.
            for choice in config.trainer.choices: #config.trainer.choices=["A", "B"]
                # 앞에 ': '를 붙이는 이유: 프롬프트 형식에서 응답 표기 직후 토큰을 안정적으로 뽑기 위함. add_special_tokens=False:BOS/EOS 같은 스페셜 토큰이 끼어들면 안 되므로.
                token_ids = self.tokenizer(f': {choice}', add_special_tokens=False)['input_ids']
                #": A"는 보통 [:, ▁A]처럼 2개 이상 토큰으로 쪼개집니다. 이 중에 **실제로 분류를 갈라주는 건 마지막 토큰(= ▁A)**이죠. 그래서 [-1]만 뽑습니다.
                if not token_ids: raise ValueError(f"Tokenizer returned empty list for choice: '{choice}'")
                choices_list.append(token_ids[-1])
            # CPU 텐서로 보관, 라운드 시작 시 디바이스로 옮김
            self.choices_cpu = torch.tensor(choices_list, dtype=torch.long)
            logger.info(f'Choice token IDs: {self.choices_cpu.tolist()}')
        except Exception as e:
            logger.error(f"Error during trainer initialization: {e}")
            raise ValueError('Failed to initialize trainer.choices.')

    def _hook_on_fit_start_init(self, ctx):  #accelerate 및 deepspeed 지원 추가
        super().__init__(ctx)

        self.choices = self.choices_cpu.to(ctx.device, non_blocking=True) #non_blocking=True는 pinned memory 환경에서 async copy 허용 → 성능 최적화.      


    def _hook_on_batch_forward(self, ctx): #참고

        #LLM 학습용 Dataset은 보통 tokenized_dict를 반환->{'input_ids': Tensor, 'labels': Tensor, 'attention_mask': Tensor}
        #accelerator는 입력 텐서를 자동으로 디바이스에 올려줘서 .to(ctx.device)가 필요 없음.


        #input_ids: [101,  42,  53,  78,   2]
        #labels:    [101,  42,  53,  78,   2]  # 혹은 일부 -100 포함
        #이렇게 거의 같지만, loss 계산에서 무시할 토큰은 labels에서만 바뀜.

        # ---- 임베딩 경계 초과 가드 ----
        try:
            base_model = self._unwrap(getattr(self.ctx, "model", None))
            if base_model is not None and "input_ids" in self.ctx.data_batch:
                emb_rows = base_model.get_input_embeddings().weight.shape[0]
                # 빠른 장치 무관 max (GPU 텐서면 .amax)
                mx = int(self.ctx.data_batch["input_ids"].amax().item())
                if mx >= emb_rows:
                    rk = getattr(self.accelerator, "process_index", "?") if getattr(self, "accelerator", None) else "?"
                    logger.error(f"[TOK_OVF] rank={rk} max_id={mx} >= emb_rows={emb_rows} | split={self.ctx.cur_split}")
                    raise RuntimeError(f"Token id overflow: {mx} >= {emb_rows}")
        except Exception:
            pass

        # 1) 입력 준비
        input_ids = ctx.data_batch["input_ids"].to(ctx.device)
        labels = ctx.data_batch["labels"].to(ctx.device)
        attention_mask = ctx.data_batch["attention_mask"].to(ctx.device)

        # 2) 모델 실행 
        if ctx.cfg.llm.accelerator.use: #참고, #ctx.model 기반으로 outputs 뽑아낸다.
            outputs = ctx.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        elif ctx.cfg.llm.deepspeed.use: #ctx.model_engine 기반으로 outputs 뽑아낸다.
            outputs = ctx.model_engine(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        else: #참고, #ctx.model 기반으로 outputs 뽑아낸다.
            outputs = ctx.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )


        # 3) 커스텀 loss 계산 
        logits = outputs.logits #일반적으로 LLM에서는 [batch_size, seq_len, vocab_size] shape의 tensor



        ################         이 부분이 달라짐       #######################
        #new_labels = [-100, -100, -100, -100, 0, -100]
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)

        # NaN/Inf 모두 방어
        if not torch.isfinite(loss): #LM 학습에서는 종종 NaN loss가 발생할 수 있음-> label이 모두 -100 (ignore index). precision 문제 (e.g., bf16/float16). exploding gradients, bad initialization
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH) #다른 hook에서 이 값이 True면 이 배치를 건너뜀. (예: loss.backward() 스킵)
            logger.warning(f"Skip batch: non-finite loss={loss.item()}")
            return
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # 5) 첫 유효 토큰 정확도
        #LLM 학습에서는 labels에 -100(IGNORE_INDEX)이 섞여 있음 → loss/acc에 포함되지 않음.
        #여기서는 배치마다 “첫 번째 유효 토큰”만 뽑아서 accuracy를 계산.
        #new_labels = [-100, -100, -100, -100, 0, -100]

        IGN = DefaultToken.IGNORE_INDEX.value # 보통 -100
        valid_mask = (new_labels != IGN) # 유효 토큰 위치만 True
        has_valid = valid_mask.any(dim=1) # 각 샘플에 유효 토큰이 있는지 여부

        B = new_labels.size(0)
        arangeB = torch.arange(B, device=new_labels.device) #[0,1, ..., B-1]
        first_idx = torch.argmax(valid_mask.int(), dim=1) #샘플별 처음으로 유효 토큰에 등장하는 위치

        sel_b = arangeB[has_valid] #유효 토큰 있는 Index들만
        sel_t = first_idx[has_valid] #유효 토큰 있는 샘프들 기준 첫 유효 토큰 위치 집합.

        if sel_b.numel() > 0:
            logits_1st = new_logits[sel_b, sel_t, :] #new logit 중 유효 토큰 있는 것들 한해서 전체 vocab 중 classifier에 해당하는 축소된 logit 반환. 즉  (|sel_b|, 1, C) 차원.
            labels_1st = new_labels[sel_b, sel_t] #|sel_b|, 1 Classification true label만 뽑음.
            pred_1st = torch.argmax(logits_1st, dim=-1) #|sel_b| 개에 대해 예측 label 뽑음.

            sample_correct = int((pred_1st == labels_1st).sum().item()) #맞춘 갯수
            sample_count = int(has_valid.sum().item()) #전체 sample 갯수.
        else:
            sample_correct, sample_count = 0, 0



        ctx.sample_correct_batch = sample_correct #배치 내에서 유효한 것 중 맞춘 것 갯수
        ctx.sample_count_batch = sample_count #배치 내의 유효한 샘플 수.



        # 평탄화 후 전체 토큰 단위 예측/정답 저장
        flat_labels = new_labels.view(-1) #[B*(T-1)]
        flat_logits = new_logits.view(-1, new_logits.size(-1)) #[B*(T-1), V]



        keep = (flat_labels != IGN) #무시 하지말아야할 토큰 찾기.
        flat_logits = flat_logits[keep, :] #무시해야할 토큰들만 제거 #[N_valid, V]
        flat_labels = flat_labels[keep] #무시해야할 토큰들만 제거 #[N_valid]

        _, predicted = flat_logits.max(1) #[N_valid]



        ctx.y_true = CtxVar(flat_labels, LIFECYCLE.BATCH) #[N_valid]
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH) #[N_valid]
        ctx.y_prob = CtxVar(flat_logits, LIFECYCLE.BATCH) #[N_valid, V]

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH) #Scalar.
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH) #B. 일반적으로 B=N_valid.

 

 



def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer': return RewardChoiceTrainer

register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
