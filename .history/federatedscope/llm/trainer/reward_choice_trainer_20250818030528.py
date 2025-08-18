import os
import gc
import sys
import math
import copy
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
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


def _get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    try:
        ws = int(os.environ.get('WORLD_SIZE', '1'))
        rk = int(os.environ.get('RANK', '0'))
        return (ws > 1), ws, rk
    except Exception:
        return False, 1, 0

class EvalShardSampler(Sampler):
    def __init__(self, dataset_len: int, rank: int, world_size: int):
        self.indices = list(range(rank, int(dataset_len), int(world_size)))
    def __iter__(self): return iter(self.indices)
    def __len__(self): return len(self.indices)
    
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
        
    def _reset_and_build_dataloader(self, split_name): #학습/평가에 들어가기 직전마다 로더를 다시 만들고(샤딩 반영)
        data_key = f"{split_name}_data"
        loader_key = split_name
        ctx_loader_key = f"{split_name}_loader"

        client_data_obj = self.ctx.data

        # 기존 로더 제거. 같은 라운드/루틴 반복 시 낡은 이터레이터/샘플러 상태가 섞이지 않도록 기존 로더를 깨끗이 지움.
        if loader_key in client_data_obj:
            del client_data_obj[loader_key]

        # 원본 데이터가 있으면 로더 생성
        if hasattr(client_data_obj, data_key) and getattr(client_data_obj, data_key) is not None:
            dataset = WrapDataset(getattr(client_data_obj, data_key))
            base_loader = get_dataloader(dataset, self.cfg, split_name) #sharding 이전의 상황. 샤딩 전(sampler=없거나 기본).


            #dist_ready: torch.distributed 프로세스 그룹이 초기화되어 있는지 여부(bool)

            #world_size: 전체 프로세스 수(예: 4)

            #rank: 현재 프로세스의 전역 랭크(0,1,2,3 …)

            dist_ready, world_size, rank = _get_dist_info()

            if world_size > 1:
                if split_name == "train":

                   # DistributedSampler (훈련용)
                    #### 목적: 각 에폭마다 셔플하고, 각 rank에 동일 개수의 샘플을 배분.

                    #### 작동:
                    ######## 에폭마다 set_epoch(e)로 시드 고정 → 전 rank가 동일한 퍼뮤테이션을 공유.

                    ######## 전체 인덱스를 섞은 뒤 동일한 길이로 균등 분할.

                    ######## drop_last=False면 길이가 나누어떨어지지 않을 때 앞쪽 인덱스를 재사용(패딩) 해서 num_samples = ceil(N / world_size)를 맞춤 → 한 에폭 내 중복 가능(훈련에서는 허용/권장).

                    #### 길이:

                    ######## len(sampler) = num_samples = ceil(N / world_size) (drop_last=False)

                    ######## 특징: 셔플/패딩/균등분할/set_epoch 지원 → 훈련 친화적.


                    # num_samples = ceil(10/4)=3, total_size=12 → 앞에서 2개 패딩(중복)

                    # rank0: [0,1,2]
                    # rank1: [3,4,5]
                    # rank2: [6,7,8]
                    # rank3: [9,0,1] ⟵ 패딩된 중복

                    # 장점: 각 rank 3개로 균등 / 단점: 중복 존재(훈련 OK)

                    sampler = DistributedSampler(
                        base_loader.dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True,
                    )
                    self._train_dist_sampler = sampler

                    loader = DataLoader(
                        dataset=base_loader.dataset,
                        batch_size=base_loader.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=getattr(base_loader, "num_workers", 0),
                        pin_memory=getattr(base_loader, "pin_memory", False),
                        drop_last=getattr(base_loader, "drop_last", False),
                        collate_fn=getattr(base_loader, "collate_fn", None),
                    )
                else:
                    # EvalShardSampler

                    # rank0: [0,4,8] (3개)
                    # rank1: [1,5,9] (3개)
                    # rank2: [2,6] (2개)
                    # rank3: [3,7] (2개)

                    # 장점: 중복 없음, 전체 정확히 10개 평가 / 단점: 각 rank 길이가 균등하지 않을 수 있음(평가 OK)

                    sampler = EvalShardSampler(len(base_loader.dataset), rank, world_size)

                    loader = DataLoader(
                        dataset=base_loader.dataset,
                        batch_size=base_loader.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                        drop_last=False,
                        collate_fn=getattr(base_loader, "collate_fn", None),
                    )

                sharded = True
                local_count = len(sampler)
            else:
                loader = base_loader
                sharded = False
                local_count = len(base_loader.dataset)

            client_data_obj[loader_key] = loader
            setattr(self.ctx, ctx_loader_key, loader)

            logger.info(
                f"Dataloader for '{split_name}' has been reset and recreated. "
                f"(sharded={sharded}, "
                f"world_size={1 if not sharded else world_size}, "
                f"rank={0 if not sharded else rank}, "
                f"local_count={local_count}, "
                f"total={len(base_loader.dataset)})"
            )



    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None, round_num=-1):
        # [수정] current_round를 맨 먼저 정의하여 NameError 해결
        current_round = round_num
        self.ctx.current_round_num = current_round

        scheduler_cfg = getattr(self.cfg.train, "scheduler", None)
        if scheduler_cfg:
            initial_lr = self.cfg.train.optimizer.lr
            milestones = getattr(scheduler_cfg, "milestones", [])
            gamma = getattr(scheduler_cfg, "gamma", 1.0)
            
            num_decays = sum(1 for milestone in sorted(milestones) if current_round >= milestone)
            new_lr = initial_lr * (gamma ** num_decays)
            
            self.ctx.new_lr_to_be_set = new_lr
            logger.info(
                f"[Stateless LR Controller] In Round #{current_round}, planning to set LR to {new_lr:.2e}"
            )

        self._reset_and_build_dataloader(target_data_split_name)
        return super().train(target_data_split_name, hooks_set)

    def evaluate(self, target_data_split_name="test", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_eval
        self._reset_and_build_dataloader(target_data_split_name)
        with torch.no_grad():
            if self.ctx.check_split(target_data_split_name, skip=True):
                self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
            else:
                self.ctx.eval_metrics = dict()
        return self.ctx.eval_metrics

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        
        if hasattr(ctx, "new_lr_to_be_set"):
            new_lr = ctx.new_lr_to_be_set
            if hasattr(ctx, 'optimizer') and ctx.optimizer is not None:
                opt = ctx.optimizer
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"Successfully applied new LR {new_lr:.2e} to the optimizer.")
                del ctx.new_lr_to_be_set
        
        current_round = int(getattr(ctx, "current_round_num", getattr(ctx, "cur_state", 0)))
        self.choices = self.choices_cpu.to(ctx.device, non_blocking=True)
        for sp in ["train", "val", "test"]:
            setattr(ctx, f"num_samples_{sp}", 0); setattr(ctx, f"loss_total_{sp}", 0.0); setattr(ctx, f"correct_{sp}", 0)
        ctx.num_samples = 0; ctx.loss_batch_total = 0.0; ctx.loss_regular_total = 0.0
        ctx.sample_seen = 0; ctx.sample_correct_accum = 0

        sampler = getattr(self, "_train_dist_sampler", None)

        print(sampler)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(current_round)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch["input_ids"].to(ctx.device)
        labels = ctx.data_batch["labels"].to(ctx.device)
        attention_mask = ctx.data_batch["attention_mask"].to(ctx.device)

        # 모델 실행 (가속/디프스피드/일반)
        if ctx.cfg.llm.accelerator.use:
            outputs = ctx.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        elif ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        else:
            outputs = ctx.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        logits = outputs.logits
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)

        # NaN 방지
        if torch.isnan(loss):
            ctx.skip_this_batch = True
            logger.warning("Skip the batch due to NaN loss.")
            return
        else:
            ctx.skip_this_batch = False

        # 첫 유효 토큰 기준 샘플 정확도 계산
        IGN = DefaultToken.IGNORE_INDEX.value
        valid_mask = (new_labels != IGN)
        has_valid = valid_mask.any(dim=1)

        B = new_labels.size(0)
        arangeB = torch.arange(B, device=new_labels.device)
        first_idx = torch.argmax(valid_mask.int(), dim=1)

        sel_b = arangeB[has_valid]
        sel_t = first_idx[has_valid]

        if sel_b.numel() > 0:
            logits_1st = new_logits[sel_b, sel_t, :]
            labels_1st = new_labels[sel_b, sel_t]
            pred_1st = torch.argmax(logits_1st, dim=-1)

            sample_correct = int((pred_1st == labels_1st).sum().item())
            sample_count = int(has_valid.sum().item())
        else:
            sample_correct, sample_count = 0, 0

        ctx.sample_correct_batch = sample_correct
        ctx.sample_count_batch = sample_count

        # 평탄화 후 전체 토큰 단위 예측/정답 저장
        flat_labels = new_labels.view(-1)
        flat_logits = new_logits.view(-1, new_logits.size(-1))

        keep = (flat_labels != IGN)
        flat_logits = flat_logits[keep, :]
        flat_labels = flat_labels[keep]

        _, predicted = flat_logits.max(1)

        ctx.y_true = CtxVar(flat_labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(flat_logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

 

 
    def _hook_on_batch_end(self, ctx):
        # 스킵 처리
        skip = getattr(ctx, "skip_this_batch", False)
        if isinstance(skip, CtxVar):
            try:
                skip = bool(skip)
            except Exception:
                skip = False
        if skip:
            return

        # (옵션) 배치별 LR 디버깅
        if ctx.cur_mode == MODE.TRAIN:
            opt = getattr(ctx, "optimizer", None)
            if opt is not None:
                current_lr = opt.param_groups[0]["lr"]
                round_idx = getattr(ctx, "current_round_num", "?")
                batch_idx = getattr(ctx, "batch_idx", "?")
                # logger.info(f"[LR BATCH CHECK] round={round_idx} batch={batch_idx} -> LR={current_lr:.2e}")

        # 누적 집계 키
        split = ctx.cur_split
        loss_total_key = f"loss_total_{split}"
        correct_key = f"correct_{split}"
        num_samples_key = f"num_samples_{split}"

        # 키 초기화
        if not hasattr(ctx, loss_total_key):
            setattr(ctx, loss_total_key, 0.0)
        if not hasattr(ctx, correct_key):
            setattr(ctx, correct_key, 0)
        if not hasattr(ctx, num_samples_key):
            setattr(ctx, num_samples_key, 0)

        # 배치 단위 통계
        batch_raw_samples = int(ctx.batch_size)
        batch_logic_seen = int(getattr(ctx, "sample_count_batch", 0))
        batch_logic_corr = int(getattr(ctx, "sample_correct_batch", 0))

        # 손실 총합/샘플 수
        setattr(
            ctx,
            loss_total_key,
            getattr(ctx, loss_total_key) + ctx.loss_batch.item() * batch_raw_samples,
        )
        setattr(
            ctx,
            num_samples_key,
            getattr(ctx, num_samples_key) + batch_raw_samples,
        )

        # 논리적(선택된) 토큰 기준 정확도 집계
        ctx.sample_seen = int(getattr(ctx, "sample_seen", 0)) + batch_logic_seen
        ctx.sample_correct_accum = int(getattr(ctx, "sample_correct_accum", 0)) + batch_logic_corr

        # 기타 누적
        ctx.num_samples = int(getattr(ctx, "num_samples", 0)) + batch_raw_samples
        ctx.loss_batch_total = float(getattr(ctx, "loss_batch_total", 0.0)) + ctx.loss_batch.item() * batch_raw_samples
        ctx.loss_regular_total = float(getattr(ctx, "loss_regular_total", 0.0)) + float(ctx.get("loss_regular", 0.0))

        # 검증/테스트: 예측 저장
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            if not hasattr(ctx, "ys_true"):
                ctx.ys_true = []
            if not hasattr(ctx, "ys_pred"):
                ctx.ys_pred = []

            pred = torch.argmax(ctx.y_prob, dim=-1)
            ctx.ys_true.append(ctx.y_true)
            ctx.ys_pred.append(pred)



 
 

    def _ddp_reduce_split(self, ctx, split: str):
        acc = getattr(self, "accelerator", None); device = ctx.device
        n = torch.tensor([getattr(ctx, f"num_samples_{split}", 0)], device=device, dtype=torch.long)
        loss_sum = torch.tensor([getattr(ctx, f"loss_total_{split}", 0.0)], device=device, dtype=torch.float32)
        if acc is not None and acc.num_processes > 1:
            n = acc.reduce(n, reduction="sum"); loss_sum = acc.reduce(loss_sum, reduction="sum")
        return int(n.item()), float(loss_sum.item())

    def _hook_on_fit_end(self, ctx):
        split = getattr(ctx, "cur_split", None)
        if split not in ("train", "val", "test"):
            return

        acc = getattr(self, "accelerator", None)
        is_main = (acc.is_main_process if acc is not None else True)
        device = ctx.device

        # 전역 합계/손실 집계
        n_global, loss_global = self._ddp_reduce_split(ctx, split)
        avg_global = float(loss_global / max(n_global, 1))

        # 샘플/정답 집계 텐서화
        t_seen = torch.tensor(
            [int(getattr(ctx, "sample_seen", 0))],
            device=device,
            dtype=torch.long,
        )
        t_corr = torch.tensor(
            [int(getattr(ctx, "sample_correct_accum", 0))],
            device=device,
            dtype=torch.long,
        )

        # 분산이면 합산
        if acc is not None and acc.num_processes > 1:
            t_seen = acc.reduce(t_seen, reduction="sum")
            t_corr = acc.reduce(t_corr, reduction="sum")

        seen_global = int(t_seen.item())
        corr_global = int(t_corr.item())
        acc_global = float(corr_global / max(seen_global, 1)) if seen_global > 0 else 0.0

        if is_main:
            if not hasattr(ctx, "eval_metrics") or ctx.eval_metrics is None:
                ctx.eval_metrics = {}

            ctx.eval_metrics.update({
                f"{split}_total":    n_global,
                f"{split}_loss":     loss_global,
                f"{split}_avg_loss": avg_global,
                f"{split}_acc":      acc_global,
            })

            logger.info(
                f"[{split}|reduce] total={n_global}, "
                f"loss_sum={loss_global:.6f}, avg_loss={avg_global:.6f}, "
                f"acc={acc_global:.6f} "
                f"(local_seen={getattr(ctx,'sample_seen',0)}, "
                f"local_corr={getattr(ctx,'sample_correct_accum',0)})"
            )
        else:
            ctx.eval_metrics = {}



def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer': return RewardChoiceTrainer

register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
