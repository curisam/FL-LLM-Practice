import torch
import logging
import gc
import math

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None

import accelerate
from accelerate import Accelerator

from transformers import AdamW

# from torch.optim import AdamW

from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataloader.dataloader import get_tokenizer



from federatedscope.core.auxiliaries.decorators import use_diff




logger = logging.getLogger(__name__)

import sys

sys.setrecursionlimit(100000)





class LLMTrainer(GeneralTorchTrainer): #**Large Language Model (LLM)**을 학습할 수 있도록 확장된 버전.

#즉, 일반적인 Trainer로는 너무 느리거나 메모리를 너무 많이 잡아먹는 LLM을 학습하려면,다음 두 가지가 필요함.
#### 1. 효율적인 메모리 관리와 GPU 분산 학습 → DeepSpeed -> 실제로 적용 안함.
#### 2. Mixed precision (예: bf16)과 gradient accumulation → Accelerate-> 이게 실제로 적용됨.


    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        num_train_batch = len(data['train'])#train dataloader의 length. micro batch size 기준 정해진 것.


        """

        우리가 생각하는 것은 effective_batch

        effective batch size = batch_size × grad_accum_step (1-gpu)

        effective batch size = per_device_batch_size × grad_accum_step × world_size (프로세스 수) (다중 gpu, DDP)



        이를 고려하여 grad_accum_step, world_size 등을 고려하면 된다.


        LLM 학습용: llm.grad_accum_step → LLM forward/backward 반복 횟수

        일반 학습용: grad.grad_accum_count → non-LLM forward/backward 반복 횟수

        
        """



        #Gradient Accumulation Step 계산→ 한 번의 optimizer.step()을 여러 mini-batch에 걸쳐 수행할 수 있도록 설정합니다. 
        self.grad_accum_step = min(
            num_train_batch,
            max(config.llm.grad_accum_step, config.grad.grad_accum_count)) #config.llm.grad_accum_step
        
        
        #Accelerator 사용 여부에 따라 Accelerator 초기화->이건 🤖 HuggingFace accelerate 라이브러리로, bf16 학습 및 자동 Device 배분, gradient accumulate, DDP 등을 쉽게 해주는 wrapper입니다.

        # if config.llm.accelerator.use: #TRUE. 

        #     """
        #     Accelerator.state 내부 동기화 구조

        #         accelerator.state.process_index  # 현재 프로세스 번호
        #         accelerator.state.num_processes  # 전체 프로세스 수
        #         accelerator.state.device         # = accelerator.device
        #     """

        #     self.accelerator = Accelerator(
        #         gradient_accumulation_steps=self.grad_accum_step,
        #         mixed_precision='bf16')
            
        #     device = self.accelerator.device #self.accelerator.device는 해당 프로세스가 맡은 GPU device (torch.device) 객체



        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')

        #tokenizer 준비
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len)
        
        self.eval_metrics = config.eval.metrics #[loss, acc]

    def register_default_hooks_train(self):
        super().register_default_hooks_train()

        #추가적으로 memory 절약용 hook 등록
        self.register_hook_in_train(self._hook_on_fit_end_free_space,
                                    "on_fit_end")

    def register_default_hooks_ft(self):
        super().register_default_hooks_ft()
        #추가적으로 memory 절약용 hook 등록:
        self.register_hook_in_ft(self._hook_on_fit_end_free_space,
                                 "on_fit_end")

    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        #추가적으로 memory 절약용 hook 등록:
        self.register_hook_in_eval(self._hook_on_fit_end_free_space,
                                   "on_fit_end")

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step=-1): #batch 1개 도는 상황.
        # 1) grad_accum_step
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            grad_accum_step = self.grad_accum_step
        else:
            grad_accum_step = 1  #EVAL 모드면 누적 없이 한 배치씩 처리

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return

        # 🔁 분산 셔플 시드 고정(Train 전용). effective batch를 process num만큼 쪼개는 과정. (Multiprocess일 때)
        # DDP일 때 모든 랭크가 동일한 셔플 시드 체계를 공유하도록 매 에폭마다 set_epoch.
        if split == "train":
            sampler = getattr(loader, "sampler", None)
            try:
                from torch.utils.data.distributed import DistributedSampler
                if isinstance(sampler, DistributedSampler): #True
                    import ipdb; ipdb.set_trace(context=15)
                    epoch_seed = int(getattr(self.ctx, "cur_epoch_i", 0))
                    sampler.set_epoch(epoch_seed)
            except Exception:
                pass


        iter_name = f"{split}_loader_iter"

        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            if run_step == -1: #train 떄 일반적으로 이것에 걸림.
                num_batches = getattr(self.ctx, f"num_{split}_batch") ##일반적인 epoch당 batch 개수

                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    # (microbatch size*grad_accum_step) 사이즈를 num_batches만큼 처리하기 위함
                    run_step = num_batches * max(1, grad_accum_step)

                else: #for k in range(grad_accum_step): 루프가 추가로 더 있을 예정.
                    run_step = math.ceil(num_batches / max(1, grad_accum_step))



            # 이터레이터: 필요시 재시작(데이터가 더 적은 경우를 대비)
            if not hasattr(self.ctx, iter_name):
                setattr(self.ctx, iter_name, iter(loader))
            data_loader_iter = getattr(self.ctx, iter_name)

        else:
            # ✅ EVAL: 샤딩된 로더 길이만큼만 1회
            if run_step == -1:
                run_step = len(loader)
            # 이터레이터: 매 평가 루틴마다 새로 시작, StopIteration → 종료
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

        exhausted = False
        for update_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(update_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator'): #accelerator 모드에서 실행. 
                # ✅ grad_accum_step 루프 제거: accumulate 블록만 사용
                try:
                    self.ctx.data_batch = next(data_loader_iter)
                except StopIteration:
                    if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                        # TRAIN: 스텝이 남아 있으면 데이터 재시작
                        data_loader_iter = iter(loader)
                        setattr(self.ctx, iter_name, data_loader_iter)
                        try:
                            self.ctx.data_batch = next(data_loader_iter)
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        # EVAL: 재시작하지 않고 종료
                        exhausted = True
                        break

                with self.accelerator.accumulate(self.ctx.model): 
                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)


            else:
                for k in range(grad_accum_step):
                    self.ctx.cur_batch_i = CtxVar(update_i * grad_accum_step + k,
                                                LIFECYCLE.BATCH)
                    try:
                        self.ctx.data_batch = next(data_loader_iter)
                    except StopIteration:
                        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                            data_loader_iter = iter(loader)
                            setattr(self.ctx, iter_name, data_loader_iter)
                            try:
                                self.ctx.data_batch = next(data_loader_iter)
                            except StopIteration:
                                exhausted = True
                                break
                        else:
                            exhausted = True
                            break

                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)

            if exhausted:
                break

    







    @lifecycle(LIFECYCLE.BATCH) #특별한 일 없으면 .run_step=1로 실행되는 상황.
    #배치 루프 오버라이드, 한 에폭(epoch) 동안의 로컬 학습 “배치 처리” 전체를 담당하는 메서드
    def _run_batch(self, hooks_set, run_step=-1):  #Accelerator.accumulate 기반 gradient accumulation 구현, 새로 추가된 거임. run_step: 외부 루프 횟수; 기본값 -1 이면 아래에서 새로 계산

        """
        self._cfg.train.batch_or_epoch에 따라 루프를 다르게 적용.

        run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")은 micro batch 기준 한 epoch당 iteration 갯수임.


        1. epoch일 떄:

            run_step 수를 micro batch가 아닌 effective batch 수 기준으로 수정. -> grad_accum_step로 나눠준다.

            grad_accum_step * run_step =getattr(self.ctx, f"num_{self.ctx.cur_split}_batch") 되게 함.


        2. batch일 떄:
            run_step 수를 micro batch가 기준 한 epoch당 iteration 갯수로 유지

            이로 인해 한 epoch을 effect batch 사이즈의 batch를 micro batch가 기준 한 epoch당 iteration 갯수만큼 돌리려 함.
        
        """

 

        #grad_accum_step 계산

        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]: #train, ft 인 상황 한해서만
            grad_accum_step = self.grad_accum_step #onfig에 설정된 self.grad_accum_step만큼 배치 단위로 그라디언트를 누적(accumulate)
        else: #eval
            grad_accum_step = 1 #EVAL 모드면 누적 없이 한 배치씩 처리


        
        if run_step == -1: #run_epoch에서 -1로 반영.
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")#원래 한 에폭당 배치 수 계산, e.g.) 데이터셋에 100개의  micro 미니 배치(batch)가 있으면 run_step = 100 이 됩니다. effective batch size 기준이 아님.


            if self._cfg.train.batch_or_epoch == 'epoch': # 한 에폭 당 업데이트 횟수 = #effective batch size 기준으로 run_step이 바뀜.
                run_step = math.ceil(run_step / grad_accum_step) #effective batch size 기준으로 run_step이 바뀜.
                #grad_accum_step이 4라면, 4개 미니배치를 모아서 한 번의 backward+update 를 하겠다는 뜻입니다.
                #따라서 업데이트 횟수 = 100 ÷ 4 = 25 (나머지 처리 위해 ceil) → run_step = 25
                #만약 이 나눗셈이 없으면, 외부 루프가 100번 돌아가면서 내부에서 또 4번씩 mini-batch를 처리해 버려, 실제로 400개의 배치를 읽게 돼 버립니다.

        # --- 데이터로더 이터레이터 준비 ---
        iter_name = f"{self.ctx.cur_split}_loader_iter"
        if not hasattr(self.ctx, iter_name):
            data_loader = self.ctx.get(f"{self.ctx.cur_split}_loader")
            if data_loader is not None:
                setattr(self.ctx, iter_name, iter(data_loader))
        if not hasattr(self.ctx, iter_name):
            return

        data_loader_iter = getattr(self.ctx, iter_name)

        for batch_i in range(run_step): #self._cfg.train.batch_or_epoch == 'epoch'일 떄  batch_i는 “업데이트 스텝 인덱스” (0…24) 입니다. effective batch의 index. 
            #batch_i가 effective batch를 의미하는 상황.
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH) #ctx.cur_batch_i에 저장하면, 훅이나 로깅 로직이 이 스텝 번호를 참조 가능. 
            
            if hasattr(self, 'accelerator'): #accelerator 모드에서 실행. with self.accelerator.accumulate(self.ctx.model) 블록을 빠져나가는 시점에 backward()를 모아둔 gradient를 합산하고 optimizer.step() + optimizer.zero_grad() 까지 한 번에 수행.
                # Build gradient accumulation upon accelerator
                for _ in range(grad_accum_step): # 내부에서 grad_accum_step번 mini-batch forward/backward
                    with self.accelerator.accumulate(self.ctx.model): #이 블록 안에서 여러 micro-batch가 모여 처리됨.

                        # --- 데이터 배치 직접 가져오기 (다시 추가) ---
                        try:
                            self.ctx.data_batch = next(data_loader_iter)
                        except StopIteration:
                            # 이터레이터 재설정
                            data_loader = self.ctx.get(f"{self.ctx.cur_split}_loader")
                            data_loader_iter = iter(data_loader)
                            setattr(self.ctx, iter_name, data_loader_iter)
                            self.ctx.data_batch = next(data_loader_iter)

                        # data_batch 설정 후 훅 호출
                        for hook in hooks_set["on_batch_start"]:
                            hook(self.ctx)

                        for hook in hooks_set["on_batch_forward"]:
                            hook(self.ctx)

                        for hook in hooks_set["on_batch_backward"]:
                            hook(self.ctx)

                        for hook in hooks_set["on_batch_end"]:
                            hook(self.ctx)



            else: #(비-accelerator 모드일 때) 실제 미니배치 인덱스 관리. fixed된 effective batch 내에서 세분화하여 작업.
                for idx in range(grad_accum_step): #이 for문이 더 들어가는게 accelerate 버전과의 차이. idx는 fixed된 effective batch 내에서 몇번째 micro batch인지에 대한 설명.
                    self.ctx.cur_batch_i = CtxVar(
                        batch_i * grad_accum_step + idx, LIFECYCLE.BATCH) #batch_i * grad_accum_step + idx 는 전체 에폭 내에서 지금 처리 중인 실제 미니배치 번호 (0…99). 

                    #여기서는 batch_i가 micro batch를 의미하는 상황. 훅(on_batch_*)가 micro-batch 단위로 매 forward/backward마다 호출됩니다.
                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)


                if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                    if (self.ctx.cur_batch_i + 1) % grad_accum_step == 0:
                        current_lr = self.ctx.optimizer.param_groups[0]['lr']
                        round_num = getattr(self.ctx, 'cur_round_i', 'N/A')
                        client_id = getattr(self.ctx, 'ID', 'N/A')
                        logger.info(
                          f"Client #{client_id} Round #{round_num} "
                          f"Local Step #{self.ctx.cur_batch_i+1}: LR = {current_lr:.2e}"
                        )


            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1:# 학습/파인튜닝 모드 확인 및 마지막 에폭인지 확인
                if batch_i >= self.ctx.num_train_batch_last_epoch - 1: #마지막 에폭의 실제 스텝(업데이트) 수 초과 여부 판정
                    break #조기 종료(break)



 
    # def _hook_on_fit_start_numerical_precision(self, ctx):
    #     if self.cfg.train.is_enable_half:
    #         if not ctx.cfg.llm.deepspeed.use and \
    #            not ctx.cfg.llm.accelerator.use:
    #             ctx.model.to(torch.bfloat16)

    # --- [수정 5] _hook_on_fit_start_init에서 로더 생성 로직 제거 ---
    def _hook_on_fit_start_init(self, ctx):  #accelerate 및 deepspeed 지원 추가

        """세 가지 경우로 나뉩니다:
        ✅ Accelerator 사용 시

        self.accelerator.prepare(...) → model, optimizer, dataloader, scheduler 모두 감싸서 Mixed precision, DDP 등을 처리해줌
        
        ✅ Deepspeed 사용 시 ->이건 해당안하니 안봐도 될 것 같음.
        deepspeed.initialize(...)→ deepspeed 방식으로 모델을 분산/효율적으로 학습 가능하도록 초기화


        ✅ 기본 PyTorch만 사용하는 경우
        → 그냥 .to(device) 후 옵티마이저 세팅
        
        """

        # --- 로더 복원: 이전 라운드에서 지웠던 train/val/test 로더를 다시 바인딩 ---
        # self.ctx.data 에 따라 두 케이스를 모두 지원:
        #   1) 이미 DataLoader 를 들고 있는 dict 형태: self.ctx.data['train'|'val'|'test']
        #   2) raw dataset 을 들고 있는 컨테이너: ctx.get(f"{split}_data") 또는 self.ctx.data.<split>_data
        for split in ["train", "val", "test"]:
            loader_key = f"{split}_loader"
            if getattr(ctx, loader_key, None) is None:
                # (1) dict에 DataLoader가 직접 들어온 경우
                if isinstance(self.ctx.data, dict) and split in self.ctx.data:
                    setattr(ctx, loader_key, self.ctx.data[split])
                else:
                    # (2) raw dataset에서 새로 로더를 만든다
                    raw = ctx.get(f"{split}_data", None)
                    if raw is None and hasattr(self.ctx.data, f"{split}_data"):
                        raw = getattr(self.ctx.data, f"{split}_data")
                    if raw is not None:
                        dl = get_dataloader(WrapDataset(raw), self.cfg, split)
                        setattr(ctx, loader_key, ReIterator(dl))
        # (안전) 이전 라운드 잔여 이터레이터가 남아있지 않도록 보장
        for it_name in ["train_loader_iter", "val_loader_iter", "test_loader_iter"]:
            if hasattr(ctx, it_name):
                delattr(ctx, it_name)



        # --- [핵심 추가 1] 매 라운드 accelerator 새로 생성 ---
        if ctx.cfg.llm.accelerator.use:
            logger.info("Re-creating Accelerator for the new round.")
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_step,
                mixed_precision='bf16'
            )
            ctx.device = self.accelerator.device

            # 원본 모델 추출
            unwrapped_model = ctx.model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            if hasattr(unwrapped_model, 'sharding'):
                unwrapped_model.sharding()

            # (학습 시) 옵티마이저/스케줄러 준비
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                cfg_optimizer = ctx.cfg[ctx.cur_mode].optimizer
                if cfg_optimizer.type == 'AdamW':
                    adamw_kwargs = {'lr': cfg_optimizer.lr, 'betas': cfg_optimizer.betas}
                    ctx.optimizer = AdamW(unwrapped_model.parameters(), **adamw_kwargs)
                else:
                    ctx.optimizer = get_optimizer(unwrapped_model, **cfg_optimizer)
                ctx.scheduler = get_scheduler(ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)
                current_lr = ctx.optimizer.param_groups[0]['lr']
                round_num = getattr(ctx, 'cur_round_i', 'N/A')
                logger.info(f"Round #{round_num} - Initializing with LR: {current_lr}")

                # ⛔️ 여기서 DataLoader를 prepare에 넘기지 않습니다!
                ctx.model, ctx.optimizer = self.accelerator.prepare(unwrapped_model, ctx.optimizer)
            else:
                # 평가 모드: 모델만 prepare
                ctx.model = self.accelerator.prepare(unwrapped_model)


        elif ctx.cfg.llm.deepspeed.use: #FALSE
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            # if ctx.cfg.train.is_enable_half:
            #     ctx.fp16 = ctx.model_engine.fp16_enabled()

        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                
                if not hasattr(self, 'scheduler') or self.scheduler is None:
                    self.scheduler = get_scheduler(ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)
                ctx.scheduler = self.scheduler

                # [디버깅 로그] 현재 LR 출력
                current_lr = ctx.optimizer.param_groups[0]['lr']
                round_num = getattr(ctx, 'cur_round_i', 'N/A')
                logger.info(f"Round #{round_num} - Current Learning Rate: {current_lr}")

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

        if hasattr(ctx, 'ys_pred'): 
            ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.model.train()
        else:  # MODE.TEST or MODE.VAL
            ctx.model.eval()

            
    def _hook_on_epoch_start(self, ctx):
        pass

 

    def _hook_on_batch_forward(self, ctx): #ctx.data_batch → input_ids, attention_mask, labels로 명시 처리 + 모델 호출 방식 변경. 기존 toprch trainer는 CNN, MLP, BERT 용이라 LLM용은 아니라서 OVERRIDE 해야함.

        """ 
        (기존)->일반 모델 (예: CNN, MLP, BERT 등)용
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        loss = ctx.criterion(pred, label)

        🔍 특징
        ctx.data_batch는 (x, label) 튜플 형태

        model(x)만으로 예측 가능

        loss = criterion(pred, label)로 손실 계산

        매우 단순하고 직관적인 구조



        (OVERRIDE 버전)->LLM 학습용 override


        input_ids = ctx.data_batch['input_ids']
        labels = ctx.data_batch['labels']
        attention_mask = ctx.data_batch['attention_mask']
        ...
        outputs = model(input_ids=..., labels=..., attention_mask=...)


        🔍 주요 변경 이유 요약
        변경 요소	이유
        data_batch가 dict로 바뀜	LLM은 (input_ids, attention_mask, labels) 형태로 학습
        model(input_ids=..., labels=...) 사용	Huggingface LLM은 forward()에서 loss까지 같이 반환함
        ctx.model_engine(...) 분기	DeepSpeed 등 사용할 경우 Accelerator나 Engine이 모델을 wrapping
        skip_this_batch 추가	LLM 학습에서 NaN loss가 빈번 → 방어 로직 필요


        """


        if ctx.cfg.llm.accelerator.use: #참고


            #LLM 학습용 Dataset은 보통 tokenized_dict를 반환->{'input_ids': Tensor, 'labels': Tensor, 'attention_mask': Tensor}
            #accelerator는 입력 텐서를 자동으로 디바이스에 올려줘서 .to(ctx.device)가 필요 없음.


            #input_ids: [101,  42,  53,  78,   2]
            #labels:    [101,  42,  53,  78,   2]  # 혹은 일부 -100 포함
            #이렇게 거의 같지만, loss 계산에서 무시할 토큰은 labels에서만 바뀜.
            
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

            #모델 호출이 model(x) → model(input_ids=..., labels=...)로 바뀜→ loss를 직접 criterion으로 계산하지 않고, 모델 자체가 계산해서 outputs.loss로 줌. loss는 cross-entropy로 적용하게 hugging face가 설계. forward를 override하면 loss 바꾸는 것 가능.
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask) # 자동으로 모델과 배치 텐서 모두 디바이스로 이동

        elif ctx.cfg.llm.deepspeed.use: #ctx.model_engine 기반으로 outputs 뽑아낸다.

            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)


            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:#참고, #ctx.model 기반으로 outputs 뽑아낸다.
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)



            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits #일반적으로 LLM에서는 [batch_size, seq_len, vocab_size] shape의 tensor
        loss = outputs.loss #일부 모델에서 reduction='none'이면 shape가 [batch_size] 혹은 [batch_size, seq_len]일 수도 있지만, 대부분은 1개의 스칼라 값

        #NaN loss가 발생시키는 batch 건너뛰기 위한 방어 로직
        if torch.isnan(loss): #LM 학습에서는 종종 NaN loss가 발생할 수 있음-> label이 모두 -100 (ignore index). precision 문제 (e.g., bf16/float16). exploding gradients, bad initialization
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH) #다른 hook에서 이 값이 True면 이 배치를 건너뜀. (예: loss.backward() 스킵)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH) #shape: [batch_size, seq_len]
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH) # shape: [batch_size, seq_len, vocab_size]
 
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx): #accelerate, deepspeed에 따른 backward, step, zero_grad 처리 분기

        """
        Accelerator 또는 Deepspeed 사용 시 전용 backward/step 로직
            self.accelerator.backward()

            ctx.model_engine.backward() 등

        그렇지 않은 경우, 일반 loss.backward()에 grad_accum_step 고려하여 나눠줌

        """


        if ctx.skip_this_batch: #스킵 플래그(skip_this_batch)가 켜져 있으면 아무 작업도 하지 않고 바로 리턴
            return

        if ctx.cfg.llm.accelerator.use:
            # Accelerate가 grad accumulation을 관리하므로 손실을 나누지 않습니다.
            self.accelerator.backward(ctx.loss_task)
            # ✅ 누적 경계(sync_gradients=True)에서만 step/zero_grad 실행
            if getattr(self.accelerator, "sync_gradients", True):
                # (보완) 경계에서만 gradient clipping 수행
                if getattr(ctx, "grad_clip", 0) and ctx.grad_clip > 0:
                    try:
                        self.accelerator.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        elif ctx.cfg.llm.deepspeed.use: #FALSE
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()

            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0: #조건부 업데이트. cur_batch_i+1) % grad_accum_step == 0일 때만
                if ctx.grad_clip > 0: #그래디언트 클리핑. ->ctx.grad_clip > 0일 경우 clip_grad_norm_ 적용
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                   ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        # # move the training data to cpu.
        # ctx.data_batch['input_ids'].cpu()
        # ctx.data_batch['labels'].cpu()
        # ctx.data_batch['attention_mask'].cpu()

        # move the training data to cpu (안전하게 키 확인)
        if 'input_ids' in ctx.data_batch:
            ctx.data_batch['input_ids'].cpu()
        if 'labels' in ctx.data_batch:
            ctx.data_batch['labels'].cpu()
        if 'attention_mask' in ctx.data_batch:
            ctx.data_batch['attention_mask'].cpu()


    def _hook_on_batch_end(self, ctx): #NaN loss일 경우 retry 로직 추가

        """
        기본 로직 + loss == NaN일 경우 batch를 retry

        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                self._run_batch(...)
        """
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))


    def _hook_on_fit_end(self, ctx): #기존과 거의 동일, 평균 loss만 계산. override되는 과정에서 accuracy는 빠진다.

        return

    def _hook_on_fit_end_free_space(self, ctx):
        # 옵티마이저, 스케줄러, 임시 텐서, 이터레이터만 정리
        attributes_to_delete = [
            'optimizer', 'scheduler', 'loss_batch', 'loss_task', 'loss_regular',
            'loss_batch_total', 'loss_regular_total', 'y_true', 'y_prob',
            'ys_true', 'ys_pred',
            'data_batch', 'grad', 'model_engine', 'skip_this_batch',
            # 데이터로더와 이터레이터를 모두 삭제하여 메모리 누수를 방지
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter'
        ]

        # 로그 추가 (선택사항이지만 디버깅에 유용)
        if hasattr(ctx, 'rank') and ctx.rank == 0:
            deleted_attrs = [attr for attr in attributes_to_delete if hasattr(ctx, attr)]
            logger.info(f"[Memory Cleanup] Deleting attributes: {deleted_attrs}")


        for attr in attributes_to_delete:
            if hasattr(ctx, attr):
                # delattr을 사용하여 ctx 객체에서 해당 속성 참조를 완전히 제거
                delattr(ctx, attr)

        # --- [핵심] accelerator 객체 자체를 삭제 ---
        if hasattr(self, 'accelerator'):
            del self.accelerator
            logger.info("Accelerator object has been deleted.")

        gc.collect()
        torch.cuda.empty_cache()



    def _hook_on_batch_forward_flop_count(self, ctx): #PASS해도 될듯

        """
        모델 구조에 따라 LLM의 input (input_ids, attention_mask)을 이용해 FLOPs를 계산.

        fvcore.nn.FlopCountAnalysis를 사용함.
        """



        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
