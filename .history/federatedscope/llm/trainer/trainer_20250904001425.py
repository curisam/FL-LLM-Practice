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
from accelerate import Accelerator, DistributedDataParallelKwargs

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

from federatedscope.llm.misc.debug_utils import log_tok_model_sync




logger = logging.getLogger(__name__)

import sys

sys.setrecursionlimit(100000)


# 모든 랭크 동기화용 배리어
from federatedscope.llm.utils_dist import barrier_all




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
        

        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')

        #tokenizer 준비
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len)
        
        self.eval_metrics = config.eval.metrics #[loss, acc]

        self.ctx.current_round_num = 0
        self.ctx.cur_round_i = 0


        if config.llm.accelerator.use:   # ✅ 조건문 추가
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True, # 어댑터 스왑 등 unused 파라미터 대비
                gradient_as_bucket_view=False, # bf16 버킷뷰 이슈 회피
                broadcast_buffers=False
            )
            mp = getattr(config.llm, "mixed_precision", "bf16")
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_step,
                mixed_precision=mp,
                kwargs_handlers=[ddp_kwargs],
            )
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = device

        # __init__ 맨 끝쪽에 가까운 위치
        self._prepared_once = False

        try:
            import torch
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                # 플래시/메모리효율 경로 차단, 수학 커널만
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass


    def _set_round_ctx(self, rnd):
        rnd = int(rnd)
        # 컨텍스트와 트레이너 양쪽에 alias를 모두 채워서 어디서든 참조되게
        for tgt in (self.ctx, self):
            setattr(tgt, "current_round_num", rnd)
            setattr(tgt, "current_round", rnd)
            setattr(tgt, "cur_round_i", rnd)
            setattr(tgt, "round_idx", rnd)


    def _unwrap(self, m):
        """Accelerate/DDP 래핑된 모델에서 원본 HF 모델을 안전하게 얻는다."""

        if self.accelerator is not None: # Accelerate 인스턴스 메서드 우선
            m_unwrapped = self.accelerator.unwrap_model(m)
        else:
            m_unwrapped = m
        # DDP 래퍼면 .module 체인으로 벗기기
        while hasattr(m_unwrapped, "module"):
            m_unwrapped = m_unwrapped.module
        return m_unwrapped



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
    #업데이트 1회를 어떻게 만드는지를 결정.
    def _run_batch(self, hooks_set, run_step=-1): #한 epoch 내에서 batch 1개 도는 걸 여러번 반복. run_epoch에서는 run_step=-1 적용. 다른 곳에서는 run_step=1 적용.
        # 1) grad_accum_step
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            """
            Train 단계:

            gradient: 자동 집계 (DDP).

            loss/metrics: print/logging용으로는 집계 처리 있음 (aggregate).

            raw 기록은 rank0만 남는 구조일 수 있음.

            Eval 단계:

            loss/metrics: 반드시 accelerator.gather_for_metrics() → monitor.aggregate() 거쳐서 global average 기록.
            """
            grad_accum_step = self.grad_accum_step

        else:
            #loss.backward() 자체를 하지 않음 → gradient 없음. 
            # 따라서: 1. optimizer.step() 호출 없음. 
            # 2. gradient accumulation 개념 자체가 동작 안 함. 
            # 3. DDP(DistributedDataParallel)는 forward 시점에만 통신(all-gather) 필요 → backward 없음 → all-reduce도 없음.

 
            
            grad_accum_step = 1  
 

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return



        # DDP일 때 매 epoch마다 모든 랭크가 동일한 셔플 시드 체계를 공유하기 위해 매 에폭마다 set_epoch. 
        # 각 프로세스가 가진 DataLoader 하나가 DistributedSampler(rank, num_replicas) 때문에 ‘자기 몫’만 읽게 함.

        """
        각 랭크는 자기 shard에서 다음 마이크로배치를 순서대로 소모.

        effective batch: 한 “업데이트 1회” 동안

        랭크별로 per_process_batch_size × grad_accum_step 샘플 소비

        전 랭크 합치면 per_process_batch_size × grad_accum_step × world_size (= 글로벌 유효배치)

        """

        if split == "train":
            sampler = getattr(loader, "sampler", None)
            try:
                from torch.utils.data.distributed import DistributedSampler
                if isinstance(sampler, DistributedSampler): #True
                    epoch_seed = int(getattr(self.ctx, "cur_epoch_i", 0))
                    sampler.set_epoch(epoch_seed)
            except Exception:
                pass


        iter_name = f"{split}_loader_iter"

        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            if run_step == -1: #train 떄 일반적으로 이것에 걸림.
                num_batches = getattr(self.ctx, f"num_{split}_batch") ##일반적인 epoch당 batch 개수. 다만 batch_or_epoch == "batch"이면 iteration 수와 비교했을 때 min 값으로 설정됨.

                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    # (microbatch size*grad_accum_step) 사이즈를 num_batches만큼 처리하기 위함

                    #루프 본문: with self.accelerator.accumulate: 안에서 마이크로배치 1개 처리
                    #결과: 마이크로배치 총 소비량이 N×g가 되어, 데이터셋을 g배 더 돎
                    run_step = num_batches * max(1, grad_accum_step)

                    self.ctx.num_train_batch_last_epoch = self.ctx.num_train_batch_last_epoch * max(1, grad_accum_step)

                else: #안쪽에서 for k in range(grad_accum_step)로 마이크로배치 grad_accum_step개 처리 → 업데이트 1회
                    # run_step = math.ceil(num_batches / max(1, grad_accum_step))
                    run_step = num_batches


             # 이터레이터: 필요시 재시작(데이터가 더 적은 경우를 대비)
            if not hasattr(self.ctx, iter_name):
                setattr(self.ctx, iter_name, iter(loader))
            data_loader_iter = getattr(self.ctx, iter_name)

        else:
            # ✅ TEST, VAL: 샤딩된 로더 길이만큼만 1회(grad_accum_step = 1)
            if run_step == -1:
                run_step = len(loader) #microbatch size 기준 길이.
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

                #Accelerate는 내부 카운터로 self.grad_accum_step-1번은 no_sync(통신/step 없음), self.grad_accum_step번째에만 all-reduce + step.
                #따라서 업데이트 1회 = 이 블록이 self.grad_accum_step번 실행된 시점.
                with self.accelerator.accumulate(self.ctx.model): #마이크로배치 1개를 처리하는 단위 블록.
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

            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1:# 학습/파인튜닝 모드 확인 및 마지막 에폭인지 확인
                if update_i >= self.ctx.num_train_batch_last_epoch - 1: #마지막 에폭의 실제 스텝(업데이트) 수 초과 여부 판정
                    break #조기 종료(break)    


 
    # def _hook_on_fit_start_numerical_precision(self, ctx):
    #     if self.cfg.train.is_enable_half:
    #         if not ctx.cfg.llm.deepspeed.use and \
    #            not ctx.cfg.llm.accelerator.use:
    #             ctx.model.to(torch.bfloat16)


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


        # 리셋 직후, 다음 단계로 넘어가기 전에 랭크 동기화
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            torch.cuda.synchronize()
        
        if ctx.cfg.llm.accelerator.use:
            # ✅ Accelerator는 이미 __init__에서 1회 생성됨
            ctx.device = self.accelerator.device

            # 1) 모델: 첫 라운드 1회만 prepare
            base = self._unwrap(self.model)

            if not getattr(self, "_prepared_once", False):
                if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                    opt = get_optimizer(base, **ctx.cfg[ctx.cur_mode].optimizer)
                    ctx.model, ctx.optimizer = self.accelerator.prepare(base, opt)
                else:
                    ctx.model = self.accelerator.prepare(base)
                    ctx.optimizer = None
                # prepared 모델을 self.model에 영구 보관 → 이후 라운드 재사용
                self.model = ctx.model
                self._prepared_once = True


            else:
                # 이미 준비된 래핑 모델 재사용
                ctx.model = self.model
                if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                    # 옵티마이저는 라운드마다 재생성하되, 모델은 re-prepare 금지
                    opt = get_optimizer(self._unwrap(ctx.model), **ctx.cfg[ctx.cur_mode].optimizer)
                    # 옵티마이저만 prepare (모델은 다시 건드리지 않음)
                    ctx.optimizer = self.accelerator.prepare(opt)
                else:
                    ctx.optimizer = None



            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # 안전 접근: 없으면 None
                opt = getattr(ctx, "optimizer", None)
                if opt is None:
                    ctx.optimizer = get_optimizer(unwrapped_model, **ctx.cfg[ctx.cur_mode].optimizer)
                # 스케줄러는 설정이 없을 수 있어서 None으로 통일
                sch = getattr(ctx, "scheduler", None)
                if sch is None:
                    ctx.scheduler = None

                # prepare (모델/옵티마이저)
                ctx.model, ctx.optimizer = self.accelerator.prepare(unwrapped_model, ctx.optimizer)
            else:
                ctx.model = self.accelerator.prepare(unwrapped_model)
        else:
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = None

                # [디버깅 로그] 현재 LR 출력
                current_lr = ctx.optimizer.param_groups[0]['lr']
                round_num = getattr(ctx, 'current_round_num', getattr(ctx, 'cur_round_i', 'N/A'))
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

        # ✅ 토크나이저 길이와 vocab 일부 로깅 + 추가 토큰 확인
        try:
            tok = getattr(self, "tokenizer", None) or getattr(self.ctx, "tokenizer", None)
            mdl = self._unwrap(getattr(ctx, "model", None))

            if tok is not None and mdl is not None:
                tok_len = len(tok)
                emb_len = mdl.get_input_embeddings().weight.shape[0]

                # 현재 라운드 번호
                rnd = getattr(ctx, "current_round_num", getattr(ctx, "cur_round_i", "?"))
                # vocab dict
                vocab_dict = tok.get_vocab()
                vocab_size = len(vocab_dict)

                # Hugging Face tokenizer는 special_tokens_map 속성에 현재 등록된 special tokens 보유
                special_tokens = getattr(tok, "special_tokens_map", {})

                logger.info(
                    f"[ROUND{rnd}] tokenizer_len={tok_len} "
                    f"(vocab size={vocab_size}) | emb_rows={emb_len} | "
                    f"special_tokens={special_tokens}"
                )

                # vocab 앞뒤 일부만 출력
                vocab_items = list(vocab_dict.keys())
                logger.debug(f"[ROUND{rnd}] vocab_head={vocab_items[:5]} vocab_tail={vocab_items[-5:]}")

                # mismatch 체크
                if tok_len != emb_len:
                    logger.warning(f"[ROUND{rnd}] ⚠️ tokenizer/embedding mismatch!")

                # 추가된 토큰 개수와 어떤 토큰인지 확인
                added_count = tok_len - vocab_size
                if added_count > 0:
                    added_list = []
                    for k, v in special_tokens.items():
                        if v not in vocab_dict:
                            continue
                        added_list.append((k, v))
                    logger.info(f"[ROUND{rnd}] added_special={added_count}, tokens={added_list}")

        except Exception as e:
            logger.warning(f"[Tokenizer log skipped] {e}")



 
    #local process에서만 일어나는 일. 각 rank가 자기 shard된 미니배치를 forward 하고 loss를 계산하는 것.
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

        if not hasattr(self, "_logged_first_fwd_round") or self._logged_first_fwd_round != getattr(self.ctx, "current_round_num", None):
            try:
                tok = getattr(self, "tokenizer", None) or getattr(self.ctx, "tokenizer", None)
                mdl = getattr(self.ctx, "model", None)
                if tok is not None and mdl is not None:
                    log_tok_model_sync(tok, self._unwrap(mdl), tag=f"before-first-forward@round{getattr(self.ctx,'current_round_num','?')}")
            except Exception:
                pass
            self._logged_first_fwd_round = getattr(self.ctx, "current_round_num", None)






        #LLM 학습용 Dataset은 보통 tokenized_dict를 반환->{'input_ids': Tensor, 'labels': Tensor, 'attention_mask': Tensor}
        #accelerator는 입력 텐서를 자동으로 디바이스에 올려줘서 .to(ctx.device)가 필요 없음.


        #input_ids: [101,  42,  53,  78,   2]
        #labels:    [101,  42,  53,  78,   2]  # 혹은 일부 -100 포함
        #이렇게 거의 같지만, loss 계산에서 무시할 토큰은 labels에서만 바뀜.
        
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.accelerator.use: #참고


            #모델 호출이 model(x) → model(input_ids=..., labels=...)로 바뀜→ loss를 직접 criterion으로 계산하지 않고, 모델 자체가 계산해서 outputs.loss로 줌. loss는 cross-entropy로 적용하게 hugging face가 설계. forward를 override하면 loss 바꾸는 것 가능.
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask) # 자동으로 모델과 배치 텐서 모두 디바이스로 이동

        elif ctx.cfg.llm.deepspeed.use: #ctx.model_engine 기반으로 outputs 뽑아낸다.

            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:#참고, #ctx.model 기반으로 outputs 뽑아낸다.

            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits #일반적으로 LLM에서는 [batch_size, seq_len, vocab_size] shape의 tensor
        loss = outputs.loss #일부 모델에서 reduction='none'이면 shape가 [batch_size] 혹은 [batch_size, seq_len]일 수도 있지만, 대부분은 1개의 스칼라 값

        # NaN/Inf 모두 방어
        if not torch.isfinite(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning(f"Skip batch: non-finite loss={loss.item()}")
            return
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # #NaN loss가 발생시키는 batch 건너뛰기 위한 방어 로직
        # if torch.isnan(loss): #LM 학습에서는 종종 NaN loss가 발생할 수 있음-> label이 모두 -100 (ignore index). precision 문제 (e.g., bf16/float16). exploding gradients, bad initialization
        #     ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH) #다른 hook에서 이 값이 True면 이 배치를 건너뜀. (예: loss.backward() 스킵)
        #     logger.warning('Skip the batch due to the loss is NaN, '
        #                    'it may be caused by exceeding the precision or '
        #                    'invalid labels.')
        #     return
        # else:
        #     ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH) #shape: [batch_size, seq_len]
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH) # shape: [batch_size, seq_len, vocab_size]
 
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)


    # grad accumulation loop는 rank별로 local에서만 동작.

    # 하지만 backward마다 DDP hook이 자동으로 all-reduce 실행.

    # 결국 grad_accum_step은 “optimizer.step() 호출 시점만 늦추는 장치”임.

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
            # backward마다 DDP hook이 자동으로 각 process의 결과 바탕으로 all-reduce 실행. Accelerate가 grad accumulation을 관리하므로 손실을 나누지 않습니다.
            self.accelerator.backward(ctx.loss_task) 
            # ✅ 누적 경계(sync_gradients=True)에서만 step/zero_grad 실행
            #(i % grad_accum_step != 0)에서는 sync_gradients=False. 즉, backward는 실행하지만 DDP all-reduce 통신은 발생하지 않음 → 로컬에서만 grad 누적.
            # accumulation 끝 (i % grad_accum_step == 0)에서는 sync_gradients=True. 이때만 optimizer.step()이 실행되고, all-reduce가 발생해서 4개 프로세스의 grad 평균이 맞춰짐.


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

        # move the training data to cpu (안전하게 키 확인)
        for k in ('input_ids', 'labels', 'attention_mask'):
            if k in ctx.data_batch:
                t = ctx.data_batch[k]
                ctx.data_batch[k] = t.detach().to('cpu', non_blocking=True)




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

        # A) 모든 랭크가 학습 루프를 완전히 빠져나왔음을 보장
        barrier_all()

        # B) prepare된 객체부터 제거 (self.model 원본은 유지)
        to_del = [
            'optimizer', 'scheduler', 
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter',
            'loss_batch','loss_task','loss_regular',
            'loss_batch_total', 'loss_regular_total', 
            'y_true', 'y_prob', 'ys_true', 'ys_pred',
            'data_batch', 'grad', 'model_engine', 'skip_this_batch',
            # 데이터로더와 이터레이터를 모두 삭제하여 메모리 누수를 방지

        ]

        deleted_attrs = []
        for k in to_del:
            if hasattr(ctx, k):
                delattr(ctx, k); deleted_attrs.append(k)
        if getattr(ctx, 'rank', 0) == 0:
            logger.info(f"[Memory Cleanup] Deleted ctx attrs: {deleted_attrs}")        

        
        try:
            tok = getattr(self, "tokenizer", None) or getattr(self.ctx, "tokenizer", None)
            mdl = getattr(self, "model", None) or getattr(self.ctx, "model", None)
            if tok is not None and mdl is not None:
                log_tok_model_sync(tok, self._unwrap(mdl), tag="before-accel-free_memory")
        except Exception:
            pass

        # ✅ Accelerator는 삭제하지 않음, free_memory만 호출
        if self.accelerator is not None:
            try:
                torch.cuda.synchronize()
                self.accelerator.free_memory()
                torch.cuda.synchronize()
            except Exception:
                pass

            logger.info("Accelerator memory has been freed (object preserved).")

            # 삭제 직후  ← 태그 주의: after-accel-delete
            try:
                tok = getattr(self, "tokenizer", None) or getattr(self.ctx, "tokenizer", None)
                mdl = getattr(self, "model", None) or getattr(self.ctx, "model", None)
                if tok is not None and mdl is not None:
                    log_tok_model_sync(tok, self._unwrap(mdl), tag="after-accel-free_memory")
            except Exception:
                pass

 


        # (선택) 원본 grad 버퍼 비우기
        if hasattr(self, 'model'):
            for p in self.model.parameters():
                p.grad = None
        

        gc.collect()
        torch.cuda.empty_cache()

        # D) 정리 완료 동기화
        barrier_all()

        try:
            if torch.cuda.is_available():
                rnd = getattr(self.ctx, "current_round_num",
                    getattr(self.ctx, "cur_round_i", "?"))
                res = torch.cuda.memory_reserved() // (1024**2)
                aloc = torch.cuda.memory_allocated() // (1024**2)
                logger.info(f"[VRAM] round={rnd} reserved={res}MB allocated={aloc}MB")
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass




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
