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


import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
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



import os, json



logger = logging.getLogger(__name__)

import sys

sys.setrecursionlimit(100000)


# 모든 랭크 동기화용 배리어
from federatedscope.llm.utils_dist import barrier_all



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

        self.local_only = bool(getattr(config.eval, "local_only", False))

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

        # __init__ 끝부분(Accelerator 생성 이후)
        self._prepared_once = False

        # SDPA 커널 선택: 플래시/메모리효율 off, 수학 커널 on
        try:
            import torch
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass


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

        self._ct_ft = int(self.cfg.federate.total_round_num) == 1 # CT-FT 모드(라운드=1)
        self._mid_eval_every = int(getattr(self.cfg.eval, "every_n_train_steps", -1)) if self._ct_ft else -1
        self._global_updates = 0

        # ⬇️ Early-Stop 상태값 (CT-FT일 때만 유효)
        self._es_enabled   = self._ct_ft and bool(getattr(self.cfg.eval, "early_stop_on_test_acc", True))
        self._es_patience  = int(getattr(self.cfg.eval, "early_stop_patience", 10))
        self._es_min_delta = float(getattr(self.cfg.eval, "early_stop_min_delta", 0.0))
        self._es_best      = float("-inf")
        self._es_wait      = 0
        self._es_triggered = False


        self._mid_eval_pending = False   
        self._mid_eval_running = False

        self._epoch_i_cache_for_train_loop = 0
        self._num_epoch_cache_for_train_loop = 1
        self._n_last_cache = None

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
                        persistent_workers=False, 
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
                        persistent_workers=False, 
                    )

                sharded = True
                local_count = len(sampler) #rank 별로 처리할 샘플(인덱스)의 개수
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

            try:
                tok = getattr(self, "tokenizer", None) or getattr(self.ctx, "tokenizer", None)
                mdl = getattr(self, "model", None) or getattr(self.ctx, "model", None)
                if tok is not None and mdl is not None:
                    log_tok_model_sync(tok, mdl, tag=f"after-reset-{split_name}")
            except Exception:
                pass
            
            barrier_all()
            if torch.cuda.is_available():                      
                torch.cuda.synchronize()





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
        if m is None:
            return None
        if getattr(self, "accelerator", None) is not None:
            try:
                m = self.accelerator.unwrap_model(m)
            except Exception:
                pass
        while hasattr(m, "module"):
            m = m.module
        return m



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




    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None, round_num=-1):

        logger.info(f"[mid-eval] every_n_train_steps={self._mid_eval_every}")

        self._set_round_ctx(round_num)  # ← 여기!
        # [수정] current_round를 맨 먼저 정의하여 NameError 해결
        current_round = round_num

        # ★ CT-FT & baseline 옵션이면, 파인튜닝 시작 전에 1회 평가
        if self._ct_ft and self._mid_eval_every > 0 and bool(getattr(self.cfg.eval, "baseline_before_ft", True)):
            self._mid_eval_once()




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
        try:
            # 여기까지 준비 끝 → 실제 학습/평가는 super().train 안에서 모두 진행
            return super().train(target_data_split_name, hooks_set)
        finally:
            # ✅ 이 호출에서 필요한 일 전부 끝난 이후에만 정리
            for k in ('train_loader', 'train_loader_iter', 'optimizer', 'scheduler', 'grad_scaler', 'scaler'):
                if hasattr(self.ctx, k):
                    delattr(self.ctx, k)

    def evaluate(self, target_data_split_name="test", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_eval
        self._reset_and_build_dataloader(target_data_split_name)
        with torch.no_grad():
            if self.ctx.check_split(target_data_split_name, skip=True):
                self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
            else:
                self.ctx.eval_metrics = dict()
        return self.ctx.eval_metrics


    @lifecycle(LIFECYCLE.EPOCH) #한 에폭(epoch) 단위로 반복 실행. 끝나면 에폭용 임시변수(CtxVar(..., "epoch")) 일괄 삭제
    def _run_epoch(self, hooks_set, run_step=-1):


        if run_step == -1:
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_epoch")#총 epoch 수. batch든 epoch 모드이든 total data 루프 몇 번 도는지 계산됨. test/val일때는 1.
        for epoch_i in range(run_step):
            self.ctx.cur_epoch_i = CtxVar(epoch_i, "epoch")

            for hook in hooks_set["on_epoch_start"]:
                hook(self.ctx)

            self._run_batch(hooks_set)#llm trainer로 override. run_step=-1로 된다는 것 유의. 마지막 epoch의 부족한 batch update도 고려.

            # ★ [ADD] 배치 루프에서 트리거되면 에폭도 즉시 종료
            if getattr(self, "_es_triggered", False):
                break


            for hook in hooks_set["on_epoch_end"]:
                hook(self.ctx)




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
                num_batches = getattr(self.ctx, f"num_{split}_batch") ##sharding 고려하지 않고 일반적인 epoch당 micro batch 기준 총  batch 개수. 다만 batch_or_epoch == "batch"이면 iteration 수와 비교했을 때 min 값으로 설정됨.

                logger.info(f"[run-batch-setup] split={split}, "
                            f"len(loader)={len(loader)}, num_batches(ctx)={num_batches}, "
                            f"grad_accum_step={grad_accum_step}, "
                            f"will_run_step(loops)={num_batches * (grad_accum_step if (self.accelerator is not None) else 1)}")




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
            # ✅ 매 배치마다 기본값 세팅
            self.ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            e = getattr(self.ctx, "cur_epoch_i", None)
            if e is not None:
                self._epoch_i_cache_for_train_loop = int(e)
            ne = getattr(self.ctx, "num_train_epoch", None)
            if ne is not None:
                self._num_epoch_cache_for_train_loop = int(ne)
            nlast = getattr(self.ctx, "num_train_batch_last_epoch", None)
            if nlast is not None:
                self._n_last_cache = int(nlast)
            elif self._n_last_cache is None:
                # 폴백: split별 배치 수
                self._n_last_cache = int(getattr(self.ctx, f"num_{split}_batch", 0))

            if hasattr(self, 'accelerator') and self.accelerator is not None: #accelerator 모드에서 실행. 
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

                # ← 여기! 한 배치의 라이프사이클이 끝난 안전 지점
                if self.ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE) and self._mid_eval_pending:
                    self._mid_eval_pending = False
                    self._mid_eval_once()

                # ★ [ADD] 조기종료이면 루프 탈출
                if self.ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE) and getattr(self, "_es_triggered", False):
                    exhausted = True
                    break
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
                # ← 여기! 누적이 끝나 optimizer.step()가 발생했을 때만 플래그가 켜짐
                if self.ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE) and self._mid_eval_pending:
                    self._mid_eval_pending = False
                    self._mid_eval_once()
                # ★ [ADD] 조기종료이면 루프 탈출
                if self.ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE) and getattr(self, "_es_triggered", False):
                    exhausted = True
                    break

            if exhausted:
                break

            # if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1: 
            #     logger.info(f"[break-check] update_i={update_i}, "
            #                 f"num_train_batch_last_epoch={self.ctx.num_train_batch_last_epoch}")

            #     if update_i >= self.ctx.num_train_batch_last_epoch - 1: 
            #         break    

            if self.ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE): # 학습/파인튜닝 모드 확인 및 마지막 에폭인지 확인
                cur_epoch_i = int(getattr(self.ctx, "cur_epoch_i",
                                        self._epoch_i_cache_for_train_loop))
                num_train_epoch = int(getattr(self.ctx, "num_train_epoch",
                                            self._num_epoch_cache_for_train_loop))
                n_last = int(getattr(self.ctx, "num_train_batch_last_epoch",
                                    self._n_last_cache if self._n_last_cache is not None
                                    else getattr(self.ctx, f"num_{split}_batch", 0)))

                if cur_epoch_i == max(0, num_train_epoch - 1): #마지막 에폭의 실제 스텝(업데이트) 수 초과 여부 판정
                    if update_i >= max(0, n_last - 1):
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
            barrier_all()
            torch.cuda.synchronize()
        
        if ctx.cfg.llm.accelerator.use:
            # ✅ Accelerator는 이미 __init__에서 1회 생성됨
            ctx.device = self.accelerator.device
            base = self._unwrap(self.model)

            if not self._prepared_once:
                # 최초 1회만 모델 prepare
                ctx.model = self.accelerator.prepare(base)
                self.model = ctx.model
                self._prepared_once = True


            else:
                # 준비된 모델 재사용 (re-prepare 금지)
                ctx.model = self.model

            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(self._unwrap(ctx.model), **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = None
        else:
            ctx.model = self.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = None

        # ✅ 검증만 유지(추가/리사이즈는 없음)
        try:
            tok = getattr(self, "tokenizer", None)
            base = self._unwrap(ctx.model)
            if tok is not None and base is not None:
                tlen = len(tok)
                elen = base.get_input_embeddings().weight.size(0)
                if tlen != elen:
                    raise RuntimeError(f"[Guard] tokenizer/embedding mismatch: {tlen} vs {elen}")
        except Exception as e:
            logger.warning(f"[Tokenizer guard] {e}")


        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.model.train()
        else:  # MODE.TEST or MODE.VAL
            ctx.model.eval()


        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

        ctx.sample_seen = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.sample_correct_accum = CtxVar(0, LIFECYCLE.ROUTINE)



        if hasattr(ctx, 'ys_pred'): 
            ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)


        if hasattr(ctx, "new_lr_to_be_set"):
            new_lr = ctx.new_lr_to_be_set
            if hasattr(ctx, 'optimizer') and ctx.optimizer is not None:
                opt = ctx.optimizer
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"Successfully applied new LR {new_lr:.2e} to the optimizer.")
                del ctx.new_lr_to_be_set
     
        current_round = int(getattr(ctx, "current_round_num", getattr(ctx, "cur_state", 0)))
 
        # train/val/test 성능 측정용 카운터 리셋
        for sp in ["train", "val", "test"]:
            setattr(ctx, f"num_samples_{sp}", 0)
            setattr(ctx, f"loss_total_{sp}", 0.0)
            setattr(ctx, f"correct_{sp}", 0)

        self.choices = self.choices_cpu.to(ctx.device, dtype=torch.long, non_blocking=True) #non_blocking=True는 pinned memory 환경에서 async copy 허용 → 성능 최적화.


        try:
            # GeneralTorchTrainer가 계산해 둔 컨텍스트 값 확인
            boe = getattr(self.ctx, "batch_or_epoch", None)
            it  = getattr(self.ctx, "train_iter", None)
            nb  = getattr(self.ctx, "num_train_batch", None)
            nb_last = getattr(self.ctx, "num_train_batch_last_epoch", None)
            ne  = getattr(self.ctx, "num_train_epoch", None)
            logger.info(f"[fit-start] batch_or_epoch={boe}, train_iter={it}, "
                        f"num_train_batch={nb}, num_train_batch_last_epoch={nb_last}, "
                        f"num_train_epoch={ne}, grad_accum_step={self.grad_accum_step}, "
                        f"accum_in_accel={getattr(self.accelerator,'gradient_accumulation_steps',None)}")
            

            
        except Exception as e:
            logger.warning(f"[fit-start] debug read failed: {e}")


        # === [PATCH] batch 모드에서만 스텝 기반 스케줄을 강제 ===
        try:
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE] \
            and getattr(ctx.cfg.train, "batch_or_epoch", "batch") == "batch":
                lus = int(ctx.cfg.train.local_update_steps)  # 예: 800
                # 의미상 한 에폭으로 고정
                ctx.num_train_epoch = 1
                # "num_batches"를 곧바로 원하는 optimizer step 개수로 지정
                ctx.num_train_batch = lus
                ctx.num_train_batch_last_epoch = lus
                logger.info(f"[force-step-schedule] epoch=1, num_batches={lus}, "
                            f"grad_accum_step={self.grad_accum_step} "
                            f"(=> total micro-batches = {lus * max(1, self.grad_accum_step)})")
        except Exception as e:
            logger.warning(f"[force-step-schedule] skip due to: {e}")



 
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
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

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

        logits = outputs.logits #일반적으로 LLM에서는 [batch_size, seq_len, vocab_size] :=[B,T,V] shape의 tensor
        loss = outputs.loss #일부 모델에서 reduction='none'이면 shape가 [batch_size] 혹은 [batch_size, seq_len]일 수도 있지만, 대부분은 1개의 스칼라 값

        # NaN/Inf 모두 방어
        if not torch.isfinite(loss): #LM 학습에서는 종종 NaN loss가 발생할 수 있음-> label이 모두 -100 (ignore index). precision 문제 (e.g., bf16/float16). exploding gradients, bad initialization
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH) #다른 hook에서 이 값이 True면 이 배치를 건너뜀. (예: loss.backward() 스킵)
            logger.warning(f"Skip batch: non-finite loss={loss.item()}")
            return
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

 

        # 4) 일반 LM 메트릭용 저장
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH) #shape: [batch_size, seq_len]
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH) # shape: [batch_size, seq_len, vocab_size]
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)


    
        # 5) A/B 선택 정확도(첫 유효 토큰) 추가 집계
        #    - self.choices: [C] (예: 2) 형태의 토큰 ID 텐서 (디바이스에 올라와 있어야 함)
        #    - labels에서 ' A' 또는 ' B'가 들어간 첫 위치를 찾고, 그 위치의 로짓에서 두 클래스만 비교
        try:
            choice_ids = self.choices  # shape [C], 예: tensor([id_A, id_B], device=...)
            # labels가 choice 중 하나인 위치 마스크: [B, T]
            """
            labels[..., None]          # == labels.unsqueeze(-1) → [B, T, 1]
            choice_ids[None, None, :]  # → [1, 1, C]
            (labels[..., None] == choice_ids[None, None, :])# → [B, T, C]
            """
            is_choice = (labels[..., None] == choice_ids[None, None, :]).any(dim=-1)# → [B, T]
            has_choice = is_choice.any(dim=1)  # [B]

            if has_choice.any():
                B, T, V = logits.shape
                device = labels.device

                #샘플별 “첫 번째” choice 위치 찾기
                """
                is_choice.int()는 True/False → 1/0.
                argmax(dim=1)는 각 샘플(b)에 대해 가장 먼저 1이 나오는 t를 돌려줌 → “첫 choice 위치”.
                """
                first_idx = torch.argmax(is_choice.int(), dim=1)                  # [B]

                """
                has_choice==False인 샘플은 제외하기 위해, 선택 마스크 has_choice로 인덱싱:
                    sel_b: 실제로 choice가 있는 샘플들의 배치 인덱스들 (M개)

                    sel_t: 그 샘플 각각에서의 “첫 choice 토큰”의 위치 (길이 M)
                """
                sel_b = torch.arange(B, device=device)[has_choice]                # [M]
                sel_t = first_idx[has_choice]                                     # [M]

                # 수정
                sel_t_pred = sel_t - 1                          # 예측용 로짓의 시간축 인덱스
                valid = sel_t_pred >= 0                         # 혹시 0인 케이스 방지
                sel_b      = sel_b[valid]
                sel_t      = sel_t[valid]
                sel_t_pred = sel_t_pred[valid]


                # ⬇️ 비어 있으면 안전 종료 (argmax가 빈 텐서에서 터질 수 있음)
                if sel_b.numel() == 0:
                    ctx.sample_correct_batch = 0
                    ctx.sample_count_batch   = 0
                    return



                # M개 샘플 각각에 대해 “첫 choice 위치 t=sel_t”에서의 어휘 전체(V) 로짓을 뽑음.
                logits_at   = logits[sel_b, sel_t_pred, :]   # [M, V]

                # 해당 위치의 정답 토큰 ID (id_A 또는 id_B)
                targets_tok = labels[sel_b, sel_t]                                # [M]


                # 선택지 두(여러) 클래스 로짓만 뽑기 → [M, C]
                logits_choice = logits_at.index_select(dim=1, index=choice_ids)



                # (targets_tok == choice_ids) 비교로 [M, C] bool 만들고, argmax로 위치(0 or 1)를 뽑음.
                #   id_A면 0, id_B면 1이 됨.
                target_idx = (targets_tok.unsqueeze(1) == choice_ids.unsqueeze(0)).long().argmax(dim=1)  # [M]

                #예측 클래스
                pred_idx = torch.argmax(logits_choice, dim=-1)                    # [M]

                #정확도 계산
                sample_correct = int((pred_idx == target_idx).sum().item()) #정답과 비교해 M개 중 맞춘 개수
                sample_count   = int(target_idx.numel())
            else:
                sample_correct, sample_count = 0, 0

            ctx.sample_correct_batch = sample_correct
            ctx.sample_count_batch   = sample_count
        except Exception as e:
            # 메트릭 실패해도 학습은 계속
            logger.warning(f"[choice-metrics] skipped due to error: {e}")
            ctx.sample_correct_batch = 0
            ctx.sample_count_batch   = 0





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


        if bool(getattr(ctx, "skip_this_batch", False)): #스킵 플래그(skip_this_batch)가 켜져 있으면 아무 작업도 하지 않고 바로 리턴
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

                # logger.info(f"[UPD] optimizer_step={self._global_updates+1} (sync_gradients=True), "
                #             f"num_train_batch_last_epoch={getattr(self.ctx,'num_train_batch_last_epoch',None)}")

                # ★ 여기서 '바로' 호출: pending/바깥체크 X
                self._global_updates += 1
                if self._ct_ft and self._mid_eval_every > 0 and (self._global_updates % self._mid_eval_every) == 0:
                    logger.info(f"[MID-EVAL-TRIGGER] (inline) at optimizer_step={self._global_updates}")
                    self._mid_eval_pending = True
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

                self._global_updates += 1
                if self._ct_ft and self._mid_eval_every > 0 and (self._global_updates % self._mid_eval_every) == 0:
                    self._mid_eval_pending = True


        # move the training data to cpu (안전하게 키 확인)
        db = getattr(ctx, "data_batch", None)
        if isinstance(db, dict):
            for k in ('input_ids', 'labels', 'attention_mask'):
                t = db.get(k, None)
                if torch.is_tensor(t):
                    db[k] = t.detach().to('cpu', non_blocking=True)


        # for k in ('input_ids', 'labels', 'attention_mask'):
        #     if k in ctx.data_batch:
        #         t = ctx.data_batch[k]
        #         ctx.data_batch[k] = t.detach().to('cpu', non_blocking=True)




    def _hook_on_batch_end(self, ctx): #👉 한 배치(batch)를 끝낼 때 호출되는 후처리 hook
        #1) 스킵 처리
        """
        기본 로직 + loss == NaN일 경우 batch를 retry

        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                self._run_batch(...)
        """
        if bool(getattr(ctx, "skip_this_batch", False)):
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        #2) (옵션) 배치별 LR 디버깅. 스케쥴러 적용 확인용.
        if ctx.cur_mode == MODE.TRAIN:
            opt = getattr(ctx, "optimizer", None)
            if opt is not None:
                current_lr = opt.param_groups[0]["lr"]
                round_idx = getattr(ctx, "current_round_num", "?")
                batch_idx = getattr(ctx, "batch_idx", "?")
                # logger.info(f"[LR BATCH CHECK] round={round_idx} batch={batch_idx} -> LR={current_lr:.2e}")

        
        #3) 통계 집계시 키 초기화
        # 누적 집계 키
        split = ctx.cur_split

        loss_total_key = f"loss_total_{split}"
        num_samples_key = f"num_samples_{split}"


        if not hasattr(ctx, loss_total_key):
            setattr(ctx, loss_total_key, 0.0)

        if not hasattr(ctx, num_samples_key):
            setattr(ctx, num_samples_key, 0)

        #4) 배치 단위 통계
        batch_raw_samples = int(ctx.batch_size) #배치 내의 전체 샘플 수
        batch_logic_seen = int(getattr(ctx, "sample_count_batch", 0)) #배치 내의 유효한 샘플 수
        batch_logic_corr = int(getattr(ctx, "sample_correct_batch", 0)) #배치 내의 유효한 샘플들 중 맞춘 것들의 갯수

        #split 동안 누적 sample 수 맟 loss 총합
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


        #5) 정확도 누적,  논리적(선택된) 토큰 기준 정확도 집계
        ctx.sample_seen = int(getattr(ctx, "sample_seen", 0)) + batch_logic_seen
        ctx.sample_correct_accum = int(getattr(ctx, "sample_correct_accum", 0)) + batch_logic_corr

        #6) Loss 세부 집계
        ctx.num_samples = int(getattr(ctx, "num_samples", 0)) + batch_raw_samples
        ctx.loss_batch_total = float(getattr(ctx, "loss_batch_total", 0.0)) + ctx.loss_batch.item() * batch_raw_samples
        ctx.loss_regular_total = float(getattr(ctx, "loss_regular_total", 0.0)) + float(ctx.get("loss_regular", 0.0))

        #7) 검증/테스트: Prediction 정보 저장
        if ctx.cur_mode in [MODE.TEST, MODE.VAL]:
            if not hasattr(ctx, "ys_true"):
                ctx.ys_true = []
            if not hasattr(ctx, "ys_pred"):
                ctx.ys_pred = []

            pred = torch.argmax(ctx.y_prob, dim=-1)
            ctx.ys_true.append(ctx.y_true)
            ctx.ys_pred.append(pred)

        # if ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE) and getattr(self, "_mid_eval_pending", False):
        #     logger.info(f"[MID-EVAL-TRIGGER] at optimizer_step={self._global_updates}")
        #     self._mid_eval_pending = False
        #     self._mid_eval_once()

    def _hook_on_fit_end(self, ctx): 

        """
        라운드 종료 시점 정리/로그 훅.
        ✅ 집계(reduce)는 _run_routine에서 완료되었다고 가정.
        - split(train/val/test)별 최종 metric(ctx.eval_metrics)만 읽어 로그 출력
        - Accelerator/메모리 정리 등은 다른 훅에 맡기거나 여기서 가볍게 처리
        """        
        #1) split 유효성 체크
        split = getattr(ctx, "cur_split", None)
        if split not in ("train", "val", "test"):
            return

        # 2) accelerator 및 main 여부
        acc = getattr(self, "accelerator", None)
        is_main = (acc.is_main_process if acc is not None else True)

        # 3) 🔧 집계는 하지 않고, _run_routine에서 채운 최종 metric만 로드
        #    (train/eval을 같은 라운드에서 둘 다 돌려도, split별 키로 공존 가능)
        m = getattr(ctx, "eval_metrics", None)
        if not isinstance(m, dict) or len(m) == 0:
            # ⚠️ 혹시 상위가 아직 안 채웠거나 싱크 문제면 조용히 리턴
            return

        # 4) 로깅용 안전 추출 (키 없을 수 있으니 get 사용)
        total    = m.get(f"{split}_total", 0)
        loss_sum = m.get(f"{split}_loss", 0.0)
        avg_loss = m.get(f"{split}_avg_loss", 0.0)
        seen     = m.get(f"{split}_seen", 0)          # 🆕 seen/ correct도 로깅
        correct  = m.get(f"{split}_correct", 0)
        accuracy  = m.get(f"{split}_acc", 0.0) 

        #7) metric 저장 및 로그 출력 (메인 프로세스만)
        if is_main:
            logger.info(
                f"[{split}|final] total={total}, "
                f"loss_sum={float(loss_sum):.6f}, avg_loss={float(avg_loss):.6f}, "
                f"seen={int(seen)}, correct={int(correct)}, accuracy={float(accuracy):.6f}"
            )

    def _hook_on_fit_end_free_space(self, ctx):

        # A) 모든 랭크가 학습 루프를 완전히 빠져나왔음을 보장
        barrier_all()

        # B) prepare된 객체부터 제거 (self.model 원본은 유지)
        to_del = [
            'val_loader', 'test_loader',
            'val_loader_iter', 'test_loader_iter',
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
            tok = getattr(self, "tokenizer", None)
            mdl = getattr(self, "model", None) or getattr(self.ctx, "model", None)
            if tok is not None and mdl is not None:
                log_tok_model_sync(tok, self._unwrap(mdl), tag="before-accel-free_memory")
        except Exception:
            pass

        if getattr(self, "accelerator", None) is not None:
            import torch
            try:
                torch.cuda.synchronize()
                self.accelerator.free_memory()
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            logger.info("Accelerator memory has been freed (object preserved).")

 



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
        
    def _save_best_local_only(self):
        """LoRA 전용(학습가능 파라미터) 체크포인트: 클라이언트당 1개만 유지(덮어쓰기)."""
        if not bool(getattr(self.cfg.eval, "local_only", False)):
            return

        acc = getattr(self, "accelerator", None)
        is_main = (acc.is_main_process if acc is not None else True)
        if not is_main:
            return

        cid  = int(getattr(getattr(self, "ctx", object()), "client_ID", 0))

        save_dir = "./checkpoints_1.0_local_only"
        os.makedirs(save_dir, exist_ok=True)

        # ★ 파일명 고정(= step 제거). 덮어쓰기 1개만 유지됨.
        save_path = os.path.join(save_dir, f"local_only_tldr_choice_qwen_client_{cid:03d}.ckpt")

        base = self._unwrap(self.ctx.model)
        sd = base.state_dict()        # AdapterModel.state_dict(return_trainable=True) → LoRA 학습가능 파라미터만
        torch.save(sd, save_path)
        logger.info(f"[local-only] saved(best) -> {save_path}")




    def _mid_eval_once(self):
        """CT-FT에서 every_n_train_steps마다:
        (1) train 스냅샷 -> 전 클라이언트 공용 파일에 append
        (2) train 카운터 초기화
        (3) val/test 평가 -> 클라이언트별 파일에 append
        (4) val/test 카운터 초기화
        """
 
        # --- 재진입 가드 ---
        if getattr(self, "_mid_eval_running", False):
            return
        self._mid_eval_running = True

        # --- 호출 전 상태 저장 ---
        prev_mode = getattr(self.ctx, "cur_mode", None)
        prev_split = getattr(self.ctx, "cur_split", None)   # ★ 추가
        prev_training = None
        m = getattr(self.ctx, "model", None)
        if m is not None:
            try:
                prev_training = self._unwrap(m).training  # HF/Accelerate 래핑 고려
            except Exception:
                prev_training = m.training

        try:
            # ====== ⬇ 기존 본문 그대로 둠 (아래 줄부터 네 코드) ======
            using_accel = hasattr(self, 'accelerator') and self.accelerator is not None
            if using_accel:
                self.accelerator.wait_for_everyone()
            is_main = (not using_accel) or self.accelerator.is_main_process

            cid  = int(getattr(getattr(self, "ctx", object()), "client_ID", 0))
            rnd  = int(getattr(getattr(self, "ctx", object()), "round", 0))
            step = int(getattr(self, "_global_updates", 0))
            adapter_idx = getattr(getattr(self, "ctx", object()), "current_adapter_idx", None)

            outdir = getattr(self.cfg.eval, "outdir", "runs")
            os.makedirs(outdir, exist_ok=True)
            train_path = os.path.join(outdir, "train_results.raw")
            per_client_path = os.path.join(outdir, f"mid_eval/client_{cid:03d}.raw")
            os.makedirs(os.path.dirname(per_client_path), exist_ok=True)

            logger.info(f"[mid-eval] start: is_main={is_main}, step={step}, splits={['val','test']}")

            train_snap = self._train_snapshot_metrics()
            if not train_snap:
                train_snap = {
                    "train_total": 0, "train_loss": 0.0, "train_avg_loss": 0.0,
                    "train_seen": 0, "train_correct": 0, "train_acc": 0.0,
                }

            if is_main:
                rec = {
                    "client": cid, "round": rnd, "step": step,
                    "phase": "mid", "split": "train",
                    "adapter_idx": (int(adapter_idx) if adapter_idx is not None else None),
                    "total": train_snap.get("train_total", 0),
                    "loss": train_snap.get("train_loss", 0.0),
                    "avg_loss": train_snap.get("train_avg_loss", 0.0),
                    "seen": train_snap.get("train_seen", 0),
                    "correct": train_snap.get("train_correct", 0),
                    "acc": train_snap.get("train_acc", 0.0),
                }
                with open(train_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            self._reset_split_counters('train')
            if using_accel:
                self.accelerator.wait_for_everyone()

            for sp in ['val','test']:
                self._reset_split_counters(sp)
                if using_accel:
                    self.accelerator.wait_for_everyone()

                out = self.evaluate(target_data_split_name=sp)

                if is_main and isinstance(out, dict):
                    rec = {
                        "client": cid, "round": rnd, "step": step,
                        "phase": "mid", "split": sp,
                        "adapter_idx": (int(adapter_idx) if adapter_idx is not None else None),
                        "total":    out.get(f"{sp}_total", 0),
                        "loss":     out.get(f"{sp}_loss", 0.0),
                        "avg_loss": out.get(f"{sp}_avg_loss", 0.0),
                        "seen":     out.get(f"{sp}_seen", 0),
                        "correct":  out.get(f"{sp}_correct", 0),
                        "acc":      out.get(f"{sp}_acc", 0.0),
                    }
                    with open(per_client_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # ⬇️ ES는 test일 때(그리고 CT-FT일 때)만 수행
                if self._ct_ft and sp == 'test' and self._es_enabled and isinstance(out, dict):
                    # acc 키를 견고하게 가져오기 (우선순위: test_acc -> f'{sp}_acc' -> acc)
                    cur_acc = float(out['test_acc'])  # 여기 고정!

                    if cur_acc > (self._es_best + self._es_min_delta):
                        self._es_best = cur_acc
                        self._es_wait = 0
                        if is_main:
                            logger.info(f"[EarlyStop] new best test_acc={cur_acc:.6f}")
                        self._save_best_local_only()
                    else:
                        self._es_wait += 1
                        if is_main:
                            logger.info(f"[EarlyStop] no improvement (wait={self._es_wait}/{self._es_patience}), "
                                        f"best={self._es_best:.6f}, curr={cur_acc:.6f}")
                        if self._es_wait >= self._es_patience:
                            self._es_triggered = True
                            if is_main:
                                logger.info("[EarlyStop] patience reached -> request stop")

                # 마지막에 카운터 리셋
                self._reset_split_counters(sp)



            if using_accel:
                self.accelerator.wait_for_everyone()
            # ====== ⬆ 기존 본문 그대로 둠 ======

        finally:
            # --- 상태 복구 ---
            if prev_mode is not None:
                self.ctx.cur_mode = prev_mode
            if prev_split is not None:
                self.ctx.cur_split = prev_split            # ★ 추가

            m = getattr(self.ctx, "model", None)
            if m is not None:
                if prev_training:
                    m.train()
                else:
                    m.eval()
            self._mid_eval_running = False

    def _reset_split_counters(self, split: str):
        """해당 split의 누적 카운터를 0으로 초기화."""
        setattr(self.ctx, f'num_samples_{split}', 0)
        setattr(self.ctx, f'loss_total_{split}', 0.0)
        # 공용 집계(정확도용)
        self.ctx.sample_seen = 0
        self.ctx.sample_correct_accum = 0
        # (옵션) 로컬 스냅샷 캐시
        if hasattr(self.ctx, 'local_results_for_log'):
            self.ctx.local_results_for_log = {}

    def _train_snapshot_metrics(self):
        """현재까지의 train 누적값을 reduce해서 그대로 반환(윈도우/prev 없음)."""
        split = "train"
        num_samples = int(self.ctx.get(f'num_samples_{split}', 0))
        loss_total  = float(self.ctx.get(f'loss_total_{split}', 0.0))
        seen        = int(self.ctx.get('sample_seen', 0))
        correct     = int(self.ctx.get('sample_correct_accum', 0))

        using_accel = hasattr(self, 'accelerator') and self.accelerator is not None \
                    and getattr(self.accelerator, 'num_processes', 1) > 1

        if using_accel:
            import torch
            dev = self.accelerator.device
            total_all   = self.accelerator.reduce(torch.tensor([num_samples], device=dev, dtype=torch.long),    reduction='sum')[0].item()
            loss_all    = self.accelerator.reduce(torch.tensor([loss_total],  device=dev, dtype=torch.float32), reduction='sum')[0].item()
            seen_all    = self.accelerator.reduce(torch.tensor([seen],        device=dev, dtype=torch.long),    reduction='sum')[0].item()
            correct_all = self.accelerator.reduce(torch.tensor([correct],     device=dev, dtype=torch.long),    reduction='sum')[0].item()
        else:
            total_all, loss_all, seen_all, correct_all = num_samples, loss_total, seen, correct

        if total_all <= 0:
            return {}

        return {
            "train_total":   int(total_all),
            "train_loss":    float(loss_all),
            "train_avg_loss": float(loss_all / max(1, total_all)),
            "train_seen":    int(seen_all),
            "train_correct": int(correct_all),
            "train_acc":     float(correct_all / max(1, seen_all)),
        }



def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
