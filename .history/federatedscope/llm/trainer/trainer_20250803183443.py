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


class LLMTrainer(GeneralTorchTrainer):
    """
    LLM 학습을 위한 Trainer (Accelerate / Deepspeed 옵션 지원)
    """

    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        num_train_batch = len(data['train'])  # micro-batch 기준
        # Gradient Accumulation Step 계산
        self.grad_accum_step = min(
            num_train_batch,
            max(config.llm.grad_accum_step, config.grad.grad_accum_count)
        )

        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')

        # tokenizer 준비
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len)
        self.eval_metrics = config.eval.metrics

    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        # memory 절약용 hook
        self.register_hook_in_train(self._hook_on_fit_end_free_space,
                                    "on_fit_end")

    def register_default_hooks_ft(self):
        super().register_default_hooks_ft()
        # memory 절약용 hook
        self.register_hook_in_ft(self._hook_on_fit_end_free_space,
                                 "on_fit_end")

    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        # memory 절약용 hook
        self.register_hook_in_eval(self._hook_on_fit_end_free_space,
                                   "on_fit_end")

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step=-1):
        # 1) grad_accum_step
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            grad_accum_step = self.grad_accum_step
        else:
            grad_accum_step = 1  # eval은 누적 없음

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return

        # 🔁 train: 분산 샘플러는 에폭마다 시드 고정
        if split == "train":
            sampler = getattr(loader, "sampler", None)
            try:
                from torch.utils.data.distributed import DistributedSampler
                if isinstance(sampler, DistributedSampler):
                    epoch_seed = int(getattr(self.ctx, "cur_epoch_i", 0))
                    sampler.set_epoch(epoch_seed)
            except Exception:
                pass

        iter_name = f"{split}_loader_iter"

        # ---- 반복 횟수 결정 ----
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            if run_step == -1:
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    # ✅ Accelerate 경로: 항상 마이크로배치 수(= 한 바퀴)만큼만 수행
                    run_step = len(loader)
                else:
                    # 수동 누적 경로: 외부 루프는 '업데이트 스텝' 단위로 수행
                    num_batches = len(loader)
                    run_step = math.ceil(num_batches / max(1, grad_accum_step))

            # 이터레이터 준비(필요시 신규 생성)
            if not hasattr(self.ctx, iter_name):
                setattr(self.ctx, iter_name, iter(loader))
            data_loader_iter = getattr(self.ctx, iter_name)

        else:
            # ✅ EVAL: 샤딩된 로더 길이만큼만 1회
            if run_step == -1:
                run_step = len(loader)
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

        exhausted = False
        for update_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(update_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator'):
                # ✅ grad_accum_step 루프 제거: accumulate 블록만 사용
                try:
                    self.ctx.data_batch = next(data_loader_iter)
                except StopIteration:
                    # ⛔️ 재시작하지 않음: 정확히 한 바퀴만
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
                # 수동 누적: 하나의 업데이트 스텝(update_i) 안에서 grad_accum_step개의 마이크로배치 처리
                for k in range(grad_accum_step):
                    self.ctx.cur_batch_i = CtxVar(update_i * grad_accum_step + k,
                                                  LIFECYCLE.BATCH)
                    try:
                        self.ctx.data_batch = next(data_loader_iter)
                    except StopIteration:
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

    # --- Fit 시작 시 초기화 (Accelerate/DeepSpeed/기본) ---
    def _hook_on_fit_start_init(self, ctx):
        """
        세 가지 경우:
          1) Accelerate 사용: model/optimizer만 prepare (dataloader는 감싸지 않음)
          2) Deepspeed 사용
          3) 기본 PyTorch
        + 메모리 정리 훅에서 로더/이터레이터를 지우므로, 여기서 안전하게 로더 복원
        """

        # --- 로더 복원: self.ctx.data 에 따라 두 케이스 지원 ---
        for split in ["train", "val", "test"]:
            loader_key = f"{split}_loader"
            if getattr(ctx, loader_key, None) is None:
                if isinstance(self.ctx.data, dict) and split in self.ctx.data:
                    # dict에 DataLoader가 직접 들어온 경우
                    setattr(ctx, loader_key, self.ctx.data[split])
                else:
                    # raw dataset 에서 새로 로더를 생성
                    raw = ctx.get(f"{split}_data", None)
                    if raw is None and hasattr(self.ctx.data, f"{split}_data"):
                        raw = getattr(self.ctx.data, f"{split}_data")
                    if raw is not None:
                        dl = get_dataloader(WrapDataset(raw), self.cfg, split)
                        setattr(ctx, loader_key, ReIterator(dl))

        # 이전 라운드 잔여 이터레이터 제거
        for it_name in ["train_loader_iter", "val_loader_iter", "test_loader_iter"]:
            if hasattr(ctx, it_name):
                delattr(ctx, it_name)

        # --- Accelerate ---
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

            # Optimizer / Scheduler
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                cfg_optimizer = ctx.cfg[ctx.cur_mode].optimizer
                if cfg_optimizer.type == 'AdamW':
                    adamw_kwargs = {'lr': cfg_optimizer.lr,
                                    'betas': cfg_optimizer.betas}
                    ctx.optimizer = AdamW(unwrapped_model.parameters(),
                                          **adamw_kwargs)
                else:
                    ctx.optimizer = get_optimizer(unwrapped_model, **cfg_optimizer)
                ctx.scheduler = get_scheduler(ctx.optimizer,
                                              **ctx.cfg[ctx.cur_mode].scheduler)
                current_lr = ctx.optimizer.param_groups[0]['lr']
                round_num = getattr(ctx, 'cur_round_i', 'N/A')
                logger.info(f"Round #{round_num} - Initializing with LR: {current_lr}")

                # ⛔️ DataLoader는 prepare에 넘기지 않음
                ctx.model, ctx.optimizer = self.accelerator.prepare(
                    unwrapped_model, ctx.optimizer
                )
            else:
                # 평가 모드: 모델만 prepare
                ctx.model = self.accelerator.prepare(unwrapped_model)

        elif ctx.cfg.llm.deepspeed.use:
            # --- Deepspeed ---
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            ctx.device = ctx.model_engine.local_rank

        else:
            # --- 기본 PyTorch ---
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                if not hasattr(self, 'scheduler') or self.scheduler is None:
                    self.scheduler = get_scheduler(ctx.optimizer,
                                                   **ctx.cfg[ctx.cur_mode].scheduler)
                ctx.scheduler = self.scheduler
                current_lr = ctx.optimizer.param_groups[0]['lr']
                round_num = getattr(ctx, 'cur_round_i', 'N/A')
                logger.info(f"Round #{round_num} - Current Learning Rate: {current_lr}")

        # 통계 변수 초기화
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        if hasattr(ctx, 'ys_pred'):
            ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.model.train()
        else:
            ctx.model.eval()

    def _hook_on_epoch_start(self, ctx):
        pass

    def _hook_on_batch_forward(self, ctx):
        """
        LLM 학습용: ctx.data_batch(dict)에서 input_ids/labels/attention_mask 사용
        """
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        # NaN loss 방어
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to NaN loss.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        # ✅ backward에서 사용할 표준 키도 명시
        ctx.loss_task = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        """
        Accelerate / Deepspeed / 기본 경로별 backward/step/zero_grad 처리
        """
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.accelerator.use:
            # Accelerate는 grad accumulation 경계를 내부 관리
            self.accelerator.backward(ctx.loss_task)
            if getattr(self.accelerator, "sync_gradients", True):
                # 경계에서만 gradient clipping/step/zero_grad
                if getattr(ctx, "grad_clip", 0) and ctx.grad_clip > 0:
                    try:
                        self.accelerator.clip_grad_norm_(ctx.model.parameters(),
                                                         ctx.grad_clip)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                       ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()
            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if getattr(ctx, "grad_clip", 0) and ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                   ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        # move the training data to cpu (안전하게 키 확인)
        if isinstance(ctx.data_batch, dict):
            if 'input_ids' in ctx.data_batch:
                ctx.data_batch['input_ids'].cpu()
            if 'labels' in ctx.data_batch:
                ctx.data_batch['labels'].cpu()
            if 'attention_mask' in ctx.data_batch:
                ctx.data_batch['attention_mask'].cpu()

    def _hook_on_batch_end(self, ctx):
        # NaN loss인 배치는 스킵 / 필요시 재시도
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        return

    # --- 메모리 정리 ---
    def _hook_on_fit_end_free_space(self, ctx):
        attributes_to_delete = [
            'optimizer', 'scheduler', 'loss_batch', 'loss_task', 'loss_regular',
            'loss_batch_total', 'loss_regular_total', 'y_true', 'y_prob',
            'ys_true', 'ys_pred',
            'data_batch', 'grad', 'model_engine', 'skip_this_batch',
            # 데이터로더와 이터레이터를 모두 삭제하여 메모리 누수를 방지
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter'
        ]

        if hasattr(ctx, 'rank') and ctx.rank == 0:
            deleted_attrs = [attr for attr in attributes_to_delete if hasattr(ctx, attr)]
            logger.info(f"[Memory Cleanup] Deleting attributes: {deleted_attrs}")

        for attr in attributes_to_delete:
            if hasattr(ctx, attr):
                delattr(ctx, attr)

        if hasattr(self, 'accelerator'):
            del self.accelerator
            logger.info("Accelerator object has been deleted.")

        gc.collect()
        torch.cuda.empty_cache()

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
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
                logger.warning("Set `cfg.eval.count_flops=False` to avoid OOM.")
                logger.error(e)
                logger.warning(
                    "current flop count implementation assumes "
                    "data_batch has [input_ids, labels, attn_mask]")
                ctx.monitor.flops_per_sample = -1

        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
