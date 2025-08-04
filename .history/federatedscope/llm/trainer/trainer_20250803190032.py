import torch
import logging
import gc
import math
import itertools  # ★ StopIteration 방지용 cycle

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None

import accelerate
from accelerate import Accelerator

from transformers import AdamW  # transformers.AdamW 사용 (경고는 무시 가능)

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
    LLM 학습용 Trainer
    - Accelerate: bf16/grad_accum/DDP 래핑
    - Deepspeed: (옵션) 현재는 사용 안함
    - 고정 iteration 모드: train.local_update_steps 를 '진짜 업데이트 스텝 수'로 간주
    """

    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        num_train_batch = len(data['train'])

        # grad accumulation step 결정
        self.grad_accum_step = min(
            num_train_batch,
            max(config.llm.grad_accum_step, config.grad.grad_accum_count)
        )

        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root, config.llm.tok_len)
        self.eval_metrics = config.eval.metrics

    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        # 메모리 정리 훅
        self.register_hook_in_train(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_ft(self):
        super().register_default_hooks_ft()
        self.register_hook_in_ft(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        self.register_hook_in_eval(self._hook_on_fit_end_free_space, "on_fit_end")

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
        cycle_name = f"{split}_cycle_iter"

        # ---- 반복 횟수 결정 ----
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # 고정 iteration 모드: train.local_update_steps = '업데이트 스텝' 수
            desired_updates = getattr(self.cfg.train, "local_update_steps", None)
            assert desired_updates and desired_updates > 0, \
                "train.local_update_steps (>0) 가 설정되어야 합니다. (예: 30)"

            if hasattr(self, 'accelerator') and self.accelerator is not None:
                # Accelerate 경로: 외부 루프 = '마이크로배치' 단위
                # 총 마이크로배치 수 = 업데이트 스텝 × grad_accum_step
                if run_step == -1:
                    run_step = desired_updates * max(1, grad_accum_step)

                # cycle 로더 준비(StopIteration 방지)
                if not hasattr(self, cycle_name) or getattr(self, cycle_name) is None:
                    setattr(self, cycle_name, itertools.cycle(loader))
                data_loader_iter = getattr(self, cycle_name)

            else:
                # 비-Accelerate 경로: 외부 루프 = '업데이트 스텝' 단위
                if run_step == -1:
                    run_step = desired_updates

                # 표준 이터레이터 준비 (내부에서 grad_accum_step 회전)
                if not hasattr(self.ctx, iter_name):
                    setattr(self.ctx, iter_name, iter(loader))
                data_loader_iter = getattr(self.ctx, iter_name)

            logger.info(f"[{split}] updates={desired_updates}, "
                        f"grad_accum={grad_accum_step}, run_step={run_step}, "
                        f"accelerate={hasattr(self, 'accelerator')}")

        else:
            # ✅ EVAL/TEST: 정확히 1 epoch
            if run_step == -1:
                run_step = len(loader)
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

        exhausted = False
        for update_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(update_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator') and self.accelerator is not None and \
               self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Accelerate 경로: 외부 루프 = 마이크로배치
                # cycle 이라 StopIteration 없음
                self.ctx.data_batch = next(data_loader_iter)
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
                # 비-Accelerate (또는 EVAL/TEST) 경로
                # EVAL/TEST: grad_accum_step=1 이므로 내부 루프 1회
                for k in range(grad_accum_step):
                    # 실제 마이크로배치 인덱스(학습 시에만 의미)
                    self.ctx.cur_batch_i = CtxVar(update_i * grad_accum_step + k,
                                                  LIFECYCLE.BATCH)
                    try:
                        self.ctx.data_batch = next(data_loader_iter)
                    except StopIteration:
                        # 학습: 재시작하여 고정 스텝 보장
                        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                            # iter(loader) 재생성
                            data_loader_iter = iter(loader)
                            setattr(self.ctx, iter_name, data_loader_iter)
                            try:
                                self.ctx.data_batch = next(data_loader_iter)
                            except StopIteration:
                                exhausted = True
                                break
                        else:
                            # 평가: 재시작하지 않음
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

    def _hook_on_fit_start_init(self, ctx):
        """
        라운드 시작 초기화:
        - dataloader 복원/생성 (train/val/test)
        - Accelerator 매 라운드 재생성 (필요시)
        - optimizer/scheduler 준비
        - (중요) accelerate.prepare 에 dataloader 는 넘기지 않음
        """
        # --- 로더 복원/생성 ---
        for split in ["train", "val", "test"]:
            loader_key = f"{split}_loader"
            if getattr(ctx, loader_key, None) is None:
                if isinstance(self.ctx.data, dict) and split in self.ctx.data:
                    setattr(ctx, loader_key, self.ctx.data[split])
                else:
                    raw = ctx.get(f"{split}_data", None)
                    if raw is None and hasattr(self.ctx.data, f"{split}_data"):
                        raw = getattr(self.ctx.data, f"{split}_data")
                    if raw is not None:
                        dl = get_dataloader(WrapDataset(raw), self.cfg, split)
                        setattr(ctx, loader_key, ReIterator(dl))

        # 이전 라운드 잔여 이터레이터 제거
        for it_name in [
            "train_loader_iter", "val_loader_iter", "test_loader_iter",
            "train_cycle_iter", "val_cycle_iter", "test_cycle_iter"
        ]:
            if hasattr(ctx, it_name):
                delattr(ctx, it_name)
        # (self 에도 cycle 이 있을 수 있으니 제거)
        for it_name in ["train_cycle_iter", "val_cycle_iter", "test_cycle_iter"]:
            if hasattr(self, it_name):
                delattr(self, it_name)

        # --- Accelerator 재생성 ---
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

            # 학습 모드: optimizer/scheduler 준비
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

                # ⛔ dataloader 는 prepare 에 넘기지 않음
                ctx.model, ctx.optimizer = self.accelerator.prepare(unwrapped_model, ctx.optimizer)
            else:
                # 평가 모드: 모델만 prepare
                ctx.model = self.accelerator.prepare(unwrapped_model)

        elif ctx.cfg.llm.deepspeed.use:
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = deepspeed.initialize(
                    config=ctx.cfg.llm.deepspeed.ds_config,
                    model=ctx.model,
                    model_parameters=filter(lambda p: p.requires_grad, ctx.model.parameters()),
                )
            ctx.device = ctx.model_engine.local_rank
        else:
            # 기본 PyTorch
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                if not hasattr(self, 'scheduler') or self.scheduler is None:
                    self.scheduler = get_scheduler(ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)
                ctx.scheduler = self.scheduler
                current_lr = ctx.optimizer.param_groups[0]['lr']
                round_num = getattr(ctx, 'cur_round_i', 'N/A')
                logger.info(f"Round #{round_num} - Current Learning Rate: {current_lr}")

        # 통계 초기화
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
        # dataloader 준비는 fit_start_init에서 처리
        pass

    def _hook_on_batch_forward(self, ctx):
        # LLM 전용: dict 배치 가정
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        # NaN 방어
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.loss_task = CtxVar(loss, LIFECYCLE.BATCH)   # ★ backward에서 사용
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(ctx.loss_task)
            if getattr(self.accelerator, "sync_gradients", True):
                if getattr(ctx, "grad_clip", 0) and ctx.grad_clip > 0:
                    try:
                        self.accelerator.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
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
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        # 메모리 절약: 배치 텐서 CPU로
        if isinstance(ctx.data_batch, dict):
            if 'input_ids' in ctx.data_batch:
                ctx.data_batch['input_ids'].cpu()
            if 'labels' in ctx.data_batch:
                ctx.data_batch['labels'].cpu()
            if 'attention_mask' in ctx.data_batch:
                ctx.data_batch['attention_mask'].cpu()

    def _hook_on_batch_end(self, ctx):
        # NaN 배치 재시도 옵션
        if ctx.skip_this_batch:
            if getattr(ctx.cfg.llm, "retry_on_nan_loss", False):
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        # 평균 계산은 기존 Monitor/상위 로직을 사용한다면 생략해도 무방
        return

    def _hook_on_fit_end_free_space(self, ctx):
        # 학습 끝난 뒤 객체 정리 (메모리 누수 방지)
        attributes_to_delete = [
            'optimizer', 'scheduler', 'loss_batch', 'loss_task', 'loss_regular',
            'loss_batch_total', 'loss_regular_total', 'y_true', 'y_prob',
            'ys_true', 'ys_pred', 'data_batch', 'grad', 'model_engine',
            'skip_this_batch',
            # dataloader & iterators
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter',
            'train_cycle_iter', 'val_cycle_iter', 'test_cycle_iter',
        ]

        if hasattr(ctx, 'rank') and ctx.rank == 0:
            deleted_attrs = [attr for attr in attributes_to_delete if hasattr(ctx, attr)]
            logger.info(f"[Memory Cleanup] Deleting attributes: {deleted_attrs}")

        for attr in attributes_to_delete:
            if hasattr(ctx, attr):
                delattr(ctx, attr)

        # self 에 잠재적으로 남아 있을 수 있는 cycle 제거
        for it_name in ['train_cycle_iter', 'val_cycle_iter', 'test_cycle_iter']:
            if hasattr(self, it_name):
                delattr(self, it_name)

        if hasattr(self, 'accelerator'):
            del self.accelerator
            logger.info("Accelerator object has been deleted.")

        gc.collect()
        torch.cuda.empty_cache()

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does not contain a valid monitor.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model, inputs=(input_ids, attention_mask)
                    ).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)
                    ).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("FLOPs count may cause OOM; set eval.count_flops=False to avoid.")
                logger.error(e)
                logger.warning(
                    "Current flop count assumes ctx.data_batch has "
                    "[input_ids, labels, attention_mask] and model signature matches."
                )
                ctx.monitor.flops_per_sample = -1

        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
