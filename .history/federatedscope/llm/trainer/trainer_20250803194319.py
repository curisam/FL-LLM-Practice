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

from accelerate import Accelerator
from transformers import AdamW

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

logger = logging.getLogger(__name__)
import sys; sys.setrecursionlimit(100000)


def _unwrap_loader_and_get_bs(loader):
    """
    ReIterator나 커스텀 래퍼를 감안해 batch_size를 최대한 튼튼하게 추출.
    """
    if loader is None:
        return None
    # 직접 보유
    for key in ("batch_size", "bs"):
        v = getattr(loader, key, None)
        if isinstance(v, int) and v > 0:
            return v
    # 래퍼 내부 추정
    for inner in ("loader", "dataloader", "data_loader", "dl"):
        inner_obj = getattr(loader, inner, None)
        if inner_obj is not None:
            for key in ("batch_size", "bs"):
                v = getattr(inner_obj, key, None)
                if isinstance(v, int) and v > 0:
                    return v
    return None


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        num_train_batch = len(data['train'])  # 기본 길이(참고용)

        # 원본 누적 스텝(설정 기반)
        base_accum = max(config.llm.grad_accum_step, getattr(config.grad, "grad_accum_count", 1))
        self.base_accum = max(1, base_accum)
        self.grad_accum_step = self.base_accum  # 라운드마다 보정해서 갱신함

        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root, config.llm.tok_len)
        self.eval_metrics = config.eval.metrics

    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        self.register_hook_in_train(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_ft(self):
        super().register_default_hooks_ft()
        self.register_hook_in_ft(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        self.register_hook_in_eval(self._hook_on_fit_end_free_space, "on_fit_end")

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step=-1):
        # grad_accum은 라운드 시작 시(_hook_on_fit_start_init) 보정된 self.grad_accum_step 사용
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            grad_accum_step = self.grad_accum_step
        else:
            grad_accum_step = 1

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return

        # 분산 샘플러 시드 고정
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
                # 여기서 num_batches는 "updates(=local_update_steps)"로 세팅되어 있음
                num_batches = getattr(self.ctx, f"num_{split}_batch")
                # Accelerate 사용 시: optimizer step 30회를 만들려면 micro를 (updates * eff_accum) 만큼 소비해야 함
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    run_step = num_batches * max(1, grad_accum_step)
                else:
                    # 수동 누적: 외부 루프는 업데이트 스텝 단위
                    run_step = math.ceil(num_batches / max(1, grad_accum_step))

            # 이터레이터 준비/복구
            if not hasattr(self.ctx, iter_name):
                setattr(self.ctx, iter_name, iter(loader))
            data_loader_iter = getattr(self.ctx, iter_name)
        else:
            if run_step == -1:
                run_step = len(loader)
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

        # 디버깅 로그
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            planned_micro = run_step if hasattr(self, 'accelerator') else run_step * max(1, grad_accum_step)
            logger.info(
                f"[{split}|{'accelerate' if hasattr(self,'accelerator') else 'vanilla'}] "
                f"planned_micro={planned_micro}, done_micro=0, this_run_micro={planned_micro}, "
                f"grad_accum={grad_accum_step}, obs_micro_bs={getattr(self.ctx,'_obs_micro_bs', 'NA')}, "
                f"target_micro_bs={getattr(self.ctx,'_target_micro_bs','NA')}"
            )

        exhausted = False
        for update_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(update_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator'):
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
                    self.ctx.cur_batch_i = CtxVar(update_i * grad_accum_step + k, LIFECYCLE.BATCH)
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

    def _hook_on_fit_start_init(self, ctx):
        # --- 로더 복원/바인딩 ---
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
        for it_name in ["train_loader_iter", "val_loader_iter", "test_loader_iter"]:
            if hasattr(ctx, it_name):
                delattr(ctx, it_name)

        # --- 관측 micro-BS / 타깃 micro-BS / 누적 스텝 보정 ---
        train_loader = ctx.get("train_loader")
        obs_micro_bs = _unwrap_loader_and_get_bs(train_loader)
        # target은 설정의 dataloader.batch_size를 우선
        target_micro_bs = getattr(self.cfg.dataloader, "batch_size", None)
        if not isinstance(target_micro_bs, int) or target_micro_bs <= 0:
            target_micro_bs = getattr(self.cfg.train, "batch_size", None)
        if not isinstance(target_micro_bs, int) or target_micro_bs <= 0:
            target_micro_bs = 2  # 최후의 안전값

        # 기본 누적(step)에서 관측 BS로 맞춰 자동 보정
        eff_accum = self.base_accum
        if isinstance(obs_micro_bs, int) and obs_micro_bs > 0:
            eff_accum = max(1, round(self.base_accum * target_micro_bs / obs_micro_bs))

        # 컨텍스트에 기록 (디버깅 보기 좋게)
        ctx._obs_micro_bs = obs_micro_bs
        ctx._target_micro_bs = target_micro_bs

        # --- Accelerator 생성 (보정된 누적 스텝으로) ---
        if ctx.cfg.llm.accelerator.use:
            logger.info("Re-creating Accelerator for the new round.")
            self.grad_accum_step = eff_accum  # 중요: 라운드 보정치 반영
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

                # 모델/옵티마이저만 prepare (DataLoader는 건드리지 않음)
                ctx.model, ctx.optimizer = self.accelerator.prepare(unwrapped_model, ctx.optimizer)
            else:
                ctx.model = self.accelerator.prepare(unwrapped_model)

        elif ctx.cfg.llm.deepspeed.use:
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad, ctx.model.parameters()),
                    )
            ctx.device = ctx.model_engine.local_rank

        else:
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
        pass

    def _hook_on_batch_forward(self, ctx):
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

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to NaN loss (precision/labels issue).')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
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

        # move training data to cpu
        for k in ('input_ids', 'labels', 'attention_mask'):
            if isinstance(ctx.data_batch, dict) and k in ctx.data_batch and hasattr(ctx.data_batch[k], "cpu"):
                ctx.data_batch[k] = ctx.data_batch[k].cpu()

    def _hook_on_batch_end(self, ctx):
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

    def _hook_on_fit_end_free_space(self, ctx):
        attributes_to_delete = [
            'optimizer', 'scheduler', 'loss_batch', 'loss_task', 'loss_regular',
            'loss_batch_total', 'loss_regular_total', 'y_true', 'y_prob',
            'ys_true', 'ys_pred', 'data_batch', 'grad', 'model_engine',
            'skip_this_batch',
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter'
        ]
        deleted_attrs = [attr for attr in attributes_to_delete if hasattr(ctx, attr)]
        if hasattr(ctx, 'rank') and ctx.rank == 0:
            logger.info(f"[Memory Cleanup] Deleting attributes: {deleted_attrs}")

        for attr in attributes_to_delete:
            if hasattr(ctx, attr):
                delattr(ctx, attr)

        if hasattr(self, 'accelerator'):
            del self.accelerator
            logger.info("Accelerator object has been deleted.")

        gc.collect()
        torch.cuda.empty_cache()


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        return LLMTrainer

register_trainer('llmtrainer', call_llm_trainer)
