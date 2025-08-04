import torch
import logging
import gc
import math
import itertools  # for cycle to avoid StopIteration on short datasets

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
import sys
sys.setrecursionlimit(100000)


def _get_cfg(obj, path_list, default=None):
    """안전하게 설정값 읽기: path_list = [('llm','per_device_batch_size'), ('train','batch_size')]"""
    for path in path_list:
        cur = obj
        ok = True
        for key in path:
            if not hasattr(cur, key):
                ok = False
                break
            cur = getattr(cur, key)
        if ok and cur is not None:
            return cur
    return default


def _loader_batch_size(loader):
    """DataLoader 혹은 ReIterator → batch_size 추출"""
    if loader is None:
        return None
    bs = getattr(loader, "batch_size", None)
    if bs is None and hasattr(loader, "loader"):
        bs = getattr(loader.loader, "batch_size", None)
    return bs


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        num_train_batch = len(data['train'])
        self.grad_accum_step = min(
            num_train_batch,
            max(config.llm.grad_accum_step, config.grad.grad_accum_count)
        )
        # 목표 per-device micro batch (없으면 2로 고정)
        self.target_micro_bs = _get_cfg(config, [
            ('llm', 'per_device_batch_size'),
            ('train', 'batch_size'),
            ('data', 'batch_size'),
        ], default=2)

        super().__init__(model, data, device, config, only_for_eval, monitor)
        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root, config.llm.tok_len)
        self.eval_metrics = config.eval.metrics

        # 러untime 상태
        self._eff_grad_accum = self.grad_accum_step
        self._observed_micro_bs = None

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
        # 모드/로더 준비
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            base_grad_accum = self.grad_accum_step
        else:
            base_grad_accum = 1

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return

        # 🔁 train: 분산 샘플러 에폭 시드 고정
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

        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            desired_updates = getattr(self.cfg.train, "local_update_steps", None)
            assert desired_updates and desired_updates > 0, \
                "cfg.train.local_update_steps (>0) 가 필요합니다. (예: 30)"

            using_accel = hasattr(self, 'accelerator') and self.accelerator is not None

            # --- 효과적 grad_accum (라운드 시작 시 계산됨) ---
            eff_accum = self._eff_grad_accum if using_accel else base_grad_accum

            if using_accel:
                # 계획/진행 마이크로스텝(=미니배치 호출 횟수)
                planned_micro = desired_updates * max(1, eff_accum)
                done_micro = getattr(self.ctx, "_train_microsteps_done", 0)
                remain_micro = max(0, planned_micro - done_micro)
                if run_step == -1:
                    run_step = remain_micro
                if run_step <= 0:
                    logger.info(f"[{split}] nothing to do (accelerate): planned={planned_micro}, done={done_micro}")
                    return

                if not hasattr(self, cycle_name) or getattr(self, cycle_name) is None:
                    setattr(self, cycle_name, itertools.cycle(loader))
                data_loader_iter = getattr(self, cycle_name)

                logger.info(
                    f"[{split}|accelerate] planned_micro={planned_micro}, "
                    f"done_micro={done_micro}, this_run_micro={run_step}, "
                    f"eff_grad_accum={eff_accum}, "
                    f"obs_micro_bs={self._observed_micro_bs}, target_micro_bs={self.target_micro_bs}"
                )

                executed_micro = 0
                for micro_i in range(run_step):
                    executed_micro += 1
                    self.ctx.cur_batch_i = CtxVar(micro_i, LIFECYCLE.BATCH)
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
                self.ctx._train_microsteps_done = done_micro + executed_micro

            else:
                # 비-accelerate 경로(현재 환경에선 사용 안하지만 유지)
                planned_updates = desired_updates
                done_updates = getattr(self.ctx, "_train_updates_done", 0)
                remain_updates = max(0, planned_updates - done_updates)
                if run_step == -1:
                    run_step = remain_updates
                if run_step <= 0:
                    logger.info(f"[{split}] nothing to do (vanilla): planned={planned_updates}, done={done_updates}")
                    return

                if not hasattr(self.ctx, iter_name):
                    setattr(self.ctx, iter_name, iter(loader))
                data_loader_iter = getattr(self.ctx, iter_name)

                logger.info(
                    f"[{split}|vanilla] planned_updates={planned_updates}, "
                    f"done_updates={done_updates}, this_run_updates={run_step}, "
                    f"grad_accum={base_grad_accum}"
                )

                executed_updates = 0
                exhausted = False
                for update_i in range(run_step):
                    executed_updates += 1
                    for k in range(base_grad_accum):
                        self.ctx.cur_batch_i = CtxVar(update_i * base_grad_accum + k, LIFECYCLE.BATCH)
                        try:
                            self.ctx.data_batch = next(data_loader_iter)
                        except StopIteration:
                            data_loader_iter = iter(loader)
                            setattr(self.ctx, iter_name, data_loader_iter)
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
                self.ctx._train_updates_done = done_updates + executed_updates

        else:
            # EVAL/TEST: 정확히 1 epoch
            if run_step == -1:
                run_step = len(loader)
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

            for step_i in range(run_step):
                self.ctx.cur_batch_i = CtxVar(step_i, LIFECYCLE.BATCH)
                try:
                    self.ctx.data_batch = next(data_loader_iter)
                except StopIteration:
                    break
                for hook in hooks_set["on_batch_start"]:
                    hook(self.ctx)
                for hook in hooks_set["on_batch_forward"]:
                    hook(self.ctx)
                for hook in hooks_set["on_batch_backward"]:
                    hook(self.ctx)
                for hook in hooks_set["on_batch_end"]:
                    hook(self.ctx)

    def _hook_on_fit_start_init(self, ctx):
        # --- 로더 준비/복원 ---
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

        # 이전 라운드 이터레이터/사이클 제거
        for it_name in [
            "train_loader_iter", "val_loader_iter", "test_loader_iter",
            "train_cycle_iter", "val_cycle_iter", "test_cycle_iter"
        ]:
            if hasattr(ctx, it_name):
                delattr(ctx, it_name)
        for it_name in ["train_cycle_iter", "val_cycle_iter", "test_cycle_iter"]:
            if hasattr(self, it_name):
                delattr(self, it_name)

        # 진행 카운터 리셋
        ctx._train_microsteps_done = 0
        ctx._train_updates_done = 0

        # 관측 micro-bs (train 기준)
        train_loader = ctx.get("train_loader")
        obs_bs = _loader_batch_size(train_loader)
        if not obs_bs:
            obs_bs = self.target_micro_bs
        self._observed_micro_bs = int(obs_bs)

        # --- Accelerator / Optimizer / Scheduler ---
        if ctx.cfg.llm.accelerator.use:
            # 목표 샘플 수 유지: eff_accum = orig_accum * target_bs / observed_bs
            eff_accum = max(1, int(round(self.grad_accum_step * float(self.target_micro_bs) / float(self._observed_micro_bs))))
            self._eff_grad_accum = eff_accum

            logger.info(
                f"Re-creating Accelerator for the new round. "
                f"(orig_accum={self.grad_accum_step}, eff_accum={eff_accum}, "
                f"obs_micro_bs={self._observed_micro_bs}, target_micro_bs={self.target_micro_bs})"
            )

            self.accelerator = Accelerator(
                gradient_accumulation_steps=eff_accum,
                mixed_precision='bf16'
            )
            ctx.device = self.accelerator.device

            # 모델 언랩 & 샤딩 훅
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

                # dataloader는 prepare에 넘기지 않음
                ctx.model, ctx.optimizer = self.accelerator.prepare(unwrapped_model, ctx.optimizer)
            else:
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
            logger.warning('Skip the batch due to the loss is NaN.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.loss_task = CtxVar(loss, LIFECYCLE.BATCH)
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
            # 비-accelerate에서는 원래 설정 유지
            (ctx.loss_task / self.grad_accum_step).backward()
            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if getattr(ctx, "grad_clip", 0) and ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        # 메모리 절약
        if isinstance(ctx.data_batch, dict):
            if 'input_ids' in ctx.data_batch:
                ctx.data_batch['input_ids'].cpu()
            if 'labels' in ctx.data_batch:
                ctx.data_batch['labels'].cpu()
            if 'attention_mask' in ctx.data_batch:
                ctx.data_batch['attention_mask'].cpu()

    def _hook_on_batch_end(self, ctx):
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
        return

    def _hook_on_fit_end_free_space(self, ctx):
        attributes_to_delete = [
            'optimizer', 'scheduler', 'loss_batch', 'loss_task', 'loss_regular',
            'loss_batch_total', 'loss_regular_total', 'y_true', 'y_prob',
            'ys_true', 'ys_pred', 'data_batch', 'grad', 'model_engine',
            'skip_this_batch',
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter',
            'train_cycle_iter', 'val_cycle_iter', 'test_cycle_iter',
            '_train_microsteps_done', '_train_updates_done',
        ]

        if hasattr(ctx, 'rank') and ctx.rank == 0:
            deleted_attrs = [attr for attr in attributes_to_delete if hasattr(ctx, attr)]
            logger.info(f"[Memory Cleanup] Deleting attributes: {deleted_attrs}")

        for attr in attributes_to_delete:
            if hasattr(ctx, attr):
                delattr(ctx, attr)

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
            logger.warning(f"The trainer {type(self)} does not contain a valid monitor.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(ctx.model.model, inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("FLOPs count may cause OOM; set eval.count_flops=False to avoid.")
                logger.error(e)
                ctx.monitor.flops_per_sample = -1

        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
