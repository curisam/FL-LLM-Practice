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
# transformers.AdamW 경고가 거슬리면 torch.optim.AdamW로 바꿔도 됩니다.
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
from federatedscope.core.auxiliaries.decorators import use_diff

logger = logging.getLogger(__name__)
import sys
sys.setrecursionlimit(100000)


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        num_train_batch = len(data['train'])
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
        # 서브클래스가 이 훅을 안 가질 수도 있으니 방어
        if hasattr(self, "_hook_on_fit_end_free_space"):
            self.register_hook_in_train(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_ft(self):
        super().register_default_hooks_ft()
        if hasattr(self, "_hook_on_fit_end_free_space"):
            self.register_hook_in_ft(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        if hasattr(self, "_hook_on_fit_end_free_space"):
            self.register_hook_in_eval(self._hook_on_fit_end_free_space, "on_fit_end")

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step=-1):
        # grad_accum_step은 accelerator 내부에서 카운팅만 쓰고, 외부 루프는 항상 "배치 단위"로 돈다.
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            grad_accum_step = self.grad_accum_step
        else:
            grad_accum_step = 1

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return

        # 분산 샘플러는 epoch마다 시드 고정
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
                # 핵심: Accelerate 경로는 "로더 길이만큼만" 돈다.
                num_batches = getattr(self.ctx, f"num_{split}_batch")
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    run_step = len(loader)  # 절대 * grad_accum_step 하지 않음
                else:
                    # 수동 누적: 외부 루프는 '업데이트 스텝' 단위로
                    run_step = math.ceil(num_batches / max(1, grad_accum_step))
            # 이터레이터 준비 (재시작 금지)
            if not hasattr(self.ctx, iter_name):
                setattr(self.ctx, iter_name, iter(loader))
            data_loader_iter = getattr(self.ctx, iter_name)
        else:
            if run_step == -1:
                run_step = len(loader)
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

        exhausted = False
        for update_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(update_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator') and self.accelerator is not None:
                # 한 라운드에 로더 한 바퀴만. StopIteration => break (재시작 금지)
                try:
                    self.ctx.data_batch = next(data_loader_iter)
                except StopIteration:
                    exhausted = True
                    break

                # Accelerate가 누적 경계/step을 관리
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
                # 수동 누적: 업데이트 1회 당 grad_accum_step개의 마이크로배치
                for k in range(grad_accum_step):
                    self.ctx.cur_batch_i = CtxVar(update_i * grad_accum_step + k, LIFECYCLE.BATCH)
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

    def _hook_on_fit_start_init(self, ctx):
        """
        - 매 라운드 시작 시 로더 바인딩(있으면 재사용, 없으면 생성)
        - Accelerate는 model/optimizer(+옵션)를 prepare, dataloader는 이 파일에서는 넘기지 않음
        - 한 라운드에 로더 한 바퀴만 돌도록 이터레이터는 _run_batch에서만 만들고 재시작 금지
        """
        # 로더 복원/생성
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
        for it_name in ["train_loader_iter", "val_loader_iter", "test_loader_iter"]:
            if hasattr(ctx, it_name):
                delattr(ctx, it_name)

        # Accelerate / Deepspeed / 기본
        if ctx.cfg.llm.accelerator.use:
            logger.info("Re-creating Accelerator for the new round.")
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_step,
                mixed_precision='bf16'
            )
            ctx.device = self.accelerator.device

            # 모델 언랩 + 필요 시 sharding
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
            logger.warning('Skip the batch due to NaN loss.')
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
                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        # 안전하게 CPU로 이동
        if isinstance(ctx.data_batch, dict):
            for k in ('input_ids', 'labels', 'attention_mask'):
                if k in ctx.data_batch and hasattr(ctx.data_batch[k], 'cpu'):
                    ctx.data_batch[k].cpu()

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
            'ys_true', 'ys_pred', 'data_batch', 'grad', 'model_engine', 'skip_this_batch',
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
                f"The trainer {type(self)} does not contain a valid monitor."
            )
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
                logger.warning("Set `cfg.eval.count_flops = False` if OOM occurs during flop count.")
                logger.error(e)
                logger.warning(
                    "FLOPs count assumes ctx.data_batch has [input_ids, labels, attention_mask] "
                    "and ctx.model takes (input_ids, attention_mask)."
                )
                ctx.monitor.flops_per_sample = -1

        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder

register_trainer('llmtrainer', call_llm_trainer)
