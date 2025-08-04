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
            "train_loader_iter", "val_loader_iter"
