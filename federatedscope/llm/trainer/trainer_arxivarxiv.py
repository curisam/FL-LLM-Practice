import torch
import logging
import gc
import math

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except Exception:
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


class LLMTrainer(GeneralTorchTrainer):
    """
    - Accelerate/Deepspeed/Vanilla 지원
    - Reward-choice 데이터(샘플당 다중 시퀀스)에서 '논리 배치 크기'로만 집계
    - 배치별 n_choices 자동 추정 (하드코딩 X)
    - 더블 카운팅 가드
    """

    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        num_train_batch = len(data['train'])
        self.grad_accum_step = min(
            num_train_batch,
            max(getattr(config.llm, "grad_accum_step", 1),
                getattr(getattr(config, "grad", object()), "grad_accum_count", 1))
        )

        super().__init__(model, data, device, config, only_for_eval, monitor)

        # tokenizer
        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root, config.llm.tok_len)
        self.eval_metrics = config.eval.metrics

        # 목표 마이크로 배치 크기(= yaml dataloader.batch_size)
        self.target_micro_bs = int(getattr(config.dataloader, "batch_size", 1))

        # 선택지 힌트(예: trainer.choices=['A','B'] → 2)
        self.n_choices_hint = 0
        try:
            ch = getattr(config.trainer, "choices", None)
            if isinstance(ch, (list, tuple)) and len(ch) > 0:
                self.n_choices_hint = len(ch)
        except Exception:
            pass

    # ---------------- Hook registrations ----------------
    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        self.register_hook_in_train(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_ft(self):
        super().register_default_hooks_ft()
        self.register_hook_in_ft(self._hook_on_fit_end_free_space, "on_fit_end")

    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        self.register_hook_in_eval(self._hook_on_fit_end_free_space, "on_fit_end")

    # ---------------- Core batch loop ----------------
    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step: int = -1):
        # grad_accum_step
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            grad_accum_step = self.grad_accum_step
        else:
            grad_accum_step = 1  # eval은 누적 없음

        split = self.ctx.cur_split
        loader = self.ctx.get(f"{split}_loader")
        if loader is None:
            return

        # 분산 샘플러 에폭 시드
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
                num_updates = int(getattr(self.ctx, f"num_{split}_batch"))
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    # accelerate: 외부 루프가 마이크로 스텝
                    run_step = num_updates * max(1, grad_accum_step)
                else:
                    # 수동 누적: 외부가 업데이트 스텝
                    run_step = math.ceil(num_updates / max(1, grad_accum_step))

            if not hasattr(self.ctx, iter_name):
                setattr(self.ctx, iter_name, iter(loader))
            data_loader_iter = getattr(self.ctx, iter_name)
        else:
            # ✅ EVAL/TEST: 샤딩된 길이만큼 1회
            if run_step == -1:
                run_step = len(loader)
            data_loader_iter = iter(loader)
            setattr(self.ctx, iter_name, data_loader_iter)

        # ---- 디버그 카운터 & 중복 합산 가드 ----
        if not hasattr(self.ctx, "debug_done_micro"):
            self.ctx.debug_done_micro = 0
        if not hasattr(self.ctx, "debug_first_batch_logged"):
            self.ctx.debug_first_batch_logged = False
        # (epoch, global_batch_idx)를 토큰으로 사용
        self.ctx._last_count_token = None

        using_accel = hasattr(self, "accelerator")
        if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            planned_micro = (int(getattr(self.ctx, f"num_{split}_batch")) *
                             max(1, grad_accum_step) if using_accel
                             else int(run_step) * max(1, grad_accum_step))
            logger.info(
                f"[{split}|{'accelerate' if using_accel else 'vanilla'}] "
                f"planned_micro={planned_micro}, done_micro={self.ctx.debug_done_micro}, "
                f"this_run_micro={run_step if using_accel else run_step*grad_accum_step}, "
                f"grad_accum={grad_accum_step}, target_micro_bs={self.target_micro_bs}"
            )

        exhausted = False
        for update_i in range(run_step):
            # global batch index (마이크로 기준)
            global_micro_i = update_i if using_accel else (update_i * grad_accum_step)

            if using_accel:
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
                    self.ctx.cur_batch_i = CtxVar(global_micro_i, LIFECYCLE.BATCH)
                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)

                self.ctx.debug_done_micro += 1

                if not self.ctx.debug_first_batch_logged:
                    logger.info(
                        f"[{split}|accelerate] raw_bs={getattr(self.ctx, 'debug_raw_bs', None)}, "
                        f"inferred_n_choices={getattr(self.ctx, 'debug_inferred_choices', None)}, "
                        f"logical_bs={getattr(self.ctx, 'batch_size', None)}, "
                        f"target_micro_bs={self.target_micro_bs}"
                    )
                    self.ctx.debug_first_batch_logged = True

            else:
                for k in range(grad_accum_step):
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

                    self.ctx.cur_batch_i = CtxVar(global_micro_i + k, LIFECYCLE.BATCH)
                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)
                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)

                    self.ctx.debug_done_micro += 1

                    if not self.ctx.debug_first_batch_logged:
                        logger.info(
                            f"[{split}|vanilla] raw_bs={getattr(self.ctx, 'debug_raw_bs', None)}, "
                            f"inferred_n_choices={getattr(self.ctx, 'debug_inferred_choices', None)}, "
                            f"logical_bs={getattr(self.ctx, 'batch_size', None)}, "
                            f"target_micro_bs={self.target_micro_bs}"
                        )
                        self.ctx.debug_first_batch_logged = True

            if exhausted:
                break

    # ---------------- Helper ----------------
    @staticmethod
    def _safe_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    def _infer_n_choices_from_batch(self, batch, raw_bs: int) -> int:
        """
        배치에서 선택지 개수 추정:
        1) batch['choice_group_size']가 있으면 우선 사용
        2) cfg.trainer.choices 길이 힌트
        3) 휴리스틱: 2..8 중 raw_bs를 나누며 (raw_bs//cand) ~= target_micro_bs 이면 채택
        실패 시 1
        """
        # 1) 명시값
        try:
            if isinstance(batch, dict) and 'choice_group_size' in batch:
                v = int(batch['choice_group_size'])
                if v >= 1:
                    return v
        except Exception:
            pass

        # 2) 설정 힌트
        if getattr(self, "n_choices_hint", 0) >= 1:
            hint = int(self.n_choices_hint)
            if hint >= 1 and raw_bs % hint == 0:
                return hint

        # 3) 휴리스틱
        for cand in range(2, 9):
            if raw_bs % cand == 0:
                logical = raw_bs // cand
                if logical == self.target_micro_bs or abs(logical - self.target_micro_bs) <= 1:
                    return cand

        return 1

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


    # def _hook_on_fit_start_init(self, ctx):
    #     """
    #     라운드 시작 시 필요한 초기화를 수행합니다.
    #     - Accelerator는 최초 1회만 생성하고, 이후 라운드에서는 재사용합니다.
    #     - cfg.llm.accelerator.use / mixed_precision 을 안전하게 반영합니다.
    #     - model/optimizer/dataloaders 는 최초 1회만 accelerator.prepare 로 래핑합니다.
    #     - AcceleratorState/PartialState 의 _reset_state() 는 절대 호출하지 않습니다.
    #     """
    #     # -----------------------
    #     # 1) cfg 안전 접근
    #     # -----------------------
    #     llm_cfg = getattr(self.cfg, "llm", None)
    #     accel_cfg = getattr(llm_cfg, "accelerator", None) if llm_cfg is not None else None

    #     use_accel = bool(getattr(accel_cfg, "use", False)) if accel_cfg is not None else False
    #     mixed_precision = getattr(accel_cfg, "mixed_precision", "no") if accel_cfg is not None else "no"
    #     if not use_accel:
    #         mixed_precision = "no"  # 강제 비활성화
    #     mp = str(mixed_precision).lower()
    #     if mp not in ("no", "fp16", "bf16"):
    #         mp = "no"

    #     grad_accum = int(getattr(llm_cfg, "grad_accum_step", 1)) if llm_cfg is not None else 1

    #     # -----------------------
    #     # 2) Accelerator 생성/재사용
    #     # -----------------------
    #     logger = getattr(self, "logger", None)

    #     if use_accel:
    #         # 이미 있으면 그대로 재사용 (재생성 금지)
    #         if getattr(self, "accelerator", None) is None:
    #             if logger:
    #                 logger.info(f"Creating Accelerator (mixed_precision={mp}, grad_accum_steps={grad_accum}).")
    #             # device 선택은 Accelerate가 알아서 함; CPU 강제 필요 시 cpu=True 옵션 추가 가능
    #             self.accelerator = Accelerator(
    #                 mixed_precision=mp,               # "no" | "fp16" | "bf16"
    #                 gradient_accumulation_steps=grad_accum,
    #             )
    #         else:
    #             if logger:
    #                 logger.debug("Reusing existing Accelerator instance.")
    #         ctx.accelerator = self.accelerator
    #         try:
    #             ctx.device = self.accelerator.device
    #         except Exception:
    #             pass

    #     # ── 여기를 추가 ──
    #     # 아직 모델이 CPU 위에 있을 수 있으니, accelerator.device 로 명시적 이동
    #     try:
    #         if hasattr(self, "model") and self.model is not None:
    #             self.model.to(self.accelerator.device)
    #             logger.info(f"Model moved to device: {self.accelerator.device}")
    #     except Exception as e:
    #         logger.warning(f"Failed to .to(device) the model: {e}")
    #     # ───────────────────


    #     else:
    #         # 가속기 비사용
    #         if logger:
    #             logger.info("Running without Accelerator (mixed_precision=no).")
    #         ctx.accelerator = None
    #         # device 힌트(가능하면 유지)
    #         try:
    #             import torch
    #             if isinstance(getattr(self.cfg, "device", None), int) and self.cfg.device >= 0 and torch.cuda.is_available():
    #                 ctx.device = torch.device(f"cuda:{self.cfg.device}")
    #             else:
    #                 ctx.device = torch.device("cpu")
    #         except Exception:
    #             pass

    #     # ---------------------------------------
    #     # 3) criterion / 모델 / 옵티마 / 로더 device/prepare
    #     #    - Accelerator를 쓰는 경우 최초 1회만 prepare
    #     # ---------------------------------------
    #     # (1) criterion to device (가능한 경우)
    #     try:
    #         if hasattr(self, "criterion") and self.criterion is not None and hasattr(ctx, "device"):
    #             try:
    #                 self.criterion.to(ctx.device)
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     # (2) accelerator.prepare (최초 1회만)
    #     if use_accel and getattr(self, "_prepared_with_accelerator", False) is not True:
    #         prep_items = []
    #         index_map = {}

    #         # model
    #         if hasattr(self, "model") and self.model is not None:
    #             index_map["model"] = len(prep_items)
    #             prep_items.append(self.model)
    #         # optimizer
    #         if hasattr(self, "optimizer") and self.optimizer is not None:
    #             index_map["optimizer"] = len(prep_items)
    #             prep_items.append(self.optimizer)
    #         # dataloaders (train/val/test 있을 때만)
    #         if hasattr(self, "train_loader") and self.train_loader is not None:
    #             index_map["train_loader"] = len(prep_items)
    #             prep_items.append(self.train_loader)
    #         if hasattr(self, "val_loader") and self.val_loader is not None:
    #             index_map["val_loader"] = len(prep_items)
    #             prep_items.append(self.val_loader)
    #         if hasattr(self, "test_loader") and self.test_loader is not None:
    #             index_map["test_loader"] = len(prep_items)
    #             prep_items.append(self.test_loader)

    #         if len(prep_items) > 0:
    #             try:
    #                 prepared = self.accelerator.prepare(*prep_items)
    #                 if len(prep_items) == 1:
    #                     prepared = (prepared,)
    #                 # 되돌려 꽂기
    #                 for name, idx in index_map.items():
    #                     obj = prepared[idx]
    #                     if name == "model":
    #                         self.model = obj
    #                     elif name == "optimizer":
    #                         self.optimizer = obj
    #                     elif name == "train_loader":
    #                         self.train_loader = obj
    #                     elif name == "val_loader":
    #                         self.val_loader = obj
    #                     elif name == "test_loader":
    #                         self.test_loader = obj

    #                 self._prepared_with_accelerator = True
    #                 if logger:
    #                     logger.info("accelerator.prepare(...) applied to model/optimizer/dataloaders.")
    #             except Exception as e:
    #                 if logger:
    #                     logger.warning(f"[accelerator.prepare] skipped due to error: {e}")

    #     # -----------------------
    #     # 4) 이후 훅들이 참조할 컨텍스트 마커
    #     # -----------------------
    #     try:
    #         ctx.is_accelerated = bool(use_accel and getattr(self, "accelerator", None) is not None)
    #     except Exception:
    #         pass

    # def _hook_on_fit_start_init(self, ctx):
    #     """
    #     라운드 시작 시 필요한 초기화를 수행합니다.
    #     - Accelerate Accelerator는 한 번만 생성하고 이후 라운드에서는 재사용합니다.
    #     - cfg.llm.accelerator.use / mixed_precision 설정을 안전하게 반영합니다.
    #     - 이미 구성된 model / optimizer / dataloaders 가 있으면 최초 1회 accelerator.prepare 로 래핑합니다.
    #     """
    #     # -----------------------
    #     # 1) cfg 안전 접근
    #     # -----------------------
    #     llm_cfg = getattr(self.cfg, "llm", None)
    #     accel_cfg = getattr(llm_cfg, "accelerator", None) if llm_cfg is not None else None

    #     use_accel = bool(getattr(accel_cfg, "use", False)) if accel_cfg is not None else False
    #     mixed_precision = getattr(accel_cfg, "mixed_precision", "no") if accel_cfg is not None else "no"
    #     if not use_accel:
    #         mixed_precision = "no"  # 강제 비활성화

    #     grad_accum = int(getattr(llm_cfg, "grad_accum_step", 1)) if llm_cfg is not None else 1

    #     # -----------------------
    #     # 2) Accelerator 생성/재사용
    #     # -----------------------
    #     logger = getattr(self, "logger", None)
    #     if use_accel:
    #         try:
    #             from accelerate import Accelerator
    #         except Exception as e:
    #             if logger:
    #                 logger.warning(f"[accelerate] import failed ({e}), fallback to no accelerator.")
    #             use_accel = False
    #             mixed_precision = "no"

    #     if use_accel:
    #         if getattr(self, "accelerator", None) is None:
    #             # 처음 한 번만 생성
    #             if logger:
    #                 logger.info(f"Creating Accelerator (mixed_precision={mixed_precision}, grad_accum_steps={grad_accum}).")
    #             self.accelerator = Accelerator(
    #                 gradient_accumulation_steps=grad_accum,
    #                 mixed_precision=mixed_precision,  # "no" | "fp16" | "bf16"
    #             )
    #         else:
    #             # 라운드별 재생성 금지(내부 상태 오류 방지)
    #             if logger:
    #                 try:
    #                     # 가끔 기존 코드에서 재생성 로그를 찍는 경우가 있었음 → 재사용으로 변경
    #                     logger.debug("Reusing existing Accelerator instance.")
    #                 except Exception:
    #                     pass
    #         # 컨텍스트 공유
    #         ctx.accelerator = self.accelerator
    #         try:
    #             ctx.device = self.accelerator.device
    #         except Exception:
    #             pass
    #     else:
    #         # 가속기 비사용
    #         if logger:
    #             logger.info("Running without Accelerator (mixed_precision=no).")
    #         ctx.accelerator = None
    #         # device 힌트(가능하면 유지)
    #         try:
    #             import torch
    #             # cfg.device 가 -1 이면 cuda:0 자동선정 대신 cpu로
    #             if isinstance(getattr(self.cfg, "device", None), int) and self.cfg.device >= 0 and torch.cuda.is_available():
    #                 ctx.device = torch.device(f"cuda:{self.cfg.device}")
    #             else:
    #                 ctx.device = torch.device("cpu")
    #         except Exception:
    #             pass

    #     # ---------------------------------------
    #     # 3) criterion / 모델 / 옵티마 / 로더 device/prepare
    #     #    - 이미 준비되어 있을 수 있으므로 있는 것만 처리
    #     #    - Accelerator를 쓰는 경우 최초 1회만 prepare
    #     # ---------------------------------------
    #     # (1) criterion to device
    #     try:
    #         if hasattr(self, "criterion") and self.criterion is not None and hasattr(ctx, "device"):
    #             # 일부 loss는 .to 지원하지 않을 수 있음 -> try-guard
    #             try:
    #                 self.criterion.to(ctx.device)
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     # (2) accelerator.prepare (최초 1회만)
    #     #     - 이미 prepare된 객체를 또 prepare하면 에러/비효율 → 가드 플래그 사용
    #     if use_accel and getattr(self, "_prepared_with_accelerator", False) is not True:
    #         prep_items = []
    #         index_map = {}

    #         # model
    #         if hasattr(self, "model") and self.model is not None:
    #             index_map["model"] = len(prep_items)
    #             prep_items.append(self.model)
    #         # optimizer
    #         if hasattr(self, "optimizer") and self.optimizer is not None:
    #             index_map["optimizer"] = len(prep_items)
    #             prep_items.append(self.optimizer)
    #         # dataloaders (train/val/test 있을 때만)
    #         if hasattr(self, "train_loader") and self.train_loader is not None:
    #             index_map["train_loader"] = len(prep_items)
    #             prep_items.append(self.train_loader)
    #         if hasattr(self, "val_loader") and self.val_loader is not None:
    #             index_map["val_loader"] = len(prep_items)
    #             prep_items.append(self.val_loader)
    #         if hasattr(self, "test_loader") and self.test_loader is not None:
    #             index_map["test_loader"] = len(prep_items)
    #             prep_items.append(self.test_loader)

    #         if len(prep_items) > 0:
    #             try:
    #                 prepared = self.accelerator.prepare(*prep_items)
    #                 # 단일 인자일 경우 object 하나가, 복수일 경우 튜플이 반환
    #                 if len(prep_items) == 1:
    #                     prepared = (prepared,)

    #                 # 되돌려 꽂기
    #                 for name, idx in index_map.items():
    #                     obj = prepared[idx]
    #                     if name == "model":
    #                         self.model = obj
    #                     elif name == "optimizer":
    #                         self.optimizer = obj
    #                     elif name == "train_loader":
    #                         self.train_loader = obj
    #                     elif name == "val_loader":
    #                         self.val_loader = obj
    #                     elif name == "test_loader":
    #                         self.test_loader = obj

    #                 self._prepared_with_accelerator = True
    #                 if logger:
    #                     logger.info("accelerator.prepare(...) applied to model/optimizer/dataloaders.")
    #             except Exception as e:
    #                 if logger:
    #                     logger.warning(f"[accelerator.prepare] skipped due to error: {e}")
    #                 # prepare 실패해도 학습 자체는 계속 진행

    #     # -----------------------
    #     # 4) 기타 컨텍스트 마커/초기화
    #     # -----------------------
    #     # 이후 훅들에서 참조할 수 있게 장치/가속기 정보 남겨두기
    #     try:
    #         ctx.is_accelerated = bool(use_accel and getattr(self, "accelerator", None) is not None)
    #     except Exception:
    #         pass




    
    # # federatedscope/llm/trainer/trainer.py  안의 클래스 메서드 교체
    # def _hook_on_fit_start_init(self, ctx):
    #     """
    #     (생략) 기존 주석/로그 그대로 두고, Accelerator 생성 직전에
    #     accelerate 전역 상태를 리셋하여 라운드마다 재생성해도 안전하도록 함.
    #     """
    #     # --- 기존 코드에서 필요한 선행 초기화들 그대로 유지 ---
    #     # 예: 데이터로더 재생성, seed 설정 등등 ...
    #     # self.data, self.model 등 준비 끝난 시점이라고 가정

    #     # --- Accelerate 전역 상태를 안전하게 리셋 ---
    #     try:
    #         from accelerate.state import AcceleratorState, PartialState
    #         try:
    #             AcceleratorState._reset_state()
    #         except Exception:
    #             pass
    #         try:
    #             PartialState._reset_state()
    #         except Exception:
    #             pass
    #     except Exception:
    #         pass

    #     # --- Accelerator 생성 (1차 시도) ---
    #     from accelerate import Accelerator
    #     try:
    #         self.accelerator = Accelerator(
    #             mixed_precision=getattr(getattr(self.cfg, "llm", None), "accelerator", None).mixed_precision
    #                             if getattr(self.cfg, "llm", None) and getattr(self.cfg.llm, "accelerator", None)
    #                             else "no",
    #             cpu=(str(getattr(self.cfg, "device", "")).lower() == "cpu"),
    #         )
    #     except AttributeError:
    #         # ─ 재시도: 상태 다시 리셋 후 1회 더
    #         try:
    #             from accelerate.state import AcceleratorState, PartialState
    #             try:
    #                 AcceleratorState._reset_state()
    #             except Exception:
    #                 pass
    #             try:
    #                 PartialState._reset_state()
    #             except Exception:
    #                 pass
    #         except Exception:
    #             pass
    #         self.accelerator = Accelerator(
    #             mixed_precision=getattr(getattr(self.cfg, "llm", None), "accelerator", None).mixed_precision
    #                             if getattr(self.cfg, "llm", None) and getattr(self.cfg.llm, "accelerator", None)
    #                             else "no",
    #             cpu=(str(getattr(self.cfg, "device", "")).lower() == "cpu"),
    #         )




    # # ---------------- Fit lifecycle ----------------
    # def _hook_on_fit_start_init(self, ctx):
    #     # 로더 복원/생성
    #     for split in ["train", "val", "test"]:
    #         loader_key = f"{split}_loader"
    #         if getattr(ctx, loader_key, None) is None:
    #             if isinstance(self.ctx.data, dict) and split in self.ctx.data:
    #                 setattr(ctx, loader_key, self.ctx.data[split])
    #             else:
    #                 raw = ctx.get(f"{split}_data", None)
    #                 if raw is None and hasattr(self.ctx.data, f"{split}_data"):
    #                     raw = getattr(self.ctx.data, f"{split}_data")
    #                 if raw is not None:
    #                     dl = get_dataloader(WrapDataset(raw), self.cfg, split)
    #                     setattr(ctx, loader_key, ReIterator(dl))
    #     # 이터레이터 초기화
    #     for it_name in ["train_loader_iter", "val_loader_iter", "test_loader_iter"]:
    #         if hasattr(ctx, it_name):
    #             try:
    #                 delattr(ctx, it_name)
    #             except Exception:
    #                 pass

    #     # Accelerator 준비
    #     if ctx.cfg.llm.accelerator.use:
    #         logger.info("Re-creating Accelerator for the new round.")
    #         self.accelerator = Accelerator(
    #             gradient_accumulation_steps=self.grad_accum_step,
    #             mixed_precision='bf16'
    #         )
    #         ctx.device = self.accelerator.device

    #         # 원본 모델
    #         unwrapped_model = ctx.model
    #         while hasattr(unwrapped_model, 'module'):
    #             unwrapped_model = unwrapped_model.module
    #         if hasattr(unwrapped_model, 'sharding'):
    #             unwrapped_model.sharding()

    #         if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
    #             cfg_optimizer = ctx.cfg[ctx.cur_mode].optimizer
    #             if getattr(cfg_optimizer, "type", "AdamW") == 'AdamW':
    #                 adamw_kwargs = {'lr': cfg_optimizer.lr, 'betas': cfg_optimizer.betas}
    #                 ctx.optimizer = AdamW(unwrapped_model.parameters(), **adamw_kwargs)
    #             else:
    #                 ctx.optimizer = get_optimizer(unwrapped_model, **cfg_optimizer)
    #             ctx.scheduler = get_scheduler(ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

    #             current_lr = ctx.optimizer.param_groups[0]['lr']
    #             round_num = getattr(ctx, 'cur_round_i', 'N/A')
    #             logger.info(f"Round #{round_num} - Initializing with LR: {current_lr}")

    #             ctx.model, ctx.optimizer = self.accelerator.prepare(unwrapped_model, ctx.optimizer)
    #         else:
    #             ctx.model = self.accelerator.prepare(unwrapped_model)

    #     elif ctx.cfg.llm.deepspeed.use:
    #         assert deepspeed is not None, "Please install deepspeed."
    #         if not hasattr(ctx, 'model_engine'):
    #             ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
    #                 deepspeed.initialize(
    #                     config=ctx.cfg.llm.deepspeed.ds_config,
    #                     model=ctx.model,
    #                     model_parameters=filter(lambda p: p.requires_grad, ctx.model.parameters()),
    #                 )
    #         ctx.device = ctx.model_engine.local_rank
    #     else:
    #         ctx.model.to(ctx.device)
    #         if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
    #             ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
    #             if not hasattr(self, 'scheduler') or self.scheduler is None:
    #                 self.scheduler = get_scheduler(ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)
    #             ctx.scheduler = self.scheduler
    #             current_lr = ctx.optimizer.param_groups[0]['lr']
    #             round_num = getattr(ctx, 'cur_round_i', 'N/A')
    #             logger.info(f"Round #{round_num} - Current Learning Rate: {current_lr}")

    #     # 통계 초기화
    #     ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
    #     ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
    #     ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
    #     ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
    #     ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
    #     if hasattr(ctx, 'ys_pred'):
    #         ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

    #     if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
    #         ctx.model.train()
    #     else:
    #         ctx.model.eval()

    #     # 디버깅 상태 초기화
    #     ctx.debug_done_micro = 0
    #     ctx.debug_first_batch_logged = False
    #     ctx._last_count_token = None

    def _hook_on_epoch_start(self, ctx):
        pass

    # ---------------- Forward/Backward ----------------
    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.accelerator.use:
            outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        elif ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip batch due to NaN loss.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # --- 선택지 개수 추정 & 논리 배치 산출 ---
        raw_bs = int(labels.size(0))
        inferred = self._infer_n_choices_from_batch(ctx.data_batch, raw_bs)
        if inferred < 1:
            inferred = 1
        if raw_bs % inferred == 0:
            logical_bs = raw_bs // inferred
        else:
            logical_bs = raw_bs
            logger.warning(f"[logical-batch warn] raw_bs={raw_bs}, inferred_n_choices={inferred} → use raw_bs.")

        # 디버그 정보
        ctx.debug_raw_bs = raw_bs
        ctx.debug_inferred_choices = inferred

        # 저장
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.loss_task = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(int(logical_bs), LIFECYCLE.BATCH)  # ✅ 논리 배치

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

        # move data to cpu
        for key in ('input_ids', 'labels', 'attention_mask'):
            if key in ctx.data_batch and hasattr(ctx.data_batch[key], 'cpu'):
                try:
                    ctx.data_batch[key] = ctx.data_batch[key].cpu()
                except Exception:
                    pass

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if getattr(ctx.cfg.llm, "retry_on_nan_loss", False):
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        # ---- 더블 카운팅 가드 ----
        # 동일 (epoch, global_micro_idx) 는 한 번만 합산
        epoch_i = int(getattr(ctx, "cur_epoch_i", 0))
        global_i = int(getattr(ctx, "cur_batch_i", 0))
        token = (epoch_i, global_i)
        if getattr(ctx, "_last_count_token", None) == token:
            logger.warning(f"[count guard] duplicated on_batch_end detected at token={token}; skip.")
            return
        ctx._last_count_token = token
        # --------------------------------

        # ✅ 논리 배치 기준 집계
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        using_accel = hasattr(self, "accelerator")
        world = getattr(self.accelerator.state, "num_processes", 1) if using_accel else 1
        rank = getattr(self.accelerator.state, "process_index", 0) if using_accel else 0

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            num_updates = int(getattr(self.ctx, f"num_{self.ctx.cur_split}_batch"))
            planned_micro = (num_updates * max(1, self.grad_accum_step)
                             if using_accel
                             else math.ceil(num_updates / max(1, self.grad_accum_step)) * max(1, self.grad_accum_step))
            expected_local_total = planned_micro * self.target_micro_bs  # 1 프로세스 기대 논리 샘플 수
            logger.info(
                f"[agg debug] using_accel={using_accel}, world={world}, rank={rank}, "
                f"local_total={int(ctx.num_samples)}, expected_local_total={expected_local_total}, "
                f"done_micro={getattr(ctx, 'debug_done_micro', -1)}, planned_micro={planned_micro}, "
                f"raw_bs_first={getattr(ctx, 'debug_raw_bs', None)}, "
                f"inferred_n_choices_first={getattr(ctx, 'debug_inferred_choices', None)}, "
                f"logical_bs_first={getattr(ctx, 'batch_size', None)}, "
                f"target_micro_bs={self.target_micro_bs}"
            )
            if int(ctx.num_samples) != int(expected_local_total):
                logger.warning(
                    f"[agg mismatch] local_total({int(ctx.num_samples)}) != expected({expected_local_total}). "
                    f"Check dataloader batch_size / choice grouping / duplicate hooks."
                )
        else:
            logger.info(
                f"[agg debug|{self.ctx.cur_split}] using_accel={using_accel}, world={world}, rank={rank}, "
                f"local_total={int(ctx.num_samples)}, "
                f"logical_bs_first={getattr(ctx, 'batch_size', None)}"
            )

    # ---------------- Cleanup ----------------
    def _hook_on_fit_end_free_space(self, ctx):
        attributes_to_delete = [
            'optimizer', 'scheduler', 'loss_batch', 'loss_task', 'loss_regular',
            'loss_batch_total', 'loss_regular_total', 'y_true', 'y_prob',
            'ys_true', 'ys_pred',
            'data_batch', 'grad', 'model_engine', 'skip_this_batch',
            'train_loader', 'val_loader', 'test_loader',
            'train_loader_iter', 'val_loader_iter', 'test_loader_iter',
            'debug_done_micro', 'debug_first_batch_logged', '_last_count_token',
            'debug_raw_bs', 'debug_inferred_choices',
        ]
        for attr in attributes_to_delete:
            if hasattr(ctx, attr):
                try:
                    delattr(ctx, attr)
                except Exception:
                    pass

        if hasattr(self, 'accelerator'):
            try:
                del self.accelerator
                logger.info("Accelerator object has been deleted.")
            except Exception:
                pass

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------- FLOPs (optional) ----------------
    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(f"The trainer {type(self)} does not contain a valid monitor.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(ctx.model.model, inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("FLOPs count failed. Set `cfg.eval.count_flops=False` if OOM occurs.")
                logger.error(e)
                ctx.monitor.flops_per_sample = -1

        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        return LLMTrainer

register_trainer('llmtrainer', call_llm_trainer)










