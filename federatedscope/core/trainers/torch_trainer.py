import os
import logging

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = None

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.data import ClientData
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from federatedscope.core.monitors.monitor import Monitor


from federatedscope.llm.utils_dist import barrier_all

logger = logging.getLogger(__name__)


#mode: train, test, val, finetune
#lifecycle: routine, epoch, batch





class GeneralTorchTrainer(Trainer):

    def get_model_para(self): #모든 adapter parae들만 떼온다.
        if self.cfg.federate.process_num > 1 or \
                self.cfg.federate.share_local_model or \
                self.cfg.llm.deepspeed.use or \
                self.cfg.llm.accelerator.use:
            
            # ._param_filter  통해 바뀌는 거 없음. self.ctx.model.state_dict() 반환. 
            # #requires_grad=True인 파라미터 포함인 것 혹은 self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함. 
            trainable_params = self._param_filter(self.accelerator.unwrap_model(self.ctx.model).state_dict())
            
            if isinstance(self.ctx.model, torch.nn.parallel.DistributedDataParallel):
                # DDP 모델에서 추출했으므로 'module.' 접두사를 제거
                """
                DDP(model).state_dict()는 키가 module.xxx로 시작합니다 (래퍼가 한 겹 더 싸는 셈).

                서버/체크포인트/비-DDP 경로와 키를 맞추려면 접두사를 떼야 함.
                """
                cleaned_trainable_params = {
                    k.replace('module.', '', 1): v.cpu()  #문자열 안에서 처음 나타나는 module. 딱 한 번만 지울 것.
                    for k, v in trainable_params.items()
                }
                return cleaned_trainable_params
            
            # DDP가 아닌 경우 (하지만 accelerator.use=True면 보통 DDP임)
            return {k: v.cpu() for k, v in trainable_params.items()}
        else:
            return self._param_filter(self.ctx.model.cpu().state_dict())
        
    def setup_data(self, ctx):
        """
        Initialization data by ``cfg``.
        """
        if isinstance(ctx.data, ClientData):
            ctx.data.setup(ctx.cfg)#ClientData에는 원래 train_data, val_data, test_data attribute가 있는데 이걸 기반으로 data loader를 train, test, val attribute에 저장.
        else:
            logger.warning(f'The data type should be `ClientData` to '
                           f'enable new `config`, but got '
                           f'{type(ctx.data)} instead.')

    def parse_data(self, data):
        """Populate "${split}_data", "${split}_loader" and "num_${
        split}_data" for different data splits
        """

        """"
        Trainer 에 전달된 data 객체(보통 딕셔너리 형태)를 보고, 내부적으로 다음 세 가지 컨텍스트 변수를 자동으로 만들어 주는 역할을 합니다:

        "{split}_data"

        "{split}_loader"

        "num_{split}_data
        """

        #초기화, 나중에 ctx.merge_from_dict(init_dict) 호출을 통해 이 init_dict 에 담긴 키·값들이 ctx 에 한 번에 setattr 됩니다.
        init_dict = dict()

        #data 타입 검사. Trainer 에 주어진 data 가 딕셔너리가 아니면 바로 에러를 냅니다.
        if isinstance(data, dict):
            for split in data.keys():#split 키 반복. 딕셔너리의 키 중에 'train', 'val', 'test' 가 아닌 이름이 있으면 무시
                if split not in ['train', 'val', 'test']:
                    continue
                #초기값 설정. 뒤에서 실제 값을 채워 주기 전, 기본값을 None 혹은 0 으로 세팅합니다.
                init_dict["{}_data".format(split)] = None
                init_dict["{}_loader".format(split)] = None
                init_dict["num_{}_data".format(split)] = 0


                if data.get(split, None) is not None:
                    if isinstance(data.get(split), Dataset):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split))
                    elif isinstance(data.get(split), DataLoader):###### ##이거에 해당할 듯.
                        init_dict["{}_loader".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split).dataset)
                    elif isinstance(data.get(split), dict):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(split))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict #이렇게 만들어진 { 'train_loader': …, 'num_train_data': …, } 를 Trainer 가 받아서, ctx 에 일괄 등록하게 됩니다. 

 
    def _normalize_keys_for_unwrapped(self, sd: dict):
        # 언랩 원본 기준으로 통일: 'module.' 있으면 제거
        return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        # strict=False를 사용하므로, YAML 파일의 설정이 우선되도록 합니다.
        strict = self.cfg.federate.get('strict_global_model', False)

        # 1) 받은 파라미터들을 모두 텐서로 변환
        server_params = {k: param2tensor(v) for k, v in model_parameters.items()} #서버에서 온 param. module. 접두사 안붙어있음.
        server_params = self._normalize_keys_for_unwrapped(server_params)

        # 2) 언랩된 원본 모듈을 잡는다 (self.model이 항상 원본이어야 함)
        base = self.model
        while hasattr(base, 'module'):
            base = base.module

        # # 3) (선택) 동기화 후 로드
        # barrier_all()
        # missing, unexpected = base.load_state_dict(server_params, strict=strict)
        # if getattr(getattr(self, 'ctx', None), 'rank', 0) == 0:
        #     logger.info(f"[Model Sync] strict={strict}, missing={len(missing)}, unexpected={len(unexpected)}")
        # barrier_all()

        cur = base.state_dict()

        # ← 여기만 변경
        filt = {k: v for k, v in server_params.items()
                if k in cur and cur[k].shape == v.shape}
        skipped = len(server_params) - len(filt)

        barrier_all()
        missing, unexpected = base.load_state_dict(filt, strict=False)
        if getattr(getattr(self, 'ctx', None), 'rank', 0) == 0:
            logger.info(f"[Model Sync] lenient load | loaded={len(filt)} "
                        f"skipped={skipped} missing={len(missing)} unexpected={len(unexpected)}")
        barrier_all()



            
 

    def evaluate(self, target_data_split_name="test"): #그대로 GeneralTorchTrainer 상속
        with torch.no_grad():
            super(GeneralTorchTrainer, self).evaluate(target_data_split_name)

        return self.ctx.eval_metrics  #loss, avg_loss, acc, total을 기록한 dictionary

    def register_default_hooks_train(self):


        #on_fit_start (루틴의 맨 처음), 하드웨어(setting) → 2) 분산·병렬 처리 세팅 → 3) 모델·옵티마이저 세팅 → 4) 모델 크기 측정 이 순서대로 한 번만 실행됩니다.
        self.register_hook_in_train(
            self._hook_on_fit_start_numerical_precision, "on_fit_start") #(예: bfloat16 같은) 수치 정밀도 세팅
        self.register_hook_in_train(self._hook_on_data_parallel_init,
                                    "on_fit_start")#멀티-GPU 환경이라면 nn.DataParallel 래퍼 씌우기
        self.register_hook_in_train(self._hook_on_fit_start_init,
                                    "on_fit_start")#모델을 GPU/CPU에 올리고 옵티마이저 · 스케줄러 초기화. loss_batch_total, num_samples, ys_true 같은 통계 변수(CtxVar) 초기화.
        self.register_hook_in_train(
            self._hook_on_fit_start_calculate_model_size, "on_fit_start") #한 번만 모델 파라미터 총량을 계산해서 모니터에 기록
        

        # #on_epoch_start (매 에폭의 시작)
        # self.register_hook_in_train(self._hook_on_epoch_start,
        #                             "on_epoch_start")#“이제 1 에폭을 돌 거예요. 데이터 로더의 포인터를 맨 앞으로 돌려주세요.”

        #on_batch_start (매 배치의 시작)
        self.register_hook_in_train(self._hook_on_batch_start_init,
                                    "on_batch_start")#다음 배치를 꺼내서 준비해주세요.”



        #모델 순전파, 손실 계산, 그라디언트 업데이트 전 준비 과정.” 3단계로 나눠서, 기본 손실→정규화 손실→연산량 모니터를 각각 한 덩어리씩 처리합니다.
        self.register_hook_in_train(self._hook_on_batch_forward,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_forward_regularizer,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_forward_flop_count,
                                    "on_batch_forward")

        #on_batch_backward (역전파), “이 배치에서 계산된 그라디언트로 파라미터 업데이트.”
        self.register_hook_in_train(self._hook_on_batch_backward,
                                    "on_batch_backward")

        #on_batch_end (배치 종료)
        self.register_hook_in_train(self._hook_on_batch_end, "on_batch_end")#“이 배치 한 바퀴 다 돌았으니, 통계(샘플 수·손실·예측값)를 모아두세요.”


        #on_fit_end (루틴(훈련/평가) 전체 종료)
        
        self.register_hook_in_train(self._hook_on_fit_end, "on_fit_end")#“한 번의 fit이 끝났습니다. 모아둔 통계들로 최종 평가 지표를 뽑아보고 리턴하세요.”

    def register_default_hooks_ft(self):
        #on_fit_start (루틴의 맨 처음), 하드웨어(setting) → 2) 분산·병렬 처리 세팅 → 3) 모델·옵티마이저 세팅 → 4) 모델 크기 측정 이 순서대로 한 번만 실행됩니다.
        self.register_hook_in_ft(self._hook_on_fit_start_numerical_precision,
                                 "on_fit_start")
        self.register_hook_in_ft(self._hook_on_data_parallel_init,
                                 "on_fit_start")
        self.register_hook_in_ft(self._hook_on_fit_start_init, "on_fit_start")
        self.register_hook_in_ft(self._hook_on_fit_start_calculate_model_size,
                                 "on_fit_start")


        # #on_epoch_start (매 에폭의 시작) 
        # self.register_hook_in_ft(self._hook_on_epoch_start, "on_epoch_start")

        #on_batch_start (매 배치의 시작)
        self.register_hook_in_ft(self._hook_on_batch_start_init,
                                 "on_batch_start")

        #모델 순전파, 손실 계산, 그라디언트 업데이트 전 준비 과정.” 3단계로 나눠서, 기본 손실→정규화 손실→연산량 모니터를 각각 한 덩어리씩 처리합니다.
        self.register_hook_in_ft(self._hook_on_batch_forward,
                                 "on_batch_forward")
        self.register_hook_in_ft(self._hook_on_batch_forward_regularizer,
                                 "on_batch_forward")
        self.register_hook_in_ft(self._hook_on_batch_forward_flop_count,
                                 "on_batch_forward")


        #on_batch_backward (역전파), “이 배치에서 계산된 그라디언트로 파라미터 업데이트.”
        self.register_hook_in_ft(self._hook_on_batch_backward,
                                 "on_batch_backward")
        
        #on_batch_end (배치 종료) 
        self.register_hook_in_ft(self._hook_on_batch_end, "on_batch_end")

        #on_fit_end (루틴(훈련/평가) 전체 종료)
        self.register_hook_in_ft(self._hook_on_fit_end, "on_fit_end")



    def register_default_hooks_eval(self):
        # test/val
        #on_fit_start (루틴의 맨 처음), 하드웨어(setting) → 2) 분산·병렬 처리 세팅 → 3) 모델·옵티마이저 세팅
        # eval의 "on_fit_start" 시점에 이 리스트에 담긴 훅들이 등록된 순서대로 차례차례 실행
        self.register_hook_in_eval(self._hook_on_fit_start_numerical_precision,
                                   "on_fit_start") #new_hook: 추가할 함수, #trigger: 이벤트 이름 (예: "on_fit_start")
        self.register_hook_in_eval(self._hook_on_data_parallel_init,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_fit_start_init,
                                   "on_fit_start")


        # #on_epoch_start (매 에폭의 시작)
        # self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
        
        
        #on_batch_start (매 배치의 시작)
        self.register_hook_in_eval(self._hook_on_batch_start_init,
                                   "on_batch_start")
        
        #on_batch_forward #모델 순전파.
        self.register_hook_in_eval(self._hook_on_batch_forward,
                                   "on_batch_forward")
        
        #on_batch_end (배치 종료) 
        self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
        
    
        #on_fit_end (루틴(훈련/평가) 전체 종료)
        self.register_hook_in_eval(self._hook_on_fit_end, "on_fit_end")

    def _hook_on_fit_start_numerical_precision(self, ctx): #평가·훈련 전반에서 모델과 데이터를 올바른 수치 정밀도로 설정.
        # if self.cfg.train.is_enable_half:
        #     ctx.model.to(torch.bfloat16)
        pass

    def _hook_on_data_parallel_init(self, ctx): #Multi-GPU 환경일 때 PyTorch의 DataParallel 또는 DistributedDataParallel 래퍼를 씌움.


        """
        Note:
          The modified attributes and according operations are shown below,
           further modifications should be made to `ctx.model` other object:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Wrap ``nn.Module` to \
            `nn.DataParallel`
            ==================================  ===========================
        """
        if isinstance(ctx.model, torch.nn.DataParallel): #PyTorch의 DataParallel 래퍼로 감싸진 모델인지 확인하는 구문
            return

        if len(ctx.cfg.train.data_para_dids): #ctx.cfg.train.data_para_dids=[]
            ctx.model = \
                torch.nn.DataParallel(ctx.model,
                                      device_ids=ctx.cfg.train.data_para_dids) #DataParallel 래퍼로 모델을 감싼다.

    def _hook_on_fit_start_init(self, ctx):#모델을 디바이스(GPU/CPU)에 올리고 (학습 모드인 경우) 옵티마이저·스케줄러 초기화. 통계 변수 초기화 (loss_batch_total, loss_regular_total, num_samples, ys_true, ys_prob) 이 변수들은 CtxVar(..., LIFECYCLE.ROUTINE)으로 감싸 루틴 종료 시 자동 소멸
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device) #델을 디바이스(GPU/CPU)에 올림.

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]: #["train", "finetune"]
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines

            """
            ctx.cfg[train].optimizer

                betas:
                - 0.9
                - 0.95
                lr: 1.0e-05
                type: AdamW


            ctx.cfg[train].scheduler
                gamma: 0.5
                milestones:
                - 100
                - 150
                type: ''
                warmup_ratio: 0.0


            """
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

        # TODO: the number of batch and epoch is decided by the current mode
        #  and data split, so the number of batch and epoch should be
        #  initialized at the beginning of the routine

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_fit_start_calculate_model_size(self, ctx): #모델 전체 사이즈 측정.
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track model size
            ==================================  ===========================
        """
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return
        if ctx.monitor.total_model_size == 0:
            ctx.monitor.track_model_size(ctx.models) #self.total_model_size에 측정한 값 저장.

    # def _hook_on_epoch_start(self, ctx): #“지금 돌고 있는 데이터 분할(ctx.cur_split, 예: "train", "test")에 맞는 DataLoader 를 ctx에 보관하거나 재설정(reset)해 주는 것.
    #     """
    #     Note:
    #       The modified attributes and according operations are shown below:
    #         ==================================  ===========================
    #         Attribute                           Operation
    #         ==================================  ===========================
    #         ``ctx.{ctx.cur_split}_loader``      Initialize DataLoader
    #         ==================================  ===========================
    #     """
    #     """

    #     WrapDataset → 프레임워크용 래퍼.
    #     어떤 형태의 입력이 와도(dict, 리스트/배열, torch.utils.data.Subset, HF datasets.Dataset 등) **길이(__len__)와 항목 추출(__getitem__)**을 일관되게 제공.

    #     get_dataloader(...)가 이 래퍼 위에서 shuffle/sampler/collate_fn을 표준화해서 쓸 수 있게 도와줌.


    #     get_dataloader → 배치 사이즈, 셔플 등 cfg 기반으로 DataLoader 생성

    #     ReIterator → “다 썼다가 또 next() 호출해도 처음부터 다시” 가능한 반복자(iterator)로 감싸 ctx.train_loader 에 저장

    #     """
    #     # prepare dataloader
    #     if ctx.get("{}_loader".format(ctx.cur_split)) is None:
    #         loader = get_dataloader(
    #             WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
    #             self.cfg, ctx.cur_split)
    #         setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))#ctx.{}_loader를 ReIterator(loader)로 지정!!
    #     elif not isinstance(ctx.get("{}_loader".format(ctx.cur_split)),
    #                         ReIterator):
    #         setattr(ctx, "{}_loader".format(ctx.cur_split),
    #                 ReIterator(ctx.get("{}_loader".format(ctx.cur_split))))
    #     else:#이미 ReIterator라면 → 내부 포인터를 reset(). 내부적으로 처음에 사용하던 DataLoader를 다시 생성하거나,내부 포인터(pointer)를 “처음 배치” 위치로 돌려놓습니다. 그래서 또다시 next()를 호출하면 “첫 번째 배치”부터 다시 돌려받을 수 있게 됩니다.
    #         ctx.get("{}_loader".format(ctx.cur_split)).reset()



    def _hook_on_batch_start_init(self, ctx):
        # LLMTrainer가 accelerate 모드에서 이 훅을 우회하므로,
        # 이 코드는 accelerate 미사용 시에만 호출됩니다.
        # 따라서 원래 코드를 그대로 두어도 괜찮지만,
        # 명확성을 위해 비워두는 것도 좋은 방법입니다.
        # => 충돌을 피하기 위해 비워두는 것을 추천합니다.
        pass



    def _hook_on_batch_forward(self, ctx): # ctx.data_batch에서 (입력, 정답)을 꺼내 디바이스로 이동-> logits = ctx.model(inputs) 수행 (순전파) -> ctx.y_true, ctx.y_prob, ctx.loss_batch, ctx.batch_size에 각각 CtxVar로 저장.
        # 이 정보가 나중에 on_batch_end 훅에서 누적·기록
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch] # [[...inputs...], [...labels...]] 라는 길이 2짜리 리스트. x->[...inputs...], label ->[...labels...].
        pred = ctx.model(x)
        if len(label.size()) == 0: # 0-차원일 때만

            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_flop_count(self, ctx):#한 배치에 대한 모델의 연산량(FLOPs)을 계산해 모니터(ctx.monitor)에 기록합니다. self.cfg.eval.count_flops=FALSE라 사실상 무시해도 됨.
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
        if not isinstance(ctx.monitor, Monitor): #모니터 유효성 검사
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0: #PASS
            # calculate the flops_per_sample
            try:
                x, y = [_.to(ctx.device) for _ in ctx.data_batch]
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(ctx.model, x).total()
                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied "
                        "by internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, "
                        "please customize the count hook")
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except:
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general "
                    "trainer case: "
                    "1) ctx.data_batch = [x, y]; and"
                    "2) the ctx.model takes only x as input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size

    def _hook_on_batch_forward_regularizer(self, ctx): #순전파 손실(ctx.loss_batch)에 정규화 항을 더해, 실제 학습에서 사용할 loss_task 를 만듭니다. 무시해도 됨.
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.loss_regular``                Calculate the regular loss
            ``ctx.loss_task``                   Sum the ``ctx.loss_regular`` \
            and ``ctx.loss``
            ==================================  ===========================
        """
        ctx.loss_regular = CtxVar(
            self.cfg.regularizer.mu * ctx.regularizer(ctx), LIFECYCLE.BATCH) #ctx.regularizer=dummy regularizer. 
        ctx.loss_task = CtxVar(ctx.loss_batch + ctx.loss_regular,
                               LIFECYCLE.BATCH)#gradient 업데이트 되는데 직접적으로 이용되는 loss

    def _hook_on_batch_backward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.optimizer``                   Update by gradient
            ``ctx.loss_task``                   Backward propagation
            ``ctx.scheduler``                   Update by gradient
            ==================================  ===========================
        """
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0: #-1.0이라 False
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()
        if ctx.scheduler is not None:#CN({'__cfg_check_funcs__': [], '__help_info__': {}, 'is_ready_for_run': False, 'type': '', 'warmup_ratio': 0.0})
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx): # 배치 하나가 끝날 때마다 loss_batch * batch_size를 loss_batch_total에 더하고 ys_true, ys_prob 리스트에 예측·실제 레이블을 확장(extend). num_samples에 배치 크기 누적
        #이렇게 모은 값으로 에폭·루틴이 끝난 뒤 평균 loss, accuracy 등을 계산.
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.num_samples``                 Add ``ctx.batch_size``
            ``ctx.loss_batch_total``            Add batch loss
            ``ctx.loss_regular_total``          Add batch regular loss
            ``ctx.ys_true``                     Append ``ctx.y_true``
            ``ctx.ys_prob``                     Append ``ctx.ys_prob``
            ==================================  ===========================
        """
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())

    def _hook_on_fit_end(self, ctx): #모든 배치가 처리된 후, loss_batch_total / num_samples로 평균 loss 계산.  ys_true, ys_prob로 정확도·기타 지표 계산(MetricCalculator 사용). 최종 결과를 ctx.eval_metrics = { ... }에 저장. 
        # evaluate()가 이 딕셔너리를 반환.
        """
        Evaluate metrics.

        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.ys_true``                     Convert to ``numpy.array``
            ``ctx.ys_prob``                     Convert to ``numpy.array``
            ``ctx.monitor``                     Evaluate the results
            ``ctx.eval_metrics``                Get evaluated results from \
            ``ctx.monitor``
            ==================================  ===========================
        """
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx) #loss, avg_loss, acc, total을 기록한 dictionary (MODEL 상관 없음)
        setattr(ctx, 'eval_metrics', results)

    def save_model(self, path, cur_round=-1): #lora 모델 fl 라운드 이름과 같이 저장.
        assert self.ctx.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.ctx.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path): #path로부터 lora 모델 불러와서 broadcast.
        assert self.ctx.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.ctx.device)
            self.ctx.model.load_state_dict(ckpt['model']) #strict=True로 반영. self.ctx.model.state_dict()(active lora adapter만 불러옴.)와의 key 불일치 확인.
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def discharge_model(self): #볼 필요 없어보임. 라운드(혹은 루틴) 끝날 때 모델을 GPU에서 CPU로 내려서 GPU 메모리를 비우는 함수
        """
        Discharge the model from GPU device
        """
        # Avoid memory leak
        if torch is None:
            return

        if not self.cfg.federate.share_local_model and \
                not self.cfg.llm.deepspeed.use: #False라서 안걸림.
            self.ctx.model.to(torch.device("cpu"))
