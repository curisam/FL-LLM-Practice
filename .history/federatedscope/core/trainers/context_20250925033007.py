import logging
import collections

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.model_builder import \
    get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
from federatedscope.core.trainers.enums import MODE
from federatedscope.core.trainers.utils import calculate_batch_epoch_num


from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

logger = logging.getLogger(__name__)

from collections import defaultdict

"""

# —————————————
# (1) CtxVar 클래스 선언
# —————————————
class CtxVar:

    #값(obj)과 이를 언제까지 유지할지(lifecycle)을 함께 저장합니다.
    #lifecycle: "batch", "epoch", "routine", 또는 None
    
    LIFECYCLES = ["batch", "epoch", "routine", None]

    def __init__(self, obj, lifecycle=None):
        assert lifecycle in CtxVar.LIFECYCLES, f"Invalid lifecycle: {lifecycle}"
        self.obj = obj
        self.lifecycle = lifecycle


# —————————————
# (2) LifecycleDict 최소 구현
# —————————————
class LifecycleDict(dict):
    def __init__(self):
        super().__init__()
        # 각 단계별로 지워야 할 키를 모아 놓는 곳
        self.lifecycles = defaultdict(set)

    def __setattr__(self, key, value):
        # CtxVar로 감싸진 변수면 생명주기 기록
        if isinstance(value, CtxVar):
            self.lifecycles[value.lifecycle].add(key)
            super().__setitem__(key, value.obj)
        else:
            super().__setitem__(key, value)

    def clear(self, lifecycle):
        
        해당 단계(lifecycle)에 등록된 키들만 한꺼번에 삭제합니다.
        lifecycle: "batch", "epoch", "routine", 또는 None
        
        for key in list(self.lifecycles[lifecycle]):
            if key in self:
                del self[key]
            self.lifecycles[lifecycle].remove(key)


# —————————————
# (3) 예시 사용
# —————————————
if __name__ == "__main__":
    ctx = LifecycleDict()

    # (a) 배치 단계에서만 유지할 변수
    ctx.temp_loss = CtxVar(0.42, lifecycle="batch")#__setattr__ 적용됨.


    print("After setting temp_loss:")
    print("  ctx =", dict(ctx))
    print("  lifecycles =", dict(ctx.lifecycles))

    # → ctx = {'temp_loss': 0.42}
    # → lifecycles = {'batch': {'temp_loss'}}

    # (b) 배치 종료 시점에 자동 정리
    ctx.clear("batch")
    print("\nAfter ctx.clear('batch'):")
    print("  ctx =", dict(ctx))
    print("  lifecycles =", dict(ctx.lifecycles))

    # → ctx = {}
    # → lifecycles = {'batch': set()}

    # (c) 전체 루틴 동안 유지할 변수
    ctx.all_preds = CtxVar([0.1, 0.9], lifecycle="routine")
    print("\nAfter setting all_preds:")
    print("  ctx =", dict(ctx))
    print("  lifecycles =", dict(ctx.lifecycles))
    # → ctx = {'all_preds': [0.1, 0.9]}
    # → lifecycles = {'routine': {'all_preds'}}

    # (d) 루틴 종료 시점에 정리
    ctx.clear("routine")
    print("\nAfter ctx.clear('routine'):")
    print("  ctx =", dict(ctx))
    print("  lifecycles =", dict(ctx.lifecycles))
    # → ctx = {}
    # → lifecycles = {'routine': set()}


"""



class LifecycleDict(dict):

    #평범한 파이썬 dict를 살짝 업그레이드한 클래스입니다.
    #일반 dict처럼 키-값 쌍을 저장하지만, “이 값은 언제까지 쓸 건지”를 표시해 둘 수 있는 기능이 추가돼 있어요.

    """A customized dict that provides lifecycle management
    Arguments:
        init_dict: initialized dict
    """
    __delattr__ = dict.__delitem__

    def __getattr__(self, item): 
        try:
            return self[item] # ctx.foo → ctx['foo']
        except KeyError:
            raise AttributeError("Attribute {} is not found".format(item))

    def __init__(self, init_dict=None):
        if init_dict is not None:
            super(LifecycleDict, self).__init__(init_dict)
        # {} (모든 키에 대해 빈 집합; 실제론 {"batch":set(),"epoch":set(),"routine":set()} 처럼 동작)

        self.lifecycles = collections.defaultdict(set) 

    def __setattr__(self, key, value):
        # ① CtxVar로 감싸진 변수는 lifecycle별 “나중에 지워야 할 키”로 표시
        if isinstance(value, CtxVar):

            #e.g. value=CtxVar(123, lifecycle="batch") value.obj=123, value.lifecycle="batch"
            self.lifecycles[value.lifecycle].add(key)  #value.lifecycle [batch, epoch, routine]

            #self.lifecycles['batch']=key.

            # ② 실제 값은 dictionary에 저장
            super(LifecycleDict, self).__setitem__(key, value.obj)


            ##self.lifecycles['batch']]key]=123
        else:
            # 일반 값은 그냥 dict에 저장
            super(LifecycleDict, self).__setitem__(key, value)

    def clear(self, lifecycle):
        # 해당 생명주기에 표시된 키 모두 삭제
        keys = list(self.lifecycles[lifecycle])
        for key in keys:
            if key in self:
                del self[key] # ctx.foo 삭제
            self.lifecycles[lifecycle].remove(key) #ctx.lifecycles[lifecycle]에서 key("foo")도 삭제


class Context(LifecycleDict): #LifecycleDict를 상속받아 모델, 설정(cfg), 데이터, 장치(device), 훈련·평가 모드·스플릿 등 FederatedScope 의 학습·평가 파이프라인 전반에 필요한 상태를 한곳에 묶어 두는 저장소
    """
    Record and pass variables among different hook functions.

    Arguments:
        model: training model
        cfg: config
        data (dict): a dict contains train/val/test dataset or dataloader
        device: running device
        init_dict (dict): a dict used to initialize the instance of Context
        init_attr (bool): if set up the static variables
    Note:
        - The variables within an instance of class `Context` can be set/get \
        as an attribute.
        ```
        ctx.${NAME_VARIABLE} = ${VALUE_VARIABLE}
        ```
        where ``${NAME_VARIABLE}`` and ``${VALUE_VARIABLE}``
        is the name and value of the variable.

        - To achieve automatically lifecycle management, you can \
        wrap the variable with ``CtxVar`` and a lifecycle parameter \
        as follows
        ```
        ctx.${NAME_VARIABLE} = CtxVar(${VALUE_VARIABLE}, ${LIFECYCLE})
        ```
        The parameter ``${LIFECYCLE}`` can be chosen from \
        ``LIFECYCLE.BATCH``, ``LIFECYCLE.EPOCH`` and ``LIFECYCLE.ROUTINE``. \
        Then the variable ``ctx.${NAME_VARIABLE}`` will be deleted at \
        the end of the corresponding stage
            - ``LIFECYCLE.BATCH``: the variables will \
            be deleted after running a batch
            - ``LIFECYCLE.EPOCH``: the variables will be \
            deleted after running a epoch
            - ``LIFECYCLE.ROUTINE``: the variables will be \
            deleted after running a routine
        More details please refer to our
        [tutorial](https://federatedscope.io/docs/trainer/).

        We classify and show the default attributes below:

        Data-related attributes
          - ``ctx.data``: the raw data (not split) the trainer holds
          - ``ctx.num_samples``: the number of samples used in training
          - ``ctx.train_data``, ``ctx.val_data``, ``ctx.test_data``: the \
          split data the trainer holds
          - ``ctx.train_loader``, ``ctx.val_loader``, ``ctx.test_loader``: \
          the DataLoader of each split data
          - ``ctx.num_train_data``, ``ctx.num_val_data``, \
          ``ctx.num_test_data``: the number of samples of  the split data \
          Model-related attributes
          - ``ctx.model``: the model used
          - ``ctx.models``: the multi models if use
          - ``ctx.mirrored_models``: the mirrored models
          - ``ctx.trainable_para_names``: the trainable parameter names of \
          the model
        Optimizer-related attributes
          - ``ctx.optimizer``: see ``torch.optim``
          - ``ctx.scheduler``: decays the learning rate of each parameter group
          - ``ctx.criterion``: loss/criterion function
          - ``ctx.regularizer``: regular terms
          - ``ctx.grad_clip``: gradient clipping
        Mode-related attributes
          - ``ctx.cur_mode``: mode of trainer, which is one of ``['train', \
          'val', 'test']``
          - ``ctx.mode_stack``: stack of mode, only used for switching mode
          - ``ctx.cur_split``: split of data, which is one of ``['train', \
          'val', 'test']`` (Note: use ``train`` data in ``test`` mode is \
          allowed)
          - ``ctx.split_stack``: stack of split, only used for switching data \
          split
        Metric-related attributes
          - ``ctx.loss_batch_total``: Loss of current batch
          - ``ctx.loss_regular_total``: Loss of regular term
          - ``ctx.y_true``:  true label of batch data
          - ``ctx.y_prob``: output of the model with batch data as input
          - ``ctx.ys_true``: true label of data
          - ``ctx.ys_prob``: output of the model
          - ``ctx.eval_metrics``: evaluation metrics calculated by \
          ``ctx.monitor``
          - ``ctx.monitor``: used for monitor trainer's behavior and statistics
        Other (statistics) attributes (@property, query from ``cfg`` if not \
        set)
          - ``ctx.cfg``: configuration of FL course
          - ``ctx.device``: current device, such as ``cpu`` and ``gpu0``.
          - ``ctx.num_train_batch_last_epoch``, \
          ``ctx.num_total_train_batch``: the number of batch
          - ``ctx.num_train_epoch``, ``ctx.num_val_epoch``, \
          ``ctx.num_test_epoch``: the number of epoch in each data split
          - ``ctx.num_train_batch``, ``ctx.num_val_batch``, \
          ``ctx.num_test_batch``: the number of batch in each data split
    """
    def __init__(self, model, cfg, data=None, device=None):
        super(Context, self).__init__({})
        # 1) 기본 입력 저장
        self.cfg = cfg
        self.model = model
        self.data = data  ##ClientData 클래스 형태. {'train': loader, 'val':…, 'test':…, 'train_datase': 데이터셋, ... }의 형태.
        self.device = device

        # 2) mode / split 전환 스택

        """
        “지금 훈련 중인지 평가 중인지”
        “훈련일 때는 train 데이터, 평가일 때는 val 또는 test 데이터를”
        자동으로 관리.
        """

        ####### mode ####### ctx.track_mode('train') → ctx.cur_mode = 'train' & 모델을 model.train() 모드로 전환
        self.cur_mode = None # 현재 실행 중인 모드 ('train', 'val', 'test'), 내부 훅이나 로직에서 “지금 뭐 하는 중인가?”를 분기 처리할 때 사용

        ####### split ####### ctx.track_split('train') → ctx.cur_split = 'train' & ctx.train_loader를 사용하겠다고 표시
        self.cur_split = None #지금 쓰고 있는 데이터 분할 이름 ('train'/'val'/'test'), 평가 시 “어느 split을 돌고 있나?”를 기록



        #### 루틴이 끝나면 ctx.reset_mode()와 ctx.reset_split() 으로 원래 모드·분할로 되돌려 준다.


        self.mode_stack = list() # 모드를 임시로 바꿔야 할 때(중첩 평가 등) 이전 모드를 저장하는 스택,  push/pop 형태로 들어갔다 나오는 구조


        self.split_stack = list() # split도 중첩 변경이 필요할 때 이전 split을 저장하는 스택


        # 3) LifecycleDict.lifecycles 초기화 (딕셔너리 리셋)
        self.lifecycles = collections.defaultdict(set)

        # Setup optimize-related context variable # 4) 백엔드(torch vs tf)별 옵티마이저·criterion·regularizer·grad_clip 설정
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model) #일단 모든 파라미터 다 불러옴. base, 모든 adapter들.
            # TODO: make `criterion` and `regularizer` @property and cached
            #  to compare whether changes happen
            self.criterion = get_criterion(self.cfg.criterion.type,
                                           self.device) #CE Loss
            self.regularizer = get_regularizer(self.cfg.regularizer.type) #self.cfg.regularizer.type=''
            self.grad_clip = self.cfg.grad.grad_clip #-1.0
            if self.cfg.federate.process_num > 1:
                self.model.to(self.device)
        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

    # Train related property, query from `cfg` if not set
    @property #함수 호출 없이 attribvute 같이 사용 가능. class.num_train_batch() -> class.num_train_batch 와 같이 적용되는 효과.
    def num_train_batch(self):
        if self.get('num_train_batch'):
            return self.get('num_train_batch')
        
        return self._calculate_batch_epoch_num(mode='train')[0] #일반적인 epoch당 batch 개수. batch 버전에서는 기존과 iteration 수의 min으로 지정.

    @property
    def num_train_batch_last_epoch(self):
        if self.get('num_train_batch_last_epoch'):
            return self.get('num_train_batch_last_epoch')
        return self._calculate_batch_epoch_num(mode='train')[1] #마지막 epoch에서의 batch 개수


    @property
    def num_train_epoch(self):
        if self.get('num_train_epoch'):
            return self.get('num_train_epoch')
        return self._calculate_batch_epoch_num(mode='train')[2] #train 동안 수행되는 총 epoch 수

    @property
    def num_total_train_batch(self):
        if self.get('num_total_train_batch'):
            return self.get('num_total_train_batch')
        return self._calculate_batch_epoch_num(mode='train')[3] #train 동안 수행되는 총 iteration 수

    # Val related property, query from `cfg` if not set
    @property
    def num_val_batch(self):
        if self.get('num_val_batch'):
            return self.get('num_val_batch')
        return self._calculate_batch_epoch_num(mode='val')[0]

    @property
    def num_val_epoch(self):
        if self.get('num_val_epoch'):
            return self.get('num_val_epoch')
        return self._calculate_batch_epoch_num(mode='val')[2]

    # Test related property, query from `cfg` if not set
    @property
    def num_test_batch(self):
        if self.get('num_test_batch'):
            return self.get('num_test_batch')
        return self._calculate_batch_epoch_num(mode='test')[0]

    @property
    def num_test_epoch(self):
        if self.get('num_test_epoch'):
            return self.get('num_test_epoch')
        return self._calculate_batch_epoch_num(mode='test')[2]

    # def _calculate_batch_epoch_num(self, mode='train'):#“local update step” 수, “gradient accumulation” 수, “batch_or_epoch” 설정을 반영. 리턴값은 (한 에폭 배치 수, 마지막 에폭 배치 수, 에폭 수, 전체 배치 수)
    #     #Val/Test: 에폭 수 고정 1, 배치 수만 데이터 크기 기준으로 계산. num_batch_last_epoch, num_total_batch = None, None으로 된다.
    #     # 1) 사용할 split 결정
    #     if self.cur_split is None:
    #         logger.warning(
    #             f'cur_split `{self.cur_split}` not found in data_split, '
    #             f'will use `train` split to calculate `ctx.var`.')
    #         cur_split = 'train'
    #     else:
    #         cur_split = self.cur_split

    #     num_batch_last_epoch, num_total_batch = None, None

    #     #2) train/finetune 모드인 경우. 여기서의 batch size는 micro batch size의미.
    #     if mode in ['train', 'finetune']: #self.cfg.grad.grad_accum_count 이거는 언제나 1인듯.
    #         #self.get(f'num_{cur_split}_data')는 torch_trainer.py의 parse_data()로부터 계산되어서 쪼개지기 전 dataset 자체의 순수 갯수임. ddp의 sharding 이랑 무관. 합쳐진 전부의 갯수.
 
    #         num_batch, num_batch_last_epoch, num_epoch, num_total_batch = \
    #             calculate_batch_epoch_num(
    #                 self.cfg.train.local_update_steps *
    #                 self.cfg.grad.grad_accum_count,
    #                 self.cfg.train.batch_or_epoch,
    #                 self.get(f'num_{cur_split}_data'),
    #                 self.cfg.dataloader.batch_size,
    #                 self.cfg.dataloader.drop_last)# (num_batch_per_epoch, num_batch_last_epoch, num_epoch, num_total_batch)를 반환, num_batch_last_epoch는 마지막 epoch에서 iteration 수
    #         #100 batch 7916 2 False
    #         # print(self.cfg.train.local_update_steps *self.cfg.grad.grad_accum_count, self.cfg.train.batch_or_epoch, self.get(f'num_{cur_split}_data'), self.cfg.dataloader.batch_size, self.cfg.dataloader.drop_last)
    #         #(100, 100, 1, 100)
    #         # print(num_batch, num_batch_last_epoch, num_epoch, num_total_batch) 

    #     # 3) val/test 모드인 경우
    #     elif mode in ['val', 'test']: #num_batch_last_epoch, num_total_batch = None, None로 반환.
    #         num_epoch = 1
    #         # 전체 데이터 크기 나누기 배치 사이즈 (+ drop_last 여부). 한 epoch 돌 동안 전체 batch 수.
    #         num_batch = self.get(f'num_{cur_split}_data') // self.cfg.dataloader.batch_size + int(not self.cfg.dataloader.drop_last and bool(self.get(f'num_{cur_split}_data') % self.cfg.dataloader.batch_size))
    #     else:
    #         raise ValueError(f'Invalid mode {mode}.')
        


        
    #     return num_batch, num_batch_last_epoch, num_epoch, num_total_batch

    def _calculate_batch_epoch_num(self, mode='train'):
        # 0) mode -> split 매핑 (cur_split에 의존 X)
        if mode in ('train', 'finetune'):
            split = 'train'
        elif mode in ('val', 'test'):
            split = mode
        else:
            raise ValueError(f'Invalid mode {mode}.')

        # 공통 입력값
        data_size = int(self.get(f'num_{split}_data'))
        bs        = int(self.cfg.dataloader.batch_size)
        drop_last = bool(self.cfg.dataloader.drop_last)

        if mode in ('train', 'finetune'):
            lus = int(self.cfg.train.local_update_steps)
            gac = int(self.cfg.grad.grad_accum_count)

            num_batch, num_batch_last_epoch, num_epoch, num_total_batch = calculate_batch_epoch_num(
                lus * gac,
                self.cfg.train.batch_or_epoch,
                data_size,
                bs,
                drop_last
            )

            # ✅ DEBUG 레벨 로깅 (필요할 때만 켜세요)
            # logger.info(
            #     "[calc/train] split=%s size=%d bs=%d drop_last=%s lus=%d gac=%d "
            #     "=> num_batch=%d, last_epoch=%s, num_epoch=%d, total=%s",
            #     split, data_size, bs, drop_last, lus, gac,
            #     num_batch, str(num_batch_last_epoch), num_epoch, str(num_total_batch)
            # )
            return num_batch, num_batch_last_epoch, num_epoch, num_total_batch

        else:  # val/test
            num_epoch = 1
            num_batch = data_size // bs + int(not drop_last and bool(data_size % bs))

            # logger.info(
            #     "[calc/%s] split=%s size=%d bs=%d drop_last=%s => num_batch=%d, num_epoch=%d",
            #     mode, split, data_size, bs, drop_last, num_batch, num_epoch
            # )
            return num_batch, None, num_epoch, None




    def track_mode(self, mode): # 지금부터 “mode” (훈련인지 평가인지) 를 바꾼다고 스택에 기록하고, cur_mode 에 반영한 뒤, 실제 모델을 .train() 또는 .eval() 상태로 전환합니다.
        self.mode_stack.append(mode)
        self.cur_mode = self.mode_stack[-1]
        self.change_mode(self.cur_mode)

    def reset_mode(self): #모드 스택에서 가장 마지막에 추가했던 mode를 꺼내(pop), cur_mode 를 그 이전 상태로 되돌린 뒤, change_mode 를 통해 모델 상태도 함께 복원합니다.
        self.mode_stack.pop()
        self.cur_mode = self.mode_stack[-1] if len(
            self.mode_stack) != 0 else None
        if len(self.mode_stack) != 0:
            self.change_mode(self.cur_mode)

    def reset_split(self): #원래 split으로 돌려주는 것. dataset 관련
        self.split_stack.pop()
        self.cur_split = self.split_stack[-1] if \
            len(self.split_stack) != 0 else None


    def change_mode(self, mode): #model.train()  or model.eval()할 지 여부
        # change state
        if self.cfg.backend == 'torch':
            getattr(
                self.model, 'train'
                if mode == MODE.TRAIN or mode == MODE.FINETUNE else 'eval')()
        else:
            pass

    def track_split(self, dataset):
        # stack-style to enable mixture usage such as evaluation on train
        # dataset
        self.split_stack.append(dataset)
        self.cur_split = self.split_stack[-1]
        

    def check_split(self, target_split_name, skip=False): #ctx.check_split("test", skip=True): test_loader가 없으면 평가를 건너뛰게 한다. 
        if self.get(f"{target_split_name}_data") is None and self.get(
                f"{target_split_name}_loader") is None:
            if skip:
                logger.warning(
                    f"No {target_split_name}_data or"
                    f" {target_split_name}_loader in the trainer, "
                    f"will skip evaluation."
                    f"If this is not the case you want, please check "
                    f"whether there is typo for the name")
                return False
            else:
                raise ValueError(f"No {target_split_name}_data or"
                                    f" {target_split_name}_loader in the trainer")
        else:
            return True

        
    #딕셔너리로 받은 값을 한 번에 ctx에 넣는다.
    def merge_from_dict(self, other_dict):
        for key, value in other_dict.items():
            setattr(self, key, value)


class CtxVar(object):
    """
    Basic variable class

    Arguments:
        lifecycle: specific lifecycle of the attribute
    """

    """
    훈련·평가 중에는 임시로 쓰는 변수가 많습니다.

        "batch": 배치 한 번 처리할 때만 쓰는 손실값

        "epoch": 한 에폭이 끝날 때까지만 모아두는 예측 결과

        "routine": 전체 평가 루틴 동안만 유지되는 메트릭

    이걸 일일이 “끝나면 삭제해 주세요”라고 쓰기 번거로우니,
    CtxVar에 **값(value)**과 **언제 지울지(lifecycle)**를 붙여서 한 번에 관리하는 겁니다.
    """

    LIFECYCLES = ["batch", "epoch", "routine", None]

    def __init__(self, obj, lifecycle=None):
        assert lifecycle in CtxVar.LIFECYCLES #["batch", "epoch", "routine", None]
        self.obj = obj # 실제 저장할 값
        self.lifecycle = lifecycle # "batch"/"epoch"/"routine" 중 하나

        #obj: 우리가 보관하고 싶은 실제 값 (예: 손실값, 리스트, 텐서 등)
        #lifecycle:
        #   "batch" → 배치가 끝나면 삭제
        #   "epoch" → 에폭이 끝나면 삭제
        #   "routine"→ 전체 루틴(evaluate/train)이 끝나면 삭제
        #   None → 자동 삭제 대상이 아님


        ##########  예사 ##############
        """ 
        # 배치 하나가 끝나면 사라져도 되는 임시 손실값
        ctx.temp_loss = CtxVar(0.0, lifecycle="batch")

        # 전체 평가가 끝날 때까지 모아둘 true/pred 리스트
        ctx.ys_true = CtxVar([], lifecycle="routine")
        ctx.ys_pred = CtxVar([], lifecycle="routine")
        """

def lifecycle(lifecycle):
    """
    Manage the lifecycle of the variables within context, \
    and blind these operations from user.

    Arguments:
        lifecycle: the type of lifecycle, choose from "batch/epoch/routine"
    """
    if lifecycle == "routine":
        # 전체 루틴(evaluate/train) 메서드에 붙일 때

        def decorate(func):
            def wrapper(self, mode, hooks_set, dataset_name=None):
                # 1) 루틴 시작 전: 모드(mode)와 split(데이터 분할) 기록
                self.ctx.track_mode(mode)
                self.ctx.track_split(dataset_name or mode)
                
                # 2) 실제 루틴 실행
                res = func(self, mode, hooks_set, dataset_name)
                
                
                # 3) routine 생명주기로 표시된 모든 변수 삭제
                # Clear the variables at the end of lifecycles
                self.ctx.clear(lifecycle)
                
                # 4) 모드와 split 스택 복원
                # rollback the model and data_split
                self.ctx.reset_mode()
                self.ctx.reset_split()
                
                # 5) 모델을 CPU로 내리기(메모리 절약)
                # Move the model into CPU to avoid memory leak
                self.discharge_model()

                return res

            return wrapper
    else:
        # batch 또는 epoch 단계 메서드에 붙일 때
        def decorate(func):
            def wrapper(self, *args, **kwargs):
                # 1) 실제 배치/에폭 수행
                res = func(self, *args, **kwargs)

                # Clear the variables at the end of lifecycles
                # 2) 해당 단계 생명주기로 표시된 변수 삭제
                self.ctx.clear(lifecycle)
                return res

            return wrapper

    return decorate
