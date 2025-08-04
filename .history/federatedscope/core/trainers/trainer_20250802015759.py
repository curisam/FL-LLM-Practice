import collections
import copy
import logging
import os
import gc         # <--- 확인 또는 추가
import torch      # <--- 확인 또는 추가 (이미 있을 가능성 높음)
import objgraph   # <--- 확인 또는 추가
import random # 파일 이름 중복 방지를 위해 추가

from federatedscope.core.trainers.base_trainer import BaseTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.trainers.utils import format_log_hooks, filter_by_specified_keywords
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle


from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.ReIterator import ReIterator

from accelerate.utils import gather_object

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
        Register, organize and run the train/test/val procedures
    """

    HOOK_TRIGGER = [
        "on_fit_start", "on_epoch_start", "on_batch_start", "on_batch_forward",
        "on_batch_backward", "on_batch_end", "on_epoch_end", "on_fit_end"
    ]

    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        self._cfg = config

        self.ctx = Context(model, self.cfg, data, device) #data는 ClientData 클래스 형태.

        # Parse data and setup init vars in ctx
        self._setup_data_related_var_in_ctx(self.ctx)

        assert monitor is not None, \
            f"Monitor not found in trainer with class {type(self)}"
        self.ctx.monitor = monitor
        # the "model_nums", and "models" are used for multi-model case and
        # model size calculation
        self.model_nums = 1
        self.ctx.models = [model]
        # "mirrored_models": whether the internal multi-models adopt the
        # same architects and almost the same behaviors,
        # which is used to simply the flops, model size calculation
        self.ctx.mirrored_models = False

        # Atomic operation during training/evaluation
        self.hooks_in_train = collections.defaultdict(list) #키(key)→리스트(value) 매핑을 관리하는 dict. 존재하지 않는 키 접근 시 자동으로 빈 리스트를 만들어 주는 장점 있음.

        # By default, use the same trigger keys
        self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)
        self.hooks_in_ft = copy.deepcopy(self.hooks_in_train)

        # register necessary hooks into self.hooks_in_train and
        # self.hooks_in_eval
        if not only_for_eval: #eval 만 할거면 train pass
            self.register_default_hooks_train()

        if self.cfg.finetune.before_eval: #eval 이전에 fine tunning 할 것인지 여부. False
            self.register_default_hooks_ft()

         # 평가용 훅들(on_fit_start, on_batch_start, on_batch_end, on_fit_end 등)
        self.register_default_hooks_eval()

        if self.cfg.federate.mode == 'distributed':#standalone
            self.print_trainer_meta_info()
        else:
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            pass

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, new_cfg):
        self._cfg = new_cfg
        self.ctx.cfg = new_cfg
        self._setup_data_related_var_in_ctx(self.ctx)

    def parse_data(self, data):
        """
        Populate ``${split}_data``, ``${split}_loader`` and \
        ``num_${split}_data`` for different data splits
        """
        raise NotImplementedError

    def setup_data(self, ctx):
        """
        Initialization data by ``cfg``.
        """
        pass

    def _setup_data_related_var_in_ctx(self, ctx):
        """
        Populate ``${split}_data``, ``${split}_loader`` and \
        ``num_${split}_data`` for different data splits, and setup init var \
        in ctx.
        """
        self.setup_data(ctx)
        init_dict = self.parse_data(ctx.data)
        ctx.merge_from_dict(init_dict) #이 떄 ctx는 train_loader, num_train_data 같은 attribute가 생김.

    def register_default_hooks_train(self):
        pass

    def register_default_hooks_eval(self):
        pass

    def register_default_hooks_ft(self):
        pass

    def reset_hook_in_train(self, target_trigger, target_hook_name=None):
        hooks_dict = self.hooks_in_train
        del_one_hook_idx = self._reset_hook_in_trigger(hooks_dict,
                                                       target_hook_name,
                                                       target_trigger)
        return del_one_hook_idx

    def reset_hook_in_eval(self, target_trigger, target_hook_name=None):
        hooks_dict = self.hooks_in_eval
        del_one_hook_idx = self._reset_hook_in_trigger(hooks_dict,
                                                       target_hook_name,
                                                       target_trigger)
        return del_one_hook_idx

    def replace_hook_in_train(self, new_hook, target_trigger,
                              target_hook_name):
        del_one_hook_idx = self.reset_hook_in_train(
            target_trigger=target_trigger, target_hook_name=target_hook_name)
        self.register_hook_in_train(new_hook=new_hook,
                                    trigger=target_trigger,
                                    insert_pos=del_one_hook_idx)

    def replace_hook_in_eval(self, new_hook, target_trigger, target_hook_name):
        del_one_hook_idx = self.reset_hook_in_eval(
            target_trigger=target_trigger, target_hook_name=target_hook_name)
        self.register_hook_in_eval(new_hook=new_hook,
                                   trigger=target_trigger,
                                   insert_pos=del_one_hook_idx)
        
        

    def _reset_hook_in_trigger(self, hooks_dict, target_hook_name,
                               target_trigger):
        # clean/delete existing hooks for a specific trigger,
        # if target_hook_name given, will clean only the specific one;
        # otherwise, will clean all hooks for the trigger.

        #target_trigger: "on_batch_end" 같은, 훅 리스트를 지울 키

        #target_hook_name: 삭제할 훅의 함수 이름(funcB.__name__ == 'funcB')
        ####None 이면 “모두 삭제”

        #3. 반환값
        # -1: 전체 훅 리스트를 비웠을 때

        # >=0: 삭제된 훅이 원래 차지하던 인덱스

        # None: (사실 논리상 잘 나오진 않지만) 훅 이름을 지정했는데 찾지 못했을 때



        """
        HOOK_TRIGGER = [
            "on_fit_start", "on_epoch_start", "on_batch_start", "on_batch_forward",
            "on_batch_backward", "on_batch_end", "on_epoch_end", "on_fit_end"
        ]
        """

        assert target_trigger in self.HOOK_TRIGGER, \
            f"Got {target_trigger} as hook trigger, you should specify a " \
            f"string within {self.HOOK_TRIGGER}."
        del_one_hook_idx = None
        if target_hook_name is None:
            hooks_dict[target_trigger] = []
            del_one_hook_idx = -1  # -1 indicates del the whole list
        else:
            for hook_idx in range(len(hooks_dict[target_trigger])):
                if target_hook_name == hooks_dict[target_trigger][
                        hook_idx].__name__:
                    del_one = hooks_dict[target_trigger].pop(hook_idx)
                    logger.info(f"Remove the hook `{del_one.__name__}` from "
                                f"hooks_set at trigger `{target_trigger}`")
                    del_one_hook_idx = hook_idx
                    break
            if del_one_hook_idx is None:
                logger.warning(
                    f"In hook del procedure, can't find the target hook "
                    f"named {target_hook_name}")
        return del_one_hook_idx

    def register_hook_in_train(self,
                               new_hook, #new_hook: 추가할 함수
                               trigger,#trigger: 이벤트 이름 (예: "on_fit_start")
                               insert_pos=None,
                               base_hook=None,
                               insert_mode="before"):
        hooks_dict = self.hooks_in_train
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def register_hook_in_ft(self,
                            new_hook,
                            trigger,
                            insert_pos=None,
                            base_hook=None,
                            insert_mode="before"):
        hooks_dict = self.hooks_in_ft
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)

    def register_hook_in_eval(self,
                              new_hook, #new_hook: 추가할 함수
                              trigger, #trigger: 이벤트 이름 (예: "on_fit_start")
                              insert_pos=None,
                              base_hook=None,
                              insert_mode="before"):


        hooks_dict = self.hooks_in_eval #hooks_dict: 실제 훅 저장소 (defaultdict(list))
        self._register_hook(base_hook, hooks_dict, insert_mode, insert_pos,
                            new_hook, trigger)#hook_dict, new_hook, trigger만 실제로 보면 됨.

    def _register_hook(self, base_hook, hooks_dict, insert_mode, insert_pos,
                       new_hook, trigger): #hooks_dict, new_hook, trigger 만 쓰임.
        
        # 1) trigger 유효성 검사
        assert trigger in self.HOOK_TRIGGER, \
            f"Got {trigger} as hook trigger, you should specify a string " \
            f"within {self.HOOK_TRIGGER}."
        # parse the insertion position
        # 2) 대상 리스트 가져오기 (없으면 빈 리스트 자동 생성)
        target_hook_set = hooks_dict[trigger]


        # 3) insert_pos 계산
        if insert_pos is not None: #FALSE
            assert (insert_pos == -1) or (insert_pos == len(target_hook_set)
                                          == 0) or \
                   (0 <= insert_pos <= (len(target_hook_set))), \
                   f"Got {insert_pos} as insert pos, you should specify a " \
                   f"integer (1) =-1 " \
                   f"or (2) =0 for null target_hook_set;" \
                   f"or (3) within [0, {(len(target_hook_set))}]."
        elif base_hook is not None: #FALSE
            base_hook_pos = target_hook_set.index(base_hook)
            insert_pos = base_hook_pos - 1 if insert_mode == "before" else \
                base_hook_pos + 1
            # bounding the insert_pos in rational range
            insert_pos = 0 if insert_pos < 0 else insert_pos
            insert_pos = -1 if insert_pos > len(
                target_hook_set) else insert_pos
        else:# 아무 옵션 없으면 뒤에 붙이기
            insert_pos = -1  # By default, the new hook is called finally
        
        
        # register the new hook
        # 4) 실제 등록: append vs insert
        if insert_pos == -1: #일반적으로 이거 적용.
            hooks_dict[trigger].append(new_hook)
        else: #insert_pos에 register!!
            hooks_dict[trigger].insert(insert_pos, new_hook)



    @use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train
        self.ctx.check_split(target_data_split_name)
        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        return num_samples, self.get_model_para(), self.ctx.eval_metrics






    def evaluate(self, target_data_split_name="test", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_eval
        if self.ctx.check_split(target_data_split_name, skip=True):
            self._run_routine(MODE.TEST, hooks_set, target_data_split_name)
        else:
            self.ctx.eval_metrics = dict()
        return self.ctx.eval_metrics





    def finetune(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_ft

        self.ctx.check_split(target_data_split_name)

        self._run_routine(MODE.FINETUNE, hooks_set, target_data_split_name)


    # @lifecycle(LIFECYCLE.ROUTINE)
    # def _run_routine(self, mode, hooks_set, dataset_name=None):
    #     self.ctx.track_mode(mode)
    #     self.ctx.track_split(dataset_name or mode)
    #     self.ctx.eval_metrics = {}

    #     # 1) on_fit_start
    #     for hook in hooks_set["on_fit_start"]:
    #         hook(self.ctx)

    #     # 2) epoch/batch 루프
    #     self._run_epoch(hooks_set)

    #     # 3) 로컬 결과 계산 (집계 전에)
    #     split = self.ctx.cur_split
    #     local_results = {}
    #     num_samples = self.ctx.get(f'num_samples_{split}', 0)

    #     if num_samples > 0:
    #         # train, val, test 모두 누적된 값으로 직접 계산
    #         loss_total = self.ctx.get(f'loss_total_{split}', 0.0)
    #         correct = self.ctx.get(f'correct_{split}', 0)
    #         local_results = {
    #             f'{split}_total': num_samples,
    #             f'{split}_loss': loss_total,
    #             f'{split}_avg_loss': loss_total / num_samples,
    #             f'{split}_acc': correct / num_samples,
    #         }
            


    #     # 분산 환경 통합 로직
    #     using_accelerate = (
    #         hasattr(self, 'accelerator')
    #         and self.accelerator is not None
    #         and getattr(self.accelerator, 'num_processes', 1) > 1
    #     )

    #     if using_accelerate:
    #         try:
    #             from accelerate.utils import gather_object
    #             gathered = gather_object(local_results)  # 모든 프로세스에서 호출됨

    #             final_results = {}
    #             if gathered and isinstance(gathered[0], dict):
    #                 total = sum(r.get(f'{split}_total', 0) for r in gathered)
    #                 if total > 0:
    #                     total_loss = sum(r.get(f'{split}_loss', 0) for r in gathered)
    #                     wacc = sum(r.get(f'{split}_acc', 0) * r.get(f'{split}_total', 0) for r in gathered)
    #                     final_results = {
    #                         f'{split}_total': total,
    #                         f'{split}_loss': total_loss,
    #                         f'{split}_avg_loss': total_loss / total,
    #                         f'{split}_acc': wacc / total,
    #                     }

    #             # 집계가 비었으면 로컬로 폴백
    #             self.ctx.eval_metrics = final_results if final_results else local_results
    #         except Exception as e:
    #             # 문제가 생기면 로컬값으로 폴백
    #             logger.warning(f"[aggregate warn] gather_object failed: {e}")
    #             self.ctx.eval_metrics = local_results
    #     else:
    #         self.ctx.eval_metrics = local_results

    #     # 5) on_fit_end (여기서 메모리 정리/accelerator 삭제 OK)
    #     for hook in hooks_set["on_fit_end"]:
    #         hook(self.ctx)


    #     return self.ctx.num_samples     


    @lifecycle(LIFECYCLE.ROUTINE)
    def _run_routine(self, mode, hooks_set, dataset_name=None):
        # 0) 모드/스플릿 트래킹 및 초기화
        self.ctx.track_mode(mode)
        self.ctx.track_split(dataset_name or mode)
        self.ctx.eval_metrics = {}

        # 1) on_fit_start 훅
        for hook in hooks_set["on_fit_start"]:
            hook(self.ctx)

        # 2) epoch/batch 루프
        self._run_epoch(hooks_set)

        # 3) 로컬 결과 계산 (집계 전에 먼저 계산)
        split = self.ctx.cur_split
        local_results = {}
        num_samples = self.ctx.get(f'num_samples_{split}', 0)
        loss_total  = self.ctx.get(f'loss_total_{split}', 0.0)
        correct     = self.ctx.get(f'correct_{split}', 0)

        if num_samples > 0:
            local_results = {
                f'{split}_total': num_samples,
                f'{split}_loss':  loss_total,
                f'{split}_avg_loss': loss_total / max(1, num_samples),
                f'{split}_acc':   correct / max(1, num_samples),
            }

        # 4) ✅ on_fit_end "이전"에 분산 집계(reduce) 수행
        using_accelerate = (
            hasattr(self, 'accelerator')
            and self.accelerator is not None
            and getattr(self.accelerator, 'num_processes', 1) > 1
        )

        # --- ⬇️ 이 디버그 로그를 "여기"에 넣으세요 ⬇️ ---
        logger.info(
            f"[agg debug] using_accel={using_accelerate}, "
            f"world={getattr(self.accelerator,'num_processes',1) if hasattr(self,'accelerator') else 'NA'}, "
            f"rank={getattr(self.accelerator,'process_index',-1) if hasattr(self,'accelerator') else 'NA'}, "
            f"local_total={num_samples}"
        )
        # --- ⬆️ 여기 까지 ---


        # 서버로 반환할 "샘플 수"도 전역 합계로 만들기 위해 지역변수를 둠
        ret_num_samples = num_samples

        if using_accelerate and num_samples > 0:
            try:
                import torch
                total_t   = torch.tensor([num_samples], device=self.accelerator.device, dtype=torch.long)
                loss_t    = torch.tensor([loss_total],  device=self.accelerator.device, dtype=torch.float32)
                correct_t = torch.tensor([correct],     device=self.accelerator.device, dtype=torch.long)

                # 모든 프로세스에서 합계(reduction='sum'). 모든 랭크에 동일한 값이 돌아옵니다.
                total_all   = self.accelerator.reduce(total_t,   reduction='sum')[0].item()
                loss_all    = self.accelerator.reduce(loss_t,    reduction='sum')[0].item()
                correct_all = self.accelerator.reduce(correct_t, reduction='sum')[0].item()

                if total_all > 0:
                    final_results = {
                        f'{split}_total': total_all,
                        f'{split}_loss':  loss_all,
                        f'{split}_avg_loss': loss_all / total_all,
                        f'{split}_acc':   correct_all / total_all,
                    }
                    self.ctx.eval_metrics = final_results
                    ret_num_samples = total_all  # ← 서버 가중치에 쓰이는 "샘플 수"도 전역 합계로 반환
                else:
                    self.ctx.eval_metrics = local_results
            except Exception as e:
                logger.warning(f"[aggregate warn] reduce failed: {e}")
                self.ctx.eval_metrics = local_results
        else:
            # 단일 프로세스 혹은 샘플 없음
            self.ctx.eval_metrics = local_results

        # 5) on_fit_end 훅 (여기서 Accelerator를 삭제/메모리 정리해도 안전)
        for hook in hooks_set["on_fit_end"]:
            hook(self.ctx)

        # 6) 반환값: 전역 합계(멀티프로세스면 480 / 단일이면 로컬 값)
        return ret_num_samples    

    @lifecycle(LIFECYCLE.EPOCH) #한 에폭(epoch) 단위로 반복 실행. 끝나면 에폭용 임시변수(CtxVar(..., "epoch")) 일괄 삭제
    def _run_epoch(self, hooks_set, run_step=-1):


        if run_step == -1:
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_epoch")#총 epoch 수. batch든 epoch 모드이든 total data 루프 몇 번 도는지 계산됨.
        for epoch_i in range(run_step):
            self.ctx.cur_epoch_i = CtxVar(epoch_i, "epoch")

            for hook in hooks_set["on_epoch_start"]:
                hook(self.ctx)

            self._run_batch(hooks_set)#llm trainer로 override. run_step=-1로 된다는 것 유의. 마지막 epoch의 부족한 batch update도 고려.

            for hook in hooks_set["on_epoch_end"]:
                hook(self.ctx)

    @lifecycle(LIFECYCLE.BATCH) #한 배치(batch) 단위로 반복 실행. 끝나면 배치용 임시변수(CtxVar(..., "batch")) 일괄 삭제.
    def _run_batch(self, hooks_set, run_step=-1):
        if run_step == -1:
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")
        for batch_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_backward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1:
                if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                    break

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): {model_name: model_val}
            strict (bool): ensure the k-v paris are strictly same
        """
        pass

    def get_model_para(self):
        """

        :return: model_parameters (dict): {model_name: model_val}
        """
        pass

    def print_trainer_meta_info(self):
        """
            print some meta info for code-users, e.g., model type; the para
            names will be filtered out, etc.,
        """
        logger.info(f"Model meta-info: {type(self.ctx.model)}.")
        logger.debug(f"Model meta-info: {self.ctx.model}.")
        # logger.info(f"Data meta-info: {self.ctx['data']}.")

        ori_para_names = set(self.ctx.model.state_dict().keys())

        #개인화 필터링 후 보존될 파라미터
        preserved_paras = self._param_filter(self.ctx.model.state_dict()) 
        preserved_para_names = set(preserved_paras.keys())

        #필터링된(제외된) 파라미터 계산
        filtered_para_names = ori_para_names - preserved_para_names


        #파라미터 통계 로그
        logger.info(f"Num of original para names: {len(ori_para_names)}.")
        logger.info(f"Num of original trainable para names:"
                    f" {len(self.ctx['trainable_para_names'])}.")
        logger.info(
            f"Num of preserved para names in local update:"
            f" {len(preserved_para_names)}. \n"
            f"Preserved para names in local update: {preserved_para_names}.")
        logger.info(
            f"Num of filtered para names in local update:"
            f" {len(filtered_para_names)}. \n"
            f"Filtered para names in local update: {filtered_para_names}.")
        

        #훅(hook) 설정 로그

        logger.info(f"After register default hooks,\n"
                    f"\tthe hooks_in_train is:\n\t"
                    f"{format_log_hooks(self.hooks_in_train)};\n"
                    f"\tthe hooks_in_eval is:\n\
            t{format_log_hooks(self.hooks_in_eval)}")

    def _param_filter(self, state_dict, filter_keywords=None): #학습가능한 파라미터만 필터링해서 보냄.
        """
        model parameter filter when transmit between local and gloabl,
        which is useful in personalization.
        e.g., setting cfg.personalization.local_param= ['bn', 'norms']
        indicates the implementation of
        "FedBN: Federated Learning on Non-IID Features via Local Batch
        Normalization, ICML2021", which can be found in
        https://openreview.net/forum?id=6YEQUn0QICG

        Arguments:
            state_dict (dict): PyTorch Module object's state_dict.
        Returns:
            state_dict (dict): remove the keys that match any of the given
            keywords.
        """
        # 1) 기본 모드(local/global)일 땐 아무 것도 공유하지 않음.
        #### 개인화가 아닌 표준 연합학습 모드이므로 아예 빈 딕셔너리를 반환해 “로컬 파라미터를 전혀 보내지 않겠다”고 설정
        if self.cfg.federate.method in ["local", "global"]:
            return {}
        

        # 2) 필터 키워드 목록이 주어지지 않았다면 cfg.personalization.local_param 사용
        #### 예를 들어 cfg.personalization.local_param = ["bn","norm"] 처럼 설정해 두면 "bn"이나 "norm"이 이름에 들어간 파라미터는 로컬에만 남기겠다는 의미.
        if filter_keywords is None:
            filter_keywords = self.cfg.personalization.local_param #[]

        # 3) “공유 가능한 파라미터”인지 검사하는 필터 함수. trainable_para_names만 필터링 한다.

        #trainable_filter 는 당연히 함수
        #share_non_trainable_para = True 인 경우 어떤 input p가 들어가도 무조건 True 를 반환합니다.
        #share_non_trainable_para = False 인 경우 p in self.ctx.trainable_para_names를 반영한 Boolean 반환!!

        #share_non_trainable_para = False로 구현 중.  trainable_para_names만 필터링 한다.


        trainable_filter = lambda p: True if \
            self.cfg.personalization.share_non_trainable_para else \
            lambda p: p in self.ctx.trainable_para_names
        


        # 4) “키워드 필터” 함수: 이름에 특정 키워드가 포함되는지 검사
        keyword_filter = filter_by_specified_keywords


        # 5) 최종 전송 파라미터만 골라 딕셔너리로 반환
        return dict(
            filter(
                lambda elem: trainable_filter(elem[1]) and keyword_filter(
                    elem[0], filter_keywords), state_dict.items()))  #실질적으로 trainable_filter(elem[1]) 이것만 고려하면 됨.

    def save_model(self, path, cur_round=-1):
        raise NotImplementedError(
            "The function `save_model` should be implemented according to "
            "the ML backend (Pytorch, Tensorflow ...).")

    def load_model(self, path):
        raise NotImplementedError(
            "The function `load_model` should be implemented according to "
            "the ML backend (Pytorch, Tensorflow ...).")
