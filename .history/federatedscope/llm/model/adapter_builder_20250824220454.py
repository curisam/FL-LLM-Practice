import torch
import torch.nn as nn
from collections import OrderedDict
from peft import get_peft_model, TaskType, PeftModel

import accelerate
from accelerate import dispatch_model, infer_auto_device_map, \
    load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

from transformers import (OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM,
                          LlamaForCausalLM, LlamaForSequenceClassification,
                          Qwen2ForCausalLM, GemmaForCausalLM)


from federatedscope.llm.misc.accel_utils import in_distributed_mode, allow_device_map_auto_by_env


MODEL_UNIT = {
    LlamaForCausalLM: ['LlamaDecoderLayer'],
    LlamaForSequenceClassification: ['LlamaDecoderLayer'],
    BloomForCausalLM: ['BloomBlock'],
    GPT2LMHeadModel: ['GPT2Block'],
    OPTForCausalLM: ['OPTDecoderLayer'],
    Qwen2ForCausalLM: ['Qwen2DecoderLayer'],
    GemmaForCausalLM: ['GemmaDecoderLayer']
}

import logging
import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


def enable_adapter(model, package, adapter, **kwargs):#package: peft, adapter:lora로 들어감.
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """
        if adapter == 'lora': ########## 이거에 해당 ##############
            """
            get_peft_model이 원래 LM을 PeftModel로 감싸면서 지정된 모듈(q/k/v/o, gate/up/down_proj)에 
            LoRA 어댑터(A/B 행렬)를 “끼워 넣었고”, 그 결과 state_dict 키에 base_model.model.   ...lora_A/B...가 생기고, 
            베이스 가중치는 동결, LoRA 가중치만 학습/전송 대상으로 바뀐 것이야.           
            
            """

            """
            앞부분(prefix): PEFT가 래핑하면서 경로에 base_model.model. 이 붙어요.

            LoRA 파라미터(suffix): 타깃 모듈 이름 뒤에 .lora_A.<어댑터이름>.weight, **.lora_B.<어댑터이름>.weight**가 붙어요.
            어댑터 이름을 따로 안 주면 기본이 **default**라서
            ...q_proj.lora_A.default.weight, ...q_proj.lora_B.default.weight 이런 식으로 생깁니다.

            정리 예시:

            베이스 가중치(동결):
            base_model.model.model.layers.0.self_attn.q_proj.weight

            LoRA 추가 가중치(학습):
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
            
            """

            """
            
            1. 저장/전송 관점

            PEFT는 보통 LoRA 가중치만 가볍게 저장/공유할 수 있어.

            네 FL 파이프라인에서 “학습 가능한 이름만” 필터하면 자연스럽게
            *.lora_A.*, *.lora_B.*만 서버로 보내게 됨(베이스는 동결이므로 제외).


            2. 학습 가능 파라미터가 바뀜

            기본 가중치(...weight, ...bias)는 동결(requires_grad=False),
            LoRA A/B만 학습(requires_grad=True).

            bias='none'이므로 bias는 존재하되 학습은 안 함(동결 상태로 남음).
            
            """

            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            #LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, r=8, target_modules={'o_proj', 'k_proj', 'v_proj', 'q_proj', 'down_proj', 'gate_proj', 'up_proj'}, lora_alpha=16, lora_dropout=0.05, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={})
            model = get_peft_model(model, peft_config) #PEFT 클래스 객체. state_dict를 할때 backbone, active 여부와 상관없이 등록된 adapter 전부가 lora 전부 포함되어서 나옴.
        elif adapter == 'prefix':
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError

        model.print_trainable_parameters() #trainable params: 4,399,104 || all params: 498,172,032 || trainable%: 0.8830491712549612 출력으로 끝.
        return model, peft_config

    if package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        # TODO:  After supporting adapterhub, we will move the following
        #   parameters in yaml file for users' convenient
        if adapter == 'lora':
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            from transformers.adapters import AdapterConfig, ConfigUnion

            # TODO: configure these args in cfg
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
        return model, config

    raise NotImplementedError


class AdapterModel(nn.Module):
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        super().__init__()



        self.model = None
        try:
            self.model_unit = MODEL_UNIT[type(model)]
        except:
            self.model_unit = None

        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft') #peft
            adapter_method = kwargs.pop('adapter_method', 'lora') #lora

            self.model, self.peft_config = \
                enable_adapter(model,
                               adapter_package,
                               adapter_method,
                               **kwargs) #self.model: PEFT 클래스.
            self.adapter_names = ['default']


        else:
            self.model = model 


    def get_input_embeddings(self):
        return self.model.get_input_embeddings() #self.model은 PEFT 클래스

    def forward(self, disable_adapter=False, *args, **kwargs):
        if isinstance(self.model, PeftModel) and disable_adapter: #disable_adapter=True: 베이스로만 forward.
            with self.model.disable_adapter():
                return self.model(*args, **kwargs)

        return self.model.forward(*args, **kwargs) #self.model은 PEFT 클래스

    def generate(self, disable_adapter=False, *args, **kwargs): #일단 pass
        try:
            if isinstance(self.model, PeftModel) and disable_adapter:
                with self.model.disable_adapter():
                    res = self.model.generate(*args, **kwargs)

            else:
                res = self.model.generate(*args, **kwargs) #self.model은 PEFT 클래스
        except RuntimeError as e:
            # When does evaluation in HELM,
            # half precision will cause RuntimeError,
            # the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                if isinstance(self.model, PeftModel) and disable_adapter:
                    with self.model.disable_adapter():
                        res = self.model.generate(*args, **kwargs)
                else:
                    res = self.model.generate(*args, **kwargs) #self.model은 PEFT 클래스
            else:
                raise RuntimeError(e) 
        return res

    #학습 가능한 파라미터만(대부분 LoRA) 추려서 반환.
    def state_dict(self, return_trainable=True, *args, **kwargs): #기존 PEFT 클래스: 활성 어댑터 가중치만 반환(LoRA만)
        if return_trainable:
            return self.get_trainable_state_dict() #**학습 대상(주로 LoRA A/B, modules_to_save 등)**만 골라 반환.
        else:
            return self.model.state_dict(*args, **kwargs) #backbone, lora 전부 내보냄. 정말로 풀 가중치(백본+어댑터)를 저장”하려면 아래 save_model(..., merge_adapter=True)를 쓰거나, self.merge_and_unload().state_dict()처럼 merge 이후에 덤프

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict=False) #전달받은 state_dict를 부분 로딩(missing/unexpected 허용)으로 주입

    def get_trainable_state_dict(self): #현재 모델(self.model)에서 전송/저장을 위한 학습 대상 파라미터만 뽑아 OrderedDict로 반환.
        """
        기본 규칙:

        requires_grad=True인 파라미터 포함 (보통 LoRA A/B, modules_to_save 등)

        멀티 어댑터 보정: self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함. 활성화되지 않은 어댑터의 파라미터까지 포함 가능.       
        """
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad: #1차 기준: requires_grad=True인 파라미터만 수집. → LoRA A/B, modules_to_save 등이 여기에 들어감.
                grad_params.append(name)
            # Special case for multiple adapters
            for adap_name in self.adapter_names: #멀티 어댑터 보정: 특정 어댑터 이름이 경로에 들어있으면 requires_grad=False여도 포함.

                #이유: 라운드/상황에 따라 어떤 어댑터는 비활성(동결) 상태여도 전송/보관이 필요할 수 있음.
                if (adap_name in name) and (name not in grad_params): #param.requires_grad=False but adap_name은 있는 경우.
                    grad_params.append(name)
                    break #break로 중복 방지. self.adapter_names 중 하나에서 append되었으면 다른건 그냥 바로 pass.

        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self,
                   path,
                   state=0,
                   merge_adapter=False,
                   return_trainable=True): #체크포인트를 파일로 저장. 
        if merge_adapter and isinstance(self.model, PeftModel):
            merged_model = self.model.merge_and_unload() #LoRA를 베이스에 합쳐 순수 HF 모델 가중치로 저장
            ckpt = {'cur_round': state, 'model': merged_model.state_dict()}
        elif return_trainable:
            ckpt = {'cur_round': state, 'model': self.state_dict()} #학습 대상만 저장 (get_trainable_state_dict() 경유)
        else:
            ckpt = {'cur_round': state, 'model': self.model.state_dict()} #self.model.state_dict() 그대로 저장
        torch.save(ckpt, path)



    def sharding(self): #Accelerate의 infer_auto_device_map + dispatch_model를 이용해 단일 노드 다중 GPU에서 모델을 디바이스별로 자동 샤딩.
        if in_distributed_mode() or not allow_device_map_auto_by_env():
            # 분산 환경이면 dispatch_model 사용 안 함
            self.device_map = None
            return

        if not hasattr(self, 'device_map'):
            max_memory = get_balanced_memory(
                self.model,
                max_memory=None,
                no_split_module_classes=self.model_unit,
                low_zero=False,
            ) #GPU별 메모리 예산 계산
            from accelerate import infer_auto_device_map, dispatch_model
            self.device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                no_split_module_classes=self.model_unit,
            ) #디바이스 매핑 추론
            self.model = dispatch_model(self.model, device_map=self.device_map)
            """
            adap = AdapterModel(base_model, use_adapter=True, adapter_package='peft', adapter_method='lora', r=8, lora_alpha=16)
            adap.sharding()
            print(adap.device_map)  # 예) {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 1, ...}
            
            """



    def print_model_map(self): #파라미터별로 할당된 디바이스를 로깅. 샤딩/디바이스 맵이 제대로 적용됐는지 눈으로 검증할 때 유용.
        for i in self.model.named_parameters(): #i[0]: 파라미터 이름 (str), i[1]: 파라미터 텐서 (torch.nn.Parameter).
            logger.info(f"{i[0]} -> {i[1].device}")

    def merge_and_unload(self): #이 반환값은 새 모델 객체고, 현재 self.model은 바뀌지 않는 점에 주의(이 함수는 “반환”만 함).
        if isinstance(self.model, PeftModel) and \
                callable(self.model.merge_and_unload): #self.model은 PEFT 클래스. #PeftModel이면 LoRA를 베이스에 합쳐 순수 HF 모델을 반환.
            return self.model.merge_and_unload()
        else:
            return self.model #아니면 원래 모델 그대로.
         
    def append_adapters(self, adapter_names, peft_config=None): #여러 개의 어댑터를 동일한 peft_config로 추가할 준비.외부에서 별도 config 주지 않으면 현재 보유한 self.peft_config 재사용.
        assert isinstance(self.model, PeftModel)
        peft_config = self.peft_config if peft_config is None else peft_config
        for name in adapter_names:
            self.model.add_adapter(name, peft_config)
            self.adapter_names.append(name)

    def set_active_adapter(self, adapter_name):#유효성 검사 후 해당 어댑터를 활성 상태로 전환. 활성 어댑터만 학습/추론에 사용됨(PEFT 내부 동작).
        assert adapter_name in self.adapter_names
        self.model.set_adapter(adapter_name)  #self.model은 PEFT 클래스

    def get_active_adapter(self):
        return self.model.active_adapter #현재 활성 어댑터 이름을 리턴.


    @property
    def config(self):
        return self.model.config

    @property
    def layers(self):
        _layers = []
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList):
                # This one should be encoders/decoders
                _layers.append(module)

        if len(_layers) == 1:
            return _layers[0]
        return _layers

    def set_layers(self, layers):
        if isinstance(self.layers, nn.ModuleList) and isinstance(
                layers, nn.ModuleList):
            self.layers._modules = layers._modules

        elif isinstance(layers, list) and isinstance(self.layers, list):
            # This consists of multiple ModuleLists
            assert len(self.layers) == len(layers)
            for src, tgt in zip(self.layers, layers):
                assert isinstance(tgt, nn.ModuleList)
                src._modules = tgt._modules

        else:
            raise ValueError(
                'Layers cannot be set due to the mismatched type. ')

    @property
    def trainable_param_name_pattern(self):
        if isinstance(self.model, PeftModel):
            return self.model.active_adapter
        return None

    def set_trainable_modules(self, modules=None):
        # First, set all modules to untrainable
        for module in self.model.modules():
            module.requires_grad_(False)

        # Second, search for the capable modules
        if modules is None:
            # Set the encoders/decoders to be trainable
            modules = self.layers

        if isinstance(modules, nn.ModuleList):
            # Make it to the list
            trainable_modules = [modules]

        elif isinstance(modules, list):
            trainable_modules = modules

        else:
            raise ValueError(f'{modules} cannot be trainable because '
                             f'{type(modules)}.')

        pattern = self.trainable_param_name_pattern
        for module in trainable_modules:
            for layer in module:
                for name, param in layer.named_parameters():
                    if pattern is None or pattern in name:
                        param.requires_grad = True

    # TODO: Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)




class LLMDataParallel(nn.DataParallel):
    def __init__(self, adap_model, device_ids=None, output_device=None, dim=0):
        assert isinstance(adap_model, AdapterModel)
        super().__init__(adap_model.model,
                         device_ids=device_ids,
                         output_device=output_device,
                         dim=dim)
        self.model = adap_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def state_dict(self, return_trainable=True, *args, **kwargs):
        return self.model.state_dict(return_trainable, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict)

    def save_model(self, path, state=0):
        self.model.save_model(path, state)