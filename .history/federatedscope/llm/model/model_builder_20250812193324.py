import os

from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.core.configs.config import global_cfg
import torch

import logging

from federatedscope.llm.misc.accel_utils import should_use_device_map_auto

logger = logging.getLogger(__name__)


def get_model_from_huggingface(model_name, config, **kwargs): #kwargs={}
    from transformers import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    if config.train.is_enable_half: ############ 적용 #######
        kwargs['torch_dtype'] = torch.bfloat16

    if config.model.llm_type == 'SequenceClassification':
        from transformers import AutoModelForSequenceClassification
        if len(config.model.llm_kwargs) > 0:
            kwargs.update(config.model.llm_kwargs[0])
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs)
    else:############### 이걸로 적용 ###############################
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs) #LoRA 적용 안됨.


def get_model_from_modelscope(model_name, config, **kwargs):
    from modelscope import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config, load_from_prev_ckpt=False, **kwargs):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')

    # --- [1] 🔒 device_map='auto' 방어 ---
    if 'device_map' in kwargs and kwargs['device_map'] == 'auto': #False
        if not should_use_device_map_auto():
            print("[get_llm] ⚠️ device_map='auto' 무시됨 (DDP 모드이므로 unsafe)")
            kwargs['device_map'] = None


    if config.model.load_from_local_pretrained_fs_config != '': #False
        # load model from local pretrained model
        pretrained_cfg = global_cfg.clone()
        pretrained_cfg.merge_from_file(
            config.model.load_from_local_pretrained_fs_config)
        assert pretrained_cfg.model.type.split('@')[0] == model_name, \
            'Two models cannot match. Failed to load from pretrained.'
        pretrained_model = get_llm(pretrained_cfg, **kwargs)
        if config.model.load_from_local_pretrained_model_path != '':
            path = config.model.load_from_local_pretrained_model_path
            ckpt = torch.load(path, map_location='cpu')
            logger.info('Successfully import the pretrained model '
                        f'from the checkpoint {path}. ')
            pretrained_model.load_state_dict(ckpt['model'])
        model = pretrained_model.merge_and_unload()
        logger.info(f'Merge and unload to {type(model)}...')
    elif model_hub == 'huggingface_llm':############# 이 부분이 걸림, 여기서는 LoRA 안붙임. ##############
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config,
                                           **kwargs) 
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name,
                                          config=config,
                                          **kwargs)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)
    

    if model_config.llm_type == 'SequenceClassification': #False
        model.config.pad_token_id = tokenizer.pad_token_id


    model.resize_token_embeddings(len(tokenizer))


    if num_new_tokens > 0:  
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        if model_config.llm_type != 'SequenceClassification':
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {} #{'adapter_package': 'peft', 'adapter_method': 'lora', 'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
    
    
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    if config.llm.adapter.use and config.llm.adapter.local_only: # config.llm.adapter.local_only=False로 걸림
        #adapter_names=[f"Adapter_{i}" for i in range(count)] → 예: ["Adapter_0","Adapter_1","Adapter_2"]를 만든 다음
        #각 이름에 대해 **동일한 LoRA 설정을 가진 여러 “이름 있는 어댑터”**를 모델 안에 추가.
        #state_dict엔 각 어댑터 이름 기준으로 lora_A.<adapter_name>.weight, lora_B.<adapter_name>.weight가 추가됨.
        #**forward는 항상 하나(또는 set_adapter로 지정한 리스트)**만 적용. 나머지 어댑터는 “붙어만 있고” 적용되지 않아요.

        """
    활성 어댑터(= forward에 쓰이는 것) 바꾸는 법
    PEFT 모델에서:      
            
    # 현재 활성 어댑터 확인
    print(model.active_adapter)        # 예: "default"

    # 활성 어댑터 바꾸기
    model.set_adapter("Adapter_1")
    print(model.active_adapter)        # "Adapter_1"

    이렇게 바꾸면 이후 forward에 Adapter_1의 LoRA가 쓰이고, default/Adapter_0 등은 적용되지 않음.

    여러 어댑터를 동시에 쓰고 싶다면? 
    PEFT는 리스트를 줘서 여러 어댑터를 합산하는 것도 지원해요(단, 동일 구조/호환 전제):    
    model.set_adapter(["Adapter_0", "Adapter_1"])  # 두 LoRA 델타를 합쳐서 사용
      
        
        """


        model.append_adapters(adapter_names=[
            f'Adapter_{i+1}' for i in range(config.federate.client_num)
        ])

    if config.llm.adapter.count > 1: #3으로 적용되어서 걸림, 실질적으로 forward에 쓰이는 것은 1개뿐
        model.append_adapters(adapter_names=[
            f'Adapter_{i}' for i in range(config.llm.adapter.count)
        ])

    if load_from_prev_ckpt: #적용 안됨
        # Here we load from the most recent one
        num_ckpt = config.federate.total_round_num // config.federate.save_freq
        prefix = ['final_'] + \
            [str(i*config.federate.save_freq) + '_'
             for i in range(num_ckpt, -1, -1)] + ['']
        dirname, filename = os.path.split(config.federate.save_to)
        for pre in prefix:
            ckpt_path = os.path.join(dirname, pre + filename)
            logger.info(f'Attempt to load from {ckpt_path}')
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                logger.info(f'Model of Round {ckpt["cur_round"]} loads '
                            f'from the checkpoint {ckpt_path}')
                break

    return model
