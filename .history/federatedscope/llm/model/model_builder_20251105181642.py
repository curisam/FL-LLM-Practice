import os

from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.core.configs.config import global_cfg
import torch

import logging

from federatedscope.llm.misc.accel_utils import should_use_device_map_auto

from federatedscope.llm.misc.debug_utils import log_tok_model_sync

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

    log_tok_model_sync(tokenizer, model, tag="after-build")


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
    
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)#model:AdapterModel 클래스  model.model은 PEFT 클래스.



    if config.llm.adapter.count > 1: #3으로 적용되어서 걸림, 실질적으로 forward에 쓰이는 것은 1개뿐
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

        """
        model이 PEFT CLASS일때
        model.peft_config는 모델 안에 “어떤 어댑터들이 어떤 설정(LoRA r/alpha/target_modules/bias 등)으로 붙어있는지”를 담은 딕셔너리
        키=어댑터 이름("default", "Adapter_1"…), 값=그 어댑터의 LoraConfig(혹은 다른 PEFT config).

        현재 어댑터 목록 확인: list(model.peft_config.keys())

        각 어댑터 세부 설정 조회

        cfg = model.peft_config["default"]
        print(cfg.r, cfg.lora_alpha, cfg.lora_dropout, cfg.target_modules, cfg.bias)
        
        """


        model.append_adapters(adapter_names=[
            f'Adapter_{i}' for i in range(config.llm.adapter.count)
        ])#"default", "Adapter_0", "Adapter_1", "Adapter_2"의 adapter가 있게 됨. Adapter class인 model은 append_adapters 존재.




    # 2) ckpt 불러와 그대로 로드 (여분 어댑터는 자동 무시)
    ckpt_path = getattr(model_config, "load_from_local_pretrained_model_path", None)  


    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd   = ckpt.get("model", ckpt)  # 일부는 바로 state_dict일 수 있음

        res = model.load_state_dict(sd, strict=False)  # 핵심: strict=False
        try:
            #mis: model에는 았지만 sd에는 없는 key의 수
            #unexp: sd에는 있지만 model에는 없는 key의 수.
            miss = len(res.missing_keys); unexp = len(res.unexpected_keys)
        except:
            miss = unexp = -1
        logger.info(f"[Warmup-Init] loaded from {ckpt_path} "
                    f"(round={ckpt.get('cur_round','?')}) | "
                    f"missing={miss} unexpected={unexp}")
        

    # 검사할 어댑터 이름(예시)
    name_a = "Adapter_0"
    name_b = "Adapter_1"

    for m in model._iter_lora_modules():
        if not hasattr(m, "lora_A"):
            continue
        keys = list(m.lora_A.keys())
        print("module:", type(m), "lora_A keys:", keys)

        if name_a in m.lora_B and name_b in m.lora_B:

            mod_a = m.lora_A[name_a]
            mod_b = m.lora_A[name_b]

            # weight 또는 첫 번째 파라미터를 안전하게 얻는 헬퍼
            def _get_weight_tensor(mod):
                if hasattr(mod, "weight"):
                    return mod.weight
                if isinstance(mod, (torch.nn.Parameter, torch.Tensor)):
                    return mod
                params = list(mod.parameters())
                if len(params):
                    return params[0]
                raise RuntimeError(f"Cannot find weight tensor in module {type(mod)}")


            a_w = _get_weight_tensor(mod_a)
            b_w = _get_weight_tensor(mod_b)

            # detach + CPU로 이동해서 안전 비교
            a_cpu = a_w.detach().cpu()
            b_cpu = b_w.detach().cpu()

            import ipdb; ipdb.set_trace(context=15)


            print("same object?", mod_a is mod_b)  # 모듈 객체 동일 여부
            try:
                same_storage = a_w.storage().data_ptr() == b_w.storage().data_ptr()
            except Exception:
                same_storage = False
            print("same storage?", same_storage)
            print("values equal?", torch.equal(a_cpu, b_cpu))
            print("max diff:", float((a_cpu - b_cpu).abs().max().item()))
            print("a device/dtype:", a_w.device, a_w.dtype, "b device/dtype:", b_w.device, b_w.dtype)
            print("a mean/std:", float(a_cpu.mean().item()), float(a_cpu.std().item()))
            print("b mean/std:", float(b_cpu.mean().item()), float(b_cpu.std().item()))








    return model #model:AdapterModel 클래스  model.model은 PEFT 클래스.



