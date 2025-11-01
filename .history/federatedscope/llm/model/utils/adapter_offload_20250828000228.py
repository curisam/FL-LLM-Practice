import gc
from typing import Iterable, List, Union
import torch
import torch.nn as nn

# -------- 공용 헬퍼 --------
def _iter_lora_like_modules(model: nn.Module):
    """
    LoRA 주입 모듈을 yield (PEFT 호환: lora_A/B, lora_embedding_A/B, lora_dropout 등)
    """
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") or hasattr(m, "lora_embedding_A"):
            yield name, m

def list_adapter_names(model: nn.Module) -> List[str]:
    """
    모델에 로드된 어댑터 이름을 최대한 안전하게 수집.
    - PEFT의 peft_config 키
    - lora_A / lora_embedding_A 딕셔너리 키
    - 없으면 active_adapter나 'default'로 폴백
    """
    names = set()

    # 1) peft_config 기반 (가장 안정적)
    if hasattr(model, "peft_config") and isinstance(model.peft_config, dict):
        names.update(list(model.peft_config.keys()))

    # 2) 모듈 내부의 lora 딕셔너리 키 수집
    for _, m in _iter_lora_like_modules(model):
        if hasattr(m, "lora_A"):
            names.update(list(getattr(m, "lora_A").keys()))
        if hasattr(m, "lora_embedding_A"):
            names.update(list(getattr(m, "lora_embedding_A").keys()))

    # 3) active_adapter 폴백
    if not names:
        active = getattr(model, "active_adapter", None)
        if isinstance(active, str) and active:
            names.add(active)

    # 4) 최종 폴백
    if not names:
        names.add("default")

    return sorted(list(names))

def _move_adapter_on_module(m: nn.Module, adapter: str, device: torch.device):
    # Linear류 LoRA
    if hasattr(m, "lora_A") and adapter in getattr(m, "lora_A", {}):
        m.lora_A[adapter].to(device)
    if hasattr(m, "lora_B") and adapter in getattr(m, "lora_B", {}):
        m.lora_B[adapter].to(device)
    if hasattr(m, "lora_dropout"):
        try:
            # dict 또는 ModuleDict 일 수 있음
            if adapter in m.lora_dropout:
                m.lora_dropout[adapter].to(device)
        except Exception:
            pass
    # Embedding류 LoRA
    if hasattr(m, "lora_embedding_A") and adapter in getattr(m, "lora_embedding_A", {}):
        m.lora_embedding_A[adapter].to(device)
    if hasattr(m, "lora_embedding_B") and adapter in getattr(m, "lora_embedding_B", {}):
        m.lora_embedding_B[adapter].to(device)

def set_active_adapter_safe(model: nn.Module, adapter: Union[str, Iterable[str]]):
    """
    환경마다 set_active_adapter / set_adapter 명이 다를 수 있어 안전하게 호출.
    """
    if hasattr(model, "set_active_adapter"):
        return model.set_active_adapter(adapter)
    if hasattr(model, "set_adapter"):
        return model.set_adapter(adapter)
    # 마지막 폴백: 속성만 세팅 (PEFT 대부분은 메서드가 있음)
    try:
        model.active_adapter = adapter if isinstance(adapter, str) else list(adapter)[0]
    except Exception:
        pass

def enable_grad_for_adapter(model: nn.Module, adapter: str, requires_grad: bool = True):
    """
    특정 어댑터 파라미터의 requires_grad를 토글 (학습 시작 시 활성 어댑터에 True 보장)
    """
    for n, p in model.named_parameters():
        is_adapter_param = (".lora_" in n) or (".modules_to_save." in n)
        # 어댑터 이름이 파라미터 경로에 들어가는 구조(PEFT 기본)
        if is_adapter_param and adapter in n:
            p.requires_grad = requires_grad

def offload_inactive_adapters(
    model: nn.Module,
    active: Union[str, Iterable[str]],
    device: torch.device,
    offload_device: torch.device = torch.device("cpu"),
    freeze_offloaded: bool = True,
    unfreeze_active: bool = True,
    empty_cache: bool = True,
):
    """
    활성 어댑터만 device(GPU)로, 비활성 어댑터는 CPU로 이동.
    - 어댑터가 1개('default') 뿐이면 사실상 no-op.
    - 평가/그루핑/학습 어디서든 호출 가능.
    """
    all_names = set(list_adapter_names(model))
    if isinstance(active, str):
        active_set = {active}
    else:
        active_set = set(active)

    # 이름 교집합 보정 (혹시 잘못 전달된 이름 보호)
    active_set = {a for a in active_set if a in all_names}
    if not active_set and all_names:
        active_set = {next(iter(all_names))}

    # 이동
    for _, m in _iter_lora_like_modules(model):
        # 이 모듈이 보유한 어댑터 집합
        adapters_here = set()
        if hasattr(m, "lora_A"):
            adapters_here.update(list(getattr(m, "lora_A").keys()))
        if hasattr(m, "lora_embedding_A"):
            adapters_here.update(list(getattr(m, "lora_embedding_A").keys()))
        # 어댑터별 이동
        for aname in adapters_here:
            tgt = device if aname in active_set else offload_device
            _move_adapter_on_module(m, aname, tgt)

    # grad 토글
    if freeze_offloaded:
        for n, p in model.named_parameters():
            is_adapter_param = (".lora_" in n) or (".modules_to_save." in n)
            if is_adapter_param:
                # 활성 어댑터만 True, 나머지는 False
                on_active = any(an in n for an in active_set)
                p.requires_grad = bool(on_active) if unfreeze_active else (not on_active)

    if device.type == "cuda" and empty_cache:
        torch.cuda.empty_cache()
        gc.collect()
