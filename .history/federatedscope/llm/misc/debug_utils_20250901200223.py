import os, logging, torch, gc
from typing import Optional

logger = logging.getLogger(__name__)

def _all_ranks() -> bool:
    return os.environ.get("FS_LOG_SUMMARY_ALL_RANKS", "0") == "1"

def _is_main() -> bool:
    if _all_ranks():
        return True
    r = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try: return int(r) == 0
    except: return True

def log_memory(tag: str):
    if not _is_main(): return
    if not torch.cuda.is_available(): 
        logger.info(f"[DBG_MEM][{tag}] cuda=N/A"); 
        return
    dev_count = torch.cuda.device_count()
    torch.cuda.synchronize()
    ss = []
    for d in range(dev_count):
        try:
            torch.cuda.synchronize(d)
            stats = torch.cuda.memory_stats(d)
            alloc = stats.get("allocated_bytes.all.current", 0)
            reserv = stats.get("reserved_bytes.all.current", 0)
            max_reserv = stats.get("reserved_bytes.all.peak", 0)
            ss.append(f"dev{d}:alloc={alloc}B,resv={reserv}B,peak={max_reserv}B")
        except Exception as e:
            ss.append(f"dev{d}:err={e}")
    logger.info(f"[DBG_MEM][{tag}] {' | '.join(ss)}")

def log_loader(loader, split: str, tag: str):
    if not _is_main(): return
    if loader is None:
        logger.info(f"[DBG_LOADER][{tag}][{split}] loader=None"); return
    bs = getattr(loader, "batch_size", None)
    nw = getattr(loader, "num_workers", None)
    samp = getattr(loader, "sampler", None)
    slen = None
    try: slen = len(samp) if samp is not None else None
    except: pass
    logger.info(f"[DBG_LOADER][{tag}][{split}] id={id(loader)}, "
                f"bs={bs}, num_workers={nw}, sampler={type(samp).__name__}, len(sampler)={slen}")

def log_accel_state(accel, model, optimizer, tag: str):
    if not _is_main(): return
    amp_dtype = None
    try:
        amp_dtype = getattr(getattr(accel, "mixed_precision", None), "value", None) or \
                    getattr(accel, "mixed_precision", None)
    except: pass
    # 모델 파라미터/LoRA 하나만 포인터 찍기
    first_ptr = None
    try:
        for p in model.parameters():
            first_ptr = int(p.data_ptr()); break
    except: pass
    lora_ptr = None
    try:
        for m in model.modules():
            if hasattr(m, "lora_A") and isinstance(m.lora_A, dict) and len(m.lora_A) > 0:
                anyA = next(iter(m.lora_A.values()))
                lora_ptr = int(anyA.weight.data_ptr()); break
    except: pass
    logger.info(f"[DBG_ACCEL][{tag}] accel_id={id(accel)}, model_id={id(model)}, "
                f"opt_id={id(optimizer) if optimizer else None}, amp={amp_dtype}, "
                f"first_param_ptr={first_ptr}, lora_ptr={lora_ptr}")

def log_first_forward_probe(model, tag: str):
    """첫 forward 직전에 1회만 찍히는 pre-hook (원하면 사용)"""
    if not _is_main(): return
    handle_attr = "_dbg_first_fwd_hook"
    try:
        if hasattr(model, handle_attr) and getattr(model, handle_attr):
            getattr(model, handle_attr).remove()
            setattr(model, handle_attr, None)
    except: pass

    def _prehook(m, inputs):
        logger.info(f"[DBG_FWD][pre][{tag}] about_to_forward; model_id={id(m)}")
        try:
            getattr(m, handle_attr).remove()
        except: pass
        setattr(m, handle_attr, None)

    if hasattr(model, "register_forward_pre_hook"):
        h = model.register_forward_pre_hook(lambda m, i: _prehook(m, i))
        setattr(model, handle_attr, h)

def hard_gc(tag: str):
    """가비지 강제수거 + 캐시 비우기 지점 표식"""
    if not _is_main(): return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info(f"[DBG_GC][{tag}] gc+empty_cache done")
