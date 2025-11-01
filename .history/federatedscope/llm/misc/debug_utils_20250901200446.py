import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def _is_main_rank() -> bool:
    # 전체 랭크를 보고 싶으면 env FS_LOG_SUMMARY_ALL_RANKS=1
    if os.environ.get("FS_LOG_SUMMARY_ALL_RANKS", "0") == "1":
        return True
    r = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(r) == 0
    except Exception:
        return True

def log_tok_model_sync(tokenizer, model, tag: str = ""):
    """토크나이저 길이, 임베딩 테이블 크기, 임베딩 weight 포인터(주소),
    (가능하면) LoRA 파라미터 포인터를 한 줄로 남긴다."""
    if not _is_main_rank():
        return

    # tokenizer 길이
    try:
        tok_len = getattr(tokenizer, "vocab_size", None)
        if tok_len is None:
            tok_len = len(tokenizer)
    except Exception:
        tok_len = "<err>"

    # 임베딩 모듈/크기/포인터
    emb = None
    try:
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
        elif hasattr(getattr(model, "model", None), "get_input_embeddings"):
            emb = model.model.get_input_embeddings()
    except Exception:
        pass

    num_embeddings: Optional[int] = None
    emb_ptr: Optional[int] = None
    if emb is not None and hasattr(emb, "weight"):
        try:
            num_embeddings = getattr(emb, "num_embeddings", None) or emb.weight.shape[0]
            emb_ptr = int(emb.weight.data_ptr())
        except Exception:
            pass

    # LoRA 파라미터 포인터(있으면 하나만)
    lora_ptr = None
    try:
        for m in model.modules():
            if hasattr(m, "lora_A") and isinstance(m.lora_A, dict) and len(m.lora_A) > 0:
                anyA = next(iter(m.lora_A.values()))
                lora_ptr = int(anyA.weight.data_ptr())
                break
    except Exception:
        pass

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    logger.info(f"[DBG_EMB][tag={tag}][rank={rank}] "
                f"tok_len={tok_len}, num_embeddings={num_embeddings}, "
                f"emb_ptr={emb_ptr}, lora_ptr={lora_ptr}")
