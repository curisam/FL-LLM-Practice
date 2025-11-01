# federatedscope/llm/misc/debug_utils.py
import os
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

def _is_main_rank() -> bool:
    """
    로그 폭주 방지: 기본 rank0만 로그. 모든 랭크에서 보고 싶으면
    FS_LOG_SUMMARY_ALL_RANKS=1 환경변수로 켜세요.
    """
    if os.environ.get("FS_LOG_SUMMARY_ALL_RANKS", "0") == "1":
        return True
    r = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(r) == 0
    except Exception:
        return True

def _unwrap_module(m):
    """
    Accelerate/DDP의 .module, Adapter/Peft의 .model 레이어를 안전하게 벗겨서
    실제 HF 베이스 모듈까지 내려간다.
    """
    seen = set()
    cur = m
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # 1) DDP/Accelerate 래핑(.module)
        if hasattr(cur, "module"):
            cur = cur.module
            continue
        # 2) Adapter/Peft 래핑(.model) → 베이스에 get_input_embeddings()가 있다면 그쪽으로
        try:
            has_gie_here = callable(getattr(cur, "get_input_embeddings", None))
            if hasattr(cur, "model") and hasattr(cur.model, "get_input_embeddings") and not has_gie_here:
                cur = cur.model
                continue
        except Exception:
            pass
        break
    return cur

def log_tok_model_sync(tokenizer, model, tag: str = ""):
    """
    토크나이저 길이, (언랩된) 모델의 input/output 임베딩 크기와 weight 포인터(주소),
    그리고 LoRA 파라미터(가능하면) 포인터를 한 줄로 남긴다.
    """
    if not _is_main_rank():
        return

    # 0) 토크나이저 길이
    try:
        tok_len = getattr(tokenizer, "vocab_size", None)
        if tok_len is None:
            tok_len = len(tokenizer)
    except Exception:
        tok_len = "<err>"

    # 1) 언랩된 베이스 모델
    base = _unwrap_module(model)

    # 2) input/output 임베딩 모듈 찾기
    in_emb = None
    out_emb = None
    try:
        if base is not None and hasattr(base, "get_input_embeddings"):
            in_emb = base.get_input_embeddings()
    except Exception:
        pass
    try:
        if base is not None and hasattr(base, "get_output_embeddings"):
            out_emb = base.get_output_embeddings()
    except Exception:
        pass

    # 3) 임베딩 크기/포인터
    in_num: Optional[int] = None
    in_ptr: Optional[int] = None
    if in_emb is not None and hasattr(in_emb, "weight"):
        try:
            in_num = getattr(in_emb, "num_embeddings", None) or in_emb.weight.shape[0]
            in_ptr = int(in_emb.weight.data_ptr())
        except Exception:
            pass

    out_num: Optional[int] = None
    out_ptr: Optional[int] = None
    if out_emb is not None and hasattr(out_emb, "weight"):
        try:
            # 일부 모델은 output embedding이 tie 되어 input과 동일 포인터일 수 있음
            out_num = getattr(out_emb, "num_embeddings", None) or out_emb.weight.shape[0]
            out_ptr = int(out_emb.weight.data_ptr())
        except Exception:
            pass

    # 4) LoRA 파라미터 포인터(있으면 하나만)
    lora_ptr = None
    try:
        scan_root = base if base is not None else model
        for m in scan_root.modules():
            # PEFT의 LoraLayer 패턴: lora_A/lora_B dict 보유
            if hasattr(m, "lora_A") and isinstance(m.lora_A, dict) and len(m.lora_A) > 0:
                anyA = next(iter(m.lora_A.values()))
                if hasattr(anyA, "weight"):
                    lora_ptr = int(anyA.weight.data_ptr())
                    break
    except Exception:
        pass

    # 5) 로그 출력
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    base_cls = type(base).__name__ if base is not None else None
    in_cls = type(in_emb).__name__ if in_emb is not None else None
    out_cls = type(out_emb).__name__ if out_emb is not None else None

    logger.info(
        f"[DBG_EMB][tag={tag}][rank={rank}] "
        f"tok_len={tok_len} | "
        f"base={base_cls} | "
        f"in_emb=({in_cls}) num={in_num} ptr={in_ptr} | "
        f"out_emb=({out_cls}) num={out_num} ptr={out_ptr} | "
        f"lora_ptr={lora_ptr}"
    )
