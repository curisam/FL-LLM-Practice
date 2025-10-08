#!/usr/bin/env bash
set -euo pipefail

# ---- 커널/환경 플래그 (네가 쓰던 그대로) ----
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_MODULE_LOADING=EAGER
export FS_LOG_SUMMARY_ALL_RANKS=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

# ---- 고정 인자 ----
CFG="fedbiscuit_script/tldr/ensemble_local_only.yaml"
ACCEL="fedbiscuit_script/accelerator_config_bf16_ver1.yaml"
BASE_OUT="exp/tldr/choice_qwen/cluster/local_only_1.0/raw"

# ---- 직렬 루프: 001..053 ----
for i in $(seq 1 1); do
  SRC=$(printf "%03d" "$i")
  CKPT_REL="checkpoints_1.0_local_only_official/official/local_only_tldr_choice_qwen_client_${SRC}.ckpt"
  CKPT="$(realpath "$CKPT_REL")"
  OUTDIR="${BASE_OUT}/src_${SRC}"
  PORT=$((29500 + i))  # 매 런 고유 포트 (안전)

  # 존재 확인 + 로그
  [[ -f "$CKPT" ]] || { echo "[ERR] ckpt not found: $CKPT"; exit 1; }
  echo "==== SRC ${SRC} ===="
  echo "CKPT     : $CKPT"
  echo "OUTDIR   : $OUTDIR"
  echo "CKPT_SHA1: $(sha1sum "$CKPT" | cut -d' ' -f1)"

  # 출력 디렉토리 초기화(원하면 주석)
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

  # 실행
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch \
    --config_file "$ACCEL" \
    --main_process_port "$PORT" \
    federatedscope/main.py \
    --cfg "$CFG" \
    model.load_from_local_pretrained_model_path "$CKPT" \
    eval.outdir "$OUTDIR" \
    outdir "$OUTDIR" \
    llm.accelerator.use True

  echo "==== SRC ${SRC} done ===="
done

echo "=== ALL DONE ==="
