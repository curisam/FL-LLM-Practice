#!/usr/bin/env bash
set -euo pipefail

# ---- 커널 안전 모드(Flash/xFormers 끄고, SDPA는 코드에서 eager/수학커널)
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1

# ---- HF Hub 완전 로컬 (런 중 네트워크 접근 차단)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- NCCL/디버깅
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# ---- CUDA 할당자(파편화 완화)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# ---- CPU 스레드 억제
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- 실행 옵션(디버깅 플래그)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_MODULE_LOADING=EAGER
export FS_LOG_SUMMARY_ALL_RANKS=1
export TRANSFORMERS_VERBOSITY=error

CFG=fedbiscuit_script/tldr/ensemble_local_only.yaml
ACCEL=fedbiscuit_script/accelerator_config_bf16_1gpu.yaml
BASE_OUT=exp/tldr/choice_qwen/cluster/local_only_1.0/raw

# 001..053 루프
for i in $(seq 7 53); do
  SRC=$(printf "%03d" "$i")

  CKPT="checkpoints_1.0_local_only/local_only_tldr_choice_qwen_client_${SRC}.ckpt"
  OUTDIR="${BASE_OUT}/src_${SRC}"

  echo "==== SRC ${SRC} 시작 ===="
  echo "CKPT:   ${CKPT}"
  echo "OUTDIR: ${OUTDIR}"

  CUDA_VISIBLE_DEVICES=0 \
  accelerate launch \
    --config_file "$ACCEL" \
    --main_process_port 29500 \
    federatedscope/main.py \
    --cfg "$CFG" \
    model.load_from_local_pretrained_model_path "$CKPT" \
    eval.outdir "$OUTDIR" \
    llm.accelerator.use True

  echo "==== SRC ${SRC} 완료 ===="
done