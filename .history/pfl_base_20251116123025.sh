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

# ---- CUDA 할당자
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ---- CPU 스레드 억제
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- 실행 옵션
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_MODULE_LOADING=EAGER
export FS_LOG_SUMMARY_ALL_RANKS=1
export TRANSFORMERS_VERBOSITY=error


# ---- 레포 루트 기준으로 PYTHONPATH 설정 (스크립트 위치 기준; set -u 안전)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || echo "${SCRIPT_DIR}")"
: "${PYTHONPATH:=}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"


CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file fedbiscuit_script/accelerator_config_bf16_ver1.yaml \
  --main_process_port 29500 \
  federatedscope/main.py \
  --cfg fedbiscuit_script/tldr/finetune_base.yaml \
  llm.accelerator.use True
