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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ---- CPU 스레드 억제(랭크간 지터 감소)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- 실행 옵션(디버깅 플래그)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_MODULE_LOADING=EAGER
export FS_LOG_SUMMARY_ALL_RANKS=1
export TRANSFORMERS_VERBOSITY=error

# ---- 사용자 설정(필요시 수정)
GPU_LIST="${GPU_LIST:-0,4,6,7}"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver3.yaml"
MAIN_PORT="${MAIN_PORT:-29500}"
APP_MAIN="federatedscope/main.py"

CFG1="fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml"
CFG2="fedbiscuit_script/tldr/tldr_choice_qwen_fedbis_oracle_u3_1.0.yaml"


LOG_DIR="${LOG_DIR:-runs_logs}"
mkdir -p "$LOG_DIR"

run_one () {
  local cfg="$1"
  local tag
  tag="$(basename "${cfg%%.yaml}")_$(date +%F_%H-%M-%S)"
  echo "=== [$(date +%T)] START: ${cfg}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --main_process_port "$MAIN_PORT" \
    "$APP_MAIN" \
    --cfg "$cfg" \
    llm.accelerator.use True \
  2>&1 | tee -a "${LOG_DIR}/${tag}.log"
  echo "=== [$(date +%T)] DONE : ${cfg}"
}

# ---- 순차 실행
run_one "$CFG1"
run_one "$CFG2"


echo "=== ALL DONE."
