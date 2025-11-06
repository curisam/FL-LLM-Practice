#!/usr/bin/env bash
set -euo pipefail

# ---- 커널 안전 모드
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1

# ---- HF Hub 완전 로컬
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

# ---- 사용자 설정
GPU_LIST="${GPU_LIST:-4,5,6,7}"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver1.yaml"
MAIN_PORT="${MAIN_PORT:-29501}"

# ★ 파일 경로(X) → 모듈 이름(O)
APP_MAIN="federatedscope.main"

CFG1="fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u10_warmup_1.0.yaml"


LOG_DIR="${LOG_DIR:-runs_logs}"
mkdir -p "$LOG_DIR"

# ---- 사전 점검: import federatedscope
python - <<'PY' || { echo "[FATAL] cannot import federatedscope"; exit 1; }
import federatedscope, sys
print("[OK] federatedscope at:", federatedscope.__file__)
print("[OK] FedBiscuit in sys.path?:", any("FedBiscuit" in p for p in sys.path))
PY

run_one () {
  local cfg="$1"
  local tag
  tag="$(basename "${cfg%%.yaml}")_$(date +%F_%H-%M-%S)"
  echo "=== [$(date +%T)] START: ${cfg}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --main_process_port "$MAIN_PORT" \
    --module "$APP_MAIN" \
    --cfg "$cfg" \
    llm.accelerator.use True \
  2>&1 | tee -a "${LOG_DIR}/${tag}.log"
  echo "=== [$(date +%T)] DONE : ${cfg}"
}

# ---- 순차 실행
run_one "$CFG1"


echo "=== ALL DONE."