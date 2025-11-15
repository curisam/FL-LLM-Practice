#!/usr/bin/env bash
set -euo pipefail

############## [ 사용자 옵션 ] ##############
# (A) 로컬 모델 경로가 있을 때 지정 (권장)
# 예) /home/seongyoon/models/Qwen2-0.5B
MODEL_DIR="${MODEL_DIR:-}"      # 비워두면 허브 ID 사용
MODEL_ID="${MODEL_ID:-Qwen/Qwen2-0.5B}"

# (B) 완전 오프라인 환경인데 캐시가 비어있다면,
#     1회만 온라인으로 받아 캐시를 채울지 여부 (0/1)
ONLINE_WARMUP="${ONLINE_WARMUP:-0}"

# 캐시 루트(공용 디스크 쓰고 싶으면 변경)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
#############################################

# ---- 커널 안전 모드(Flash/xFormers 끄기)
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
GPU_LIST="${GPU_LIST:-0}"
ACCEL_CFG="${ACCEL_CFG:-fedbiscuit_script/accelerator_config_bf16_1gpu_0.yaml}"
MAIN_PORT="${MAIN_PORT:-29500}"
APP_MAIN="${APP_MAIN:-federatedscope.main}"
CFG1="${CFG1:-fedbiscuit_script/tldr/tldr_choice_qwen_fedbis_1.0.yaml}"

LOG_DIR="${LOG_DIR:-runs_logs}"
mkdir -p "$LOG_DIR"

# ---------- 유틸: 모델 준비 ----------
have_config_json() {
  local path="$1"
  [ -f "$path/config.json" ] && return 0
  # transformers 캐시 구조에 들어있는 경우도 커버
  find "$path" -maxdepth 3 -name 'config.json' | grep -q .
}

ensure_model_ready() {
  # 1) 로컬 경로를 직접 쓸 경우
  if [[ -n "$MODEL_DIR" ]]; then
    if ! have_config_json "$MODEL_DIR"; then
      echo "[ERR] MODEL_DIR='$MODEL_DIR' 에 config.json이 없습니다." >&2
      echo "      로컬로 모델을 받아 두거나 경로를 확인하세요." >&2
      exit 1
    fi
    echo "[INFO] 로컬 모델 사용: $MODEL_DIR"
    return 0
  fi

  # 2) 허브 ID를 쓸 경우: 오프라인 캐시에 존재하는지 확인
  #    캐시 구조상 모델 폴더명을 찾기 어려우니 단순히 캐시 아래에서 모델 아이디 조각 탐색
  if find "$HF_HOME" -maxdepth 5 -type f -name 'config.json' | grep -q 'Qwen2-0.5B'; then
    echo "[INFO] 캐시된 모델 발견(Qwen2-0.5B). 오프라인으로 진행합니다."
    return 0
  fi

  # 3) 캐시에 없고, 1회 워밍업 허용 시
  if [[ "$ONLINE_WARMUP" == "1" ]]; then
    echo "[INFO] 캐시에 모델이 없어 1회 온라인 워밍업을 수행합니다."
    # 임시로 오프라인 플래그 해제
    ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
      python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2-0.5B", local_files_only=False)
print("[INFO] snapshot_download 완료")
PY
    )
    echo "[INFO] 워밍업 완료. 다시 오프라인으로 실행합니다."
    return 0
  fi

  # 4) 캐시에 없고 워밍업도 거부된 경우
  echo "[ERR] 오프라인 모드인데 캐시에 모델이 없습니다."
  echo "      옵션 중 하나를 사용하세요:"
  echo "      (A) MODEL_DIR=/path/to/Qwen2-0.5B 를 지정해 로컬 경로 사용"
  echo "      (B) ONLINE_WARMUP=1 로 1회 캐시 워밍업 허용"
  exit 1
}

run_one () {
  local cfg="$1"
  local tag
  tag="$(basename "${cfg%%.yaml}")_$(date +%F_%H-%M-%S)"

  # 실행 전 모델 준비
  ensure_model_ready

  # 추가 cfg override 구성
  # - 로컬 경로가 있으면 허브 ID 대신 강제로 주입
  # - Qwen 계열은 trust_remote_code가 필요한 경우가 있어 True로 넘김
  local extra_overrides=(
    "llm.accelerator.use" "True"
    "llm.trust_remote_code" "True"
  )
  if [[ -n "$MODEL_DIR" ]]; then
    extra_overrides+=("llm.model_name" "$MODEL_DIR")
  else
    extra_overrides+=("llm.model_name" "$MODEL_ID")
  fi

  echo "=== [$(date +%T)] START: ${cfg}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --main_process_port "$MAIN_PORT" \
    --module "$APP_MAIN" \
    --cfg "$cfg" \
    "${extra_overrides[@]}" \
  2>&1 | tee -a "${LOG_DIR}/${tag}.log"
  echo "=== [$(date +%T)] DONE : ${cfg}"
}

# ---- 순차 실행
run_one "$CFG1"

echo "=== ALL DONE."
