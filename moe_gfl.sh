#!/usr/bin/env bash
set -euo pipefail

# --- 안전한 PYTHONPATH 초기화 (set -u 대응)
: "${PYTHONPATH:=}"
export PYTHONPATH="/home/seongyoon/jupyter/FedBiscuit${PYTHONPATH:+:${PYTHONPATH}}"

# --- 환경 설정
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPU_LIST="3,4,5,6"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver2.yaml"
MAIN_PORT="${MAIN_PORT:-29500}"
CFG="fedbiscuit_script/tldr/full_moe_config.yaml"

# ✅ 쉼표 없이 정의 (bash 배열은 요소 사이에 공백)
EMAS=(0.9 0.5 0.0)

run_one () {
  local ema="$1"
  local suffix
  suffix="$(echo "${ema}" | sed 's/\./_/g')"   # 0.5 -> 0_5

  local save_path="checkpoints_1.0/tldr_choice_qwen_fullmoe_u3_${suffix}.ckpt"
  local outdir_path="exp/tldr/choice_qwen/fullmoe_u3_1.0_${suffix}"
  local log_path="runs_logs/fullmoe_ema${suffix}.log"

  # 출력/로그 디렉터리 보장
  mkdir -p "$(dirname "$save_path")" "$(dirname "$outdir_path")" runs_logs

  echo "=== [$(date +%T)] START: ema_beta=${ema} (try float override)"

  # --- 시도 1: 스키마가 float인 경우
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --main_process_port "$MAIN_PORT" \
    --module federatedscope.main \
    --cfg "$CFG" \
    llm.adapter.ema_beta "${ema}" \
    federate.save_to "$save_path" \
    outdir "$outdir_path" \
    llm.accelerator.use True \
  2>&1 | tee -a "$log_path"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    echo ">>> float override 실패. tuple-index override로 재시도: ema_beta.0=${ema}"
    # --- 시도 2: 스키마가 tuple/list인 경우 0번째 원소만 덮어씀
    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    accelerate launch \
      --config_file "$ACCEL_CFG" \
      --main_process_port "$MAIN_PORT" \
      --module federatedscope.main \
      --cfg "$CFG" \
      llm.adapter.ema_beta.0 "${ema}" \
      federate.save_to "$save_path" \
      outdir "$outdir_path" \
      llm.accelerator.use True \
    2>&1 | tee -a "$log_path"
  fi

  echo "=== [$(date +%T)] DONE : ema_beta=${ema}"
}

for ema in "${EMAS[@]}"; do
  run_one "$ema"
done

echo "=== ALL DONE."
