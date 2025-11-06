#!/usr/bin/env bash
set -euo pipefail

: "${PYTHONPATH:=}"
export PYTHONPATH="/home/seongyoon/jupyter/FedBiscuit${PYTHONPATH:+:${PYTHONPATH}}"

export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 색상 강제(파이프/원격에서도 컬러 유지)
export TERM=${TERM:-xterm-256color}
export PY_COLORS=1
export FORCE_COLOR=1

GPU_LIST="4"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_1gpu_0.yaml"
MAIN_PORT="${MAIN_PORT:-29501}"
CFG="fedbiscuit_script/tldr/fusion_moe_config.yaml"

EMAS=(0.5)

run_one () {
  local ema="$1"
  local suffix
  suffix="$(echo "${ema}" | sed 's/\./_/g')"

  local save_path="checkpoints_1.0/tldr_choice_qwen_fusionmoe_u3_${suffix}.ckpt"
  local outdir_path="exp/tldr/choice_qwen/fusionmoe_u3_1.0_${suffix}"
  local log_path="runs_logs/fullmoe_ema${suffix}.log"

  mkdir -p "$(dirname "$save_path")" "$(dirname "$outdir_path")" runs_logs

  echo "=== [$(date +%T)] START: ema_beta=${ema} (try float override)"

  # --- 시도 1: float로 직접 세팅 (pty 부여 + tee 로깅 유지)
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  script -q -c "
    accelerate launch \
      --config_file \"$ACCEL_CFG\" \
      --main_process_port \"$MAIN_PORT\" \
      --module federatedscope.main \
      --cfg \"$CFG\" \
      llm.adapter.ema_beta \"$ema\" \
      federate.save_to \"$save_path\" \
      outdir \"$outdir_path\" \
      llm.accelerator.use True
  " /dev/null | tee -a "$log_path"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    echo ">>> float override 실패. tuple-index override로 재시도: ema_beta.0=${ema}"
    # --- 시도 2: tuple/list의 0번 원소 덮어쓰기 (동일하게 pty 부여)
    CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    script -q -c "
      accelerate launch \
        --config_file \"$ACCEL_CFG\" \
        --main_process_port \"$MAIN_PORT\" \
        --module federatedscope.main \
        --cfg \"$CFG\" \
        llm.adapter.ema_beta.0 \"$ema\" \
        federate.save_to \"$save_path\" \
        outdir \"$outdir_path\" \
        llm.accelerator.use True
    " /dev/null | tee -a "$log_path"
  fi

  echo "=== [$(date +%T)] DONE : ema_beta=${ema}"
}

for ema in "${EMAS[@]}"; do
  run_one "$ema"
done

echo "=== ALL DONE."
