#!/usr/bin/env bash
set -euo pipefail

: "${PYTHONPATH:=}"
export PYTHONPATH="/home/seongyoon/jupyter/FedBiscuit${PYTHONPATH:+:${PYTHONPATH}}"

export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TERM=${TERM:-xterm-256color}
export PY_COLORS=1
export FORCE_COLOR=1

GPU_LIST="0,1,2,3"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_1gpu_0.yaml"
MAIN_PORT="${MAIN_PORT:-29501}"
CFG="fedbiscuit_script/tldr/fusion_moe_config.yaml"

# 여러 값 테스트
EMAS=(0.5 0.9 0.0)

run_one () {
  local ema="$1"
  local suffix
  suffix="$(echo "${ema}" | sed 's/\./_/g')"

  local save_path="checkpoints_1.0/tldr_choice_qwen_fusionmoe_u3_${suffix}.ckpt"
  local outdir_path="exp/tldr/choice_qwen/fusionmoe_u3_1.0_${suffix}"

  mkdir -p "$(dirname "$save_path")" "$(dirname "$outdir_path")"

  echo "=== [$(date +%T)] START: ema_beta=${ema}"

  # 1차 시도: llm.adapter.ema_beta 직접 오버라이드
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
  " /dev/null
  rc=$?
  set -e

  # 실패 시 tuple-index 방식 재시도
  if [[ $rc -ne 0 ]]; then
    echo ">>> direct override 실패 → tuple-index로 재시도: ema_beta.0=${ema}"
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
    " /dev/null
  fi

  echo "=== [$(date +%T)] DONE : ema_beta=${ema}"
}

for ema in "${EMAS[@]}"; do
  run_one "$ema"
done

echo "=== ALL DONE."
