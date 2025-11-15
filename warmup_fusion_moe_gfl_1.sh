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
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver1.yaml"
MAIN_PORT="${MAIN_PORT:-29500}"
CFG="fedbiscuit_script/tldr/warmup_fusion_moe_config.yaml"

# 여러 값 테스트
EMAS=(0.9 0.5)

run_one () {
  local ema="$1"
  
  # EMA 값을 문자열로 처리하여 그대로 경로에 반영
  local save_path="checkpoints_1.0/tldr_choice_qwen_warmup_fusionmoe_u3_${ema}.ckpt"
  local outdir_path="exp/tldr/choice_qwen/gfl/warmup_fusionmoe_u3_1.0_${ema}"

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

  echo "=== [$(date +%T)] DONE : ema_beta=${ema}"
}

for ema in "${EMAS[@]}"; do
  run_one "$ema"
done

echo "=== ALL DONE."
