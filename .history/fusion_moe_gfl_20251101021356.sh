#!/usr/bin/env bash
set -euo pipefail

# 환경 설정
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPU_LIST="3,4,5,6"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver2.yaml"
MAIN_PORT="${MAIN_PORT:-29500}"
APP_MAIN="federatedscope/main.py"
CFG="fedbiscuit_script/tldr/fusion_moe_config.yaml"

# Define the EMA values you want to try
EMAS=(0.9 0.5 0.0)

run_one () {
  local ema="$1"        # 예: 0.9  (← 숫자만 받는다고 가정)
  local suffix
  suffix="$(echo "${ema}" | sed 's/\./_/g')"   # 0.9 -> 0_9

  local save_path="checkpoints_1.0/tldr_choice_qwen_fusionmoe_u3_${suffix}.ckpt"
  local outdir_path="exp/tldr/choice_qwen/fusionmoe_u3_1.0_${suffix}"

  # 1) 임시 오버레이 YAML 생성 (튜플로 강제)
  local overlay="__ema_overlay_${suffix}.yaml"
  cat > "$overlay" <<EOF
llm:
  adapter:
    ema_beta: !!python/tuple [${ema}]
EOF

  echo "=== [$(date +%T)] START: ema_beta=(${ema},)"

  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --main_process_port "$MAIN_PORT" \
    "$APP_MAIN" \
    --cfg "$CFG" \
    --cfg "$overlay" \
    federate.save_to "$save_path" \
    outdir "$outdir_path" \
    llm.accelerator.use True \
  2>&1 | tee -a "runs_logs/fullmoe_ema${suffix}.log"

  rm -f "$overlay"

  echo "=== [$(date +%T)] DONE : ema_beta=(${ema},)"
}
