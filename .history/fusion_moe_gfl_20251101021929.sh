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

# 모듈 실행(경로 실행 대신) ▶ yacs의 CLI 파서가 더 안정적으로 동작
# APP_MAIN="federatedscope/main.py"
CFG="fedbiscuit_script/tldr/fusion_moe_config.yaml"

# 실험할 EMA 값 (숫자만)
EMAS=(0.9 0.5 0.0)

run_one () {
  local ema="$1"
  local suffix
  suffix="$(echo "${ema}" | sed 's/\./_/g')"   # 0.9 -> 0_9

  local save_path="checkpoints_1.0/tldr_choice_qwen_fusionmoe_u3_${suffix}.ckpt"
  local outdir_path="exp/tldr/choice_qwen/fusionmoe_u3_1.0_${suffix}"
  local log_path="runs_logs/fullmoe_ema${suffix}.log"

  # 출력/로그 디렉터리 보장
  mkdir -p "$(dirname "$save_path")" "$(dirname "$outdir_path")" "runs_logs"

  echo "=== [$(date +%T)] START: ema_beta=(${ema},)"

  # 핵심: overlay 제거, 모듈 실행 + CLI에 파이썬 '튜플' 리터럴 그대로 전달
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  accelerate launch \
    --config_file "$ACCEL_CFG" \
    --main_process_port "$MAIN_PORT" \
    --module federatedscope.main \
    --cfg "$CFG" \
    llm.adapter.ema_beta "(${ema},)" \
    federate.save_to "$save_path" \
    outdir "$outdir_path" \
    llm.accelerator.use True \
  2>&1 | tee -a "$log_path"

  echo "=== [$(date +%T)] DONE : ema_beta=(${ema},)"
}

for ema in "${EMAS[@]}"; do
  run_one "$ema"
done

echo "=== ALL DONE."