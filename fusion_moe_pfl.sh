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

GPU_LIST="4,5,6,7"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver2.yaml"
MAIN_PORT="${MAIN_PORT:-29501}"
CFG="fedbiscuit_script/tldr/finetune_fusion_moe_config.yaml"

# 여러 값 테스트
EMAS=(0.9 0.5 0.0)

run_one () {
  local ema="$1"
  local outdir_path="exp/tldr/choice_qwen/pfl/fusionmoe_u3_1.0_${ema}"

  # 동적으로 경로 업데이트
  local load_grouping_weights_path="exp/gfl/fusionmoe_u3_1.0_${ema}/grouping_weights_round175_beta${ema}.json"
  local pretrained_model_path="checkpoints_1.0/final_tldr_choice_qwen_fusionmoe_u3_${ema}_round_200.ckpt"
  local eval_outdir="exp/tldr/choice_qwen/pfl/fusionmoe_u3_1.0_${ema}/raw"
  local save_outdir="exp/tldr/choice_qwen/pfl/fusionmoe_u3_1.0_${ema}/"

  # 디렉토리 생성
  mkdir -p "$outdir_path" "$save_outdir"

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
      llm.adapter.load_grouping_weights_path \"$load_grouping_weights_path\" \
      model.load_from_local_pretrained_model_path \"$pretrained_model_path\" \
      eval.outdir \"$eval_outdir\" \
      outdir \"$save_outdir\" \
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
