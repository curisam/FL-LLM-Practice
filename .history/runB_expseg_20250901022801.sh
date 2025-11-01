cat > runB_expseg.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# (필요시) conda activate fs-llm
PORT=29501
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ts=$(date +%Y%m%d_%H%M%S)
log="runB_expseg_${ts}.log"

echo "[INFO] PORT=$PORT, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
accelerate launch \
  --config_file fedbiscuit_script/accelerator_config_bf16.yaml \
  --main_process_port "$PORT" \
  federatedscope/main.py \
  --cfg fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml \
  llm.accelerator.use True 2>&1 | tee "$log"
SH
chmod +x runB_expseg.sh
