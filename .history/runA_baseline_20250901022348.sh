cat > runA_baseline.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# (필요시) conda activate fs-llm
PORT=29500                      # 포트 겹치면 다른 번호로 변경
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1   # 디버그용(느리면 지워도 OK)
unset PYTORCH_CUDA_ALLOC_CONF   # allocator 완전 기본

ts=$(date +%Y%m%d_%H%M%S)
log="runA_baseline_${ts}.log"

echo "[INFO] PORT=$PORT, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
accelerate launch \
  --config_file fedbiscuit_script/accelerator_config_bf16.yaml \
  --main_process_port "$PORT" \
  federatedscope/main.py \
  --cfg fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml \
  llm.accelerator.use True 2>&1 | tee "$log"
SH
chmod +x runA_baseline.sh
