cat > runA_baseline.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
PORT=29500
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1        # 디버그용 (느리면 지워도 됨)
unset PYTORCH_CUDA_ALLOC_CONF        # allocator 기본
export PYTHONUNBUFFERED=1            # 파이썬 출력 버퍼링 끄기

ts=$(date +%Y%m%d_%H%M%S)
log="runA_baseline_${ts}.log"

echo "[INFO] PORT=$PORT, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
stdbuf -oL -eL \
accelerate launch \
  --config_file fedbiscuit_script/accelerator_config_bf16.yaml \
  --main_process_port "$PORT" \
  federatedscope/main.py \
  --cfg fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml \
  llm.accelerator.use True |& tee "$log"
SH
chmod +x runA_baseline.sh
