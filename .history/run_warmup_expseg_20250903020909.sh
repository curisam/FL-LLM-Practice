#!/bin/bash
PORT=29501  # 충돌 시 다른 값으로 바꿔주세요

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
TORCH_USE_CUDA_DSA=1 \
CUDA_MODULE_LOADING=EAGER \
FS_LOG_SUMMARY_ALL_RANKS=1 \
TRANSFORMERS_VERBOSITY=error \
accelerate launch \
  --config_file fedbiscuit_script/accelerator_config_bf16.yaml \
  --main_process_port $PORT \
  federatedscope/main.py \
  --cfg fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml \
  llm.accelerator.use True
