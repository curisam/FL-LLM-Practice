# 권장: 로컬 단일 노드 고정
export MASTER_ADDR=127.0.0.1

# 디버그 환경변수
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# allocator 실험 옵션은 빼둠 (OK)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=29500 \
  federatedscope/main.py \
  --cfg fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml \
  federate.master_port 29500 \
  llm.accelerator.use True 2>&1 | tee crash.log