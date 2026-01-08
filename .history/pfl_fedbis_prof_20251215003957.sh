#!/bin/bash

# ---- PYTHONPATH (레포 루트에서 실행한다고 가정)
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---- 커널 안전 모드(Flash/xFormers 끄고, SDPA는 코드에서 eager/수학커널)
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1

# ---- HF Hub 완전 로컬 (런 중 네트워크 접근 차단)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- NCCL/디버깅
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# ---- CUDA 할당자(파편화 완화)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ---- CPU 스레드 억제(랭크간 지터 감소)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- 실행 옵션(디버깅 플래그)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_MODULE_LOADING=EAGER
export FS_LOG_SUMMARY_ALL_RANKS=1
export TRANSFORMERS_VERBOSITY=error

GPU_LIST="4,5,6,7"
ACCEL_CFG="fedbiscuit_script/accelerator_config_bf16_ver2.yaml"
MAIN_PORT=29501
CFG1="fedbiscuit_script/tldr/finetune_fedbis.yaml"

LOG_DIR="runs_logs"
mkdir -p "${LOG_DIR}"

run_one () {
  local cfg="$1"
  local lr="$2"
  local step="$3"

  # 경로에 그대로 lr, step을 넣고 싶다고 했으니까 그대로 사용
  local ckpt_path="checkpoints_1.0/lr_${lr}_step_${step}/final_tldr_choice_qwen_fedbis_round_200.ckpt"
  local eval_outdir="exp/tldr/choice_qwen/pfl/lr_${lr}_step_${step}/fedbis_1.0/raw"
  local outdir="exp/tldr/choice_qwen/pfl/lr_${lr}_step_${step}/fedbis_1.0"

  # 필요한 디렉토리 생성 (ckpt 디렉토리는 미리 안 만들어져 있으면 여기서 만들어도 무방)
  mkdir -p "$(dirname "${ckpt_path}")" "${eval_outdir}" "${outdir}"

  local tag="pfl_finetune_lr_${lr}_step_${step}_$(date +%F_%H-%M-%S)"

  echo "=== [$(date +%T)] START: cfg=${cfg}, lr=${lr}, step=${step}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  accelerate launch \
    --config_file "${ACCEL_CFG}" \
    --main_process_port "${MAIN_PORT}" \
    federatedscope/main.py \
    --cfg "${cfg}" \
    model.load_from_local_pretrained_model_path "${ckpt_path}" \
    eval.outdir "${eval_outdir}" \
    outdir "${outdir}" \
    llm.accelerator.use True \
  2>&1 | tee -a "${LOG_DIR}/${tag}.log"
  echo "=== [$(date +%T)] DONE : cfg=${cfg}, lr=${lr}, step=${step}"
}

# ---- 여기서 탐색하고 싶은 lr / step 값들 정의
LR_LIST=("1e-6")        # 필요하면 "1e-7" 추가
STEP_LIST=(1 5 15 30)      # 필요하면 30 추가

for lr in "${LR_LIST[@]}"; do
  for step in "${STEP_LIST[@]}"; do
    run_one "${CFG1}" "${lr}" "${step}"
  done
done

echo "=== ALL DONE."





