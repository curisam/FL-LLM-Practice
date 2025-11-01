#!/usr/bin/env bash
set -euo pipefail

# ======= 사용자 설정 (필요시 수정) =======
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29501
CFG="fedbiscuit_script/tldr/tldr_choice_qwen_fedbiscuit_u3_1.0.yaml"
MAINPY="federatedscope/main.py"
LOGDIR="debug_artifacts_$(date +%Y%m%d_%H%M%S)"
# allocator 옵션은 일단 비활성(불안정 회피). 필요하면 아래 줄 주석 해제하여 대체 사용.
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
# =====================================

mkdir -p "$LOGDIR"
echo "[*] Logs will be saved in: $LOGDIR"

# 0) 디버그 환경변수
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# 0) 디버그 환경변수 아래에 추가
export MASTER_ADDR=127.0.0.1
# 단일 노드에서 NCCL 이슈 의심시 다음 2줄도 시도 가능
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1

# 1) 버전/환경 수집
echo "[*] Collecting environment info..."
python -m torch.utils.collect_env | tee "$LOGDIR/env_collect.txt" || true
accelerate env | tee "$LOGDIR/accelerate_env.txt" || true
python - <<'PY' | tee "$LOGDIR/py_versions.txt" || true
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
try:
    import accelerate; print("accelerate:", accelerate.__version__)
except Exception: print("accelerate: <none>")
try:
    import transformers; print("transformers:", transformers.__version__)
except Exception: print("transformers: <none>")
try:
    import peft; print("peft:", peft.__version__)
except Exception: print("peft: <none>")
try:
    import bitsandbytes as bnb; print("bitsandbytes:", getattr(bnb,'__version__','?'))
except Exception: print("bitsandbytes: <none>")
PY
nvidia-smi | tee "$LOGDIR/nvidia_smi.txt" || true
nvidia-smi topo -m | tee "$LOGDIR/nvidia_topo.txt" || true

python - <<'PY' | tee "$LOGDIR/dist_env.txt"
import os
print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
PY

# 2) 코어덤프 허용(가능한 경우)
ulimit -c unlimited || true
echo "[*] core_pattern: $(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo '<unknown>')"

# 3) 학습 실행 (torchrun 사용, 없으면 python -m fallback)
echo "[*] Starting training with torchrun..."
export CUDA_VISIBLE_DEVICES="$GPUS"

RUN_CMD=()
if command -v torchrun >/dev/null 2>&1; then
  RUN_CMD=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "$MAINPY" --cfg "$CFG" federate.master_port "$MASTER_PORT" llm.accelerator.use True)
else
RUN_CMD=(python -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "$MAINPY" --cfg "$CFG" federate.master_port "$MASTER_PORT" llm.accelerator.use True)
fi

# 크래시 여부와 상관없이 이후 단계 진행하도록 상태코드 저장
set +e
"${RUN_CMD[@]}" 2>&1 | tee "$LOGDIR/crash.log"
RET=$?
set -e
echo "[*] Training exited with code: $RET"

# 4) 로그 요약본 추출
echo "[*] Extracting crash log snippets..."
head -n 60 "$LOGDIR/crash.log" > "$LOGDIR/crash_head.txt" || true
tail -n 200 "$LOGDIR/crash.log" > "$LOGDIR/crash_tail.txt" || true
grep -nE "ERROR|Signal|SIGSEGV|Segmentation|Traceback|RuntimeError|NCCL|torch.distributed" "$LOGDIR/crash.log" \
  | tail -n 200 > "$LOGDIR/crash_grep.txt" || true

# 5) 코어덤프가 있다면 gdb 백트레이스 추출
COREF=$(ls -t core* 2>/dev/null | head -n1 || true)
if [ -n "${COREF:-}" ]; then
  echo "[*] Found core file: $COREF — generating gdb backtrace..."
  gdb -q "$(which python)" "$COREF" \
    -ex "set pagination off" \
    -ex "thread apply all bt" \
    -ex "quit" | tee "$LOGDIR/gdb_bt.txt" || true
else
  # systemd-coredump 경로
  if command -v coredumpctl >/dev/null 2>&1; then
    echo "[*] Trying systemd-coredump (coredumpctl)..."
    coredumpctl list | tail -n 10 | tee "$LOGDIR/coredump_list.txt" || true
    coredumpctl info python | tee "$LOGDIR/coredump_info.txt" || true
    coredumpctl gdb python -q \
      -ex "set pagination off" -ex "thread apply all bt" -ex "quit" \
      | tee "$LOGDIR/gdb_bt.txt" || true
  fi
fi

echo ""
echo "================ SUMMARY ================"
echo "Log dir: $LOGDIR"
echo "- crash.log          (full run log)"
echo "- crash_head.txt     (first 60 lines)"
echo "- crash_tail.txt     (last 200 lines)"
echo "- crash_grep.txt     (errors/NCCL/stack keywords)"
echo "- env_collect.txt, accelerate_env.txt, py_versions.txt"
echo "- nvidia_smi.txt, nvidia_topo.txt, dist_env.txt"
echo "- gdb_bt.txt (if available), coredump_* (if available)"
echo "========================================"
echo ""
echo "Next: 위 파일들 중에서 특히"
echo "  crash_tail.txt, crash_grep.txt, gdb_bt.txt(있으면)"
echo "  그리고 env_collect.txt/accelerate_env.txt/nvidia_topo.txt"
echo "이 6개를 나한테 붙여주면 바로 파악 가능!"
