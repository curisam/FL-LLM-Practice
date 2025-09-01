which python
python - <<'PY'
import sys, torch
print("sys.executable:", sys.executable)
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
try:
    import torchvision
    print("torchvision:", torchvision.__version__, "위치:", getattr(torchvision, "__file__", "<없음>"))
except Exception as e:
    print("torchvision import error:", e)
try:
    import torchaudio
    print("torchaudio:", torchaudio.__version__, "위치:", getattr(torchaudio, "__file__", "<없음>"))
except Exception as e:
    print("torchaudio import error:", e)
PY

# conda가 뭐라고 생각하는지 확인
conda list | egrep 'torch|vision|audio|pytorch-cuda'
