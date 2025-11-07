#!/bin/bash
set -euxo pipefail
# All output goes here:
exec >/root/boot.log 2>&1

echo "[boot] begin $(date)"

# --- Base system deps ---
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-pip git ffmpeg libgl1 libglib2.0-0 pciutils curl ca-certificates

# Allow pip to write to system site-packages on Debian/Ubuntu
python3 -m pip install --upgrade pip --break-system-packages || true

# ---------------- Torch install with automatic cu121 fallback ----------------
# 1) Detect CUDA tag for a first attempt
TAG="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  VER="$(nvidia-smi | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | awk '{print $3}' || true)"
  if [ -n "$VER" ]; then
    M=${VER%.*}; m=${VER#*.}; N=$((10#$M*100 + 10#$m))
    if   [ "$N" -ge 1204 ]; then TAG="cu124"
    elif [ "$N" -ge 1201 ]; then TAG="cu121"
    elif [ "$N" -ge 1108 ]; then TAG="cu118"
    else TAG="cpu"; fi
  fi
fi
echo "[boot] detected CUDA tag: $TAG"

# 2) Pick versions for the first attempt (known-good pairs)
FIRST_INDEX="https://download.pytorch.org/whl/cpu"
FIRST_TORCH="torch==2.5.1"
FIRST_VISION="torchvision==0.20.1"
if [ "$TAG" = "cu124" ]; then
  FIRST_INDEX="https://download.pytorch.org/whl/cu124"
  FIRST_TORCH="torch==2.6.0"
  FIRST_VISION="torchvision==0.21.0"
elif [ "$TAG" = "cu121" ]; then
  FIRST_INDEX="https://download.pytorch.org/whl/cu121"
  FIRST_TORCH="torch==2.5.1"
  FIRST_VISION="torchvision==0.20.1"
elif [ "$TAG" = "cu118" ]; then
  FIRST_INDEX="https://download.pytorch.org/whl/cu118"
  FIRST_TORCH="torch==2.4.1"
  FIRST_VISION="torchvision==0.19.1"
fi
echo "[boot] first index: $FIRST_INDEX  pair: $FIRST_TORCH / $FIRST_VISION"

# 3) Clean + install the first attempt
python3 -m pip uninstall -y torch torchvision torchaudio || true
python3 -m pip install --break-system-packages --index-url "$FIRST_INDEX" "$FIRST_TORCH" "$FIRST_VISION"

# 4) Import test; if it fails, force the cu121 pair
FALLBACK=0
python3 - <<'PY' || FALLBACK=1
print("Testing torch/vision import…")
import torch, torchvision
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
PY

if [ "$FALLBACK" = "1" ]; then
  echo "[boot] import failed; falling back to cu121 pair"
  python3 -m pip uninstall -y torch torchvision torchaudio || true
  python3 -m pip install --break-system-packages --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1" "torchvision==0.20.1"
  python3 - <<'PY'
import torch, torchvision
print("Fallback OK ->", torch.__version__, "/", torchvision.__version__)
PY
fi

# Optional: if you want to avoid torchvision entirely with Transformers, uncomment:
# export TRANSFORMERS_NO_TORCHVISION=1

# ---------------- Other Python deps ----------------
python3 -m pip install --break-system-packages \
  fastapi "uvicorn[standard]" pillow transformers accelerate requests opencv-python num2words \
  --upgrade --ignore-installed blinker itsdangerous

# ---------------- Model choice by VRAM (unless user overrides) ----------------
if [ -z "${SMOLVLM2_MODEL:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || echo "0")
  echo "[boot] GPU VRAM (MiB): $VRAM_MB"
  if [ "$VRAM_MB" -ge 15000 ]; then
    export SMOLVLM2_MODEL="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  else
    export SMOLVLM2_MODEL="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
  fi
fi
echo "[boot] SMOLVLM2_MODEL=${SMOLVLM2_MODEL:-unset}"

# ---------------- Fetch & launch server ----------------
cd /root
curl -fsSL https://raw.githubusercontent.com/murzoh/videoanalysis/main/server_smolvlm2.py -o server_smolvlm2.py
chmod +x server_smolvlm2.py
sed -i 's/\r$//' server_smolvlm2.py   # strip CRLF if any

# Run in background; logs at /root/server.log, pid at /root/server.pid
nohup python3 /root/server_smolvlm2.py > /root/server.log 2>&1 &
echo $! > /root/server.pid

# Sentinel + wrap up
echo ok > /root/STARTUP_WAS_HERE.txt
echo "[boot] done $(date)"
