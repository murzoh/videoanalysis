#!/bin/bash
set -euxo pipefail
exec >/root/boot.log 2>&1

echo "[boot] begin $(date)"

# --- Base system deps ---
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-pip git ffmpeg libgl1 libglib2.0-0 pciutils curl ca-certificates

# Allow pip to write into system site-packages on Ubuntu/Debian
python3 -m pip install --upgrade pip --break-system-packages || true

# --- Detect CUDA and pick torch index (cu124/cu121/cu118/cpu) ---
TAG="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  VER="$(nvidia-smi | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | awk '{print $3}' || true)"
  if [ -n "${VER}" ]; then
    M=${VER%.*}; m=${VER#*.}; N=$((10#$M*100 + 10#$m))
    if   [ "$N" -ge 1204 ]; then TAG="cu124"
    elif [ "$N" -ge 1201 ]; then TAG="cu121"
    elif [ "$N" -ge 1108 ]; then TAG="cu118"
    else TAG="cpu"; fi
  fi
fi
IDX="https://download.pytorch.org/whl/${TAG}"
[ "$TAG" = "cpu" ] && IDX="https://download.pytorch.org/whl/cpu"
echo "[boot] torch index = $IDX"

# --- Install PyTorch + friends ---
python3 -m pip install --break-system-packages --index-url "$IDX" torch torchvision

# Other libs (avoid Debian conflicts with --ignore-installed for blinker/itsdangerous)
python3 -m pip install --break-system-packages \
  fastapi "uvicorn[standard]" pillow transformers accelerate requests opencv-python \
  --upgrade --ignore-installed blinker itsdangerous

# --- Optional: choose model by VRAM unless env provided ---
if [ -z "${SMOLVLM2_MODEL:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || echo "0")
  if [ "$VRAM_MB" -ge 15000 ]; then
    export SMOLVLM2_MODEL="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
  else
    export SMOLVLM2_MODEL="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
  fi
fi
echo "[boot] SMOLVLM2_MODEL=${SMOLVLM2_MODEL:-unset}"

# --- Fetch server script from the repo (RAW URL) ---
cd /root
curl -fsSL https://raw.githubusercontent.com/murzoh/videoanalysis/main/server_smolvlm2.py -o server_smolvlm2.py
chmod +x server_smolvlm2.py

# Guard against CRLF if edited on Windows
sed -i 's/\r$//' server_smolvlm2.py

# --- Launch server in background and log it ---
nohup python3 /root/server_smolvlm2.py > /root/server.log 2>&1 &
echo $! > /root/server.pid

# Sentinel
echo ok >/root/STARTUP_WAS_HERE.txt
echo "[boot] done $(date)"
