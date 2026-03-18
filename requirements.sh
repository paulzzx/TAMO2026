#!/usr/bin/env bash
set -euo pipefail

# TAMO dependencies for CentOS 7 / glibc 2.17 / Python 3.9.
# Install order matters:
# 1) torch
# 2) PyG compiled extensions
# 3) remaining Python packages
#
# Removed from the old script because the repo does not import them:
# - torchvision
# - torchaudio
# - ogb
# - matplotlib
# - nvitop
#
# Optional:
# - pyg-lib: not imported directly by this repo, so it is skipped by default.

echo "[1/3] Installing PyTorch 2.2.1 (CUDA 11.8 wheels)..."
pip install torch==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "[2/3] Installing PyG extension wheels matched to torch 2.2 + cu118..."
pip install \
  torch-scatter==2.1.2+pt22cu118 \
  torch-sparse==0.6.18+pt22cu118 \
  torch-cluster==1.6.3+pt22cu118 \
  torch-spline-conv==1.2.2+pt22cu118 \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install torch-geometric==2.6.1

echo "[3/3] Installing the remaining runtime dependencies..."
pip install \
  transformers==4.36.2 \
  peft==0.8.2 \
  accelerate==0.27.2 \
  huggingface_hub==0.20.3 \
  sentencepiece==0.1.99 \
  datasets==2.16.1 \
  numpy==1.26.4 \
  pandas==1.5.3 \
  pyarrow==12.0.1 \
  scipy==1.10.1 \
  gensim==4.3.2 \
  wandb==0.16.6 \
  sacrebleu==2.4.0 \
  tqdm==4.66.2 \
  requests==2.31.0 \
  six==1.16.0 \
  pcst_fast==1.0.10

echo
echo "Done."
echo "Installed core versions:"
echo "  torch==2.2.1+cu118"
echo "  torch-geometric==2.6.1"
echo "  torch-sparse==0.6.18+pt22cu118"
echo "  torch-scatter==2.1.2+pt22cu118"
echo "  numpy==1.26.4"
echo "  pandas==1.5.3"
echo "  pyarrow==12.0.1"
echo "  gensim==4.3.2"
echo "  sentencepiece==0.1.99"
echo "  wandb==0.16.6"
echo
echo "Optional package not installed by default:"
echo "  pyg-lib==0.4.0+pt22cu118"
