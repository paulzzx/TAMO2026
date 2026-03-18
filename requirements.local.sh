#!/usr/bin/env bash
set -euo pipefail

# TAMO local dependencies for macOS development.
# This is a looser environment than the Linux server setup and is intended for:
# - data preprocessing
# - dataset loading
# - pure-text model path debugging
# - general code-path validation
#
# Recommended local model choices:
#   --model_name llm
#   --model_name pt_llm
#   --model_name mistral
#   --model_name pt_mistral
#
# If you need hypergraph models locally, install platform-specific PyG extensions
# separately after this script.

echo "[1/2] Installing local core dependencies..."
pip install -r requirements.local.txt

echo "[2/2] Local environment ready."
echo "If you need hypergraph models on macOS, install these separately if available:"
echo "  torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib"
