#!/usr/bin/env bash
set -euo pipefail

# Server smoke test for TAMO.
#
# Defaults:
#   SMOKE_MODEL_NAME=7b
#   SMOKE_PROMPT_TYPE=llama2
#   SMOKE_DATASET=structprobe
#
# Examples:
#   bash server_smoke_test.sh
#   SMOKE_MODEL_NAME=mistral_7b SMOKE_PROMPT_TYPE=mistral SMOKE_DATASET=wtq_orig bash server_smoke_test.sh
#   SMOKE_SKIP_MODEL=1 bash server_smoke_test.sh

export SMOKE_MODEL_NAME="${SMOKE_MODEL_NAME:-7b}"
export SMOKE_PROMPT_TYPE="${SMOKE_PROMPT_TYPE:-llama2}"
export SMOKE_DATASET="${SMOKE_DATASET:-structprobe}"
export SMOKE_SKIP_MODEL="${SMOKE_SKIP_MODEL:-0}"

echo "Running TAMO server smoke test..."
echo "  SMOKE_MODEL_NAME=${SMOKE_MODEL_NAME}"
echo "  SMOKE_PROMPT_TYPE=${SMOKE_PROMPT_TYPE}"
echo "  SMOKE_DATASET=${SMOKE_DATASET}"
echo "  SMOKE_SKIP_MODEL=${SMOKE_SKIP_MODEL}"

python server_smoke_test.py
