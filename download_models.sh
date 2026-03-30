#!/usr/bin/env bash
set -euo pipefail

# Download the local model assets expected by TAMO.
#
# Runtime model key -> local directory:
#   7b                 -> ./models/meta-llama/Llama-2-7b-hf
#   7b_chat            -> ./models/meta-llama/Llama-2-7b-chat
#   table_llama_7b     -> ./models/tablellama
#   mistral_7b         -> ./models/mistralai/Mistral-7B-v0.1
#   mistral_7b_instruct-> ./models/mistralai/Mistral-7B-Instruct-v0.2
#
# Auxiliary local assets used elsewhere in the repo:
#   ./models/sentence-transformers/all-roberta-large-v1
#   ./models/google-bert/bert-base-uncased
#
# Note:
# - Llama 2 checkpoints are gated on Hugging Face and require authentication.
# - The runtime also accepts the legacy alias
#   ./models/meta-llama/Llama-2-7b-chat-hf for 7b_chat.

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install huggingface_hub and log in before running this script." >&2
  exit 1
fi

echo "[1/3] Downloading auxiliary local assets..."
huggingface-cli download --resume-download sentence-transformers/all-roberta-large-v1 --local-dir ./models/sentence-transformers/all-roberta-large-v1
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir ./models/google-bert/bert-base-uncased

echo "[2/3] Downloading Llama-family local assets..."
huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf --local-dir ./models/meta-llama/Llama-2-7b-hf
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat --local-dir ./models/meta-llama/Llama-2-7b-chat
huggingface-cli download --resume-download osunlp/TableLlama --local-dir ./models/tablellama

echo "[3/3] Downloading Mistral local assets..."
huggingface-cli download --resume-download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistralai/Mistral-7B-v0.1
huggingface-cli download --resume-download mistralai/Mistral-7B-Instruct-v0.2 --local-dir ./models/mistralai/Mistral-7B-Instruct-v0.2

echo
echo "Model download complete."
