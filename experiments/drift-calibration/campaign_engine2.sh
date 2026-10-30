#!/bin/sh
# Second-engine arms for the noisy null: the same GGUF files through transformers on CPU.
#   sh campaign_engine2.sh <model-dir> <work-dir>      (run with the project .venv's python on PATH)
set -eu
M=$1; W=$2; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"
c() { python "$HERE/collect_hf.py" --model-dir "$M" --tokenizer "$M/qwen3-0.6b-hf" --out "$W/runs-engine2.jsonl" "$@"; }
c --gguf qwen3-0.6b-Q8_0.gguf --tag q8-hf1 --batch 1
c --gguf qwen3-0.6b-Q8_0.gguf --tag q8-hf8 --batch 8
c --gguf qwen3-0.6b-Q4_K_M.gguf --tag q4-hf1 --batch 1
touch "$W/engine2.done"
