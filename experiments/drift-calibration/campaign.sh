#!/bin/sh
# One calibration campaign: the reference samples, three unchanged "days", one batched-serving
# variant, three days of the planted downgrade (Q4_K_M), and the bf16 original. CPU only.
#   sh campaign.sh <model-dir> <work-dir>
set -eu
M=$1; W=$2; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"
RUN="$W/runs.jsonl"; LOG="$W/server.log"
c() { python3 "$HERE/collect.py" --out "$RUN" --log "$LOG" "$@"; }

c --model "$M/qwen3-0.6b-Q8_0.gguf" --tag reference --sets open --open-samples 20
for d in 1 2 3; do c --model "$M/qwen3-0.6b-Q8_0.gguf" --tag "q8-day$d"; done
c --model "$M/qwen3-0.6b-Q8_0.gguf" --tag q8-batched --threads 4 --parallel 4 --concurrency 4
for d in 1 2 3; do c --model "$M/qwen3-0.6b-Q4_K_M.gguf" --tag "q4-day$d"; done
c --model "$M/qwen3-0.6b-bf16.gguf" --tag bf16
touch "$W/collect.done"
python3 "$HERE/analyse.py" --runs "$RUN" --scorer "$M/qwen3-0.6b-hf" --out "$W/results.json" > "$W/analyse.out" 2>&1
touch "$W/campaign.done"
