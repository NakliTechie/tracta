#!/bin/sh
# One family: the provider serves BASE (published Q8_0); the planted changes are its own published
# Q4_K_M and a swap to another model. The reference bank is the provider's own baseline answers
# (history-based B1), scored later by one fixed tiny scorer, so no provider weights are needed.
#   sh campaign_family.sh <models-dir> <work-dir> <base.gguf> <downgrade.gguf> <swap.gguf>
# Paths to GGUF files are relative to <models-dir> and must be listed in models.json.
# CPU only. Every arm uses the same serving setting (4 slots, 4 concurrent requests).
set -eu
M=$1; W=$2; BASE=$3; DOWN=$4; SWAP=$5; HERE=$(cd "$(dirname "$0")" && pwd)
RUN="$W/runs.jsonl"
[ -e "$RUN" ] && { echo "refusing: $RUN exists; use a fresh work dir" >&2; exit 1; }
mkdir -p "$W"
printf '{"base": "%s", "downgrade": "%s", "swap": "%s"}\n' "$BASE" "$DOWN" "$SWAP" > "$W/family.json"
c() { python3 "$HERE/collect.py" --out "$RUN" --log "$W/server.log" --threads 8 --parallel 4 --concurrency 4 "$@"; }
c --model "$M/$BASE" --tag reference --sets open --open-samples 10
c --model "$M/$BASE" --tag q8-audit --sets open --open-samples 10
c --model "$M/$DOWN" --tag q4-audit --sets open --open-samples 5
c --model "$M/$SWAP" --tag swap-audit --sets open --open-samples 5
c --model "$M/$BASE" --tag q8-day1 --sets boundary,tokens
c --model "$M/$BASE" --tag q8-day2 --sets boundary,tokens --threads 4 --parallel 1 --concurrency 1
c --model "$M/$DOWN" --tag q4-day1 --sets boundary,tokens
c --model "$M/$SWAP" --tag swap-day1 --sets boundary,tokens
touch "$W/family.done"
