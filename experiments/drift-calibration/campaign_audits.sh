#!/bin/sh
# Fresh audits for operational false-alarm and detection rates (B1): 10 new answers per open prompt
# for the unchanged model and the downgrade, on both engines. analyse.py cuts each answer sample into
# disjoint audits (8 audits of 25 prompts per sample, so 80 per arm), scored against the frozen
# reference bank from campaign.sh.
#   sh campaign_audits.sh <model-dir> <work-dir>      (run with the project .venv's python on PATH)
set -eu
M=$1; W=$2; HERE=$(cd "$(dirname "$0")" && pwd)
RUN="$W/runs-audits.jsonl"
[ -e "$RUN" ] && { echo "refusing: $RUN exists; use a fresh work dir" >&2; exit 1; }
mkdir -p "$W"
python "$HERE/collect.py" --model "$M/qwen3-0.6b-Q8_0.gguf" --tag q8-audit --sets open --open-samples 10 --out "$RUN" --log "$W/server.log"
python "$HERE/collect.py" --model "$M/qwen3-0.6b-Q4_K_M.gguf" --tag q4-audit --sets open --open-samples 10 --out "$RUN" --log "$W/server.log"
h() { python "$HERE/collect_hf.py" --model-dir "$M" --tokenizer "$M/qwen3-0.6b-hf" --out "$RUN" --sets open --open-samples 10 --batch 8 "$@"; }
h --gguf qwen3-0.6b-Q8_0.gguf --tag q8-hfaudit
h --gguf qwen3-0.6b-Q4_K_M.gguf --tag q4-hfaudit
touch "$W/audits.done"
