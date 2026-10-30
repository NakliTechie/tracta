#!/bin/sh
# The 3-4B family campaigns, same design as families-small.sh. Paths are relative to the tracta
# models root: 3-4B files sit in <family>/, 0-2B files in published/<family>/.
#   sh families-4b.sh <models-root> <work-root>
set -eu
M=$1; R=$2; HERE=$(cd "$(dirname "$0")" && pwd)
run() { [ -e "$R/$1/family.done" ] || sh "$HERE/campaign_family.sh" "$M" "$R/$1" "$2" "$3" "$4" || echo "family $1 failed; continuing" >&2; }
run qwen3.5-4b qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf published/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf
run smollm3-3b smollm3-3b/SmolLM3-3B-Q8_0.gguf smollm3-3b/SmolLM3-3B-Q4_K_M.gguf published/smollm2-1.7b/SmolLM2-1.7B-Instruct-Q8_0.gguf
run llama-3.2-3b llama-3.2-3b/Llama-3.2-3B-Instruct-Q8_0.gguf llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf published/llama-3.2-1b/Llama-3.2-1B-Instruct-Q8_0.gguf
run phi-4-mini phi-4-mini/Phi-4-mini-instruct.Q8_0.gguf phi-4-mini/Phi-4-mini-instruct-Q4_K_M.gguf qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf
touch "$R/families-4b.done"
