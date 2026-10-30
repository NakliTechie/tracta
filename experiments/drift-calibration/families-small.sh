#!/bin/sh
# The 0-2B family campaigns: base (published Q8_0) | downgrade (its published Q4_K_M) | swap.
# Swaps go to a smaller sibling where the family publishes one, else to a similar-size model of
# another family.
#   sh families-small.sh <models-dir> <work-root>
set -eu
M=$1; R=$2; HERE=$(cd "$(dirname "$0")" && pwd)
run() { [ -e "$R/$1/family.done" ] || sh "$HERE/campaign_family.sh" "$M" "$R/$1" "$2" "$3" "$4" || echo "family $1 failed; continuing" >&2; }
run smollm2-1.7b smollm2-1.7b/SmolLM2-1.7B-Instruct-Q8_0.gguf smollm2-1.7b/SmolLM2-1.7B-Instruct-Q4_K_M.gguf smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf
run gemma-3-1b gemma-3-1b/gemma-3-1b-it-Q8_0.gguf gemma-3-1b/gemma-3-1b-it-Q4_K_M.gguf gemma-3-270m/gemma-3-270m-it-Q8_0.gguf
run qwen3.5-0.8b qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
run llama-3.2-1b llama-3.2-1b/Llama-3.2-1B-Instruct-Q8_0.gguf llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
run lfm2.5-350m lfm2.5-350m/LFM2.5-350M-Q8_0.gguf lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf
run qwen3-0.6b qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf lfm2.5-350m/LFM2.5-350M-Q8_0.gguf
touch "$R/families-small.done"
