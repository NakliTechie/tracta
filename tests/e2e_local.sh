#!/bin/sh
# End-to-end: drive the real CLI against local llama-server "providers" built from the pinned files.
# Baseline twice on the Q8_0, then check the Q8_0 (expect no change), its published Q4_K_M and a
# smaller sibling (expect changed). CPU only.
#   sh tests/e2e_local.sh <models-dir> <work-dir> [base] [downgrade] [swap]
# TRACTA_BIN selects the tool under test (default: the Python package; e.g. TRACTA_BIN=dist/tracta for the Go binary).
set -u
M=$1; W=$2
BASE=${3:-gemma-3-1b/gemma-3-1b-it-Q8_0.gguf}; DOWN=${4:-gemma-3-1b/gemma-3-1b-it-Q4_K_M.gguf}; SWAP=${5:-gemma-3-270m/gemma-3-270m-it-Q8_0.gguf}
PORT=18791; ROOT=$W/watches; HERE=$(cd "$(dirname "$0")/.." && pwd)
[ -e "$ROOT" ] && { echo "refusing: $ROOT exists" >&2; exit 1; }
mkdir -p "$W"
serve() {
  llama-server -m "$M/$1" --device none -ngl 0 --no-op-offload --host 127.0.0.1 --port $PORT -c 2048 --jinja > "$W/server-$(basename $1).log" 2>&1 &
  SP=$!
  until curl -s 127.0.0.1:$PORT/health | grep -q ok; do kill -0 $SP 2>/dev/null || { echo "server failed: $1" >&2; exit 1; }; sleep 1; done
}
stop() { kill $SP; wait $SP 2>/dev/null; }
t() {
  sub=$1; shift
  if [ -n "${TRACTA_BIN:-}" ]; then (cd "$HERE" && "$TRACTA_BIN" "$sub" --root "$ROOT" "$@")
  else (cd "$HERE" && python3 -m tracta --root "$ROOT" "$sub" "$@"); fi
}
fail=0
expect() { # expect <exit code> <label> -- tracta args
  want=$1; label=$2; shift 3; t "$@"; got=$?
  if [ "$got" = "$want" ]; then echo "PASS $label (exit $got)"; else echo "FAIL $label (exit $got, want $want)"; fail=1; fi
}
serve "$BASE"
t baseline --name e2e --endpoint http://127.0.0.1:$PORT/v1 --model base --rps 0 --logprobs
expect 3 "one baseline gives no verdict" -- check --name e2e --rps 0
t baseline --name e2e --rps 0 --logprobs
expect 0 "unchanged Q8_0" -- check --name e2e --rps 0
stop
serve "$DOWN"; expect 2 "published Q4_K_M" -- check --name e2e --rps 0; stop
serve "$SWAP"; expect 2 "swap to $(basename $SWAP)" -- check --name e2e --rps 0; stop
exit $fail
