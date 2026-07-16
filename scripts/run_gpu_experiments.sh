#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

# Use the venv created by prepare_gpu.sh; fall back to python3 for
# environments that manage dependencies themselves.
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BASE="${BASE_CHECKPOINT:-checkpoints/minimoe_step_0019073.pt}"
SFT="${SFT_CHECKPOINT:-checkpoints/minimoe_sft.pt}"
mkdir -p results

run_distributed() {
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$@"
  else
    "$PYTHON" "$@"
  fi
}

if [[ "$MODE" == "smoke" ]]; then
  run_distributed src/experiments.py smoke --base "$BASE" --sft "$SFT" --output results/smoke.json
  exit 0
fi

run_distributed src/experiments.py hellaswag --checkpoint "$BASE" --output results/hellaswag_base.json
run_distributed src/experiments.py hellaswag --checkpoint "$SFT" --output results/hellaswag_sft.json
run_distributed src/experiments.py routing --checkpoint "$BASE" --output results/routing_base.json
run_distributed src/experiments.py routing --checkpoint "$SFT" --output results/routing_sft.json
run_distributed src/experiments.py ablations --checkpoint "$BASE" --output results/ablations.json
run_distributed src/experiments.py prompts --base "$BASE" --sft "$SFT" --output results/prompt_outputs.json
"$PYTHON" src/plot_experiments.py --results-dir results --figures-dir results/figures
