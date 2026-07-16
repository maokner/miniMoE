#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "prepare_gpu.sh requires Linux." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Attach an NVIDIA GPU and install its driver." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found." >&2
  exit 1
fi

nvidia-smi

# PEP 668 (Ubuntu 24.04+, Debian 12+) forbids pip installs into the system
# Python, so everything runs in a repo-local venv. --system-site-packages keeps
# a preinstalled CUDA PyTorch (NGC, Lambda, and similar images) visible instead
# of re-downloading a multi-gigabyte wheel.
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Install a CUDA-enabled PyTorch build.")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs {torch.cuda.device_count()}")
PY

mkdir -p checkpoints

fetch_checkpoint() {
  local url="$1" file="$2" sha="$3"
  if [[ -f "$file" ]] && echo "$sha  $file" | sha256sum --check --status; then
    echo "$file already downloaded and verified."
    return
  fi
  rm -f "$file"
  # Download to a side file so an interrupted transfer can resume across
  # reruns; verify before moving into place so a corrupt download never
  # masquerades as a ready checkpoint.
  curl -fL --retry 5 --continue-at - -o "$file.download" "$url"
  if ! echo "$sha  $file.download" | sha256sum --check --strict; then
    rm -f "$file.download"
    echo "SHA-256 verification failed for $file; the corrupt download was removed. Rerun to retry." >&2
    exit 1
  fi
  mv "$file.download" "$file"
}

fetch_checkpoint \
  "https://huggingface.co/mokner123/miniMoE/resolve/main/minimoe_step_0019073.pt" \
  checkpoints/minimoe_step_0019073.pt \
  67040b0bce7edc1ec49116fbdac4e819dc092b83bc503a6c4722603e3e313532
fetch_checkpoint \
  "https://huggingface.co/mokner123/miniMoE/resolve/main/minimoe_sft.pt" \
  checkpoints/minimoe_sft.pt \
  f8f9e91d05f00fe1d1579dc2d5ce0f2f728d0183f9424733fd93542e979d9529

echo "GPU environment and both checkpoints are ready."
