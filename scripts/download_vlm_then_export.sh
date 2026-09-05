#!/usr/bin/env bash
# Finish SmolVLM2 weights (needed by tether export), then restart FP16 export.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/download_smolvlm2.log"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla-export}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

{
  echo "==> $(date -Is) hf download SmolVLM2-500M-Instruct model.safetensors (~1.89 GiB, resume)"
  hf download HuggingFaceTB/SmolVLM2-500M-Instruct model.safetensors
  echo "==> $(date -Is) weights ready — starting tether export"
} 2>&1 | tee -a "${LOG}"

cd "${ROOT}"
VLA_CONDA_ENV=vla-export nohup bash scripts/export_onnx.sh > results/export_onnx_launch.log 2>&1 &
echo "EXPORT_PID=$!" | tee -a "${LOG}"
sleep 8
nohup bash scripts/watch_export_then_bench.sh >> results/watch_export_then_bench.log 2>&1 &
echo "WATCHER_PID=$!" | tee -a "${LOG}"
pgrep -af 'tether export' || true
