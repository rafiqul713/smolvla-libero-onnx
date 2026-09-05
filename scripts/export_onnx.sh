#!/usr/bin/env bash
# Week 2: export SmolVLA to ONNX for TensorRT (run after FP32 baselines complete).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/exports/smolvla_libero"
LOG="${ROOT}/results/export_onnx.log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla-export}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"

export HF_HUB_DISABLE_XET=1
mkdir -p "${ROOT}/exports"

{
  echo "==> $(date -Is) Export HuggingFaceVLA/smolvla_libero -> ${OUT} (FP16, desktop)"
  tether export HuggingFaceVLA/smolvla_libero \
    --output "${OUT}" \
    --target desktop \
    --precision fp16

  echo "==> $(date -Is) Export complete (skipping full tether verify: 30 eps/task LIBERO)."
  echo "    Next: bash scripts/bench_trt.sh ${OUT} fp16_trt"
} 2>&1 | tee "${LOG}"

echo "Log: ${LOG}"
echo "Next: VLA_CONDA_ENV=vla-export bash scripts/bench_trt.sh ${OUT} fp16_trt"
