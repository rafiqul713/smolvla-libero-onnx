#!/usr/bin/env bash
# Week 3: INT8 PTQ export (run AFTER FP16 export + latency bench).
# tether 0.12 has no separate --calibration-data flag; INT8 is --precision int8.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/exports/smolvla_libero_int8"
LOG="${ROOT}/results/export_int8.log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla-export}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_ort_cuda12.sh"

export HF_HUB_DISABLE_XET=1
mkdir -p "${ROOT}/exports"

{
  echo "==> $(date -Is) INT8 export HuggingFaceVLA/smolvla_libero -> ${OUT}"
  tether export HuggingFaceVLA/smolvla_libero \
    --output "${OUT}" \
    --target desktop \
    --precision int8 \
    --num-steps 10
  echo "==> $(date -Is) INT8 export complete"
} 2>&1 | tee "${LOG}"

echo "Log: ${LOG}"
echo "Next: VLA_CONDA_ENV=vla-export bash scripts/bench_trt.sh ${OUT} int8_trt"
