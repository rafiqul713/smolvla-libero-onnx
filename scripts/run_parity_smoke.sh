#!/usr/bin/env bash
# ONNX parity: INT8 vs FP16 reference (200 shared inputs). FP16 vs PyTorch assumed at export.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/parity/pipeline.log"
mkdir -p "${ROOT}/results/parity"
exec > >(tee -a "${LOG}") 2>&1
echo "[$(date -Is)] parity smoke start (ONNX cross-check)"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla-export
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_ort_cuda12.sh"

# FP16 self-check (reference cache for INT8)
python3 "${ROOT}/scripts/parity_smoke_test.py" \
  --reference "${ROOT}/exports/smolvla_libero" \
  --candidate "${ROOT}/exports/smolvla_libero" \
  --tag fp16 --samples 200

python3 "${ROOT}/scripts/parity_smoke_test.py" \
  --reference "${ROOT}/exports/smolvla_libero" \
  --candidate "${ROOT}/exports/smolvla_libero_int8" \
  --tag int8 --samples 200

python3 "${ROOT}/scripts/merge_parity_results.py"
echo "[$(date -Is)] parity smoke done"
