#!/usr/bin/env bash
# =============================================================================
# run_onnx_libero_eval.sh — closed-loop LIBERO for FP16 or INT8 ONNX
# =============================================================================
#
# Purpose:
#   One suite per invocation. Uses libero_onnx_eval.py (LeRobot eval loop +
#   SmolVLAOnnxPolicy). This produced the headline 41/89 and 40/89 numbers.
#
# Usage:
#   bash scripts/run_onnx_libero_eval.sh fp16 libero_spatial
#   bash scripts/run_onnx_libero_eval.sh int8 libero_object
#
# Runtime: ~3 h per suite on RTX 2060 (100 episodes = 10 tasks × 10 eps).
# Output: results/{fp16|int8}_trt/{suite}/eval_info.json
#
# Note: Uses 10 eps/task here (standard SmolVLA LIBERO protocol). Verify chain uses 30.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?Usage: $0 fp16|int8 libero_spatial|libero_object}"
SUITE="${2:?Usage: $0 fp16|int8 libero_spatial|libero_object}"

case "${TAG}" in
  fp16) EXPORT="${ROOT}/exports/smolvla_libero" ;;
  int8) EXPORT="${ROOT}/exports/smolvla_libero_int8" ;;
  *) echo "Unknown tag: ${TAG}"; exit 1 ;;
esac

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_ort_cuda12.sh"

OUT="${ROOT}/results/${TAG}_trt/${SUITE}"
mkdir -p "${OUT}"
LOG="${OUT}/eval.log"

echo "==> ONNX LIBERO eval: ${TAG} ${SUITE} -> ${OUT}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

python3 "${ROOT}/scripts/libero_onnx_eval.py" \
  --export "${EXPORT}" \
  --tag "${TAG}" \
  --suite "${SUITE}" \
  --n-episodes 10 \
  --seed 42 \
  --output "${OUT}" \
  2>&1 | tee "${LOG}"

# Refresh headline JSON
python3 "${ROOT}/scripts/merge_libero_onnx_results.py"
echo "==> Done ${TAG} ${SUITE}"
