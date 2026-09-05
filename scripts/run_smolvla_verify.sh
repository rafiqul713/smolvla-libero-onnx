#!/usr/bin/env bash
# =============================================================================
# run_smolvla_verify.sh — wrapper for paired PyTorch vs ONNX LIBERO verify
# =============================================================================
#
# Purpose:
#   Shell wrapper: sets conda + env vars, then calls run_smolvla_verify.py.
#   Fills the gap because upstream `tether verify` only supports Pi05/Triton.
#
# Usage:
#   bash scripts/run_smolvla_verify.sh fp16 libero_spatial 30
#   bash scripts/run_smolvla_verify.sh int8 libero_object 10   # quick test
#
# Args:
#   $1  fp16 | int8     → picks exports/smolvla_libero or exports/smolvla_libero_int8
#   $2  libero_spatial | libero_object
#   $3  episodes per task (default 30 for ICRA; main baseline uses 10)
#
# Conda: vla (LIBERO + LeRobot). ORT CUDA libs via env_ort_cuda12.sh.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?Usage: $0 fp16|int8 libero_spatial|libero_object [num_episodes]}"
SUITE="${2:?Usage: $0 fp16|int8 libero_spatial|libero_object [num_episodes]}"
EPS="${3:-30}"

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

OUT="${ROOT}/results/verify/${TAG}_${SUITE}_${EPS}eps.json"
mkdir -p "$(dirname "${OUT}")"

echo "==> SmolVLA verify: ${TAG} ${SUITE} (${EPS} eps/task)"
export PYTHONUNBUFFERED=1
python3 "${ROOT}/scripts/run_smolvla_verify.py" \
  --export "${EXPORT}" \
  --suite "${SUITE}" \
  --num-episodes "${EPS}" \
  --seed 42 \
  --output "${OUT}"
