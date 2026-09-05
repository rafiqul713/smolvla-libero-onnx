#!/usr/bin/env bash
# =============================================================================
# run_verify_chain.sh — serialized FP16 paired verify (Spatial then Object)
# =============================================================================
#
# Purpose:
#   Run native PyTorch vs ONNX on the SAME tether rollout loop (ICRA evidence).
#   One suite at a time — RTX 2060 cannot run two heavy LIBERO jobs in parallel.
#
# Usage:
#   nohup bash scripts/run_verify_chain.sh >> results/verify/verify_chain.log 2>&1 &
#
# Output (skip if file already exists — safe to restart):
#   results/verify/fp16_libero_spatial_30eps.json
#   results/verify/fp16_libero_object_30eps.json
#
# Runtime (RTX 2060): ~20–40 h total (30 eps/task × 10 tasks × 2 arms × 2 suites)
#
# Resume: re-run the same command; completed suites are skipped automatically.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/verify/verify_chain.log"
mkdir -p "${ROOT}/results/verify"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
exec >>"${LOG}" 2>&1

echo "[$(date -Is)] verify chain start"

run_one() {
  local tag="$1" suite="$2" eps="${3:-30}"
  local out="${ROOT}/results/verify/${tag}_${suite}_${eps}eps.json"
  # Each suite writes one JSON only when BOTH PyTorch and ONNX arms finish.
  if [[ -f "${out}" ]]; then
    echo "[$(date -Is)] skip (done): ${tag} ${suite}"
    return 0
  fi
  echo "[$(date -Is)] run: ${tag} ${suite} (${eps} eps/task)"
  bash "${ROOT}/scripts/run_smolvla_verify.sh" "${tag}" "${suite}" "${eps}"
}

for spec in "fp16 libero_spatial" "fp16 libero_object"; do
  set -- ${spec}
  run_one "$1" "$2" 30
done

echo "[$(date -Is)] verify chain complete"
