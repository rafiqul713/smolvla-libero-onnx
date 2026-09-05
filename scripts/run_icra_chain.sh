#!/usr/bin/env bash
# =============================================================================
# run_icra_chain.sh — closed-loop LIBERO for FP16+INT8, then merge summary
# =============================================================================
#
# Purpose:
#   Run missing ONNX LIBERO evals (4 jobs), merge into baseline_summary.json.
#
# Usage:
#   nohup bash scripts/run_icra_chain.sh >> results/icra_chain.log 2>&1 &
#
# Jobs (skip if eval_info.json exists):
#   fp16 libero_spatial, fp16 libero_object, int8 libero_spatial, int8 libero_object
#
# Serialized on RTX 2060 — ~12 h for all four if starting from scratch.
# Log: results/icra_chain.log
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/icra_chain.log"
exec >>"${LOG}" 2>&1

echo "[$(date -Is)] ICRA chain start"

# Block until a libero_onnx_eval.py process for this tag+suite finishes.
wait_for() {
  local tag="$1" suite="$2"
  local log="${ROOT}/results/${tag}_trt/${suite}/eval.log"
  while pgrep -f "libero_onnx_eval.py.*--tag ${tag}.*--suite ${suite}" >/dev/null 2>&1; do
    echo "[$(date -Is)] waiting: ${tag} ${suite}"
    sleep 120
  done
  if [[ -f "${log}" ]]; then
    tail -3 "${log}" || true
  fi
}

for spec in "fp16 libero_spatial" "fp16 libero_object" "int8 libero_spatial" "int8 libero_object"; do
  set -- ${spec}
  tag=$1 suite=$2
  out="${ROOT}/results/${tag}_trt/${suite}/eval_info.json"
  if [[ -f "${out}" ]]; then
    echo "[$(date -Is)] skip (done): ${tag} ${suite}"
    continue
  fi
  # Avoid starting duplicate eval if a previous nohup job is still running.
  wait_for "${tag}" "${suite}"
  if [[ -f "${out}" ]]; then
    echo "[$(date -Is)] skip (done after wait): ${tag} ${suite}"
    continue
  fi
  echo "[$(date -Is)] run: ${tag} ${suite}"
  bash "${ROOT}/scripts/run_onnx_libero_eval.sh" "${tag}" "${suite}"
done

python3 "${ROOT}/scripts/merge_libero_onnx_results.py"

echo "[$(date -Is)] ICRA chain complete"
