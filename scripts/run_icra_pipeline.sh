#!/usr/bin/env bash
# ICRA preparation: full LIBERO re-eval for FP16 + INT8 (serialized on RTX 2060).
# Uses fixed closed-loop adapter (LeRobot lang_tokens → ONNX, not SmolLM2 retokenize).
# Optional: set RUN_TRT_EP=1 on a 12 GB+ GPU for TensorRT EP latency.
# Optional paired verify: bash scripts/run_smolvla_verify.sh fp16 libero_spatial
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/icra_pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] ICRA pipeline start"

for TAG in fp16 int8; do
  for SUITE in libero_spatial libero_object; do
    echo "[$(date -Is)] LIBERO ${TAG} ${SUITE}"
    bash "${ROOT}/scripts/run_onnx_libero_eval.sh" "${TAG}" "${SUITE}"
  done
done

if [[ "${RUN_TRT_EP:-0}" == "1" ]]; then
  echo "[$(date -Is)] TensorRT EP bench (needs >=12 GB VRAM)"
  bash "${ROOT}/scripts/run_trt_ep_bench.sh" || echo "TRT EP bench failed (see log)"
fi

python3 "${ROOT}/scripts/merge_libero_onnx_results.py"

echo "[$(date -Is)] ICRA pipeline complete (~12 h LIBERO)"
