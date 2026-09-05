#!/usr/bin/env bash
# INT8 export then GPU latency bench (run after FP16 CUDA numbers recorded).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/int8_pipeline.log"
exec > >(tee -a "${LOG}") 2>&1
echo "[$(date -Is)] INT8 pipeline start"
VLA_CONDA_ENV=vla-export bash "${ROOT}/scripts/export_int8.sh"
python3 "${ROOT}/scripts/adapt_onnx_for_trt.py" "${ROOT}/exports/smolvla_libero_int8/model.onnx" 2>/dev/null || true
TETHER_TRT_EP=0 VLA_CONDA_ENV=vla-export bash "${ROOT}/scripts/bench_trt.sh" \
  "${ROOT}/exports/smolvla_libero_int8" int8_trt
bash "${ROOT}/scripts/record_latency_results.sh" "${ROOT}/results/int8_trt/latency.json" int8_trt
python3 "${ROOT}/scripts/post_int8_update.py"
echo "[$(date -Is)] INT8 pipeline done"
