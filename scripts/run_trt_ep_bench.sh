#!/usr/bin/env bash
# =============================================================================
# run_trt_ep_bench.sh — TensorRT Execution Provider latency benchmark
# =============================================================================
#
# Purpose:
#   Measure SmolVLA FP16 latency with ORT TensorRT EP (fused engine).
#   This is the "title claim" backend — CUDA EP numbers are the fallback.
#
# Usage:
#   RUN_TRT_EP=1 bash scripts/run_trt_ep_bench.sh          # local 2060 (may OOM)
#   # On RunPod RTX 4090 (12 GB+): same command, usually succeeds.
#
# Steps:
#   1. adapt_onnx_for_trt.py — strip ScatterND attrs TRT rejects
#   2. bench_trt.sh with TETHER_TRT_EP=1
#   3. record_latency_results.sh → baseline_summary.json
#
# RTX 2060: TETHER_TRT_WORKSPACE_MB capped at 1536 (engine build still may fail).
# Log: results/fp16_trt_trt_ep/bench.log
# Output: results/fp16_trt_trt_ep/latency.json
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/fp16_trt_trt_ep/bench.log"
mkdir -p "${ROOT}/results/fp16_trt_trt_ep"
exec > >(tee -a "${LOG}") 2>&1
echo "[$(date -Is)] TRT EP bench start (serialized; 6GB workspace cap)"

# TensorRT 10 rejects ScatterND with reduction=none attribute; strip before build.
python3 "${ROOT}/scripts/adapt_onnx_for_trt.py" "${ROOT}/exports/smolvla_libero/model.onnx" 2>/dev/null || true

# Workspace: small on 6 GB (often still OOM), large on RunPod 4090.
# Prefer scripts/run_runpod_trt.sh on cloud — it refuses GPUs under 12 GB.
if [[ -z "${TETHER_TRT_WORKSPACE_MB:-}" ]]; then
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)"
  if [[ "${VRAM_MB}" =~ ^[0-9]+$ ]] && (( VRAM_MB >= 12000 )); then
    export TETHER_TRT_WORKSPACE_MB=8192
  else
    export TETHER_TRT_WORKSPACE_MB=1536
    echo "WARN: VRAM=${VRAM_MB:-?} MiB — TRT build often fails below 12 GB. Use docs/RUNPOD_TRT.md"
  fi
fi
echo "TETHER_TRT_WORKSPACE_MB=${TETHER_TRT_WORKSPACE_MB}"
TETHER_TRT_EP=1 VLA_CONDA_ENV=vla-export bash "${ROOT}/scripts/bench_trt.sh" \
  "${ROOT}/exports/smolvla_libero" fp16_trt_trt_ep

bash "${ROOT}/scripts/record_latency_results.sh" \
  "${ROOT}/results/fp16_trt_trt_ep/latency.json" fp16_trt_trt_ep 2>/dev/null || true

echo "[$(date -Is)] TRT EP bench done"
