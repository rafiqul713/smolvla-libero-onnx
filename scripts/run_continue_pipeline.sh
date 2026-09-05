#!/usr/bin/env bash
# Serialize remaining steps: parity smoke -> TRT EP bench (one GPU job at a time).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/continue_pipeline.log"
exec > >(tee -a "${LOG}") 2>&1
echo "[$(date -Is)] continue pipeline start (serialized)"

bash "${ROOT}/scripts/run_parity_smoke.sh"
echo "[$(date -Is)] parity done; starting TRT EP bench"
bash "${ROOT}/scripts/run_trt_ep_bench.sh"
echo "[$(date -Is)] TRT EP done"
echo "[$(date -Is)] continue pipeline complete"
