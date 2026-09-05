#!/usr/bin/env bash
# Wait for the running FP16 `tether export`, skip the expensive `tether verify`
# (30 episodes/task LIBERO), then run the 1000-iter TRT latency bench.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${ROOT}/results/watch_export_then_bench.lock"
mkdir -p "${ROOT}/results"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "[$(date -Is)] another watcher holds the lock — exiting."
  exit 0
fi

echo "[$(date -Is)] waiting for tether export to finish..."
while pgrep -f 'tether export HuggingFaceVLA/smolvla_libero' >/dev/null 2>&1; do
  sleep 30
done

echo "[$(date -Is)] export process gone. Stopping tether verify if it started."
pkill -f 'tether verify' 2>/dev/null || true
sleep 2

EXPORT="${ROOT}/exports/smolvla_libero"
if [[ ! -d "${EXPORT}" ]] || ! find "${EXPORT}" -name '*.onnx' -o -name '*.engine' | grep -q .; then
  echo "[$(date -Is)] ERROR: no ONNX/engine under ${EXPORT}"
  ls -la "${EXPORT}" 2>/dev/null || true
  tail -30 "${ROOT}/results/export_onnx.log" || true
  exit 1
fi

echo "[$(date -Is)] starting FP16 TRT latency bench"
VLA_CONDA_ENV=vla-export bash "${ROOT}/scripts/bench_trt.sh" "${EXPORT}" fp16_trt
echo "[$(date -Is)] FP16 bench finished"
