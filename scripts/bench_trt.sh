#!/usr/bin/env bash
# Latency bench for a tether export (FP16 or INT8).
#
# WHAT THIS MEASURES (Table II / tab:summary ONNX p99 601 ms and 532 ms):
#   Wraps `tether inspect bench` (CUDA EP by default): 50 warmup + 1000 timed
#   calls, then parses tether's markdown report into results/{tag}/latency.json.
#   This is the harness behind the paper's main-deployment ONNX latency numbers.
#
# HOW THE OTHER ONNX BENCH DIFFERS (scripts/bench_onnx_lang_width.py):
#   That script drives onnxruntime Session.run directly with synthetic inputs
#   shaped from the graph (used for Table V / Fig. 2 width 16/24/32 comparison).
#   Same nominal 50/1000 protocol and CUDA EP, but a different driver, input
#   construction, and timing path — so width-16 p99 can differ (601 ms here vs
#   583.6 ms there) without any model change. Do not mix the two p99 columns.
#
# Uses `tether inspect bench` — same warmup/N protocol as benchmark_latency.py.
# Parses tether's text report into results/{tag}/latency.json for the paper.
#
# Usage:
#   bash scripts/bench_trt.sh exports/smolvla_libero fp16_trt
#   TETHER_TRT_EP=1 bash scripts/bench_trt.sh exports/smolvla_libero fp16_trt_trt_ep
#
# Env vars:
#   WARMUP=50   discarded iterations (GPU cache warm-up)
#   ITERS=1000  timed runs for p50/p99
#   VLA_CONDA_ENV  default vla-export (tether + ORT)
#   TETHER_TRT_EP  set by env_ort_cuda12.sh / caller (0=CUDA EP, 1=TensorRT EP)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT="${1:-${ROOT}/exports/smolvla_libero}"
TAG="${2:-fp16_trt}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-1000}"
OUT_DIR="${ROOT}/results/${TAG}"
LOG="${OUT_DIR}/bench.log"
JSON="${OUT_DIR}/latency.json"

mkdir -p "${OUT_DIR}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla-export}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_ort_cuda12.sh"

if [[ ! -d "${EXPORT}" ]]; then
  echo "Export dir missing: ${EXPORT}" >&2
  exit 1
fi

{
  echo "==> $(date -Is) tether inspect bench ${EXPORT}"
  echo "    warmup=${WARMUP} iterations=${ITERS} tag=${TAG}"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
  tether inspect bench "${EXPORT}" \
    --device cuda \
    --warmup "${WARMUP}" \
    --iterations "${ITERS}" \
    --report "${OUT_DIR}/bench_report.md"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
  echo "==> $(date -Is) bench done"
} 2>&1 | tee "${LOG}"

# Inline Python: tether prints markdown; we extract p50/p99 and provider name.
python3 - "${LOG}" "${JSON}" "${TAG}" "${EXPORT}" <<'PY'
import json, re, sys, time
from pathlib import Path
log, out, tag, export = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
text = log.read_text(errors="replace")
nums = {}
for key, pat in [
    ("mean_ms", r"(?i)mean[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    ("p50_ms", r"(?i)p50[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    ("p95_ms", r"(?i)p95[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    ("p99_ms", r"(?i)p99[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*ms"),
]:
    m = re.search(pat, text)
    if m:
        nums[key] = float(m.group(1))
vram = None
for line in text.splitlines():
    if "MiB" in line and "," in line:
        parts = [p.strip().replace(" MiB", "").replace("MiB", "") for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            vram = int(parts[1])
prov = None
for line in text.splitlines():
    if "provider_mode" in line or "Active EP:" in line:
        prov = line.strip()
    if "active_providers=" in line.replace(" ", ""):
        m = re.search(r"active_providers=\[([^\]]+)\]", line.replace("\n", " "))
        if m:
            prov = m.group(1).strip("'\"")
            break
if not prov:
    m = re.search(r"Active EP:\s*(.+)", text)
    if m:
        prov = m.group(1).strip()
result = {
    "gpu": "NVIDIA GeForce RTX 2060",
    "precision": tag,
    "export_dir": export,
    "provider": prov,
    "runs": 1000,
    "warmup": 50,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    **nums,
    "peak_vram_mb": vram,
    "log": str(log),
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "Saved: ${JSON}"
echo "Next: if FP16 looks sane, bash scripts/export_int8.sh"
