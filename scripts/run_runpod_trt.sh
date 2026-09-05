#!/usr/bin/env bash
# =============================================================================
# run_runpod_trt.sh — TensorRT EP latency on a 12 GB+ GPU (RunPod 4090)
# =============================================================================
#
# Purpose:
#   One-shot helper for RunPod. Refuses RTX 2060-class cards so you do not
#   waste another evening on OOM engine builds.
#
# Usage (on RunPod, after cloning repo + copying exports/smolvla_libero):
#   bash scripts/run_runpod_trt.sh              # setup + TRT bench
#   bash scripts/run_runpod_trt.sh --skip-setup # env already ready
#
# Docs: docs/RUNPOD_TRT.md
# Output: results/fp16_trt_trt_ep/latency.json
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_SETUP=0
MIN_VRAM_MB=12000
WORKSPACE_MB="${TETHER_TRT_WORKSPACE_MB:-8192}"

for arg in "$@"; do
  case "${arg}" in
    --skip-setup) SKIP_SETUP=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

echo "==> RunPod / large-GPU TensorRT EP helper"
echo "    root=${ROOT}"

# --- GPU gate: never silently burn hours on 6 GB ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Run this on a CUDA GPU machine." >&2
  exit 1
fi

GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
VRAM_MB="$(echo "${GPU_LINE}" | sed -E 's/.*, *([0-9]+) *MiB.*/\1/')"
GPU_NAME="$(echo "${GPU_LINE}" | cut -d',' -f1 | xargs)"
echo "    GPU: ${GPU_NAME} (${VRAM_MB} MiB)"

if [[ -z "${VRAM_MB}" || ! "${VRAM_MB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse GPU memory from: ${GPU_LINE}" >&2
  exit 1
fi

if (( VRAM_MB < MIN_VRAM_MB )); then
  cat >&2 <<EOF
ERROR: GPU has only ${VRAM_MB} MiB VRAM (< ${MIN_VRAM_MB}).

This machine will almost certainly OOM during TensorRT engine build
(same failure as RTX 2060 6 GB). Do NOT retry here.

Use RunPod RTX 4090 (24 GB) instead — see docs/RUNPOD_TRT.md
EOF
  exit 2
fi

# Cap workspace to ~half of free-ish budget but keep a floor for TRT tactics.
if (( VRAM_MB < 16000 )); then
  WORKSPACE_MB="${TETHER_TRT_WORKSPACE_MB:-4096}"
else
  WORKSPACE_MB="${TETHER_TRT_WORKSPACE_MB:-8192}"
fi
export TETHER_TRT_WORKSPACE_MB="${WORKSPACE_MB}"
echo "    TETHER_TRT_WORKSPACE_MB=${TETHER_TRT_WORKSPACE_MB}"

# --- Export must be present (gitignored) ---
EXPORT="${ROOT}/exports/smolvla_libero"
if [[ ! -f "${EXPORT}/model.onnx" || ! -f "${EXPORT}/model.onnx.data" ]]; then
  cat >&2 <<EOF
ERROR: missing ONNX export under ${EXPORT}

exports/ is gitignored. From your home PC, copy it:

  scp -P PORT -i KEY -r exports/smolvla_libero USER@POD:~/VLAResearch/exports/

Need: model.onnx + model.onnx.data (+ tether_config.json)
EOF
  exit 3
fi
du -sh "${EXPORT}"

# --- Optional first-time install ---
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${SKIP_SETUP}" -eq 0 ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx 'vla-export'; then
    echo "==> Creating conda env vla-export (python 3.12)"
    conda create -y -n vla-export python=3.12
  fi
  conda activate vla-export
  echo "==> Running scripts/setup_trt.sh (tether + tensorrt-cu12 + ORT)"
  VLA_CONDA_ENV=vla-export bash "${ROOT}/scripts/setup_trt.sh"
else
  conda activate "${VLA_CONDA_ENV:-vla-export}"
  echo "==> --skip-setup: using env ${CONDA_DEFAULT_ENV}"
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env_ort_cuda12.sh"

echo "==> Provider sanity"
python3 - <<'PY'
import onnxruntime as ort
print("onnxruntime", ort.__version__)
print("providers", ort.get_available_providers())
assert "TensorrtExecutionProvider" in ort.get_available_providers(), (
    "TensorrtExecutionProvider missing — setup_trt.sh may have failed"
)
PY

echo "==> Starting TRT EP bench (engine build can take 5–30 min)"
export TETHER_TRT_EP=1
export VLA_CONDA_ENV="${VLA_CONDA_ENV:-vla-export}"
bash "${ROOT}/scripts/run_trt_ep_bench.sh"

LAT="${ROOT}/results/fp16_trt_trt_ep/latency.json"
if [[ ! -f "${LAT}" ]]; then
  echo "ERROR: ${LAT} missing — see results/fp16_trt_trt_ep/bench.log" >&2
  exit 4
fi

echo "==> Result:"
python3 - <<PY
import json
from pathlib import Path
p = Path("${LAT}")
d = json.loads(p.read_text())
print(json.dumps({k: d.get(k) for k in ("provider", "p50_ms", "p99_ms", "mean_ms", "peak_vram_mb")}, indent=2))
prov = str(d.get("provider") or "")
if "tensorrt" not in prov.lower() and "Tensorrt" not in prov:
    print("WARNING: provider string does not look like TensorRT — check bench.log")
else:
    print("OK: TensorRT EP looks active")
print("Copy this file home:", p)
PY

echo "==> Done. Terminate the RunPod pod after you scp latency.json home."
echo "    Home PC: docs/RUNPOD_TRT.md section E"
