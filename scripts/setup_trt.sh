#!/usr/bin/env bash
# Week 2 prep: install reflex-vla (ONNX + ORT-TRT). Run once; no GPU eval during LIBERO baseline.
# See docs/EXPERIMENT_GUIDE.md Phase 3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/results/setup-trt.log"
mkdir -p "${ROOT}/results" "${ROOT}/exports"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla}"

export HF_HUB_DISABLE_XET=1

{
  # The [gpu] extra pulls the generic `tensorrt` meta-package, which resolves
  # CUDA-13 wheels (tensorrt_cu13) — unusable on driver 535 (CUDA 12 only).
  # So: install [serve] extra, then the GPU stack explicitly with cu12 wheels.
  echo "==> $(date -Is) Installing fastcrest-tether (package was renamed from reflex-vla)"
  pip install --retries 10 --timeout 60 'fastcrest-tether[serve] @ git+https://github.com/FastCrest/reflex-vla'

  echo "==> $(date -Is) Installing CUDA-12 GPU stack (tensorrt-cu12, onnxruntime-gpu, cudnn9)"
  pip install --retries 10 --timeout 60 \
    'tensorrt-cu12>=10.0,<11' \
    'onnxruntime-gpu>=1.25.1' \
    'nvidia-cudnn-cu12>=9.5,<10'

  echo "==> $(date -Is) Sanity: torch CUDA still works + tensorrt imports"
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_ok:", torch.cuda.is_available())
import tensorrt
print("tensorrt", tensorrt.__version__)
import onnxruntime
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
PY

  echo "==> $(date -Is) Checking CLI"
  reflex --help | head -5 || tether --help | head -5 || true

  echo "==> $(date -Is) Done"
} 2>&1 | tee "${LOG}"

echo "Install log: ${LOG}"
echo "Next (after Week 1 baselines): bash scripts/export_onnx.sh"
