#!/usr/bin/env bash
# Replace CUDA-13 ONNX Runtime (1.29) with a CUDA-12 wheel so the RTX 2060
# (driver 535) can use CUDAExecutionProvider.
set -euo pipefail
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla-export}"

echo "==> $(date -Is) uninstall ORT 1.29 (CUDA 13)"
pip uninstall -y onnxruntime onnxruntime-gpu || true

echo "==> $(date -Is) install onnxruntime-gpu 1.22.0 (CUDA 12 / cuDNN 9)"
pip install --retries 10 --timeout 120 "onnxruntime-gpu==1.22.0"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/env_ort_cuda12.sh"

python - <<'PY'
import os
print("LD_LIBRARY_PATH starts:", os.environ.get("LD_LIBRARY_PATH","")[:180])
import onnxruntime as ort
print("onnxruntime", ort.__version__)
print("providers", ort.get_available_providers())
so = "/".join(ort.__file__.split("/")[:-1] + ["capi", "libonnxruntime_providers_cuda.so"])
import subprocess, pathlib
p = pathlib.Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
if p.exists():
    out = subprocess.check_output(["ldd", str(p)], text=True, stderr=subprocess.STDOUT)
    missing = [ln.strip() for ln in out.splitlines() if "not found" in ln]
    print("missing CUDA libs:", missing or "none")
PY
echo "Done. Next: bash scripts/bench_trt.sh exports/smolvla_libero fp16_trt"
