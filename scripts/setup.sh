#!/usr/bin/env bash
# One-time environment setup for SmolVLA + LIBERO on RTX 2060.
# See docs/EXPERIMENT_GUIDE.md Phase 0–1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VLA_CONDA_ENV:-vla}"

echo "==> Checking NVIDIA GPU"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed. Install/update the NVIDIA driver first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Creating conda env '${ENV_NAME}' (Python 3.12) if missing"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.12 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Upgrading pip"
python -m pip install -U pip wheel setuptools
# Slow links / large wheels (torch, robosuite) need a long timeout
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

echo "==> Installing LeRobot + LIBERO + SmolVLA extras (large downloads; may take 30–90+ min)"
pip install --retries 10 'lerobot[libero,smolvla,evaluation]'
pip install --retries 10 'mujoco==3.3.2'
pip install --retries 10 huggingface_hub

# Default PyPI torch wheels may target CUDA 13 and fail on driver 535 (CUDA 12.2).
# Pin a CUDA 12.1 build that works with RTX 2060 + driver 535.x.
echo "==> Replacing torch with CUDA 12.1 wheels (driver 535 / CUDA 12.2 compatible)"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install --retries 10 torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# Optional (dataset video); ignore failure
pip install --retries 10 'torchcodec<0.12.0' || true

echo "==> Recording pinned versions"
mkdir -p "${ROOT}/results"
pip freeze > "${ROOT}/results/requirements-frozen.txt"
conda list --explicit > "${ROOT}/results/conda-vla-explicit.txt" 2>/dev/null || true

echo "==> Quick import check"
python - <<'PY'
import mujoco
import torch
print("mujoco:", mujoco.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Setup complete. Before eval, run:"
echo "  conda activate ${ENV_NAME}"
echo "  source ${ROOT}/scripts/env.sh"
echo ""
echo "Smoke test (2 min):"
echo "  lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero \\"
echo "    --env.type=libero --env.task=libero_spatial --env.task_ids=[0] \\"
echo "    --eval.n_episodes=2 --eval.batch_size=1 --policy.n_action_steps=1 \\"
echo "    --policy.num_steps=10 --policy.device=cuda --policy.use_amp=true \\"
echo "    --output_dir=results/fp32_smoke --seed=42"
