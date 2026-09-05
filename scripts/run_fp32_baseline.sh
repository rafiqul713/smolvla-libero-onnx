#!/usr/bin/env bash
# Week 1 FP32 PyTorch baseline on RTX 2060 (see docs/EXPERIMENT_GUIDE.md Phase 2).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:?Usage: $0 libero_spatial|libero_object}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-vla}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"

# Avoid slow HuggingFace xet downloads; SmolVLM2 base weights are in smolvla_libero checkpoint.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

OUT="${ROOT}/results/fp32_pytorch/${SUITE}"
mkdir -p "${OUT}"

echo "==> FP32 baseline: ${SUITE} -> ${OUT}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.load_vlm_weights=false \
  --env.type=libero \
  --env.task="${SUITE}" \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.n_action_steps=1 \
  --policy.num_steps=10 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --output_dir="${OUT}" \
  --seed=42 \
  2>&1 | tee "${OUT}/eval.log"

echo "==> Done. Check ${OUT}/eval.log for pc_success."
