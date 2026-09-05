#!/usr/bin/env bash
# =============================================================================
# env.sh — shared environment for LIBERO / MuJoCo eval on Linux
# =============================================================================
#
# Usage (always source, never execute):
#   source scripts/env.sh
#
# Why each variable:
#   MUJOCO_GL=egl          Headless GPU rendering (no display needed)
#   PYOPENGL_PLATFORM=egl  Match MuJoCo GL backend
#   TOKENIZERS_PARALLELISM  Avoid HF tokenizer fork warnings
#   OMP/MKL/OPENBLAS=1     LIBERO is single-env; extra threads add CPU heat
#   HF_HUB_DISABLE_XET     xet downloads often hang on home networks
# =============================================================================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

# MuJoCo + robosuite run one env; pinning threads reduces CPU package temp.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "VLA eval environment loaded (MUJOCO_GL=egl, mujoco must be 3.3.2)"
