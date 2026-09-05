#!/usr/bin/env bash
# Point the dynamic linker at CUDA-12 libraries shipped with the conda env
# (PyTorch / nvidia-* wheels). Required because onnxruntime-gpu 1.29 looks for
# CUDA 13, and even a CUDA-12 ORT wheel must find libcublasLt.so.12 at runtime.
#
# Usage: source scripts/env_ort_cuda12.sh   (after conda activate vla-export)

_sp="${CONDA_PREFIX}/lib/python3.12/site-packages"
_paths=()
for _d in \
  "${_sp}/nvidia/cublas/lib" \
  "${_sp}/nvidia/cuda_runtime/lib" \
  "${_sp}/nvidia/cudnn/lib" \
  "${_sp}/nvidia/cufft/lib" \
  "${_sp}/nvidia/curand/lib" \
  "${_sp}/nvidia/cusolver/lib" \
  "${_sp}/nvidia/cusparse/lib" \
  "${_sp}/nvidia/nvjitlink/lib" \
  "${_sp}/nvidia/cuda_nvrtc/lib" \
  "${_sp}/tensorrt_libs" \
  "${_sp}/tensorrt_cu12_libs" \
  "${CONDA_PREFIX}/lib"
do
  if [[ -d "${_d}" ]]; then
    _paths+=("${_d}")
  fi
done
# tensorrt wheels sometimes nest libs
for _d in "${_sp}"/tensorrt*/lib "${_sp}"/nvidia/tensorrt/lib; do
  [[ -d "${_d}" ]] && _paths+=("${_d}")
done

export LD_LIBRARY_PATH="$(IFS=:; echo "${_paths[*]}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Prefer GPU CUDA EP. Set TETHER_TRT_EP=1 after adapting ONNX for TensorRT EP.
export TETHER_TRT_EP="${TETHER_TRT_EP:-0}"
echo "ORT CUDA-12 library path set (TETHER_TRT_EP=${TETHER_TRT_EP})"
