# Reproduce (RTX 2060–class GPU)

Pinned stack used in these experiments:

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX 2060 6 GB (driver 535) |
| Python | 3.12 |
| PyTorch | 2.5.1+cu121 |
| LeRobot | 0.6.1 (eval), 0.5.1 pin (export env) |
| MuJoCo | 3.3.2 |
| tether | 0.12.0 (`fastcrest-tether`) |
| ONNX Runtime GPU | 1.22.0 (CUDA 12) |
| Checkpoint | `HuggingFaceVLA/smolvla_libero` |

## Environment

```bash
# Create eval env (example)
conda create -n vla python=3.12 -y
conda activate vla
# Install LeRobot 0.6.1, MuJoCo 3.3.2, LIBERO deps per LeRobot docs

# Export / ORT tooling env
conda create -n vla-export python=3.12 -y
conda activate vla-export
bash scripts/setup_trt.sh
```

```bash
source scripts/env.sh
source scripts/env_ort_cuda12.sh   # after activating vla-export / ORT
```

## Main commands

```bash
# FP32 LIBERO baseline (Spatial / Object)
bash scripts/run_fp32_baseline.sh

# ONNX export (tether --precision is a flag; monolithic path yields FP32 graph)
bash scripts/export_onnx.sh
bash scripts/export_int8.sh

# Latency — tether inspect harness (Table II style)
bash scripts/bench_trt.sh exports/smolvla_libero fp16_trt
bash scripts/bench_trt.sh exports/smolvla_libero_int8 int8_trt

# Action parity (requested-FP16 vs requested-INT8 ONNX)
bash scripts/run_parity_smoke.sh

# Closed-loop ONNX LIBERO (width 16)
bash scripts/run_onnx_libero_eval.sh fp16 libero_spatial
bash scripts/run_onnx_libero_eval.sh fp16 libero_object
bash scripts/run_onnx_libero_eval.sh int8 libero_spatial
bash scripts/run_onnx_libero_eval.sh int8 libero_object

# Paired verify (long; 30 eps/task)
bash scripts/run_smolvla_verify.sh fp16 libero_spatial 30
bash scripts/run_smolvla_verify.sh fp16 libero_object 30

# Language-width ablation (export 24/32, uniform ORT bench, closed-loop, summary)
bash scripts/run_lang_width_ablation.sh
# Or step-by-step:
#   python scripts/export_smolvla_lang_width.py --lang-len 24 --output exports/smolvla_libero_lang24
#   python scripts/bench_onnx_lang_width.py --export exports/smolvla_libero_lang24 --out results/lang_ablation/w24/latency.json
#   python scripts/lang_truncation_stats.py --out results/lang_ablation/truncation_stats.json
#   python scripts/summarize_lang_ablation.py
```

TensorRT EP needs **≥12 GB** VRAM for engine build on this graph; on 6 GB it typically OOMs. Optional helper: `scripts/run_runpod_trt.sh` (refuses GPUs under 12 GB).

## Results

| File | Contents |
|------|----------|
| `results/baseline_summary.json` | Main table + paired verify + Wilson/McNemar fields |
| `results/lang_ablation/summary.json` | Width 16/24/32 success, latency, Wilson CIs, two-proportion test |
| `results/lang_ablation/truncation_stats.json` | Fed/raw token lengths |

**Latency harnesses are not interchangeable:** `bench_trt.sh` (`tether inspect`) vs `bench_onnx_lang_width.py` (uniform ORT).
