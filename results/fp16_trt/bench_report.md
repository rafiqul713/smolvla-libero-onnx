# Tether Bench Report

_Generated 2026-08-27T19:47:32Z • tether 0.12.0 @ _

## Per-chunk latency (post-warmup)

- **n** = 1000 samples (warmup_discarded = 0)
- **min** = 490.11 ms
- **mean** = 528.60 ms (95% CI [527.32, 529.88])
- **p50** = 526.86 ms
- **p95** = 559.43 ms
- **p99** = 597.96 ms
- **p99.9** = 644.81 ms
- **max** = 656.51 ms
- **std** = 20.70 ms
- **jitter** (std/mean) = 0.0392
- **hz** (1/mean) = 1.9 Hz

## Reproducibility envelope

- **timestamp**: `2026-08-27T19:47:32Z`
- **tether**: `0.12.0`
- **git**: ``
- **python**: `3.12.13`
- **platform**: `Linux-6.8.0-138-generic-x86_64`
- **gpu**: `NVIDIA GeForce RTX 2060`
- **cuda**: `535.309.01`
- **onnxruntime**: `1.22.0`
- **device**: `cuda`
- **inference_mode**: `smolvla_onnx_monolithic`
- **provider_mode**: `onnx_gpu`
- **active_providers**: `CUDAExecutionProvider, CPUExecutionProvider`
- **seed**: `0`
- **export_dir**: `exports/smolvla_libero`

### ONNX files

| file | sha256 (16) | bytes |
|---|---|---:|
| `model.onnx` | `a1ef4656922c8a8c` | 29728053 |
| `model.onnx.data` | `97cd598ea5dd1001` | 2299663689 |

## Notes

- warmup=50 discarded BEFORE the recorded latencies (see iterations loop in cli.benchmark_cmd)

## What this measures

Per-chunk wall-clock latency of `server.predict()` — the full denoising loop for flow-matching VLAs (10 Euler steps for pi0 / pi0.5 / SmolVLA, 4 DDIM steps for GR00T) including VLM-prefix + expert-denoise + postprocess. Warmup samples discarded so TRT engine build / ORT graph optimization does not contaminate the steady-state distribution. Methodology lifted from ISB-1 (sibling project EasyInference); see `reference/NOTES.md` for the source pattern.
