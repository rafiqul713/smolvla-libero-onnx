# SmolVLA ONNX Deployment on LIBERO (RTX 2060)

Code and measured results accompanying the letter:

> *When Faster VLA Deployment Changes Closed-Loop Behavior: Task Success–Latency Analysis of SmolVLA Across PyTorch and ONNX Variants*

**Hardware:** personal NVIDIA RTX 2060 6 GB only (no Jetson, no physical robot).

## Highlights (main eval, 100 episodes/suite, seed 42)

| Config | Spatial | Object | p99 |
|--------|--------:|-------:|----:|
| PyTorch+AMP | 70.0% | 88.0% | 1181 ms |
| Requested-FP16 ONNX (width 16) | 41.0% | 89.0% | 601 ms† |
| Requested-INT8 ONNX (width 16) | 40.0% | 89.0% | 532 ms† |
| ONNX width 24 | 75.0% | 90.0% | 587.7 ms‡ |
| ONNX width 32 | 71.0% | 87.0% | 591.3 ms‡ |

† `tether inspect bench` (CUDA EP). ‡ Uniform ORT bench (`scripts/bench_onnx_lang_width.py`) — do not mix harnesses.

**Graph audit:** requested-FP16 and requested-INT8 exports are **byte-identical FP32** graphs (no quantization ops). The 601 vs 532 ms gap is **not** INT8 quantization.

**Language width:** default static width 16 truncates 5/10 Spatial instructions; width 24 recovers Spatial (75.0%).

Paired verify (300 eps/suite): Spatial 56.7% → 33.0%; Object 73.3% → 74.0%.

## Layout

| Path | Contents |
|------|----------|
| `scripts/` | Export, latency, LIBERO eval, parity, verify, **language-width ablation** |
| `results/` | `baseline_summary.json`, eval/latency/parity/verify, `lang_ablation/` |
| `docs/REPRODUCE.md` | How to reproduce |
| `INCLUDED.md` | What is shipped |

ONNX exports (`exports/`, ~2+ GB) are **not** included. Re-export with tether.

## Quick start

```bash
source scripts/env.sh
# See docs/REPRODUCE.md for conda / tether / MuJoCo pins
bash scripts/run_parity_smoke.sh
```

Language-width ablation (long):

```bash
bash scripts/run_lang_width_ablation.sh
```

Headline numbers: `results/baseline_summary.json`  
Width ablation: `results/lang_ablation/summary.json`

## License

MIT — see `LICENSE`.
