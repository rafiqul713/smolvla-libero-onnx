# Scripts

Experiment helpers for SmolVLA ONNX export, latency, LIBERO eval, paired verify, and language-width ablation.

See [docs/REPRODUCE.md](../docs/REPRODUCE.md) for the full command list.

| Script | Role |
|--------|------|
| `export_onnx.sh` / `export_int8.sh` | tether export (`--precision` flag; see paper graph audit) |
| `bench_trt.sh` | `tether inspect bench` latency → `results/*/latency.json` |
| `bench_onnx_lang_width.py` | Uniform ORT CUDA-EP latency (width ablation) |
| `libero_onnx_eval.py` | Closed-loop ONNX LIBERO (reads static lang width) |
| `run_smolvla_verify.py` | Paired PyTorch vs ONNX (+ McNemar) |
| `parity_smoke_test.py` | Open-loop action parity |
| `export_smolvla_lang_width.py` | Re-export at `--lang-len` 16/24/32 |
| `lang_truncation_stats.py` | Fed/raw token truncation stats |
| `check_lang_width_openloop.py` | Open-loop w16 vs w24 vs PyTorch |
| `run_lang_width_ablation.sh` | Full width ablation pipeline |
| `summarize_lang_ablation.py` | Collect ablation + Wilson CIs + two-proportion test |
| `merge_libero_onnx_results.py` | Merge eval into `baseline_summary.json` (+ CIs/McNemar) |
| `adapt_onnx_for_trt.py` | ScatterND cleanup for TRT EP |
