#!/usr/bin/env python3
"""Uniform ONNX Runtime CUDA-EP latency bench for the language-width ablation.

WHAT THIS MEASURES (Table V / tab:ctx and Fig. 2 ONNX p99 583.6 / 587.7 / 591.3 ms):
    Direct onnxruntime InferenceSession.run timing (CUDA EP): 50 warmup + 1000
    timed calls, batch 1, 10 flow steps baked into the graph. All three static
    language widths (16/24/32) use this same script so width comparison is fair.

HOW THE OTHER ONNX BENCH DIFFERS (scripts/bench_trt.sh → `tether inspect bench`):
    That harness is used for Table II requested-FP16/INT8 p99 (601 ms / 532 ms).
    Same nominal 50/1000 + CUDA EP, but tether's inspect driver and report path
    differ from this Session.run loop — so width-16 p99 can be 601 ms there vs
    583.6 ms here with the same export. Do not treat the two as interchangeable.

Inputs are built from the session's own static shapes; language mask is all-True
(worst case: every language position attended).

Peak memory: process-level `nvidia-smi --query-compute-apps` for this PID
(same accounting style as the paper: indicative, includes runtime context).

Usage (conda env: vla-export, after `source scripts/env_ort_cuda12.sh`):
    python scripts/bench_onnx_lang_width.py --export exports/smolvla_libero_lang24 \
        --out results/lang_ablation/w24/latency.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _nvidia_smi_used_mb(pid: int) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.strip().splitlines():
            p, mem = [x.strip() for x in line.split(",")[:2]]
            if int(p) == pid:
                return int(mem)
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    onnx_path = args.export.resolve() / "model.onnx"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = sess.get_providers()
    if providers[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CUDA EP not active: {providers}")

    rng = np.random.default_rng(args.seed)
    feeds = {}
    lang_len = None
    for inp in sess.get_inputs():
        shape = [int(d) if isinstance(d, int) else 1 for d in inp.shape]
        if inp.name.startswith("img_"):
            feeds[inp.name] = rng.uniform(-1, 1, size=shape).astype(np.float32)
        elif inp.name.startswith("mask_"):
            feeds[inp.name] = np.ones(shape, dtype=bool)
        elif inp.name == "lang_tokens":
            lang_len = shape[1]
            feeds[inp.name] = rng.integers(0, 49152, size=shape, dtype=np.int64)
        elif inp.name == "lang_masks":
            feeds[inp.name] = np.ones(shape, dtype=bool)
        elif inp.name == "state":
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
        elif inp.name == "noise":
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
        else:
            raise SystemExit(f"Unexpected input {inp.name}")

    for _ in range(args.warmup):
        sess.run(None, feeds)

    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)  # synchronous: ORT copies output to host before returning
        times.append((time.perf_counter() - t0) * 1000.0)
    used_mb = _nvidia_smi_used_mb(os.getpid())

    arr = np.asarray(times)
    res = {
        "export": str(args.export),
        "lang_len": lang_len,
        "provider": providers[0],
        "onnxruntime": ort.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "min_ms": round(float(arr.min()), 2),
        "nvidia_smi_used_mb": used_mb,
        "onnx_size_mb": round(sum(f.stat().st_size for f in args.export.resolve().glob("model.onnx*")
                                  if not f.name.endswith(".pre_trt_adapt")) / 1e6),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
