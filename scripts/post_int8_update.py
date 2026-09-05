#!/usr/bin/env python3
"""After INT8 bench: update summary silent_quant_failure flag."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "baseline_summary.json"
INT8 = ROOT / "results" / "int8_trt" / "latency.json"
FP16_P99 = 601.43


def main() -> None:
    if not INT8.exists():
        print("No INT8 latency yet")
        return
    s = json.loads(SUMMARY.read_text())
    lat = json.loads(INT8.read_text())
    int8 = s["our_results"]["int8_trt"]
    p99 = lat.get("p99_ms")
    int8["p50_ms"] = lat.get("p50_ms")
    int8["p95_ms"] = lat.get("p95_ms")
    int8["p99_ms"] = p99
    int8["mean_ms"] = lat.get("mean_ms")
    int8["peak_vram_mb"] = lat.get("peak_vram_mb")
    int8["provider"] = lat.get("provider")
    if p99 is not None and FP16_P99:
        ratio = p99 / FP16_P99
        int8["silent_quant_failure"] = ratio > 0.95  # within 5% = no real speedup
        int8["speedup_vs_fp16_p99"] = round(FP16_P99 / p99, 2) if p99 else None
        print(f"INT8 p99={p99} vs FP16 {FP16_P99}: silent_failure={int8['silent_quant_failure']}")
    SUMMARY.write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
