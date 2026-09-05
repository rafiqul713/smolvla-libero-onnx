#!/usr/bin/env bash
# =============================================================================
# record_latency_results.sh — copy bench numbers into baseline_summary.json
# =============================================================================
#
# Usage:
#   bash scripts/record_latency_results.sh results/fp16_trt/latency.json fp16_trt
#   bash scripts/record_latency_results.sh results/fp16_trt_trt_ep/latency.json fp16_trt_trt_ep
#
# Args:
#   $1  path to latency.json from bench_trt.sh
#   $2  key under our_results in baseline_summary.json (fp16_trt | int8_trt | fp16_trt_trt_ep)
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAT="${1:?latency.json path}"
TAG="${2:?summary key: fp16_trt|int8_trt|fp16_trt_trt_ep}"
SUMMARY="${ROOT}/results/baseline_summary.json"

python3 - <<PY
import json
from datetime import date
from pathlib import Path

lat = json.loads(Path("${LAT}").read_text())
summary = json.loads(Path("${SUMMARY}").read_text())
key = "${TAG}"
row = summary["our_results"][key]
row["p50_ms"] = lat.get("p50_ms")
row["p95_ms"] = lat.get("p95_ms")
row["p99_ms"] = lat.get("p99_ms")
row["mean_ms"] = lat.get("mean_ms")
row["latency_runs"] = lat.get("runs", 1000)
row["provider"] = lat.get("provider", "see bench_report.md")
row["peak_vram_mb"] = lat.get("peak_vram_mb")
row["latency_json"] = "${LAT}"
summary["last_updated"] = str(date.today())
Path("${SUMMARY}").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Updated our_results.{key}: p50={row.get('p50_ms')} p99={row.get('p99_ms')}")
PY
