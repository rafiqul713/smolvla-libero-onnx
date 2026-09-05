#!/usr/bin/env bash
# Copy pc_success from eval_info.json into baseline_summary.json.
# Usage: bash scripts/record_baseline_results.sh libero_spatial|libero_object
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:?Usage: $0 libero_spatial|libero_object}"
INFO="${ROOT}/results/fp32_pytorch/${SUITE}/eval_info.json"
SUMMARY="${ROOT}/results/baseline_summary.json"

if [[ ! -f "${INFO}" ]]; then
  echo "ERROR: ${INFO} not found. Wait for eval to finish." >&2
  exit 1
fi

python3 - <<PY
import json
from datetime import date
from pathlib import Path

info_path = Path("${INFO}")
summary_path = Path("${SUMMARY}")
suite = "${SUITE}"
key = "libero_spatial_pct" if suite == "libero_spatial" else "libero_object_pct"

info = json.loads(info_path.read_text())
pc = info.get("overall", {}).get("pc_success")
if pc is None:
    pc = info.get("per_group", {}).get(suite, {}).get("pc_success")
if pc is None:
    raise SystemExit(f"Could not find pc_success in {info_path}")

summary = json.loads(summary_path.read_text())
summary["our_results"]["fp32_pytorch"][key] = pc
spatial = summary["our_results"]["fp32_pytorch"].get("libero_spatial_pct")
obj = summary["our_results"]["fp32_pytorch"].get("libero_object_pct")
if spatial is not None and obj is not None:
    summary["our_results"]["fp32_pytorch"]["status"] = "complete"
else:
    summary["our_results"]["fp32_pytorch"]["status"] = "partial"
summary["last_updated"] = str(date.today())
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Recorded {key} = {pc}%")
PY

echo "Updated ${SUMMARY}"
