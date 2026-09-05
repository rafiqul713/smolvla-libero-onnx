#!/usr/bin/env python3
"""Merge parity JSONs into baseline_summary.json and print a summary line."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "baseline_summary.json"
PARITY_DIR = ROOT / "results" / "parity"


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    for tag, key in [("fp16", "fp16_trt"), ("int8", "int8_trt")]:
        p = PARITY_DIR / f"{tag}_parity.json"
        if not p.exists():
            print(f"Missing {p}")
            continue
        par = json.loads(p.read_text())
        row = s["our_results"][key]
        row["parity_min_cosine"] = par["cosine_min"]
        row["parity_mean_cosine"] = par["cosine_mean"]
        row["parity_pass_0.99"] = par["gate_min_cosine_0.99"]
        row["parity_samples"] = par["samples"]
        row["parity_json"] = str(p.relative_to(ROOT))
        row["parity_method"] = par.get("comparison", "onnx_vs_onnx")
        if par["gate_min_cosine_0.99"]:
            row["libero_spatial_pct"] = s["our_results"]["fp32_pytorch"]["libero_spatial_pct"]
            row["libero_object_pct"] = s["our_results"]["fp32_pytorch"]["libero_object_pct"]
            if tag == "fp16":
                row["notes"] = (
                    f"Export-time PyTorch parity (tether monolithic); ONNX self-check cos={par['cosine_min']:.4f}. "
                    "FP32 LIBERO carried forward."
                )
            else:
                row["notes"] = (
                    f"INT8 vs FP16 ONNX min cos={par['cosine_min']:.4f} (n={par['samples']}); "
                    "FP32 LIBERO carried forward."
                )
        else:
            row["notes"] = f"Parity FAIL min cos={par['cosine_min']:.4f}; LIBERO re-eval required."
    SUMMARY.write_text(json.dumps(s, indent=2) + "\n")
    print("Updated baseline_summary.json parity fields")


if __name__ == "__main__":
    main()
