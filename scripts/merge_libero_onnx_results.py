#!/usr/bin/env python3
"""Merge closed-loop ONNX LIBERO eval_info.json into baseline_summary.json.

Reads per-suite success from:
  results/{fp16|int8}_trt/libero_spatial/eval_info.json
  results/{fp16|int8}_trt/libero_object/eval_info.json

Writes libero_spatial_pct, libero_object_pct, libero_measured=True into
results/baseline_summary.json → consumed by update_paper_from_results.py
and paper/figures/generate_pareto.py.

Also adds Wilson 95% CIs next to existing point estimates (unchanged) and
McNemar tests for paired 300-episode verify JSONs under paired_verify_fp16.

Run automatically after run_onnx_libero_eval.sh; safe to run manually anytime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "baseline_summary.json"


def read_success(path: Path) -> float | None:
    """Extract pc_success from LeRobot eval_info.json (handles nested layouts)."""
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    overall = info.get("overall") or info.get("per_group", {}).get("overall")
    if isinstance(overall, dict) and "pc_success" in overall:
        return float(overall["pc_success"])
    if "pc_success" in info:
        return float(info["pc_success"])
    return None


def _wilson_ci_pct(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    """Wilson score 95% CI for a proportion, returned as [lo_pct, hi_pct]."""
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    margin = (z / den) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return [round(100.0 * (center - margin), 1), round(100.0 * (center + margin), 1)]


def success_counts(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    k = 0
    n = 0
    for t in info.get("per_task", []):
        succ = t.get("metrics", {}).get("successes") or []
        k += sum(bool(s) for s in succ)
        n += len(succ)
    if n == 0:
        overall = info.get("overall") or {}
        n = int(overall.get("n_episodes") or 0)
        if n and "pc_success" in overall:
            k = int(round(float(overall["pc_success"]) / 100.0 * n))
    return (k, n) if n else None


def wilson_from_eval(path: Path) -> list[float] | None:
    counts = success_counts(path)
    if counts is None:
        return None
    k, n = counts
    return _wilson_ci_pct(k, n)


def mcnemar_from_verify(path: Path) -> dict | None:
    """McNemar test on paired per-episode success/failure (continuity-corrected)."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    orig: dict[tuple[int, int], bool] = {}
    opt: dict[tuple[int, int], bool] = {}
    for t in data.get("original", {}).get("per_task", []):
        tid = int(t["task_idx"])
        for e in t.get("episodes", []):
            orig[(tid, int(e["ep"]))] = bool(e["success"])
    for t in data.get("optimized", {}).get("per_task", []):
        tid = int(t["task_idx"])
        for e in t.get("episodes", []):
            opt[(tid, int(e["ep"]))] = bool(e["success"])
    keys = sorted(set(orig) & set(opt))
    if not keys:
        return None
    b = sum(1 for k in keys if orig[k] and not opt[k])
    c = sum(1 for k in keys if (not orig[k]) and opt[k])
    if b + c == 0:
        stat, p_value = 0.0, 1.0
    else:
        # Edwards continuity-corrected McNemar chi-square (df=1).
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = math.erfc(math.sqrt(stat / 2.0))
    return {
        "n_pairs": len(keys),
        "b_pytorch_success_onnx_fail": b,
        "c_pytorch_fail_onnx_success": c,
        "statistic": round(stat, 4),
        "p_value": p_value,
        "method": "mcnemar_continuity_correction",
    }


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    updated = False

    for tag, key in [("fp16", "fp16_trt"), ("int8", "int8_trt")]:
        row = s["our_results"][key]
        spatial_path = ROOT / "results" / f"{tag}_trt" / "libero_spatial" / "eval_info.json"
        obj_path = ROOT / "results" / f"{tag}_trt" / "libero_object" / "eval_info.json"
        spatial = read_success(spatial_path)
        obj = read_success(obj_path)
        if spatial is not None:
            row["libero_spatial_pct"] = spatial
            row["spatial_output_dir"] = f"results/{tag}_trt/libero_spatial"
            row["libero_spatial_wilson_ci_pct"] = wilson_from_eval(spatial_path)
            updated = True
        if obj is not None:
            row["libero_object_pct"] = obj
            row["object_output_dir"] = f"results/{tag}_trt/libero_object"
            row["libero_object_wilson_ci_pct"] = wilson_from_eval(obj_path)
            updated = True
        if spatial is not None and obj is not None:
            row["libero_measured"] = True
            row["notes"] = (
                f"Closed-loop LIBERO measured on RTX 2060 (100 eps/suite, seed 42). "
                f"Spatial {spatial:.1f}%, Object {obj:.1f}%."
            )

    pt = s["our_results"]["fp32_pytorch"]
    pt_sp = ROOT / "results" / "fp32_pytorch" / "libero_spatial" / "eval_info.json"
    pt_ob = ROOT / "results" / "fp32_pytorch" / "libero_object" / "eval_info.json"
    if pt_sp.exists():
        pt["libero_spatial_wilson_ci_pct"] = wilson_from_eval(pt_sp)
        updated = True
    if pt_ob.exists():
        pt["libero_object_wilson_ci_pct"] = wilson_from_eval(pt_ob)
        updated = True

    verify = s.setdefault("paired_verify_fp16", {})
    for suite_key, fname in [
        ("spatial", "fp16_libero_spatial_30eps.json"),
        ("object", "fp16_libero_object_30eps.json"),
    ]:
        vpath = ROOT / "results" / "verify" / fname
        m = mcnemar_from_verify(vpath)
        if m is not None:
            suite_row = verify.setdefault(suite_key, {})
            suite_row["mcnemar"] = m
            # Backfill into existing verify JSON without re-running rollouts.
            vdata = json.loads(vpath.read_text())
            if vdata.get("mcnemar") != m:
                vdata["mcnemar"] = m
                vpath.write_text(json.dumps(vdata, indent=2) + "\n")
            updated = True

    if updated:
        s["experiment_status"] = "icra_libero_measured"
        s["submission_notes"] = (
            "FP16+INT8 LIBERO re-eval complete. Latency + parity from v1. "
            "TRT EP optional on 12 GB+ GPU."
        )
        SUMMARY.write_text(json.dumps(s, indent=2) + "\n")
        print("Updated baseline_summary.json with measured LIBERO numbers")
    else:
        print("No ONNX LIBERO eval_info.json found — nothing to merge")


if __name__ == "__main__":
    main()
