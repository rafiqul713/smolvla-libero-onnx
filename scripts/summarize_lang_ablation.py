#!/usr/bin/env python3
"""Collect the language-width ablation into results/lang_ablation/summary.json.

Width 16 closed-loop numbers come from the paper's existing runs
(results/fp16_trt/*/eval_info.json — the export in exports/smolvla_libero,
whose static language width is 16). Widths 24/32 come from results/lang_ablation/.
Latency for all widths comes from the uniform bench (scripts/bench_onnx_lang_width.py).

Also reports Wilson 95% CIs for suite-level (n=100) and width-16 truncation
group (n=50) success proportions. Point estimates are unchanged.

Also reports an independent two-proportion (continuity-corrected chi-square)
test of PyTorch+AMP Spatial (n=100) vs.\ width-24 ONNX Spatial (n=100).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lang_ablation"
WIDTHS = (16, 24, 32)
# Fed-token tasks truncated at static width 16 (Table IV grouping).
TRUNC_AT_16_TASK_IDS = frozenset({0, 1, 4, 5, 6})


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


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


def _two_proportion_chi2(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Independent 2x2 continuity-corrected chi-square (not McNemar)."""
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    chi2, p_value, dof, _ = chi2_contingency(table, correction=True)
    return {
        "method": "chi2_contingency_continuity_correction",
        "k1": k1,
        "n1": n1,
        "k2": k2,
        "n2": n2,
        "statistic": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": float(p_value),
    }


def _success_counts(p: Path) -> tuple[int, int] | None:
    d = _load(p)
    if not d:
        return None
    k = 0
    n = 0
    for t in d.get("per_task", []):
        succ = t.get("metrics", {}).get("successes") or []
        k += sum(bool(s) for s in succ)
        n += len(succ)
    if n == 0:
        n = int(d.get("overall", {}).get("n_episodes") or 0)
        k = int(round(float(d["overall"]["pc_success"]) / 100.0 * n)) if n else 0
    return k, n


def _success(p: Path):
    d = _load(p)
    if not d:
        return None
    return round(float(d["overall"]["pc_success"]), 1)


def _success_with_wilson(p: Path) -> tuple[float | None, list[float] | None]:
    pct = _success(p)
    counts = _success_counts(p)
    if counts is None:
        return pct, None
    k, n = counts
    return pct, _wilson_ci_pct(k, n)


def _group_n50(p: Path) -> dict | None:
    """Success counts and Wilson CIs for previously-truncated-at-16 vs untruncated (n=50 each)."""
    d = _load(p)
    if not d:
        return None
    trunc_k = untrunc_k = 0
    trunc_n = untrunc_n = 0
    for t in d.get("per_task", []):
        tid = t.get("task_id")
        if tid is None:
            tid = t.get("task_idx")
        succ = t.get("metrics", {}).get("successes") or []
        if tid in TRUNC_AT_16_TASK_IDS:
            trunc_k += sum(bool(s) for s in succ)
            trunc_n += len(succ)
        else:
            untrunc_k += sum(bool(s) for s in succ)
            untrunc_n += len(succ)
    if trunc_n == 0 and untrunc_n == 0:
        return None
    return {
        "prev_trunc_at_16_successes": trunc_k,
        "prev_trunc_at_16_n": trunc_n,
        "prev_trunc_at_16_wilson_ci_pct": _wilson_ci_pct(trunc_k, trunc_n),
        "untrunc_at_16_successes": untrunc_k,
        "untrunc_at_16_n": untrunc_n,
        "untrunc_at_16_wilson_ci_pct": _wilson_ci_pct(untrunc_k, untrunc_n),
    }


def _per_task(p: Path):
    d = _load(p)
    if not d:
        return None
    out = {}
    for t in d.get("per_task", []):
        succ = t.get("metrics", {}).get("successes")
        if succ:
            out[str(t.get("task_id"))] = round(100.0 * sum(bool(s) for s in succ) / len(succ), 1)
    return out or None


def main() -> None:
    trunc = _load(OUT / "truncation_stats.json") or {}
    rows = {}
    for w in WIDTHS:
        if w == 16:
            sp = ROOT / "results" / "fp16_trt" / "libero_spatial" / "eval_info.json"
            ob = ROOT / "results" / "fp16_trt" / "libero_object" / "eval_info.json"
            sp_rep = ROOT / "results" / "int8_trt" / "libero_spatial" / "eval_info.json"
            ob_rep = ROOT / "results" / "int8_trt" / "libero_object" / "eval_info.json"
        else:
            sp = OUT / f"w{w}" / "libero_spatial" / "eval_info.json"
            ob = OUT / f"w{w}" / "libero_object" / "eval_info.json"
            sp_rep = ob_rep = Path("/nonexistent")
        lat = _load(OUT / f"w{w}" / "latency.json") or {}
        sp_pct, sp_ci = _success_with_wilson(sp)
        ob_pct, ob_ci = _success_with_wilson(ob)
        sp_rep_pct, sp_rep_ci = _success_with_wilson(sp_rep)
        ob_rep_pct, ob_rep_ci = _success_with_wilson(ob_rep)
        rows[str(w)] = {
            "spatial_pct": sp_pct,
            "object_pct": ob_pct,
            "spatial_wilson_ci_pct": sp_ci,
            "object_wilson_ci_pct": ob_ci,
            "spatial_pct_replicate": sp_rep_pct,
            "object_pct_replicate": ob_rep_pct,
            "spatial_wilson_ci_pct_replicate": sp_rep_ci,
            "object_wilson_ci_pct_replicate": ob_rep_ci,
            "spatial_per_task": _per_task(sp),
            "object_per_task": _per_task(ob),
            "spatial_n50_groups": _group_n50(sp),
            "truncated_spatial_tasks": (trunc.get("suites", {}).get("libero_spatial", {})
                                        .get("truncated_at_width", {}).get(str(w))),
            "truncated_object_tasks": (trunc.get("suites", {}).get("libero_object", {})
                                       .get("truncated_at_width", {}).get(str(w))),
            "p50_ms": lat.get("p50_ms"),
            "p99_ms": lat.get("p99_ms"),
            "nvidia_smi_used_mb": lat.get("nvidia_smi_used_mb"),
            "onnx_size_mb": lat.get("onnx_size_mb"),
        }
    pt_sp = ROOT / "results" / "fp32_pytorch" / "libero_spatial" / "eval_info.json"
    pt_ob = ROOT / "results" / "fp32_pytorch" / "libero_object" / "eval_info.json"
    pt_sp_pct, pt_sp_ci = _success_with_wilson(pt_sp)
    pt_ob_pct, pt_ob_ci = _success_with_wilson(pt_ob)
    w24_sp = OUT / "w24" / "libero_spatial" / "eval_info.json"
    pt_counts = _success_counts(pt_sp)
    w24_counts = _success_counts(w24_sp)
    two_prop = None
    if pt_counts is not None and w24_counts is not None:
        k1, n1 = pt_counts
        k2, n2 = w24_counts
        two_prop = _two_proportion_chi2(k1, n1, k2, n2)
        two_prop["comparison"] = "pytorch_amp_spatial_vs_onnx_width24_spatial"
    summary = {
        "protocol": "lerobot-eval closed loop, 10 eps/task x 10 tasks, seed 42, n_action_steps=1, 10 flow steps, "
                    "batch 1, ONNX Runtime CUDA EP, RTX 2060 6GB; latency = 50 warmup + 1000 timed calls",
        "precision_note": "all arms are FP32 monolithic ONNX (tether 0.12.0 monolithic path ignores --precision)",
        "ci_note": "Wilson score 95% intervals; suite n=100; Table IV groups n=50 (tasks truncated at width 16 vs not)",
        "pytorch_amp_baseline": {
            "spatial_pct": pt_sp_pct if pt_sp_pct is not None else 70.0,
            "object_pct": pt_ob_pct if pt_ob_pct is not None else 88.0,
            "spatial_wilson_ci_pct": pt_sp_ci,
            "object_wilson_ci_pct": pt_ob_ci,
            "spatial_n50_groups": _group_n50(pt_sp),
            "p99_ms": 1181,
        },
        "widths": rows,
        "pytorch_vs_width24_spatial_two_proportion": two_prop,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["widths"], indent=2))
    if two_prop is not None:
        print(json.dumps({"pytorch_vs_width24_spatial_two_proportion": two_prop}, indent=2))


if __name__ == "__main__":
    main()
