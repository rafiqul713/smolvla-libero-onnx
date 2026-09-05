#!/usr/bin/env python3
"""SmolVLA paired LIBERO verify: native PyTorch vs monolithic ONNX.

Mirrors ``tether verify`` (original + optimized arms on the same rollout loop).
Upstream tether v0.12 only ships Pi05/Triton verify; this script fills the
SmolVLA ONNX gap for ICRA-grade paired rollouts.

Flow:
  1. ORIGINAL arm — full PyTorch SmolVLAPolicy via ``select_action``
  2. Free GPU memory (delete PyTorch weights)
  3. OPTIMIZED arm — SmolVLAOnnxPolicy (same ``select_action`` path, ONNX inside)

Important: both arms use ``use_native=True`` so control flow matches. Do NOT use
the tether ``predict_action_chunk`` path for OPTIMIZED — it uses a different
replan loop and produced 0% ONNX success in smoke tests.

Usage:
    python scripts/run_smolvla_verify.py \\
        --export exports/smolvla_libero \\
        --suite libero_spatial \\
        --num-episodes 30 \\
        --output results/verify/fp16_spatial.json

Output JSON fields: original_success_rate_pct, optimized_success_rate_pct,
parity_gap_pct (optimized minus original), and mcnemar (paired binary test).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mcnemar_paired(original: dict, optimized: dict) -> dict:
    """Continuity-corrected McNemar on paired per-episode success/failure."""
    orig: dict[tuple[int, int], bool] = {}
    opt: dict[tuple[int, int], bool] = {}
    for t in original.get("per_task", []):
        tid = int(t["task_idx"])
        for e in t.get("episodes", []):
            orig[(tid, int(e["ep"]))] = bool(e["success"])
    for t in optimized.get("per_task", []):
        tid = int(t["task_idx"])
        for e in t.get("episodes", []):
            opt[(tid, int(e["ep"]))] = bool(e["success"])
    keys = sorted(set(orig) & set(opt))
    b = sum(1 for k in keys if orig[k] and not opt[k])
    c = sum(1 for k in keys if (not orig[k]) and opt[k])
    if b + c == 0:
        stat, p_value = 0.0, 1.0
    else:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num-episodes", type=int, default=30)
    ap.add_argument("--task-indices", type=str, default=None, help="e.g. 0,1,2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize-size", type=int, default=512)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    export_dir = args.export.resolve()
    if not (export_dir / "model.onnx").exists():
        raise SystemExit(f"Missing model.onnx in {export_dir}")

    task_indices = None
    if args.task_indices:
        task_indices = [int(x.strip()) for x in args.task_indices.split(",")]

    from tether.eval.libero_rollout import run_libero_rollout

    from libero_onnx_eval import SmolVLAOnnxPolicy
    from smolvla_onnx_libero_adapter import load_smolvla_policy_shim

    policy, preprocessor, postprocessor = load_smolvla_policy_shim()
    common = dict(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=args.suite,
        num_episodes=args.num_episodes,
        task_indices=task_indices,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        seed=args.seed,
    )

    logger.info("ORIGINAL arm (native PyTorch select_action)")
    original = run_libero_rollout(
        inference=None, use_native=True, label="ORIGINAL", **common,
    )

    # ONNX policy loads its own weights; drop PyTorch model first to avoid OOM.
    logger.info("OPTIMIZED arm (monolithic ONNX via select_action)")
    import torch

    del policy
    torch.cuda.empty_cache()
    onnx_policy = SmolVLAOnnxPolicy(export_dir)
    # Same rollout code path as ORIGINAL — only _get_action_chunk differs (ONNX).
    optimized = run_libero_rollout(
        policy=onnx_policy,
        inference=None,
        use_native=True,
        label="OPTIMIZED",
        **{k: v for k, v in common.items() if k != "policy"},
    )

    out = {
        "export": str(export_dir),
        "suite": args.suite,
        "num_episodes_per_task": args.num_episodes,
        "seed": args.seed,
        "original_success_rate_pct": original.get("success_rate_pct"),
        "optimized_success_rate_pct": optimized.get("success_rate_pct"),
        "original": original,
        "optimized": optimized,
        "parity_gap_pct": (
            (optimized.get("success_rate_pct") or 0) - (original.get("success_rate_pct") or 0)
        ),
        "mcnemar": mcnemar_paired(original, optimized),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(
        {
            "output": str(args.output),
            "original_pct": out["original_success_rate_pct"],
            "optimized_pct": out["optimized_success_rate_pct"],
            "gap_pct": out["parity_gap_pct"],
            "mcnemar": out["mcnemar"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
