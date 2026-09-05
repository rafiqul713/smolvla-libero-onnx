#!/usr/bin/env python3
"""ONNX action parity: reference export vs candidate (FP16 vs INT8).

Uses tether SmolVLAOnnxServer with identical inputs (images, instruction,
state, noise). Pass gate: min cosine >= 0.99 (EXPERIMENT_GUIDE Option B).

Usage:
    python scripts/parity_smoke_test.py \\
      --reference exports/smolvla_libero --candidate exports/smolvla_libero_int8 --tag int8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def make_cases(n: int, seed: int, chunk: int, action_dim: int, state_dim: int):
    rng = np.random.RandomState(seed)
    instructions = [
        "pick up the alphabet soup and place it in the basket",
        "put the mug on the plate",
        "open the drawer",
        "place the bowl between the plate and the mug",
    ]
    cases = []
    for i in range(n):
        images = [rng.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(3)]
        state = rng.randn(state_dim).astype(np.float32)
        noise = rng.randn(1, chunk, action_dim).astype(np.float32)
        cases.append((images, instructions[i % len(instructions)], state, noise))
    return cases


def load_server(export: Path, device: str):
    from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    srv = SmolVLAOnnxServer(export, device=device, providers=providers, strict_providers=False)
    srv.load()
    return srv


def predict_all(server, cases: list) -> list[np.ndarray]:
    out_list: list[np.ndarray] = []
    for i, (images, instr, state, noise) in enumerate(cases):
        out = server.predict(image=images, instruction=instr, state=state, noise=noise)
        if "error" in out:
            raise RuntimeError(out["error"])
        out_list.append(np.asarray(out["actions"], dtype=np.float32))
        if (i + 1) % 50 == 0:
            print(f"  predict {i + 1}/{len(cases)}")
    return out_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ref_dir = args.reference.resolve()
    cand_dir = args.candidate.resolve()
    cfg = json.loads((ref_dir / "tether_config.json").read_text())
    chunk = int(cfg.get("chunk_size", 50))
    action_dim = int(cfg.get("action_dim", 32))
    state_dim = int(cfg.get("max_state_dim", 32))

    out_dir = ROOT / "results" / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"{args.tag}_parity.json"
    cache_path = out_dir / f"{args.tag}_reference_actions.npz"

    cases = make_cases(args.samples, args.seed, chunk, action_dim, state_dim)
    t0 = time.time()

    if cache_path.exists() and args.tag != "fp16":
        data = np.load(cache_path)
        reference = [data[f"a{i}"] for i in range(args.samples)]
        print(f"Loaded reference actions from {cache_path}")
    else:
        print(f"Reference ONNX: {ref_dir}")
        ref_srv = load_server(ref_dir, args.device)
        reference = predict_all(ref_srv, cases)
        del ref_srv
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        np.savez_compressed(cache_path, **{f"a{i}": reference[i] for i in range(len(reference))})
        print(f"Saved reference cache: {cache_path}")

    print(f"Candidate ONNX: {cand_dir}")
    cand_srv = load_server(cand_dir, args.device)
    cosines: list[float] = []
    max_abs: list[float] = []
    for i, case in enumerate(cases):
        images, instr, state, noise = case
        out = cand_srv.predict(image=images, instruction=instr, state=state, noise=noise)
        if "error" in out:
            raise RuntimeError(out["error"])
        cand_a = np.asarray(out["actions"], dtype=np.float32)
        cosines.append(cosine_similarity(reference[i], cand_a))
        max_abs.append(max_abs_diff(reference[i], cand_a))
        if (i + 1) % 50 == 0:
            print(f"  compare {i + 1}/{len(cases)} min_cos={min(cosines):.6f}")

    result = {
        "tag": args.tag,
        "reference_export": str(ref_dir),
        "candidate_export": str(cand_dir),
        "comparison": "onnx_reference_vs_onnx_candidate",
        "samples": args.samples,
        "seed": args.seed,
        "cosine_min": min(cosines),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_p50": float(np.percentile(cosines, 50)),
        "cosine_p99": float(np.percentile(cosines, 99)),
        "max_abs_max": max(max_abs),
        "max_abs_mean": float(np.mean(max_abs)),
        "gate_min_cosine_0.99": min(cosines) >= 0.99,
        "gate_min_cosine_0.9999": min(cosines) >= 0.9999,
        "wall_s": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")
    if not result["gate_min_cosine_0.99"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
