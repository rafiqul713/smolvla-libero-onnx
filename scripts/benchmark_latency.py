#!/usr/bin/env python
"""Latency benchmark for SmolVLA on LIBERO observations (RTX 2060).

Mirrors the exact inference path of `lerobot-eval` (same policy factory,
same processor pipelines) so latency numbers are comparable to the
accuracy runs. With n_action_steps=1 each `select_action` call executes
the full model: vision encoder + language backbone + flow-matching steps.

Usage (after activating the `vla` conda env and sourcing scripts/env.sh):
    python scripts/benchmark_latency.py --precision fp32
    python scripts/benchmark_latency.py --precision fp32_amp
"""

import argparse
import json
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch

import lerobot.scripts.lerobot_eval as ev
from lerobot.envs.configs import LiberoEnv

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "HuggingFaceVLA/smolvla_libero"


def nvidia_smi_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp32_amp"], default="fp32_amp")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--output", default=None, help="Output JSON path")
    args = ap.parse_args()

    use_amp = args.precision == "fp32_amp"
    out_path = Path(args.output) if args.output else ROOT / "results" / "fp32_pytorch" / f"latency_{args.precision}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[1/4] Building one LIBERO env ({args.suite}, task 0) for a real observation ...")
    env_cfg = LiberoEnv(task=args.suite, task_ids=[0])
    envs = ev.make_env(env_cfg, n_envs=1)
    env = next(iter(next(iter(envs.values())).values()))

    print(f"[2/4] Loading policy {MODEL_ID} (load_vlm_weights=False, n_action_steps=1, num_steps=10) ...")
    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(MODEL_ID)
    policy_cfg.pretrained_path = MODEL_ID
    policy_cfg.device = "cuda"
    if hasattr(policy_cfg, "load_vlm_weights"):
        policy_cfg.load_vlm_weights = False
    policy_cfg.n_action_steps = 1  # every select_action runs the full model
    if hasattr(policy_cfg, "num_steps"):
        policy_cfg.num_steps = 10

    policy = ev.make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": str(policy.config.device)}}
    preprocessor, postprocessor = ev.make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=MODEL_ID,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, _ = ev.make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    print("[3/4] Preparing one preprocessed observation ...")
    policy.reset()
    raw_obs, _ = env.reset(seed=42)
    observation = ev.preprocess_observation(raw_obs)
    try:
        observation["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        observation["task"] = [""] * env.num_envs
    observation = env_preprocessor(observation)
    observation = preprocessor(observation)

    amp_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
    torch.cuda.reset_peak_memory_stats()

    print(f"[4/4] Warmup {args.warmup} calls, then timing {args.runs} calls ({args.precision}) ...")
    with torch.inference_mode(), amp_ctx:
        for _ in range(args.warmup):
            policy.reset()
            policy.select_action(observation)
        torch.cuda.synchronize()

        latencies_ms = []
        for i in range(args.runs):
            policy.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.select_action(observation)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{args.runs} done, running p50={statistics.median(latencies_ms):.1f} ms")

    latencies_ms.sort()
    n = len(latencies_ms)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "precision": f"{args.precision}_pytorch",
        "model": MODEL_ID,
        "n_action_steps": 1,
        "num_flow_steps": 10,
        "runs": n,
        "warmup": args.warmup,
        "mean_ms": round(sum(latencies_ms) / n, 2),
        "p50_ms": round(latencies_ms[int(n * 0.50)], 2),
        "p95_ms": round(latencies_ms[int(n * 0.95)], 2),
        "p99_ms": round(latencies_ms[int(n * 0.99)], 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
        "nvidia_smi_used_mb": nvidia_smi_used_mb(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
