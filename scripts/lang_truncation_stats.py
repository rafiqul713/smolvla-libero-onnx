#!/usr/bin/env python3
"""Per-task instruction token lengths *as actually fed to the policy* and how many
tasks are truncated at each candidate static ONNX language width.

Uses the checkpoint's own preprocessor steps: `smolvla_new_line_processor` (appends
"\n" to the instruction) followed by `tokenizer_processor` (SmolVLM2 tokenizer,
max_length=48, padding="longest"). `n_tokens_fed` is therefore the exact number of
valid language tokens that reach the ONNX boundary; `n_tokens_raw_no_newline` is the
raw-instruction count used in earlier token statistics (one token fewer).

Usage (conda env: vla):
    python scripts/lang_truncation_stats.py --out results/lang_ablation/truncation_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WIDTHS = (16, 24, 32)


def _task_strings(suite: str) -> list[str]:
    from libero.libero import benchmark

    bm = benchmark.get_benchmark_dict()[suite]()
    return [bm.get_task(i).language for i in range(bm.n_tasks)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    args = ap.parse_args()

    from pathlib import Path as _P
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = SmolVLAConfig.from_pretrained(args.model)
    cfg.pretrained_path = _P(args.model)
    cfg.device = "cpu"
    pre, _post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # Use the checkpoint's own tokenizer step (same object the closed-loop eval uses).
    tok_step = next(s for s in pre.steps if hasattr(s, "_tokenize_text"))
    tok = tok_step.input_tokenizer

    result = {"tokenizer_step": type(tok_step).__name__, "tokenizer_name": getattr(tok_step, "tokenizer_name", None),
              "tokenizer_max_length": cfg.tokenizer_max_length,
              "pad_language_to": cfg.pad_language_to, "widths": list(WIDTHS), "suites": {}}
    for suite in ("libero_spatial", "libero_object"):
        strings = _task_strings(suite)
        rows = []
        for i, s in enumerate(strings):
            # The checkpoint's pipeline runs `smolvla_new_line_processor` (appends "\n" to the
            # task) *before* `tokenizer_processor`; reproduce both so the count is what the
            # policy actually receives at the ONNX boundary.
            s_fed = s if s.endswith("\n") else s + "\n"
            enc = tok_step._tokenize_text(s_fed)
            n_fed = int(enc["attention_mask"].sum())
            ids_plain = tok(s, add_special_tokens=False)["input_ids"]
            rows.append({"task_id": i, "instruction": s, "n_tokens_fed": n_fed,
                         "n_tokens_raw_no_newline": len(ids_plain)})
        fed = [r["n_tokens_fed"] for r in rows]
        result["suites"][suite] = {
            "tasks": rows,
            "mean_tokens_fed": round(sum(fed) / len(fed), 1),
            "max_tokens_fed": max(fed),
            "truncated_at_width": {str(w): sum(1 for n in fed if n > w) for w in WIDTHS},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for suite, d in result["suites"].items():
        print(suite, "mean", d["mean_tokens_fed"], "max", d["max_tokens_fed"], "truncated:", d["truncated_at_width"])


if __name__ == "__main__":
    main()
