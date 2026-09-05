#!/usr/bin/env python3
"""Export SmolVLA to monolithic ONNX with a configurable static language width.

Why this exists
---------------
`tether export` (0.12.0, --monolithic path) hard-codes the language input to
[1, 16] tokens (see tether/exporters/monolithic.py::export_smolvla_monolithic).
LeRobot's preprocessor pads SmolVLA instructions to "longest" with
tokenizer_max_length=48, so some LIBERO-Spatial instructions are longer than 16
tokens and get truncated at the ONNX boundary.

This script reproduces tether's SmolVLA monolithic export step-for-step (same
patches, same torch.export / torch.onnx.export calls, same post-export fixes,
same weight-fusion pass, same tether_config.json writer) but takes the language
width as an argument, so the *only* variable that changes between exports is the
static `lang_tokens` / `lang_masks` shape.

Precision note: like `tether export --monolithic`, this produces an FP32 ONNX
graph. tether 0.12.0 ignores `--precision` on the monolithic path.

Usage (conda env: vla-export):
    python scripts/export_smolvla_lang_width.py --lang-len 24 \
        --output exports/smolvla_libero_lang24
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("export_lang_width")


def export(model_id: str, output_dir: Path, *, lang_len: int, num_steps: int, target: str) -> dict:
    from tether.exporters import monolithic as M

    M._require_monolithic_deps()

    import torch
    import torch.nn as nn
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    M.apply_export_patches()
    M._raise_if_smolvla_export_patches_failed()

    logger.info("Loading %s", model_id)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval().to("cpu").to(torch.float32)
    policy.model.config.num_steps = num_steps
    M._force_eager_attn(policy.model)
    logger.info("Loaded in %.1fs; num_steps=%d; lang_len=%d", time.time() - t0, num_steps, lang_len)

    class SmolVLAMonolithicWrapper(nn.Module):
        def __init__(self, smolvla_model):
            super().__init__()
            self.model = smolvla_model

        def forward(
            self,
            img_cam1, img_cam2, img_cam3,
            mask_cam1, mask_cam2, mask_cam3,
            lang_tokens, lang_masks,
            state, noise,
        ):
            images = [img_cam1, img_cam2, img_cam3]
            img_masks = [mask_cam1, mask_cam2, mask_cam3]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise,
            )

    wrapper = SmolVLAMonolithicWrapper(policy.model).eval()
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    # Identical to tether's dummy inputs except for the language width.
    dummy = dict(
        img_cam1=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        img_cam2=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        img_cam3=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        mask_cam1=torch.ones(B, dtype=torch.bool),
        mask_cam2=torch.ones(B, dtype=torch.bool),
        mask_cam3=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 49152, (B, lang_len), dtype=torch.long),
        lang_masks=torch.ones(B, lang_len, dtype=torch.bool),
        state=torch.randn(B, state_dim, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("torch.export.export ...")
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(wrapper, tuple(dummy.values()), dynamic_shapes=None, strict=False)
    logger.info("torch.export: %.1fs", time.time() - t0)

    logger.info("torch.onnx.export ...")
    t0 = time.time()
    torch.onnx.export(
        ep, tuple(dummy.values()), str(onnx_path),
        input_names=list(dummy.keys()), output_names=["actions"],
        opset_version=19,
    )
    logger.info("ONNX conversion: %.1fs", time.time() - t0)

    fixes = M._fix_onnx_where_dtype_mismatches(onnx_path)
    logger.info("post-export Cast fixes: %d", fixes)

    M._write_tether_config(
        output_dir, policy.config, num_steps=num_steps,
        model_id=model_id, model_type="smolvla", target=target,
    )
    # Record the ablation variable in the config so downstream tools can read it.
    cfg_path = output_dir / "tether_config.json"
    cfg_json = json.loads(cfg_path.read_text())
    cfg_json["lang_max_tokens"] = lang_len
    cfg_json["precision_note"] = "FP32 ONNX (tether 0.12.0 monolithic path; no FP16/INT8 conversion applied)"
    cfg_json["export_script"] = "scripts/export_smolvla_lang_width.py"
    cfg_path.write_text(json.dumps(cfg_json, indent=2, default=str))

    # Same optional weight-fusion pass as tether.exporters.monolithic.export_monolithic.
    try:
        from tether.exporters.weight_fusion import fuse_weights
        fuse_weights(str(onnx_path), num_steps=num_steps)
        logger.info("Weight fusion pass complete")
    except Exception as e:  # non-fatal, mirrors tether
        logger.warning("Weight fusion pass failed (non-fatal): %s", e)

    size_mb = onnx_path.stat().st_size / 1e6
    total_mb = sum(f.stat().st_size for f in output_dir.glob("*.data")) / 1e6 + size_mb
    return {"status": "ok", "onnx_path": str(onnx_path), "size_mb": total_mb,
            "num_steps": num_steps, "lang_len": lang_len}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--lang-len", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--target", default="desktop")
    args = ap.parse_args()
    t0 = time.time()
    res = export(args.model, args.output.resolve(), lang_len=args.lang_len,
                 num_steps=args.num_steps, target=args.target)
    res["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
