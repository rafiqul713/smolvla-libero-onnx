#!/usr/bin/env python3
"""Closed-loop LIBERO eval for SmolVLA ONNX exports (FP16 / INT8).

Uses the LeRobot rollout loop with SmolVLAPolicy preprocessing but ONNX inference
via tether's monolithic export (same graph as latency benchmarks).

Critical: pass LeRobot ``lang_tokens`` / ``lang_masks`` (truncated to ONNX's 16
tokens) — do NOT re-tokenize instructions with the export's SmolLM2 bundle.

Usage:
    python scripts/libero_onnx_eval.py \\
        --export exports/smolvla_libero --tag fp16 --suite libero_spatial
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

ROOT = Path(__file__).resolve().parents[1]

# Static shapes baked into tether ONNX export — must match or ORT rejects inputs.
ONNX_MAX_LANG = 16      # default lang width of `tether export`; overridden from the graph at load
ONNX_NUM_CAMERAS = 3    # LIBERO has 2 real cameras; 3rd is padded + mask=False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _truncate_lang(tokens: Tensor, masks: Tensor, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice language tensors to ONNX static shape [1, max_len]."""
    tok = tokens[0:1, :max_len].detach().cpu().numpy().astype(np.int64)
    msk = masks[0:1, :max_len].detach().cpu().numpy().astype(bool)
    if tok.shape[1] < max_len:
        pad = max_len - tok.shape[1]
        tok = np.pad(tok, ((0, 0), (0, pad)), constant_values=0)
        msk = np.pad(msk, ((0, 0), (0, pad)), constant_values=False)
    return tok, msk


def _pad_cameras(
    images: list[Tensor], img_masks: list[Tensor], num_cams: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Convert prepare_images output to ONNX [1,C,H,W] float32 + bool masks."""
    cams: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for img, m in zip(images, img_masks, strict=False):
        cams.append(img[0:1].detach().cpu().numpy().astype(np.float32))
        masks.append(m[0:1].detach().cpu().numpy().astype(bool))

    while len(cams) < num_cams:
        # Dummy camera: -1 pixels, mask=False tells the model to ignore it.
        cams.append(np.ones_like(cams[0]) * -1.0)
        masks.append(np.zeros((1,), dtype=bool))
    return cams[:num_cams], masks[:num_cams]


def _run_onnx(
    session,
    input_names: list[str],
    *,
    images: list[np.ndarray],
    img_masks: list[np.ndarray],
    lang_tokens: np.ndarray,
    lang_masks: np.ndarray,
    state: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    ort_inputs = {
        "img_cam1": images[0],
        "img_cam2": images[1],
        "img_cam3": images[2],
        "mask_cam1": img_masks[0],
        "mask_cam2": img_masks[1],
        "mask_cam3": img_masks[2],
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "noise": noise,
    }
    ort_inputs = {k: v for k, v in ort_inputs.items() if k in input_names}
    return session.run(None, ort_inputs)[0]


class SmolVLAOnnxPolicy(SmolVLAPolicy):
    """SmolVLA with PyTorch preprocessing; ONNX Runtime for action generation.

    Subclasses SmolVLAPolicy so ``select_action`` / action queues behave like
    FP32 eval. Only ``_get_action_chunk`` is overridden to call ORT instead of
    the PyTorch flow-matching forward pass.
    """

    name = "smolvla_onnx"

    def __init__(self, export_dir: Path, model_id: str = "HuggingFaceVLA/smolvla_libero"):
        from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer

        config = SmolVLAConfig.from_pretrained(model_id)
        config.device = "cuda"
        config.load_vlm_weights = False  # weights come from smolvla_libero checkpoint
        config.use_amp = False
        super().__init__(config)

        self._export_dir = export_dir
        onnx_cfg = json.loads((export_dir / "tether_config.json").read_text())
        self._chunk = int(onnx_cfg.get("chunk_size", 50))
        self._action_dim = int(onnx_cfg.get("action_dim", 32))
        self._state_dim = int(onnx_cfg.get("max_state_dim", 32))

        self._server = SmolVLAOnnxServer(
            export_dir,
            device="cuda",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            strict_providers=False,
        )
        self._server.load()
        self._session = self._server._session
        self._input_names = [i.name for i in self._session.get_inputs()]
        # Language width is a static shape baked into the graph: read it from the
        # ONNX input instead of assuming 16 (context-length ablation exports use 24/32).
        lang_in = next(i for i in self._session.get_inputs() if i.name == "lang_tokens")
        self._lang_len = int(lang_in.shape[1]) if isinstance(lang_in.shape[1], int) else ONNX_MAX_LANG
        logger.info("ONNX static language width: %d tokens", self._lang_len)

        # PyTorch weights unused after ORT session is ready — free VRAM for LIBERO.
        del self.model
        torch.cuda.empty_cache()
        self.eval()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        from lerobot.utils.constants import ACTION

        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        batch = self._prepare_batch(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)

        lang_tokens, lang_masks = _truncate_lang(
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            self._lang_len,
        )
        # CRITICAL: use LeRobot preprocessor tokens — never re-tokenize with export SmolLM2.
        cam_arrays, mask_arrays = _pad_cameras(images, img_masks, ONNX_NUM_CAMERAS)

        if noise is None:
            noise_np = np.random.randn(1, self._chunk, self._action_dim).astype(np.float32)
        else:
            noise_np = noise.detach().cpu().numpy()
            if noise_np.ndim == 2:
                noise_np = noise_np[None, ...]

        state_np = state[0:1].detach().cpu().numpy().astype(np.float32)
        if state_np.shape[1] < self._state_dim:
            state_np = np.pad(
                state_np,
                ((0, 0), (0, self._state_dim - state_np.shape[1])),
                constant_values=0.0,
            )
        elif state_np.shape[1] > self._state_dim:
            state_np = state_np[:, : self._state_dim]

        actions_np = _run_onnx(
            self._session,
            self._input_names,
            images=cam_arrays,
            img_masks=mask_arrays,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state_np,
            noise=noise_np.astype(np.float32),
        )

        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=state.device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        dim = self.config.action_feature.shape[0]
        return actions[:, :, :dim]


def _task_capture_hook(policy: SmolVLAOnnxPolicy):
    def hook(_idx: int, transition: dict) -> None:
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = comp.get("task")
        if task is None:
            return
        policy._current_instruction = task[0] if isinstance(task, (list, tuple)) else str(task)

    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suite", required=True, choices=["libero_spatial", "libero_object"])
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--task-ids", type=str, default=None, help="e.g. 0 or 0,1 for quick tests")
    args = ap.parse_args()

    export_dir = args.export.resolve()
    if not (export_dir / "model.onnx").exists():
        raise SystemExit(f"Missing model.onnx in {export_dir}")

    out_dir = args.output or ROOT / "results" / f"{args.tag}_trt" / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.configs.default import EvalConfig
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all

    model_path = Path("HuggingFaceVLA/smolvla_libero")
    policy_cfg = SmolVLAConfig.from_pretrained(str(model_path))
    policy_cfg.pretrained_path = model_path
    policy_cfg.device = "cuda"
    policy_cfg.use_amp = False
    policy_cfg.load_vlm_weights = False
    policy_cfg.n_action_steps = 1   # re-infer every control step (strictest eval)
    policy_cfg.num_steps = 10       # flow-matching denoise steps inside model

    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    env_cfg = LiberoEnv(task=args.suite, task_ids=task_ids)
    eval_cfg = EvalConfig(n_episodes=args.n_episodes, batch_size=1, use_async_envs=False)
    cfg = EvalPipelineConfig(env=env_cfg, eval=eval_cfg, policy=policy_cfg, output_dir=out_dir, seed=args.seed)

    logger.info("Export: %s  Suite: %s  Out: %s", export_dir, args.suite, out_dir)
    envs = make_env(cfg.env, n_envs=1, use_async_envs=False)

    policy = SmolVLAOnnxPolicy(export_dir)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=0,
            videos_dir=None,
            return_episode_data=False,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )

    close_envs(envs)
    (out_dir / "eval_info.json").write_text(json.dumps(info, indent=2) + "\n")
    pc = info["overall"]["pc_success"]
    logger.info("Done: %s %s -> %.1f%% success", args.tag, args.suite, pc)
    print(json.dumps({"tag": args.tag, "suite": args.suite, "pc_success": pc, "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
