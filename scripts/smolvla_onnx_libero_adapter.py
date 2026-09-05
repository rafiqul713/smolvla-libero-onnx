#!/usr/bin/env python3
"""SmolVLA monolithic ONNX adapter for tether.eval.libero_rollout.

Legacy adapter: implements InferenceProtocol for ``run_libero_rollout(inference=...)``.
The verify script now prefers SmolVLAOnnxPolicy + use_native=True (same select_action
path as libero_onnx_eval.py). Keep this module for experiments with tether's raw
predict_action_chunk path.

Key rule: pass LeRobot ``lang_tokens`` truncated to 16 — never re-tokenize with the
export's bundled SmolLM2 tokenizer (that bug caused 0% LIBERO success).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Must match static ONNX input shapes in exports/smolvla_libero/model.onnx
ONNX_MAX_LANG = 16
ONNX_NUM_CAMERAS = 3


class SmolVLAOnnxLIBEROAdapter:
    """ONNX Runtime inference for tether's LIBERO rollout primitive."""

    def __init__(self, export_dir: str | Path, device: str = "cuda"):
        from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer

        self._export_dir = Path(export_dir)
        self._server = SmolVLAOnnxServer(
            self._export_dir,
            device=device,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            strict_providers=False,
        )
        self._server.load()
        self._session = self._server._session
        self._input_names = [i.name for i in self._session.get_inputs()]
        cfg = self._server.config
        self._chunk = int(cfg.get("chunk_size", 50))
        self._action_dim = int(cfg.get("action_dim", 32))
        self._state_dim = int(cfg.get("max_state_dim", 32))

    def reset_cache(self) -> None:
        return None

    def get_stats(self) -> dict[str, Any]:
        return {"active_providers": list(self._session.get_providers())}

    @staticmethod
    def _as_chw(img: Any) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        return arr

    @staticmethod
    def _as_mask(mask: Any) -> np.ndarray:
        m = np.asarray(mask)
        if m.ndim == 0:
            return np.array([bool(m)], dtype=bool)
        if m.ndim == 1:
            return m.astype(bool)
        return m[0].astype(bool)

    def predict_action_chunk(
        self,
        *,
        img_base: Any,
        img_wrist_l: Any,
        img_wrist_r: Any,
        mask_base: Any,
        mask_wrist_l: Any,
        mask_wrist_r: Any,
        lang_tokens: Any,
        lang_masks: Any,
        noise: Any,
        state: Any,
        episode_id: str,
    ) -> Any:
        del episode_id
        images = [self._as_chw(img_base), self._as_chw(img_wrist_l), self._as_chw(img_wrist_r)]
        masks = [self._as_mask(mask_base), self._as_mask(mask_wrist_l), self._as_mask(mask_wrist_r)]

        tok = np.asarray(lang_tokens, dtype=np.int64)
        msk = np.asarray(lang_masks, dtype=bool)
        if tok.ndim == 1:
            tok = tok[None, :]
        if msk.ndim == 1:
            msk = msk[None, :]
        tok = tok[:, :ONNX_MAX_LANG]
        msk = msk[:, :ONNX_MAX_LANG]
        if tok.shape[1] < ONNX_MAX_LANG:
            # Pad to fixed [1, 16] — ONNX graph has static lang input width.
            pad = ONNX_MAX_LANG - tok.shape[1]
            tok = np.pad(tok, ((0, 0), (0, pad)))
            msk = np.pad(msk, ((0, 0), (0, pad)), constant_values=False)

        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim == 1:
            state_arr = state_arr[None, :]
        if state_arr.shape[1] < self._state_dim:
            state_arr = np.pad(
                state_arr,
                ((0, 0), (0, self._state_dim - state_arr.shape[1])),
                constant_values=0.0,
            )
        elif state_arr.shape[1] > self._state_dim:
            state_arr = state_arr[:, : self._state_dim]

        noise_arr = np.asarray(noise, dtype=np.float32)

        ort_inputs = {
            "img_cam1": images[0][None, ...] if images[0].ndim == 3 else images[0],
            "img_cam2": images[1][None, ...] if images[1].ndim == 3 else images[1],
            "img_cam3": images[2][None, ...] if images[2].ndim == 3 else images[2],
            "mask_cam1": masks[0],
            "mask_cam2": masks[1],
            "mask_cam3": masks[2],
            "lang_tokens": tok,
            "lang_masks": msk,
            "state": state_arr,
            "noise": noise_arr,
        }
        ort_inputs = {k: v for k, v in ort_inputs.items() if k in self._input_names}
        return self._session.run(None, ort_inputs)[0]


def load_smolvla_policy_shim(model_id: str = "HuggingFaceVLA/smolvla_libero"):
    """Load SmolVLA policy + LeRobot pre/post processors for tether rollout.

    Adds ``_preprocess_images`` hook so tether's rollout can pad a 3rd dummy camera
    (required by monolithic ONNX, optional in native PyTorch).
    """
    import torch
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

    cfg = SmolVLAConfig.from_pretrained(model_id)
    cfg.device = "cuda"
    cfg.load_vlm_weights = False
    cfg.use_amp = False
    policy = SmolVLAPolicy.from_pretrained(model_id, config=cfg)
    policy.eval().to("cuda")

    def _preprocess_images(batch_pp):
        images, img_masks = policy.prepare_images(batch_pp)
        while len(images) < ONNX_NUM_CAMERAS:
            images.append(torch.ones_like(images[0]) * -1.0)
            img_masks.append(torch.zeros_like(img_masks[0]))
        return images, img_masks

    policy._preprocess_images = _preprocess_images  # type: ignore[attr-defined]

    proc_path = model_id
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        proc_path,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        proc_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor
