"""Open-loop check: do the width-16 and width-24 ONNX graphs (and PyTorch) agree on identical inputs?

Run once per mode (conda env vla, after `source scripts/env_ort_cuda12.sh`):
    python scripts/check_lang_width_openloop.py onnx16
    python scripts/check_lang_width_openloop.py onnx24
    python scripts/check_lang_width_openloop.py torch
Outputs results/lang_ablation/openloop_<mode>.json; the aggregated comparison is
results/lang_ablation/openloop_width_check.json.
"""
import gc, json, sys
import numpy as np, torch
import onnxruntime as ort

ROOT = "/home/rafiqul/Documents/Projects/Research/VLAResearch"
MODEL = "HuggingFaceVLA/smolvla_libero"
mode = sys.argv[1]  # onnx16 | onnx24 | torch
rng = np.random.default_rng(0)

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from pathlib import Path
cfg = SmolVLAConfig.from_pretrained(MODEL); cfg.pretrained_path = Path(MODEL); cfg.device = "cpu"
pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=cfg.pretrained_path,
                                  preprocessor_overrides={"device_processor": {"device": "cpu"}})
tok_step = next(s for s in pre.steps if hasattr(s, "_tokenize_text"))
instr = ["pick up the black bowl between the plate and the ramekin and place it on the plate",  # 16+ tokens
         "pick up the alphabet soup and place it in the basket"]                                 # short
encs = [tok_step._tokenize_text(s) for s in instr]

imgs = [rng.uniform(-1, 1, size=(1, 3, 512, 512)).astype(np.float32) for _ in range(3)]
state = rng.standard_normal((1, 32)).astype(np.float32)
noise = rng.standard_normal((1, 50, 32)).astype(np.float32)

def lang(enc, width):
    ids = enc["input_ids"].numpy().astype(np.int64)[:1]; m = enc["attention_mask"].numpy().astype(bool)[:1]
    n = int(m.sum()); ids = ids[:, :n]; m = m[:, :n]
    ids = ids[:, :width]; m = m[:, :width]
    if ids.shape[1] < width:
        p = width - ids.shape[1]
        ids = np.pad(ids, ((0, 0), (0, p))); m = np.pad(m, ((0, 0), (0, p)), constant_values=False)
    return ids, m, n

outs = {}
if mode.startswith("onnx"):
    w = int(mode[4:]); d = f"{ROOT}/exports/smolvla_libero" + ("" if w == 16 else f"_lang{w}")
    sess = ort.InferenceSession(f"{d}/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider"
    for i, enc in enumerate(encs):
        ids, m, n = lang(enc, w)
        feeds = {"img_cam1": imgs[0], "img_cam2": imgs[1], "img_cam3": imgs[2],
                 "mask_cam1": np.array([True]), "mask_cam2": np.array([True]), "mask_cam3": np.array([False]),
                 "lang_tokens": ids, "lang_masks": m, "state": state, "noise": noise}
        a = sess.run(None, feeds)[0]
        outs[f"instr{i}"] = {"n_tokens": n, "truncated": n > w, "actions": a[0, :, :7].tolist()}
else:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg2 = SmolVLAConfig.from_pretrained(MODEL); cfg2.device = "cuda"; cfg2.load_vlm_weights = False; cfg2.use_amp = False
    pol = SmolVLAPolicy.from_pretrained(MODEL, config=cfg2).eval().to("cuda")
    pol.model.config.num_steps = 10
    with torch.no_grad():
        for i, enc in enumerate(encs):
            ids, m, n = lang(enc, 48)  # PyTorch: full (untruncated) instruction, padded like LeRobot "longest"
            ids = ids[:, :n]; m = m[:, :n]
            im = [torch.tensor(x).cuda() for x in imgs]
            mk = [torch.tensor([True]).cuda(), torch.tensor([True]).cuda(), torch.tensor([False]).cuda()]
            a = pol.model.sample_actions(im, mk, torch.tensor(ids).cuda(), torch.tensor(m).cuda(),
                                         torch.tensor(state).cuda(), noise=torch.tensor(noise).cuda())
            outs[f"instr{i}"] = {"n_tokens": n, "truncated": False, "actions": a[0, :, :7].float().cpu().tolist()}
json.dump(outs, open(f"{ROOT}/results/lang_ablation/openloop_{mode}.json", "w"))
print(mode, {k: (v["n_tokens"], v["truncated"]) for k, v in outs.items()})
