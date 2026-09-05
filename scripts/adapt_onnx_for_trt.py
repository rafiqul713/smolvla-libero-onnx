#!/usr/bin/env python3
"""Strip ScatterND `reduction` attributes so TensorRT EP will consider the graph.

Tether refuses TensorRT when any ScatterND node has a `reduction` attribute
(even reduction=none). TensorRT 10.x does not import that ONNX attribute.
Our exported SmolVLA graph has 1086 ScatterND nodes, all reduction=none,
so removing the attribute is numerically a no-op and unblocks TRT.

Does not rewrite weights: loads ONNX without external data and saves in place
next to the existing model.onnx.data file.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx


def adapt(onnx_path: Path, backup: bool = True) -> int:
    if backup:
        bak = onnx_path.with_suffix(".onnx.pre_trt_adapt")
        if not bak.exists():
            shutil.copy2(onnx_path, bak)
            print(f"backup: {bak}")
    # load_external_data=False: weights stay in model.onnx.data — we only edit the graph.
    model = onnx.load(str(onnx_path), load_external_data=False)
    n = 0
    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue
        # TensorRT 10 rejects reduction=none even when it means "no extra reduction".
        kept = [a for a in node.attribute if a.name != "reduction"]
        if len(kept) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(kept)
            n += 1
    tmp = onnx_path.with_suffix(".onnx.tmp")
    onnx.save(model, str(tmp))
    tmp.replace(onnx_path)
    print(f"stripped reduction from {n} ScatterND nodes -> {onnx_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", nargs="?", default="exports/smolvla_libero/model.onnx")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    adapt(Path(args.onnx_path), backup=not args.no_backup)


if __name__ == "__main__":
    main()
