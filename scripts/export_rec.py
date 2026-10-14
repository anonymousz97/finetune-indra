"""Write a (fine-tuned) recognizer back into a drop-in Indra model directory.

    uv run scripts/export_rec.py --state work/run-comma/best_state.pt \
        --vocab work/run-comma/vocab.json --out work/model-comma

The result is a folder Datalab's UNCHANGED pipeline loads with
`indra_ocr.load(path=...)`: every other model (detector, reading order,
orientation, script classifier, CJK recognizer) is copied as shipped, rec.onnx
carries the new weights, and meta.json's `vocab` holds the extended vocabulary.

**The fine-tuned weights are written INTO Datalab's original graph; the graph is
never re-exported.** Re-exporting with torch.onnx.export was tried first and
failed: tracing froze a shape, building the padding mask for the example input's
160 timesteps, so any line of another width could not broadcast against it. The
architecture does not change -- only parameter values, plus the classifier head's
row count -- and nothing downstream of `rec.head` hard-codes the number of
classes (ArgMax and the softmax confidence run over the last axis). So each
trainable tensor is copied into the initializer its ONNX node reads, and the
masking, the `lengths` computation and the decoding tail stay Datalab's own.

Each torch parameter is mapped to its initializer through the node that owns it
(onnx2torch names a converted module after its ONNX node), and the script REFUSES
to write a folder if any trainable parameter goes unmapped or if the patched graph
disagrees with the PyTorch module on any line it is checked on.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from shipped import MODEL_DIR  # noqa: E402  (pinned revision; see shipped.py)


def patch_weights(model: torch.nn.Module, graph: onnx.GraphProto) -> int:
    """Copy every trainable tensor into the initializer its ONNX node reads."""
    inits = {t.name: t for t in graph.initializer}
    modules = dict(model.named_modules())
    roles = {"Conv": ["weight", "bias"], "BatchNormalization": ["weight", "bias"]}
    done = set()
    for node in graph.node:
        if node.op_type not in roles:
            continue
        # onnx2torch names a module after its node but swaps "." for "/" (dots are
        # torch's module-path separator): /rec/blocks.0/dw/Conv -> rec/blocks/0/dw/Conv
        mod = modules.get(node.name.lstrip("/").replace(".", "/"))
        if mod is None:
            continue
        for attr, init_name in zip(roles[node.op_type], node.input[1:]):
            t = getattr(mod, attr, None)
            if t is None or init_name not in inits:
                continue
            arr = t.detach().cpu().numpy().astype(np.float32)
            inits[init_name].CopyFrom(numpy_helper.from_array(arr, init_name))
            done.add(id(t))
    missing = [n for n, p in model.named_parameters() if id(p) not in done]
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} trainable parameter(s) not mapped: {missing[:5]}")
    return len(done)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, help="a pickled module (.pt); local convenience")
    ap.add_argument("--state", type=Path, help="a state_dict (best_state.pt): portable, preferred")
    ap.add_argument("--vocab", type=Path, help="vocab.json from training; omit to keep the shipped one")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.state:
        # Rebuild from Datalab's ONNX, size the head from the checkpoint, load weights.
        from convert_rec import rec_module
        sd = torch.load(args.state, map_location="cpu", weights_only=True)
        # Conversion scratch goes to a temp dir: args.out is the folder that gets
        # PUBLISHED, and a leftover _convert/ there would ship to the Hub.
        import tempfile
        model = rec_module(MODEL_DIR / "rec.onnx", Path(tempfile.mkdtemp()))
        head_key = next(k for k in sd if k.endswith(".weight") and sd[k].shape[1:] == (128, 1)
                        and sd[k].shape[0] >= 1678)
        name = head_key[: -len(".weight")]
        n_cls = sd[head_key].shape[0]
        setattr(model, name, torch.nn.Conv1d(128, n_cls, 1))
        model.load_state_dict(sd)
        model = model.eval()
    else:
        model = torch.load(args.model, weights_only=False).cpu().eval()
    proto = onnx.load(str(MODEL_DIR / "rec.onnx"))
    n = patch_weights(model, proto.graph)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(args.out / "rec.onnx"))

    # The patched graph must agree with the module, at several widths (the very
    # thing the traced re-export got wrong), on per-timestep argmax ids.
    sess = ort.InferenceSession(str(args.out / "rec.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for widths in ([1200, 640, 96, 88], [304, 2048], [64]):
        wm = max(widths)
        x = (rng.random((len(widths), 1, 32, wm)) > 0.85).astype(np.float32)
        for k, w in enumerate(widths):
            x[k, 0, :, w:] = 0
        w_arr = np.array(widths, np.int64)
        ids_o, _, len_o = sess.run(None, {"x": x, "widths": w_arr})
        with torch.no_grad():
            ids_pt = model(torch.from_numpy(x), torch.from_numpy(w_arr)).argmax(-1).numpy()
        for k in range(len(widths)):
            L = int(len_o[k])
            assert np.array_equal(ids_o[k, :L], ids_pt[k, :L]), f"ids differ at width {widths[k]}"
    print(f"patched {n} parameter tensors into Datalab's graph; ids match the module at 7 widths")

    for f in MODEL_DIR.iterdir():
        if f.name not in ("rec.onnx", "meta.json"):
            shutil.copy2(f, args.out / f.name)
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    if args.vocab:
        meta["vocab"] = args.vocab.read_text()
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"wrote drop-in model folder {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
