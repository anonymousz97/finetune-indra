"""Turn Indra's inference-only rec.onnx into a trainable PyTorch module, and prove it.

    uv run scripts/convert_rec.py --test-jsonl data/comma/validation.jsonl --out work/rec_torch.pt

Indra ships its line recognizer as an ONNX graph with greedy decoding baked in:
the graph ends `rec.head` (a 1x1 Conv1d, [1678, 128, 1]) -> Transpose -> ArgMax,
and outputs token ids rather than scores. Training needs the scores, so the graph
is cut at `/rec/Transpose_output_0` -- the [B, T, 1678] logits that feed the
ArgMax -- and that subgraph is converted to PyTorch with onnx2torch.

A conversion that is merely plausible is worthless here: a subtly wrong padding
mask or a mis-mapped activation would still "train", and the damage would show
up only as a mysteriously bad fine-tune. So this does not trust the conversion.
On real line crops -- produced by Indra's own detector and crop rule from real
holdout pages, not synthetic tensors -- it checks the converted module against
the original ONNX graph three ways: raw logits (max absolute difference),
per-timestep argmax ids, and the final decoded strings. It refuses to save
unless the decoded text is identical on every line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx2torch import convert

from indra_ocr.pipeline import OrtOCR, crop_lines
from indra_ocr.rec_preproc import pad_batch, width_batches
from indra_ocr.vocab import Vocab

from shipped import MODEL_DIR  # noqa: E402  (pinned revision; see shipped.py)
LOGITS = "/rec/Transpose_output_0"


def patch_clips(m: onnx.ModelProto) -> int:
    """Rewrite one-sided Clip nodes as Min/Max, which onnx2torch can convert.

    The graph has two Clips, each with one bound omitted (an empty input name --
    valid ONNX, meaning "unbounded"). onnx2torch cannot convert an omitted bound,
    nor a bound computed at run time:
      /rec/Clip             no min, max computed -> Min(x, max)   (clamps the
                            valid-timestep count to the sequence length)
      /rec/blocks.0/se/Clip min 1.0, no max      -> Max(x, 1.0)   (keeps the masked
                            squeeze-excitation mean from dividing by zero)
    Same output names, so nothing downstream changes. Equivalence is then CHECKED
    on real crops below, not assumed from this argument.
    """
    from onnx import helper
    g, n_fixed = m.graph, 0
    for i, node in enumerate(list(g.node)):
        if node.op_type != "Clip":
            continue
        x = node.input[0]
        lo = node.input[1] if len(node.input) > 1 else ""
        hi = node.input[2] if len(node.input) > 2 else ""
        if lo and hi:
            continue  # two-sided: leave it
        bound, op = (hi, "Min") if hi else (lo, "Max")
        new = helper.make_node(op, [x, bound], list(node.output), name=node.name + "_" + op)
        g.node.remove(node)
        g.node.insert(i, new)
        n_fixed += 1
    return n_fixed


def rec_module(onnx_path: Path, workdir: Path) -> torch.nn.Module:
    """rec.onnx -> trainable PyTorch module: cut at the logits, rewrite the Clips, convert.

    The exact path the equivalence test in main() validates (452/452 identical
    decoded lines). Rebuilding from ONNX on whichever machine trains -- instead of
    shipping a pickled module -- keeps a torch-version mismatch between laptop and
    cluster from ever entering the weights.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    sub = workdir / "rec_logits.onnx"
    onnx.utils.extract_model(str(onnx_path), str(sub), ["x", "widths"], [LOGITS])
    m = onnx.load(str(sub))
    patch_clips(m)
    onnx.checker.check_model(m)
    return convert(m).eval()


def greedy(ids: np.ndarray, n: int) -> list[int]:
    seq = ids[:n]
    keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
    return seq[keep].tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path, action="append", required=True,
                    help="jsonl(s) of real pages to test on; image paths relative to its folder")
    ap.add_argument("--pages", type=int, default=6, help="pages per jsonl to test on")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sub = args.out.with_name("rec_logits.onnx")
    # An UNPATCHED extraction is the reference, so the logit comparison below
    # covers the Clip rewrite as well as the conversion -- not just the latter.
    ref_sub = args.out.with_name("rec_logits_reference.onnx")
    onnx.utils.extract_model(str(MODEL_DIR / "rec.onnx"), str(ref_sub), ["x", "widths"], [LOGITS])
    onnx.utils.extract_model(str(MODEL_DIR / "rec.onnx"), str(sub), ["x", "widths"], [LOGITS])
    patched = onnx.load(str(sub))
    print(f"rewrote {patch_clips(patched)} one-sided Clip node(s) as Min/Max")
    onnx.checker.check_model(patched)
    onnx.save(patched, str(sub))
    model = convert(patched).eval()
    print(f"converted: {sum(p.numel() for p in model.parameters()):,} trainable parameters")

    full = ort.InferenceSession(str(MODEL_DIR / "rec.onnx"), providers=["CPUExecutionProvider"])
    part = ort.InferenceSession(str(ref_sub), providers=["CPUExecutionProvider"])
    ocr = OrtOCR(MODEL_DIR, threads=4, providers=["CPUExecutionProvider"])
    vocab = Vocab(json.loads(json.loads((MODEL_DIR / "meta.json").read_text())["vocab"])["chars"])

    n_lines = n_text_same = n_ids_same = 0
    worst = 0.0
    for tj in args.test_jsonl:
        rows = [json.loads(l) for l in tj
                .read_text(encoding="utf-8").split("\n") if l.strip()][: args.pages]
        for r in rows:
            gray = cv2.imread(str(tj.parent / r["images"][0]), cv2.IMREAD_GRAYSCALE)
            boxes = ocr.detect(gray)
            if not len(boxes):
                continue
            crops = crop_lines(gray, boxes[:, :4], 32, getattr(ocr, "mask", None))
            for idx in width_batches([c.shape[1] for c in crops], 64):
                arr, widths = pad_batch([crops[j] for j in idx])
                ids_ref, _, lengths = full.run(None, {"x": arr, "widths": widths})
                logits_ref = part.run(None, {"x": arr, "widths": widths})[0]
                with torch.no_grad():
                    logits_pt = model(torch.from_numpy(arr), torch.from_numpy(widths)).numpy()
                worst = max(worst, float(np.abs(logits_pt - logits_ref).max()))
                ids_pt = logits_pt.argmax(-1)
                for k in range(len(idx)):
                    n = int(lengths[k])
                    n_lines += 1
                    n_ids_same += int(np.array_equal(ids_pt[k, :n], ids_ref[k, :n]))
                    a = vocab.decode(greedy(ids_pt[k], n))
                    b = vocab.decode(greedy(ids_ref[k], n))
                    n_text_same += int(a == b)
    print(f"tested {n_lines} real line crops from {len(args.test_jsonl) * args.pages} pages")
    print(f"  max |logit diff|          : {worst:.2e}")
    print(f"  identical argmax ids      : {n_ids_same}/{n_lines}")
    print(f"  identical decoded strings : {n_text_same}/{n_lines}")
    if n_text_same != n_lines:
        print("REFUSING TO SAVE: the converted module does not reproduce the original")
        return 1
    torch.save(model, args.out)
    print(f"EQUIVALENT -- saved trainable module to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
