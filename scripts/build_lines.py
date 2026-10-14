"""Build line-level (crop, label) pairs for fine-tuning Indra's recognizer.

    uv run scripts/build_lines.py --dataset-dir data/comma --out work/lines

Indra recognizes LINES; our datasets are pages. Dataset 48 carries a box per
line -- `<|box_start|>(x0,y0),(x1,y1)<|box_end|>text`, coordinates normalised
0-1000 to the page crop, proven aligned to the image for the joint bbox+HTR work
-- so each page yields exact (line image, line text) pairs. Its validation split
is the same 243 pages as dataset 32's holdout, so the recognizer can be scored on
ground-truth line crops as well as end to end.

Two decisions, both forced by measurement on holdout pages:

* **The geometry is the DETECTOR's, the text is ALTO's.** ALTO line boxes include
  ascender and descender space; Indra's detector draws tighter ink boxes, measured
  at a median 0.72x the ALTO height on the same lines. Crops are resized to 32 px
  high, so training on raw ALTO boxes would show the model text at a different
  scale from inference. Tightening ALTO boxes to their ink was tried first and did
  nothing (1.00x): a line box overlaps its neighbours' ascenders and descenders,
  so nearly every row holds some ink. So each ALTO line is instead matched to the
  detector box Indra itself would crop -- IoU >= 0.5 AND covering >= 80% of the
  line's width, because a box over half a split line paired with the whole line's
  text would teach the model to hallucinate. Unmatched lines fall back to the ALTO
  box shrunk to the detector's median height about its centre. The match rate and
  resulting height ratio are printed, so the choice is checked, not assumed.
* **Crops come from Indra's own `crop_line`.** Its docstring: "Training and
  inference must both call crop_line, so the text scale and the margins match."

Labels are canonicalised with Indra's own `canonical()` (NFC plus its look-alike
folds), exactly as its training did. Lines whose label is empty are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from indra_ocr.pipeline import OrtOCR
from indra_ocr.rec_preproc import crop_line
from indra_ocr.vocab import canonical

from shipped import MODEL_DIR  # noqa: E402  (pinned revision; see shipped.py)
BOX = re.compile(r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>(.*)")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


DET_HEIGHT = 0.72  # measured: detector box height / ALTO box height, median, holdout


def detector_box(alto, det: list) -> tuple[tuple, bool]:
    """(box to crop, matched?) -- the detector's box for this line when one fits."""
    width = alto[2] - alto[0]
    good = [d for d in det if iou(alto, d) >= 0.5 and (d[2] - d[0]) >= 0.8 * width]
    if good:
        return tuple(max(good, key=lambda d: iou(alto, d))), True
    cy, h = (alto[1] + alto[3]) / 2, (alto[3] - alto[1]) * DET_HEIGHT
    return (alto[0], cy - h / 2, alto[2], cy + h / 2), False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="folder with <split>.jsonl (boxed targets) and the images they reference")
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset_dir
    ocr = OrtOCR(MODEL_DIR, threads=8, providers=["CPUExecutionProvider"])

    for split in args.splits.split(","):
        rows = [json.loads(l) for l in (ds / f"{split}.jsonl").read_text(encoding="utf-8").split("\n")
                if l.strip()]
        crops, labels, pages, heights, matched = [], [], [], [], 0
        for p, r in enumerate(rows):
            gray = cv2.imread(str(ds / r["images"][0]), cv2.IMREAD_GRAYSCALE)
            H, W = gray.shape
            det = [tuple(b) for b in ocr.detect(gray)[:, :4].tolist()]
            target = next(m for m in r["messages"] if m["role"] == "assistant")["content"]
            for line in target.split("\n"):
                m = BOX.match(line)
                if not m:
                    continue
                a, b, c, d, text = m.groups()
                text = canonical(text.strip())
                if not text:
                    continue
                box = (int(a) * W / 1000, int(b) * H / 1000, int(c) * W / 1000, int(d) * H / 1000)
                crop_box, hit = detector_box(box, det)
                matched += hit
                heights.append((crop_box[3] - crop_box[1]) / max(1.0, box[3] - box[1]))
                crops.append(crop_line(gray, crop_box))
                labels.append(text)
                pages.append(p)
            if (p + 1) % 200 == 0:
                print(f"  {split}: {p + 1}/{len(rows)} pages, {len(labels):,} lines", flush=True)
        widths = np.array([c.shape[1] for c in crops], np.int32)
        # One flat uint8 array (ink map x255) + offsets: compact, and loads in seconds.
        flat = np.concatenate([(c * 255).round().astype(np.uint8).ravel() for c in crops])
        np.savez_compressed(args.out / f"{split}.npz", flat=flat, widths=widths,
                            pages=np.array(pages, np.int32))
        (args.out / f"{split}-labels.json").write_text(json.dumps(labels, ensure_ascii=False))
        msg = f"{split}: {len(rows)} pages -> {len(labels):,} lines, median width {int(np.median(widths))}px"
        msg += (f" | {matched / max(1, len(labels)):.0%} cropped from a matched detector box,"
                f" crop height {np.median(heights):.2f}x ALTO (detector measured {DET_HEIGHT}x)")
        print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
