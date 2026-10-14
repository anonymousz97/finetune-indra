"""Turn aligned (detector box, reference line) pairs into recognizer training crops.

    uv run scripts/build_aligned.py --pairs work/nara-align --tag=-r1 \
        --dataset-root data/nara --out work/nara-lines-r1

Reads the `<split><tag>-pairs.jsonl` files align_lines.py writes and produces the
same `<split>.npz` + `<split>-labels.json` layout build_lines.py does, so
train_rec.py trains on either without change. Crops use the DETECTOR's box --
Indra's exact inference geometry -- through Indra's own `crop_line`. Labels get
Indra's `canonical()` (NFC plus its look-alike folds) with tabs turned into
spaces: a tab is invisible on the page, and the NARA transcripts contain 1,109.

The holdout split's pairs are pseudo-labels too, so they are a training-progress
signal only, and a biased one: alignment keeps the lines Indra could already
partly read. The number that counts is end-to-end page CER on the 762 holdout
pages through Datalab's unchanged pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from indra_ocr.rec_preproc import crop_line
from indra_ocr.vocab import canonical



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="the folder the pairs' image paths are relative to")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset_root

    for split in ("train", "validation"):
        src = args.pairs / f"{split}{args.tag}-pairs.jsonl"
        crops, labels, pages, sims = [], [], [], []
        for line in src.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row["pairs"]:
                continue
            gray = cv2.imread(str(ds / row["image"]), cv2.IMREAD_GRAYSCALE)
            for box, ref, sim, _ in row["pairs"]:
                text = canonical(ref.replace("\t", " ").strip())
                if not text:
                    continue
                crops.append(crop_line(gray, box))
                labels.append(text)
                pages.append(row["page"])
                sims.append(sim)
        widths = np.array([c.shape[1] for c in crops], np.int32)
        flat = np.concatenate([(c * 255).round().astype(np.uint8).ravel() for c in crops])
        np.savez_compressed(args.out / f"{split}.npz", flat=flat, widths=widths,
                            pages=np.array(pages, np.int32), sims=np.array(sims, np.float32))
        (args.out / f"{split}-labels.json").write_text(json.dumps(labels, ensure_ascii=False))
        print(f"{split}: {len(labels):,} line crops from {len(set(pages)):,} pages | "
              f"median pair similarity {np.median(sims):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
