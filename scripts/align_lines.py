"""Pseudo-label NARA line crops by aligning Indra's detected lines to page transcripts.

    uv run scripts/align_lines.py --dataset-dir data/nara --split train \
        --out work/nara-align --workers 10

NARA (si-test) has page-level transcriptions and NO line geometry, unlike
deep-jsonl's ALTO. So training lines are recovered by alignment: Indra's own
detector finds the lines and its recognizer reads them; each page's detected
lines are then aligned to the reference lines with a monotonic dynamic program
(both run top to bottom), and a detected line is paired with a reference line
only when its reading is similar enough to be confident they are the same line.

Why each rule:

* **Monotonic, gaps free.** Reading order is shared, so pairs may not cross; but
  either side may skip lines -- the detector over-segments (1.55x on comma) and
  finds marginal marks the reference omits, and misses some lines too.
* **A similarity threshold, not a forced match.** A pair is accepted only if
  1 - normalised edit distance >= `--min-sim`. A wrong pair teaches the model to
  read one line as another, which is worse than losing a line.
* **A length-ratio check.** A detected FRAGMENT of a split line reads as part of
  its reference line; similarity alone can let that through, and a half-line
  image paired with the whole line's text teaches hallucination. Pairs whose
  lengths differ by more than `--max-len-ratio` are rejected.
* **Iterable.** `--model-dir` reads lines with a fine-tuned recognizer, so round 2
  can re-align with better readings and recover lines round 1 had to reject.

The detector boxes are what get cropped -- exactly Indra's inference geometry.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from shipped import MODEL_DIR as SHIPPED  # noqa: E402  (pinned revision)
_W = {}


def similarity(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def align(det: list[str], ref: list[str], min_sim: float, max_ratio: float) -> list[tuple[int, int, float]]:
    """Monotonic alignment maximising total similarity over accepted pairs."""
    n, m = len(det), len(ref)
    S = np.zeros((n + 1, m + 1))
    back = np.zeros((n + 1, m + 1), np.int8)  # 0 diag(match) 1 skip det 2 skip ref
    sims = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = det[i - 1], ref[j - 1]
            ok = a and b and max(len(a), len(b)) / max(1, min(len(a), len(b))) <= max_ratio
            s = similarity(a, b) if ok else 0.0
            sims[(i, j)] = s
            cand = [(S[i - 1, j - 1] + s if s >= min_sim else -1, 0), (S[i - 1, j], 1), (S[i, j - 1], 2)]
            S[i, j], back[i, j] = max(cand)
    out, i, j = [], n, m
    while i > 0 and j > 0:
        b = back[i, j]
        if b == 0:
            out.append((i - 1, j - 1, sims[(i, j)])); i -= 1; j -= 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def _init(model_dir: str):
    import indra_ocr
    _W["conv"] = indra_ocr.load(path=model_dir, threads=1, mode="auto", furniture=True,
                                providers=["CPUExecutionProvider"])


def _page(args):
    """One page -> (page index, [(box, ref_text, sim, det_text)], n_det, n_ref)."""
    p, img, target, min_sim, max_ratio = args
    try:
        _, page = next(iter(_W["conv"].convert(img)))
    except Exception:
        return p, [], 0, 0
    lines = [l for l in (page.lines or []) if (l.text or "").strip()]
    ref = [r.strip() for r in target.split("\n") if r.strip()]
    pairs = align([l.text.strip() for l in lines], ref, min_sim, max_ratio)
    return p, [(list(map(float, lines[i].box[:4])), ref[j], s, lines[i].text.strip())
               for i, j, s in pairs], len(lines), len(ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="folder with <split>.jsonl (page-level targets) and its images")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=SHIPPED)
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--max-len-ratio", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset_dir
    rows = [json.loads(l) for l in (ds / f"{args.split}.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    jobs = [(p, str(ds / r["images"][0]),
             next(m for m in r["messages"] if m["role"] == "assistant")["content"],
             args.min_sim, args.max_len_ratio) for p, r in enumerate(rows)]
    out = args.out / f"{args.split}{args.tag}-pairs.jsonl"
    t0 = time.time()
    n_det = n_ref = n_pair = 0
    with mp.get_context("spawn").Pool(args.workers, _init, (str(args.model_dir),)) as pool, \
            out.open("w", encoding="utf-8") as fh:
        for k, (p, pairs, nd, nr) in enumerate(pool.imap_unordered(_page, jobs, chunksize=4), 1):
            n_det += nd; n_ref += nr; n_pair += len(pairs)
            fh.write(json.dumps({"page": p, "image": rows[p]["images"][0], "pairs": pairs,
                                 "n_det": nd, "n_ref": nr}, ensure_ascii=False) + "\n")
            if k % 500 == 0 or k == len(jobs):
                print(f"  {k:,}/{len(jobs):,} pages  {(time.time() - t0) / k:.2f}s/page  "
                      f"paired {n_pair:,} of {n_ref:,} reference lines ({n_pair / max(1, n_ref):.0%})",
                      flush=True)
    print(f"wrote {out}: {n_pair:,} pairs | {n_ref:,} reference lines | {n_det:,} detected lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
