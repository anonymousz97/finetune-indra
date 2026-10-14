"""Run Datalab's Indra OCR over a dataset split and write raw predictions.

    uv run scripts/run_indra.py --dataset-dir data/comma --out work/eval

Runs in its OWN environment (pip install indra-ocr): Indra is a ~1M-parameter
ONNX pipeline with no PyTorch, unrelated to the transformers stack the project
venv carries, and keeping it separate means its dependencies cannot touch the
metric code. Scoring is a second step, `score_indra.py`, in the project env,
through `textmetrics.score_samples` -- the exact function evaluate.py calls --
so Indra's numbers are directly comparable with every model in results.md.

Settings, and why:

* `furniture=True`. Indra's default DROPS page headers and footers, but every
  reference here transcribes the whole image -- folio numbers, running heads,
  catalogue-card headings. Leaving the default would score Indra for obeying its
  own layout policy rather than for misreading.
* `mode="auto"`: these are JPEGs, so there is no text layer and auto is OCR.
* `providers=["CPUExecutionProvider"]`: Indra is documented and benchmarked on
  CPU, and pinning it keeps the run reproducible on a machine that also offers
  CoreML.
* Two renderings of each page are kept. `prediction` is one line per detected
  line, blank lines dropped -- Indra's `text` format minus the empty lines it
  places between layout blocks, which is the line-by-line task. `verbatim` is
  page.text exactly as Indra emits it, kept so the choice of rendering can be
  shown not to decide the result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import indra_ocr



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="folder with <split>.jsonl and the images it references")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="a drop-in model folder (export_rec.py); default: the shipped models")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()

    ds = args.dataset_dir
    rows = [json.loads(l) for l in (ds / f"{args.split}.jsonl").read_text(encoding="utf-8").split("\n")
            if l.strip()]
    conv = indra_ocr.load(path=str(args.model_dir) if args.model_dir else None, threads=args.threads, mode="auto", furniture=True,
                          providers=["CPUExecutionProvider"])
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"predictions{args.tag}.jsonl"
    started = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            img = r["images"][0]
            target = next(m for m in r["messages"] if m["role"] == "assistant")["content"]
            t = time.time()
            try:
                _, page = next(iter(conv.convert(str(ds / img))))
                verbatim = page.text or ""
                meta = {"source": page.source, "rotation": page.rotation, "n_lines": len(page.lines or [])}
                err = None
            except Exception as exc:  # one unreadable page must not end the run
                verbatim, meta, err = "", {}, f"{type(exc).__name__}: {exc}"
            lines = [ln for ln in verbatim.split("\n") if ln.strip()]
            fh.write(json.dumps({
                "images": [img], "target": target, "target_kind": "text",
                "prediction": "\n".join(lines), "verbatim": verbatim,
                "seconds": round(time.time() - t, 3), "error": err, **meta,
            }, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(rows):
                print(f"ds{args.dataset}: {i}/{len(rows)} pages "
                      f"({(time.time() - started) / i:.2f}s/page)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
