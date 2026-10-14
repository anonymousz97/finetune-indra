"""Score Indra prediction files with the CHAI metric module.

    uv run scripts/score.py work/eval/predictions.jsonl work/eval/predictions-ft.jsonl

Each file is what run_indra.py writes (one JSON object per page with `target` and
`prediction`). Scoring goes through `textmetrics.score_samples` -- the same
function the CHAI-Document-HPC evaluation uses -- so numbers here are directly
comparable with that project's VLM results.

CER is quoted after NFD normalisation (`cer_nfd`). Manuscript corpora write the
same glyph two ways -- `ũ` precomposed, or `u` + combining tilde -- and charging
two edits for a difference that is not on the page measures the encoding, not
the reading. On plain English the normalisation is a no-op.

Micro CER weights a page by its length; macro weights every page equally. A big
gap between them means the errors are concentrated -- usually a few sparse pages
where decoding ran away. Treat differences under ~0.02 CER as noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textmetrics  # noqa: E402

KEYS = [("cer_nfd", "CER"), ("cer_nfd_macro", "macro"), ("wer_nfd", "WER"),
        ("line_recall_nfd", "line recall")]


def score(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    rows = [{"images": r.get("images"), "target": r["target"],
             "target_kind": r.get("target_kind", "text"), "prediction": r["prediction"]} for r in rows]
    agg = textmetrics.score_samples(rows)
    agg["pages"] = len(rows)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--json", type=Path, help="also write the aggregates here")
    args = ap.parse_args()
    results = {str(p): score(p) for p in args.predictions}
    width = max(len(p.name) for p in args.predictions) + 2
    print(f"{'file':{width}s} {'pages':>6s}" + "".join(f"{label:>13s}" for _, label in KEYS))
    for p in args.predictions:
        a = results[str(p)]
        print(f"{p.name:{width}s} {a['pages']:6d}" + "".join(f"{a[k]:13.4f}" for k, _ in KEYS))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
