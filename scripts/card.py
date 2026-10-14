"""Model card for an Indra recognizer fine-tune -- with the license terms it MUST carry.

    uv run scripts/card.py --model-dir work/model-comma --run work/run-comma \
        --base-pred work/eval/predictions.jsonl --tuned-pred work/eval/predictions-ft.jsonl \
        --repo you/my-indra-finetune --pages 9906 --language la \
        --data comma-project/deep-jsonl --compare "Qwen3.5-0.8B full FT, same pages=0.1458"

Writes README.md into --model-dir and copies Datalab's LICENSE beside it. Every
number is read from files: the training history, and predictions scored with
`textmetrics.score_samples`.

**The license is not yours to choose.** Indra's weights are under Datalab's modified
OpenRAIL-M, and a fine-tune is a "Derivative of the Model" (§1e). It may be
redistributed (§2, §4) only if you: include a copy of the License (§4b); mark every
modified file (§4c); carry the Attachment A use restrictions as an enforceable
provision with notice to users (§4a, §5); credit Datalab, link the original and
state your changes (§7); and do not suggest endorsement (§7, §10). It is
SHARE-ALIKE (§8): the derivative and its output take this same license -- so the
card declares `license: openrail`, never Apache or MIT. Attachment A also bars
use by any entity over US$20M in revenue or funding except for personal or
research purposes. Read the LICENSE yourself before publishing; this card is a
summary, and the License governs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score  # noqa: E402
from shipped import REPO, REVISION  # noqa: E402

UNMODIFIED = ["det.onnx", "order.onnx", "orient.onnx", "script.onnx", "rec_cjk.onnx"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True, help="the folder export_rec.py wrote")
    ap.add_argument("--run", type=Path, required=True, help="the train_rec.py output folder")
    ap.add_argument("--base-pred", type=Path, required=True, help="run_indra.py output, shipped models")
    ap.add_argument("--tuned-pred", type=Path, required=True, help="run_indra.py output, this model")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pages", type=int, required=True, help="pages the training lines came from")
    ap.add_argument("--language", default="la")
    ap.add_argument("--data", default="", help="Hub dataset id the training data came from")
    ap.add_argument("--tags", default="", help="comma-separated extra tags")
    ap.add_argument("--compare", action="append", default=[],
                    help='extra reference rows, "label=CER" (same holdout, same metric!)')
    args = ap.parse_args()

    hist = json.loads((args.run / "history.json").read_text())
    z, f = score(args.base_pred), score(args.tuned_pred)
    best = min(hist["epochs"], key=lambda e: e["cer_nfd"])
    last = hist["epochs"][-1]
    new = hist["new_chars"]
    mins = sum(e["seconds"] for e in hist["epochs"]) / 60
    row = lambda label, d: (f"| {label} | {d['cer_nfd']:.4f} | {d['cer_nfd_macro']:.4f} | "
                            f"{d['wer_nfd']:.4f} | {d['line_recall_nfd']:.4f} |")
    extra = "".join(f"| {c.split('=')[0]} | {float(c.split('=')[1]):.4f} | | | |\n" for c in args.compare)
    tags = ["ocr", "htr", "handwritten-text-recognition", "cultural-heritage", "onnx", "ctc",
            *[t.strip() for t in args.tags.split(",") if t.strip()]]
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    data_yaml = f"datasets:\n  - {args.data}\n" if args.data else ""
    selection = ("the final epoch" if best["epoch"] == last["epoch"] else
                 f"epoch {best['epoch']} ({best['cer_nfd']:.4f}; the final epoch scored {last['cer_nfd']:.4f})")

    card = f"""---
license: openrail
license_name: openrail-m-modified-datalab
license_link: LICENSE
base_model: datalab-to/indra-ocr
{data_yaml}language:
  - {args.language}
pipeline_tag: image-to-text
tags:
{tags_yaml}
---

# {args.repo.split('/')[-1]}

A fine-tune of the line recognizer in Datalab's
[**Indra OCR**](https://huggingface.co/datalab-to/indra-ocr).

> **Not affiliated with or endorsed by Datalab.** This is an independent derivative
> of their model. Indra OCR and its weights are © Datalab; this repo is distributed
> under their license, below.

## What was changed, and what was not

Indra is a pipeline of small ONNX models. **Only the line recognizer was retrained.**

| file | status |
|---|---|
| `rec.onnx` | **MODIFIED** — recognizer weights fine-tuned; classifier head widened from 1,678 to {1678 + len(new)} classes |
| `meta.json` | **MODIFIED** — `vocab` extended by {len(new)} characters; all other fields unchanged |
| {', '.join(f'`{u}`' for u in UNMODIFIED)} | unmodified copies of Datalab's files |
| `LICENSE` | Datalab's license, verbatim |

The weights were written back **into Datalab's original graph** rather than re-exported,
so masking, sequence lengths and decoding are theirs.
{f"Characters added: {' '.join(new)}" if new else ""}

## Results

End to end with Datalab's unchanged detector and reading-order models, on
{z['pages']} held-out pages. CER after NFD normalisation.

| model | CER (NFD) | macro | WER (NFD) | line recall |
|---|---|---|---|---|
{row("Indra OCR, as shipped", z)}
{row("**this model**", f)}
{extra}
On ground-truth line crops the recognizer scores {best['cer_nfd']:.4f} (shipped: {hist['zero_shot']['cer_nfd']:.4f}).
The gap to the end-to-end figure is Indra's line detector, which this fine-tune does
not change.

## Usage

```python
# pip install indra-ocr
import indra_ocr
from huggingface_hub import snapshot_download

conv = indra_ocr.load(path=snapshot_download("{args.repo}"), furniture=True)
for _, page in conv.convert("page.jpg"):
    print(page.text)
```

`furniture=True` keeps page headers and footers, which Indra drops by default. Runs on CPU.

## Training

CTC fine-tune on {hist['lines_used']:,} line crops from {args.pages:,} pages, {len(hist['epochs'])} epochs
({mins:.0f} minutes). Checkpoint: {selection}.

## License and use restrictions

Distributed under Datalab's **modified OpenRAIL-M license** — see [`LICENSE`](LICENSE),
included verbatim. It is **share-alike**: this model, any derivative of it, and its
output are governed by that same license.

By using this model you agree to the use-based restrictions in Attachment A of the
License, which apply to it as an enforceable provision. In summary — the License
governs, not this summary — you may not use it, its derivatives or its output:
unlawfully or to infringe intellectual-property rights; **for any purpose if you or
your employer have over US$20M in annual gross revenue or over US$20M in funding,
except where your use is limited to personal use or research purposes**; or if you
or your employer offer a product or service that competes with Datalab's. Broader
licensing is available from Datalab at https://www.datalab.to/.
"""
    (args.model_dir / "README.md").write_text(card, encoding="utf-8")
    # Fetched explicitly at the PINNED revision: Indra's own loader caches only the
    # model files, and the license shipped must be the one for these exact weights.
    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download(REPO, "LICENSE", revision=REVISION), args.model_dir / "LICENSE")
    print(f"wrote {args.model_dir / 'README.md'} and LICENSE | "
          f"shipped {z['cer_nfd']:.4f} -> fine-tuned {f['cer_nfd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
