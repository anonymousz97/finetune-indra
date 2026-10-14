# finetune-indra

Fine-tune the line recognizer of Datalab's [**Indra OCR**](https://huggingface.co/datalab-to/indra-ocr)
on your own transcriptions — and drop the result straight back into Datalab's
unchanged pipeline.

Indra is a ~1M-parameter, CPU-only OCR pipeline shipped as ONNX graphs, with no
training code. This repo recovers a trainable PyTorch recognizer from the shipped
`rec.onnx`, proves the conversion is exact, trains it with CTC, and writes the new
weights **back into Datalab's own graph**, so detection, reading order, masking and
decoding all stay theirs.

## What it achieved

Medieval Latin under the CATMuS graphemic standard (abbreviations kept, e.g. `ꝑ`, `ꝰ`,
combining tildes), scored end to end on **243 held-out pages, each in a different hand**:

| model | params | trained on | CER (NFD) | line recall | runs on |
|---|---|---|---|---|---|
| Indra, as shipped | 1.2 M | — | 0.5834 | 0.0003 | CPU |
| **Indra, fine-tuned** | 1.2 M | 1,116 pages | **0.1780** | 0.0737 | CPU |
| **Indra, fine-tuned** | 1.2 M | 9,906 pages | **0.1499** | 0.1290 | CPU |
| Qwen3.5-2B, LoRA | 2 B | the same 1,116 | 0.1895 | 0.1078 | GPU |
| Qwen3.5-0.8B, full fine-tune | 0.8 B | the same 9,906 | 0.1458 | 0.1672 | GPU |
| LightOnOCR-3-0.8B, full fine-tune | 0.8 B | the same 9,906 | 0.1206 | 0.1678 | GPU |

On identical training pages the fine-tuned Indra **ties** a 2B VLM (1,116 pages) and a
full fine-tune of Qwen3.5-0.8B (9,906 pages) — both gaps are inside the ~0.02 CER
run-to-run noise — at roughly 1/700th of the parameters, on a CPU, at ~0.17 s/page.
It trails the strongest VLM by ~0.03 CER and trails all of them on line recall.

**The recognizer is no longer the main limit.** Read from ground-truth line crops it
scores 0.1048 (9,906 pages); end to end it scores 0.1499. The ~0.045 difference is
Indra's line *detector*, which this repo does not retrain: on manuscript pages it
draws ~1.55× as many lines as the references contain, splitting some and missing a
few. That is also why line recall stays low.

## ⚠️ License — read before publishing anything

The code here is yours to license. **Indra's weights are not.** They are under
Datalab's *modified OpenRAIL-M* license (shipped with the model as `LICENSE`), and a
fine-tune is a "Derivative of the Model". You may redistribute one, but only if you:

- include the License verbatim (§4b);
- mark every modified file (§4c) and state your changes (§7);
- carry the Attachment A use restrictions as an enforceable provision (§4a, §5);
- credit Datalab, link the original, and **not** suggest endorsement (§7, §10).

It is **share-alike** (§8): the derivative *and its output* take the same license — not
Apache, not MIT. Attachment A also bars use by any entity over **US$20M** in revenue
or funding **except for personal use or research purposes**.

`scripts/card.py` writes a card that satisfies the repo-side conditions and copies the
License at the pinned revision. Whether *your* use fits the terms is your call.

## Install

```bash
uv sync
```

`torch` / `torchvision` are pinned to **2.11.0 / 0.26.0**: on Linux from the CUDA 12.8
index (needed for Blackwell, sm_120), on macOS from PyPI (Apple-silicon GPU via MPS).
Datalab's models are fetched at one pinned revision (`scripts/shipped.py`); set
`INDRA_MODEL_DIR` to point at a local copy instead.

## Your data

Each dataset is a folder holding `train.jsonl`, `validation.jsonl` and the images they
reference. Image paths are relative to that folder:

```
examples/line-boxes/
├── train.jsonl
├── validation.jsonl
└── images/
    ├── bsb00064015_00054.jpg
    └── bsb00046453_00369.jpg
```

Each line of a `.jsonl` file is one page in the CHAI-Document export shape. Only the
**assistant** turn is read — Indra takes no prompt, so the user turn's content does
not matter. The transcription has one line of text per written line, in one of two
forms. [`examples/`](examples/) has a working dataset of each, with provenance and rights.

### Format A — a box per line (`examples/line-boxes/`)

Each line starts with its box, `(x0,y0),(x1,y1)` normalised **0–1000** to the image. This
is the clean route ([step 3a](#3a--line-crops-from-line-boxes)). The first three lines
of the example, a public-domain BSB manuscript transcribed to the CATMuS standard:

```json
{"images": ["images/bsb00064015_00054.jpg"],
 "messages": [
   {"role": "user", "content": "<image>\nTranscribe this page, one line of text per written line."},
   {"role": "assistant", "content": "<|box_start|>(37,25),(903,81)<|box_end|>renuntiate mͥ utteego ueniam ado\n<|box_start|>(34,80),(999,135)<|box_end|>rẽ cũ. ł untib; aũ magis muia apparint\n<|box_start|>(34,128),(993,186)<|box_end|>stella. & quasi dicatũ p̃starit illis ante\n..."}
 ]}
```

(Pretty-printed here; in the file each page is a single line.) ALTO, PAGE-XML or any
other line geometry converts to this: divide each line's pixel box by the image width
and height and multiply by 1000.

### Format B — page text only (`examples/page-text/`)

Plain lines, no geometry. Training lines are recovered by aligning Indra's own
detected lines to yours ([step 3b](#3b--line-crops-from-page-text-only-alignment)), which
works but is noisier. The example is an 1853 US National Archives pension certificate:

```json
{"images": ["images/54013945_0128.jpg"],
 "messages": [
   {"role": "user", "content": "<image>\nTranscribe this page, one line of text per written line."},
   {"role": "assistant", "content": "{ State of North Carolina\n{ Comptroller's Office\nI, William J. Clarke, Comptroller of public accounts in and for the State aforesaid, do hereby certify ...\n..."}
 ]}
```

**Alignment only works as well as your transcription follows the page's lines.** That
third "line" is a whole paragraph spanning several written lines; it cannot pair with
any single detected line and is dropped. If you can, transcribe one line per line.

## Try it on the examples

Every step, end to end, in a few minutes on a laptop. Nothing here produces a useful
model — the examples are one or two pages each — but it proves your install works and
shows what each step prints.

```bash
uv sync

# 1. Benchmark the shipped model
uv run scripts/run_indra.py --dataset-dir examples/line-boxes --out work/eval --tag=-shipped
uv run scripts/score.py work/eval/predictions-shipped.jsonl

# 2. Prove the conversion on real pages (expect "identical decoded strings: N/N")
uv run scripts/convert_rec.py --test-jsonl examples/line-boxes/validation.jsonl --pages 1 --out work/rec_torch.pt

# 3a. Line crops from boxes (expect "100% cropped from a matched detector box")
uv run scripts/build_lines.py --dataset-dir examples/line-boxes --out work/lines

# 3b. ...or from page text, by alignment
uv run scripts/align_lines.py --dataset-dir examples/page-text --split train --out work/align --workers 2 --tag=-r1
uv run scripts/align_lines.py --dataset-dir examples/page-text --split validation --out work/align --workers 2 --tag=-r1
uv run scripts/build_aligned.py --pairs work/align --tag=-r1 --dataset-root examples/page-text --out work/aligned

# 4–6. Train, export into Datalab's graph, evaluate
uv run scripts/train_rec.py --lines work/lines --out work/run --epochs 3
uv run scripts/export_rec.py --state work/run/best_state.pt --vocab work/run/vocab.json --out work/model
uv run scripts/run_indra.py --dataset-dir examples/line-boxes --out work/eval --model-dir work/model --tag=-ft
uv run scripts/score.py work/eval/predictions-shipped.jsonl work/eval/predictions-ft.jsonl
```

Step 4 reports the vocabulary growing by three characters — combining tilde, `ͥ`, `ꝑ`.
Those are CATMuS signs the shipped recognizer cannot produce at all, and they are why
the classifier head is widened.

## Step by step

### 1 · Benchmark the shipped model

```bash
uv run scripts/run_indra.py --dataset-dir data/mine --out work/eval --tag=-shipped
uv run scripts/score.py work/eval/predictions-shipped.jsonl
```

`run_indra.py` keeps page headers and footers (`furniture=True`); Indra drops them by
default, and a full-page transcription includes them. Note `--tag=-shipped`, with
`=`: argparse reads a hyphen-leading value as a new flag.

### 2 · Convert the recognizer and prove it

```bash
uv run scripts/convert_rec.py --test-jsonl data/mine/validation.jsonl --out work/rec_torch.pt
```

Cuts `rec.onnx` at its logits (the graph has greedy decoding baked in), converts it
with onnx2torch, and **refuses to save** unless the PyTorch module reproduces the ONNX
graph's decoded text on every real line crop it tests. Expect
`identical decoded strings: N/N`. Training does not need this file — it rebuilds the
same module itself — but run it once to confirm your versions convert exactly.

### 3a · Line crops from line boxes

```bash
uv run scripts/build_lines.py --dataset-dir data/mine --out work/lines
```

The text comes from your boxes; the **geometry comes from Indra's own detector**. Each
reference line is matched to the detector box Indra would crop at inference
(IoU ≥ 0.5 and ≥ 80% of the line's width), falling back to the reference box resized
to the detector's typical height. Expect ~94% of lines matched, and a reported crop
height close to the detector's.

### 3b · Line crops from page text only (alignment)

```bash
uv run scripts/align_lines.py --dataset-dir data/mine --split train --out work/align --min-sim 0.3 --tag=-r1
uv run scripts/align_lines.py --dataset-dir data/mine --split validation --out work/align --min-sim 0.3 --tag=-r1
uv run scripts/build_aligned.py --pairs work/align --tag=-r1 --dataset-root data/mine --out work/lines
```

Indra detects and reads every line, then a monotonic alignment pairs a detected line
with a reference line only when its reading is similar enough. **`--min-sim 0.3` was
set by inspection**: between 0.3 and 0.6 the pairs are mostly correct even when Indra's
reading is badly wrong (`no reulan stsschorgur aThots` ↔ `no written discharges at that`);
below 0.3 correct readings start attaching to the wrong lines.

Alignment favours lines Indra can already partly read, so the model learns least about
what it reads worst. Run a second round: export the round-1 model (step 5), re-run
`align_lines.py --model-dir work/model-r1 --tag=-r2`, rebuild, retrain. Better readings
recover more lines.

### 4 · Train

```bash
uv run scripts/train_rec.py --lines work/lines --out work/run --epochs 6
```

- **Vocabulary is extended, not replaced.** Characters in your labels that Indra lacks
  are appended and the classifier head widened; Datalab's original classes are untouched.
  The shipped vocabulary omits combining marks by design.
- **New classes get a warm start** from the letter their Unicode name implies.
- **BatchNorm stays frozen**; validation is reported **before** training as the control.
- Uses CUDA, else Apple MPS, else CPU. ~40k lines train an epoch in ~1 minute on an
  Apple-silicon GPU. Saves `best_state.pt` (portable) and `vocab.json`.

### 5 · Export into Datalab's graph

```bash
uv run scripts/export_rec.py --state work/run/best_state.pt --vocab work/run/vocab.json --out work/model
```

Writes the trained weights into the **original** `rec.onnx` and the extended vocabulary
into `meta.json`, copies Datalab's other models unchanged, and checks the patched graph
against the PyTorch module at several line widths. The folder is a drop-in for
`indra_ocr.load(path=...)`.

### 6 · Evaluate end to end

```bash
uv run scripts/run_indra.py --dataset-dir data/mine --out work/eval --model-dir work/model --tag=-ft
uv run scripts/score.py work/eval/predictions-shipped.jsonl work/eval/predictions-ft.jsonl
```

### 7 · Publish

```bash
uv run scripts/card.py --model-dir work/model --run work/run \
    --base-pred work/eval/predictions-shipped.jsonl --tuned-pred work/eval/predictions-ft.jsonl \
    --repo you/your-indra-finetune --pages 1116 --language la --tags catmus,medieval-latin
hf upload you/your-indra-finetune work/model . --repo-type model
```

Read the License section above first.

## On a SLURM cluster

`cluster/` holds the jobs used on Yale's Bouchet cluster: `setup`, `verify`, `build`,
`train`. Set `INDRA_WORK` to a workspace (scratch is ideal), copy `scripts/` to
`$INDRA_WORK/code`, and submit from a folder containing `logs/`. Partition names are
Bouchet's; change them for yours. Build crops **where the images already live** — the
crops for ~33k pages are tens of GB, so nothing large should cross the network.

`verify.sbatch` checks the CUDA build on a CPU node by reading torch's
**compile-time** architecture flags; `torch.cuda.get_arch_list()` returns `[]` whenever
no GPU is visible and would wrongly fail every build there. The real runtime check
(device present, `sm_120` listed, a kernel launched) runs at the top of `train.sbatch`.

## Things that will bite you

- **CTC timesteps are `ceil(width / 4)`**, measured from the graph. Indra's source
  comment calls the 8-px width multiple the "timestep stride"; it is not, and using it
  halves every sequence length.
- **onnx2torch cannot convert one-sided `Clip` nodes**, and the recognizer has two.
  `convert_rec.patch_clips` rewrites them as `Min`/`Max`; the equivalence test covers it.
- **Do not re-export with `torch.onnx.export`.** Tracing froze the padding mask at the
  example input's width, breaking every other width. Patch weights into the original
  graph instead (`export_rec.py`).
- **onnx2torch renames modules**, turning `.` into `/` (`/rec/blocks.0/dw/Conv` →
  `rec/blocks/0/dw/Conv`). `export_rec.py` refuses to write a model if any trainable
  parameter goes unmapped — without that guard it would export half-updated weights
  that still run.
- **Pin the model revision.** Datalab's `main` can move; before/after comparisons
  across revisions measure the release, not your fine-tune.
- **The detector bounds the gain.** Recognizer fine-tuning cannot fix line
  segmentation; compare recognizer-on-ground-truth CER with end-to-end CER to see how
  much of your error is detection.

## Layout

```
scripts/
  shipped.py        pinned location of Datalab's models
  convert_rec.py    rec.onnx -> PyTorch, with the equivalence proof
  build_lines.py    crops from line-boxed transcriptions
  align_lines.py    pseudo-label lines from page-level transcriptions
  build_aligned.py  crops from aligned pairs
  train_rec.py      CTC fine-tune
  export_rec.py     weights back into Datalab's graph
  run_indra.py      run Datalab's pipeline over a split
  score.py          CER / WER / line recall (textmetrics.py)
  card.py           model card with the license terms built in
  textmetrics.py    the metric module from CHAI-Document-HPC, verbatim
cluster/            SLURM jobs (optional)
```
