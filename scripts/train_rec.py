"""Fine-tune Indra's line recognizer with CTC on line crops from build_lines.py.

    uv run scripts/train_rec.py --lines work/lines --out work/run-comma

By default it starts by REBUILDING the recognizer from Datalab's pinned rec.onnx
(`--init shipped`, via convert_rec.rec_module) -- the conversion proven to
reproduce their ONNX graph exactly (452/452 decoded lines). Rebuilding instead of
loading a pickle keeps a torch-version mismatch between machines out of the
weights; checkpoints are saved as portable state_dicts. What it decides, and why:

* **Vocabulary is extended, not replaced.** The shipped recognizer has 1,677
  tokens and omits combining marks by design; 59% of CATMuS training lines contain
  a character outside it (combining tilde, vertical tilde, p-with-stroke, con,
  tironian et, superscript letters...). Every out-of-vocabulary character in the
  TRAINING labels is appended, and the 1x1 classifier `rec.head` is widened to
  match. The original 1,678 rows are kept intact, so old characters start exactly
  where Datalab left them.
* **New classes get a warm start.** Each new class's weights are copied from the
  letter its Unicode name implies -- "LATIN SMALL LETTER P WITH STROKE ..." from
  `p`, "COMBINING LATIN SMALL LETTER I" from `i`, combining tildes from `~` -- and
  from the mean row when no base letter applies. Their bias starts at the mean
  bias, so no new class is favoured before it has been trained.
* **The single BatchNorm stays frozen** (eval mode). Updating running statistics
  from small width-bucketed batches is a standard way to destabilise a fine-tune.
* **CTC input lengths are `ceil(width / 4)`** -- MEASURED from the original graph's
  `lengths` output on nine widths. The source comment calls the 8-px width
  multiple the "timestep stride"; it is not, and using it would halve every
  sequence length the loss sees.
* **Validation is on ground-truth line crops** from the 243-page holdout, scored
  as CER after NFD normalisation (the project's headline metric), with the
  zero-shot recognizer scored the same way BEFORE training as the control.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from indra_ocr.rec_preproc import width_batches
from indra_ocr.vocab import Vocab

from shipped import MODEL_DIR  # noqa: E402  (pinned revision; see shipped.py)


def load_lines(d: Path, split: str):
    z = np.load(d / f"{split}.npz")
    labels = json.loads((d / f"{split}-labels.json").read_text())
    widths = z["widths"].astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(widths * 32)])
    flat = z["flat"]
    crops = [flat[offs[i]:offs[i + 1]].reshape(32, widths[i]) for i in range(len(widths))]
    return crops, labels, widths


def base_letter(ch: str, vocab_index: dict) -> str | None:
    """The existing token a new character should borrow its initial weights from."""
    name = unicodedata.name(ch, "")
    if "TILDE" in name and "~" in vocab_index:
        return "~"
    m = re.search(r"LETTER ([A-Z])\b", name)
    if m:
        c = m.group(1).lower() if "SMALL" in name or "COMBINING" in name else m.group(1)
        if c in vocab_index:
            return c
    return None


def widen_head(model, n_old: int, new_chars: list[str], vocab_index: dict):
    head_name, head = next((n, m) for n, m in model.named_modules()
                           if isinstance(m, torch.nn.Conv1d) and m.out_channels == n_old)
    new = torch.nn.Conv1d(head.in_channels, n_old + len(new_chars), 1)
    with torch.no_grad():
        new.weight[:n_old] = head.weight
        new.bias[:n_old] = head.bias
        mean_w, mean_b = head.weight.mean(0), head.bias.mean()
        sources = []
        for k, ch in enumerate(new_chars):
            src = base_letter(ch, vocab_index)
            new.weight[n_old + k] = head.weight[vocab_index[src]] if src else mean_w
            new.bias[n_old + k] = mean_b
            sources.append(src or "mean")
    parent = model
    *path, leaf = head_name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)
    return head_name, sources


def ctc_ok(ids: list[int], t: int) -> bool:
    repeats = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return len(ids) + repeats <= t


@torch.no_grad()
def evaluate(model, crops, labels, widths, vocab, device, batch=64) -> dict:
    model.eval()
    errs = chars = exact = 0
    for idx in width_batches(widths.tolist(), batch):
        x, w = pad(crops, idx, widths)
        logits = model(x.to(device), w.to(device)).float().cpu()
        best = logits.argmax(-1).numpy()
        for k, j in enumerate(idx):
            n = math.ceil(int(widths[j]) / 4)
            seq = best[k, :n]
            keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
            hyp = unicodedata.normalize("NFD", vocab.decode(seq[keep].tolist()))
            ref = unicodedata.normalize("NFD", labels[j])
            errs += Levenshtein.distance(hyp, ref)
            chars += len(ref)
            exact += hyp == ref
    freeze_bn(model)
    return {"cer_nfd": errs / max(1, chars), "line_exact": exact / max(1, len(labels))}


def pad(crops, idx, widths):
    wm = int(math.ceil(max(int(widths[j]) for j in idx) / 8) * 8)
    x = np.zeros((len(idx), 1, 32, wm), np.float32)
    for k, j in enumerate(idx):
        x[k, 0, :, : widths[j]] = crops[j] / 255.0
    return torch.from_numpy(x), torch.from_numpy(widths[idx].astype(np.int64))


def freeze_bn(model):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lines", type=Path, required=True)
    ap.add_argument("--init", default="shipped",
                    help="'shipped' or an .onnx: rebuild from ONNX (portable); a .pt: torch.load")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

    tr_c, tr_l, tr_w = load_lines(args.lines, "train")
    va_c, va_l, va_w = load_lines(args.lines, "validation")
    base_chars = json.loads(json.loads((MODEL_DIR / "meta.json").read_text())["vocab"])["chars"]
    base = Vocab(base_chars)
    new_chars = sorted({c for l in tr_l for c in base.oov(l)})
    vocab = Vocab(base_chars + new_chars)
    if args.init == "shipped" or str(args.init).endswith(".onnx"):
        from convert_rec import rec_module
        src = MODEL_DIR / "rec.onnx" if args.init == "shipped" else Path(args.init)
        model = rec_module(src, args.out / "_convert")
    else:
        model = torch.load(args.init, weights_only=False)
    model = model.to(device)

    # Control first: the UNMODIFIED recognizer on the same ground-truth crops.
    zero = evaluate(model, va_c, va_l, va_w, base, device)
    print(f"zero-shot recognizer on {len(va_l):,} GT holdout lines: "
          f"CER_nfd {zero['cer_nfd']:.4f}  exact lines {zero['line_exact']:.1%}", flush=True)

    head_name, sources = widen_head(model.cpu(), base.n_classes, new_chars, base.index)
    model = model.to(device)
    print(f"vocab {base.n_classes - 1} -> {vocab.n_classes - 1} tokens (+{len(new_chars)}); "
          f"widened {head_name}; warm starts: "
          + ", ".join(f"{c!r}<-{s}" for c, s in list(zip(new_chars, sources))[:14]) + " ...", flush=True)

    enc = [vocab.encode(l) for l in tr_l]
    keep = [i for i, e in enumerate(enc) if e and ctc_ok(e, math.ceil(int(tr_w[i]) / 4))]
    va_oov = sum(1 for l in va_l if vocab.oov(l))
    print(f"train lines usable: {len(keep):,}/{len(tr_l):,} "
          f"(dropped {len(tr_l) - len(keep)}: unencodable or too long for CTC) | "
          f"holdout lines with a char unseen in training: {va_oov}", flush=True)

    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    batches_per_epoch = len(width_batches([int(tr_w[i]) for i in keep], args.batch))
    total = args.epochs * batches_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    history, best = [], (zero["cer_nfd"], -1)
    for epoch in range(args.epochs):
        freeze_bn(model)
        batches = width_batches([int(tr_w[i]) for i in keep], args.batch)
        random.shuffle(batches)
        t0, losses = time.time(), []
        for b in batches:
            idx = [keep[i] for i in b]
            x, w = pad(tr_c, idx, tr_w)
            logp = model(x.to(device), w.to(device)).float().log_softmax(-1).transpose(0, 1).cpu()
            tgt = [enc[j] for j in idx]
            loss = ctc(logp, torch.tensor([t for s in tgt for t in s]),
                       torch.tensor([math.ceil(int(tr_w[j]) / 4) for j in idx]),
                       torch.tensor([len(s) for s in tgt]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        ev = evaluate(model, va_c, va_l, va_w, vocab, device)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **ev,
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch + 1:2d}  loss {np.mean(losses):.4f}  holdout CER_nfd {ev['cer_nfd']:.4f}  "
              f"exact lines {ev['line_exact']:.1%}  ({time.time() - t0:.0f}s)", flush=True)
        if ev["cer_nfd"] < best[0]:
            best = (ev["cer_nfd"], epoch + 1)
            # state_dict is the portable artifact (no torch-version-bound pickle);
            # best.pt is kept for local convenience only.
            torch.save(model.cpu().state_dict(), args.out / "best_state.pt")
            torch.save(model, args.out / "best.pt"); model.to(device)
    (args.out / "vocab.json").write_text(json.dumps({"blank": 0, "chars": vocab.chars}, ensure_ascii=False))
    (args.out / "history.json").write_text(json.dumps(
        {"zero_shot": zero, "epochs": history, "best_epoch": best[1], "new_chars": new_chars,
         "warm_starts": dict(zip(new_chars, sources)), "lines_used": len(keep)}, indent=1, ensure_ascii=False))
    print(f"best holdout CER_nfd {best[0]:.4f} at epoch {best[1]} (zero-shot {zero['cer_nfd']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
