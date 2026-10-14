"""Transcription metrics — the single implementation, used on both sides.

This file is synced to the cluster and imported by `evaluate.py` there; the local
app imports *this same path* through `app/metrics.py`. That is deliberate: an HPC
number and a local number are only comparable if they came from the same code,
and a second implementation is the easiest way to silently disagree about, say,
whether trailing whitespace counts.

Metrics reported per sample and aggregated two ways:

* **micro** — total edits / total reference length. This is the number to quote:
  it weights a page by how much text is on it, so one long page cannot be
  averaged away by ten short ones.
* **macro** — mean of per-sample rates. Reported alongside because a large gap
  between the two says the errors are concentrated in a few pages.

`*_norm` variants lowercase, fold whitespace, and drop punctuation before
scoring. For a graphemic standard like CATMuS the raw number is the real one;
the normalized number separates "read the wrong letters" from "made different
editorial choices about case and punctuation".

`*_nfd` variants put **both sides into Unicode NFD** and change nothing else.
This is not an editorial choice, it is the removal of a representation artefact:
`ũ` (U+0169) and `u` + U+0303 are the same glyph, and this corpus contains both —
77% of the training targets are internally mixed, and `ũ` alone appears 8,642
times composed against 14,843 decomposed. Scoring them as different characters
charges the model two edits (a substitution and a deletion) for a difference that
does not exist on the page. NFD rather than NFC because six of the twelve commonest
abbreviated glyphs here (`t̃ r̃ m̃ p̃ c̃ q̃`) have **no** precomposed form at all, so NFC
cannot make the corpus self-consistent and NFD can. Measured on the best run this
is ~16% of all edit operations.

`cer` itself is deliberately left alone: every number published in `results.md`
was produced under it, and silently redefining it would make the history
incomparable. `cer_nfd` is reported alongside and is the number to quote going
forward.
"""

from __future__ import annotations

import json
import re
import unicodedata

#: Sentinel the exporting app emits for a page with no text at all.
NO_TEXT = "VLM-NO-TEXT"

try:  # rapidfuzz's Levenshtein is C++; pure Python is ~1000x slower on pages.
    from rapidfuzz.distance import Levenshtein as _LEV

    def edit_distance(a, b) -> int:
        return _LEV.distance(a, b)

    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fallback path
    HAVE_RAPIDFUZZ = False

    def edit_distance(a, b) -> int:
        """Levenshtein distance with a rolling row. O(len(a) * len(b))."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_WS_ALL = re.compile(r"\s")


def normalize(text: str) -> str:
    """Case-folded, whitespace-collapsed, punctuation-stripped, NFKC form."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def canonical(text: str) -> str:
    """Unicode NFD, and nothing else.

    Lossless and reversible, so this is a change of spelling rather than a change
    of content: `canonical` never merges two characters that a palaeographer would
    distinguish. See the module docstring for why NFD and not NFC.
    """
    return unicodedata.normalize("NFD", text or "")


#: Allographs — positional or graphic variants of one letter. Folding these *is* an
#: editorial choice (CATMuS is graphemic and keeps them apart), so it never touches
#: `cer` or `cer_nfd`; `cer_fold` exists to say how much of the remaining error is
#: the model disagreeing about allograph policy rather than misreading the script.
_ALLOGRAPH = str.maketrans({
    "\u017f": "s",   # ſ  LATIN SMALL LETTER LONG S
    "\u0131": "i",   # ı  LATIN SMALL LETTER DOTLESS I
    "\u0237": "j",   # ȷ  LATIN SMALL LETTER DOTLESS J
    "\u1e9e": "\u00df",  # ẞ -> ß
    "\uf76c": "s",   # private-use long s seen in some exports
})


def graphemic(text: str) -> str:
    """NFD plus allograph folding. A diagnostic, never the headline number."""
    return canonical(text).translate(_ALLOGRAPH)


def spaceless(text: str) -> str:
    """NFD with **all** whitespace removed — line breaks included.

    Word division is the largest single error class here: whitespace is 16.7% of
    all edit operations, split almost evenly between insertions and deletions. Some
    of that is a genuine reading failure, but much of it is not a reading question
    at all — medieval Latin is written with inconsistent spacing, and a model that
    reads every letter correctly while disagreeing about where one word ends is
    penalised the same as one that misreads the letters.

    There is a specific and measurable version of this: the model inserts a space
    immediately before an abbreviation-marked character, mid-word, treating it as
    the start of a new word — `dotꝰ` → `do tͥ`, `inqs` → `in qͥs`, `deiͦ` → `cle iͦ`.
    Inserted spaces sit before a mark-carrying character 11.1% of the time against a
    5.0% base rate, an enrichment of 2.2x.

    So `cer_nows` isolates "did it read the right characters" from "did it agree
    about word boundaries". It is a diagnostic beside `cer_nfd`, not a replacement:
    for a line-by-line CATMuS target the spacing is part of the transcription, and
    a metric that discards it would also hide a model that ran every word together.
    Read the two together — a large gap means the letters are right and the
    divisions are not.
    """
    return _WS_ALL.sub("", canonical(text))


def lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def words(text: str) -> list[str]:
    return (text or "").split()


def is_no_text(text: str) -> bool:
    return (text or "").strip().upper() == NO_TEXT or not (text or "").strip()


def strip_thinking(text: str) -> str:
    """Drop a `<think>...</think>` block and any stray fences.

    Qwen3.5's chat template does not close the thinking block consistently across
    sizes (see the note in `render.py`), so a generation can arrive with an open
    or unbalanced block even when training never taught the model to think. A
    lenient strip here keeps a template mismatch from being scored as a
    transcription failure — the mismatch shows up in `thinking_leak` instead.
    """
    out = text or ""
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    elif out.lstrip().startswith("<think>"):
        # Open block, never closed: the whole generation is reasoning.
        out = ""
    out = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", out.strip())
    return out.strip()


# --------------------------------------------------------------------------
# Joint layout + HTR targets (opt in; nothing below fires on a text-only page)
# --------------------------------------------------------------------------
#
# A joint target prefixes every transcription line with that line's bounding
# box, using the tokenizer's OWN grounding tokens rather than an ad-hoc syntax:
#
#     <|box_start|>(45,17),(955,61)<|box_end|>rbTame̾ nchi lomimꝰ tam ...
#
# `<|box_start|>` / `<|box_end|>` are single tokens in the Qwen3.5 vocabulary
# (248049 / 248050), so the base model already associates them with grounding
# from pretraining and each box costs ~18 tokens rather than a dozen characters
# the model must learn to punctuate.
#
# COORDINATES ARE NORMALISED 0-1000 INTEGERS, in the frame of the CROPPED image.
# Not pixels: the processor's smart-resize changes the pixel dimensions the model
# actually sees (a page is reduced to fit the visual-token budget), so a pixel
# target would be systematically wrong at inference by whatever factor the resize
# applied. Normalised coordinates are invariant to it.
#
# The box is a PREFIX rather than a suffix or a separate block so that stripping
# boxes recovers the text-only target exactly — which is what keeps a joint run's
# CER directly comparable with every text-only number already published.

BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
BOX_SCALE = 1000

_BOX_RE = re.compile(
    re.escape(BOX_START) + r"\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*"
    r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*" + re.escape(BOX_END))

#: The same box with its markers missing. Needed because a decoder that calls
#: `skip_special_tokens=True` deletes `<|box_start|>`/`<|box_end|>` — they ARE
#: special tokens — and leaves the coordinates behind as ordinary text. The model
#: emitted a perfectly good box; the tokenizer removed the punctuation around it.
#:
#: Scoring the marker-less form is not a leniency, it is the difference between
#: reading the output and mis-reading it: without this, the coordinates count as
#: transcription characters (CER 0.4953 against a true 0.13), every line carries a
#: numeric prefix so `line_recall` is exactly 0, and `box_coverage` is 0 for a
#: model localising to within three normalised units.
#:
#: **Anchored at the start of the line**, deliberately. An unanchored pattern
#: would find "(1,2),(3,4)" anywhere in a transcription and silently treat real
#: text as geometry; a box is only ever a line prefix here.
_BARE_BOX_RE = re.compile(
    r"^\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*"
    r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def has_boxes(text: str) -> bool:
    """Whether this string carries line boxes, in either spelling.

    Checked line by line for the bare form because it is anchored: a prediction
    whose markers were eaten by `skip_special_tokens` has no `BOX_START` at all.
    """
    if BOX_START in (text or ""):
        return True
    return any(_BARE_BOX_RE.match(line) for line in (text or "").split("\n"))


def format_box(x0: float, y0: float, x1: float, y1: float) -> str:
    """One box marker. The single place the on-the-wire format is written.

    Both the target builder and the scorer go through this and `parse_boxes`, so
    the thing that writes a target and the thing that reads a prediction cannot
    disagree about the syntax — the same reason `render.py` owns the prompt.
    """
    clamp = lambda v: max(0, min(BOX_SCALE, int(round(v))))
    return f"{BOX_START}({clamp(x0)},{clamp(y0)}),({clamp(x1)},{clamp(y1)}){BOX_END}"


def parse_boxes(text: str) -> tuple[list[tuple[int, int, int, int] | None], list[str]]:
    """(boxes, texts) per line. A line with no parseable box yields None.

    None rather than a zero box: "the model emitted no box here" and "the model
    emitted a degenerate box at the origin" are different failures, and averaging
    a zero IoU over the first would hide that the model simply stopped localising.
    """
    boxes: list[tuple[int, int, int, int] | None] = []
    texts: list[str] = []
    for line in (text or "").split("\n"):
        m = _BOX_RE.search(line)
        if m:
            boxes.append(tuple(int(g) for g in m.groups()))  # type: ignore[arg-type]
            texts.append(_BOX_RE.sub("", line, count=1).strip())
            continue
        m = _BARE_BOX_RE.match(line)
        if m:
            boxes.append(tuple(int(g) for g in m.groups()))  # type: ignore[arg-type]
            texts.append(line[m.end():].strip())
            continue
        boxes.append(None)
        texts.append(line.strip())
    return boxes, texts


def strip_boxes(text: str) -> str:
    """The text-only view of a possibly box-laden target.

    A no-op on any string without `BOX_START`, which is every target this project
    has published — so routing all scoring through it is provably additive. That
    is verified rather than asserted: `app/rescore.py` re-scores every stored
    prediction and refuses to write if a single pre-existing metric moves.
    """
    if not has_boxes(text):
        return text
    return "\n".join(parse_boxes(text)[1])


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_scores(reference: str, hypothesis: str) -> dict:
    """Localisation quality, matched by READING ORDER — line i against line i.

    Reading-order matching, not nearest-box matching, and deliberately so. The
    source line boxes overlap heavily — measured on this corpus, a line's box is
    a median **1.63x the line pitch**, because the annotation is generous about
    ascenders and descenders — so a spatial matcher can confidently pair a
    predicted box with the neighbouring line and report a good IoU for output
    that has slipped by one line. Reading order is unambiguous because both
    sequences are ordered by construction, and a model that drops or invents a
    line is *supposed* to be penalised for the misalignment that follows.

    Pages where the reference has no boxes return {} so a mixed split does not
    average localisation over pages that never asked for it.
    """
    ref_boxes, _ = parse_boxes(reference)
    if not any(b is not None for b in ref_boxes):
        return {}
    hyp_boxes, _ = parse_boxes(hypothesis)
    ious: list[float] = []
    emitted = 0
    for i, rb in enumerate(ref_boxes):
        if rb is None:
            continue
        hb = hyp_boxes[i] if i < len(hyp_boxes) else None
        if hb is None:
            ious.append(0.0)
            continue
        emitted += 1
        ious.append(_iou(rb, hb))
    n_ref = sum(1 for b in ref_boxes if b is not None)
    return {
        "box_ref": n_ref,
        "box_emitted": emitted,
        "box_coverage": _rate(emitted, n_ref, default=0.0),
        "box_iou": _mean(ious),
        # Numerators, so `aggregate` can micro-average over a split by summing
        # rather than averaging per-page means — a 60-line page and a 12-line
        # page do not carry the same weight in a micro average.
        "box_iou_sum": sum(ious),
        "box_iou50_n": sum(1 for v in ious if v >= 0.5),
        "box_iou75_n": sum(1 for v in ious if v >= 0.75),
        # The share of lines localised well enough to be usable. 0.5 is the
        # conventional detection threshold; 0.75 is reported beside it because at
        # 1.63x pitch a 0.5 IoU can still be most of a neighbouring line.
        "box_iou50": _rate(sum(1 for v in ious if v >= 0.5), n_ref, default=0.0),
        "box_iou75": _rate(sum(1 for v in ious if v >= 0.75), n_ref, default=0.0),
    }


def score_pair(reference: str, hypothesis: str) -> dict:
    """Per-sample metrics for one (reference, hypothesis) transcription pair.

    A joint layout+HTR target carries a box marker per line. Those are stripped
    before every text metric and scored separately, so a joint run's `cer_nfd` is
    the CER of its transcription and stays directly comparable with the
    text-only numbers in `results.md` — rather than partly measuring how well the
    model punctuates coordinates. `strip_boxes` is a no-op on a page without
    boxes, so this cannot move an existing number; `app/rescore.py` proves it.
    """
    boxes = box_scores(reference or "", hypothesis or "")
    reference, hypothesis = strip_boxes(reference or ""), strip_boxes(hypothesis or "")
    ref, hyp = (reference or "").strip(), (hypothesis or "").strip()
    ref_n, hyp_n = normalize(ref), normalize(hyp)
    ref_w, hyp_w = words(ref), words(hyp)
    ref_l, hyp_l = lines(ref), lines(hyp)
    # Canonical (NFD) and allograph-folded views. Both sides get the same
    # treatment, so these can only remove a difference that is not on the page.
    ref_c, hyp_c = canonical(ref), canonical(hyp)
    ref_f, hyp_f = graphemic(ref), graphemic(hyp)
    ref_s, hyp_s = spaceless(ref), spaceless(hyp)

    char_edits = edit_distance(ref, hyp)
    word_edits = edit_distance(ref_w, hyp_w)
    norm_edits = edit_distance(ref_n, hyp_n)
    nfd_edits = edit_distance(ref_c, hyp_c)
    fold_edits = edit_distance(ref_f, hyp_f)
    nows_edits = edit_distance(ref_s, hyp_s)
    nfd_word_edits = edit_distance(words(ref_c), words(hyp_c))

    matched = _lcs_length(ref_l, hyp_l)
    # Exact-line matching is *more* sensitive to the encoding split than CER is:
    # one differently-composed tilde fails a whole 41-character line.
    matched_nfd = _lcs_length([canonical(x) for x in ref_l],
                              [canonical(x) for x in hyp_l])
    return {
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
        "ref_words": len(ref_w),
        "ref_lines": len(ref_l),
        "hyp_lines": len(hyp_l),
        "char_edits": char_edits,
        "word_edits": word_edits,
        "norm_edits": norm_edits,
        "norm_chars": len(ref_n),
        "nfd_edits": nfd_edits,
        "nfd_chars": len(ref_c),
        "nfd_word_edits": nfd_word_edits,
        "nfd_ref_words": len(words(ref_c)),
        "fold_edits": fold_edits,
        "fold_chars": len(ref_f),
        "nows_edits": nows_edits,
        "nows_chars": len(ref_s),
        "cer": _rate(char_edits, len(ref)),
        "wer": _rate(word_edits, len(ref_w)),
        "cer_norm": _rate(norm_edits, len(ref_n)),
        "cer_nfd": _rate(nfd_edits, len(ref_c)),
        "wer_nfd": _rate(nfd_word_edits, len(words(ref_c))),
        "cer_fold": _rate(fold_edits, len(ref_f)),
        "cer_nows": _rate(nows_edits, len(ref_s)),
        "lines_exact_nfd": matched_nfd,
        "line_recall_nfd": _rate(matched_nfd, len(ref_l), default=1.0),
        # Lines the model reproduced exactly, in order. Blunt but the thing a
        # palaeographer actually looks at: did it get the line breaks right.
        "lines_exact": matched,
        "line_recall": _rate(matched, len(ref_l), default=1.0),
        **boxes,
        "line_delta": len(hyp_l) - len(ref_l),
        "exact": ref == hyp,
        "empty_hyp": not hyp,
        # Degeneracy diagnostics. A small model under greedy decoding can fall into
        # a repetition loop and fill the whole token budget — observed emitting
        # 4,121 characters against a 1,615-character page, most of it ". . . . .".
        # That is a *decoding* failure, not a reading failure, and it moves micro
        # CER a long way: without separating it you conclude the model cannot read
        # when on most pages it can. Reported alongside, never folded into CER.
        "max_repeat_run": _max_repeat_run(hyp_w),
        "length_ratio": (len(hyp) / len(ref)) if ref else (0.0 if not hyp else 999.0),
    }


def _max_repeat_run(tokens: list[str]) -> int:
    """Longest run of one repeated token."""
    best = run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        best = max(best, run)
    return best


def is_degenerate(scores: dict) -> bool:
    """A repetition loop rather than a bad reading.

    Both conditions together: the output is meaningfully longer than the reference
    *and* contains a long run of one repeated token. Either alone has honest causes
    — a verbose but valid transcription, or a page that really does repeat a word.
    """
    return scores.get("length_ratio", 0) > 1.3 and scores.get("max_repeat_run", 0) >= 10


def _rate(numerator: int, denominator: int, default: float = 0.0) -> float:
    if denominator == 0:
        # No reference text: any output at all is pure insertion. Report 1.0 so
        # a hallucinated page is not silently scored as perfect.
        return default if numerator == 0 else 1.0
    return numerator / denominator


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two line lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def score_json_pair(reference: str, hypothesis: str) -> dict:
    """Field-level metrics for layout (JSON) targets.

    Kept deliberately simple: JSON validity, region-count agreement, and F1 over
    flattened `key=value` leaves. A layout sweep would want IoU on the boxes;
    this prototype's datasets are transcription, so this exists so a mixed corpus
    still produces a number instead of an exception.
    """
    try:
        hyp_obj = json.loads(strip_thinking(hypothesis))
        valid = True
    except (json.JSONDecodeError, TypeError):
        hyp_obj, valid = None, False
    try:
        ref_obj = json.loads(reference)
    except (json.JSONDecodeError, TypeError):
        ref_obj = None

    ref_leaves = set(_flatten(ref_obj)) if ref_obj is not None else set()
    hyp_leaves = set(_flatten(hyp_obj)) if hyp_obj is not None else set()
    tp = len(ref_leaves & hyp_leaves)
    precision = tp / len(hyp_leaves) if hyp_leaves else 0.0
    recall = tp / len(ref_leaves) if ref_leaves else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "json_valid": valid,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "ref_regions": len((ref_obj or {}).get("regions") or []) if ref_obj else 0,
        "hyp_regions": len((hyp_obj or {}).get("regions") or []) if hyp_obj else 0,
    }


def _flatten(obj, prefix: str = "") -> list[str]:
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out.extend(_flatten(v, f"{prefix}{k}."))
        return out
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out.extend(_flatten(v, f"{prefix}{i}."))
        return out
    return [f"{prefix}={obj}"]


def aggregate(samples: list[dict]) -> dict:
    """Roll per-sample scores up into the numbers the results table shows."""
    text = [s for s in samples if s.get("target_kind") == "text" and not s.get("no_text")]
    no_text = [s for s in samples if s.get("no_text")]
    layout = [s for s in samples if s.get("target_kind") == "layout"]

    out: dict = {"n": len(samples), "n_text": len(text), "n_no_text": len(no_text),
                 "n_layout": len(layout)}

    if text:
        totals = {k: sum(s["scores"].get(k, 0) for s in text) for k in
                  ("char_edits", "ref_chars", "word_edits", "ref_words",
                   "norm_edits", "norm_chars", "lines_exact", "ref_lines",
                   "nfd_edits", "nfd_chars", "nfd_word_edits", "nfd_ref_words",
                   "fold_edits", "fold_chars", "lines_exact_nfd",
                   "nows_edits", "nows_chars")}
        out.update({
            "cer": _rate(totals["char_edits"], totals["ref_chars"]),
            "wer": _rate(totals["word_edits"], totals["ref_words"]),
            "cer_norm": _rate(totals["norm_edits"], totals["norm_chars"]),
            "line_recall": _rate(totals["lines_exact"], totals["ref_lines"], default=1.0),
            # Same scoring with both sides in Unicode NFD. Not an editorial
            # relaxation: it removes a representation split that is present in the
            # reference data itself. See the module docstring.
            "cer_nfd": _rate(totals["nfd_edits"], totals["nfd_chars"]),
            "wer_nfd": _rate(totals["nfd_word_edits"], totals["nfd_ref_words"]),
            "cer_fold": _rate(totals["fold_edits"], totals["fold_chars"]),
            # NFD with all whitespace removed: reading accuracy separated from
            # agreement about word division. See `spaceless`.
            "cer_nows": _rate(totals["nows_edits"], totals["nows_chars"]),
            "line_recall_nfd": _rate(totals["lines_exact_nfd"], totals["ref_lines"],
                                     default=1.0),
            "cer_nfd_macro": _mean(s["scores"].get("cer_nfd", 0.0) for s in text),
            "cer_macro": _mean(s["scores"]["cer"] for s in text),
            "wer_macro": _mean(s["scores"]["wer"] for s in text),
            "exact_pages": sum(1 for s in text if s["scores"]["exact"]) / len(text),
            "empty_outputs": sum(1 for s in text if s["scores"]["empty_hyp"]) / len(text),
            "ref_chars": totals["ref_chars"],
            # The raw numerators and denominators, so a *subset* of an eval split
            # (one corpus, the human-corrected corpora) can be micro-averaged
            # exactly by summing them. Recombining published rates instead needs
            # a division that has already lost the counts.
            "totals": dict(totals),
            "degenerate": sum(1 for s in text if is_degenerate(s["scores"])) / len(text),
            # The same micro CER over the pages that did *not* collapse into a
            # repetition loop. The gap between this and `cer` is the share of the
            # error that a decoding fix could remove without the model learning
            # anything new.
            "cer_clean": _clean_cer(text),
        })
    if no_text:
        # Did the model correctly say "there is nothing here" rather than invent
        # a transcription? The failure mode that matters on binding shots.
        out["no_text_accuracy"] = sum(
            1 for s in no_text if is_no_text(s.get("prediction", ""))
        ) / len(no_text)
    if layout:
        out.update({
            "json_valid": _mean(s["scores"].get("json_valid", 0) for s in layout),
            "field_f1": _mean(s["scores"].get("field_f1", 0.0) for s in layout),
        })
    boxed = [s for s in text if s["scores"].get("box_ref")]
    if boxed:
        # Micro-averaged over lines, like CER is over characters: a page with 60
        # lines should weigh five times a page with 12. `n_box_pages` is reported
        # because a joint model that stops emitting boxes on hard pages would
        # otherwise show a flattering IoU over the easy ones it still tried.
        n_ref = sum(s["scores"]["box_ref"] for s in boxed)
        out.update({
            "n_box_pages": len(boxed),
            "box_coverage": _rate(sum(s["scores"]["box_emitted"] for s in boxed), n_ref),
            "box_iou": _rate(sum(s["scores"]["box_iou_sum"] for s in boxed), n_ref),
            "box_iou50": _rate(sum(s["scores"]["box_iou50_n"] for s in boxed), n_ref),
            "box_iou75": _rate(sum(s["scores"]["box_iou75_n"] for s in boxed), n_ref),
            "box_iou_macro": _mean(s["scores"]["box_iou"] for s in boxed),
            "box_ref_total": n_ref,
        })
    leaked = sum(1 for s in samples if s.get("thinking_leak"))
    if leaked:
        out["thinking_leak"] = leaked / len(samples)
    truncated = sum(1 for s in samples if s.get("truncated"))
    if truncated:
        out["truncated"] = truncated / len(samples)
    return out


def _clean_cer(text: list[dict]) -> float:
    kept = [s for s in text if not is_degenerate(s["scores"])]
    if not kept:
        return 1.0
    return _rate(sum(s["scores"]["char_edits"] for s in kept),
                 sum(s["scores"]["ref_chars"] for s in kept))


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def score_samples(samples: list[dict]) -> dict:
    """Score a list of `{target, prediction, target_kind}` rows end to end.

    Mutates each row with `scores` / `no_text` / `thinking_leak` and returns the
    aggregate. Both `evaluate.py` (cluster) and the local endpoint runner call
    exactly this, so an HPC row and a local row are directly comparable.
    """
    for s in samples:
        raw = s.get("prediction_raw", s.get("prediction", "")) or ""
        prediction = strip_thinking(raw)
        s["prediction"] = prediction
        s["thinking_leak"] = "<think>" in raw or "</think>" in raw
        target = s.get("target", "") or ""
        kind = s.get("target_kind") or ("layout" if target.lstrip().startswith("{") else "text")
        s["target_kind"] = kind
        s["no_text"] = is_no_text(target)
        if kind == "layout":
            s["scores"] = score_json_pair(target, prediction)
        else:
            s["scores"] = score_pair(target, prediction)
    return aggregate(samples)
