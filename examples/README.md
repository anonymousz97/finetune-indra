# Example data

Two tiny datasets, one per input format. They exist to show the format and to let
you run every step end to end in a few minutes. They are far too small to train a
useful model.

| folder | format | pages | route |
|---|---|---|---|
| `line-boxes/` | transcription with a box per line | 1 train, 1 validation | step 3a, `build_lines.py` |
| `page-text/` | transcription only | 2 train, 1 validation | step 3b, `align_lines.py` + `build_aligned.py` |

Only the **assistant** turn is read. The user turn is there because CHAI-Document
exports carry one; Indra takes no prompt, so its content does not matter.

## Provenance and rights

The images are reproduced here because each one is in the public domain. The
transcriptions come from the datasets named below.

### `line-boxes/` — Bayerische Staatsbibliothek, Munich

| file | manuscript | IIIF manifest | rights |
|---|---|---|---|
| `bsb00064015_00054.jpg` | BSB **Clm 19105** — Chromatius of Aquileia, *Chromatii et Heliodori epistula ad Hieronymum et eius responsiva* | [bsb00064015](https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00064015/manifest) | [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/) |
| `bsb00046453_00369.jpg` | BSB **Clm 14452** — *Voll-Lektionar* | [bsb00046453](https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00046453/manifest) | [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/) |

Images: München, Bayerische Staatsbibliothek. The rights statement for each was read
from its own IIIF manifest. Each image is the main text block of the page, requested
as a IIIF region and delivered at `!1500,1500`.

Transcriptions and line geometry: [`comma-project/deep-jsonl`](https://huggingface.co/datasets/comma-project/deep-jsonl),
CATMuS graphemic standard. Each line's box is `(x0,y0),(x1,y1)` normalised **0–1000** to
the image, derived from the source ALTO.

### `page-text/` — United States National Archives

Revolutionary War pension files, via [`wjbmattingly/si-test`](https://huggingface.co/datasets/wjbmattingly/si-test):

| file | document | date |
|---|---|---|
| `54013945_0128.jpg` | North Carolina Comptroller's Office, certificate of Revolutionary War payments to John Bostic | 1853 |
| `111763377_0033.jpg` | War Department, printed "Revolutionary Claim" pension certificate for Jesse Wood, with handwritten entries | 1845 |
| `111768895_0005.jpg` | Widow's declaration of Elizabeth Woodard, sworn before the Niagara County judge | after 1832 |

All three were written in the 19th century and are in the public domain by age; two
are government records. They were chosen for that reason. Pension files also hold
later private letters (some from the 1930s), which may still be in copyright, so
check dates before adding NARA pages of your own to a public repository.

**These transcriptions are not strictly line-by-line.** On `54013945_0128.jpg` a whole
paragraph is one transcription "line" that spans several written lines. Alignment can
only pair a reference line with a single detected line, so lines like that are dropped.
It is a large part of why only ~29% of NARA reference lines pair. Alignment works as
well as your transcriptions follow the page's actual lines.
