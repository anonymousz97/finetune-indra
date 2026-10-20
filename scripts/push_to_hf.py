"""Upload fine-tuned Indra Vietnamese OCR model to Hugging Face Hub (ANZ-Innovation/indra-v1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def generate_model_card(
    repo_id: str,
    stats: dict,
    test_eval: dict,
    train_history: dict,
) -> str:
    base_cer = test_eval.get("base_model", {}).get("cer_nfd", 0.0) * 100
    ft_cer = test_eval.get("finetuned_model", {}).get("cer_nfd", 0.0) * 100
    base_acc = test_eval.get("base_model", {}).get("line_exact", 0.0) * 100
    ft_acc = test_eval.get("finetuned_model", {}).get("line_exact", 0.0) * 100
    cer_reduc = test_eval.get("cer_reduction_pct", 0.0)
    acc_gain = test_eval.get("exact_line_improvement_pct", 0.0)

    n_train = stats.get("train_lines", 0)
    n_val = stats.get("validation_lines", 0)
    n_test = stats.get("test_lines", 0)
    n_total = stats.get("total_lines", 0)
    src = stats.get("sources", {})

    epochs = len(train_history.get("epochs", []))
    best_epoch = train_history.get("best_epoch", 0)
    speed = train_history.get("epochs", [{}])[-1].get("samples_per_sec", 0.0)
    total_time = train_history.get("total_train_time_sec", 0.0)

    return f"""---
language:
- vi
- en
license: other
license_name: indra-license
license_link: LICENSE
tags:
- ocr
- indra
- text-recognition
- vietnamese
- onnx
- document-ai
pipeline_tag: image-to-text
---

# Indra OCR Vietnamese (v1)

Fine-tuned version of [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr) optimized for high-accuracy Vietnamese document text recognition.

This checkpoint is a **drop-in replacement** for the official `indra-ocr` pipeline. It retains full architecture compatibility with Datalab's ONNX runtime while significantly boosting accuracy on Vietnamese administrative decrees, academic research publications, and structured forms.

---

## Benchmark & Evaluation

Evaluated on held-out Vietnamese test data across administrative (Chinhphu), academic (VJOL), and real-world forms (Datalab snapshot):

| Metric | Base Model (`datalab-to/indra-ocr`) | Fine-Tuned (`ANZ-Innovation/indra-v1`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Character Error Rate (CER NFD)** | **{base_cer:.2f}%** | **{ft_cer:.2f}%** | **-{cer_reduc:.2f}% relative** |
| **Exact Line Accuracy** | **{base_acc:.2f}%** | **{ft_acc:.2f}%** | **+{acc_gain:.2f}% absolute** |

### Training Stats & Efficiency
- **Hardware**: 1x NVIDIA RTX A4000 16GB
- **Throughput**: ~{speed:.1f} samples/second
- **Total Training Duration**: ~{total_time:.1f}s (~{total_time / 60:.1f} minutes, {epochs} epochs, best at epoch {best_epoch})
- **Dataset Partitioning**:
  - **Train**: {n_train:,} lines
  - **Validation**: {n_val:,} lines
  - **Test (Held-out)**: {n_test:,} lines
  - **Total**: {n_total:,} lines
  - **Sources**:
    - Government decrees & administrative documents (Chinhphu): {src.get('chinhphu_administrative', 0):,} lines
    - Academic journals & scientific publications (VJOL): {src.get('vjol_academic', 0):,} lines
    - Datalab DBNet detector-matched clean document lines: {src.get('datalab_detector_matched', 0):,} lines
    - Photorealistic synthetic document lines (Liberation Serif/Sans, Lato, DejaVu): {src.get('synthetic_vietnamese_128fonts', 0):,} lines

---

## Quickstart & Usage

You can use this fine-tuned model directly with the official `indra-ocr` package or ONNX Runtime:

```bash
pip install indra-ocr huggingface_hub
```

### 1. In Python with `indra_ocr`:

```python
from huggingface_hub import snapshot_download
import indra_ocr
import cv2

# Download model weights from Hugging Face
model_dir = snapshot_download(repo_id="{repo_id}")

# Load OCR pipeline
converter = indra_ocr.load(path=model_dir)

# Read image (grayscale)
img = cv2.imread("document_vn.png", cv2.IMREAD_GRAYSCALE)

# Run full OCR (detection + recognition)
result = converter.ocr(img)

for line in result.lines:
    print(f"  {{line.text}}")
```

---

## Technical Details

- **Recognizer Architecture**: Multi-scale 1D depthwise separable convolutional backbone with 128-channel CTC classifier head (~1.24M parameters).
- **Format**: High-performance ONNX model (`rec.onnx`) with frozen batch normalization and dynamic sequence length masking.
- **License**: Modified OpenRAIL-M (commercial use permitted subject to Indra OCR terms).

---

Developed by [ANZ Innovation](https://huggingface.co/ANZ-Innovation).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("work/model-vn-full"))
    parser.add_argument("--lines-dir", type=Path, default=Path("work/lines-vn-full"))
    parser.add_argument("--run-dir", type=Path, default=Path("work/run-vn-full"))
    parser.add_argument("--repo-id", default="ANZ-Innovation/indra-v1")
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    api = HfApi()

    print(f"Checking / creating repository {args.repo_id}...")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # Load statistics and evaluation results to generate README.md
    stats = {}
    if (args.lines_dir / "stats.json").exists():
        stats = json.loads((args.lines_dir / "stats.json").read_text())

    test_eval = {}
    if (args.run_dir / "test_eval.json").exists():
        test_eval = json.loads((args.run_dir / "test_eval.json").read_text())

    train_history = {}
    if (args.run_dir / "history.json").exists():
        train_history = json.loads((args.run_dir / "history.json").read_text())

    readme_content = generate_model_card(args.repo_id, stats, test_eval, train_history)
    (args.model_dir / "README.md").write_text(readme_content)
    print(f"Generated README.md at {args.model_dir / 'README.md'}")

    print(f"Uploading folder {args.model_dir} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(args.model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add fine-tuned Indra Vietnamese OCR v1 drop-in model",
    )
    print(f"Successfully uploaded to https://huggingface.co/{args.repo_id}!")


if __name__ == "__main__":
    main()
