"""Preprocess and build a diverse Vietnamese line-level OCR dataset from:
1. Chinhphu PDFs (administrative, government circulars, legal decrees, official documents)
2. VJOL PDFs (academic journal articles, scientific terminology, research papers)
3. Datalab snapshot (forms, resumes/sơ yếu lý lịch, vehicle inspection certificates, tables)

Saves directly into the compressed .npz + -labels.json format for train_rec.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import unicodedata
import zipfile
from pathlib import Path

import cv2
import fitz
import numpy as np

from indra_ocr.rec_preproc import crop_line
from indra_ocr.vocab import canonical


def strip_html_tags(text: str) -> list[str]:
    t = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    t = re.sub(r"</(?:p|div|tr|li|h[1-6])>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    lines = [unicodedata.normalize("NFC", line).strip() for line in t.splitlines()]
    return [l for l in lines if len(l) > 1]


def extract_from_pdf(pdf_path: str, max_pages: int = 0) -> list[tuple[np.ndarray, str]]:
    results = []
    try:
        doc = fitz.open(pdf_path)
        pages_to_process = len(doc) if max_pages <= 0 else min(len(doc), max_pages)
        for page_idx in range(pages_to_process):
            page = doc[page_idx]
            d = page.get_text("dict")
            blocks = [b for b in d.get("blocks", []) if "lines" in b]
            if not blocks:
                continue

            # Render at 150 DPI for crisp line crops
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
            scale_x = pix.width / page.rect.width
            scale_y = pix.height / page.rect.height

            for b in blocks:
                for l in b["lines"]:
                    line_text = "".join(s["text"] for s in l["spans"]).strip()
                    line_text = canonical(unicodedata.normalize("NFC", line_text))
                    if len(line_text) < 2 or len(line_text) > 120:
                        continue
                    x0 = l["bbox"][0] * scale_x
                    y0 = l["bbox"][1] * scale_y
                    x1 = l["bbox"][2] * scale_x
                    y1 = l["bbox"][3] * scale_y
                    if x1 - x0 < 10 or y1 - y0 < 5:
                        continue
                    box = [x0, y0, x1, y1]
                    try:
                        crop = crop_line(gray, box)
                        if crop.shape[1] >= 16:
                            results.append((crop, line_text))
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return results


def extract_from_datalab_zip(zip_path: str, max_pages: int = 0) -> list[tuple[np.ndarray, str]]:
    results = []
    if not os.path.exists(zip_path):
        return results

    print(f"   Opening zip file {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        name_list = z.namelist()
        name_set = set(name_list)
        raw_files = [n for n in name_list if n.endswith(".raw.json") and "preview/" in n]
        random.seed(42)
        random.shuffle(raw_files)
        selected_files = raw_files if max_pages <= 0 else raw_files[:max_pages]
        print(f"   Total raw files in zip: {len(raw_files)}, processing: {len(selected_files)} pages...")

        for i, rf in enumerate(selected_files):
            if (i + 1) % 200 == 0 or (i + 1) == len(selected_files):
                print(f"   -> Datalab archive: processed {i + 1}/{len(selected_files)} pages ({len(results)} lines extracted)...")
            try:
                base_id = os.path.basename(rf).replace(".raw.json", "")
                img_name = f"dataset/images/{base_id}.png"
                if img_name not in name_set:
                    continue

                img_bytes = z.open(img_name).read()
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                gray = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue

                d = json.loads(z.open(rf).read().decode("utf-8"))
                children = d.get("json", {}).get("children", [{}])[0].get("children", [])
                for sc in children:
                    b_type = sc.get("block_type")
                    if b_type not in ["Text", "SectionHeader", "PageHeader", "Form", "Table"]:
                        continue
                    bbox = sc.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    html_content = sc.get("html") or ""
                    lines = strip_html_tags(html_content)
                    if not lines:
                        continue

                    # If block has 1 line, its bbox maps directly to the line
                    if len(lines) == 1:
                        text = canonical(lines[0])
                        if len(text) >= 2 and len(text) <= 120:
                            box = bbox
                            if box[2] - box[0] >= 10 and box[3] - box[1] >= 5:
                                crop = crop_line(gray, box)
                                if crop.shape[1] >= 16:
                                    results.append((crop, text))
                    else:
                        # For multi-line blocks, slice proportionally along vertical height
                        total_lines = len(lines)
                        block_h = bbox[3] - bbox[1]
                        line_h = block_h / total_lines
                        for idx, l_text in enumerate(lines):
                            text = canonical(l_text)
                            if len(text) < 2 or len(text) > 120:
                                continue
                            y0 = bbox[1] + idx * line_h
                            y1 = y0 + line_h
                            box = [bbox[0], y0, bbox[2], y1]
                            if box[2] - box[0] >= 10 and box[3] - box[1] >= 5:
                                crop = crop_line(gray, box)
                                if crop.shape[1] >= 16:
                                    results.append((crop, text))
            except Exception:
                continue
    return results


def save_split(out_dir: Path, split_name: str, pairs: list[tuple[np.ndarray, str]]):
    crops = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    widths = np.array([c.shape[1] for c in crops], dtype=np.int32)
    flat = np.concatenate([(c * 255.0).round().astype(np.uint8).ravel() for c in crops])
    np.savez_compressed(
        out_dir / f"{split_name}.npz",
        flat=flat,
        widths=widths,
        pages=np.zeros(len(crops), dtype=np.int32),
    )
    (out_dir / f"{split_name}-labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    print(
        f"  Split [{split_name:10s}]: {len(labels):,d} lines | "
        f"median width: {int(np.median(widths))}px | "
        f"total chars: {sum(len(l) for l in labels):,d}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinhphu-dir", default="/home/evg-user/surya-viet/data/chinhphu_pdf")
    parser.add_argument("--vjol-dir", default="/home/evg-user/surya-viet/data/vjol_raw")
    parser.add_argument("--datalab-zip", default="/home/evg-user/surya-viet/finetune/data/dataset__datalab__24092026.zip")
    parser.add_argument("--max-pdf-pages", type=int, default=0, help="0 means all pages")
    parser.add_argument("--max-datalab-pages", type=int, default=0, help="0 means all pages")
    parser.add_argument("--out", type=Path, default=Path("work/lines-vn"))
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=== Extracting Diverse Vietnamese Text Lines ===")

    # 1. Government & Legal (Chinhphu)
    print("\n1. Scanning Chinhphu PDFs (Administrative / Government decrees)...")
    cp_pdfs = sorted(glob.glob(os.path.join(args.chinhphu_dir, "*.pdf")))
    cp_pairs = []
    for i, pdf in enumerate(cp_pdfs):
        lines = extract_from_pdf(pdf, max_pages=args.max_pdf_pages)
        cp_pairs.extend(lines)
        if (i + 1) % 10 == 0 or (i + 1) == len(cp_pdfs):
            print(f"   Chinhphu: {i + 1}/{len(cp_pdfs)} PDFs processed ({len(cp_pairs)} lines)...")
    print(f"   Extracted {len(cp_pairs):,d} lines from {len(cp_pdfs)} Chinhphu PDFs")

    # 2. Academic / Scientific (VJOL)
    print("\n2. Scanning VJOL PDFs (Academic research papers)...")
    vjol_pdfs = sorted(glob.glob(os.path.join(args.vjol_dir, "*.pdf")))
    vjol_pairs = []
    for i, pdf in enumerate(vjol_pdfs):
        lines = extract_from_pdf(pdf, max_pages=args.max_pdf_pages)
        vjol_pairs.extend(lines)
        if (i + 1) % 5 == 0 or (i + 1) == len(vjol_pdfs):
            print(f"   VJOL: {i + 1}/{len(vjol_pdfs)} PDFs processed ({len(vjol_pairs)} lines)...")
    print(f"   Extracted {len(vjol_pairs):,d} lines from {len(vjol_pdfs)} VJOL PDFs")

    # 3. Forms / Tables / Resumes (Datalab zip)
    print("\n3. Scanning Datalab Archive (Forms, resumes, vehicle certificates, tables)...")
    datalab_pairs = extract_from_datalab_zip(args.datalab_zip, max_pages=args.max_datalab_pages)
    print(f"   Extracted {len(datalab_pairs):,d} lines from Datalab archive")

    # Group and shuffle within domains, then combine
    random.shuffle(cp_pairs)
    random.shuffle(vjol_pairs)
    random.shuffle(datalab_pairs)

    def split_domain(domain_pairs):
        n = len(domain_pairs)
        n_val = max(100, int(n * 0.08))
        n_test = max(100, int(n * 0.08))
        n_train = n - n_val - n_test
        return domain_pairs[:n_train], domain_pairs[n_train:n_train + n_val], domain_pairs[n_train + n_val:]

    cp_tr, cp_va, cp_te = split_domain(cp_pairs)
    vjol_tr, vjol_va, vjol_te = split_domain(vjol_pairs)
    dl_tr, dl_va, dl_te = split_domain(datalab_pairs)

    train_pairs = cp_tr + vjol_tr + dl_tr
    val_pairs = cp_va + vjol_va + dl_va
    test_pairs = cp_te + vjol_te + dl_te

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    print(f"\n=== Dataset Summary ===")
    print(f"Total lines: {len(train_pairs) + len(val_pairs) + len(test_pairs):,d}")
    print(f"  - Administrative / Government : {len(cp_pairs):,d} lines")
    print(f"  - Academic / Scientific       : {len(vjol_pairs):,d} lines")
    print(f"  - Forms / Tables / Documents  : {len(datalab_pairs):,d} lines")

    print("\nSaving splits to:", args.out)
    save_split(args.out, "train", train_pairs)
    save_split(args.out, "validation", val_pairs)
    save_split(args.out, "test", test_pairs)

    stats = {
        "train_lines": len(train_pairs),
        "validation_lines": len(val_pairs),
        "test_lines": len(test_pairs),
        "total_lines": len(train_pairs) + len(val_pairs) + len(test_pairs),
        "sources": {
            "chinhphu_administrative": len(cp_pairs),
            "vjol_academic": len(vjol_pairs),
            "datalab_forms_tables": len(datalab_pairs),
        },
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("\nDataset ready!")


if __name__ == "__main__":
    main()
