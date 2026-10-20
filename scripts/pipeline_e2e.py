"""End-to-End Pipeline: Document Image/PDF -> Indra OCR v2 -> Clean Markdown -> Schema Extractor.
Measures and reports sub-millisecond latency for each stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import indra_ocr


class DocumentSchemaPipeline:
    def __init__(
        self,
        ocr_model_dir: str | Path = "work/model-vn-v2",
        extractor_base_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        extractor_lora_dir: str | Path | None = "work/qwen_extractor_lora",
        device: str | None = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing DocumentSchemaPipeline on device: {self.device}...")
        
        # 1. Load Fine-tuned Indra OCR
        print(f"Loading Indra OCR from: {ocr_model_dir}...")
        t0 = time.time()
        self.ocr_engine = indra_ocr.load(path=str(ocr_model_dir))
        print(f"Indra OCR loaded in {time.time() - t0:.2f}s!")

        # 2. Load Extractor Model
        print(f"Loading Schema Extractor from: {extractor_base_id} (LoRA: {extractor_lora_dir})...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(extractor_base_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            extractor_base_id,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=self.device
        )

        if extractor_lora_dir and Path(extractor_lora_dir).exists():
            print(f"Applying LoRA weights from {extractor_lora_dir}...")
            self.extractor_model = PeftModel.from_pretrained(base_model, str(extractor_lora_dir))
        else:
            print("Running extractor with base weights (Zero-shot)...")
            self.extractor_model = base_model
        
        self.extractor_model.eval()
        print(f"Extractor loaded in {time.time() - t0:.2f}s!")

    def _to_gray(self, img_input: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(img_input, (str, Path)):
            p = str(img_input)
            if p.lower().endswith(".pdf"):
                doc = fitz.open(p)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return np.array(img.convert("L"))
            else:
                return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img_input, Image.Image):
            return np.array(img_input.convert("L"))
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                return img_input
            if img_input.ndim == 3:
                return cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY if img_input.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported image input type: {type(img_input)}")

    def run_ocr(self, img_input: str | Path | np.ndarray | Image.Image) -> tuple[str, list[dict], float]:
        t0 = time.time()
        gray = self._to_gray(img_input)
        res = self.ocr_engine.ocr(gray)
        ocr_time_ms = (time.time() - t0) * 1000

        lines = []
        for line in res.lines:
            txt = line.text.strip()
            if txt:
                lines.append({"text": txt, "box": line.box})

        # Format into clean reading order Markdown
        markdown_text = "\n\n".join(l["text"] for l in lines)
        return markdown_text, lines, round(ocr_time_ms, 2)

    def extract_schema(
        self,
        markdown_text: str,
        target_schema: list[str] | dict[str, str],
        max_new_tokens: int = 384
    ) -> tuple[dict[str, Any], str, float]:
        t0 = time.time()
        
        system_prompt = """Bạn là trợ lý AI chuyên gia trích xuất thông tin có cấu trúc từ văn bản tài liệu sang định dạng JSON.
Nhiệm vụ của bạn là:
1. Nhận văn bản Markdown và Schema yêu cầu (dưới dạng danh sách các trường hoặc từ điển mô tả ngữ nghĩa của từng trường).
2. Trích xuất CHÍNH XÁC giá trị tương ứng cho từng trường từ văn bản.
3. QUY TẮC BẮT BUỘC (Trained Abstention): Nếu một trường trong Schema KHÔNG CÓ thông tin trong văn bản, BẮT BUỘC gán giá trị của trường đó là null. Tuyệt đối không được đoán mò hoặc bịa đặt giá trị.
4. Đầu ra CHỈ LÀ một khối JSON hợp lệ duy nhất khớp đúng các key được yêu cầu, không kèm bất kỳ lời giải thích nào khác."""

        if isinstance(target_schema, dict):
            schema_instr = "Schema yêu cầu trích xuất (kèm mô tả ngữ nghĩa cho từng trường):"
            schema_rendered = json.dumps(target_schema, ensure_ascii=False, indent=2)
        else:
            schema_instr = "Schema yêu cầu trích xuất (danh sách trường):"
            schema_rendered = json.dumps(list(target_schema), ensure_ascii=False, indent=2)

        user_prompt = f"""### Văn bản tài liệu:
```markdown
{markdown_text.strip()}
```

### {schema_instr}
```json
{schema_rendered}
```

Hãy trích xuất thông tin và trả về JSON theo đúng các key được yêu cầu. Trường nào không có thông tin hãy để giá trị là null."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.extractor_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        raw_resp = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        extract_time_ms = (time.time() - t0) * 1000

        # Parse JSON
        resp = raw_resp
        if "```json" in resp:
            resp = resp.split("```json")[-1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[-1].split("```")[0].strip()

        try:
            extracted_json = json.loads(resp)
        except Exception:
            extracted_json = {"error": "Failed to parse JSON", "raw_output": raw_resp}

        return extracted_json, raw_resp, round(extract_time_ms, 2)

    def process_document(
        self,
        img_input: str | Path | np.ndarray | Image.Image,
        target_schema: list[str] | dict[str, str]
    ) -> dict[str, Any]:
        t0 = time.time()
        markdown_text, lines, ocr_time_ms = self.run_ocr(img_input)
        extracted_json, raw_resp, extract_time_ms = self.extract_schema(markdown_text, target_schema)
        total_e2e_ms = (time.time() - t0) * 1000

        return {
            "extracted_json": extracted_json,
            "markdown_text": markdown_text,
            "detected_lines_count": len(lines),
            "latency": {
                "ocr_latency_ms": ocr_time_ms,
                "extraction_latency_ms": extract_time_ms,
                "total_e2e_latency_ms": round(total_e2e_ms, 2)
            }
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="End-to-End Indra OCR + Schema Extractor Pipeline")
    parser.add_argument("--image", default="work/demo_samples/sample_thongtu_86_btc.png")
    parser.add_argument("--ocr-dir", default="work/model-vn-v2")
    parser.add_argument("--extractor-dir", default="work/qwen_extractor_merged")
    args = parser.parse_args()

    pipeline = DocumentSchemaPipeline(
        ocr_model_dir=args.ocr_dir,
        extractor_base_id=args.extractor_dir,
        extractor_lora_dir=None
    )

    test_schema = {
        "so_hieu": "Số hiệu chính thức của thông tư hoặc văn bản",
        "co_quan_ban_hanh": "Tên cơ quan ban hành văn bản",
        "ngay_ban_hanh": "Ngày tháng năm ban hành văn bản",
        "nguoi_ky": "Họ và tên người ký văn bản",
        "tong_tien_vat": "Tổng số tiền thuế giá trị gia tăng (nếu có)",
        "thoi_han_bao_hanh": "Thời hạn bảo hành hợp đồng (nếu có)"
    }

    print(f"\nProcessing {args.image}...")
    res = pipeline.process_document(args.image, test_schema)

    print("\n--- OCR MARKDOWN TEXT ---")
    print(res["markdown_text"])

    print("\n--- EXTRACTED JSON ---")
    print(json.dumps(res["extracted_json"], ensure_ascii=False, indent=2))

    print("\n--- LATENCY REPORT ---")
    print(f"OCR Latency:        {res['latency']['ocr_latency_ms']} ms")
    print(f"Extraction Latency: {res['latency']['extraction_latency_ms']} ms")
    print(f"Total E2E Latency:  {res['latency']['total_e2e_latency_ms']} ms")
