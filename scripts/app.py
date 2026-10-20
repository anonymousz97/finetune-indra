"""Gradio Web UI Demo for:
1. Indra OCR Vietnamese (v1) - OCR Line Recognition & Comparison
2. End-to-End Document Schema Extraction (Indra OCR + Qwen2.5-0.5B Extractor)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import fitz
import gradio as gr
import numpy as np
from PIL import Image

import indra_ocr

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
FT_MODEL_DIR = ROOT_DIR / "work" / "model-vn-v2"
EXTRACTOR_DIR = ROOT_DIR / "work" / "qwen_extractor_lora"
DEMO_SAMPLES_DIR = ROOT_DIR / "work" / "demo_samples"

print("Loading OCR engines...")
print("1. Loading Fine-Tuned Model from:", FT_MODEL_DIR)
pipeline_ft = indra_ocr.load(path=str(FT_MODEL_DIR))

print("2. Loading Base Model (datalab-to/indra-ocr)...")
pipeline_base = indra_ocr.load()
print("Both OCR engines loaded successfully!")

# Global lazy-loaded E2E pipeline
_e2e_pipeline = None


def get_e2e_pipeline():
    global _e2e_pipeline
    if _e2e_pipeline is None:
        import sys
        sys.path.insert(0, str(ROOT_DIR / "scripts"))
        from pipeline_e2e import DocumentSchemaPipeline
        _e2e_pipeline = DocumentSchemaPipeline(
            ocr_model_dir=FT_MODEL_DIR,
            extractor_base_id="Qwen/Qwen2.5-0.5B-Instruct",
            extractor_lora_dir=EXTRACTOR_DIR if EXTRACTOR_DIR.exists() else None
        )
    return _e2e_pipeline


def to_gray(img: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("L"))
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image type: {type(img)}")


def draw_boxes(image: np.ndarray, lines: list) -> np.ndarray:
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis = image.copy()

    for i, line in enumerate(lines):
        box = line.box
        pts = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(vis, str(i + 1), (int(box[0]), max(15, int(box[1]) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return vis


def run_ocr(input_img, input_pdf, page_num, mode):
    img_gray = None
    if input_pdf is not None:
        doc = fitz.open(input_pdf.name)
        page_idx = min(max(1, int(page_num)), len(doc)) - 1
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_gray = np.array(img.convert("L"))
    elif input_img is not None:
        img_gray = to_gray(input_img)
    else:
        return None, "Vui lòng tải ảnh hoặc file PDF để thử nghiệm!", "Chưa có ảnh đầu vào", [], ""

    if mode == "Fine-Tuned (ANZ-Innovation/indra-v1)":
        t0 = time.time()
        res = pipeline_ft.ocr(img_gray)
        elapsed = (time.time() - t0) * 1000
        vis = draw_boxes(img_gray, res.lines)
        text = "\n".join(l.text for l in res.lines)
        info = f"⚡ Thời gian xử lý: **{elapsed:.1f} ms** | Số dòng phát hiện: **{len(res.lines)}**"
        table = [[i + 1, l.text, "", ""] for i, l in enumerate(res.lines)]
        return vis, text, info, table, ""

    elif mode == "Base Model (datalab-to/indra-ocr)":
        t0 = time.time()
        res = pipeline_base.ocr(img_gray)
        elapsed = (time.time() - t0) * 1000
        vis = draw_boxes(img_gray, res.lines)
        text = "\n".join(l.text for l in res.lines)
        info = f"⚡ Thời gian xử lý: **{elapsed:.1f} ms** | Số dòng phát hiện: **{len(res.lines)}**"
        table = [[i + 1, l.text, "", ""] for i, l in enumerate(res.lines)]
        return vis, text, info, table, ""

    else:
        # Side-by-side mode
        t0 = time.time()
        res_ft = pipeline_ft.ocr(img_gray)
        t_ft = (time.time() - t0) * 1000

        t0 = time.time()
        res_base = pipeline_base.ocr(img_gray)
        t_base = (time.time() - t0) * 1000

        vis = draw_boxes(img_gray, res_ft.lines)
        text_ft = "\n".join(l.text for l in res_ft.lines)
        text_base = "\n".join(l.text for l in res_base.lines)

        info = f"""### ⏱️ So sánh thời gian & phát hiện:
- **Fine-Tuned (ANZ-Innovation/indra-v1)**: `{t_ft:.1f} ms` | {len(res_ft.lines)} dòng
- **Base Model (datalab-to/indra-ocr)**: `{t_base:.1f} ms` | {len(res_base.lines)} dòng
"""
        max_lines = max(len(res_base.lines), len(res_ft.lines))
        comp_table = []
        for i in range(max_lines):
            b_text = res_base.lines[i].text if i < len(res_base.lines) else ""
            f_text = res_ft.lines[i].text if i < len(res_ft.lines) else ""
            status = "✅ Khớp" if b_text == f_text else "🔥 Khác biệt (Fine-tuned sửa lỗi)"
            comp_table.append([i + 1, b_text, f_text, status])

        side_by_side = f"### 📊 So sánh văn bản:\n\n**[Fine-Tuned Model]**\n```text\n{text_ft}\n```\n\n**[Base Model]**\n```text\n{text_base}\n```"
        return vis, text_ft, info, comp_table, side_by_side


# Preset schema definitions
PRESET_SCHEMAS = {
    "Văn bản hành chính / Thông tư": json.dumps({
        "so_hieu": "Số hiệu chính thức của văn bản (ví dụ: 86/2026/TT-BTC)",
        "co_quan_ban_hanh": "Tên cơ quan, tổ chức hoặc bộ ngành ban hành văn bản",
        "ngay_ban_hanh": "Ngày tháng năm ký hoặc ban hành văn bản",
        "trich_yeu": "Nội dung trích yếu hoặc chủ đề chính của văn bản",
        "nguoi_ky": "Họ và tên người có thẩm quyền ký văn bản",
        "tong_tien_vat": "Tổng số tiền thuế VAT phát sinh (trường bẫy nếu không có)"
    }, ensure_ascii=False, indent=2),

    "Hóa đơn GTGT (VAT)": json.dumps({
        "so_hoa_don": "Số hóa đơn điện tử giá trị gia tăng",
        "don_vi_ban_hang": "Tên công ty hoặc đơn vị bán hàng",
        "ma_so_thue_nguoi_ban": "Mã số thuế của công ty bán hàng",
        "tong_tien_thanh_toan": "Tổng cộng tiền thanh toán cuối cùng của hóa đơn",
        "so_tai_khoan_ngan_hang": "Số tài khoản ngân hàng thụ hưởng"
    }, ensure_ascii=False, indent=2),

    "Hợp đồng kinh tế": json.dumps({
        "so_hop_dong": "Số hiệu định danh của bản hợp đồng kinh tế",
        "ben_a": "Tên công ty hoặc tổ chức đại diện Bên A",
        "ben_b": "Tên công ty hoặc tổ chức đại diện Bên B",
        "gia_tri_hop_dong": "Tổng giá trị thanh toán của hợp đồng",
        "ngay_ky_hop_dong": "Ngày các bên ký kết hợp đồng",
        "thoi_han_thuc_hien": "Thời hạn hoàn thành hợp đồng"
    }, ensure_ascii=False, indent=2),

    "Sơ yếu lý lịch": json.dumps({
        "ho_va_ten": "Họ và tên đầy đủ của cá nhân",
        "ngay_sinh": "Ngày tháng năm sinh",
        "so_cccd": "Số thẻ căn cước công dân hoặc số định danh cá nhân",
        "so_dien_thoai": "Số điện thoại di động liên hệ",
        "vi_tri_ung_tuyen": "Vị trí công việc muốn ứng tuyển"
    }, ensure_ascii=False, indent=2)
}


def run_e2e_extraction(input_img, input_pdf, page_num, schema_str):
    img_gray = None
    if input_pdf is not None:
        doc = fitz.open(input_pdf.name)
        page_idx = min(max(1, int(page_num)), len(doc)) - 1
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_gray = np.array(img.convert("L"))
    elif input_img is not None:
        img_gray = to_gray(input_img)
    else:
        return {}, "Vui lòng tải ảnh hoặc file PDF!", "⚠️ Chưa có tài liệu đầu vào"

    try:
        schema = json.loads(schema_str)
    except Exception as e:
        return {"error": f"Lỗi cú pháp JSON Schema: {e}"}, "", f"⚠️ Lỗi cú pháp JSON Schema: {e}"

    pipeline = get_e2e_pipeline()
    res = pipeline.process_document(img_gray, schema)

    lat = res["latency"]
    lat_info = f"""### ⏱️ Thống kê độ trễ End-to-End:
- **1. Indra OCR (Phát hiện & Nhận diện {res['detected_lines_count']} dòng)**: **`{lat['ocr_latency_ms']:.1f} ms`**
- **2. Qwen2.5-0.5B Extractor (Trích xuất & gán nhãn)**: **`{lat['extraction_latency_ms']:.1f} ms`**
- **🚀 Tổng thời gian E2E**: **`{lat['total_e2e_latency_ms']:.1f} ms`** (`{lat['total_e2e_latency_ms']/1000:.2f} giây`)
"""
    return res["extracted_json"], res["markdown_text"], lat_info


def build_app():
    custom_css = """
    .gradio-container { max-width: 1300px !important; margin: auto !important; }
    """

    with gr.Blocks(title="Indra OCR & Schema Extraction Demo", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🇻🇳 Indra OCR & Schema Extractor Suite (ANZ Innovation)
            ### Pipeline khép kín: **Ảnh/PDF** $\\to$ **Indra OCR (1.2M params)** $\\to$ **Markdown** $\\to$ **Qwen2.5-0.5B Extractor** $\\to$ **JSON Chuẩn**
            Mô hình nhận diện OCR: [ANZ-Innovation/indra-v1](https://huggingface.co/ANZ-Innovation/indra-v1) | Extractor: [ANZ-Innovation/qwen2.5-0.5b-vietnamese-schema-extractor](https://huggingface.co/ANZ-Innovation/qwen2.5-0.5b-vietnamese-schema-extractor)
            """
        )

        with gr.Tabs():
            # ==================================================================
            # TAB 1: END-TO-END SCHEMA EXTRACTION
            # ==================================================================
            with gr.Tab("📑 1. Bóc Tách Schema Tự Động (Indra + Qwen Extractor)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### 📥 1. Tài liệu đầu vào:")
                        with gr.Tab("📷 Tải ảnh"):
                            e2e_img = gr.Image(type="numpy", label="Chọn ảnh tài liệu (PNG/JPG)")
                        with gr.Tab("📄 Tải PDF"):
                            e2e_pdf = gr.File(label="Tải file PDF", file_types=[".pdf"])
                            e2e_pdf_page = gr.Slider(minimum=1, maximum=30, value=1, step=1, label="Trang PDF")

                        gr.Markdown("### ⚙️ 2. Schema cần trích xuất:")
                        e2e_schema_input = gr.Textbox(
                            value=PRESET_SCHEMAS["Văn bản hành chính / Thông tư"],
                            label="JSON Schema (hỗ trợ mô tả description hoặc danh sách key)",
                            lines=10
                        )

                        gr.Markdown("**Chọn nhanh mẫu Schema định sẵn:**")
                        with gr.Row():
                            btn_p1 = gr.Button("Văn bản hành chính", size="sm")
                            btn_p2 = gr.Button("Hóa đơn VAT", size="sm")
                            btn_p3 = gr.Button("Hợp đồng", size="sm")
                            btn_p4 = gr.Button("Sơ yếu lý lịch", size="sm")

                        btn_p1.click(lambda: PRESET_SCHEMAS["Văn bản hành chính / Thông tư"], outputs=[e2e_schema_input])
                        btn_p2.click(lambda: PRESET_SCHEMAS["Hóa đơn GTGT (VAT)"], outputs=[e2e_schema_input])
                        btn_p3.click(lambda: PRESET_SCHEMAS["Hợp đồng kinh tế"], outputs=[e2e_schema_input])
                        btn_p4.click(lambda: PRESET_SCHEMAS["Sơ yếu lý lịch"], outputs=[e2e_schema_input])

                        btn_run_e2e = gr.Button("🚀 Bóc Tách Schema End-to-End", variant="primary", size="lg")

                        # Examples
                        example_samples = []
                        for p in ["sample_thongtu_86_btc.png", "sample_quyetdinh_109_btc.png", "sample_so_yeu_ly_lich.png"]:
                            full_p = DEMO_SAMPLES_DIR / p
                            if full_p.exists():
                                example_samples.append([str(full_p)])

                        if example_samples:
                            gr.Examples(
                                examples=example_samples,
                                inputs=[e2e_img],
                                label="Bấm vào mẫu có sẵn để tải nhanh:",
                            )

                    with gr.Column(scale=7):
                        e2e_lat_md = gr.Markdown("⚡ Trạng thái: Sẵn sàng thực thi")
                        gr.Markdown("### 📤 3. Kết Quả JSON Trích Xuất (Khớp 100% Schema):")
                        e2e_json_output = gr.JSON(label="Dữ liệu JSON có cấu trúc")

                        with gr.Accordion("📝 Xem văn bản Markdown trung gian từ Indra OCR", open=False):
                            e2e_md_output = gr.Textbox(label="Nội dung Markdown", lines=10, buttons=["copy"])

                btn_run_e2e.click(
                    fn=run_e2e_extraction,
                    inputs=[e2e_img, e2e_pdf, e2e_pdf_page, e2e_schema_input],
                    outputs=[e2e_json_output, e2e_md_output, e2e_lat_md]
                )

            # ==================================================================
            # TAB 2: OCR COMPARISON
            # ==================================================================
            with gr.Tab("🔍 2. Nhận Diện Dòng OCR (Indra v2 vs Base Model)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Tab("📷 Tải ảnh"):
                            input_img = gr.Image(type="numpy", label="Chọn ảnh tài liệu (PNG/JPG)")
                        with gr.Tab("📄 Tải PDF"):
                            input_pdf = gr.File(label="Tải tài liệu PDF", file_types=[".pdf"])
                            pdf_page = gr.Slider(minimum=1, maximum=30, value=1, step=1, label="Trang PDF")

                        mode_selector = gr.Radio(
                            choices=[
                                "So sánh song song (Side-by-Side: Base vs Fine-Tuned)",
                                "Fine-Tuned (ANZ-Innovation/indra-v1)",
                                "Base Model (datalab-to/indra-ocr)",
                            ],
                            value="So sánh song song (Side-by-Side: Base vs Fine-Tuned)",
                            label="Chế độ kiểm thử (Mode)",
                        )

                        btn_run = gr.Button("🚀 Nhận diện Text ngay", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        info_md = gr.Markdown("⚡ Trạng thái: Sẵn sàng")
                        output_vis = gr.Image(label="Ảnh phát hiện dòng (Bounding Boxes)")
                        
                        with gr.Tab("📝 Văn bản nhận diện"):
                            output_text = gr.Textbox(label="Văn bản bóc tách", lines=12, buttons=["copy"])
                        
                        with gr.Tab("⚖️ Đối chiếu song song"):
                            side_by_side_md = gr.Markdown("Chạy chế độ 'So sánh song song' để xem đối chiếu chi tiết.")

                        with gr.Tab("📋 Chi tiết từng dòng"):
                            output_table = gr.Dataframe(
                                headers=["Dòng #", "Base Model", "Fine-Tuned Model", "Trạng thái"],
                                interactive=False,
                            )

                btn_run.click(
                    fn=run_ocr,
                    inputs=[input_img, input_pdf, pdf_page, mode_selector],
                    outputs=[output_vis, output_text, info_md, output_table, side_by_side_md],
                )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True)
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
