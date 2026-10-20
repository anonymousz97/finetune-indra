"""Gradio Web UI Demo for Indra OCR Vietnamese (v1).
Supports single image/PDF inference and side-by-side comparison between
Base Model (datalab-to/indra-ocr) and Fine-Tuned Model (ANZ-Innovation/indra-v1).
"""

from __future__ import annotations

import argparse
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
DEMO_SAMPLES_DIR = ROOT_DIR / "work" / "demo_samples"

print("Loading OCR engines...")
print("1. Loading Fine-Tuned Model from:", FT_MODEL_DIR)
pipeline_ft = indra_ocr.load(path=str(FT_MODEL_DIR))

print("2. Loading Base Model (datalab-to/indra-ocr)...")
pipeline_base = indra_ocr.load()
print("Both engines loaded successfully!")


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
    raise ValueError("Unsupported image format")


def to_rgb(img: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3:
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            if img.shape[2] == 3:
                return img
    raise ValueError("Unsupported image format")


def draw_bboxes(img_rgb: np.ndarray, lines: list, color=(0, 140, 255)) -> np.ndarray:
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    for i, l in enumerate(lines):
        x0, y0, x1, y1 = [int(round(v)) for v in l.box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        badge_y0 = max(0, y0 - th - 4)
        cv2.rectangle(vis, (x0, badge_y0), (x0 + tw + 4, max(0, y0)), color, -1)
        cv2.putText(vis, label, (x0 + 2, max(0, y0 - 2)), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def run_ocr(
    image: np.ndarray | Image.Image | None,
    pdf_file: str | None,
    pdf_page: int,
    mode: str,
):
    if image is None and not pdf_file:
        return None, "Vui lòng tải lên ảnh hoặc chọn file PDF để nhận diện.", None, None, None

    # Handle PDF input
    if pdf_file and image is None:
        try:
            doc = fitz.open(pdf_file)
            page_idx = max(0, min(len(doc) - 1, int(pdf_page) - 1))
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            rgb = to_rgb(img_arr)
            gray = to_gray(img_arr)
        except Exception as e:
            return None, f"Lỗi đọc PDF: {e}", None, None, None
    else:
        rgb = to_rgb(image)
        gray = to_gray(image)

    t0 = time.time()

    if mode == "Fine-Tuned (ANZ-Innovation/indra-v1)":
        res_ft = pipeline_ft.ocr(gray)
        latency = (time.time() - t0) * 1000
        vis = draw_bboxes(rgb, res_ft.lines, color=(0, 160, 255))
        text_out = "\n".join(l.text for l in res_ft.lines if l.text.strip())
        info = f"⚡ Thời gian xử lý: **{latency:.1f} ms** | Số dòng phát hiện: **{len(res_ft.lines)}**"
        
        table = [[i + 1, f"{l.conf:.2f}", f"[{int(l.box[0])}, {int(l.box[1])}, {int(l.box[2])}, {int(l.box[3])}]", l.text]
                 for i, l in enumerate(res_ft.lines)]
        return vis, text_out, info, table, None

    elif mode == "Base Model (datalab-to/indra-ocr)":
        res_base = pipeline_base.ocr(gray)
        latency = (time.time() - t0) * 1000
        vis = draw_bboxes(rgb, res_base.lines, color=(255, 100, 0))
        text_out = "\n".join(l.text for l in res_base.lines if l.text.strip())
        info = f"⚡ Thời gian xử lý: **{latency:.1f} ms** | Số dòng phát hiện: **{len(res_base.lines)}**"
        
        table = [[i + 1, f"{l.conf:.2f}", f"[{int(l.box[0])}, {int(l.box[1])}, {int(l.box[2])}, {int(l.box[3])}]", l.text]
                 for i, l in enumerate(res_base.lines)]
        return vis, text_out, info, table, None

    else:
        # Side-by-side comparison
        t_base = time.time()
        res_base = pipeline_base.ocr(gray)
        dur_base = (time.time() - t_base) * 1000

        t_ft = time.time()
        res_ft = pipeline_ft.ocr(gray)
        dur_ft = (time.time() - t_ft) * 1000

        vis = draw_bboxes(rgb, res_ft.lines, color=(0, 160, 255))
        text_ft = "\n".join(l.text for l in res_ft.lines if l.text.strip())
        text_base = "\n".join(l.text for l in res_base.lines if l.text.strip())

        info = (
            f"⚡ **Fine-Tuned**: {dur_ft:.1f} ms ({len(res_ft.lines)} dòng) | "
            f"⚡ **Base Model**: {dur_base:.1f} ms ({len(res_base.lines)} dòng)"
        )

        # Comparative table
        max_lines = max(len(res_base.lines), len(res_ft.lines))
        comp_table = []
        for i in range(max_lines):
            b_text = res_base.lines[i].text if i < len(res_base.lines) else ""
            f_text = res_ft.lines[i].text if i < len(res_ft.lines) else ""
            status = "✅ Khớp" if b_text == f_text else "🔥 Khác biệt (Fine-tuned sửa lỗi)"
            comp_table.append([i + 1, b_text, f_text, status])

        side_by_side = f"### 📊 So sánh văn bản:\n\n**[Fine-Tuned Model]**\n```text\n{text_ft}\n```\n\n**[Base Model]**\n```text\n{text_base}\n```"
        return vis, text_ft, info, comp_table, side_by_side


def build_app():
    custom_css = """
    .gradio-container { max-width: 1250px !important; margin: auto !important; }
    .header-box { text-align: center; margin-bottom: 20px; }
    """

    with gr.Blocks(title="Indra OCR Vietnamese Demo") as demo:
        gr.Markdown(
            """
            # 🇻🇳 Indra OCR Vietnamese (v1) - Interactive Testing UI
            ### Model Drop-in: [ANZ-Innovation/indra-v1](https://huggingface.co/ANZ-Innovation/indra-v1) | Base: [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr)
            Mô hình được tối ưu hóa chuyên sâu cho tài liệu hành chính, thông tư, quyết định chính phủ, bài báo khoa học và biểu mẫu tiếng Việt.
            """
        )

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

                # Sample images
                example_samples = []
                for p in ["sample_thongtu_86_btc.png", "sample_quyetdinh_109_btc.png", "sample_so_yeu_ly_lich.png"]:
                    full_p = DEMO_SAMPLES_DIR / p
                    if full_p.exists():
                        example_samples.append([str(full_p)])

                if example_samples:
                    gr.Markdown("### 📌 Mẫu tài liệu thử nghiệm có sẵn:")
                    gr.Examples(
                        examples=example_samples,
                        inputs=[input_img],
                        label="Bấm vào mẫu để tải nhanh:",
                    )

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
    parser.add_argument("--share", action="store_true", default=True, help="Enable public gradio.live share link")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
