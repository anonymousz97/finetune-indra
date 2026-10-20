"""Generate high-quality Vietnamese Schema Extraction dataset with Trained Abstention.
Supports Dual-Mode Schemas:
  1. Rich semantic descriptions (e.g. {"so_hieu": "Số hiệu văn bản..."})
  2. Simple key lists (e.g. ["so_hieu", "co_quan"])
  3. Key variations (e.g. "so_hieu", "số_hiệu", "so_van_ban")
Covers 5 document categories with 100% deterministic alignment and verified substring checks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ==============================================================================
# VOCABULARY & ENTITY POOLS FOR VIETNAMESE DOCUMENTS
# ==============================================================================

ORGS_GOV = [
    "CHÍNH PHỦ", "BỘ TÀI CHÍNH", "BỘ KẾ HOẠCH VÀ ĐẦU TƯ", "BỘ THÔNG TIN VÀ TRUYỀN THÔNG",
    "BỘ CÔNG THƯƠNG", "BỘ Y TẾ", "BỘ GIÁO DỤC VÀ ĐÀO TẠO", "BỘ TƯ PHÁP",
    "ỦY BAN NHÂN DÂN THÀNH PHỐ HÀ NỘI", "ỦY BAN NHÂN DÂN THÀNH PHỐ HỒ CHÍ MINH",
    "ỦY BAN NHÂN DÂN TỈNH BÌNH DƯƠNG", "ỦY BAN NHÂN DÂN TỈNH ĐỒNG NAI",
    "TỔNG CỤC HẢI QUAN", "TỔNG CỤC THUẾ", "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM"
]

ORGS_CORP = [
    "CÔNG TY CỔ PHẦN CÔNG NGHỆ VÀ TRUYỀN THÔNG VIỆT NAM",
    "CÔNG TY TNHH GIẢI PHÁP PHẦN MỀM TOÀN CẦU",
    "TẬP ĐOÀN CÔNG NGHIỆP - VIỄN THÔNG QUÂN ĐỘI",
    "CÔNG TY CỔ PHẦN TẬP ĐOÀN ĐẦU TƯ ĐỊA ỐC NOVA",
    "CÔNG TY CỔ PHẦN SỮA VIỆT NAM (VINAMILK)",
    "CÔNG TY TNHH PHÁT TRIỂN CÔNG NGHỆ THÔNG MINH ANZ",
    "NGÂN HÀNG TMCP NGOẠI THƯƠNG VIỆT NAM (VIETCOMBANK)",
    "CÔNG TY TNHH DỊCH VỤ VẬN TẢI VÀ TIẾP VẬN HẢI ÂU"
]

NAMES = [
    "Nguyễn Văn An", "Trần Thị Mai", "Lê Hoàng Long", "Phạm Quốc Tuấn",
    "Vũ Đình Trọng", "Đặng Minh Tâm", "Hoàng Kim Chi", "Bùi Thanh Sơn",
    "Ngô Quang Dũng", "Đỗ Hương Giang", "Nguyễn Đức Chi", "Trịnh Đình Dũng",
    "Phan Văn Khải", "Võ Thị Ánh Xuân", "Hồ Đức Phớc", "Nguyễn Chí Dũng"
]

CITIES = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Bình Dương", "Đồng Nai", "Quảng Ninh"]

POSITIONS = ["Giám đốc", "Tổng Giám đốc", "Thứ trưởng", "Bộ trưởng", "Chủ tịch UBND", "Kế toán trưởng", "Trưởng phòng Nhân sự", "Trưởng ban Pháp chế"]

DOC_TYPES_GOV = ["NGHỊ ĐỊNH", "THÔNG TƯ", "QUYẾT ĐỊNH", "CÔNG VĂN", "TỜ TRÌNH"]


def random_date() -> tuple[str, str]:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2021, 2026)
    fmt = random.choice([
        f"ngày {day:02d} tháng {month:02d} năm {year}",
        f"{day:02d}/{month:02d}/{year}",
        f"ngày {day} tháng {month} năm {year}"
    ])
    iso = f"{year}-{month:02d}-{day:02d}"
    return fmt, iso


def random_money() -> tuple[str, int]:
    val = random.randint(5, 500) * 1_000_000
    fmt = f"{val:,.0f} VNĐ".replace(",", ".")
    return fmt, val


def random_tax_id() -> str:
    if random.random() < 0.7:
        return f"{random.randint(1000000000, 9999999999):010d}"
    return f"{random.randint(1000000000, 9999999999):010d}-{random.randint(1, 999):03d}"


# ==============================================================================
# KEY ALIASES (FOR TRAINING OPEN-SCHEMA VARIATIONS)
# ==============================================================================

KEY_ALIASES = {
    "so_hieu": ["so_hieu", "số_hiệu", "so_van_ban", "document_id", "so_cv"],
    "co_quan_ban_hanh": ["co_quan_ban_hanh", "cơ_quan_ban_hành", "don_vi_ban_hanh", "issuer"],
    "ngay_ban_hanh": ["ngay_ban_hanh", "ngày_ban_hành", "ngay_ky", "issue_date"],
    "tong_tien_thanh_toan": ["tong_tien_thanh_toan", "tổng_tiền", "tong_tien", "total_amount"],
    "ma_so_thue_doanh_nghiep": ["ma_so_thue", "mã_số_thuế", "mst", "tax_code"],
    "so_hop_dong": ["so_hop_dong", "số_hợp_đồng", "contract_number"],
    "ho_va_ten": ["ho_va_ten", "họ_và_tên", "full_name", "ho_ten"]
}


# ==============================================================================
# GENERATORS PER DOCUMENT CATEGORY (VALUE + SEMANTIC DESCRIPTION)
# ==============================================================================

def gen_administrative_doc():
    org = random.choice(ORGS_GOV)
    doctype = random.choice(DOC_TYPES_GOV)
    year = random.randint(2021, 2026)
    num = random.randint(1, 350)
    suffix = random.choice(["NĐ-CP", "TT-BTC", "QĐ-UBND", "CV-TCT", "TT-BKHĐT"])
    so_hieu = f"{num}/{year}/{suffix}"
    city = random.choice(CITIES)
    date_str, _ = random_date()
    signer = random.choice(NAMES)
    position = random.choice(POSITIONS)
    position_render = position.upper()
    
    topics = [
        "Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Thuế giá trị gia tăng",
        "Về việc phê duyệt kế hoạch chuyển đổi số và ứng dụng trí tuệ nhân tạo năm 2026",
        "Ban hành quy chế quản lý, vận hành và khai thác hệ thống dữ liệu quốc gia",
        "Quy định về quản lý thu phí, lệ phí đối với hàng hóa xuất nhập khẩu tại cảng biển",
        "Về việc điều động, bổ nhiệm cán bộ giữ chức vụ lãnh đạo, quản lý"
    ]
    trich_yeu = random.choice(topics)

    present = {
        "so_hieu": (so_hieu, "Số hiệu chính thức của văn bản, công văn, nghị định hoặc thông tư"),
        "co_quan_ban_hanh": (org, "Tên cơ quan, tổ chức hoặc bộ ngành ban hành văn bản"),
        "loai_van_ban": (doctype, "Thể loại văn bản hành chính (Nghị định, Thông tư, Quyết định...)"),
        "ngay_ban_hanh": (date_str, "Ngày tháng năm ban hành hoặc ký văn bản"),
        "dia_diem": (city, "Địa danh hoặc thành phố nơi ban hành văn bản"),
        "nguoi_ky": (signer, "Họ và tên người có thẩm quyền ký văn bản"),
        "chuc_vu_nguoi_ky": (position_render, "Chức vụ của người ký văn bản"),
        "trich_yeu_noi_dung": (trich_yeu, "Nội dung trích yếu hoặc chủ đề chính của văn bản")
    }

    traps = {
        "tong_tien_thanh_toan": (None, "Tổng số tiền thanh toán cuối cùng đã gồm thuế VAT"),
        "ma_so_thue_doanh_nghiep": (None, "Mã số thuế của doanh nghiệp hoặc đơn vị kinh doanh"),
        "so_tai_khoan_ngan_hang": (None, "Số tài khoản ngân hàng để chuyển khoản"),
        "thoi_han_bao_hanh": (None, "Thời hạn bảo hành thiết bị hoặc dịch vụ"),
        "bien_so_xe": (None, "Biển số đăng ký của phương tiện giao thông"),
        "muc_luong_co_ban": (None, "Mức lương cơ bản thỏa thuận hàng tháng")
    }

    markdown = f"""# {org}
**Số: {so_hieu}**
*{city}, {date_str}*

## {doctype}
### {trich_yeu}

**Căn cứ Luật Tổ chức Chính phủ số 76/2015/QH13;**
**Căn cứ các Nghị định và quy định hiện hành có liên quan;**

**QUYẾT NGHỊ:**
- Điều 1: Ban hành kèm theo {doctype} này danh mục hướng dẫn áp dụng thống nhất trên toàn quốc.
- Điều 2: Các cơ quan, tổ chức, doanh nghiệp và cá nhân có liên quan chịu trách nhiệm thi hành.
- Điều 3: {doctype} này có hiệu lực thi hành kể từ ngày ký.

**Nơi nhận:**
- Thủ tướng, các Phó Thủ tướng;
- Các Bộ, cơ quan ngang Bộ;
- Lưu: VT, TH.

**{position_render}**
*(Đã ký)*
**{signer}**
"""
    return markdown, present, traps


def gen_commercial_contract():
    corp_a = random.choice(ORGS_CORP)
    corp_b = random.choice(ORGS_CORP)
    while corp_b == corp_a:
        corp_b = random.choice(ORGS_CORP)
    
    rep_a = random.choice(NAMES)
    rep_b = random.choice(NAMES)
    year = random.randint(2023, 2026)
    contract_num = f"HĐKT-{random.randint(100, 999)}/{year}"
    date_str, _ = random_date()
    money_str, _ = random_money()
    mst_a = random_tax_id()
    mst_b = random_tax_id()
    duration = f"{random.randint(3, 24)} tháng"

    present = {
        "so_hop_dong": (contract_num, "Số hiệu định danh của bản hợp đồng kinh tế"),
        "ben_a": (corp_a, "Tên công ty hoặc tổ chức đại diện cho Bên A (Bên giao việc)"),
        "nguoi_dai_dien_ben_a": (rep_a, "Họ tên người đại diện ký hợp đồng của Bên A"),
        "ma_so_thue_ben_a": (mst_a, "Mã số thuế của công ty Bên A"),
        "ben_b": (corp_b, "Tên công ty hoặc tổ chức đại diện cho Bên B (Bên thực hiện)"),
        "nguoi_dai_dien_ben_b": (rep_b, "Họ tên người đại diện ký hợp đồng của Bên B"),
        "ma_so_thue_ben_b": (mst_b, "Mã số thuế của công ty Bên B"),
        "gia_tri_hop_dong": (money_str, "Tổng giá trị thanh toán của hợp đồng kinh tế"),
        "ngay_ky_hop_dong": (date_str, "Ngày tháng năm các bên ký kết hợp đồng"),
        "thoi_han_thuc_hien": (duration, "Thời hạn hoặc thời gian hoàn thành hợp đồng")
    }

    traps = {
        "so_hieu_thong_tu": (None, "Số hiệu thông tư hoặc nghị định pháp luật"),
        "so_chung_minh_nhan_dan": (None, "Số CMND hoặc thẻ căn cước công dân"),
        "diem_trung_binh_tot_nghiep": (None, "Điểm trung bình tốt nghiệp GPA đại học"),
        "lai_suat_vay_ngan_hang": (None, "Mức lãi suất vay vốn hàng năm của ngân hàng"),
        "bien_ban_nghiem_thu": (None, "Số hiệu biên bản nghiệm thu công trình")
    }

    markdown = f"""# CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
## Độc lập - Tự do - Hạnh phúc

# HỢP ĐỒNG KINH TẾ
**Số: {contract_num}**
*Hôm nay, {date_str}, tại văn phòng hai bên tiến hành ký kết hợp đồng:*

### BÊN A (BÊN GIAO VIỆC):
- Tên đơn vị: **{corp_a}**
- Mã số thuế: **{mst_a}**
- Đại diện: Ông/Bà **{rep_a}** - Chức vụ: Giám đốc

### BÊN B (BÊN THỰC HIỆN):
- Tên đơn vị: **{corp_b}**
- Mã số thuế: **{mst_b}**
- Đại diện: Ông/Bà **{rep_b}** - Chức vụ: Đại diện theo pháp luật

**ĐIỀU KHOẢN CHÍNH:**
1. **Nội dung công việc**: Bên B nhận cung cấp giải pháp chuyển đổi số cho Bên A.
2. **Giá trị hợp đồng**: Tổng giá trị thanh toán là **{money_str}** (đã bao gồm thuế GTGT).
3. **Thời hạn thực hiện**: Trong thời gian **{duration}** kể từ ngày hợp đồng có hiệu lực.

| Đại diện Bên A | Đại diện Bên B |
| :---: | :---: |
| *(Đã ký và đóng dấu)* | *(Đã ký và đóng dấu)* |
| **{rep_a}** | **{rep_b}** |
"""
    return markdown, present, traps


def gen_financial_invoice():
    seller = random.choice(ORGS_CORP)
    buyer = random.choice(ORGS_CORP)
    while buyer == seller:
        buyer = random.choice(ORGS_CORP)
    
    invoice_num = f"{random.randint(1, 999999):07d}"
    ky_hieu = f"1C{random.randint(23, 26)}TAA"
    date_str, _ = random_date()
    mst_seller = random_tax_id()
    mst_buyer = random_tax_id()
    money_str, _ = random_money()
    bank_acc = f"{random.randint(1000000000, 999999999999):d}"
    bank_name = "Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank)"

    present = {
        "so_hoa_don": (invoice_num, "Số hóa đơn điện tử giá trị gia tăng"),
        "ky_hieu_hoa_don": (ky_hieu, "Ký hiệu mẫu số hoặc series của hóa đơn"),
        "don_vi_ban_hang": (seller, "Tên công ty hoặc đơn vị bán hàng hóa/dịch vụ"),
        "ma_so_thue_nguoi_ban": (mst_seller, "Mã số thuế của công ty bán hàng"),
        "don_vi_mua_hang": (buyer, "Tên công ty hoặc khách hàng mua hàng"),
        "ma_so_thue_nguoi_mua": (mst_buyer, "Mã số thuế của đơn vị mua hàng"),
        "tong_tien_thanh_toan": (money_str, "Tổng cộng tiền thanh toán cuối cùng của hóa đơn"),
        "ngay_lap_hoa_don": (date_str, "Ngày tháng năm phát hành hoặc lập hóa đơn"),
        "so_tai_khoan_ngan_hang": (bank_acc, "Số tài khoản ngân hàng của đơn vị thụ hưởng")
    }

    traps = {
        "so_hieu_nghi_dinh": (None, "Số hiệu nghị định của Chính phủ"),
        "nguoi_giam_ho": (None, "Họ và tên của người giám hộ hợp pháp"),
        "so_so_bao_hiem_xa_hoi": (None, "Số sổ bảo hiểm xã hội cá nhân"),
        "muc_an_hinh_su": (None, "Mức án hình phạt tù hoặc phạt tiền của tòa án"),
        "thoi_han_bao_hanh": (None, "Thời hạn bảo hành thiết bị")
    }

    markdown = f"""# HÓA ĐƠN GIÁ TRỊ GIA TĂNG (VAT)
**Ký hiệu: {ky_hieu}** | **Số: {invoice_num}**
*Ngày lập: {date_str}*

### ĐƠN VỊ BÁN HÀNG:
- Tên công ty: **{seller}**
- Mã số thuế: **{mst_seller}**
- Số tài khoản: **{bank_acc}** tại {bank_name}

### ĐƠN VỊ MUA HÀNG:
- Tên công ty: **{buyer}**
- Mã số thuế: **{mst_buyer}**
- Hình thức thanh toán: Chuyển khoản

| STT | Tên hàng hóa, dịch vụ | ĐVT | Số lượng | Đơn giá | Thành tiền |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Dịch vụ phần mềm quản lý tài liệu | Gói | 1 | {money_str} | {money_str} |

- **Tổng cộng tiền thanh toán:** **{money_str}**
- Người bán hàng: *(Đã ký điện tử hợp lệ)*
"""
    return markdown, present, traps


def gen_hr_resume():
    name = random.choice(NAMES)
    gender = random.choice(["Nam", "Nữ"])
    birth_year = random.randint(1985, 2002)
    birth_day = random.randint(1, 28)
    birth_month = random.randint(1, 12)
    dob = f"{birth_day:02d}/{birth_month:02d}/{birth_year}"
    cccd = f"{random.randint(100000000000, 999999999999):012d}"
    hometown = random.choice(CITIES)
    phone = f"09{random.randint(10000000, 99999999):08d}"
    email = f"{name.lower().replace(' ', '')}{birth_year}@gmail.com"
    position_apply = random.choice(["Kỹ sư AI/ML", "Chuyên viên Kế toán", "Trưởng phòng Kinh doanh", "Nhân viên Pháp chế"])
    school = random.choice(["Đại học Bách Khoa Hà Nội", "Đại học Quốc gia TP.HCM", "Đại học Kinh tế Quốc dân", "Đại học Ngoại thương"])

    present = {
        "ho_va_ten": (name, "Họ và tên đầy đủ của người ứng tuyển / cá nhân"),
        "gioi_tinh": (gender, "Giới tính sinh học (Nam hoặc Nữ)"),
        "ngay_thang_nam_sinh": (dob, "Ngày tháng năm sinh của ứng viên"),
        "so_cccd": (cccd, "Số thẻ căn cước công dân hoặc số định danh cá nhân"),
        "que_quan": (hometown, "Quê quán hoặc nơi sinh theo giấy tờ"),
        "so_dien_thoai": (phone, "Số điện thoại di động liên hệ"),
        "email_lien_he": (email, "Địa chỉ thư điện tử email cá nhân"),
        "vi_tri_ung_tuyen": (position_apply, "Chức danh hoặc vị trí công việc muốn ứng tuyển"),
        "truong_dao_tao": (school, "Tên trường đại học hoặc cơ sở giáo dục đã đào tạo")
    }

    traps = {
        "ma_so_thue_doanh_nghiep": (None, "Mã số thuế doanh nghiệp của công ty"),
        "so_hieu_thong_tu": (None, "Số hiệu thông tư pháp luật"),
        "tong_gia_tri_hop_dong": (None, "Tổng giá trị hợp đồng kinh tế"),
        "bien_so_xe_may": (None, "Biển số đăng ký xe máy cá nhân"),
        "lai_suat_tiet_kiem": (None, "Lãi suất tiền gửi tiết kiệm có kỳ hạn")
    }

    markdown = f"""# SƠ YẾU LÝ LỊCH CÁ NHÂN

### THÔNG TIN CƠ BẢN:
- Họ và tên: **{name}**
- Giới tính: **{gender}**
- Ngày sinh: **{dob}**
- Số CCCD/Định danh: **{cccd}**
- Quê quán: **{hometown}**
- Số điện thoại: **{phone}**
- Email: **{email}**

### VỊ TRÍ ỨNG TUYỂN:
- Mục tiêu nghề nghiệp: **{position_apply}**
- Trình độ học vấn: Cử nhân tại **{school}**
- Kinh nghiệm làm việc: Trên 3 năm kinh nghiệm trong lĩnh vực liên quan.

Tôi xin cam đoan các thông tin khai trên là hoàn toàn đúng sự thật.
Người khai: **{name}**
"""
    return markdown, present, traps


def gen_academic_report():
    title = random.choice([
        "Nghiên cứu ứng dụng mạng nơ-ron học sâu trong số hóa văn bản hành chính Việt Nam",
        "Tối ưu hóa thuật toán nhận diện chữ quang học OCR phục vụ chính phủ điện tử",
        "Khảo sát hiệu năng trích xuất thông tin cấu trúc dựa trên mô hình ngôn ngữ nhỏ",
        "Phát triển hệ thống phân loại và lập chỉ mục văn bản lưu trữ tự động"
    ])
    author = random.choice(NAMES)
    supervisor = random.choice(NAMES)
    org = random.choice(["Đại học Bách Khoa Hà Nội", "Viện Công nghệ Thông tin - Viện Hàn lâm KH&CN Việt Nam", "Đại học Công nghệ - ĐHQGHN"])
    date_str, _ = random_date()
    code = f"ĐT-NCKH-{random.randint(2022, 2026)}/{random.randint(10, 99)}"

    present = {
        "ten_de_tai": (title, "Tên chính thức của đề tài hoặc dự án nghiên cứu khoa học"),
        "ma_so_de_tai": (code, "Mã số quản lý đề tài khoa học"),
        "chu_nhiem_de_tai": (author, "Họ tên chủ nhiệm hoặc tác giả chính của đề tài"),
        "can_bo_huong_dan": (supervisor, "Họ tên người hướng dẫn hoặc cố vấn khoa học"),
        "co_quan_chu_tri": (org, "Tên trường viện hoặc cơ quan chủ trì nghiên cứu"),
        "ngay_bao_cao": (date_str, "Ngày tháng năm nghiệm thu hoặc báo cáo đề tài")
    }

    traps = {
        "so_tai_khoan_ngan_hang": (None, "Số tài khoản ngân hàng thụ hưởng"),
        "tong_tien_thue_vat": (None, "Tiền thuế GTGT phát sinh"),
        "so_hoa_don": (None, "Số hóa đơn tài chính"),
        "bien_so_xe": (None, "Biển số xe công vụ"),
        "nguoi_dai_dien_phap_luat": (None, "Người đại diện theo pháp luật của công ty")
    }

    markdown = f"""# BÁO CÁO KẾT QUẢ ĐỀ TÀI NGHIÊN CỨU KHOA HỌC
**Mã số đề tài: {code}**
*Báo cáo ngày: {date_str}*

## Tên đề tài:
### {title}

- Đơn vị chủ trì: **{org}**
- Chủ nhiệm đề tài: ThS/TS. **{author}**
- Cán bộ hướng dẫn / Cố vấn: PGS.TS. **{supervisor}**

**TÓM TẮT KẾT QUẢ:**
Đề tài đã hoàn thành các mục tiêu nghiên cứu đề ra, xây dựng thành công pipeline trích xuất thông tin với độ trễ thấp và độ chính xác cao.
"""
    return markdown, present, traps


GENERATORS = [
    gen_administrative_doc,
    gen_commercial_contract,
    gen_financial_invoice,
    gen_hr_resume,
    gen_academic_report
]


SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia trích xuất thông tin có cấu trúc từ văn bản tài liệu sang định dạng JSON.
Nhiệm vụ của bạn là:
1. Nhận văn bản Markdown và Schema yêu cầu (dưới dạng danh sách các trường hoặc từ điển mô tả ngữ nghĩa của từng trường).
2. Trích xuất CHÍNH XÁC giá trị tương ứng cho từng trường từ văn bản.
3. QUY TẮC BẮT BUỘC (Trained Abstention): Nếu một trường trong Schema KHÔNG CÓ thông tin trong văn bản, BẮT BUỘC gán giá trị của trường đó là null. Tuyệt đối không được đoán mò hoặc bịa đặt giá trị.
4. Đầu ra CHỈ LÀ một khối JSON hợp lệ duy nhất khớp đúng các key được yêu cầu, không kèm bất kỳ lời giải thích nào khác."""


def build_sample(seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)
    
    gen_fn = random.choice(GENERATORS)
    markdown, present, traps = gen_fn()

    # Sample schema: 3-5 present fields and 2-3 trap fields
    n_present = min(len(present), random.randint(3, 5))
    n_traps = min(len(traps), random.randint(2, 3))

    selected_present_keys = random.sample(list(present.keys()), n_present)
    selected_trap_keys = random.sample(list(traps.keys()), n_traps)

    all_raw_keys = selected_present_keys + selected_trap_keys
    random.shuffle(all_raw_keys)

    # Choose alias variant for keys to train semantic flexibility
    target_json = {}
    schema_with_desc = {}
    schema_keys_only = []

    for k in all_raw_keys:
        # 30% chance to pick an alias if available
        alias_k = k
        if k in KEY_ALIASES and random.random() < 0.35:
            alias_k = random.choice(KEY_ALIASES[k])

        if k in present:
            val, desc = present[k]
            assert str(val) in markdown, f"Verification failed: '{val}' not in markdown!"
            target_json[alias_k] = val
            schema_with_desc[alias_k] = desc
        else:
            val, desc = traps[k]
            target_json[alias_k] = None
            schema_with_desc[alias_k] = desc
        
        schema_keys_only.append(alias_k)

    # 55% chance: Schema is a rich Dictionary with Descriptions!
    # 45% chance: Schema is a simple list of Keys!
    use_descriptions = random.random() < 0.55
    if use_descriptions:
        schema_rendered = json.dumps(schema_with_desc, ensure_ascii=False, indent=2)
        schema_instr = "Schema yêu cầu trích xuất (kèm mô tả ngữ nghĩa cho từng trường):"
    else:
        schema_rendered = json.dumps(schema_keys_only, ensure_ascii=False, indent=2)
        schema_instr = "Schema yêu cầu trích xuất (danh sách trường):"

    user_prompt = f"""### Văn bản tài liệu:
```markdown
{markdown.strip()}
```

### {schema_instr}
```json
{schema_rendered}
```

Hãy trích xuất thông tin và trả về JSON theo đúng các key được yêu cầu. Trường nào không có thông tin hãy để giá trị là null."""

    assistant_response = json.dumps(target_json, ensure_ascii=False, indent=2)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ],
        "metadata": {
            "num_fields": len(all_raw_keys),
            "num_present": n_present,
            "num_null": n_traps,
            "has_descriptions": use_descriptions
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("work/schema_dataset"))
    parser.add_argument("--train-samples", type=int, default=2500)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    splits = {
        "train": args.train_samples,
        "validation": args.val_samples,
        "test": args.test_samples
    }

    print("Generating dual-mode schema extraction dataset (with descriptions & abstention)...")
    total_stats = {"total": 0, "present_fields": 0, "null_fields": 0, "with_descriptions": 0}

    for split_name, count in splits.items():
        out_file = args.out_dir / f"{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for i in range(count):
                sample = build_sample()
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_stats["total"] += 1
                total_stats["present_fields"] += sample["metadata"]["num_present"]
                total_stats["null_fields"] += sample["metadata"]["num_null"]
                total_stats["with_descriptions"] += int(sample["metadata"]["has_descriptions"])
        print(f"  -> Generated {count:,} samples in {out_file}")

    stats_file = args.out_dir / "stats.json"
    stats_file.write_text(json.dumps({
        "splits": splits,
        "total_samples": total_stats["total"],
        "avg_fields_per_sample": round((total_stats["present_fields"] + total_stats["null_fields"]) / total_stats["total"], 2),
        "present_fields_ratio": round(total_stats["present_fields"] / (total_stats["present_fields"] + total_stats["null_fields"]), 3),
        "null_abstention_ratio": round(total_stats["null_fields"] / (total_stats["present_fields"] + total_stats["null_fields"]), 3),
        "with_descriptions_ratio": round(total_stats["with_descriptions"] / total_stats["total"], 3)
    }, indent=2, ensure_ascii=False))

    print(f"\nCompleted! Total: {total_stats['total']:,} samples.")
    print(f"Stats written to {stats_file}")


if __name__ == "__main__":
    main()
