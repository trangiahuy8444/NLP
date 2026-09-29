"""
Module tiền xử lý văn bản tiếng Việt cho bài toán phân loại cảm xúc.
Các bước:
1. Chuẩn hóa Unicode (NFC)
2. Chuyển chữ thường (lowercase)
3. Chuẩn hóa dấu thanh và từ ngữ viết tắt phổ biến
4. Loại bỏ ký tự đặc biệt, dấu câu thừa
5. Tách từ (Tokenization)
"""

import re
import unicodedata

# Bảng chuẩn hóa một số từ viết tắt / teencode thường gặp trong đánh giá sản phẩm
NORMALIZATION_DICT = {
    "k": "không",
    "ko": "không",
    "khong": "không",
    "hok": "không",
    "hẻo": "yếu",
    "đc": "được",
    "dc": "được",
    "duoc": "được",
    "oke": "tốt",
    "ok": "tốt",
    "oki": "tốt",
    "sp": "sản phẩm",
    "cam": "camera",
    "tg": "thời gian",
    "dt": "điện thoại",
    "đt": "điện thoại",
    "ms": "mới",
    "tks": "cảm ơn",
    "thks": "cảm ơn",
    "shop": "cửa hàng",
    "vcl": "rất",
    "đt": "điện thoại"
}

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng dựng sẵn NFC."""
    return unicodedata.normalize("NFC", text)

def clean_text(text: str) -> str:
    """Làm sạch văn bản: chuẩn hóa, loại bỏ ký tự lạ, chuẩn hóa khoảng trắng."""
    if not isinstance(text, str):
        return ""
    
    text = normalize_unicode(text)
    text = text.lower()
    
    # Loại bỏ URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Chuẩn hóa các ký tự lặp (ví dụ: 'sướnggggg' -> 'sướng', 'ngonnnn' -> 'ngon')
    text = re.sub(r"([a-zà-ỹ])\1{2,}", r"\1", text)
    
    # Loại bỏ dấu câu đặc biệt, giữ lại các chữ cái tiếng Việt và khoảng trắng
    text = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]", " ", text)
    
    # Thay thế từ viết tắt
    tokens = text.split()
    tokens = [NORMALIZATION_DICT.get(token, token) for token in tokens]
    
    return " ".join(tokens)

def tokenize(text: str) -> list[str]:
    """Tách câu thành danh sách các token (từ/tiếng)."""
    cleaned = clean_text(text)
    return cleaned.split()

if __name__ == "__main__":
    sample = "Máy xài quá okelaaaa! Pin trâu vcl, dùng cả ngày k hết pin :)))"
    print("Gốc:", sample)
    print("Sau làm sạch:", clean_text(sample))
    print("Tokens:", tokenize(sample))
