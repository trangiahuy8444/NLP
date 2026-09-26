"""
Giao diện Trải nghiệm Streamlit (Streamlit Web Application)
Dành cho người dùng muốn khởi chạy qua Streamlit:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
import pandas as pd

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    import streamlit as st
except ImportError:
    print("Vui lòng cài đặt streamlit: pip install streamlit")
    sys.exit(1)

from inference_engine import (
    SentimentInferenceEngine,
    MODEL_DISPLAY_NAMES,
    MODEL_METRICS_INFO,
    SAMPLE_QUERIES
)

st.set_page_config(
    page_title="HCMUE - Phân Tích Cảm Xúc Đa Miền",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho bộ nhận diện thương hiệu Trường ĐH Sư Phạm TP.HCM
st.markdown("""
<style>
    /* Theme màu thương hiệu HCMUE */
    :root {
        --cerulean: #124874;
        --jasper: #CF373D;
    }
    .stAppHeader {
        border-top: 4px solid #124874;
    }
    h1, h2, h3 {
        color: #124874 !important;
    }
    .stButton>button {
        background-color: #124874 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        background-color: #0f3d63 !important;
        box-shadow: 0 4px 12px rgba(18, 72, 116, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    models_dir = os.path.join(BASE_DIR, "models")
    return SentimentInferenceEngine(models_dir=models_dir)

engine = load_engine()

# Sidebar
logo_path = os.path.join(BASE_DIR, "web_app", "assets", "logo_hcmue_moet.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_column_width=True)

st.sidebar.markdown("""
<div style="background-color: #f0f5fa; padding: 8px; border-radius: 8px; border: 1px solid #b8d4e6; margin-bottom: 12px; font-size: 11px;">
    <strong>Trường ĐH Sư phạm TP.HCM</strong><br>
    Khoa Khoa Học Máy Tính • Khoá 36
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🛠️ Thiết Lập Trải Nghiệm")
dataset = st.sidebar.selectbox(
    "1. Chọn Tập Dữ Liệu Thực Nghiệm (Domain)",
    options=["UIT-VSFC", "E-Commerce"],
    index=0,
    help="UIT-VSFC: Giáo dục / Khảo sát sinh viên. E-Commerce: Đánh giá mua sắm trực tuyến."
)

mode = st.sidebar.radio(
    "2. Chế Độ Suy Luận",
    options=["So sánh đồng thời cả 6 mô hình", "Trải nghiệm chi tiết 1 mô hình"],
    index=0
)

selected_model_id = "bilstm_attention"
if mode == "Trải nghiệm chi tiết 1 mô hình":
    selected_model_id = st.sidebar.selectbox(
        "Chọn mô hình:",
        options=list(MODEL_DISPLAY_NAMES.keys()),
        format_func=lambda x: MODEL_DISPLAY_NAMES[x],
        index=4
    )

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Thiết bị tính toán:** `{engine.device}`")

# Header chính
st.title("⚡ Hệ Thống Phân Tích Cảm Xúc Tiếng Việt Đa Miền")
st.markdown(
    "Ứng dụng thực tế đối chuẩn và so sánh hiệu năng **6 mô hình NLP**: "
    "*TF-IDF, Average Word2Vec, BiLSTM Scratch, BiLSTM Pretrained, BiLSTM + Self-Attention, PhoBERT Base v2* "
    "trên 2 miền dữ liệu thực tế."
)

# Mẫu câu thử nghiệm
st.subheader("💡 Mẫu Câu Thử Nghiệm Nhanh")
samples = SAMPLE_QUERIES.get(dataset, [])
sample_cols = st.columns(len(samples))
default_text = "Thầy dạy rất nhiệt tình, bài giảng dễ hiểu và hỗ trợ sinh viên chu đáo."

if "input_text" not in st.session_state:
    st.session_state["input_text"] = default_text

for i, s in enumerate(samples):
    with sample_cols[i]:
        if st.button(f"📌 {s['category']}", key=f"sample_{i}"):
            st.session_state["input_text"] = s["text"]

# Input text area
input_text = st.text_area(
    "Nhập văn bản cần phân tích cảm xúc:",
    value=st.session_state.get("input_text", default_text),
    height=100
)

col_act1, col_act2 = st.columns([1, 5])
with col_act1:
    btn_predict = st.button("🚀 Phân Tích Ngay", type="primary", use_container_width=True)

if btn_predict and input_text.strip():
    text = input_text.strip()

    if mode == "So sánh đồng thời cả 6 mô hình":
        with st.spinner("Đang chạy suy luận qua cả 6 mô hình..."):
            res = engine.predict_all(dataset, text)

        results = res["results"]
        pos_count = sum(1 for r in results if r.get("sentiment") == 1)
        neg_count = sum(1 for r in results if r.get("sentiment") == 0)

        # Consensus Box
        if pos_count >= neg_count:
            st.success(f"### 🎉 Kết Luận Đồng Thuận: TÍCH CỰC ({pos_count}/{len(results)} mô hình đồng tình)")
        else:
            st.error(f"### ⚠️ Kết Luận Đồng Thuận: TIÊU CỰC ({neg_count}/{len(results)} mô hình đồng tình)")

        # Hiển thị kết quả 6 mô hình
        cols = st.columns(3)
        for idx, r in enumerate(results):
            col = cols[idx % 3]
            with col:
                is_pos = r.get("sentiment") == 1
                color = "green" if is_pos else "red"
                with st.container():
                    st.markdown(f"#### {r['model_name']}")
                    st.markdown(f"**Nhãn:** :{color}[**{r['label']}**] | **Độ tin cậy:** `{r['confidence']}%`")
                    st.progress(r['confidence'] / 100.0)
                    st.caption(f"Độ trễ: `{r['latency_ms']} ms` | Tiêu cực: `{r['probabilities']['Tiêu cực']}%` - Tích cực: `{r['probabilities']['Tích cực']}%`")
                    st.markdown("---")

        # Hiển thị Attention Heatmap nếu có
        attn_result = next((r for r in results if r.get("attention")), None)
        if attn_result and attn_result.get("attention"):
            st.subheader("🔍 Bản Đồ Trọng Số Chú Ý (BiLSTM + Self-Attention)")
            st.write("Mức độ quan trọng ngữ nghĩa của từng từ vựng trong câu:")
            tokens_html = ""
            for item in attn_result["attention"]:
                w = item["weight"]
                alpha = min(1.0, w * 3.5)
                bg_color = f"rgba(99, 102, 241, {max(0.1, alpha)})"
                text_color = "white" if alpha > 0.6 else "#1e1b4b"
                tokens_html += f"<span style='background-color:{bg_color}; color:{text_color}; padding:4px 8px; margin:3px; border-radius:6px; display:inline-block; font-weight:600; font-size:14px;' title='Weight: {w:.4f}'>{item['token']} <span style='font-size:10px; opacity:0.8;'>({w*100:.1f}%)</span></span> "
            st.markdown(tokens_html, unsafe_allow_html=True)

    else:
        # Chế độ 1 mô hình
        with st.spinner(f"Đang suy luận qua {MODEL_DISPLAY_NAMES[selected_model_id]}..."):
            res = engine.predict_single(dataset, selected_model_id, text)

        is_pos = res.get("sentiment") == 1
        color = "green" if is_pos else "red"
        st.markdown(f"### Dự đoán: :{color}[**{res['label']}**] (Độ tin cậy: `{res['confidence']}%`)")
        st.progress(res['confidence'] / 100.0)
        st.write(f"⏱️ **Thời gian suy luận:** `{res['latency_ms']} ms`")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Xác suất Tiêu cực", f"{res['probabilities']['Tiêu cực']}%")
        with col2:
            st.metric("Xác suất Tích cực", f"{res['probabilities']['Tích cực']}%")

        if res.get("attention"):
            st.subheader("🔍 Phân Bổ Chú Ý (Self-Attention Weights)")
            tokens_html = ""
            for item in res["attention"]:
                w = item["weight"]
                alpha = min(1.0, w * 3.5)
                bg_color = f"rgba(99, 102, 241, {max(0.1, alpha)})"
                text_color = "white" if alpha > 0.6 else "#1e1b4b"
                tokens_html += f"<span style='background-color:{bg_color}; color:{text_color}; padding:4px 8px; margin:3px; border-radius:6px; display:inline-block; font-weight:600; font-size:14px;' title='Weight: {w:.4f}'>{item['token']} <span style='font-size:10px; opacity:0.8;'>({w*100:.1f}%)</span></span> "
            st.markdown(tokens_html, unsafe_allow_html=True)

# Bảng xếp hạng Benchmark
st.markdown("---")
st.subheader("📊 Bảng Đối Chuẩn Hiệu Năng Nghiên Cứu (Test Set)")
metrics = MODEL_METRICS_INFO.get(dataset, {})
df_metrics = pd.DataFrame([
    {
        "Mã": k,
        "Mô Hình": MODEL_DISPLAY_NAMES[k],
        "Accuracy (%)": v["acc"],
        "Macro F1 (%)": v["f1"],
        "Mô tả": v["desc"]
    }
    for k, v in metrics.items()
])
st.dataframe(df_metrics, use_container_width=True, hide_index=True)
