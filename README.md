# 🎓 HỌC PHẦN: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NATURAL LANGUAGE PROCESSING)
### TRƯỜNG ĐẠI HỌC SƯ PHẠM THÀNH PHỐ HỒ CHÍ MINH (HCMUE) — KHOA KHOA HỌC MÁY TÍNH
**Lớp:** Cao học Khoa học Máy tính — Khóa 36 (2025 – 2027)  
**Giảng viên hướng dẫn:** **TS. LÊ ANH CƯỜNG**  
**Nhóm học viên thực hiện:**  
1. **Trần Gia Huy** — MSSV: **KHMT836012**  
2. **Đỗ Minh Khánh Ngân** — MSSV: **KHMT836019**  
3. **Nguyễn Tấn Phát** — MSSV: **KHMT836026**  

---

## 🌐 GITHUB PAGES INTERACTIVE STUDIO
👉 **Trang báo cáo & Dashboard trực quan hóa tương tác:**  
🔗 **[https://trangiahuy8444.github.io/NLP/](https://trangiahuy8444.github.io/NLP/)**

*(Gồm đầy đủ Logo HCMUE, biểu đồ t-SNE 2D/3D, ma trận nhầm lẫn Confusion Matrix, đường cong học Loss/Acc và bản đồ nhiệt Attention Heatmap giải thích từ khóa)*

---

## 📂 TỔ CHỨC CẤU TRÚC THƯ MỤC REPOSITORY

```text
NLP/
├── README.md                           # 📖 Trang chủ hướng dẫn tổng quan toàn bộ repository
├── requirements.txt                    # 📋 Danh mục các thư viện Python (PyTorch, Transformers, Gensim, Pyvi)
├── .gitignore                          # 🛡️ Cấu hình loại bỏ file rác, file zip và weights >100MB
│
├── docs/                               # 🌐 Website GitHub Pages trực quan hóa kết quả (Live Dashboard)
│   ├── index.html                      # Trang Dashboard chính tích hợp logo & theme HCMUE
│   ├── style.css                       # Giao diện CSS hiện đại
│   ├── app.js & data.js                # Dữ liệu & logic tương tác bộ lọc kết quả
│   ├── assets/                         # Logo trường, huy hiệu HCMUE
│   └── results/                        # Toàn bộ hình ảnh biểu đồ t-SNE, Confusion Matrix
│
├── Bai_tap_giua_ky/                    # 🎓 ĐỒ ÁN TIỂU LUẬN GIỮA KỲ (Trọng tâm học phần)
│   ├── Bao_Cao_Tieu_Luan_Giua_Ky_NLP.pdf # ⭐ Bài báo cáo tiểu luận chuẩn IEEE/ACM (29 trang, clickable TOC)
│   ├── BAO_CAO_GIUA_KY.md              # Báo cáo Markdown chi tiết song song
│   ├── Thuc_Hanh_Word2Vec_LSTM.ipynb   # Jupyter Notebook thực nghiệm trực quan
│   ├── app.py                          # 🚀 Máy chủ Web App Demo suy luận thời gian thực
│   ├── streamlit_app.py                # Ứng dụng Streamlit tương tác
│   ├── run_app.sh                      # Script 1-click khởi chạy Web App trên macOS / Linux
│   ├── web_app/                        # Giao diện Web App SPA (Nhận diện thương hiệu HCMUE)
│   ├── src/                            # Toàn bộ mã nguồn thuật toán (Preprocess, BiLSTM, PhoBERT, Engine)
│   ├── data/                           # Dữ liệu UIT-VSFC (Giáo dục) & E-Commerce (TMĐT)
│   ├── models/                         # Trọng số các mô hình (<100MB: TF-IDF, Word2Vec, BiLSTM, Vocabs)
│   ├── results/                        # Biểu đồ đánh giá thực nghiệm độc lập
│   ├── tai_lieu_tham_khao/             # 10 bài báo khoa học PDF trích dẫn gốc
│   └── latex_project/                  # Trọn bộ mã nguồn LaTeX để biên dịch báo cáo
│
├── Bai_tap_ve_nha/                     # 📝 Bài tập về nhà: Mô hình ngôn ngữ N-gram
│   ├── BAI_LAM.md                      # Lời giải chi tiết
│   ├── Thuc_hanh_NGram.ipynb           # Notebook thực hành N-gram
│   └── ngram_language_model.py         # Code triển khai N-gram
│
├── IGM_N-gram/                         # 📝 Bài tập thực hành: Mô hình IGM & N-gram
│   └── Bai_ghi_Mo_hinh_ngon_ngu_N-gram.md
│
├── word2V/                             # 📝 Bài tập thực hành: Word2Vec cơ sở
│   └── main.py
│
└── HCMUE-NhanDienThuongHieu/           # 🏛️ Bộ nhận diện thương hiệu Trường Đại học Sư phạm TP.HCM
    ├── 1. Bo nhan dien/                # Quy chuẩn hướng dẫn sử dụng thương hiệu (PDF)
    ├── 2. File anh PNG/                # Logo chính thức, Logo kèm chữ MOET, Tòa nhà A
    └── Mau sac chu dao:                # Xanh Cerulean (#124874) & Đỏ Jasper (#CF373D)
```

---

## 📊 BẢNG TỔNG HỢP ĐỐI CHUẨN 6 MÔ HÌNH (TIỂU LUẬN GIỮA KỲ)

Thực nghiệm được thực hiện trên **02 tập dữ liệu thực tế độc lập** (mỗi tập 600 mẫu kiểm thử Test Set):

| # | Kiến trúc Mô hình | Phương pháp Biểu diễn | UIT-VSFC Acc | UIT-VSFC F1 | E-Com Acc | E-Com F1 | Nhận xét Khoa học |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| 1 | **TF-IDF + Logistic Reg.** | Bag-of-Words (1,500 feats) | 88.50% | 89.12% | 66.50% | 69.31% | Baseline thống kê truyền thống, mất ngữ cảnh |
| 2 | **Average Word2Vec + LR** | Word2Vec CBOW (100d) | 85.33% | 86.41% | 63.67% | 67.24% | Mất trật tự từ do trung bình cộng |
| 3 | **BiLSTM (Scratch)** | LSTM Hidden State (256d) | 87.67% | 88.35% | 68.33% | 70.45% | Tự học embedding từ ngẫu nhiên |
| 4 | **BiLSTM (Pre-trained W2V)** | Khởi tạo Word2Vec + BiLSTM | 91.83% | 92.14% | 70.50% | 72.88% | Hội tụ nhanh, tổng quát hóa tốt |
| 5 | **BiLSTM + Self-Attention** | *Đề xuất cải tiến của nhóm* | **93.17%** | **93.38%** | **74.17%** | **75.92%** | **Giải thích từ khóa XAI qua Attention Heatmap** |
| 6 | **PhoBERT Base v2** | *Transformer SOTA Tiếng Việt* | **95.83%** | **95.96%** | **78.50%** | **79.84%** | Hiểu sâu đa nghĩa và ngữ điệu phức tạp |

---

## 🚀 HƯỚNG DẪN CHẠY MÃ NGUỒN & WEB DEMO TẠI MÁY LOCAL

### 1. Cài đặt môi trường
```bash
conda activate ML
# Hoặc cài đặt các thư viện:
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng Dụng Web Trực Quan 1-Click (Giao diện Nhận diện HCMUE)
```bash
cd Bai_tap_giua_ky
./run_app.sh
# Hoặc: python app.py
```
👉 Truy cập trình duyệt tại: **`http://localhost:8501`**

### 3. Mở Jupyter Notebook nghiên cứu
```bash
cd Bai_tap_giua_ky
jupyter notebook Thuc_Hanh_Word2Vec_LSTM.ipynb
```

### 4. Chạy lại toàn bộ pipeline huấn luyện đối chuẩn 6 mô hình
```bash
cd Bai_tap_giua_ky
python src/run_experiments.py
```

---

## ⚖️ GIẤY PHÉP & BẢN QUYỀN
Dự án được thực hiện phục vụ mục đích học thuật và nghiên cứu trong khuôn khổ môn học **Xử lý ngôn ngữ tự nhiên** tại **Trường Đại học Sư phạm TP. Hồ Chí Minh (HCMUE)**.
