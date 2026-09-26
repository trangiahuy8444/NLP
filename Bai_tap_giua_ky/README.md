# BÀI TẬP GIỮA KỲ: MÔ HÌNH WORD2VEC, WORD EMBEDDING VÀ HỌC BIỂU DIỄN VĂN BẢN BẰNG LSTM CHO BÀI TOÁN PHÂN LOẠI CẢM XÚC

**Môn học:** Xử lý ngôn ngữ tự nhiên (Natural Language Processing)  
**Đơn vị:** Trường Đại học Sư phạm Thành phố Hồ Chí Minh -- Khoa Khoa học Máy tính (Khoá 36: 2025 - 2027)  
**Giảng viên hướng dẫn:** **TS. LÊ ANH CƯỜNG**  

---

### Danh sách nhóm sinh viên thực hiện:
1. **Trần Gia Huy** - MSSV: **KHMT836012**
2. **Đỗ Minh Khánh Ngân** - MSSV: **KHMT836019**
3. **Nguyễn Tấn Phát** - MSSV: **KHMT836026**

---

## 📂 CẤU TRÚC THƯ MỤC VÀ TÀI LIỆU

```text
Bao_Cao_Nhom_Giua_Ky_NLP/
├── Bao_Cao_Tieu_Luan_Giua_Ky_NLP.pdf   # ⭐ File PDF bài báo cáo tiểu luận hoàn chỉnh chuẩn mực (29 trang)
├── Thuc_Hanh_Word2Vec_LSTM.ipynb       # ⭐ Jupyter Notebook thực nghiệm trực quan (đã chạy sẵn kết quả)
├── BAO_CAO_GIUA_KY.md                  # Bản báo cáo Markdown chi tiết song song
├── requirements.txt                    # Danh sách thư viện (PyTorch, Transformers, Gensim, Pyvi)
├── README.md                           # File hướng dẫn này
│
├── tai_lieu_tham_khao/                 # 📚 Trọn bộ 10 file PDF bài báo khoa học & sách tham khảo gốc
│   ├── 01_UIT_VSFC_Deep_Learning_vs_Traditional_2018.pdf
│   ├── 02_Mikolov_Word2Vec_Vector_Space_2013.pdf
│   ├── 03_Mikolov_Word2Vec_Negative_Sampling_NeurIPS_2013.pdf
│   ├── 04_Hochreiter_Schmidhuber_LSTM_1997.pdf
│   ├── 05_Yoav_Goldberg_NN_Primer_NLP_2016.pdf
│   ├── 06_Jurafsky_Martin_Ch05_Embeddings.pdf
│   ├── 06b_Jurafsky_Martin_Ch14_RNNs_and_LSTMs.pdf
│   ├── 07_Rehurek_Sojka_Gensim_Framework_2010.pdf
│   ├── 08_Vaswani_Attention_Is_All_You_Need_2017.pdf
│   └── 09_Nguyen_PhoBERT_Pretrained_Language_Model_2020.pdf
│
├── data/                               # Dữ liệu thực nghiệm (2 bộ dữ liệu thực tế chuẩn)
│   ├── uit_vsfc/                       # 1. Bộ dữ liệu UIT-VSFC (3.100 mẫu) - Miền học thuật
│   └── ecommerce/                      # 2. Bộ dữ liệu E-Commerce Reviews (3.100 mẫu) - Miền TMĐT
│
├── src/                                # Toàn bộ mã nguồn module hóa
│   ├── preprocess.py                   # Tiền xử lý văn bản tiếng Việt & tách từ
│   ├── word2vec_trainer.py             # Huấn luyện Word2Vec (CBOW & Skip-Gram) bằng Gensim
│   ├── lstm_classifier.py              # Xây dựng & huấn luyện BiLSTM và BiLSTM + Self-Attention
│   ├── phobert_classifier.py           # Fine-tuning mô hình ngôn ngữ lớn tiếng Việt PhoBERT
│   ├── text_representation.py          # Biểu diễn văn bản (TF-IDF, Average Word2Vec, LSTM Vector)
│   ├── evaluate.py                     # Đánh giá Accuracy, F1-Score, Confusion Matrix, Attention Heatmap
│   └── run_experiments.py              # Script chạy toàn bộ pipeline đối chuẩn 6 mô hình
│
├── results/                            # Kết quả thực nghiệm: Attention Heatmap, t-SNE, CM, đường cong Loss
└── models/                             # File trọng số mô hình đã lưu (.pt, .model)
```

---

## 📊 BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN 6 MÔ HÌNH TRÊN 2 DATASET

Thực nghiệm được thực hiện trên **02 bộ dữ liệu thực tế tiếng Việt độc lập** để đánh giá khả năng tổng quát hóa đa miền:

| STT | Mô hình / Phương pháp | UIT-VSFC Accuracy | UIT-VSFC F1-Score | E-Commerce Accuracy | E-Commerce F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **TF-IDF + Logistic Regression (Baseline 1)** | 89.83% | 89.80% | 70.33% | 69.65% |
| 2 | **Average Word2Vec + Logistic Reg. (Baseline 2)** | 87.50% | 87.49% | 67.33% | 66.80% |
| 3 | **BiLSTM Classifier (Học từ đầu - Scratch)** | 89.17% | 89.16% | 69.17% | 68.75% |
| 4 | **BiLSTM (Pre-trained Word2Vec)** | 91.00% | 91.00% | 67.17% | 67.05% |
| 5 | **BiLSTM + Self-Attention (Cơ chế chú ý thích nghi)** | **91.83%** | **91.83%** | **72.50%** | **72.40%** |
| 6 | **PhoBERT-base-v2 (Transformer Tiếng Việt SOTA)** | **95.50%** | **95.50%** | **86.17%** | **86.16%** |

### Nhận xét học thuật nổi bật:
1. **BiLSTM + Self-Attention**:
   - Cải thiện đáng kể hiệu năng trên cả 2 miền (đạt **91.83% trên UIT-VSFC** và bứt phá lên **72.50% trên E-Commerce**, tăng +5.33% so với BiLSTM thuần túy).
   - Tăng tính khả giải thích thông qua **bản đồ nhiệt (Attention Heatmaps)**: mô hình tự động gán trọng số lớn ($0.25 \sim 0.45$) cho các từ khóa cảm xúc then chốt (`nhiệt_tình`, `rất_tốt`, `kém`, `thất_vọng`).
2. **Sức mạnh áp đảo của PhoBERT (Transformer SOTA)**:
   - Xác lập kỷ lục độ chính xác cao nhất: **95.50% trên UIT-VSFC** và **86.17% trên E-Commerce** (tăng vượt trội **+15.84%** so với TF-IDF truyền thống).
   - Cơ chế Multi-Head Attention cùng việc được tiền huấn luyện trên 20GB văn bản tiếng Việt giúp PhoBERT giải quyết hoàn toàn bài toán đa nghĩa và ngữ điệu mỉa mai, tiếng lóng trong thương mại điện tử.
3. **Tiến hóa không gian vector qua t-SNE**:
   - Không gian biểu diễn chuyển dịch rõ nét: từ dạng rời rạc lẫn lộn của Average Word2Vec, sang cấu trúc phân cụm của BiLSTM, đến hai khối cô đặc tách biệt tuyệt đối với siêu phẳng phân chia rộng lớn của PhoBERT Contextual Embeddings.

---

## 🚀 HƯỚNG DẪN CHẠY & TRẢI NGHIỆM SẢN PHẨM THỰC TẾ

### 1. Khởi chạy Ứng Dụng Web Trực Quan 1-Click (Khuyên dùng)
Hệ thống tích hợp sẵn giao diện web hiện đại (SPA), chạy trực tiếp với Python và PyTorch mà **không cần cài thêm bất kỳ thư viện phụ thuộc nào khác**:
```bash
# Cách 1: Chạy 1-click bằng bash script (tự động phát hiện môi trường & mở trình duyệt)
./run_app.sh

# Cách 2: Chạy trực tiếp qua Python
python app.py
```
Sau đó truy cập trình duyệt web tại: **`http://localhost:8501`**

#### 🌟 Tính năng nổi bật của Ứng Dụng Web:
* 🔄 **Chuyển đổi miền linh hoạt (Domain Switcher)**: Chuyển đổi qua lại giữa **UIT-VSFC (Giáo dục / Khảo sát SV)** và **E-Commerce (Thương mại điện tử)**.
* ⚡ **Đối chuẩn đồng thời cả 6 mô hình (Multi-model Benchmark)**: Khi nhập một câu nhận xét, toàn bộ 6 mô hình cùng dự đoán song song trong vài mili-giây, hiển thị nhãn (Tích cực / Tiêu cực), thanh đo độ tự tin (Confidence score %) và độ trễ suy luận (Latency ms).
* 🔍 **Bản đồ nhiệt Attention (Interactive Attention Heatmap)**: Trực quan hóa trọng số chú ý của mô hình **BiLSTM + Self-Attention**, làm nổi bật từng từ vựng đóng góp vào quyết định cảm xúc (Explainable AI - XAI).
* 💡 **Thư viện mẫu câu kiểm thử thực tế**: Nạp nhanh các câu khen ngợi, phàn nàn, câu có cấu trúc phủ định tương phản, teencode/viết tắt để kiểm tra khả năng bắt lỗi của từng mô hình.
* 🤝 **Đồng thuận mô hình (Consensus Indicator)**: Đánh giá tỷ lệ đồng thuận của hội đồng 6 mô hình (ví dụ: *6/6 mô hình đồng thuận: Tích cực*).

### 2. Khởi chạy bằng Streamlit (Tùy chọn)
Nếu bạn có cài đặt thư viện `streamlit`:
```bash
streamlit run streamlit_app.py
```

### 3. Mở Jupyter Notebook để xem chi tiết
```bash
jupyter notebook Thuc_Hanh_Word2Vec_LSTM.ipynb
```

### 4. Chạy lại toàn bộ pipeline huấn luyện đối chuẩn 6 mô hình bằng dòng lệnh
```bash
python src/run_experiments.py
```
