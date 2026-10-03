# 🎓 GÓI TÀI LIỆU TIỂU LUẬN & ỨNG DỤNG PHÂN TÍCH CẢM XÚC TIẾNG VIỆT
### TRƯỜNG ĐẠI HỌC SƯ PHẠM TP. HỒ CHÍ MINH — KHOA KHOA HỌC MÁY TÍNH
**Học phần:** Xử Lý Ngôn Ngữ Tự Nhiên (NLP) • Khóa 36 (2025 - 2027)  
**Giảng viên hướng dẫn:** **PGS.TS. LÊ ANH CƯỜNG**  
**Nhóm sinh viên thực hiện:**  
1. **Trần Gia Huy** — MSSV: **KHMT836012**  
2. **Đỗ Minh Khánh Ngân** — MSSV: **KHMT836019**  
3. **Nguyễn Tấn Phát** — MSSV: **KHMT836026**  
**Đề tài:** *Mô hình Word2Vec, Word Embedding và Học biểu diễn văn bản bằng LSTM cho bài toán phân loại cảm xúc.*

---

## 📦 CÁC THÀNH PHẦN TRONG GÓI GỬI NHÓM

| Tên File / Thư mục | Mô Tả Chi Tiết |
| :--- | :--- |
| 📄 **`Bao_Cao_Tieu_Luan_Giua_Ky_NLP.pdf`** | **Bản báo cáo tiểu luận hoàn chỉnh 29 trang** chuẩn học thuật (đầy đủ tên thành viên, giáo viên, bảng đối chuẩn 6 mô hình, đồ thị t-SNE, Loss curves và phân tích ngôn ngữ học). Dùng để đọc ôn bài hoặc in nộp. |
| 📝 **`BAO_CAO_GIUA_KY.md`** | Bản báo cáo định dạng Markdown song song, tiện tra cứu nội dung trên GitHub hoặc trình đọc văn bản. |
| 📓 **`Thuc_Hanh_Word2Vec_LSTM.ipynb`** | **Jupyter Notebook thực nghiệm hoàn chỉnh** đã chạy sẵn 100% output, 0 lỗi, đầy đủ biểu đồ t-SNE 2 chiều, đồ thị so sánh hiệu năng và kết luận đánh giá (Summary). |
| 🌐 **`web_app/` + `app.py`** | Hệ thống Web Demo sản phẩm thực tế với bộ nhận diện chuẩn HCMUE (Cerulean `#124874` & Jasper `#CF373D`), hỗ trợ suy luận thời gian thực. |
| 📂 **`data/`** | 2 tập dữ liệu thực tế chuẩn hóa: `uit_vsfc/` (Ý kiến sinh viên) và `ecommerce/` (Đánh giá mua sắm trực tuyến). |
| 🧠 **`models/`** | Trọng số mô hình đã huấn luyện sẵn (TF-IDF `.joblib`, Word2Vec CBOW `.model`, BiLSTM Scratch/Pretrained/Attention `.pt`, Từ điển `vocab_*.json`). |
| 🛠️ **`src/`** | Toàn bộ mã nguồn module hóa của hệ thống (`inference_engine.py`, `preprocess.py`, `lstm_classifier.py`, `word2vec_trainer.py`). |

---

## 🚀 HƯỚNG DẪN SETUP & CHẠY WEB DEMO TRÊN MÁY TÍNH

Mọi người có thể mở ứng dụng bằng một trong các cách sau tùy theo hệ điều hành:

### 🌟 CÁCH 1: Trải nghiệm Online ngay trên Trình duyệt (KHÔNG CẦN CÀI ĐẶT)
* Nếu cần xem demo nhanh trên điện thoại hoặc máy tính mà không muốn cài Python:
* 👉 Truy cập link GitHub Pages chính thức của nhóm:  
  **[https://trangiahuy8444.github.io/NLP/](https://trangiahuy8444.github.io/NLP/)**  
  *(Hệ thống đã tích hợp sẵn Web Client Engine để bấm phân tích tức thì 24/7).*

---

### 💻 CÁCH 2: Chạy 1-Click trên Windows
1. Đảm bảo máy tính đã cài đặt Python (phiên bản 3.9 đến 3.12).
2. Nhấp đúp chuột vào file: **`run_app.bat`**
3. Script sẽ tự động kiểm tra thư viện và khởi động Web Server.
4. Mở trình duyệt web tại địa chỉ: **`http://localhost:8501`**

---

### 🍏 CÁCH 3: Chạy 1-Click trên macOS / Linux
1. Mở ứng dụng **Terminal** và điều hướng vào thư mục này:
   ```bash
   cd đường_dẫn_tới_thư_mục/Goi_Bao_Cao_Gui_Nhom_NLP
   ```
2. Cấp quyền và chạy script:
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```
3. Trình duyệt sẽ tự động mở tại: **`http://localhost:8501`**

---

### 🐍 CÁCH 4: Chạy thủ công bằng lệnh Python / Conda
Nếu muốn chạy thủ công từng bước:
1. Mở Terminal / Command Prompt và kích hoạt môi trường Python:
   ```bash
   # Cài đặt thư viện nếu chưa có
   pip install -r requirements.txt
   ```
2. Khởi chạy máy chủ:
   ```bash
   python app.py --port 8501
   ```
3. Mở trình duyệt và truy cập: **`http://127.0.0.1:8501`**

---

### 📓 CÁCH 5: Mở File Thực Nghiệm Jupyter Notebook
Để mở kiểm tra code thực nghiệm Word2Vec và BiLSTM:
```bash
jupyter notebook Thuc_Hanh_Word2Vec_LSTM.ipynb
# Hoặc mở trực tiếp trên VS Code bằng cách click vào file .ipynb
```
> **Lưu ý:** File notebook đã được chạy sẵn toàn bộ biểu đồ và kết quả huấn luyện nên có thể đọc lướt xem kết quả ngay mà không bắt buộc phải chạy lại từ đầu.

---

## 🎯 CÁC TÍNH NĂNG CHÍNH ĐỂ THUYẾT MINH / BÁO CÁO

Khi trình chiếu Web Demo trước thầy cô hoặc ban giám khảo, mọi người lưu ý các điểm nổi bật sau:
1. **Chuyển đổi 2 miền dữ liệu:** Bấm chọn giữa **UIT-VSFC (Giáo dục)** và **E-Commerce (Thương mại điện tử)** để thấy sự thay đổi về đặc trưng từ vựng và bảng đối chuẩn benchmark.
2. **Chọn các câu mẫu đa dạng:** Có sẵn 10 câu mẫu thuộc các trường hợp: khen ngợi, phàn nàn, cấu trúc tương phản (*nhưng, tuy nhiên*), từ ngữ giới trẻ/teencode (*10 điểm không có nhưng, xịn xò vl*).
3. **Bảng kết luận đồng thuận (Consensus):** Thể hiện sự tổng hợp ý kiến từ cả 6 mô hình và độ tin cậy trung bình.
4. **Bản đồ nhiệt chú ý (Attention Heatmap):** Đây là điểm nhấn của **mô hình đề xuất BiLSTM + Self-Attention** — cho phép giải thích trực quan từ khóa nào khiến AI quyết định nhãn tích cực hay tiêu cực.
