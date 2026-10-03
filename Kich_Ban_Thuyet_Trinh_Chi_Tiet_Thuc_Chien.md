# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT & BỘ TÀI LIỆU PHẢN BIỆN THỰC CHIẾN
## ĐỀ TÀI: NGHIÊN CỨU & XÂY DỰNG MÔ HÌNH PHÂN LOẠI CẢM XÚC VĂN BẢN TIẾNG VIỆT
### DỰA TRÊN WORD2VEC, LSTM VÀ CƠ CHẾ CHÚ Ý (ATTENTION MECHANISM)

---

- **Đơn vị đào tạo:** Trường Đại học Sư phạm TP. Hồ Chí Minh — Khoa Công nghệ Thông tin
- **Chương trình đào tạo:** Lớp Cao học Khoa học Máy tính — Khóa K36
- **Học phần:** Xử lý Ngôn ngữ Tự nhiên Nâng cao
- **Giảng viên hướng dẫn:** **PGS. TS. Lê Anh Cường**
- **Nhóm học viên thực hiện:**
  1. **Trần Gia Huy** (Mã học viên: 8444...)
  2. **Nguyễn Thị Kim Ngân**
  3. **Hoàng Tấn Phát**

---

# MỤC LỤC TỔNG QUAN

1. [PHẦN I: KỊCH BẢN BÁO CÁO GIỮA KỲ HỘI ĐỒNG CHUYÊN MÔN (20 PHÚT)](#phan-i-kich-ban-bao-cao-giua-ky)
   - Phân chia thời lượng & Vai trò diễn giả
   - Kịch bản chi tiết từng Slide (Slide 1 đến Slide 24)
2. [PHẦN II: KỊCH BẢN BÀI GIẢNG NHẬP MÔN NLP CHO NGƯỜI MỚI BẮT ĐẦU (30 PHÚT)](#phan-ii-kich-ban-bai-giang-nhap-mon)
   - Phương pháp luận sư phạm & Tâm thế giảng dạy
   - Kịch bản diễn giải từng Slide (Slide 1 đến Slide 22)
   - Hướng dẫn sư phạm giải chi tiết 3 bài tập giải tay (Slide 13, 14, 15)
3. [PHẦN III: BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU & CÂU TRẢ LỜI MẪU (Q&A DEFENSE)](#phan-iii-bo-cau-hoi-phan-bien)
   - 10 Câu hỏi trọng tâm từ Hội đồng / Giảng viên hướng dẫn
4. [PHẦN IV: CHECKLIST KỸ THUẬT VÀ PHÒNG NGỪA RỦI RO TRƯỚC GIỜ G](#phan-iv-checklist-ky-thuat)

---

<a name="phan-i-kich-ban-bao-cao-giua-ky"></a>
# PHẦN I: KỊCH BẢN BÁO CÁO GIỮA KỲ HỘI ĐỒNG CHUYÊN MÔN
- **Thời lượng chuẩn:** 20 phút báo cáo + 10 phút hỏi đáp phản biện (Q&A).
- **Tỉ lệ trình chiếu:** Widescreen 16:9 Beamer (`Bai_Thuyet_Trinh_Giua_Ky_NLP.pdf`).
- **Phân vai trình bày:**
  * **Diễn giả 1 (Trần Gia Huy):** Mở đầu, Đặt vấn đề, Cơ sở lý luận (Word2Vec, LSTM, BiLSTM + Additive Attention, PhoBERT). *(Slide 1 - Slide 10 | ~8 phút)*
  * **Diễn giả 2 (Nguyễn Thị Kim Ngân):** Thiết kế thực nghiệm, Đối chuẩn hiệu năng 6 mô hình, Metrics AUC-ROC, XAI Attention Heatmap & Phân không gian t-SNE. *(Slide 11 - Slide 16 | ~6 phút)*
  * **Diễn giả 3 (Hoàng Tấn Phát):** Thảo luận học thuật (Domain Gap, Sarcasm), Phân tích lỗi định tính, Trình diễn Web Demo & Kết luận hướng phát triển. *(Slide 17 - Slide 24 | ~6 phút)*
  *(Lưu ý: Nếu một người báo cáo toàn bộ, chỉ cần theo đúng mạch chuyển ý liên tục bên dưới).*

---

### SLIDE 1: TRANG TIÊU ĐỀ (BẮT ĐẦU)
- **Thời lượng:** 0:00 - 0:45 (45 giây)
- **Người trình bày:** Diễn giả 1 (Trần Gia Huy)
- **Hành động:** Đứng nghiêm túc, phong thái tự tin, mắt bao quát toàn thể hội đồng, cúi đầu chào lịch sự.
- **Lời thoại:**
  > "Kính thưa Thầy hướng dẫn - Phó Giáo sư, Tiến sĩ Lê Anh Cường, cùng toàn thể quý thầy cô và các anh chị học viên có mặt trong buổi báo cáo chuyên đề hôm nay.  
  > 
  > Thay mặt nhóm nghiên cứu lớp Cao học Khoa học Máy tính K36, em là Trần Gia Huy, cùng hai thành viên là bạn Nguyễn Thị Kim Ngân và bạn Hoàng Tấn Phát, xin phép được báo cáo đề tài nghiên cứu giữa kỳ môn Xử lý Ngôn ngữ Tự nhiên: **'Nghiên cứu và Xây dựng Mô hình Phân loại Cảm xúc Văn bản Tiếng Việt Dựa trên Word2Vec, LSTM và Cơ chế Chú ý Attention'**.  
  > 
  > Bài báo cáo của nhóm chúng em hôm nay sẽ tái hiện một bức tranh khoa học hoàn chỉnh từ nền tảng lý thuyết biểu diễn ngữ nghĩa, thiết kế kiến trúc cải tiến BiLSTM kết hợp Additive Self-Attention, thực nghiệm đối chuẩn trên 3 miền ngữ liệu thực tế, cho đến đóng gói hệ thống phần mềm thời gian thực. Sau đây, em xin phép bắt đầu phần trình bày."

---

### SLIDE 2: NỘI DUNG BÁO CÁO (AGENDA)
- **Thời lượng:** 0:45 - 1:20 (35 giây)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Dùng laser pointer lướt nhanh qua 6 đề mục chính.
- **Lời thoại:**
  > "Kính thưa quý thầy cô, bài báo cáo được cấu trúc thành 6 phần logic chặt chẽ:  
  > 1. Đầu tiên là **Đặt vấn đề & Mục tiêu nghiên cứu**, chỉ ra các thách thức đặc thù của tiếng Việt.  
  > 2. Thứ hai là **Cơ sở lý luận**, tái hiện tiến trình tiến hóa biểu diễn từ Word2Vec, LSTM, cơ chế Self-Attention đến mô hình Transformer SOTA.  
  > 3. Thứ ba là **Thiết kế thực nghiệm**, công bố quy trình tiền xử lý 5 bước trên 3 tập dữ liệu đa miền.  
  > 4. Thứ tư là **Đối chuẩn kết quả & Phân tích giải thích (XAI)** qua ma trận nhầm lẫn, AUC-ROC, Attention Heatmap và chiếu t-SNE.  
  > 5. Thứ năm là **Hệ thống Demo thời gian thực**, triển khai sản phẩm ứng dụng thực tế.  
  > 6. Và cuối cùng là **Kết luận cùng định hướng nghiên cứu tiếp theo**."

---

### SLIDE 3: 1. ĐẶT VẤN ĐỀ - THÁCH THỨC BIỂU DIỄN TIẾNG VIỆT
- **Thời lượng:** 1:20 - 2:30 (1 phút 10 giây)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Nhấn giọng ở các đặc thù ngôn ngữ học của tiếng Việt.
- **Lời thoại:**
  > "Bắt đầu với bài toán Đặt vấn đề. Phân loại cảm xúc (Sentiment Analysis) đóng vai trò sống còn trong quản trị trải nghiệm khách hàng và giám sát dư luận mạng xã hội. Tuy nhiên, khi áp dụng vào tiếng Việt, các hệ thống NLP truyền thống gặp phải 3 rào cản cốt tử:  
  > 
  > - **Thứ nhất, tính đơn lập và ranh giới từ phức:** Tiếng Việt không biến đổi hình thái từ. Một từ có thể gồm nhiều âm tiết ghép lại, ví dụ từ *'bàn học'* khác hoàn toàn *'bàn bạc'*. Nếu tách sai từ tố, toàn bộ ngữ nghĩa bị sai lệch.  
  > - **Thứ hai, hiện tượng đa nghĩa và phụ thuộc xa:** Cảm xúc của câu tiếng Việt phụ thuộc mật thiết vào các liên từ đảo nghịch như *'tuy... nhưng'*, *'tuyệt vời nhưng giá hơi chát'*. Các mô hình nông như Naive Bayes hay Bag-of-Words hoàn toàn bất lực vì phá vỡ cấu trúc ngữ tự.  
  > - **Thứ ba, sự thiếu hụt tài nguyên chuẩn hóa:** Các mô hình học sâu đòi hỏi lượng tham số khổng lồ, nhưng tài nguyên gán nhãn tiếng Việt còn phân mảnh.  
  > 
  > Xuất phát từ thực tiễn đó, câu hỏi nghiên cứu của nhóm là: *Làm thế nào để xây dựng một kiến trúc vừa trích xuất ngữ nghĩa sâu sắc, vừa duy trì tính gọn nhẹ, có khả năng giải thích được (XAI) và hoạt động bền vững trước sự dịch chuyển phân phối dữ liệu đa miền?*"

---

### SLIDE 4: MỤC TIÊU NGHIÊN CỨU & 7 ĐÓNG GÓP CHÍNH
- **Thời lượng:** 2:30 - 3:45 (1 phút 15 giây)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Chỉ vào 7 bullet point trên slide, nhấn mạnh tính ứng dụng.
- **Lời thoại:**
  > "Để trả lời câu hỏi đó, đề tài của nhóm đã hoàn thành trọn vẹn 7 đóng góp học thuật và thực tiễn:  
  > 1. Hệ thống hóa tiến trình biểu diễn ngữ nghĩa từ phân tán (Distributional Semantics) đến học sâu.  
  > 2. Đề xuất thành công kiến trúc **BiLSTM kết hợp Additive Self-Attention**, khắc phục hoàn toàn nhược điểm nghẽn cổ chai thông tin (Information Bottleneck) của RNN truyền thống.  
  > 3. Thiết lập ma trận thực nghiệm quy mô trên **3 miền ngữ liệu độc lập**: UIT-VSFC (Giáo dục), E-Commerce (Thương mại điện tử), và IMDb (Đánh giá phim).  
  > 4. Đối chuẩn sòng phẳng 6 mô hình từ cổ điển (LogReg, TF-IDF) đến SOTA (PhoBERT-base).  
  > 5. Cung cấp khả năng giải thích minh bạch (Explainable AI) bằng Attention Heatmap trực quan.  
  > 6. Thực hiện phân tích định tính các ca lỗi kinh điển (Sarcasm, câu đa chiều, từ hiếm).  
  > 7. Và cuối cùng, đóng gói toàn bộ pipeline thành **Web Demo tương tác thực tế** với độ trễ suy luận dưới 50ms, sẵn sàng triển khai môi trường sản xuất."

---

### SLIDE 5: 2. TIẾN TRÌNH PHÁT TRIỂN BIỂU DIỄN TỪ TRONG NLP
- **Thời lượng:** 3:45 - 5:00 (1 phút 15 giây)
- **Người trình bày:** Diễn giả 1
- **Hành động:** So sánh giữa 2 khối hình: Không gian rời rạc vs Không gian phân tán liên tục.
- **Lời thoại:**
  > "Em xin phép chuyển sang Cơ sở lý luận: Tiến trình phát triển biểu diễn từ.  
  > - Giai đoạn đầu tiên là **Mã hóa rời rạc (Discrete Representation)** với One-Hot Encoding và Bag-of-Words/TF-IDF. Điểm nghẽn chí mạng ở đây là ma trận cực kỳ thưa, bùng nổ số chiều theo kích thước từ điển $V$, và quan trọng nhất: *khoảng cách Euclid giữa mọi cặp từ luôn bằng $\sqrt{2}$, tích vô hướng bằng 0*. Nghĩa là máy tính xem 'tuyệt vời' và 'xuất sắc' xa lạ hệt như 'tuyệt vời' với 'chiếc xe đạp'.  
  > - Bước ngoặt lịch sử diễn ra vào năm 2013 với giả thuyết phân phối của Harris & Firth: *'Từ ngữ được định nghĩa bởi các từ đi cùng với nó'*. **Word2Vec** ra đời, ánh xạ từ vựng vào không gian vector thực dày đặc liên tục $\mathbb{R}^d$ ($d \ll |V|$), nơi các từ đồng nghĩa tự động hội tụ gần nhau theo thước đo góc Cosine Similarity."

---

### SLIDE 6: KHÔNG GIAN VECTOR WORD2VEC: CBOW, SKIP-GRAM & SGNS
- **Thời lượng:** 5:00 - 6:30 (1 phút 30 giây)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Nhấn mạnh vào công thức toán học SGNS trên slide.
- **Lời thoại:**
  > "Đi sâu vào cơ chế Word2Vec, chúng ta có hai kiến trúc đối ngẫu: **CBOW** dùng từ ngữ cảnh để dự đoán từ trung tâm, và **Skip-Gram** dùng từ trung tâm để dự đoán ngữ cảnh xung quanh.  
  > 
  > Về mặt toán học, nếu tính toán Softmax toàn từ điển $V$, mẫu số sẽ có độ phức tạp $O(|V|)$, với $|V| \ge 100.000$ từ thì không thể huấn luyện nổi. Mikolov đã đưa ra giải pháp đột phá: **Skip-Gram with Negative Sampling (SGNS)**.  
  > Như công thức trên slide:
  > $$\mathcal{L}_{SGNS} = \sum_{t=1}^T \sum_{-c \le j \le c, j \neq 0} \left[ \log \sigma(v'_{w_{t+j}}^\top v_{w_t}) + \sum_{k=1}^K \mathbb{E}_{w_{n,k} \sim P_n(w)} \left[ \log \sigma(-v'_{w_{n,k}}^\top v_{w_t}) \right] \right]$$
  > Mục tiêu là cực đại hóa xác suất từ ngữ cảnh thực sự xuất hiện qua hàm Sigmoid $\sigma$, đồng thời cực tiểu hóa xác suất của $K$ từ âm bản được rút ngẫu nhiên theo phân phối làm mịn lũy thừa $\frac{3}{4}$: $P_n(w) \propto f(w)^{3/4}$. Việc này giảm độ phức tạp từ $O(|V|)$ xuống chỉ còn $O(K)$, giúp mô hình học cực nhanh trên kho dữ liệu hàng tỷ từ."

---

### SLIDE 7: SO SÁNH ĐỐI ĐẦU KỸ THUẬT: CBOW VS SKIP-GRAM
- **Thời lượng:** 6:30 - 7:15 (45 giây)
- **Diễn giả 1:**
- **Lời thoại:**
  > "Trên slide 7 là bảng so sánh kỹ thuật đối đầu giữa hai kiến trúc.  
  > - **CBOW** tối ưu về tốc độ huấn luyện, hiệu quả cao trên các từ phổ biến do cơ chế trung bình cộng ngữ cảnh làm phẳng nhiễu.  
  > - Ngược lại, **Skip-Gram** tốn chi phí tính toán hơn nhưng vượt trội hoàn toàn trong việc nắm bắt các từ ngữ hiếm (Rare words) và cấu trúc ngữ nghĩa tinh tế, vì mỗi cặp (từ đích - ngữ cảnh) đều được cập nhật gradient riêng biệt.  
  > Chính vì đặc thù ngôn ngữ mạng xã hội tiếng Việt có nhiều từ lóng và biến thể hiếm, nhóm nghiên cứu đã chọn kiến trúc **Skip-Gram với kích thước vector $d=150$** làm nền tảng trích xuất đặc trưng cho các mô hình học sâu tiếp theo."

---

### SLIDE 8: HỌC BIỂU DIỄN VĂN BẢN BẰNG LSTM & MẠNG BILSTM
- **Thời lượng:** 7:15 - 8:30 (1 phút 15 giây)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Chỉ vào 6 công thức cổng logic LSTM.
- **Lời thoại:**
  > "Khi đã có vector từ, làm thế nào biểu diễn cả một câu?  
  > Nếu chỉ cộng trung bình (Average Pooling), ta mất toàn bộ thứ tự từ. Mạng RNN truyền thống thì bị hiện tượng triệt tiêu gradient (Vanishing Gradient) khi chuỗi dài hơn 10 bước.  
  > 
  > Giải pháp là **LSTM (Long Short-Term Memory)** với đường truyền cao tốc Cell State $C_t$ được điều tiết bởi 3 cổng logic:  
  > - Cổng quên $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$ quyết định xóa bỏ bao nhiêu ký ức quá khứ.  
  > - Cổng vào $i_t$ kết hợp trạng thái ứng viên $\tilde{C}_t$ cập nhật thông tin mới.  
  > - Cổng ra $o_t$ lọc thông tin từ $C_t$ đưa ra ẩn trạng thái $h_t$.  
  > 
  > Hơn thế nữa, ngôn ngữ tự nhiên đòi hỏi ngữ cảnh hai chiều. Do đó, nhóm xây dựng **BiLSTM**: luồng xuôi đọc từ trái sang phải $\overrightarrow{h}_t$, luồng ngược đọc từ phải sang trái $\overleftarrow{h}_t$. Ẩn trạng thái tổng hợp $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2d_h}$ nắm trọn vẹn ngữ cảnh cả quá khứ lẫn tương lai tại từng vị trí từ."

---

### SLIDE 9: ĐỀ XUẤT: TÍCH HỢP ADDITIVE SELF-ATTENTION & XAI
- **Thời lượng:** 8:30 - 9:30 (1 phút)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Nhấn mạnh vào lý do không dùng vector cuối cùng $h_T$.
- **Lời thoại:**
  > "Tuy BiLSTM rất mạnh, nhưng mô hình truyền thống thường chỉ lấy vector ở bước cuối cùng $h_T$ làm đại diện câu. Điều này tạo ra hiện tượng **'Nghẽn cổ chai thông tin' (Information Bottleneck)**: toàn bộ thông điệp của câu 50 từ bị ép vào một vector duy nhất.  
  > 
  > Nhóm đề xuất tích hợp cơ chế **Additive Self-Attention** của Bahdanau/Yang:  
  > 1. Chiếu phi tuyến ẩn trạng thái $h_t$ qua mạng nơ-ron truyền thẳng để tính điểm tiềm năng: $u_t = \tanh(W_a h_t + b_a)$.  
  > 2. Đo mức độ quan trọng bằng tích vô hướng với vector ngữ cảnh học được $u_s$, rồi chuẩn hóa qua Softmax để có phân phối xác suất chú ý:  
  >    $$\alpha_t = \frac{\exp(u_t^\top u_s)}{\sum_{j=1}^T \exp(u_j^\top u_s)}$$  
  > 3. Vector tài liệu $v_D$ là tổng tổ hợp tuyến tính có trọng số: $v_D = \sum_{t=1}^T \alpha_t h_t$.  
  > 
  > Lợi ích đột phá ở đây là: mô hình tự động 'soi sáng' các từ mang tải lượng cảm xúc cao (như 'tuyệt vời', 'thất vọng'), triệt tiêu các từ dừng rác, và biến mô hình hộp đen thành một hệ thống **Explainable AI (XAI)** hoàn toàn minh bạch."

---

### SLIDE 10: SƠ ĐỒ LUỒNG KIẾN TRÚC & PHOBERT [CLS]
- **Thời lượng:** 9:30 - 10:30 (1 phút)
- **Người trình bày:** Diễn giả 1
- **Hành động:** Giới thiệu ngắn gọn sơ đồ khối và chuyển giao cho Diễn giả 2.
- **Lời thoại:**
  > "Trên Slide 10 là sơ đồ luồng kiến trúc hoàn chỉnh của mô hình đề xuất: Dữ liệu đi từ Embedding Layer $\rightarrow$ Khối BiLSTM 2 chiều $\rightarrow$ Khối Additive Attention trích xuất trọng số $\alpha_t \rightarrow$ Vector $v_D \rightarrow$ Lớp Softmax phân loại.  
  > 
  > Đồng thời, ở Slide 11, nhóm thiết lập mô hình đối chuẩn tối thượng SOTA: **PhoBERT-base** kiến trúc Transformer với 12 tầng Self-Attention. Nhóm khai thác vector ẩn tại token đặc biệt **`[CLS]` ($\in \mathbb{R}^{768}$)** đứng đầu câu, nơi đã tích hợp ngữ cảnh toàn cục thông qua cơ chế Scaled Dot-Product Attention đa đầu.  
  > 
  > Sau đây, em xin trân trọng kính mời bạn **Nguyễn Thị Kim Ngân** tiếp tục phần trình bày về Thiết kế thực nghiệm và Kết quả đối chuẩn."

---

### SLIDE 11 & 12: 3. THIẾT KẾ THỰC NGHIỆM & QUY TRÌNH TIỀN XỬ LÝ 5 BƯỚC
- **Thời lượng:** 10:30 - 12:00 (1 phút 30 giây)
- **Người trình bày:** Diễn giả 2 (Nguyễn Thị Kim Ngân)
- **Hành động:** Giọng nói rõ ràng, tự tin, nhấn mạnh quy mô dữ liệu và quy chuẩn kiểm thử.
- **Lời thoại:**
  > "Kính thưa Thầy và Hội đồng, em là Kim Ngân. Em xin phép trình bày phần Thực nghiệm đối chuẩn.  
  > Để đánh giá mô hình một cách khách quan nhất, nhóm không chỉ kiểm thử trên một tập dữ liệu đóng mà mở rộng trên **3 miền ngữ liệu thực tế**:  
  > 1. **UIT-VSFC:** Hơn 16.000 câu phản hồi sinh viên trong môi trường học thuật.  
  > 2. **Tiki / Shopee Reviews:** 8.000 mẫu đánh giá thương mại điện tử với ngôn ngữ tự do, teencode phong phú.  
  > 3. **IMDb Movie Reviews:** 50.000 mẫu đánh giá điện ảnh chuẩn mực quốc tế để kiểm tra tính tổng quát hóa đa ngôn ngữ.  
  > 
  > Toàn bộ dữ liệu trải qua **Pipeline tiền xử lý 5 bước nghiêm ngặt**:  
  > - Chuẩn hóa Unicode dựng sẵn (NFC).  
  > - Chuyển chữ thường, làm sạch URL/HTML, xử lý teencode và ký tự lặp (ví dụ 'ngonnnn' $\rightarrow$ 'ngon').  
  > - Phân đoạn từ tiếng Việt chuẩn bằng thư viện chuyên dụng `pyvi`.  
  > - Lọc bỏ stopword có chọn lọc (bảo toàn nghiêm ngặt các từ phủ định như *'không', 'chẳng, chưa'*).  
  > - Cắt ngắn và đệm chuỗi (Padding/Truncating) về độ dài cố định $L=100$ tokens."

---

### SLIDE 13 & 14: 4. ĐỐI CHUẨN TOÀN DIỆN: ACCURACY, F1-SCORE, AUC-ROC
- **Thời lượng:** 12:00 - 13:45 (1 phút 45 giây)
- **Người trình bày:** Diễn giả 2
- **Hành động:** Chỉ vào bảng số liệu trên Slide 13 và biểu đồ so sánh ở Slide 14.
- **Lời thoại:**
  > "Đây là kết quả trung tâm của công trình nghiên cứu: Bảng đối chuẩn 6 mô hình trên tập kiểm thử UIT-VSFC:  
  > - Nhóm Baseline cổ điển: Logistic Regression đạt $85.42\%$, Random Forest đạt $82.15\%$.  
  > - Khi chuyển sang Mạng nơ-ron tuần hoàn: LSTM đơn kênh đạt $88.15\%$, và khi nâng cấp lên **BiLSTM**, độ chính xác tăng lên $89.85\%$ (F1 đạt $89.72\%$, AUC-ROC đạt $0.9412$).  
  > - Đặc biệt, khi tích hợp cơ chế đề xuất **BiLSTM + Additive Attention**, hiệu năng nhảy vọt lên **$91.68\%$ Accuracy, F1-Score đạt $91.54\%$, và AUC-ROC đạt $0.9587$**! Mô hình vượt trội hơn hẳn BiLSTM thuần túy gần $2\%$ trên mọi thước đo.  
  > - Cuối cùng, mô hình SOTA **PhoBERT-base** thiết lập kỷ lục với **$93.45\%$ Accuracy, F1-Score $93.38\%$, và AUC-ROC đạt $0.9760$**.  
  > 
  > Nhưng thưa quý thầy cô, điểm đáng chú ý nhất nằm ở bài toán **Độ bền vững đa miền (Cross-Domain Robustness)** tại Slide 14:  
  > Khi đưa mô hình huấn luyện trên dữ liệu trường học sang kiểm thử trực tiếp trên dữ liệu E-Commerce (Domain Shift):  
  > - Logistic Regression bị suy giảm tới $14.15\%$.  
  > - BiLSTM bị suy giảm $10.65\%$.  
  > - Mô hình đề xuất **BiLSTM + Attention chỉ sụt giảm $7.58\%$**, duy trì độ chính xác $84.10\%$.  
  > Con số này chứng minh: Cơ chế Attention giúp mô hình tập trung vào bản chất các từ mang cảm xúc bất biến qua các miền, tạo khả năng kháng nhiễu cực kỳ ấn tượng!"

---

### SLIDE 15 & 16: TRỰC QUAN HÓA ATTENTION HEATMAP & CHIẾU KHÔNG GIAN T-SNE
- **Thời lượng:** 13:45 - 15:00 (1 phút 15 giây)
- **Người trình bày:** Diễn giả 2
- **Hành động:** Chỉ vào bản đồ nhiệt trực quan và cụm dữ liệu t-SNE.
- **Lời thoại:**
  > "Để chứng minh mô hình không phải là một hộp đen may rủi, Slide 15 minh chứng cơ chế hoạt động qua **Attention Heatmap**:  
  > Trong câu: *'Thầy dạy rất nhiệt tình nhưng phòng học hơi nóng'*:  
  > Trọng số chú ý $\alpha_t$ tự động dồn nén cao độ vào hai từ khóa định hình cảm xúc: **'nhiệt_tình' (0.38)** và **'hơi_nóng' (0.42)**, trong khi các từ chức năng như 'thầy', 'rất', 'phòng' nhận trọng số xấp xỉ 0. Cơ chế này giúp ta hoàn toàn thấu hiểu căn nguyên tại sao mạng nơ-ron đưa ra quyết định dự đoán.  
  > 
  > Tiếp theo, tại Slide 16 là **Không gian chiếu giảm chiều t-SNE**:  
  > - Với TF-IDF hoặc Word2Vec trung bình, các điểm dữ liệu Tích cực (xanh) và Tiêu cực (đỏ) hòa lẫn hỗn loạn ở vùng ranh giới.  
  > - Sang BiLSTM + Attention và đặc biệt là PhoBERT, hai cụm cảm xúc tách biệt ranh giới rõ rệt (Cluster Separability), chứng minh biểu diễn vector tiềm ẩn của mô hình đạt độ thuần khiết ngữ nghĩa rất cao.  
  > 
  > Sau đây, em xin kính mời bạn **Hoàng Tấn Phát** tiếp tục phần Thảo luận chuyên sâu, Phân tích lỗi và Demo hệ thống."

---

### SLIDE 17 & 18: THẢO LUẬN HỌC THUẬT: ĐỘ TRỄ SUY LUẬN & BỨT PHÁ ATTENTION
- **Thời lượng:** 15:00 - 16:30 (1 phút 30 giây)
- **Người trình bày:** Diễn giả 3 (Hoàng Tấn Phát)
- **Hành động:** Đĩnh đạc, phong thái kỹ sư công nghệ phần mềm thực chiến.
- **Lời thoại:**
  > "Kính thưa Thầy và Hội đồng, em là Hoàng Tấn Phát.  
  > Nhìn vào bảng kết quả, câu hỏi đặt ra là: *'PhoBERT đạt $93.45\%$, vậy tại sao chúng ta vẫn nghiên cứu và triển khai BiLSTM + Attention?'*  
  > Câu trả lời nằm ở bài toán **Đánh đổi Công nghệ (Engineering Trade-off)**:  
  > - PhoBERT có tới 135 triệu tham số, đòi hỏi tài nguyên GPU đắt đỏ, và độ trễ suy luận (Inference Latency) trên CPU lên tới **65ms/mẫu**.  
  > - Trong khi đó, mô hình đề xuất **BiLSTM + Additive Attention chỉ có 1.8 triệu tham số (nhẹ hơn 75 lần)**, chạy mượt mà trên CPU bình thường với độ trễ vỏn vẹn **12ms/mẫu** (nhanh hơn gấp 5.4 lần), mà độ chính xác chỉ thấp hơn PhoBERT vỏn vẹn $1.77\%$.  
  > 
  > Trong các kịch bản thực tế như xử lý hàng chục nghìn bình luận livestream thời gian thực hoặc thiết bị nhúng hạn chế phần cứng, BiLSTM + Attention chính là điểm cân bằng vàng giữa độ chính xác và chi phí vận hành!"

---

### SLIDE 19 & 20: PHÂN TÍCH LỖI ĐỊNH TÍNH (QUALITATIVE ERROR ANALYSIS) & GIẢI PHÁP
- **Thời lượng:** 16:30 - 17:45 (1 phút 15 giây)
- **Người trình bày:** Diễn giả 3
- **Hành động:** Đi thẳng vào các điểm hạn chế của mô hình một cách khoa học và thành thực.
- **Lời thoại:**
  > "Một nghiên cứu trung thực không thể thiếu việc mổ xẻ các ca dự đoán sai. Nhóm đã thực hiện **Phân tích lỗi định tính** và nhận diện 3 ca khó kinh điển:  
  > 1. **Nghệ thuật châm biếm / Mỉa mai (Sarcasm):** Ví dụ câu: *'Giao hàng sau 1 tháng, nhanh như chớp!'*. Cả BiLSTM và PhoBERT đều bắt từ 'nhanh như chớp' và đoán Tích cực, trong khi nhãn thực là Tiêu cực.  
  >    $\rightarrow$ *Nguyên nhân:* Thiếu tri thức thế giới (World Knowledge) về thời gian thực tế.  
  > 2. **Cấu trúc tương phản đa chiều:** *'Món ăn ngon tuyệt cú mèo nhưng nhân viên phục vụ coi thường khách'*. Mô hình bị lúng túng khi gán nhãn tổng thể nếu không áp dụng phân loại cảm xúc theo khía cạnh (Aspect-Based Sentiment).  
  > 3. **Từ lóng và biến thể Teencode thế hệ mới:** Các từ viết tắt biến thể liên tục nằm ngoài từ điển (OOV).  
  > 
  > Nhóm đề xuất 3 giải pháp kỹ thuật cụ thể: Tích hợp Đồ thị tri thức (Knowledge Graph), chuyển dịch sang bài toán Aspect-Based Sentiment, và áp dụng cơ chế Byte-Pair Encoding phụ trợ."

---

### SLIDE 21 & 22: 5. HỆ THỐNG DEMO ỨNG DỤNG THỜI GIAN THỰC
- **Thời lượng:** 17:45 - 19:00 (1 phút 15 giây)
- **Người trình bày:** Diễn giả 3
- **Hành động:** Chiếu màn hình giao diện Web Demo (hoặc chuyển sang cửa sổ trình duyệt tương tác trực tiếp 30 giây).
- **Lời thoại:**
  > "Nhóm không dừng lại ở các con số trên giấy báo cáo mà đã hoàn thiện một **Sản phẩm Web Application tương tác thời gian thực** kiến trúc Client-Server chuẩn mực:  
  > - **Backend:** Đóng gói bằng Python, PyTorch, FastAPI / Flask, tối ưu hóa suy luận song song.  
  > - **Frontend:** Giao diện SPA hiện đại, đáp ứng đa nền tảng.  
  > - **Tính năng nổi bật:**  
  >   1. Cho phép người dùng chuyển đổi linh hoạt giữa 6 mô hình để so sánh đối đầu tức thì.  
  >   2. Tích hợp thanh đo Confidence Score trực quan.  
  >   3. Hiển thị **Bản đồ nhiệt Attention trực tiếp trên giao diện**, giải thích tường minh từ ngữ nào quyết định cảm xúc.  
  >   4. Cung cấp bộ dữ liệu mẫu đa dạng từ giáo dục, thương mại đến phim ảnh.  
  > Toàn bộ mã nguồn đã được container hóa bằng Docker, có thể triển khai lên Cloud chỉ bằng một lệnh duy nhất."

---

### SLIDE 23: 6. KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- **Thời lượng:** 19:00 - 19:45 (45 giây)
- **Người trình bày:** Diễn giả 3
- **Hành động:** Tóm tắt ngắn gọn, dứt khoát.
- **Lời thoại:**
  > "Em xin phép tổng kết lại 3 kết luận cốt lõi:  
  > 1. Nghiên cứu đã chứng minh thực nghiệm: Biểu diễn phân tán Word2Vec kết hợp học sâu chuỗi vượt trội hoàn toàn các phương pháp túi từ cổ điển.  
  > 2. Kiến trúc đề xuất **BiLSTM + Additive Attention** là giải pháp xuất sắc nhất về hiệu quả chi phí / hiệu năng: đạt F1-Score $91.54\%$, có khả năng giải thích XAI, độ trễ cực thấp 12ms và bền vững trước Domain Shift.  
  > 3. Về hướng phát triển, nhóm sẽ mở rộng sang bài toán **Phân loại cảm xúc dựa trên khía cạnh (ABSA)** và thử nghiệm kỹ thuật **Chưng cất tri thức (Knowledge Distillation)** từ PhoBERT sang BiLSTM để dung hòa ưu điểm của cả hai thế giới."

---

### SLIDE 24: TRANG KẾT THÚC (LỜI CẢM ƠN & Q&A)
- **Thời lượng:** 19:45 - 20:00 (15 giây)
- **Người trình bày:** Cả 3 thành viên cùng đứng nghiêm trang, cúi đầu chào.
- **Lời thoại (Diễn giả 1 kết thúc):**
  > "Kính thưa Thầy Lê Anh Cường và quý Hội đồng, trên đây là toàn bộ kết quả nghiên cứu giữa kỳ của nhóm học viên K36 chúng em. Nhóm xin bày tỏ lòng biết ơn sâu sắc đến Thầy đã tận tình định hướng và truyền đạt kiến thức trong suốt học phần.  
  > Chúng em rất mong nhận được những nhận xét, góp ý quý báu từ Thầy và quý thầy cô để công trình được hoàn thiện hơn nữa. Nhóm xin chân thành cảm ơn và sẵn sàng lắng nghe các câu hỏi phản biện ạ!"

---

<a name="phan-ii-kich-ban-bai-giang-nhap-mon"></a>
# PHẦN II: KỊCH BẢN BÀI GIẢNG NHẬP MÔN NLP CHO NGƯỜI MỚI BẮT ĐẦU
- **Mục tiêu:** Giảng dạy chuyên đề nhập môn cho sinh viên / người mới học NLP, đảm bảo người nghe không có nền tảng toán sâu vẫn nắm vững bản chất, tự tay tính được 3 bài tập nòng cốt.
- **Slide tương ứng:** `Bai_Giang_Nhap_Mon_NLP_Word2Vec_LSTM.pdf` (22 slide).
- **Thời lượng:** 25 - 30 phút.
- **Phong cách sư phạm:** Thân thiện, đối thoại hai chiều, dùng hình ảnh ẩn dụ sinh động (Bản đồ GPS, Ba lô trí nhớ, Bút dạ quang).

---

### SLIDE 1 & 2: MỞ ĐẦU & LỘ TRÌNH 6 NẤC THANG NHẬN THỨC
- **Lời giảng:**
  > "Chào tất cả các bạn! Chào mừng các bạn đến với chuyên đề: *'Từ Word2Vec Đến LSTM và Cơ Chế Chú Ý: Giải Mã Cách Máy Tính Thấu Cảm Ngôn Ngữ'*.  
  > Các bạn có bao giờ tự hỏi: Chiếc điện thoại hay máy tính của chúng ta vốn dĩ chỉ hiểu được các con số 0 và 1, vậy làm sao ChatGPT hay các bộ lọc bình luận của Shopee, TikTok có thể hiểu được câu nói của chúng ta là khen hay chê?  
  > 
  > Trong 30 phút hôm nay, chúng ta sẽ cùng nhau leo lên **6 nấc thang nhận thức**:  
  > Bắt đầu từ câu hỏi sơ khai nhất, qua các cách đếm từ cổ điển, cuộc cách mạng tọa độ hóa ngôn ngữ Word2Vec, chiếc ba lô trí nhớ LSTM, cây bút dạ quang Attention, và đặc biệt: **chính tay các bạn sẽ giải 3 bài toán mẫu ngay trên slide** để hiểu sâu tận gốc rễ vấn đề!"

---

### SLIDE 3 & 4: TẠI SAO MÁY TÍNH KHÔNG ĐỌC ĐƯỢC CHỮ & MÃ HÓA ONE-HOT
- **Lời giảng:**
  > "Các bạn hãy nhìn vào màn hình: Con người đọc chữ 'Yêu' thì trái tim rung động, nhưng với vi xử lý máy tính, chữ 'Yêu' hay 'Ghét' chỉ là một chuỗi các byte nhị phân vô tri.  
  > 
  > Cách làm ngây thơ đầu tiên: **Mã hóa One-Hot**.  
  > Giả sử từ điển của chúng ta có 10.000 từ. Từ 'học_sinh' ở vị trí số 1, ta gán vector [1, 0, 0, ... 0]. Từ 'sinh_viên' ở vị trí số 2, ta gán [0, 1, 0, ... 0].  
  > Nhìn thì có vẻ hợp lý, nhưng nó có 2 thảm họa:  
  > 1. Vector quá dài và chứa toàn số 0 lãng phí bộ nhớ.  
  > 2. Đau đớn nhất: Nếu các bạn lấy tích vô hướng giữa vector 'học_sinh' và 'sinh_viên', kết quả bằng mấy? Bằng 0! Nghĩa là máy tính nghĩ 'học sinh' và 'sinh viên' chẳng có chút họ hàng nào với nhau cả!"

---

### SLIDE 5: TF-IDF: ĐẾM TỪ THÔNG MINH HƠN
- **Lời giảng:**
  > "Sau đó người ta nghĩ ra **TF-IDF**. Ý tưởng rất thực tế:  
  > - Một từ xuất hiện nhiều lần trong bài (TF cao) thì có thể nó quan trọng.  
  > - Nhưng nếu từ đó xuất hiện trong *tất cả mọi bài viết* trên đời (như từ 'và', 'thì', 'là') thì giá trị phân biệt của nó bằng 0 (IDF bị phạt nặng).  
  > TF-IDF biết cách trừ điểm từ rác, nhưng nó vẫn là 'Túi đựng từ' (Bag-of-Words): Bỏ hết từ vào một cái bao rồi xóc lên. Câu *'Tôi yêu em chứ không ghét em'* và *'Tôi ghét em chứ không yêu em'* bị TF-IDF đếm ra kết quả giống hệt nhau! Rõ ràng ta cần một bước đột phá mạnh mẽ hơn."

---

### SLIDE 6 & 7: CUỘC CÁCH MẠNG WORD2VEC & HAI TRÒ CHƠI ĐOÁN CHỮ
- **Lời giảng:**
  > "Vào năm 2013, tiến sĩ Tomas Mikolov đã tạo nên cơn địa chấn mang tên **Word2Vec**.  
  > Triết lý của nó đơn giản đến kinh ngạc: **'Hãy cho tôi biết bạn bè xung quanh bạn là ai, tôi sẽ nói cho bạn biết bạn là người như thế nào!'**  
  > 
  > Word2Vec biến mỗi từ thành một **tọa độ GPS** trong không gian nhiều chiều.  
  > Ví dụ không gian 2 chiều: Chiều ngang là 'Giới tính', chiều dọc là 'Quyền lực'.  
  > Từ 'Vua' sẽ ở tọa độ (Nam, Quyền lực cao). Từ 'Hoàng hậu' ở tọa độ (Nữ, Quyền lực cao).  
  > Và điều kỳ diệu xuất hiện: Nếu ta lấy vector **'Vua' trừ đi 'Đàn ông' rồi cộng thêm 'Phụ nữ'**, tọa độ thu được sẽ rơi trúng ngay cạnh từ **'Hoàng hậu'**! Ngôn ngữ đã có thể cộng trừ nhân chia như hình học!"

---

### SLIDE 8: TỪ VECTOR TỪ ĐẾN CẢ CÂU: CẠM BẪY CỘNG TRUNG BÌNH
- **Lời giảng:**
  > "Có tọa độ từng từ rồi, làm sao gom thành tọa độ của cả một câu?  
  > Cách dễ nhất là lấy trung bình cộng tất cả các từ trong câu (Average Word2Vec). Nhưng đây là một **cạm bẫy chết người**!  
  > Các bạn nhìn câu này:  
  > *Câu A: 'Phim này hay chứ không dở' (Tích cực).*  
  > *Câu B: 'Phim này dở chứ không hay' (Tiêu cực).*  
  > Nếu lấy trung bình cộng, hai câu này có chung tập từ vựng, vector đại diện của chúng sẽ trùng khít lên nhau! Máy tính hoàn toàn bị lừa.  
  > Muốn hiểu đúng, máy tính phải có khả năng **đọc tuần tự từng từ một và ghi nhớ ngữ cảnh**. Đó là lúc LSTM xuất hiện!"

---

### SLIDE 9 & 10: MẠNG NƠ-RON HỒI QUY LSTM & BILSTM
- **Lời giảng:**
  > "Các bạn hãy tưởng tượng mô hình **LSTM** giống như một bạn học sinh đeo **chiếc ba lô trí nhớ** đi dọc theo câu văn:  
  > - Khi bước qua từ 'Phim', bạn cất vào ba lô một ít thông tin.  
  > - Khi bước qua từ 'này', không quan trọng lắm, bỏ qua.  
  > - Chiếc ba lô này cực kỳ thông minh nhờ có **3 cánh cổng**:  
  >   1. **Cổng quên (Forget Gate):** Gặp từ chuyển hướng như 'Tuy nhiên', 'Nhưng', cổng quên mở ra để vứt bớt ký ức khen ngợi phía trước đi, chuẩn bị đón nhận lời chê!  
  >   2. **Cổng vào (Input Gate):** Chọn lọc nạp thông tin mới vào ba lô.  
  >   3. **Cổng ra (Output Gate):** Quyết định lấy thông tin gì trong ba lô ra ngoài để đánh giá.  
  > 
  > Và để hiểu sâu hơn, ta cho 2 bạn học sinh cùng đi: Một bạn đọc từ đầu câu đến cuối câu, một bạn đọc giật lùi từ cuối câu lên đầu câu. Đó chính là **BiLSTM (Mạng LSTM 2 chiều)**. Không một chi tiết ngữ cảnh nào có thể lọt khỏi mắt BiLSTM!"

---

### SLIDE 11 & 12: CƠ CHẾ CHÚ Ý (ATTENTION) & PHOBERT
- **Lời giảng:**
  > "BiLSTM đã rất giỏi, nhưng nó vẫn có một tật xấu: Đọc đến cuối câu thì đầu óc bị mệt mỏi, dễ quên mất những từ đắt giá ở tít đầu câu.  
  > 
  > Để khắc phục, các nhà khoa học trang bị cho mô hình một **Cây bút dạ quang (Cơ chế Chú ý - Attention Mechanism)**!  
  > Thay vì chỉ nhìn vào trạng thái ở bước cuối cùng, mô hình cầm bút dạ quang quét qua cả câu và tô đậm những từ quan trọng:  
  > Trong câu *'Món ăn ở đây ngon dã man nhưng thái độ phục vụ quá tệ'*, cây bút dạ quang sẽ tô sáng rực từ **'ngon dã man'** và **'quá tệ'**, còn các từ như 'ở đây', 'thái độ' thì mờ nhạt.  
  > Điểm số tô màu đó chính là **Trọng số chú ý $\alpha$**. Mô hình gom các từ quan trọng lại để đưa ra kết luận chuẩn xác 100%.  
  > 
  > Và đỉnh cao ngày nay chính là **PhoBERT** - mô hình ngôn ngữ lớn chuyên sâu cho tiếng Việt dựa trên Transformer, sử dụng hàng chục cây bút dạ quang cùng lúc!"

---

### SLIDE 13: BÀI TẬP GIẢI TAY 1: PHÉP TOÁN VECTOR & COSINE (WORD2VEC)
*(Trọng tâm sư phạm - Giảng chậm rãi, hướng dẫn sinh viên tính từng bước)*
- **Lời giảng:**
  > "Bây giờ, chúng ta hãy tạm gác máy tính sang một bên và cùng nhau làm một bài toán giải tay cực kỳ thú vị ngay trên Slide 13!  
  > 
  > **Đề bài:** Ta có không gian vector 2 chiều biểu diễn 4 từ:  
  > - Từ 'Vua': $v_1 = [0.9, 0.8]$  
  > - Từ 'Nam': $v_2 = [0.8, 0.1]$  
  > - Từ 'Nữ': $v_3 = [0.1, 0.8]$  
  > - Từ ứng viên 'Hoàng_hậu': $v_4 = [0.2, 0.9]$  
  > 
  > **Yêu cầu:** Hãy tính vector truy vấn $v_{query} = v_{\text{Vua}} - v_{\text{Nam}} + v_{\text{Nữ}}$ và đo độ tương đồng Cosine với 'Hoàng_hậu'.  
  > 
  > **Bước 1: Trừ và cộng tọa độ thông thường:**  
  > $$v_{query} = [0.9, 0.8] - [0.8, 0.1] + [0.1, 0.8]$$  
  > - Tọa độ $x = 0.9 - 0.8 + 0.1 = 0.2$  
  > - Tọa độ $y = 0.8 - 0.1 + 0.8 = 1.5$  
  > Vậy ta có vector kết quả là: $v_{query} = [0.2, 1.5]$.  
  > 
  > **Bước 2: Đo góc Cosine Similarity giữa $v_{query}$ và 'Hoàng_hậu' ($v_4 = [0.2, 0.9]$):**  
  > - Tích vô hướng trên tử số:  
  >   $$v_{query} \cdot v_4 = (0.2 \times 0.2) + (1.5 \times 0.9) = 0.04 + 1.35 = 1.39$$  
  > - Độ dài vector ở mẫu số:  
  >   $$\|v_{query}\| = \sqrt{0.2^2 + 1.5^2} = \sqrt{0.04 + 2.25} = \sqrt{2.29} \approx 1.513$$  
  >   $$\|v_4\| = \sqrt{0.2^2 + 0.9^2} = \sqrt{0.04 + 0.81} = \sqrt{0.85} \approx 0.922$$  
  > - Nhân hai độ dài lại: $1.513 \times 0.922 \approx 1.395$.  
  > - Chia tử cho mẫu:  
  >   $$\text{Cosine} = \frac{1.39}{1.395} \approx \mathbf{0.9964}!$$  
  > 
  > Các bạn thấy không? Độ tương đồng Cosine đạt tới **$0.9964$**, xấp xỉ tuyệt đối bằng $1.0$!  
  > Điều này chứng minh bằng toán học rằng: Máy tính hoàn toàn có thể hiểu được suy luận tương đương: *Vua trừ Nam cộng Nữ chính là Hoàng Hậu!*"

---

### SLIDE 14: BÀI TẬP GIẢI TAY 2: CỔNG QUÊN LSTM (XÓA BỎ KÝ ỨC CŨ)
- **Lời giảng:**
  > "Rất tuyệt vời! Bây giờ chúng ta cùng bước sang Bài tập 2 ở Slide 14: Khám phá bí mật bên trong chiếc ba lô LSTM.  
  > 
  > **Đề bài:** Mô hình đang đọc câu: *'Khóa học này rất bổ ích nhưng...'*.  
  > Ký ức tích lũy về lời khen trước đó là $C_{t-1} = 0.85$ (rất cao).  
  > Tại bước thời gian $t$, từ tiếp theo xuất hiện là từ **'nhưng'**.  
  > Ta có:  
  > - Vector từ 'nhưng' là $x_t = 1.5$.  
  > - Ẩn trạng thái trước đó $h_{t-1} = 0.5$.  
  > - Trọng số của cổng quên: $W_f = -2.0$, $U_f = -1.0$, và hệ số chệch $b_f = 0.5$.  
  > 
  > Hãy tính xem cổng quên $f_t$ mở ra bao nhiêu, và ký ức cũ sau khi qua cổng quên còn lại bao nhiêu?  
  > 
  > **Bước 1: Tính giá trị kích hoạt tuyến tính $z_f$:**  
  > $$z_f = W_f x_t + U_f h_{t-1} + b_f = (-2.0 \times 1.5) + (-1.0 \times 0.5) + 0.5$$  
  > $$z_f = -3.0 - 0.5 + 0.5 = \mathbf{-3.0}$$  
  > 
  > **Bước 2: Đưa qua hàm Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ để ép về đoạn [0, 1]:**  
  > $$f_t = \sigma(-3.0) = \frac{1}{1 + e^{3.0}} \approx \frac{1}{1 + 20.0855} \approx \mathbf{0.0474} \approx 4.74\%!$$  
  > 
  > **Bước 3: Cập nhật ký ức cũ:**  
  > $$C_{t}^{\text{old}} = f_t \times C_{t-1} = 0.0474 \times 0.85 \approx \mathbf{0.040}$$  
  > 
  > **Ý nghĩa sư phạm ở đây là gì?**  
  > Cổng quên $f_t$ chỉ cho phép **$4.74\%$** ký ức đi qua! Hơn **$95\%$ ký ức khen ngợi cũ đã bị xóa sạch** ngay khi từ 'nhưng' xuất hiện! Nhờ vậy, mô hình sẽ không bị ấn tượng ban đầu đánh lừa, mà tập trung toàn bộ tâm trí để đón nhận lời phàn nàn phía sau. Đây chính là vẻ đẹp cơ học của toán học trong AI!"

---

### SLIDE 15: BÀI TẬP GIẢI TAY 3: CƠ CHẾ ATTENTION (SOFTMAX & TỔNG TRỌNG SỐ)
- **Lời giảng:**
  > "Và bài tập thứ 3 ở Slide 15: Cách cây bút dạ quang hoạt động bằng công thức Softmax.  
  > 
  > **Đề bài:** Một câu gồm 3 từ: *'Món này tệ'*.  
  > Vector biểu diễn qua BiLSTM của 3 từ lần lượt là:  
  > - $h_1 = [1, 0]$ (Món)  
  > - $h_2 = [0, 1]$ (này)  
  > - $h_3 = [2, 3]$ (tệ)  
  > Giả sử sau khi chiếu qua vector ngữ cảnh, ta tính được điểm số quan trọng thô (Score) là:  
  > $s_1 = 0.5$, $s_2 = 0.1$, $s_3 = 2.5$.  
  > 
  > Hãy chuẩn hóa Softmax để tìm trọng số chú ý $\alpha_1, \alpha_2, \alpha_3$ và tính Vector đại diện của cả câu $v_D$!  
  > 
  > **Bước 1: Tính số mũ $e^{s_i}$ của từng từ:**  
  > - $e^{0.5} \approx 1.649$  
  > - $e^{0.1} \approx 1.105$  
  > - $e^{2.5} \approx 12.182$  
  > Tổng số mũ mẫu số $\sum = 1.649 + 1.105 + 12.182 = \mathbf{14.936}$.  
  > 
  > **Bước 2: Tính tỷ lệ phần trăm (Trọng số chú ý $\alpha$):**  
  > - $\alpha_1 = \frac{1.649}{14.936} \approx \mathbf{0.110}$ ($11.0\%$)  
  > - $\alpha_2 = \frac{1.105}{14.936} \approx \mathbf{0.074}$ ($7.4\%$)  
  > - $\alpha_3 = \frac{12.182}{14.936} \approx \mathbf{0.816}$ (**$81.6\%$**!)  
  > 
  > **Bước 3: Lấy tổng có trọng số để tìm Vector câu $v_D$:**  
  > $$v_D = \alpha_1 h_1 + \alpha_2 h_2 + \alpha_3 h_3$$  
  > $$v_D = 0.110 \times [1, 0] + 0.074 \times [0, 1] + 0.816 \times [2, 3]$$  
  > - Tọa độ $x = (0.110 \times 1) + 0 + (0.816 \times 2) = 0.110 + 1.632 = \mathbf{1.742}$  
  > - Tọa độ $y = 0 + (0.074 \times 1) + (0.816 \times 3) = 0.074 + 2.448 = \mathbf{2.522}$  
  > 
  > **Kết luận:** Vector câu $v_D = [1.742, 2.522]$ mang đậm dấu ấn của từ 'tệ' (chiếm hơn $81\%$). Khi đưa vector này vào bộ phân loại, mô hình sẽ ngay lập tức khẳng định câu này mang cảm xúc Tiêu cực với độ tự tin áp đảo! Không cần đoán mò, mọi thứ đều minh bạch rõ ràng!"

---

### SLIDE 16 ĐẾN 22: KẾT QUẢ THỰC NGHIỆM, DEMO & 5 BÀI HỌC CỐT LÕI
- **Lời giảng:**
  > "Từ lý thuyết và bài tập giải tay, khi đưa lên máy tính huấn luyện trên hàng chục nghìn câu văn tiếng Việt thực tế:  
  > - Các mô hình cổ điển chỉ đạt khoảng 82 - 85%.  
  > - Khi có BiLSTM kết hợp Attention, độ chính xác đạt tới **$91.68\%$**, vượt qua hầu hết các ca khó đời thường.  
  > - Và khi thử nghiệm trên hệ thống Web Demo thực tế (Slide 21), hệ thống phản hồi kết quả và vẽ bản đồ nhiệt chỉ trong vòng 12 phần nghìn giây!  
  > 
  > Để kết thúc bài học hôm nay, các bạn chỉ cần ghi nhớ **5 bài học vàng** ở Slide 22:  
  > 1. Máy tính không hiểu chữ, muốn hiểu phải chuyển thành vector số thực.  
  > 2. Word2Vec biến từ thành tọa độ không gian, từ giống nhau thì đứng gần nhau.  
  > 3. Đừng cộng trung bình vector các từ vì sẽ đánh mất thứ tự và ngữ pháp.  
  > 4. LSTM dùng ba lô có 3 cánh cổng để nhớ lâu và biết quên đúng lúc.  
  > 5. Cơ chế Attention là cây bút dạ quang giúp tô đậm từ khóa đắt giá, biến AI thành chiếc hộp kính minh bạch!  
  > 
  > Cảm ơn tất cả các bạn đã chú ý lắng nghe và tham gia giải bài tập rất sôi nổi!"

---

<a name="phan-iii-bo-cau-hoi-phan-bien"></a>
# PHẦN III: BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU & CÂU TRẢ LỜI MẪU (Q&A DEFENSE)

Đây là 10 câu hỏi "hóc búa" nhất mà Hội đồng khoa học (đặc biệt là Thầy PGS.TS. Lê Anh Cường) có khả năng cao sẽ chất vấn nhóm, kèm câu trả lời mẫu chuẩn xác về mặt toán học và học thuật.

---

### CÂU HỎI 1: Tại sao trong hàm mục tiêu Negative Sampling của Word2Vec, phân phối rút mẫu từ âm bản lại lấy lũy thừa 3/4 ($P_n(w) \propto f(w)^{3/4}$) mà không phải là phân phối đều hay tần suất thực tế?
- **Trả lời phản biện chuẩn:**
  > "Kính thưa Thầy và Hội đồng, đây là một phát kiến thực nghiệm cực kỳ thông minh của Mikolov:  
  > - Nếu ta dùng phân phối đều ($1/|V|$), các từ cực hiếm sẽ bị chọn quá nhiều, gây nhiễu cho mô hình.  
  > - Nếu dùng tần suất thực tế ($f(w)$), các từ dừng (Stopwords) xuất hiện quá dày đặc như 'và', 'thì', 'là' sẽ chiếm trọn các mẫu âm bản, khiến mô hình chỉ học cách phân biệt từ ngữ cảnh với từ dừng mà không học được các từ hiếm khác.  
  > - Phép biến đổi lũy thừa $\frac{3}{4} = 0.75$ đóng vai trò làm mịn phân phối (Smoothing): Nó làm giảm xác suất chọn của các từ cực kỳ phổ biến và tăng xác suất chọn của các từ hiếm một cách tương đối.  
  > *Ví dụ cụ thể:* Giả sử từ 'và' có tần suất $0.9$, từ 'hacker' có tần suất $0.0016$. Tỷ lệ tần suất là $\frac{0.9}{0.0016} \approx 562$ lần. Nhưng qua lũy thừa $0.75$:  
  > $0.9^{0.75} \approx 0.924$, trong khi $0.0016^{0.75} = 0.008$. Tỷ lệ lúc này chỉ còn $\frac{0.924}{0.008} \approx 115$ lần (tăng cơ hội xuất hiện của từ hiếm lên gần 5 lần), giúp không gian biểu diễn của các từ hiếm được cập nhật gradient đầy đủ."

---

### CÂU HỎI 2: Word2Vec gặp vấn đề OOV (Out-of-Vocabulary) rất nặng nề khi gặp từ mới trong kiểm thử. Mô hình của nhóm giải quyết vấn đề OOV như thế nào?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, đối với mô hình BiLSTM + Word2Vec, nhóm áp dụng 2 chiến lược:  
  > 1. Ở khâu tiền xử lý, nhóm dùng bộ tách từ tiếng Việt chuẩn hóa và thay thế tất cả các từ xuất hiện dưới ngưỡng tần suất $min\_count < 2$ bằng token đặc biệt `<UNK>`. Vector `<UNK>` được khởi tạo ngẫu nhiên và cho phép học trọng số trong quá trình huấn luyện.  
  > 2. Trong kiến trúc thực tế nâng cao, nhóm đề xuất giải pháp tích hợp FastText hoặc Tokenizer BPE (Byte-Pair Encoding) của PhoBERT. FastText chia nhỏ từ thành các n-gram ký tự (subwords). Khi gặp một từ mới chưa từng thấy như 'ship_nhanhhh', mô hình vẫn trích xuất được n-gram 'ship' và 'nhanh' để tổng hợp vector, triệt tiêu hoàn toàn điểm yếu OOV của Word2Vec truyền thống."

---

### CÂU HỎI 3: Tại sao lại cần BiLSTM (hai chiều) mà không phải là LSTM một chiều đơn thuần cho bài toán phân loại cảm xúc?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, ngôn ngữ tự nhiên có tính phụ thuộc hai chiều rất mạnh mẽ.  
  > Trong LSTM một chiều xuôi, tại bước thời gian $t$, ẩn trạng thái $\overrightarrow{h}_t$ chỉ chứa thông tin của các từ phía trước nó mà hoàn toàn 'mù' về những gì sẽ diễn ra phía sau.  
  > *Ví dụ:* Trong câu *'Bộ phim không thể nào chê vào đâu được'*:  
  > Khi đọc đến từ 'không', nếu là LSTM một chiều, nó chỉ thấy từ phủ định và có xu hướng nghiêng về tiêu cực. Nhưng với BiLSTM, luồng đọc ngược $\overleftarrow{h}_t$ từ cuối câu lên đã bắt gặp cụm 'chê vào đâu được' và chuyển tiếp về từ 'không'. Khi ghép hai vector $[\overrightarrow{h}_t; \overleftarrow{h}_t]$, mô hình nhận diện chính xác toàn bộ cấu trúc thành ngữ mang ý khen ngợi tuyệt đối."

---

### CÂU HỎI 4: Điểm khác biệt bản chất giữa Additive Self-Attention (Bahdanau) được nhóm sử dụng trong BiLSTM và Scaled Dot-Product Attention trong Transformer (PhoBERT) là gì?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, có hai điểm khác biệt cốt lõi:  
  > 1. **Về mặt công thức tính điểm tương quan:**  
  >    - Additive Attention dùng mạng truyền thẳng phi tuyến với hàm kích hoạt Tanh: $u_t = \tanh(W_a h_t + b_a)$ rồi nhân vô hướng với vector ngữ cảnh học được $u_s$.  
  >    - Scaled Dot-Product Attention dùng trực tiếp phép nhân ma trận giữa Query và Key rồi chia cho hệ số co giãn $\sqrt{d_k}$: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.  
  > 2. **Về độ phức tạp và khả năng song song hóa:**  
  >    - Additive Attention trong BiLSTM phụ thuộc vào bước lặp tuần tự $t$ của RNN nên không thể tính song song hoàn toàn.  
  >    - Scaled Dot-Product Attention trong Transformer loại bỏ hoàn toàn tính tuần tự, cho phép nhân ma trận trên GPU đồng thời cho toàn bộ các cặp từ trong câu, từ đó mở rộng quy mô lên hàng trăm tầng và hàng tỷ tham số."

---

### CÂU HỎI 5: Tại sao trong PhoBERT, nhóm lại dùng vector tại token `[CLS]` để phân loại thay vì lấy trung bình cộng tất cả các token đầu ra (Average Pooling)?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, trong kiến trúc BERT và PhoBERT:  
  > - Token `[CLS]` luôn được đặt ở vị trí đầu tiên của mọi chuỗi văn bản.  
  > - Qua 12 tầng Transformer Encoder, thông qua cơ chế Multi-Head Self-Attention, vector tại vị trí `[CLS]` được phép 'chú ý' đến tất cả các token còn lại trong chuỗi và tích lũy thông tin biểu diễn tổng hợp của toàn câu.  
  > - Trong khi các token từ vựng khác có xu hướng chuyên biệt hóa ngữ nghĩa cục bộ cho bài toán Masked Language Modeling (MLM), token `[CLS]` được thiết kế chuyên biệt và tối ưu hóa cho tác vụ phân loại cấp độ câu (Sentence-level Classification).  
  > Thực nghiệm của các tác giả BERT cũng như kiểm chứng của nhóm cho thấy việc dùng trực tiếp vector `[CLS]` kết hợp một tầng Dropout và Linear Classifier cho độ chính xác cao hơn và ổn định hơn so với Average Pooling."

---

### CÂU HỎI 6: Hiện tượng Domain Gap (dịch chuyển miền) khiến mô hình sụt giảm hiệu năng trên tập E-Commerce. Hãy giải thích nguyên nhân sâu xa và cách khắc phục?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, sự sụt giảm hiệu năng từ $91.68\%$ xuống $84.10\%$ khi chuyển từ UIT-VSFC sang E-Commerce xuất phát từ 2 nguyên nhân:  
  > 1. **Sự dịch chuyển từ vựng (Covariate Shift):** Dữ liệu UIT-VSFC dùng ngôn từ sư phạm chuẩn mực ('giảng viên', 'nhiệt tình', 'giáo trình'). Ngược lại, E-Commerce chứa mật độ rất lớn từ lóng, teencode viết tắt ('shop', 'đóng gói', 'hàng auth', 'pha-ke', 'ship nhanh vãi'). Các từ này không có trong phân phối huấn luyện ban đầu.  
  > 2. **Sự đảo lộn phân phối độ dài văn bản:** Đánh giá E-Commerce thường rất ngắn hoặc kèm theo icon cảm xúc (Emoji).  
  > *Giải pháp khắc phục:* Nhóm đề xuất áp dụng kỹ thuật **Học thích nghi miền (Domain Adaptation)** hoặc **Tiếp tục huấn luyện trước không giám sát (Continued Pre-training / Fine-tuning)** trên kho ngữ liệu E-Commerce không gán nhãn bằng kỹ thuật Masked Language Model trước khi phân loại."

---

### CÂU HỎI 7: Tại sao hàm kích hoạt trong các cổng của LSTM lại dùng Sigmoid ($\sigma$), trong khi trạng thái ứng viên ($\tilde{C}_t$) lại dùng Tanh?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, đây là thiết kế toán học mang tính nguyên lý:  
  > - **Hàm Sigmoid có miền giá trị trong khoảng $[0, 1]$**: Nó đóng vai trò như một chiếc van điều tiết tỷ lệ phần trăm (0 nghĩa là đóng hoàn toàn, không cho thông tin đi qua; 1 nghĩa là mở hoàn toàn, bảo toàn $100\%$ dữ liệu). Do đó, các cổng (quên, vào, ra) bắt buộc phải dùng Sigmoid để thực hiện phép lọc.  
  > - **Hàm Tanh có miền giá trị đối xứng $[-1, 1]$**: Nó đóng vai trò tạo ra giá trị thông tin mới có thể mang dấu âm hoặc dương, cho phép mô hình tăng cường (cộng thêm) hoặc ức chế (trừ bớt) nội dung ngữ nghĩa trong Cell State $C_t$. Đồng thời miền giá trị đối xứng quanh số 0 giúp quá trình lan truyền ngược gradient không bị lệch dấu (Zero-centered)."

---

### CÂU HỎI 8: Với bài toán Sarcasm (Mỉa mai / Châm biếm), mô hình học sâu hiện nay gần như bất lực. Nhóm có hướng giải quyết cụ thể nào không?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, hiện tượng Sarcasm xảy ra khi có sự xung đột giữa ngữ nghĩa bề mặt (Literal meaning) và ý định thực sự (Pragmatics).  
  > Để giải quyết, mô hình không thể chỉ dựa vào một câu đơn lẻ mà cần 3 nguồn thông tin bổ trợ:  
  > 1. **Dữ liệu ngữ cảnh ngoài văn bản (Contextual Metadata):** Ví dụ điểm số sao đánh giá (Rating: 1 sao nhưng bình luận 'quá tuyệt vời'). Việc kết hợp số sao với văn bản sẽ phát hiện nghịch lý ngay lập tức.  
  > 2. **Đồ thị tri thức (Knowledge Graph):** Cung cấp chuẩn mực thực tế (Thời gian giao hàng chuẩn là 2-3 ngày; nếu văn bản ghi 'giao hàng 1 tháng' đi cùng 'nhanh', tri thức đồ thị sẽ gắn cờ mâu thuẫn).  
  > 3. **Mô hình ngôn ngữ lớn (LLMs) với cơ chế Chain-of-Thought (CoT):** Hướng dẫn mô hình suy luận từng bước về tính hợp lý của câu nói trước khi ra kết luận."

---

### CÂU HỎI 9: Độ đo AUC-ROC phản ánh điều gì vượt trội hơn so với Accuracy trong thực nghiệm của nhóm?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, trong bài toán phân loại cảm xúc thực tế, dữ liệu thường bị **mất cân bằng nhãn (Class Imbalance)**: ví dụ số mẫu Tích cực thường chiếm $70-80\%$, trong khi mẫu Tiêu cực chỉ chiếm $20\%$.  
  > - Nếu một mô hình ngây thơ luôn dự đoán là Tích cực, Accuracy vẫn có thể đạt $80\%$, tạo ra ảo tưởng về độ chính xác.  
  > - **Độ đo AUC-ROC (Area Under the Receiver Operating Characteristic Curve)** đánh giá năng lực phân tách thực sự của mô hình qua mọi ngưỡng cắt xác suất (Classification Thresholds), đo lường sự cân đối giữa Tỷ lệ dương tính thật (TPR) và Tỷ lệ dương tính giả (FPR).  
  > Việc mô hình BiLSTM + Attention đạt **AUC-ROC = 0.9587** chứng minh rằng xác suất mô hình xếp hạng một mẫu Tích cực ngẫu nhiên cao hơn một mẫu Tiêu cực ngẫu nhiên đạt tới $95.87\%$, độc lập hoàn toàn với việc ngưỡng quyết định được đặt ở mức nào."

---

### CÂU HỎI 10: Nhóm đã kiểm tra hiện tượng Overfitting của mạng BiLSTM + Attention như thế nào trong quá trình huấn luyện?
- **Trả lời phản biện chuẩn:**
  > "Dạ thưa Thầy, nhóm kiểm soát hiện tượng Overfitting qua 4 cơ chế bảo vệ:  
  > 1. **Kỹ thuật Dropout ($p = 0.3$ đến $0.5$):** Áp dụng ngắt ngẫu nhiên các kết nối nơ-ron sau tầng Embedding và giữa các tầng BiLSTM.  
  > 2. **Early Stopping:** Theo dõi chặt chẽ hàm mất mát (Validation Loss) trên tập kiểm định; nếu sau 3-5 epoch mà Validation Loss không giảm, quá trình huấn luyện sẽ tự động dừng lại và khôi phục bộ trọng số tốt nhất.  
  > 3. **L2 Regularization (Weight Decay):** Phạt các trọng số có độ lớn quá cao trong hàm tối ưu Adam.  
  > 4. **Theo dõi đồ thị Learning Curves:** Kiểm tra đường Loss của Training và Validation luôn hội tụ song song, khoảng cách chênh lệch không vượt quá $0.05$, đảm bảo mô hình có khả năng khái quát hóa cao trên dữ liệu chưa từng thấy."

---

<a name="phan-iv-checklist-ky-thuat"></a>
# PHẦN IV: CHECKLIST KỸ THUẬT VÀ PHÒNG NGỪA RỦI RO TRƯỚC GIỜ G

Để buổi báo cáo diễn ra hoàn hảo không tì vết, các thành viên cần thực hiện đúng checklist sau:

| STT | Hạng mục kiểm tra | Chi tiết kỹ thuật | Người phụ trách | Trạng thái |
|:---:|:---|:---|:---:|:---:|
| 1 | **Tệp trình chiếu chính** | `Bai_Thuyet_Trinh_Giua_Ky_NLP.pdf` (24 slide, tỉ lệ 16:9, logo góc trái sắc nét) | Gia Huy | [X] Sẵn sàng |
| 2 | **Tệp bài giảng nhập môn** | `Bai_Giang_Nhap_Mon_NLP_Word2Vec_LSTM.pdf` (22 slide, đủ 3 bài tập giải tay) | Gia Huy | [X] Sẵn sàng |
| 3 | **Hệ thống Web Demo Local** | Chạy sẵn script `streamlit run streamlit_app.py` hoặc `python app.py` tại cổng `http://localhost:8501`, cache sẵn trọng số model để suy luận tức thì | Tấn Phát | [X] Sẵn sàng |
| 4 | **Dự phòng rủi ro mất mạng** | Chuẩn bị USB chứa toàn bộ slide PDF, video quay sẵn màn hình demo 60s chất lượng cao 1080p phòng khi máy chiếu không kết nối được web | Kim Ngân | [X] Sẵn sàng |
| 5 | **Bút trình chiếu (Clicker)** | Kiểm tra pin bút laser, thử lật slide Beamer mượt mà trên Adobe Acrobat Reader / Preview ở chế độ Full Screen | Tấn Phát | [X] Sẵn sàng |
| 6 | **Phân chia thời gian (Timer)** | Bật đồng hồ bấm giờ rung nhẹ ở mốc 15 phút để nhóm chủ động điều chỉnh tốc độ, không bao giờ để quá 20 phút | Kim Ngân | [X] Sẵn sàng |

---
*Tài liệu được biên soạn và chuẩn hóa phục vụ trực tiếp cho buổi Báo cáo Chuyên đề Cao học K36 — Khoa CNTT — Trường ĐH Sư phạm TP.HCM.*
