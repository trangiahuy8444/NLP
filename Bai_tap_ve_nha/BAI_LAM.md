# BÀI TẬP VỀ NHÀ: MÔ HÌNH NGÔN NGỮ

**Họ và tên:** ........................................................  
**Mã sinh viên:** ....................................................  
**Môn học:** Xử lý ngôn ngữ tự nhiên  

---

## 1. N-Gram và cách xây dựng mô hình N-Gram

### 1.1. N-Gram là gì?

N-Gram là một dãy gồm $N$ phần tử liên tiếp được lấy từ văn bản. Phần tử thường là từ hoặc token.

Ví dụ, với câu:

> Tôi học xử lý ngôn ngữ

Ta có:

- **Unigram (1-gram):** `Tôi`, `học`, `xử_lý`, `ngôn_ngữ`.
- **Bigram (2-gram):** `Tôi học`, `học xử_lý`, `xử_lý ngôn_ngữ`.
- **Trigram (3-gram):** `Tôi học xử_lý`, `học xử_lý ngôn_ngữ`.

Trong mô hình ngôn ngữ, N-Gram được dùng để xấp xỉ xác suất của từ tiếp theo. Giả định Markov cho rằng từ hiện tại chủ yếu phụ thuộc vào $N-1$ từ gần nhất:

$$
P(w_i\mid w_1,\ldots,w_{i-1})
\approx
P(w_i\mid w_{i-N+1},\ldots,w_{i-1}).
$$

Vì vậy:

- Unigram không dùng từ đứng trước.
- Bigram dùng một từ đứng trước.
- Trigram dùng hai từ đứng trước.

### 1.2. Xác suất của một câu

Theo quy tắc dây chuyền:

$$
P(w_1,\ldots,w_m)
=\prod_{i=1}^{m}P(w_i\mid w_1,\ldots,w_{i-1}).
$$

Với mô hình Bigram, công thức được xấp xỉ thành:

$$
P(w_1,\ldots,w_m)
\approx
P(w_1\mid \langle s\rangle)
\prod_{i=2}^{m}P(w_i\mid w_{i-1})
P(\langle /s\rangle\mid w_m).
$$

Trong đó $\langle s\rangle$ và $\langle /s\rangle$ lần lượt là ký hiệu bắt đầu và kết thúc câu.

Xác suất Bigram được ước lượng từ số lần xuất hiện:

$$
P(w_i\mid w_{i-1})
=\frac{C(w_{i-1},w_i)}{C(w_{i-1})}.
$$

Với Trigram:

$$
P(w_i\mid w_{i-2},w_{i-1})
=\frac{C(w_{i-2},w_{i-1},w_i)}
{C(w_{i-2},w_{i-1})}.
$$

### 1.3. Vấn đề xác suất bằng 0

Nếu một N-Gram không xuất hiện trong tập huấn luyện thì số đếm của nó bằng 0. Khi xác suất câu là tích của nhiều xác suất thành phần, chỉ một thành phần bằng 0 cũng khiến xác suất cả câu bằng 0.

Một cách xử lý đơn giản là **add-$k$ smoothing**:

$$
P_k(w_i\mid h)
=\frac{C(h,w_i)+k}{C(h)+k|V|},
$$

trong đó:

- $h$ là ngữ cảnh gồm $N-1$ từ.
- $|V|$ là kích thước từ vựng.
- $k>0$ là hệ số làm trơn.
- Khi $k=1$, phương pháp được gọi là Laplace smoothing.

### 1.4. Quy trình xây dựng N-Gram

#### Bước 1: Chuẩn bị corpus

Thu thập văn bản phù hợp với lĩnh vực sử dụng. Corpus cần đủ lớn và nên có cùng đặc trưng với dữ liệu mà mô hình sẽ gặp khi chạy thực tế.

#### Bước 2: Tiền xử lý

Các thao tác thường dùng:

1. Chuẩn hóa Unicode.
2. Chuyển chữ thường.
3. Chuẩn hóa khoảng trắng và dấu câu.
4. Tách văn bản thành câu.
5. Tách câu thành token.
6. Xử lý từ hiếm bằng token `<unk>`.
7. Thêm token bắt đầu `<s>` và kết thúc `</s>`.

Đối với tiếng Việt, cần chú ý rằng một từ có thể gồm nhiều tiếng, ví dụ “xử lý” hoặc “ngôn ngữ”. Trong bài minh họa, chương trình tách theo đơn vị chữ để giữ mã nguồn đơn giản. Khi làm trên dữ liệu thật, có thể dùng bộ tách từ tiếng Việt trước khi tạo N-Gram.

#### Bước 3: Tạo và đếm N-Gram

Với Bigram, ta đếm:

- Số lần mỗi ngữ cảnh $w_{i-1}$ xuất hiện.
- Số lần mỗi cặp $(w_{i-1},w_i)$ xuất hiện.

Với Trigram, ta đếm các ngữ cảnh $(w_{i-2},w_{i-1})$ và bộ ba $(w_{i-2},w_{i-1},w_i)$.

#### Bước 4: Tính xác suất

Dùng số đếm để tính xác suất có điều kiện. Nếu sử dụng smoothing thì cộng $k$ vào tử số và $k|V|$ vào mẫu số.

#### Bước 5: Dự đoán từ tiếp theo

Với một ngữ cảnh $h$, tính $P(w\mid h)$ cho các từ trong từ vựng rồi:

- Chọn từ có xác suất lớn nhất; hoặc
- Lấy mẫu theo phân phối xác suất để kết quả đa dạng hơn.

#### Bước 6: Tính xác suất câu

Thêm token bắt đầu và kết thúc, sau đó nhân các xác suất có điều kiện. Trong chương trình thực tế nên cộng log xác suất để tránh hiện tượng số quá nhỏ:

$$
\log P(w_1,\ldots,w_m)
=\sum_{i=1}^{m}\log P(w_i\mid h_i).
$$

#### Bước 7: Đánh giá bằng perplexity

Perplexity đo mức độ bất ngờ của mô hình trước dữ liệu kiểm thử:

$$
PP(W)
=\exp\left(
-\frac{1}{M}\sum_{i=1}^{M}\log P(w_i\mid h_i)
\right).
$$

Perplexity càng thấp thì mô hình dự đoán dữ liệu kiểm thử càng tốt. Chỉ nên so sánh các mô hình có cùng cách tách token, từ vựng và tập kiểm thử.

#### Bước 8: Sinh văn bản

Bắt đầu từ `<s>`, dự đoán hoặc lấy mẫu từ tiếp theo, cập nhật ngữ cảnh, rồi lặp lại cho tới `</s>` hoặc khi đạt số token tối đa.

### 1.5. Hướng dẫn chạy chương trình

Trong thư mục `Bai_tap_ve_nha` đã có:

- `ngram_language_model.py`: mã nguồn mô hình.
- `du_lieu_mau.txt`: corpus nhỏ để thử nghiệm.
- `Thuc_hanh_NGram.ipynb`: notebook hướng dẫn chạy từng bước.

Chương trình chỉ dùng thư viện chuẩn Python, không cần cài thêm gói.
Mặc định chương trình dùng add-$k$ với $k=0{,}1$. Muốn dùng đúng Laplace
smoothing, đặt tham số `--k 1`.

Tại thư mục bài tập, chạy Bigram:

    python3 ngram_language_model.py --n 2

Chạy Trigram:

    python3 ngram_language_model.py --n 3

Thử một câu khác:

    python3 ngram_language_model.py --n 2 --sentence "mô hình ngôn ngữ dự đoán từ tiếp theo"

Thay hệ số smoothing:

    python3 ngram_language_model.py --n 3 --k 0.1

Dùng Laplace smoothing:

    python3 ngram_language_model.py --n 2 --k 1

Dùng corpus khác:

    python3 ngram_language_model.py --corpus duong_dan_den_file.txt --n 2

Các thành phần quan trọng trong mã nguồn:

- `fit()`: tách câu, xây từ vựng và đếm N-Gram.
- `xac_suat()`: tính $P(w_i\mid h_i)$ với add-$k$ smoothing.
- `top_tu_tiep_theo()`: dự đoán những từ tiếp theo có xác suất cao.
- `log_xac_suat_cau()`: tính log xác suất của câu.
- `perplexity()`: đánh giá mô hình.
- `sinh_cau()`: sinh câu bằng phương pháp sampling.

### 1.6. Nhận xét về N-Gram

**Ưu điểm**

- Đơn giản, dễ cài đặt và dễ giải thích.
- Huấn luyện nhanh vì chủ yếu là đếm.
- Có thể hoạt động tốt trên miền hẹp nếu corpus đủ lớn.

**Hạn chế**

- Chỉ nhớ được ngữ cảnh ngắn.
- Số lượng N-Gram tăng nhanh khi $N$ tăng.
- Dữ liệu thưa khiến nhiều N-Gram chưa từng xuất hiện.
- Không hiểu quan hệ ngữ nghĩa giữa các từ tương tự.
- Bảng đếm có thể chiếm nhiều bộ nhớ.

---

## 2. Word2Vec và mô hình LSTM dùng cho mô hình ngôn ngữ

### 2.1. Word2Vec

Word2Vec là phương pháp học **word embedding**, tức là ánh xạ mỗi từ thành một vector số thực có số chiều cố định:

$$
w\longmapsto \mathbf{e}_w\in\mathbb{R}^{d}.
$$

Word2Vec dựa trên giả thuyết phân bố: các từ xuất hiện trong ngữ cảnh giống nhau thường có ý nghĩa hoặc vai trò gần nhau. Sau huấn luyện, các từ có quan hệ ngữ nghĩa thường có vector gần nhau theo cosine similarity.

#### Hai kiến trúc chính

**CBOW – Continuous Bag of Words**

CBOW dùng các từ xung quanh để dự đoán từ ở giữa:

$$
P(w_t\mid w_{t-c},\ldots,w_{t-1},w_{t+1},\ldots,w_{t+c}).
$$

CBOW huấn luyện nhanh và hoạt động tốt với những từ xuất hiện nhiều.

**Skip-gram**

Skip-gram làm theo chiều ngược lại: dùng từ trung tâm để dự đoán các từ ngữ cảnh:

$$
\prod_{\substack{-c\le j\le c\\j\ne 0}}
P(w_{t+j}\mid w_t).
$$

Skip-gram thường học biểu diễn tốt hơn cho từ ít xuất hiện, nhưng thời gian huấn luyện có thể dài hơn.

Để giảm chi phí tính softmax trên từ vựng lớn, Word2Vec thường sử dụng **negative sampling**: mô hình học phân biệt cặp từ–ngữ cảnh thật với một số cặp âm được lấy mẫu.

#### Word2Vec có phải mô hình ngôn ngữ không?

Word2Vec tự thân **không phải một mô hình ngôn ngữ hoàn chỉnh**. Nó học vector của từ nhưng không trực tiếp tính xác suất của cả chuỗi và cũng không duy trì thứ tự đầy đủ của câu. Tuy nhiên, embedding Word2Vec có thể được dùng làm đầu vào cho RNN hoặc LSTM để xây dựng mô hình ngôn ngữ.

### 2.2. LSTM

LSTM (Long Short-Term Memory) là một biến thể của mạng nơ-ron hồi quy RNN. Mạng đọc chuỗi lần lượt từ trái sang phải và duy trì:

- Trạng thái ẩn $\mathbf{h}_t$, biểu diễn thông tin dùng để dự đoán.
- Trạng thái ô nhớ $\mathbf{c}_t$, lưu thông tin dài hạn.

Tại bước $t$, token $w_t$ được biến đổi thành embedding $\mathbf{x}_t$. LSTM cập nhật các cổng:

$$
\mathbf{f}_t
=\sigma(W_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f),
$$

$$
\mathbf{i}_t
=\sigma(W_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i),
$$

$$
\tilde{\mathbf{c}}_t
=\tanh(W_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c),
$$

$$
\mathbf{c}_t
=\mathbf{f}_t\odot\mathbf{c}_{t-1}
+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t,
$$

$$
\mathbf{o}_t
=\sigma(W_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o),
$$

$$
\mathbf{h}_t
=\mathbf{o}_t\odot\tanh(\mathbf{c}_t).
$$

Ý nghĩa của các cổng:

- **Forget gate** $\mathbf{f}_t$: chọn thông tin cũ cần quên.
- **Input gate** $\mathbf{i}_t$: chọn thông tin mới cần ghi.
- **Output gate** $\mathbf{o}_t$: chọn phần bộ nhớ dùng làm đầu ra.

Để dùng LSTM làm mô hình ngôn ngữ, trạng thái ẩn được đưa qua lớp tuyến tính và softmax:

$$
P(w_{t+1}\mid w_1,\ldots,w_t)
=\operatorname{softmax}(W\mathbf{h}_t+\mathbf{b}).
$$

Trong khi huấn luyện, đầu vào là các token từ đầu đến token áp chót; nhãn là cùng chuỗi nhưng dịch sang trái một vị trí. Hàm mất mát là cross-entropy:

$$
\mathcal{L}
=-\sum_t\log P(w_{t+1}\mid w_1,\ldots,w_t).
$$

Khi sinh văn bản, mô hình bắt đầu bằng token đầu câu, dự đoán token tiếp theo, đưa token đó trở lại mạng và tiếp tục cho đến token kết thúc.

### 2.3. Vai trò của Word2Vec trong LSTM

Ma trận embedding đầu vào của LSTM có thể:

1. Khởi tạo ngẫu nhiên và được học cùng mô hình ngôn ngữ; hoặc
2. Khởi tạo bằng vector Word2Vec đã huấn luyện trước.

Cách thứ hai có thể giúp khi tập dữ liệu mô hình ngôn ngữ nhỏ. Tuy nhiên, embedding học trực tiếp trong mô hình ngôn ngữ thường phù hợp hơn với chính nhiệm vụ dự đoán token tiếp theo.

### 2.4. So sánh N-Gram, Word2Vec và LSTM

| Tiêu chí | N-Gram | Word2Vec | LSTM Language Model |
|---|---|---|---|
| Mục tiêu chính | Ước lượng xác suất từ số đếm | Học vector biểu diễn từ | Dự đoán token tiếp theo |
| Có phải LM hoàn chỉnh? | Có | Không | Có |
| Thứ tự từ | Chỉ trong cửa sổ $N$ | Hạn chế, chủ yếu dựa trên ngữ cảnh | Được xử lý tuần tự |
| Ngữ cảnh | Cố định, ngắn | Cửa sổ cục bộ | Dài hơn N-Gram |
| Khả năng hiểu từ tương tự | Kém | Tốt | Tốt nhờ embedding |
| Dữ liệu thưa | Nghiêm trọng | Giảm nhờ vector | Giảm nhờ chia sẻ tham số |
| Chi phí huấn luyện | Thấp | Trung bình | Cao hơn |

---

## 3. Mô hình ngôn ngữ và lý do ChatGPT được gọi là mô hình ngôn ngữ

### 3.1. Mô hình ngôn ngữ là gì?

Mô hình ngôn ngữ là mô hình gán xác suất cho một chuỗi token hoặc dự đoán token tiếp theo dựa trên ngữ cảnh:

$$
P(w_1,\ldots,w_T)
=\prod_{t=1}^{T}P(w_t\mid w_1,\ldots,w_{t-1}).
$$

Một mô hình ngôn ngữ tốt sẽ gán xác suất cao cho các chuỗi tự nhiên và dự đoán chính xác token tiếp theo. Nhờ đó, nó có thể:

- Tự động hoàn thành câu.
- Sinh văn bản.
- Xếp hạng các phương án trong nhận dạng tiếng nói và dịch máy.
- Hỗ trợ tóm tắt, hỏi đáp và hội thoại.

### 3.2. Vì sao ChatGPT là mô hình ngôn ngữ?

ChatGPT được gọi là mô hình ngôn ngữ vì cơ chế nền tảng của nó là nhận một chuỗi token và tạo ra phân phối xác suất cho token tiếp theo:

$$
P(w_{t+1}\mid w_1,w_2,\ldots,w_t).
$$

Sau khi chọn một token, mô hình nối token đó vào ngữ cảnh rồi dự đoán token kế tiếp. Lặp lại quá trình này nhiều lần sẽ tạo thành một câu trả lời hoàn chỉnh.

ChatGPT là **mô hình ngôn ngữ lớn** vì:

1. Mô hình có rất nhiều tham số học được từ dữ liệu.
2. Dữ liệu tiền huấn luyện có quy mô lớn và chứa nhiều kiểu văn bản.
3. Mô hình sử dụng kiến trúc Transformer với cơ chế self-attention để kết hợp thông tin từ nhiều vị trí trong ngữ cảnh.
4. Mục tiêu tiền huấn luyện cốt lõi vẫn là dự đoán token tiếp theo.

ChatGPT không dùng bảng đếm N-Gram. Nó biểu diễn token và ngữ cảnh bằng các vector liên tục, sau đó dùng nhiều lớp Transformer để tính phân phối token tiếp theo. Nhờ chia sẻ tham số và attention, mô hình có thể khái quát sang những câu chưa từng xuất hiện nguyên vẹn trong dữ liệu.

Sau tiền huấn luyện, mô hình còn được tinh chỉnh để làm theo chỉ dẫn, trả lời hữu ích và hạn chế nội dung không mong muốn. Tuy nhiên, việc sinh câu trả lời vẫn diễn ra dưới dạng dự đoán token kế tiếp. Vì vậy, gọi ChatGPT là một mô hình ngôn ngữ là chính xác.

### 3.3. Điểm cần lưu ý

Khả năng sinh văn bản trôi chảy không đồng nghĩa với việc mọi câu trả lời đều đúng. Mô hình tạo token dựa trên xác suất học được, nên đôi khi có thể tạo thông tin nghe hợp lý nhưng sai. Với thông tin quan trọng, người dùng vẫn cần kiểm tra lại bằng tài liệu đáng tin cậy.

---

## 4. Kết luận

N-Gram là cách tiếp cận thống kê đơn giản, trong đó xác suất từ tiếp theo được ước lượng bằng số đếm của các cụm từ. Mô hình dễ xây dựng nhưng gặp hạn chế về ngữ cảnh và dữ liệu thưa. Word2Vec giải quyết một phần hạn chế biểu diễn bằng cách ánh xạ từ sang vector, nhưng bản thân nó không phải mô hình ngôn ngữ hoàn chỉnh. LSTM kết hợp embedding và bộ nhớ tuần tự để dự đoán từ tiếp theo dựa trên ngữ cảnh dài hơn. ChatGPT tiếp tục tư tưởng dự đoán token tiếp theo ở quy mô lớn hơn với kiến trúc Transformer.

Qua bài thực hành, em có thể tự xây dựng một mô hình N-Gram, tính xác suất câu, dự đoán từ tiếp theo, đánh giá bằng perplexity và sinh văn bản từ corpus.

---

## 5. Tài liệu tham khảo

1. Daniel Jurafsky và James H. Martin, *Speech and Language Processing*, chương về N-Gram Language Models.
2. Tomas Mikolov và cộng sự, “Efficient Estimation of Word Representations in Vector Space”, 2013.
3. Tomas Mikolov và cộng sự, “Distributed Representations of Words and Phrases and their Compositionality”, 2013.
4. Sepp Hochreiter và Jürgen Schmidhuber, “Long Short-Term Memory”, *Neural Computation*, 1997.
5. Ashish Vaswani và cộng sự, “Attention Is All You Need”, 2017.
