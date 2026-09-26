# MÔ HÌNH NGÔN NGỮ VÀ N-GRAM

## 1. Mô hình ngôn ngữ là gì?

Mô hình ngôn ngữ (Language Model – LM) là mô hình dùng để gán xác suất cho một chuỗi từ hoặc dự đoán từ tiếp theo dựa trên các từ đã xuất hiện trước đó.

Cho một câu gồm $n$ từ:

$$
S = w_1,w_2,\ldots,w_n
$$

Mô hình ngôn ngữ cần tính:

$$
P(S)=P(w_1,w_2,\ldots,w_n)
$$

Xác suất này thể hiện mức độ tự nhiên hoặc hợp lý của câu trong một ngôn ngữ. Câu thường gặp, đúng ngữ pháp và phù hợp ngữ cảnh thường được mô hình gán xác suất cao hơn câu ít tự nhiên.

Ngoài việc đánh giá cả câu, mô hình còn có thể dự đoán từ tiếp theo:

$$
P(w_n\mid w_1,w_2,\ldots,w_{n-1})
$$

Nói cách khác, từ phần văn bản đã có, mô hình xây dựng một phân phối xác suất trên toàn bộ những từ có thể xuất hiện tiếp theo.

### Vai trò của mô hình ngôn ngữ

- Hiểu và biểu diễn quy luật của ngôn ngữ tự nhiên.
- Đánh giá câu nào tự nhiên hoặc có khả năng xuất hiện cao hơn.
- Dự đoán từ tiếp theo và sinh văn bản.
- Hỗ trợ nhận dạng tiếng nói, dịch máy, sửa chính tả, tự động hoàn thành câu, hỏi đáp và chatbot.

## 2. Mô hình ngôn ngữ trong bài toán giải mã

Trong nhiều bài toán xử lý ngôn ngữ, máy nhận được một quan sát $E$ và phải tìm chuỗi từ $V$ phù hợp nhất. Theo nguyên lý cực đại hậu nghiệm:

$$
V^*=\arg\max_V P(V\mid E)
$$

Áp dụng định lý Bayes:

$$
P(V\mid E)=\frac{P(E\mid V)P(V)}{P(E)}
$$

Vì $P(E)$ không thay đổi khi so sánh các phương án $V$, ta có:

$$
V^*=\arg\max_V P(E\mid V)P(V)
$$

Trong đó:

- $P(E\mid V)$ cho biết chuỗi $V$ phù hợp với dữ liệu quan sát đến mức nào. Ví dụ, trong nhận dạng tiếng nói, đây là mức độ phù hợp giữa tín hiệu âm thanh và chuỗi từ.
- $P(V)$ là xác suất của chuỗi từ do mô hình ngôn ngữ cung cấp. Thành phần này ưu tiên các câu tự nhiên và thường xuất hiện trong ngôn ngữ.

Tư tưởng tương tự được dùng trong dịch máy: giữa nhiều câu đích có thể có, hệ thống chọn câu vừa phù hợp với câu nguồn, vừa tự nhiên trong ngôn ngữ đích.

## 3. Phân rã xác suất câu bằng quy tắc dây chuyền

Theo quy tắc dây chuyền của xác suất:

$$
\begin{aligned}
P(w_1,w_2,\ldots,w_n)
&=P(w_1)P(w_2\mid w_1)P(w_3\mid w_1,w_2)\cdots \\
&\quad P(w_n\mid w_1,w_2,\ldots,w_{n-1}) \\
&=\prod_{i=1}^{n}P(w_i\mid w_1,\ldots,w_{i-1}).
\end{aligned}
$$

Nếu sử dụng ký hiệu bắt đầu câu `<s>` và kết thúc câu `</s>`, công thức có thể viết đầy đủ hơn:

$$
P(S)=P(w_1\mid \langle s\rangle)
\prod_{i=2}^{n}P(w_i\mid w_1,\ldots,w_{i-1})
P(\langle /s\rangle\mid w_1,\ldots,w_n).
$$

Đây là công thức chính xác. Tuy nhiên, việc ước lượng xác suất của một từ dựa trên toàn bộ lịch sử phía trước rất khó vì số lượng ngữ cảnh có thể có là cực lớn và dữ liệu huấn luyện luôn hữu hạn.

## 4. Giả định Markov và mô hình N-gram

Để đơn giản hóa, mô hình N-gram sử dụng giả định Markov: xác suất của từ hiện tại chỉ phụ thuộc vào một số hữu hạn từ đứng ngay trước nó.

Với mô hình N-gram:

$$
P(w_i\mid w_1,\ldots,w_{i-1})
\approx
P(w_i\mid w_{i-N+1},\ldots,w_{i-1}).
$$

Các trường hợp thường gặp:

- **Unigram (1-gram):** không xét từ đứng trước.

  $$
  P(w_i\mid w_1,\ldots,w_{i-1})\approx P(w_i)
  $$

- **Bigram (2-gram):** chỉ xét một từ đứng trước.

  $$
  P(w_i\mid w_1,\ldots,w_{i-1})\approx P(w_i\mid w_{i-1})
  $$

- **Trigram (3-gram):** xét hai từ đứng trước.

  $$
  P(w_i\mid w_1,\ldots,w_{i-1})\approx P(w_i\mid w_{i-2},w_{i-1})
  $$

Tương tự, mô hình 4-gram dùng ba từ trước, mô hình 5-gram dùng bốn từ trước.

### Xác suất câu trong mô hình Bigram

Với giả định Bigram:

$$
P(w_1,w_2,\ldots,w_n)
\approx P(w_1\mid \langle s\rangle)
\prod_{i=2}^{n}P(w_i\mid w_{i-1})
P(\langle /s\rangle\mid w_n).
$$

Ví dụ:

$$
\begin{aligned}
P(\text{I love you})
&=P(\text{I}\mid \langle s\rangle) \\
&\quad\times P(\text{love}\mid\text{I}) \\
&\quad\times P(\text{you}\mid\text{love}) \\
&\quad\times P(\langle /s\rangle\mid\text{you}).
\end{aligned}
$$

## 5. Ước lượng xác suất từ tập dữ liệu

Mô hình N-gram thường ước lượng xác suất bằng phương pháp hợp lý cực đại (Maximum Likelihood Estimation – MLE), tức là dựa vào số lần xuất hiện trong corpus.

Ký hiệu:

- $C(w)$: số lần từ $w$ xuất hiện.
- $C(w_{i-1},w_i)$: số lần cặp từ $(w_{i-1},w_i)$ xuất hiện liên tiếp.
- $C(w_{i-2},w_{i-1},w_i)$: số lần bộ ba từ $(w_{i-2},w_{i-1},w_i)$ xuất hiện liên tiếp.

### Unigram

$$
P(w_i)=\frac{C(w_i)}{\sum_{w\in V}C(w)}
$$

trong đó $V$ là tập từ vựng.

### Bigram

$$
P(w_i\mid w_{i-1})
=\frac{C(w_{i-1},w_i)}{C(w_{i-1})}
=\frac{C(w_{i-1},w_i)}{\sum_{x\in V}C(w_{i-1},x)}.
$$

### Trigram

$$
P(w_i\mid w_{i-2},w_{i-1})
=\frac{C(w_{i-2},w_{i-1},w_i)}{C(w_{i-2},w_{i-1})}.
$$

Tổng quát, xác suất của một N-gram bằng số lần xuất hiện của N-gram đó chia cho số lần xuất hiện của phần ngữ cảnh gồm $N-1$ từ đầu.

## 6. Vấn đề dữ liệu thưa và xác suất bằng 0

Do corpus hữu hạn, nhiều N-gram hợp lệ không xuất hiện trong tập huấn luyện. Khi đó:

$$
C(w_{i-1},w_i)=0
\quad\Rightarrow\quad
P(w_i\mid w_{i-1})=0.
$$

Xác suất câu là tích của các xác suất thành phần. Vì vậy, chỉ cần một N-gram có xác suất bằng 0 thì xác suất của cả câu cũng bằng 0, dù câu đó có thể hoàn toàn hợp lý.

Hiện tượng này gọi là **dữ liệu thưa** (sparse data) hoặc **zero-count problem**. Khi tăng $N$, mô hình có thể nắm bắt ngữ cảnh dài hơn nhưng số lượng N-gram có thể có tăng rất nhanh, nên dữ liệu càng thưa.

## 7. Làm trơn xác suất

Làm trơn (smoothing) là quá trình lấy bớt một phần khối xác suất của các N-gram đã gặp để phân phối cho những N-gram chưa từng xuất hiện. Mục tiêu là tránh xác suất bằng 0 và giúp mô hình khái quát tốt hơn trên dữ liệu mới.

### Laplace smoothing – cộng 1

Với mô hình Bigram:

$$
P_{\text{Laplace}}(w_i\mid w_{i-1})
=\frac{C(w_{i-1},w_i)+1}{C(w_{i-1})+|V|},
$$

trong đó $|V|$ là kích thước từ vựng.

Mẫu số tăng thêm $|V|$ vì ta đã cộng 1 cho mỗi từ có thể đứng sau $w_{i-1}$. Nhờ đó, mọi từ trong từ vựng đều có xác suất lớn hơn 0 và tổng các xác suất vẫn bằng 1.

### Add-k smoothing

Một dạng tổng quát hơn là cộng một số nhỏ $k>0$:

$$
P_{\text{add-}k}(w_i\mid w_{i-1})
=\frac{C(w_{i-1},w_i)+k}{C(w_{i-1})+k|V|}.
$$

Laplace smoothing chính là trường hợp $k=1$. Trong thực tế, $k$ nhỏ hơn 1 thường ít làm méo phân phối hơn. Ngoài ra còn có các kỹ thuật tốt hơn như backoff, interpolation và Kneser–Ney.

## 8. Quy trình xây dựng mô hình N-gram

### Bước 1. Thu thập dữ liệu

Chuẩn bị một corpus đủ lớn và phù hợp với lĩnh vực sử dụng. Mô hình học trực tiếp từ tần suất xuất hiện của các từ và cụm từ trong corpus.

### Bước 2. Tiền xử lý văn bản

Có thể thực hiện các thao tác như chuẩn hóa Unicode, chuẩn hóa khoảng trắng, chuyển chữ thường, xử lý dấu câu, số, ký hiệu đặc biệt và từ ngoài từ vựng. Cách tiền xử lý phải thống nhất giữa dữ liệu huấn luyện và dữ liệu kiểm thử.

### Bước 3. Tách văn bản

Tách văn bản thành câu và tách câu thành token. Thêm ký hiệu bắt đầu câu `<s>` và kết thúc câu `</s>` để mô hình học được cách bắt đầu và dừng một câu.

### Bước 4. Xây dựng N-gram

Từ từng câu, tạo ra các unigram, bigram, trigram hoặc N-gram theo bậc mô hình đã chọn.

Ví dụ, với câu `<s> I love you </s>`:

- Unigram: `<s>`, `I`, `love`, `you`, `</s>`.
- Bigram: `(<s>, I)`, `(I, love)`, `(love, you)`, `(you, </s>)`.
- Trigram: `(<s>, I, love)`, `(I, love, you)`, `(love, you, </s>)`.

### Bước 5. Đếm và tính xác suất

Đếm tần suất của các N-gram và ngữ cảnh $N-1$ từ, sau đó dùng công thức MLE để tính xác suất có điều kiện.

### Bước 6. Làm trơn

Áp dụng Laplace, add-$k$, backoff hoặc interpolation để xử lý các N-gram chưa xuất hiện và giảm ảnh hưởng của dữ liệu thưa.

### Bước 7. Đánh giá mô hình

Tách riêng tập kiểm thử; không dùng tập này để đếm N-gram. Một thước đo phổ biến là **perplexity**:

$$
PP(W)=P(w_1,w_2,\ldots,w_m)^{-1/m}
=\exp\left(-\frac{1}{m}\sum_{i=1}^{m}\ln P(w_i\mid h_i)\right),
$$

trong đó $h_i$ là ngữ cảnh của $w_i$. Perplexity càng thấp thì mô hình càng dự đoán tốt tập kiểm thử. Chỉ nên so sánh các mô hình dùng cùng dữ liệu kiểm thử, cùng cách tách token và cùng từ vựng.

### Bước 8. Sinh văn bản

Bắt đầu với `<s>`, dùng phân phối $P(w_i\mid h_i)$ để chọn từ tiếp theo, cập nhật ngữ cảnh rồi lặp lại cho tới khi sinh ra `</s>` hoặc đạt độ dài tối đa.

Có hai cách chọn từ phổ biến:

- **Greedy:** luôn chọn từ có xác suất lớn nhất; kết quả ổn định nhưng dễ lặp và ít đa dạng.
- **Sampling:** lấy mẫu theo phân phối xác suất; kết quả đa dạng hơn nhưng có thể kém tự nhiên nếu phân phối chưa tốt.

## 9. Ưu điểm và hạn chế của N-gram

### Ưu điểm

- Ý tưởng đơn giản, dễ hiểu và dễ cài đặt.
- Huấn luyện nhanh, không cần mạng nơ-ron.
- Xác suất có thể giải thích trực tiếp bằng số lần xuất hiện.
- Hoạt động khá tốt khi corpus lớn và miền dữ liệu hẹp.

### Hạn chế

- Chỉ sử dụng được ngữ cảnh ngắn, không nắm bắt tốt quan hệ xa trong câu.
- Số lượng N-gram tăng rất nhanh khi $N$ hoặc kích thước từ vựng tăng.
- Tốn bộ nhớ để lưu bảng đếm.
- Gặp vấn đề dữ liệu thưa và từ ngoài từ vựng.
- Khả năng khái quát về ngữ nghĩa kém: các từ gần nghĩa vẫn bị xem là các mục hoàn toàn khác nhau.

## 10. Từ N-gram đến Word2Vec, LSTM và mô hình ngôn ngữ hiện đại

**Word2Vec** học một vector liên tục cho mỗi từ dựa trên ngữ cảnh. Những từ xuất hiện trong ngữ cảnh tương tự thường có vector gần nhau. Word2Vec giúp biểu diễn quan hệ ngữ nghĩa giữa các từ, nhưng bản thân nó không phải là một mô hình ngôn ngữ hoàn chỉnh vì không trực tiếp gán xác suất cho cả chuỗi từ.

**Mô hình ngôn ngữ dùng LSTM** lần lượt đọc các từ và lưu thông tin trong trạng thái ẩn. So với N-gram, LSTM có thể sử dụng ngữ cảnh dài hơn và chia sẻ kiến thức giữa những ngữ cảnh tương tự nhờ biểu diễn vector. Tuy nhiên, LSTM vẫn gặp khó khăn khi chuỗi rất dài và việc huấn luyện tuần tự khó song song hóa.

Các mô hình hiện đại như **ChatGPT** là mô hình ngôn ngữ lớn dùng kiến trúc Transformer. Trong giai đoạn tiền huấn luyện, mô hình học dự đoán token tiếp theo:

$$
P(w_t\mid w_1,w_2,\ldots,w_{t-1}).
$$

Vì liên tục chọn token tiếp theo từ phân phối xác suất này, mô hình có thể tạo ra cả một đoạn văn. ChatGPT vẫn là mô hình ngôn ngữ theo nghĩa cốt lõi đó, nhưng khác N-gram ở chỗ nó biểu diễn từ và ngữ cảnh bằng vector, dùng attention để khai thác ngữ cảnh dài, và có số lượng tham số rất lớn.

## 11. Ví dụ tổng hợp

Giả sử cần tính xác suất câu `I love you` bằng mô hình Bigram.

Không làm trơn:

$$
\begin{aligned}
P(\text{I love you})
&=\frac{C(\langle s\rangle,\text{I})}{C(\langle s\rangle)}
\times\frac{C(\text{I},\text{love})}{C(\text{I})} \\
&\quad\times\frac{C(\text{love},\text{you})}{C(\text{love})}
\times\frac{C(\text{you},\langle /s\rangle)}{C(\text{you})}.
\end{aligned}
$$

Nếu một trong các cặp trên chưa từng xuất hiện, xác suất cả câu sẽ bằng 0. Với Laplace smoothing:

$$
\begin{aligned}
P_{\text{L}}(\text{I love you})
&=\frac{C(\langle s\rangle,\text{I})+1}{C(\langle s\rangle)+|V|} \\
&\quad\times\frac{C(\text{I},\text{love})+1}{C(\text{I})+|V|} \\
&\quad\times\frac{C(\text{love},\text{you})+1}{C(\text{love})+|V|} \\
&\quad\times\frac{C(\text{you},\langle /s\rangle)+1}{C(\text{you})+|V|}.
\end{aligned}
$$

Như vậy, ngay cả khi một Bigram chưa xuất hiện trong corpus, câu vẫn nhận được một xác suất dương.

## 12. Ghi nhớ nhanh

1. Mô hình ngôn ngữ gán xác suất cho chuỗi từ và dự đoán từ tiếp theo.
2. Quy tắc dây chuyền phân rã xác suất câu thành tích các xác suất có điều kiện.
3. N-gram dùng giả định Markov để chỉ xét $N-1$ từ đứng trước.
4. Xác suất N-gram được ước lượng từ số lần xuất hiện trong corpus.
5. N-gram chưa xuất hiện làm xác suất câu bằng 0; smoothing được dùng để xử lý vấn đề này.
6. Perplexity càng thấp thì mô hình càng dự đoán tốt trên cùng một tập kiểm thử.
7. Sinh văn bản là quá trình dự đoán hoặc lấy mẫu từng từ tiếp theo cho tới khi kết thúc câu.
8. Word2Vec biểu diễn từ bằng vector; LSTM và Transformer xây dựng mô hình ngôn ngữ bằng mạng nơ-ron và khai thác ngữ cảnh tốt hơn N-gram.

## 13. Bài tập về nhà ghi trên bảng

1. Tìm hiểu mô hình ngôn ngữ dựa trên N-gram.
2. Tìm hiểu Word2Vec và mô hình LSTM dùng cho mô hình ngôn ngữ.
3. Trả lời: mô hình ngôn ngữ là gì, và vì sao ChatGPT được gọi là một mô hình ngôn ngữ?
