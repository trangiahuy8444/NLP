"""
Script tạo file Jupyter Notebook Thuc_Hanh_Word2Vec_LSTM.ipynb hoàn chỉnh
Hỗ trợ cả 2 bộ dữ liệu thực tế (UIT-VSFC và E-Commerce Reviews),
chạy qua các bước từ Word2Vec, Biểu diễn văn bản đến BiLSTM phân loại.
"""
import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# BÀI TẬP GIỮA KỲ: WORD2VEC, WORD EMBEDDING & HỌC BIỂU DIỄN VĂN BẢN BẰNG LSTM\n",
            "\n",
            "**Môn học:** Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)  \n",
            "**Thực nghiệm trên 2 Bộ Dữ liệu Thực tế Tiếng Việt:**\n",
            "1. **UIT-VSFC** (Vietnamese Students' Feedback Corpus - Miền Giáo dục / Học thuật)\n",
            "2. **Vietnamese E-Commerce Reviews** (Miền Thương mại điện tử / Tiêu dùng & Mạng xã hội)\n",
            "\n",
            "**Mục tiêu chính:**\n",
            "- Huấn luyện mô hình biểu diễn từ Word2Vec (CBOW & Skip-Gram) bằng `gensim`.\n",
            "- Biểu diễn văn bản bằng vector: Phương pháp thống kê (Average Word2Vec) và mạng nơ-ron học sâu (**BiLSTM**).\n",
            "- Ứng dụng LSTM để phân loại cảm xúc (Text Sentiment Classification).\n",
            "- So sánh đối chuẩn 4 mô hình trên 2 miền dữ liệu thực tế và trực quan hóa t-SNE."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Cài đặt và Nạp Thư viện"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import random\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "from gensim.models import Word2Vec\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix\n",
            "from sklearn.manifold import TSNE\n",
            "\n",
            "# Tái lập kết quả\n",
            "torch.manual_seed(42)\n",
            "np.random.seed(42)\n",
            "random.seed(42)\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))\n",
            "print(f\"Thiết bị tính toán: {device}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Nạp Dữ liệu Thực tế (Lựa chọn Dataset)\n",
            "Bạn có thể dễ dàng chuyển đổi giữa `uit_vsfc` và `ecommerce` bằng biến `DATASET_CHOICE`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Chọn dataset: 'uit_vsfc' hoặc 'ecommerce'\n",
            "DATASET_CHOICE = 'uit_vsfc'\n",
            "\n",
            "train_path = f'data/{DATASET_CHOICE}/train.csv'\n",
            "test_path = f'data/{DATASET_CHOICE}/test.csv'\n",
            "\n",
            "train_df = pd.read_csv(train_path).dropna(subset=['text'])\n",
            "test_df = pd.read_csv(test_path).dropna(subset=['text'])\n",
            "\n",
            "print(f\"=== ĐANG SỬ DỤNG BỘ DỮ LIỆU: {DATASET_CHOICE.upper()} ===\")\n",
            "print(f\"Số lượng mẫu Train: {len(train_df)}\")\n",
            "print(f\"Số lượng mẫu Test:  {len(test_df)}\")\n",
            "print(\"\\nPhân bố nhãn trong tập Train:\")\n",
            "print(train_df['label'].value_counts())\n",
            "train_df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Tiền xử lý Dữ liệu Tiếng Việt\n",
            "Chuẩn hóa Unicode (NFC), chuyển chữ thường, xử lý từ viết tắt, ký tự đặc biệt và tách từ."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import re\n",
            "import unicodedata\n",
            "\n",
            "NORMALIZATION_DICT = {\n",
            "    \"k\": \"không\", \"ko\": \"không\", \"khong\": \"không\", \"hok\": \"không\",\n",
            "    \"đc\": \"được\", \"dc\": \"được\", \"oke\": \"tốt\", \"ok\": \"tốt\",\n",
            "    \"sp\": \"sản phẩm\", \"cam\": \"camera\", \"dt\": \"điện thoại\", \"đt\": \"điện thoại\",\n",
            "    \"gv\": \"giảng viên\", \"thầy\": \"giảng viên\", \"cô\": \"giảng viên\"\n",
            "}\n",
            "\n",
            "def clean_and_tokenize(text):\n",
            "    text = unicodedata.normalize('NFC', str(text)).lower()\n",
            "    text = re.sub(r\"https?://\\S+|www\\.\\S+\", \"\", text)\n",
            "    text = re.sub(r\"([a-zà-ỹ])\\1{2,}\", r\"\\1\", text)\n",
            "    text = re.sub(r\"[^\\w\\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]\", \" \", text)\n",
            "    tokens = text.split()\n",
            "    tokens = [NORMALIZATION_DICT.get(w, w) for w in tokens]\n",
            "    return tokens\n",
            "\n",
            "train_tokens = [clean_and_tokenize(t) for t in train_df['text']]\n",
            "test_tokens = [clean_and_tokenize(t) for t in test_df['text']]\n",
            "y_train = train_df['label'].values\n",
            "y_test = test_df['label'].values\n",
            "\n",
            "print(\"Mẫu câu sau khi tách token:\", train_tokens[0])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Huấn luyện Word2Vec (CBOW & Skip-Gram) bằng Gensim\n",
            "Học vector từ ngữ cảnh thực tế của bộ dữ liệu."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "corpus = train_tokens + test_tokens\n",
            "\n",
            "# 1. CBOW (sg=0)\n",
            "w2v_cbow = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, sg=0, epochs=30, seed=42)\n",
            "# 2. Skip-Gram (sg=1)\n",
            "w2v_sg = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, sg=1, epochs=30, seed=42)\n",
            "\n",
            "print(f\"Đã huấn luyện xong Word2Vec! Kích thước từ điển: {len(w2v_cbow.wv)} từ.\")\n",
            "\n",
            "# Khảo sát từ tương đồng\n",
            "sample_words = [w for w in ['nhiệt_tình', 'giảng_viên', 'tốt', 'kém', 'dễ', 'khó'] if w in w2v_cbow.wv]\n",
            "if len(sample_words) == 0:\n",
            "    sample_words = list(w2v_cbow.wv.index_to_key[:5])\n",
            "\n",
            "for word in sample_words[:3]:\n",
            "    sims = w2v_cbow.wv.most_similar(word, topn=3)\n",
            "    print(f\"Từ tương đồng nhất với '{word}':\", [(w, round(score, 3)) for w, score in sims])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Đối chuẩn 1: TF-IDF + Logistic Regression (Baseline)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "tfidf = TfidfVectorizer(max_features=1500)\n",
            "X_train_tfidf = tfidf.fit_transform(train_df['text'].astype(str))\n",
            "X_test_tfidf = tfidf.transform(test_df['text'].astype(str))\n",
            "\n",
            "clf_tfidf = LogisticRegression(random_state=42, max_iter=500)\n",
            "clf_tfidf.fit(X_train_tfidf, y_train)\n",
            "y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)\n",
            "\n",
            "acc_tfidf = accuracy_score(y_test, y_pred_tfidf)\n",
            "_, _, f1_tfidf, _ = precision_recall_fscore_support(y_test, y_pred_tfidf, average='weighted')\n",
            "print(f\"[Baseline 1] TF-IDF: Accuracy = {acc_tfidf*100:.2f}% | F1 = {f1_tfidf*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Đối chuẩn 2: Average Word2Vec + Logistic Regression"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_avg_word2vec(tokens, model):\n",
            "    vecs = [model.wv[w] for w in tokens if w in model.wv]\n",
            "    if len(vecs) == 0:\n",
            "        return np.zeros(model.vector_size, dtype=np.float32)\n",
            "    return np.mean(vecs, axis=0).astype(np.float32)\n",
            "\n",
            "X_train_w2v = np.array([get_avg_word2vec(t, w2v_cbow) for t in train_tokens])\n",
            "X_test_w2v = np.array([get_avg_word2vec(t, w2v_cbow) for t in test_tokens])\n",
            "\n",
            "clf_w2v = LogisticRegression(random_state=42, max_iter=500)\n",
            "clf_w2v.fit(X_train_w2v, y_train)\n",
            "y_pred_w2v = clf_w2v.predict(X_test_w2v)\n",
            "\n",
            "acc_w2v = accuracy_score(y_test, y_pred_w2v)\n",
            "_, _, f1_w2v, _ = precision_recall_fscore_support(y_test, y_pred_w2v, average='weighted')\n",
            "print(f\"[Baseline 2] Average Word2Vec: Accuracy = {acc_w2v*100:.2f}% | F1 = {f1_w2v*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Mạng BiLSTM Học Biểu diễn Văn bản và Phân loại (PyTorch)\n",
            "Xây dựng mạng BiLSTM nhận đầu vào là chuỗi từ và ánh xạ thành **Document Representation Vector** $256$ chiều."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "vocab = {'<pad>': 0, '<unk>': 1}\n",
            "for sent in train_tokens:\n",
            "    for w in sent:\n",
            "        if w not in vocab:\n",
            "            vocab[w] = len(vocab)\n",
            "\n",
            "class TextDataset(Dataset):\n",
            "    def __init__(self, texts, labels, vocab, max_len=64):\n",
            "        self.texts = texts\n",
            "        self.labels = labels\n",
            "        self.vocab = vocab\n",
            "        self.max_len = max_len\n",
            "        self.pad_idx = vocab['<pad>']\n",
            "        self.unk_idx = vocab['<unk>']\n",
            "    def __len__(self):\n",
            "        return len(self.texts)\n",
            "    def __getitem__(self, idx):\n",
            "        tokens = self.texts[idx][:self.max_len]\n",
            "        indices = [self.vocab.get(w, self.unk_idx) for w in tokens]\n",
            "        length = max(1, len(indices))\n",
            "        if len(indices) < self.max_len:\n",
            "            indices = indices + [self.pad_idx] * (self.max_len - len(indices))\n",
            "        return torch.tensor(indices, dtype=torch.long), self.labels[idx], length\n",
            "\n",
            "def collate_fn(batch):\n",
            "    texts, labels, lengths = zip(*batch)\n",
            "    return torch.stack(texts), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)\n",
            "\n",
            "train_loader = DataLoader(TextDataset(train_tokens, y_train, vocab), batch_size=32, shuffle=True, collate_fn=collate_fn)\n",
            "test_loader = DataLoader(TextDataset(test_tokens, y_test, vocab), batch_size=32, shuffle=False, collate_fn=collate_fn)\n",
            "\n",
            "class BiLSTMClassifier(nn.Module):\n",
            "    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128, num_classes=2, pretrained_emb=None):\n",
            "        super().__init__()\n",
            "        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)\n",
            "        if pretrained_emb is not None:\n",
            "            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))\n",
            "        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)\n",
            "        self.rep_dim = hidden_dim * 2\n",
            "        self.classifier = nn.Sequential(\n",
            "            nn.Dropout(0.3),\n",
            "            nn.Linear(self.rep_dim, 64),\n",
            "            nn.ReLU(),\n",
            "            nn.Dropout(0.3),\n",
            "            nn.Linear(64, num_classes)\n",
            "        )\n",
            "    def get_document_vector(self, x, lengths):\n",
            "        embed = self.embedding(x)\n",
            "        lstm_out, (hn, cn) = self.lstm(embed)\n",
            "        doc_vec = torch.cat([hn[-2], hn[-1]], dim=1)\n",
            "        return doc_vec\n",
            "    def forward(self, x, lengths):\n",
            "        doc_vec = self.get_document_vector(x, lengths)\n",
            "        logits = self.classifier(doc_vec)\n",
            "        return logits, doc_vec"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Huấn luyện BiLSTM tích hợp Pretrained Word2Vec Embeddings"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "emb_matrix = np.zeros((len(vocab), 100), dtype=np.float32)\n",
            "for word, idx in vocab.items():\n",
            "    if word in w2v_cbow.wv:\n",
            "        emb_matrix[idx] = w2v_cbow.wv[word]\n",
            "    elif word != '<pad>':\n",
            "        emb_matrix[idx] = np.random.normal(scale=0.1, size=(100,))\n",
            "\n",
            "model_lstm = BiLSTMClassifier(len(vocab), emb_dim=100, hidden_dim=128, pretrained_emb=emb_matrix).to(device)\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-3, weight_decay=1e-4)\n",
            "\n",
            "history = {'train_loss': [], 'val_acc': []}\n",
            "for epoch in range(1, 13):\n",
            "    model_lstm.train()\n",
            "    t_loss, total = 0, 0\n",
            "    for x, y, lengths in train_loader:\n",
            "        x, y, lengths = x.to(device), y.to(device), lengths.to(device)\n",
            "        optimizer.zero_grad()\n",
            "        logits, _ = model_lstm(x, lengths)\n",
            "        loss = criterion(logits, y)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        t_loss += loss.item() * len(y)\n",
            "        total += len(y)\n",
            "    \n",
            "    model_lstm.eval()\n",
            "    v_correct, v_total = 0, 0\n",
            "    with torch.no_grad():\n",
            "        for x, y, lengths in test_loader:\n",
            "            x, y, lengths = x.to(device), y.to(device), lengths.to(device)\n",
            "            logits, _ = model_lstm(x, lengths)\n",
            "            v_correct += (logits.argmax(dim=1) == y).sum().item()\n",
            "            v_total += len(y)\n",
            "    \n",
            "    v_acc = v_correct / v_total\n",
            "    history['train_loss'].append(t_loss / total)\n",
            "    history['val_acc'].append(v_acc)\n",
            "    if epoch % 3 == 0 or epoch == 1 or epoch == 12:\n",
            "        print(f\"Epoch [{epoch:02d}/12] Train Loss: {t_loss/total:.4f} | Val Accuracy: {v_acc*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. So sánh Đối chuẩn Đa Tập Dữ liệu (Cross-Dataset Comparison)\n",
            "Bảng tổng hợp kết quả chạy thực tế trên cả 2 bộ dữ liệu:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary_df = pd.read_csv('results/cross_dataset_summary.csv')\n",
            "print(\"=== BẢNG SO SÁNH KẾT QUẢ ĐA TẬP DỮ LIỆU ===\")\n",
            "summary_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Biểu đồ So sánh Hiệu năng trên 2 Miền Dữ liệu Thực tế"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.image as mpimg\n",
            "chart_img = mpimg.imread('results/cross_dataset_comparison.png')\n",
            "plt.figure(figsize=(11, 7))\n",
            "plt.imshow(chart_img)\n",
            "plt.axis('off')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Trực quan hóa Không gian Biểu diễn Văn bản bằng t-SNE\n",
            "So sánh vector thống kê Average Word2Vec và vector biểu diễn do mạng BiLSTM học được."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "img_tsne_w2v = mpimg.imread(f'results/{DATASET_CHOICE}/tsne_avg_word2vec.png')\n",
            "img_tsne_lstm = mpimg.imread(f'results/{DATASET_CHOICE}/tsne_lstm_representation.png')\n",
            "\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))\n",
            "ax1.imshow(img_tsne_w2v)\n",
            "ax1.set_title(f't-SNE Average Word2Vec ({DATASET_CHOICE.upper()})', fontweight='bold')\n",
            "ax1.axis('off')\n",
            "\n",
            "ax2.imshow(img_tsne_lstm)\n",
            "ax2.set_title(f't-SNE LSTM Representation ({DATASET_CHOICE.upper()})', fontweight='bold')\n",
            "ax2.axis('off')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Thử nghiệm Dự đoán Cảm xúc trên các Câu Mới (Inference)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def predict_sentiment(text, model, vocab, device):\n",
            "    model.eval()\n",
            "    tokens = clean_and_tokenize(text)\n",
            "    indices = [vocab.get(w, vocab['<unk>']) for w in tokens]\n",
            "    length = torch.tensor([max(1, len(indices))], dtype=torch.long).to(device)\n",
            "    if len(indices) < 64:\n",
            "        indices = indices + [vocab['<pad>']] * (64 - len(indices))\n",
            "    else:\n",
            "        indices = indices[:64]\n",
            "    x = torch.tensor([indices], dtype=torch.long).to(device)\n",
            "    with torch.no_grad():\n",
            "        logits, doc_vec = model(x, length)\n",
            "        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]\n",
            "        pred = np.argmax(probs)\n",
            "    label_str = \"TÍCH CỰC (Positive)\" if pred == 1 else \"TIÊU CỰC (Negative)\"\n",
            "    print(f\"Văn bản: '{text}'\")\n",
            "    print(f\" -> Dự đoán: {label_str} (Độ tin cậy: {probs[pred]*100:.2f}%)\")\n",
            "    print(f\" -> 5 chiều đầu của Document Vector: {doc_vec.cpu().numpy()[0][:5]}\\n\")\n",
            "\n",
            "test_sents = [\n",
            "    \"Giảng viên rất tận tâm, giải thích bài tập rõ ràng dễ hiểu!\",\n",
            "    \"Thầy dạy quá nhanh, bài tập khó và không hướng dẫn kỹ.\",\n",
            "    \"Phòng học nóng nực, máy chiếu mờ không thấy gì cả.\",\n",
            "    \"Môn học rất bổ ích, áp dụng được nhiều kiến thức thực tế.\"\n",
            "]\n",
            "for s in test_sents:\n",
            "    predict_sentiment(s, model_lstm, vocab, device)"
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "../Thuc_Hanh_Word2Vec_LSTM.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, ensure_ascii=False, indent=2)

print(f"Đã cập nhật make_notebook.py tại {out_path}")
