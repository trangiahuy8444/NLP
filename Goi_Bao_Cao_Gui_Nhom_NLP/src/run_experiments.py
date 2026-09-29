"""
Script chạy toàn bộ quy trình thực nghiệm đối chuẩn chuyên sâu đa tập dữ liệu:
So sánh đối chuẩn 6 mô hình:
  1. TF-IDF + Logistic Regression (Baseline truyền thống)
  2. Average Word2Vec + Logistic Regression (Baseline vector tĩnh)
  3. BiLSTM Classifier (Học biểu diễn từ đầu - Scratch)
  4. BiLSTM Classifier (Tích hợp Pre-trained Word2Vec)
  5. BiLSTM + Self-Attention (Tích hợp Pre-trained Word2Vec + Cơ chế chú ý tự thân)
  6. PhoBERT (Fine-tuning Mô hình ngôn ngữ lớn Transformer tiếng Việt)

Trên 2 bộ dữ liệu thực tế tiếng Việt:
  - Bộ dữ liệu 1: UIT-VSFC (Vietnamese Students' Feedback Corpus - Miền Giáo dục / Học thuật)
  - Bộ dữ liệu 2: Vietnamese E-Commerce Reviews (Miền Thương mại điện tử / Tiêu dùng)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Import các module tự xây dựng
from preprocess import tokenize, clean_text
from word2vec_trainer import (
    train_word2vec,
    save_word2vec_model,
    get_word_similarities,
    build_embedding_matrix
)
from text_representation import (
    compute_corpus_average_word2vec,
    extract_lstm_representations
)
from lstm_classifier import (
    TextDataset,
    collate_fn,
    LSTMTextClassifier,
    BiLSTMAttentionClassifier,
    train_model
)
from phobert_classifier import (
    train_phobert,
    evaluate_phobert
)
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    plot_tsne_embeddings,
    plot_attention_heatmap
)

def run_experiment_on_dataset(dataset_name: str, data_dir: str, results_dir: str, models_dir: str, device):
    print(f"\n{'='*75}")
    print(f"=== TIẾN HÀNH THỰC NGHIỆM ĐỐI CHUẨN TRÊN BỘ DỮ LIỆU: {dataset_name.upper()} ===")
    print(f"{'='*75}")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. NẠP DỮ LIỆU
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv")).dropna(subset=["text"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv")).dropna(subset=["text"])
    
    print(f"Số lượng mẫu huấn luyện (Train): {len(train_df)}")
    print(f"Số lượng mẫu kiểm thử (Test): {len(test_df)}")
    
    train_texts = [str(t) for t in train_df["text"]]
    test_texts = [str(t) for t in test_df["text"]]
    train_tokens = [tokenize(t) for t in train_texts]
    test_tokens = [tokenize(t) for t in test_texts]
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    class_names = ["Tiêu cực (0)", "Tích cực (1)"]
    
    # 2. HUẤN LUYỆN WORD2VEC (CBOW & SKIP-GRAM)
    print(f"\n--- [1/6] Huấn luyện Word2Vec trên corpus {dataset_name} ---")
    all_sentences = train_tokens + test_tokens
    
    w2v_cbow = train_word2vec(all_sentences, vector_size=100, window=5, min_count=1, sg=0, epochs=30)
    save_word2vec_model(w2v_cbow, os.path.join(models_dir, f"word2vec_cbow_{dataset_name}.model"))
    
    w2v_sg = train_word2vec(all_sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=30)
    save_word2vec_model(w2v_sg, os.path.join(models_dir, f"word2vec_skipgram_{dataset_name}.model"))
    
    print(f"Đã huấn luyện xong Word2Vec CBOW và Skip-Gram! Số từ vựng: {len(w2v_cbow.wv)}")
    
    # 3. BASELINE 1: TF-IDF + LOGISTIC REGRESSION
    print("\n--- [2/6] Mô hình 1: TF-IDF + Logistic Regression (Baseline) ---")
    tfidf = TfidfVectorizer(preprocessor=clean_text, max_features=1500)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)
    
    clf_tfidf = LogisticRegression(random_state=42, max_iter=500)
    clf_tfidf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
    metrics_tfidf = compute_metrics(y_test, y_pred_tfidf)
    print(f"Kết quả TF-IDF: Acc = {metrics_tfidf['accuracy']*100:.2f}% | F1 = {metrics_tfidf['f1_score']*100:.2f}%")
    plot_confusion_matrix(y_test, y_pred_tfidf, class_names, f"CM TF-IDF ({dataset_name})", os.path.join(results_dir, "cm_tfidf.png"))
    
    # 4. BASELINE 2: AVERAGE WORD2VEC + LOGISTIC REGRESSION
    print("\n--- [3/6] Mô hình 2: Average Word2Vec + Logistic Regression ---")
    X_train_avg_w2v = compute_corpus_average_word2vec(train_tokens, w2v_cbow)
    X_test_avg_w2v = compute_corpus_average_word2vec(test_tokens, w2v_cbow)
    
    clf_w2v = LogisticRegression(random_state=42, max_iter=500)
    clf_w2v.fit(X_train_avg_w2v, y_train)
    y_pred_w2v = clf_w2v.predict(X_test_avg_w2v)
    metrics_w2v = compute_metrics(y_test, y_pred_w2v)
    print(f"Kết quả Average Word2Vec: Acc = {metrics_w2v['accuracy']*100:.2f}% | F1 = {metrics_w2v['f1_score']*100:.2f}%")
    plot_confusion_matrix(y_test, y_pred_w2v, class_names, f"CM Avg Word2Vec ({dataset_name})", os.path.join(results_dir, "cm_avg_word2vec.png"))
    
    # 5. XÂY DỰNG TỪ ĐIỂN VÀ DATALOADER CHO MẠNG HỒI QUY
    vocab = {"<pad>": 0, "<unk>": 1}
    for sent in train_tokens:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
                
    train_dataset = TextDataset(train_tokens, y_train, vocab, max_len=64)
    test_dataset = TextDataset(test_tokens, y_test, vocab, max_len=64)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 6. MÔ HÌNH 3: BiLSTM HỌC TỪ ĐẦU (SCRATCH)
    print("\n--- [4/6] Mô hình 3: BiLSTM Classifier (Học từ đầu) ---")
    model_scratch = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=2,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
        pad_idx=0
    )
    hist_scratch = train_model(model_scratch, train_loader, test_loader, num_epochs=12, lr=1e-3, device=device)
    plot_training_curves(hist_scratch, os.path.join(results_dir, "training_curves_scratch.png"))
    
    model_scratch.eval()
    all_preds_scratch = []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits, _ = model_scratch(x, lengths)
            all_preds_scratch.extend(logits.argmax(dim=1).cpu().numpy())
    metrics_scratch = compute_metrics(y_test, all_preds_scratch)
    print(f"Kết quả BiLSTM (từ đầu): Acc = {metrics_scratch['accuracy']*100:.2f}% | F1 = {metrics_scratch['f1_score']*100:.2f}%")
    plot_confusion_matrix(y_test, all_preds_scratch, class_names, f"CM BiLSTM Scratch ({dataset_name})", os.path.join(results_dir, "cm_bilstm_scratch.png"))
    
    # 7. MÔ HÌNH 4: BiLSTM TÍCH HỢP PRETRAINED WORD2VEC
    print("\n--- [5/6] Mô hình 4: BiLSTM Classifier (Pre-trained Word2Vec) ---")
    pretrained_weights = build_embedding_matrix(w2v_cbow, vocab, embedding_dim=100)
    model_pretrained = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=2,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
        pad_idx=0,
        pretrained_embeddings=pretrained_weights,
        freeze_embeddings=False
    )
    hist_pretrained = train_model(model_pretrained, train_loader, test_loader, num_epochs=12, lr=1e-3, device=device)
    plot_training_curves(hist_pretrained, os.path.join(results_dir, "training_curves_pretrained.png"))
    
    model_pretrained.eval()
    all_preds_pretrained = []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits, _ = model_pretrained(x, lengths)
            all_preds_pretrained.extend(logits.argmax(dim=1).cpu().numpy())
    metrics_pretrained = compute_metrics(y_test, all_preds_pretrained)
    print(f"Kết quả BiLSTM (Pre-trained Word2Vec): Acc = {metrics_pretrained['accuracy']*100:.2f}% | F1 = {metrics_pretrained['f1_score']*100:.2f}%")
    plot_confusion_matrix(y_test, all_preds_pretrained, class_names, f"CM BiLSTM Pretrained ({dataset_name})", os.path.join(results_dir, "cm_bilstm_pretrained.png"))
    
    # 8. MÔ HÌNH 5: BiLSTM + SELF-ATTENTION (Mở rộng theo Hướng phát triển)
    print("\n--- [6a/6] Mô hình 5: BiLSTM + Self-Attention (Pre-trained Word2Vec + Attention) ---")
    model_attention = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        attention_dim=64,
        num_classes=2,
        num_layers=1,
        dropout=0.3,
        pad_idx=0,
        pretrained_embeddings=pretrained_weights,
        freeze_embeddings=False
    )
    hist_attention = train_model(model_attention, train_loader, test_loader, num_epochs=12, lr=1e-3, device=device)
    plot_training_curves(hist_attention, os.path.join(results_dir, "training_curves_attention.png"))
    torch.save(model_attention.state_dict(), os.path.join(models_dir, f"bilstm_attention_{dataset_name}.pt"))
    
    model_attention.eval()
    all_preds_attention = []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits, _ = model_attention(x, lengths)
            all_preds_attention.extend(logits.argmax(dim=1).cpu().numpy())
    metrics_attention = compute_metrics(y_test, all_preds_attention)
    print(f"Kết quả BiLSTM + Self-Attention: Acc = {metrics_attention['accuracy']*100:.2f}% | F1 = {metrics_attention['f1_score']*100:.2f}%")
    plot_confusion_matrix(y_test, all_preds_attention, class_names, f"CM BiLSTM+Attention ({dataset_name})", os.path.join(results_dir, "cm_bilstm_attention.png"))
    
    # TRỰC QUAN HÓA ATTENTION WEIGHTS CHO CÁC MẪU TIÊU BIỂU
    print("\n--- Trực quan hóa Attention Heatmaps trên các mẫu câu thực tế ---")
    def visualize_sample_attention(sample_text: str, label_desc: str, out_filename: str):
        toks = tokenize(sample_text)[:30]
        indices = [vocab.get(w, 1) for w in toks]
        x_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        with torch.no_grad():
            _, weights = model_attention.get_document_vector_and_weights(x_tensor)
            w_np = weights[0].cpu().numpy()[:len(toks)]
        plot_attention_heatmap(
            toks, w_np,
            title=f"Phân bố Attention ({label_desc}) - {dataset_name}",
            save_path=os.path.join(results_dir, out_filename)
        )
        
    pos_samples = test_df[test_df["label"] == 1]["text"].tolist()
    neg_samples = test_df[test_df["label"] == 0]["text"].tolist()
    if pos_samples:
        visualize_sample_attention(pos_samples[0], "Tích cực", "attention_sample_pos.png")
    if neg_samples:
        visualize_sample_attention(neg_samples[0], "Tiêu cực", "attention_sample_neg.png")
        
    # 9. MÔ HÌNH 6: PhoBERT TRANSFORMER (Mở rộng theo Hướng phát triển)
    print("\n--- [6b/6] Mô hình 6: PhoBERT Fine-tuning (vinai/phobert-base-v2) ---")
    phobert_save_path = os.path.join(models_dir, f"best_phobert_{dataset_name}.pt")
    phobert_model, phobert_tokenizer, phobert_hist = train_phobert(
        train_texts=train_texts,
        train_labels=y_train.tolist(),
        val_texts=test_texts,
        val_labels=y_test.tolist(),
        model_name="vinai/phobert-base-v2",
        num_classes=2,
        batch_size=32,
        max_len=128,
        epochs=3,
        lr=2e-5,
        device=device,
        save_model_path=phobert_save_path
    )
    
    metrics_phobert, all_preds_phobert, phobert_cls_embeddings = evaluate_phobert(
        phobert_model,
        phobert_tokenizer,
        test_texts,
        y_test.tolist(),
        batch_size=32,
        max_len=128,
        device=device
    )
    print(f"Kết quả PhoBERT: Acc = {metrics_phobert['accuracy']*100:.2f}% | F1 = {metrics_phobert['f1_score']*100:.2f}%")
    plot_confusion_matrix(y_test, all_preds_phobert, class_names, f"CM PhoBERT ({dataset_name})", os.path.join(results_dir, "cm_phobert.png"))
    
    # 10. TRÍCH XUẤT BIỂU DIỄN VĂN BẢN VÀ TRỰC QUAN HÓA t-SNE
    print("\n--- Trích xuất Document Vectors và vẽ biểu đồ t-SNE ---")
    lstm_doc_vectors = extract_lstm_representations(model_pretrained, test_loader, device=device)
    plot_tsne_embeddings(
        X_test_avg_w2v, y_test, ["Tiêu cực", "Tích cực"],
        f"t-SNE Average Word2Vec ({dataset_name})",
        os.path.join(results_dir, "tsne_avg_word2vec.png")
    )
    plot_tsne_embeddings(
        lstm_doc_vectors, y_test, ["Tiêu cực", "Tích cực"],
        f"t-SNE BiLSTM Representation ({dataset_name})",
        os.path.join(results_dir, "tsne_lstm_representation.png")
    )
    plot_tsne_embeddings(
        phobert_cls_embeddings, y_test, ["Tiêu cực", "Tích cực"],
        f"t-SNE PhoBERT Contextual Embeddings ({dataset_name})",
        os.path.join(results_dir, "tsne_phobert_representation.png")
    )
    
    return {
        "dataset": dataset_name,
        "TF-IDF": metrics_tfidf,
        "Avg_Word2Vec": metrics_w2v,
        "LSTM_Scratch": metrics_scratch,
        "LSTM_Pretrained": metrics_pretrained,
        "BiLSTM_Attention": metrics_attention,
        "PhoBERT": metrics_phobert
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base = os.path.join(base_dir, "data")
    results_base = os.path.join(base_dir, "results")
    models_base = os.path.join(base_dir, "models")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"=== BẮT ĐẦU THỰC NGHIỆM ĐỐI CHUẨN 6 MÔ HÌNH VỚI THIẾT BỊ: {device} ===")
    
    # 1. Chạy trên Dataset 1: UIT-VSFC
    res_uit = run_experiment_on_dataset(
        dataset_name="UIT-VSFC",
        data_dir=os.path.join(data_base, "uit_vsfc"),
        results_dir=os.path.join(results_base, "uit_vsfc"),
        models_dir=models_base,
        device=device
    )
    
    # 2. Chạy trên Dataset 2: E-Commerce
    res_ecom = run_experiment_on_dataset(
        dataset_name="E-Commerce",
        data_dir=os.path.join(data_base, "ecommerce"),
        results_dir=os.path.join(results_base, "ecommerce"),
        models_dir=models_base,
        device=device
    )
    
    # 3. TỔNG HỢP VÀ SO SÁNH GIỮA 2 TẬP DỮ LIỆU
    model_mapping = [
        ("TF-IDF", "TF-IDF + Logistic Regression (Baseline)"),
        ("Avg_Word2Vec", "Average Word2Vec + Logistic Regression"),
        ("LSTM_Scratch", "BiLSTM Classifier (Học từ đầu)"),
        ("LSTM_Pretrained", "BiLSTM Classifier (Pre-trained Word2Vec)"),
        ("BiLSTM_Attention", "BiLSTM + Self-Attention (Đề xuất)"),
        ("PhoBERT", "PhoBERT Base v2 (Transformer SOTA)")
    ]
    
    comparison_rows = []
    for model_key, model_name in model_mapping:
        uit_m = res_uit[model_key]
        ecom_m = res_ecom[model_key]
        comparison_rows.append({
            "Mô hình": model_name,
            "UIT-VSFC Acc": f"{uit_m['accuracy']*100:.2f}%",
            "UIT-VSFC F1": f"{uit_m['f1_score']*100:.2f}%",
            "E-Commerce Acc": f"{ecom_m['accuracy']*100:.2f}%",
            "E-Commerce F1": f"{ecom_m['f1_score']*100:.2f}%"
        })
        
    df_compare = pd.DataFrame(comparison_rows)
    df_compare.to_csv(os.path.join(results_base, "cross_dataset_summary.csv"), index=False, encoding="utf-8")
    
    print("\n" + "="*85)
    print("=== BẢNG TỔNG HỢP ĐỐI CHUẨN 6 MÔ HÌNH TRÊN 2 TẬP DỮ LIỆU THỰC TẾ ===")
    print("="*85)
    print(df_compare.to_string(index=False))
    print("="*85)
    
    # 4. VẼ BIỂU ĐỒ SO SÁNH ĐỐI CHUẨN 6 MÔ HÌNH TRÊN 2 TẬP DỮ LIỆU
    model_labels = ["TF-IDF", "Avg W2V", "BiLSTM\n(Scratch)", "BiLSTM\n(Pretrained)", "BiLSTM+\nAttention", "PhoBERT\n(Transformer)"]
    keys = ["TF-IDF", "Avg_Word2Vec", "LSTM_Scratch", "LSTM_Pretrained", "BiLSTM_Attention", "PhoBERT"]
    
    acc_uit = [res_uit[k]["accuracy"] * 100 for k in keys]
    acc_ecom = [res_ecom[k]["accuracy"] * 100 for k in keys]
    
    x = np.arange(len(model_labels))
    width = 0.38
    
    plt.figure(figsize=(12, 6.5))
    rects1 = plt.bar(x - width/2, acc_uit, width, label='UIT-VSFC (Giáo dục / Học thuật)', color='#2980b9', edgecolor='black', linewidth=0.8)
    rects2 = plt.bar(x + width/2, acc_ecom, width, label='E-Commerce (Thương mại điện tử)', color='#d35400', edgecolor='black', linewidth=0.8)
    
    plt.ylabel('Độ chính xác - Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Đối Chuẩn Hiệu Năng 6 Mô Hình trên 2 Bộ Dữ Liệu Thực Tế', fontsize=14, fontweight='bold')
    plt.xticks(x, model_labels, fontsize=10.5, fontweight='bold')
    plt.ylim(60, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    chart_path = os.path.join(results_base, "cross_dataset_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nĐã lưu biểu đồ đối chuẩn 6 mô hình tại: {chart_path}")

if __name__ == "__main__":
    main()
