"""
Module biểu diễn văn bản (Document / Sentence Embedding) bằng vector.
Hỗ trợ các phương pháp:
1. Thống kê: Average Word2Vec (Trung bình cộng các vector từ)
2. Thống kê có trọng số: TF-IDF Weighted Average Word2Vec
3. Học sâu chuỗi: Trích xuất vector biểu diễn văn bản từ tầng ẩn của mô hình LSTM
"""

from typing import List, Optional
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def get_average_word2vec(
    tokens: List[str],
    model: Word2Vec,
    vector_size: Optional[int] = None
) -> np.ndarray:
    """
    Tính vector biểu diễn văn bản bằng trung bình cộng các vector từ Word2Vec.
    
    Formula:
        v_doc = (1 / |D|) * sum_{w in D} v_w
    """
    dim = vector_size or model.vector_size
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def compute_corpus_average_word2vec(
    tokenized_docs: List[List[str]],
    model: Word2Vec
) -> np.ndarray:
    """Trích xuất vector cho toàn bộ tập văn bản bằng Average Word2Vec."""
    matrix = [get_average_word2vec(tokens, model) for tokens in tokenized_docs]
    return np.array(matrix, dtype=np.float32)

def get_tfidf_weighted_word2vec(
    tokens: List[str],
    model: Word2Vec,
    tfidf_weights: dict[str, float],
    vector_size: Optional[int] = None
) -> np.ndarray:
    """
    Tính vector biểu diễn văn bản có trọng số TF-IDF:
        v_doc = (sum_{w in D} tfidf(w) * v_w) / (sum_{w in D} tfidf(w) + eps)
    """
    dim = vector_size or model.vector_size
    weighted_vectors = []
    weights = []
    
    for w in tokens:
        if w in model.wv:
            weight = tfidf_weights.get(w, 1.0)
            weighted_vectors.append(weight * model.wv[w])
            weights.append(weight)
            
    if len(weighted_vectors) == 0 or sum(weights) == 0:
        return np.zeros(dim, dtype=np.float32)
    
    return (np.sum(weighted_vectors, axis=0) / sum(weights)).astype(np.float32)

def extract_lstm_representations(
    lstm_model,
    dataloader,
    device: torch.device
) -> np.ndarray:
    """
    Trích xuất vector biểu diễn văn bản (Document Vectors) do mô hình LSTM học được.
    
    Returns:
        np.ndarray kích thước (num_samples, hidden_dim)
    """
    lstm_model.eval()
    representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, _, lengths = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            doc_vecs = lstm_model.get_document_vector(inputs, lengths)
            representations.append(doc_vecs.cpu().numpy())
            
    return np.concatenate(representations, axis=0)
