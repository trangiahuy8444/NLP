"""
Module huấn luyện mô hình biểu diễn từ Word2Vec (CBOW và Skip-Gram) bằng thư viện Gensim.
Cung cấp các hàm kiểm tra độ tương đồng ngữ nghĩa và trích xuất ma trận embedding.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
from gensim.models import Word2Vec

def train_word2vec(
    tokenized_sentences: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    sg: int = 0,  # 0 for CBOW, 1 for Skip-gram
    epochs: int = 30,
    workers: int = 4
) -> Word2Vec:
    """
    Huấn luyện mô hình Word2Vec.
    
    Args:
        tokenized_sentences: Danh sách các câu đã được tách từ.
        vector_size: Số chiều của vector biểu diễn từ.
        window: Bán kính cửa sổ ngữ cảnh xung quanh từ mục tiêu.
        min_count: Tần suất xuất hiện tối thiểu của từ để đưa vào từ vựng.
        sg: 0 là CBOW, 1 là Skip-Gram.
        epochs: Số vòng lặp huấn luyện.
        workers: Số luồng xử lý song song.
    
    Returns:
        Mô hình Word2Vec đã huấn luyện.
    """
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
        seed=42
    )
    return model

def save_word2vec_model(model: Word2Vec, filepath: str) -> None:
    """Lưu mô hình Word2Vec ra đĩa."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)

def load_word2vec_model(filepath: str) -> Word2Vec:
    """Tải mô hình Word2Vec từ đĩa."""
    return Word2Vec.load(filepath)

def get_word_similarities(model: Word2Vec, word: str, topn: int = 5) -> List[Tuple[str, float]]:
    """Tìm top n từ tương đồng ngữ nghĩa nhất với từ cho trước."""
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    return []

def build_embedding_matrix(
    word2vec_model: Word2Vec,
    vocab: dict[str, int],
    embedding_dim: int = 100
) -> np.ndarray:
    """
    Xây dựng ma trận trọng số embedding cho tầng nn.Embedding của PyTorch.
    
    Args:
        word2vec_model: Mô hình Word2Vec đã huấn luyện.
        vocab: Từ điển ánh xạ word -> index.
        embedding_dim: Kích thước vector từ.
        
    Returns:
        numpy array kích thước (vocab_size, embedding_dim).
    """
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    for word, idx in vocab.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]
        else:
            # Từ ngoài từ điển (<pad>, <unk> hoặc OOV) khởi tạo ngẫu nhiên nhỏ
            if word == "<pad>":
                embedding_matrix[idx] = np.zeros(embedding_dim, dtype=np.float32)
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.1, size=(embedding_dim,)).astype(np.float32)
                
    return embedding_matrix
