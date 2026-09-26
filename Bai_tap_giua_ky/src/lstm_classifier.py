"""
Module xây dựng mô hình mạng LSTM (Long Short-Term Memory) bằng PyTorch
nhằm hai mục tiêu cốt lõi:
1. Học biểu diễn văn bản thành vector (Document Representation Vector) qua tầng LSTM
2. Phân loại cảm xúc văn bản (Text Classification) dựa trên vector biểu diễn đã học
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """Dataset cho văn bản phân loại với độ dài câu thay đổi."""
    def __init__(self, texts: List[List[str]], labels: List[int], vocab: Dict[str, int], max_len: int = 64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.get("<pad>", 0)
        self.unk_idx = vocab.get("<unk>", 1)
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        tokens = self.texts[idx]
        indices = [self.vocab.get(w, self.unk_idx) for w in tokens][:self.max_len]
        length = max(1, len(indices))
        
        # Padding
        if len(indices) < self.max_len:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long), self.labels[idx], length

def collate_fn(batch):
    """Ghép batch và sắp xếp theo độ dài giảm dần (nếu cần dùng pack_padded_sequence)."""
    texts, labels, lengths = zip(*batch)
    return torch.stack(texts), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


class LSTMTextClassifier(nn.Module):
    """
    Mô hình LSTM học biểu diễn văn bản và phân loại.
    
    Kiến trúc:
        Input Tokens -> Embedding Layer (Có thể nạp Pretrained Word2Vec)
                     -> LSTM Layer (Lấy trạng thái ẩn h_t)
                     -> Document Vector Representation (Last Step hoặc Mean-pooling)
                     -> Fully Connected + Dropout -> Logits Phân lớp
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        pooling: str = "last"  # 'last' hoặc 'mean'
    ):
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. Tầng Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                
        # 2. Tầng LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.rep_dim = hidden_dim * self.num_directions
        
        # 3. Tầng Phân loại (Classifier) dựa trên vector biểu diễn văn bản
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.rep_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def get_document_vector(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector biểu diễn văn bản (Document Vector) từ mô hình LSTM.
        
        Args:
            x: Tensor token indices (batch_size, seq_len)
            lengths: Tensor độ dài thực tế của câu (batch_size,)
            
        Returns:
            doc_vec: Tensor vector biểu diễn văn bản (batch_size, rep_dim)
        """
        # x: (batch_size, seq_len) -> embed: (batch_size, seq_len, emb_dim)
        embed = self.embedding(x)
        
        # Chạy qua LSTM
        lstm_out, (hn, cn) = self.lstm(embed)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        # hn: (num_layers * num_directions, batch_size, hidden_dim)
        
        if self.pooling == "mean":
            # Mask bỏ phần padding để tính trung bình chỉ trên các token thật
            mask = (x != self.embedding.padding_idx).unsqueeze(-1).float() # (batch, seq_len, 1)
            sum_hidden = torch.sum(lstm_out * mask, dim=1) # (batch, rep_dim)
            doc_vec = sum_hidden / lengths.unsqueeze(-1).clamp(min=1).float()
        else:
            # Lấy trạng thái ẩn ở bước cuối cùng (Last hidden state)
            if self.bidirectional:
                # Nối hidden state xuôi ở bước cuối và hidden state ngược ở bước đầu
                forward_last = hn[-2, :, :] # (batch_size, hidden_dim)
                backward_first = hn[-1, :, :] # (batch_size, hidden_dim)
                doc_vec = torch.cat([forward_last, backward_first], dim=1) # (batch, hidden_dim * 2)
            else:
                doc_vec = hn[-1, :, :] # (batch_size, hidden_dim)
                
        return doc_vec
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lan truyền xuôi (Forward pass):
        Bước 1: Trích xuất vector biểu diễn văn bản bằng LSTM
        Bước 2: Dự đoán nhãn phân loại từ vector biểu diễn
        
        Returns:
            logits: (batch_size, num_classes)
            doc_vec: (batch_size, rep_dim)
        """
        doc_vec = self.get_document_vector(x, lengths)
        logits = self.classifier(doc_vec)
        return logits, doc_vec


def train_model(
    model: LSTMTextClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 15,
    lr: float = 1e-3,
    device: Optional[torch.device] = None
) -> dict:
    """
    Huấn luyện mô hình LSTM với optimizer Adam và CrossEntropyLoss.
    Theo dõi loss và accuracy qua từng epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(1, num_epochs + 1):
        # 1. Training loop
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            
        train_loss = total_loss / total
        train_acc = correct / total
        
        # 2. Validation loop
        model.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                logits, _ = model(x, lengths)
                loss = criterion(logits, y)
                
                val_total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)
                
        val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
    return history


class SelfAttention(nn.Module):
    """
    Cơ chế Self-Attention (Additive Attention theo phong cách Bahdanau).
    Tự động chấm điểm và gán trọng số cho từng token dựa trên mức độ quan trọng ngữ nghĩa:
        u_t = tanh(W_a * h_t + b_a)
        alpha_t = exp(u_t^T * v_a) / sum_j(exp(u_j^T * v_a))
        v_D = sum_t(alpha_t * h_t)
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, lstm_out: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) True cho token thật, False cho pad_idx
        Returns:
            doc_vec: (batch_size, hidden_dim)
            weights: (batch_size, seq_len)
        """
        # (batch_size, seq_len, attention_dim)
        u = self.projection(lstm_out)
        # (batch_size, seq_len)
        scores = self.context_vector(u).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            
        weights = torch.softmax(scores, dim=-1) # (batch_size, seq_len)
        
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_dim) -> (batch_size, 1, hidden_dim)
        doc_vec = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return doc_vec, weights


class BiLSTMAttentionClassifier(nn.Module):
    """
    Mô hình BiLSTM kết hợp cơ chế Self-Attention.
    Đầu ra văn bản được tổng hợp từ toàn bộ chuỗi trạng thái ẩn hai chiều
    thông qua ma trận trọng số chú ý thích nghi.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.rep_dim = hidden_dim * 2 # Hai hướng (Bidirectional)
        self.attention = SelfAttention(hidden_dim=self.rep_dim, attention_dim=attention_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.rep_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def get_document_vector_and_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)
        mask = (x != self.pad_idx)
        doc_vec, weights = self.attention(lstm_out, mask=mask)
        return doc_vec, weights
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        doc_vec, _ = self.get_document_vector_and_weights(x)
        logits = self.classifier(doc_vec)
        return logits, doc_vec
