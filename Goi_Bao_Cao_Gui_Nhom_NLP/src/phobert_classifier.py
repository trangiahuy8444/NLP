"""
Module Fine-tuning Mô hình Ngôn ngữ Transformer Tiếng Việt: PhoBERT (vinai/phobert-base-v2)
Phục vụ mục tiêu:
1. Mở rộng thực nghiệm so sánh với các kiến trúc truyền thống (TF-IDF, Word2Vec, BiLSTM)
2. Fine-tune phân loại cảm xúc văn bản tiếng Việt trên 2 tập dữ liệu thực tế
3. Trích xuất vector biểu diễn văn bản ngữ cảnh (Contextual Document Embeddings)
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from evaluate import compute_metrics

class PhoBERTTextDataset(Dataset):
    """Dataset định dạng cho PhoBERT Tokenizer."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_phobert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    model_name: str = "vinai/phobert-base-v2",
    num_classes: int = 2,
    batch_size: int = 32,
    max_len: int = 128,
    epochs: int = 3,
    lr: float = 2e-5,
    device: Optional[torch.device] = None,
    save_model_path: Optional[str] = None
) -> Tuple[nn.Module, AutoTokenizer, Dict[str, List[float]]]:
    """
    Huấn luyện fine-tune PhoBERT trên tập dữ liệu tiếng Việt.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"Khởi tạo PhoBERT: {model_name} trên thiết bị: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model.to(device)

    train_dataset = PhoBERTTextDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset = PhoBERTTextDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        metrics = compute_metrics(val_targets, val_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["f1_score"])

        print(f"PhoBERT Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {metrics['accuracy']*100:.2f}% | Val F1: {metrics['f1_score']*100:.2f}%")

        if metrics["f1_score"] > best_val_f1 and save_model_path:
            best_val_f1 = metrics["f1_score"]
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model.state_dict(), save_model_path)

    if save_model_path and os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path))

    return model, tokenizer, history

def evaluate_phobert(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    labels: List[int],
    batch_size: int = 32,
    max_len: int = 128,
    device: Optional[torch.device] = None
) -> Tuple[Dict[str, float], List[int], np.ndarray]:
    """
    Đánh giá mô hình PhoBERT trên tập kiểm thử và trích xuất embedding đại diện câu.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model.to(device)
    model.eval()

    dataset = PhoBERTTextDataset(texts, labels, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

            # Lấy vector biểu diễn của token <s> (tương đương CLS) ở tầng ẩn cuối cùng
            last_hidden_state = outputs.hidden_states[-1] # (batch_size, seq_len, hidden_dim)
            cls_repr = last_hidden_state[:, 0, :].cpu().numpy() # (batch_size, hidden_dim)
            all_embeddings.append(cls_repr)

    metrics = compute_metrics(labels, all_preds)
    embeddings = np.vstack(all_embeddings)
    return metrics, all_preds, embeddings
