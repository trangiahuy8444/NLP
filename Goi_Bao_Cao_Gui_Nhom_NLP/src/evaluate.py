"""
Module đánh giá mô hình và trực quan hóa kết quả:
1. Tính các độ đo: Accuracy, Precision, Recall, F1-Score
2. Vẽ ma trận nhầm lẫn (Confusion Matrix)
3. Vẽ đường cong hàm mất mát và độ chính xác qua các Epoch
4. Trực quan hóa không gian vector biểu diễn văn bản bằng t-SNE / PCA
"""

import os
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Tính toán các chỉ số đánh giá tiêu chuẩn."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1_score": f1
    }

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str = None
):
    """Vẽ ma trận nhầm lẫn chi tiết."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold"
            )
            
    plt.ylabel("Nhãn thực tế (True Label)", fontsize=11)
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """Vẽ biểu đồ Loss và Accuracy qua từng Epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.set_title("Biến thiên Hàm Mất Mát (Loss)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # 2. Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Val Accuracy", linewidth=2)
    ax2.set_title("Biến thiên Độ Chính Xác (Accuracy)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Độ chính xác (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: List[int],
    class_names: List[str],
    title: str = "t-SNE Biểu diễn Văn bản",
    save_path: str = None
):
    """Chiếu không gian vector biểu diễn văn bản nhiều chiều về 2D bằng t-SNE."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(labels)//10)))
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    
    for idx, name in enumerate(class_names):
        mask = (np.array(labels) == idx)
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=colors[idx % len(colors)],
            label=name,
            alpha=0.75,
            edgecolors="none",
            s=40
        )
        
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE Chiều 1", fontsize=11)
    plt.ylabel("t-SNE Chiều 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_attention_heatmap(
    tokens: List[str],
    weights: np.ndarray,
    title: str = "Trực quan hóa Trọng số Chú ý (Self-Attention Weights)",
    save_path: str = None
):
    """Vẽ Heatmap thể hiện trọng số chú ý (Attention weights) của mô hình trên từng token."""
    weights = np.array(weights).reshape(1, -1)
    valid_len = len(tokens)
    weights = weights[:, :valid_len]
    
    plt.figure(figsize=(max(8, len(tokens) * 0.85), 3.2))
    plt.imshow(weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.05, float(np.max(weights)) * 1.1))
    plt.yticks([])
    plt.xticks(range(len(tokens)), tokens, rotation=35, ha="right", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.colorbar(orientation="horizontal", pad=0.4, shrink=0.5, label="Attention Weight (Trọng số chú ý)")
    
    for i in range(len(tokens)):
        w = weights[0, i]
        plt.text(i, 0, f"{w:.3f}", ha="center", va="center", 
                 color="white" if w > (float(np.max(weights))*0.6) else "black", 
                 fontsize=10, fontweight="bold")
                 
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
