#!/bin/bash
# ==============================================================================
# Script Khởi Chạy Ứng Dụng Thực Tế Phân Tích Cảm Xúc Đa Miền (NLP Demo App)
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "⚡ KHỞI CHẠY HỆ THỐNG PHÂN TÍCH CẢM XÚC TIẾNG VIỆT ĐA MIỀN"
echo "   Đối Chuẩn 6 Mô Hình: TF-IDF, Word2Vec, BiLSTM, Attention, PhoBERT"
echo "======================================================================"

# 1. Tìm kiếm Python interpreter phù hợp
PYTHON_BIN=""

if [ -f "/Users/huytran/miniconda3/envs/ML/bin/python" ]; then
    PYTHON_BIN="/Users/huytran/miniconda3/envs/ML/bin/python"
elif command -v conda &> /dev/null && conda info --envs | grep -q "ML"; then
    CONDA_PREFIX=$(conda info --envs | grep "ML" | awk '{print $NF}')
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "🔹 Sử dụng Python: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print(f'   PyTorch Version: {torch.__version__} | Device: {\"mps\" if torch.backends.mps.is_available() else (\"cuda\" if torch.cuda.is_available() else \"cpu\")}')" 2>/dev/null

PORT=8501

echo ""
echo "🚀 Đang khởi động Web Server tại: http://localhost:$PORT"
echo "👉 Ứng dụng sẽ tự động mở trong trình duyệt của bạn."
echo "👉 Để dừng ứng dụng, nhấn phím Ctrl + C trong Terminal này."
echo "======================================================================"
echo ""

# Chạy ứng dụng web
$PYTHON_BIN app.py --port $PORT
