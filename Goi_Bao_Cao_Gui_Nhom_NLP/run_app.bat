@echo off
REM ==============================================================================
REM HCMUE NLP SENTIMENT AI DEMO - Script Khởi Chạy Ứng Dụng (Windows)
REM ==============================================================================

chcp 65001 >nul
echo ======================================================================
echo 🎓 TRƯỜNG ĐẠI HỌC SƯ PHẠM THÀNH PHỐ HỒ CHÍ MINH (HCMUE)
echo    KHOA KHOA HỌC MÁY TÍNH - KHÓA 36 (2025 - 2027)
echo ----------------------------------------------------------------------
echo ⚡ HỆ THỐNG PHÂN TÍCH CẢM XÚC TIẾNG VIỆT ĐA MIỀN (NLP DEMO STUDIO)
echo    Đối Chuẩn 6 Mô Hình: TF-IDF, Word2Vec, BiLSTM, Attention, PhoBERT
echo ======================================================================

set PORT=8501
echo.
echo 🚀 Đang khởi động Web Server tại: http://localhost:%PORT%
echo 👉 Nhấn Ctrl + C để dừng máy chủ.
echo.

python app.py --port %PORT%
pause
