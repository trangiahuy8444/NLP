#!/bin/bash
# Script tự động biên dịch báo cáo LaTeX sang PDF
set -e

echo "=== ĐANG BIÊN DỊCH BÁO CÁO LATEX SANG PDF ==="
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "=== BIÊN DỊCH THÀNH CÔNG! FILE PDF ĐÃ TẠO TẠI: main.pdf ==="
ls -lh main.pdf
