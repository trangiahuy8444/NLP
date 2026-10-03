#!/bin/bash
# Script tự động biên dịch bài thuyết trình Beamer HCMUE sang PDF
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== ĐANG BIÊN DỊCH BÀI THUYẾT TRÌNH BEAMER HCMUE ==="
/Library/TeX/texbin/pdflatex -interaction=nonstopmode -output-directory="$DIR" "$DIR/presentation.tex"
/Library/TeX/texbin/pdflatex -interaction=nonstopmode -output-directory="$DIR" "$DIR/presentation.tex"

echo "=== BIÊN DỊCH THÀNH CÔNG! ==="
ls -lh "$DIR/presentation.pdf"
