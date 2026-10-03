#!/bin/bash
# Script biên dịch bài giảng Beamer nhập môn sang PDF
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== ĐANG BIÊN DỊCH BÀI GIẢNG NHẬP MÔN BEAMER HCMUE ==="
/Library/TeX/texbin/pdflatex -interaction=nonstopmode -output-directory="$DIR" "$DIR/bai_giang_nhap_mon.tex"
/Library/TeX/texbin/pdflatex -interaction=nonstopmode -output-directory="$DIR" "$DIR/bai_giang_nhap_mon.tex"

echo "=== BIÊN DỊCH THÀNH CÔNG! ==="
ls -lh "$DIR/bai_giang_nhap_mon.pdf"
