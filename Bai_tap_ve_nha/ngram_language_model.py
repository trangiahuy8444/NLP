#!/usr/bin/env python3
"""Mô hình ngôn ngữ N-gram đơn giản, chỉ dùng thư viện chuẩn Python."""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


START = "<s>"
END = "</s>"
UNK = "<unk>"


def tach_cau(van_ban: str) -> list[str]:
    """Tách văn bản thành câu theo các dấu kết thúc câu thông dụng."""
    return [cau.strip() for cau in re.split(r"[.!?]+", van_ban) if cau.strip()]


def tach_tu(cau: str) -> list[str]:
    """Tách từ Unicode, giữ số và các từ có dấu gạch nối/dấu nháy."""
    mau = r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+(?:[.,]\d+)*"
    return re.findall(mau, cau.lower(), flags=re.UNICODE)


class NGramLanguageModel:
    def __init__(self, n: int = 2, smoothing_k: float = 0.1, min_count: int = 1):
        if n < 1:
            raise ValueError("n phải lớn hơn hoặc bằng 1")
        if smoothing_k < 0:
            raise ValueError("smoothing_k không được âm")

        self.n = n
        self.smoothing_k = smoothing_k
        self.min_count = min_count
        self.vocab: set[str] = set()
        self.ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.so_cau_huan_luyen = 0

    def _chuan_hoa_tu(self, tu: str) -> str:
        if tu == START:
            return START
        return tu if tu in self.vocab else UNK

    def _chuan_hoa_ngu_canh(self, ngu_canh: Iterable[str]) -> tuple[str, ...]:
        if self.n == 1:
            return ()
        cac_tu = [self._chuan_hoa_tu(tu.lower()) for tu in ngu_canh]
        cac_tu = cac_tu[-(self.n - 1) :]
        if len(cac_tu) < self.n - 1:
            cac_tu = [START] * (self.n - 1 - len(cac_tu)) + cac_tu
        return tuple(cac_tu)

    def fit(self, van_ban: str) -> "NGramLanguageModel":
        """Huấn luyện mô hình từ một chuỗi văn bản."""
        cac_cau = [tach_tu(cau) for cau in tach_cau(van_ban)]
        cac_cau = [cau for cau in cac_cau if cau]
        tan_suat_tu = Counter(tu for cau in cac_cau for tu in cau)

        self.vocab = {
            tu for tu, so_lan in tan_suat_tu.items() if so_lan >= self.min_count
        }
        self.vocab.update({UNK, END})
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.so_cau_huan_luyen = len(cac_cau)

        for cau in cac_cau:
            cau = [tu if tu in self.vocab else UNK for tu in cau]
            chuoi = [START] * (self.n - 1) + cau + [END]

            for vi_tri in range(self.n - 1, len(chuoi)):
                if self.n == 1:
                    ngu_canh = ()
                else:
                    ngu_canh = tuple(chuoi[vi_tri - self.n + 1 : vi_tri])
                tu_tiep_theo = chuoi[vi_tri]
                self.ngram_counts[ngu_canh][tu_tiep_theo] += 1
                self.context_counts[ngu_canh] += 1

        return self

    def xac_suat(self, tu: str, ngu_canh: Iterable[str] = ()) -> float:
        """Tính P(từ | ngữ cảnh) với add-k smoothing."""
        tu = self._chuan_hoa_tu(tu.lower())
        ngu_canh_chuan = self._chuan_hoa_ngu_canh(ngu_canh)
        tu_so = self.ngram_counts[ngu_canh_chuan][tu] + self.smoothing_k
        mau_so = (
            self.context_counts[ngu_canh_chuan]
            + self.smoothing_k * len(self.vocab)
        )
        return tu_so / mau_so if mau_so > 0 else 0.0

    def top_tu_tiep_theo(
        self, ngu_canh: Iterable[str], so_luong: int = 5
    ) -> list[tuple[str, float]]:
        """Trả về các từ tiếp theo có xác suất cao nhất."""
        ket_qua = [
            (tu, self.xac_suat(tu, ngu_canh))
            for tu in self.vocab
            if tu != UNK
        ]
        return sorted(ket_qua, key=lambda item: item[1], reverse=True)[:so_luong]

    def log_xac_suat_cau(self, cau: str) -> float:
        """Tính log xác suất tự nhiên của một câu."""
        cac_tu = [self._chuan_hoa_tu(tu) for tu in tach_tu(cau)] + [END]
        lich_su = [START] * (self.n - 1)
        tong_log = 0.0

        for tu in cac_tu:
            ngu_canh = lich_su[-(self.n - 1) :] if self.n > 1 else []
            p = self.xac_suat(tu, ngu_canh)
            if p == 0:
                return -math.inf
            tong_log += math.log(p)
            lich_su.append(tu)

        return tong_log

    def xac_suat_cau(self, cau: str) -> float:
        log_p = self.log_xac_suat_cau(cau)
        return math.exp(log_p) if math.isfinite(log_p) else 0.0

    def perplexity(self, van_ban: str) -> float:
        """Tính perplexity trung bình trên một đoạn văn bản."""
        tong_log = 0.0
        tong_so_tu = 0

        for cau in tach_cau(van_ban):
            so_tu = len(tach_tu(cau)) + 1  # Cộng token kết thúc câu.
            log_p = self.log_xac_suat_cau(cau)
            if not math.isfinite(log_p):
                return math.inf
            tong_log += log_p
            tong_so_tu += so_tu

        return math.exp(-tong_log / tong_so_tu) if tong_so_tu else math.inf

    def sinh_cau(
        self,
        do_dai_toi_da: int = 20,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Sinh câu bằng cách lấy mẫu tuần tự từ phân phối xác suất."""
        if temperature <= 0:
            raise ValueError("temperature phải lớn hơn 0")

        bo_sinh_so = random.Random(seed)
        lich_su = [START] * (self.n - 1)
        ket_qua: list[str] = []

        for _ in range(do_dai_toi_da):
            ngu_canh = lich_su[-(self.n - 1) :] if self.n > 1 else []
            ngu_canh_chuan = self._chuan_hoa_ngu_canh(ngu_canh)
            cac_tu_da_gap = list(self.ngram_counts[ngu_canh_chuan])
            # Khi sinh câu, ưu tiên các chuyển tiếp đã học để văn bản dễ đọc.
            # Smoothing vẫn được dùng khi tính xác suất và perplexity.
            tap_ung_vien = cac_tu_da_gap if cac_tu_da_gap else list(self.vocab)
            ung_vien = [tu for tu in tap_ung_vien if tu != UNK]
            trong_so = [
                self.xac_suat(tu, ngu_canh) ** (1.0 / temperature)
                for tu in ung_vien
            ]
            tu_moi = bo_sinh_so.choices(ung_vien, weights=trong_so, k=1)[0]

            if tu_moi == END:
                break
            ket_qua.append(tu_moi)
            lich_su.append(tu_moi)

        return " ".join(ket_qua)


def main() -> None:
    tep_mac_dinh = Path(__file__).with_name("du_lieu_mau.txt")
    parser = argparse.ArgumentParser(
        description="Huấn luyện và thử nghiệm mô hình ngôn ngữ N-gram."
    )
    parser.add_argument("--corpus", type=Path, default=tep_mac_dinh)
    parser.add_argument("--n", type=int, default=2, help="Bậc N-gram")
    parser.add_argument("--k", type=float, default=0.1, help="Hệ số add-k smoothing")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--sentence", default="xử lý ngôn ngữ tự nhiên")
    parser.add_argument("--context", default="mô hình")
    parser.add_argument("--generate", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    van_ban = args.corpus.read_text(encoding="utf-8")
    mo_hinh = NGramLanguageModel(
        n=args.n, smoothing_k=args.k, min_count=args.min_count
    ).fit(van_ban)

    print(f"Mô hình: {args.n}-gram")
    print(f"Số câu huấn luyện: {mo_hinh.so_cau_huan_luyen}")
    print(f"Kích thước từ vựng: {len(mo_hinh.vocab)}")
    print(f"Số ngữ cảnh đã gặp: {len(mo_hinh.context_counts)}")
    print()
    print(f"Câu cần kiểm tra: {args.sentence!r}")
    print(f"Log xác suất: {mo_hinh.log_xac_suat_cau(args.sentence):.6f}")
    print(f"Xác suất: {mo_hinh.xac_suat_cau(args.sentence):.6e}")
    print(f"Perplexity: {mo_hinh.perplexity(args.sentence):.6f}")
    print()

    ngu_canh = tach_tu(args.context)
    print(f"Các từ có thể đứng sau {args.context!r}:")
    for tu, p in mo_hinh.top_tu_tiep_theo(ngu_canh):
        print(f"  {tu:<15} {p:.6f}")

    print("\nCác câu được sinh:")
    for chi_so in range(args.generate):
        cau = mo_hinh.sinh_cau(seed=args.seed + chi_so)
        print(f"  {chi_so + 1}. {cau}")


if __name__ == "__main__":
    main()
