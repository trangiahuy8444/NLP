"""
Script tải và chuẩn bị 2 bộ dữ liệu thực tế tiếng Việt chuẩn:
1. Dataset 1: UIT-VSFC (Vietnamese Students' Feedback Corpus) - Miền giáo dục / học thuật
2. Dataset 2: Vietnamese E-Commerce / Social Reviews - Miền thương mại điện tử / mạng xã hội
"""

import os
import pandas as pd
import numpy as np

def prepare_uit_vsfc(out_dir: str, train_samples: int = 2500, test_samples: int = 600):
    print("\n[1/2] Đang tải bộ dữ liệu thực tế UIT-VSFC...")
    os.makedirs(out_dir, exist_ok=True)
    
    url_train = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    url_test = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    
    df_train_raw = pd.read_parquet(url_train)
    df_test_raw = pd.read_parquet(url_test)
    
    # Lọc bài toán nhị phân (0: Tiêu cực, 2: Tích cực -> chuyển 2 thành 1)
    df_train_bin = df_train_raw[df_train_raw["sentiment"].isin([0, 2])].copy()
    df_train_bin["label"] = df_train_bin["sentiment"].map({0: 0, 2: 1})
    df_train_bin = df_train_bin.rename(columns={"sentence": "text"})[["text", "label"]]
    
    df_test_bin = df_test_raw[df_test_raw["sentiment"].isin([0, 2])].copy()
    df_test_bin["label"] = df_test_bin["sentiment"].map({0: 0, 2: 1})
    df_test_bin = df_test_bin.rename(columns={"sentence": "text"})[["text", "label"]]
    
    # Lấy mẫu cân bằng để thời gian huấn luyện tối ưu
    pos_train = df_train_bin[df_train_bin["label"] == 1].sample(n=train_samples // 2, random_state=42)
    neg_train = df_train_bin[df_train_bin["label"] == 0].sample(n=train_samples // 2, random_state=42)
    train_df = pd.concat([pos_train, neg_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    pos_test = df_test_bin[df_test_bin["label"] == 1].sample(n=test_samples // 2, random_state=42)
    neg_test = df_test_bin[df_test_bin["label"] == 0].sample(n=test_samples // 2, random_state=42)
    test_df = pd.concat([pos_test, neg_test]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False, encoding="utf-8")
    print(f" -> Đã lưu UIT-VSFC: Train={len(train_df)} mẫu, Test={len(test_df)} mẫu tại {out_dir}")

def prepare_ecommerce(out_dir: str, train_samples: int = 2500, test_samples: int = 600):
    print("\n[2/2] Đang tải bộ dữ liệu thực tế Thương mại điện tử (E-Commerce Reviews)...")
    os.makedirs(out_dir, exist_ok=True)
    
    url_train = "https://huggingface.co/datasets/h-i-e-u/vietnamese-SA-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    url_test = "https://huggingface.co/datasets/h-i-e-u/vietnamese-SA-dataset/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    
    df_train_raw = pd.read_parquet(url_train)
    df_test_raw = pd.read_parquet(url_test)
    
    # Chuẩn hóa nhãn về chữ in hoa (POS, NEG, NEU)
    df_train_raw["label"] = df_train_raw["label"].astype(str).str.upper()
    df_test_raw["label"] = df_test_raw["label"].astype(str).str.upper()
    
    # Lọc POS (1) và NEG (0)
    df_train_bin = df_train_raw[df_train_raw["label"].isin(["POS", "NEG"])].copy()
    df_train_bin["label"] = df_train_bin["label"].map({"NEG": 0, "POS": 1})
    df_train_bin = df_train_bin[["text", "label"]].dropna()
    
    df_test_bin = df_test_raw[df_test_raw["label"].isin(["POS", "NEG"])].copy()
    df_test_bin["label"] = df_test_bin["label"].map({"NEG": 0, "POS": 1})
    df_test_bin = df_test_bin[["text", "label"]].dropna()
    
    # Lấy mẫu cân bằng
    pos_train = df_train_bin[df_train_bin["label"] == 1].sample(n=train_samples // 2, random_state=42)
    neg_train = df_train_bin[df_train_bin["label"] == 0].sample(n=train_samples // 2, random_state=42)
    train_df = pd.concat([pos_train, neg_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    pos_test = df_test_bin[df_test_bin["label"] == 1].sample(n=test_samples // 2, random_state=42)
    neg_test = df_test_bin[df_test_bin["label"] == 0].sample(n=test_samples // 2, random_state=42)
    test_df = pd.concat([pos_test, neg_test]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False, encoding="utf-8")
    print(f" -> Đã lưu E-Commerce Reviews: Train={len(train_df)} mẫu, Test={len(test_df)} mẫu tại {out_dir}")

if __name__ == "__main__":
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    prepare_uit_vsfc(os.path.join(base_data_dir, "uit_vsfc"), train_samples=2500, test_samples=600)
    prepare_ecommerce(os.path.join(base_data_dir, "ecommerce"), train_samples=2500, test_samples=600)
    print("\nHoàn tất tải và chuẩn bị 2 bộ dữ liệu thực tế!")
