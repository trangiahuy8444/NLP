"""
Script download các bài báo khoa học và tài liệu tham khảo chính thức
được trích dẫn trong bài báo cáo tiểu luận môn Xử lý ngôn ngữ tự nhiên.
"""

import os
import urllib.request
import ssl

# Thiết lập context SSL nếu cần
ssl_context = ssl._create_unverified_context()

PAPERS = [
    {
        "filename": "01_UIT_VSFC_Sentiment_Analysis_2018.pdf",
        "title": "UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis (Nguyen et al., 2018)",
        "url": "https://arxiv.org/pdf/1908.08272.pdf"
    },
    {
        "filename": "02_Mikolov_Word2Vec_Vector_Space_2013.pdf",
        "title": "Efficient Estimation of Word Representations in Vector Space (Mikolov et al., ICLR 2013)",
        "url": "https://arxiv.org/pdf/1301.3781.pdf"
    },
    {
        "filename": "03_Mikolov_Word2Vec_Negative_Sampling_NeurIPS_2013.pdf",
        "title": "Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al., NeurIPS 2013)",
        "url": "https://arxiv.org/pdf/1310.4546.pdf"
    },
    {
        "filename": "04_Hochreiter_Schmidhuber_LSTM_1997.pdf",
        "title": "Long Short-Term Memory (Hochreiter & Schmidhuber, Neural Computation 1997)",
        "url": "https://www.bioinf.jku.at/publications/older/2604.pdf",
        "fallback_url": "https://people.idsia.ch/~juergen/lstm.pdf"
    },
    {
        "filename": "05_Yoav_Goldberg_NN_Primer_NLP_2016.pdf",
        "title": "A Primer on Neural Network Models for Natural Language Processing (Yoav Goldberg, 2016)",
        "url": "https://arxiv.org/pdf/1510.00726.pdf"
    },
    {
        "filename": "06_Jurafsky_Martin_Vector_Semantics_and_Embeddings.pdf",
        "title": "Speech and Language Processing - Ch 6: Vector Semantics and Embeddings (Jurafsky & Martin)",
        "url": "https://web.stanford.edu/~jurafsky/slp3/6.pdf"
    },
    {
        "filename": "06b_Jurafsky_Martin_RNNs_and_LSTMs.pdf",
        "title": "Speech and Language Processing - Ch 9: RNNs and LSTMs (Jurafsky & Martin)",
        "url": "https://web.stanford.edu/~jurafsky/slp3/9.pdf"
    },
    {
        "filename": "07_Rehurek_Sojka_Gensim_Framework_2010.pdf",
        "title": "Software Framework for Topic Modelling with Large Corpora - Gensim (Rehurek & Sojka, 2010)",
        "url": "https://radimrehurek.com/gensim/lrec2010_final.pdf"
    }
]

def download_file(url, target_path, fallback_url=None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as resp, open(target_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"  [Lỗi] Tải từ {url} thất bại: {e}")
        if fallback_url:
            print(f"  -> Thử tải từ URL dự phòng: {fallback_url}")
            try:
                req_fallback = urllib.request.Request(fallback_url, headers=headers)
                with urllib.request.urlopen(req_fallback, context=ssl_context, timeout=30) as resp, open(target_path, "wb") as f:
                    f.write(resp.read())
                return True
            except Exception as e2:
                print(f"  [Lỗi dự phòng] Thất bại: {e2}")
        return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "tai_lieu_tham_khao")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"=== BẮT ĐẦU TẢI CÁC TÀI LIỆU THAM KHẢO VÀO {target_dir} ===\n")
    
    success_count = 0
    for idx, paper in enumerate(PAPERS, 1):
        target_path = os.path.join(target_dir, paper["filename"])
        print(f"[{idx}/{len(PAPERS)}] Đang tải: {paper['title']}...")
        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
            print(f"  -> Đã tồn tại ({os.path.getsize(target_path) / 1024:.1f} KB). Bỏ qua.")
            success_count += 1
            continue
            
        ok = download_file(paper["url"], target_path, paper.get("fallback_url"))
        if ok:
            size_kb = os.path.getsize(target_path) / 1024
            print(f"  -> Thành công! Kích thước: {size_kb:.1f} KB")
            success_count += 1
        else:
            print(f"  -> THẤT BẠI: {paper['filename']}")
            
    print(f"\n=== HOÀN THÀNH: {success_count}/{len(PAPERS)} TÀI LIỆU ĐÃ TẢI XONG ===")

if __name__ == "__main__":
    main()
