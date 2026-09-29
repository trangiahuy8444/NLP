#!/usr/bin/env python3
"""
Ứng Dụng Web Trực Quan Phân Tích Cảm Xúc Đa Miền (Sentiment Analysis Web App)
Khởi chạy máy chủ HTTP REST API tích hợp giao diện tương tác người dùng thời gian thực:
- 2 Miền dữ liệu: UIT-VSFC (Giáo dục) & E-Commerce (Thương mại điện tử)
- 6 Mô hình: TF-IDF, Avg Word2Vec, BiLSTM Scratch, BiLSTM Pretrained, BiLSTM + Self-Attention, PhoBERT Base v2
"""

import os
import sys
import json
import argparse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Đảm bảo import được module từ thư mục src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
WEB_DIR = os.path.join(BASE_DIR, "web_app")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from inference_engine import SentimentInferenceEngine, MODEL_DISPLAY_NAMES, MODEL_METRICS_INFO

# Khởi tạo singleton inference engine toàn cục
engine = None

class SentimentAppRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Giảm bớt log thừa khi request static
        if "GET / " in args[0] or "POST /api/" in args[0]:
            sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]}\n")

    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html_path = os.path.join(WEB_DIR, "index.html")
            if os.path.exists(html_path):
                self._set_headers(content_type="text/html; charset=utf-8")
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(status=404)
                self.wfile.write(b"<h1>404 Not Found</h1>")
            return

        elif path.startswith("/assets/"):
            rel_path = path.lstrip("/")
            file_path = os.path.join(WEB_DIR, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                content_type = "application/octet-stream"
                if file_path.endswith(".png"):
                    content_type = "image/png"
                elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                    content_type = "image/jpeg"
                elif file_path.endswith(".svg"):
                    content_type = "image/svg+xml"
                elif file_path.endswith(".css"):
                    content_type = "text/css"
                elif file_path.endswith(".js"):
                    content_type = "application/javascript"
                self._set_headers(content_type=content_type)
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(status=404)
                self.wfile.write(b"404 Not Found")
            return

        elif path == "/api/status":
            self._set_headers()
            status_data = {
                "status": "online",
                "device": str(engine.device),
                "models_available": list(MODEL_DISPLAY_NAMES.keys()),
                "datasets": ["UIT-VSFC", "E-Commerce"]
            }
            self.wfile.write(json.dumps(status_data, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/metrics":
            self._set_headers()
            self.wfile.write(json.dumps(MODEL_METRICS_INFO, ensure_ascii=False).encode("utf-8"))
            return

        else:
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": "Endpoint không tồn tại"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/benchmark":
            dataset = payload.get("dataset", "UIT-VSFC")
            text = payload.get("text", "").strip()

            if not text:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": "Vui lòng cung cấp chuỗi văn bản 'text'"}, ensure_ascii=False).encode("utf-8"))
                return

            try:
                benchmark_results = engine.predict_all(dataset, text)
                self._set_headers()
                self.wfile.write(json.dumps(benchmark_results, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._set_headers(status=500)
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/predict":
            dataset = payload.get("dataset", "UIT-VSFC")
            model_id = payload.get("model_id", "bilstm_attention")
            text = payload.get("text", "").strip()

            if not text:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": "Vui lòng cung cấp chuỗi văn bản 'text'"}, ensure_ascii=False).encode("utf-8"))
                return

            try:
                res = engine.predict_single(dataset, model_id, text)
                self._set_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._set_headers(status=500)
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))

        else:
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": "Endpoint không tồn tại"}).encode("utf-8"))


def main():
    global engine
    parser = argparse.ArgumentParser(description="Chạy ứng dụng Web phân tích cảm xúc tiếng Việt đa miền")
    parser.add_argument("--port", type=int, default=8501, help="Cổng chạy web server (Mặc định: 8501)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Địa chỉ host (Mặc định: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG SẢN PHẨM PHÂN TÍCH CẢM XÚC ĐA MIỀN (VIETNAMESE NLP DEMO)")
    print(f"{'='*75}")

    # Khởi tạo engine
    models_dir = os.path.join(BASE_DIR, "models")
    engine = SentimentInferenceEngine(models_dir=models_dir)

    port = int(os.environ.get("PORT", os.environ.get("SPACE_PORT", args.port)))
    host = os.environ.get("HOST", args.host)
    server_address = (host, port)
    httpd = HTTPServer(server_address, SentimentAppRequestHandler)

    url = f"http://{host}:{port}"
    print(f"\n✅ Ứng dụng đã sẵn sàng phục vụ tại: {url}")
    print(f"👉 Mở trình duyệt web và truy cập địa chỉ trên để trải nghiệm!")
    print(f"Nhấn Ctrl + C để dừng máy chủ.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Đang dừng máy chủ...")
        httpd.server_close()
        print("Đã dừng an toàn. Tạm biệt!")


if __name__ == "__main__":
    main()
