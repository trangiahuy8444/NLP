#!/usr/bin/env python3
"""
Script khởi chạy Web Interactive Studio cho Đồ Án Giữa Kỳ NLP
Tự động tìm cổng khả dụng, bật HTTP Server cục bộ và mở trình duyệt mặc định.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    global PORT
    for attempt in range(10):
        try:
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                url = f"http://localhost:{PORT}/index.html"
                print("=" * 70)
                print(f"🚀 NLP MIDTERM INTERACTIVE STUDIO ĐÃ KHỞI CHẠY THÀNH CÔNG!")
                print(f"🌐 Truy cập tại: {url}")
                print(f"📁 Thư mục phục vụ: {DIRECTORY}")
                print("=" * 70)
                print("💡 Nhấn Ctrl+C để dừng máy chủ web khi học xong.")
                
                # Mở trình duyệt tự động
                try:
                    webbrowser.open(url)
                except Exception as e:
                    print(f"Lưu ý: Không thể mở trình duyệt tự động ({e}), vui lòng bấm vào link trên.")
                
                httpd.serve_forever()
        except OSError:
            print(f"⚠️ Cổng {PORT} đang bận, thử cổng {PORT + 1}...")
            PORT += 1

if __name__ == "__main__":
    start_server()
