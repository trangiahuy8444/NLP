#!/usr/bin/env python3
"""
Script Hỗ Trợ Tải Dự Án Lên Hugging Face Spaces Tự Động (1-Click Deploy)
Sử dụng thư viện chính thức huggingface_hub để upload toàn bộ mã nguồn,
weights mô hình (bao gồm cả PhoBERT 540MB) và giao diện trực tiếp lên Spaces.
"""

import os
import sys
import argparse

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("❌ Vui lòng cài đặt thư viện huggingface_hub:")
    print("   pip install huggingface_hub")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Tải ứng dụng lên Hugging Face Spaces")
    parser.add_argument("--repo", type=str, help="Tên Space trên Hugging Face (Dạng: username/space-name)")
    parser.add_argument("--token", type=str, help="Hugging Face Access Token (Bắt đầu bằng hf_...)")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 HỆ THỐNG HỖ TRỢ DEPLOY LÊN HUGGING FACE SPACES - HCMUE NLP")
    print("=" * 70)

    # 1. Nhập Token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n👉 Để tải lên, bạn cần một Access Token từ Hugging Face (Quyền Write):")
        print("   1. Đăng nhập https://huggingface.co")
        print("   2. Vào https://huggingface.co/settings/tokens")
        print("   3. Bấm 'Create new token' -> Chọn loại 'Write' -> Copy mã token")
        token = input("\nNhập Access Token (hf_...): ").strip()

    if not token:
        print("❌ Lỗi: Bạn chưa cung cấp Access Token!")
        return

    # Xác thực đăng nhập
    api = HfApi(token=token)
    username = None
    try:
        login(token=token, add_to_git_credential=True)
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"✅ Xác thực thành công! Tài khoản của bạn là: \033[1;32m{username}\033[0m")
    except Exception as e:
        print(f"❌ Xác thực thất bại: {e}")
        return

    # 2. Nhập Tên Repo Space
    repo_id = args.repo
    if not repo_id:
        default_name = f"{username}/hcmue-sentiment-ai" if username else "hcmue-sentiment-ai"
        print(f"\n👉 Nhập tên Space đích (Nhấn Enter để dùng mặc định: \033[1;34m{default_name}\033[0m):")
        user_input = input(f"Repo Space ID [{default_name}]: ").strip()
        repo_id = user_input if user_input else default_name

    # Nếu người dùng chỉ gõ tên ngắn như 'NLP' hoặc 'hcmue-sentiment-ai' thì tự động thêm username/
    if "/" not in repo_id and username:
        repo_id = f"{username}/{repo_id}"

    print(f"🎯 Đích tải lên: \033[1;36m{repo_id}\033[0m")

    # 3. Tạo Space nếu chưa tồn tại
    api = HfApi()
    try:
        api.space_info(repo_id=repo_id)
        print(f"🔹 Đã tìm thấy Space: {repo_id}")
    except Exception:
        print(f"🔹 Space '{repo_id}' chưa tồn tại, đang tự động khởi tạo với SDK Docker...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                private=False
            )
            print(f"✅ Đã tạo thành công Space: {repo_id}")
        except Exception as err:
            print(f"❌ Không thể tạo Space: {err}")
            return

    # 4. Tải toàn bộ thư mục lên Space
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📦 Đang tải toàn bộ dự án từ: {current_dir}")
    print(f"   (Bao gồm mã nguồn, giao diện HCMUE, weights PhoBERT và BiLSTM)")
    print("⏳ Quá trình upload file model lớn (~1GB) có thể mất vài phút tuỳ tốc độ mạng...")

    try:
        api.upload_folder(
            folder_path=current_dir,
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["*.DS_Store", "*__pycache__*", "*.zip", "upload_to_huggingface.py"]
        )
        print("\n" + "=" * 70)
        print("🎉 TẢI LÊN HUGGING FACE SPACES THÀNH CÔNG RỰC RỠ!")
        print("=" * 70)
        live_url = f"https://huggingface.co/spaces/{repo_id}"
        print(f"\n🌐 Đường link ứng dụng của bạn:")
        print(f"👉 \033[1;32m{live_url}\033[0m")
        print("\n💡 Lưu ý: Hugging Face sẽ mất khoảng 2 - 3 phút để Build Docker và khởi động.")
        print("   Sau khi build xong, bạn mở link trên là có thể trải nghiệm trực tuyến ngay!")
    except Exception as e:
        print(f"❌ Lỗi trong quá trình upload: {e}")

if __name__ == "__main__":
    main()
