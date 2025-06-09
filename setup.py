#!/usr/bin/env python3
"""
Setup script for Text2Cypher RAG Demo
Kiểm tra và cài đặt các dependencies cần thiết
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """Chạy command và hiển thị kết quả"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Thành công")
            return True
        else:
            print(f"❌ {description} - Lỗi: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False

def check_ollama():
    """Kiểm tra Ollama đã được cài đặt chưa"""
    print("\n📋 Kiểm tra Ollama...")
    return run_command("ollama --version", "Kiểm tra Ollama đã cài đặt")

def check_ollama_running():
    """Kiểm tra Ollama có đang chạy không"""
    print("\n📋 Kiểm tra Ollama đang chạy...")
    return run_command("ollama list", "Kiểm tra Ollama service")

def install_model():
    """Cài đặt mô hình llama3.2"""
    print("\n📋 Cài đặt mô hình llama3.2...")
    return run_command("ollama pull llama3.2", "Cài đặt llama3.2 model")

def install_python_deps():
    """Cài đặt Python dependencies"""
    print("\n📋 Cài đặt Python dependencies...")
    return run_command("pip install -r requirements.txt", "Cài đặt Python packages")

def main():
    print("🚀 Text2Cypher RAG Demo Setup")
    print("=" * 40)
    
    # Kiểm tra file requirements.txt
    if not os.path.exists("requirements.txt"):
        print("❌ Không tìm thấy requirements.txt")
        sys.exit(1)
    
    # Kiểm tra schema.txt
    if not os.path.exists("schema.txt"):
        print("❌ Không tìm thấy schema.txt")
        sys.exit(1)
    
    success = True
    
    # 1. Kiểm tra Ollama
    if not check_ollama():
        print("\n🔧 Hướng dẫn cài đặt Ollama:")
        print("   Windows: winget install Ollama.Ollama")
        print("   Hoặc tải từ: https://ollama.ai/download")
        success = False
    
    # 2. Cài đặt Python dependencies
    if not install_python_deps():
        success = False
    
    # 3. Kiểm tra Ollama đang chạy
    if not check_ollama_running():
        print("\n🔧 Khởi động Ollama:")
        print("   Chạy lệnh: ollama serve")
        print("   Sau đó chạy lại script này")
        success = False
    else:
        # 4. Cài đặt mô hình llama3.2
        if not install_model():
            print("\n🔧 Cài đặt mô hình thủ công:")
            print("   ollama pull llama3.2")
            success = False
    
    print("\n" + "=" * 40)
    
    if success:
        print("✅ Setup hoàn tất! Bạn có thể chạy:")
        print("   Streamlit: streamlit run text2cypher_demo.py")
        print("   Console: python console_demo.py")
    else:
        print("❌ Setup chưa hoàn tất. Vui lòng xem hướng dẫn ở trên.")
        sys.exit(1)

if __name__ == "__main__":
    main() 