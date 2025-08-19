#!/usr/bin/env python3
"""
Video Labeling Tool - Backend Server Runner

이 스크립트는 비디오 라벨링 도구의 백엔드 서버를 실행합니다.
YOLOv8과 DINOv2 모델을 로드하여 자동 라벨링 기능을 제공합니다.
"""

import sys
import subprocess
import os

def check_dependencies():
    """필요한 패키지가 설치되어 있는지 확인"""
    # 패키지 이름과 실제 import 이름 매핑
    required_packages = {
        'flask': 'flask',
        'ultralytics': 'ultralytics', 
        'torch': 'torch',
        'opencv-python': 'cv2',  # opencv-python은 cv2로 import
        'transformers': 'transformers',
        'flask-cors': 'flask_cors'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치해주세요:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 모든 필수 패키지가 설치되어 있습니다.")
    return True

def main():
    print("=" * 60)
    print("🚀 Video Labeling Tool - AI Backend Server")
    print("=" * 60)
    
    # 종속성 확인
    if not check_dependencies():
        sys.exit(1)
    
    print("\n📋 시스템 정보:")
    try:
        import torch
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - CUDA 사용 가능: {'✅ Yes' if torch.cuda.is_available() else '❌ No (CPU 사용)'}")
        if torch.cuda.is_available():
            print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   - PyTorch: 설치되지 않음")
    
    print("\n🔧 서버 시작 중...")
    print("   - 주소: http://localhost:5000")
    print("   - 프론트엔드는 브라우저에서 http://localhost:5000 으로 접속하세요")
    print("   - 서버를 중지하려면 Ctrl+C를 눌러주세요")
    print("\n" + "=" * 60)
    
    # Flask 서버 실행
    try:
        from app_modular import app
        print("🤖 AI 모델 로딩 중...")
        print("✅ 서버 준비 완료!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 서버가 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 