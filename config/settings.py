"""
설정 관리 모듈
"""
import os
from pathlib import Path

# 기본 설정
BASE_DIR = Path(__file__).parent.parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
TRAINING_FOLDER = BASE_DIR / 'custom_training'

# 모델 설정
AVAILABLE_DETECTORS = {
    'yolo_dinov2': 'models.detectors.yolo_dinov2_hybrid_detector.YOLODINOv2HybridDetector',  # YOLO + DINOv2 하이브리드
    'yolo_clip': 'models.detectors.yolo_clip_hybrid_detector.YOLOCLIPHybridDetector',  # YOLO + CLIP 하이브리드
}

AVAILABLE_CLASSIFIERS = {
    'dinov2': 'models.classifiers.dinov2_classifier.DINOv2Classifier',  # DINOv2 분류기
    'clip': 'models.classifiers.clip_classifier.CLIPClassifier',
    'universal': 'models.classifiers.universal_classifier.UniversalClassifier',
}

# 기본 모델 설정
DEFAULT_DETECTOR = 'yolo_dinov2'  # YOLO + DINOv2 하이브리드를 기본으로
DEFAULT_CLASSIFIER = 'dinov2'

# YOLOv11 설정 (업그레이드)
YOLO_MODEL_PATH = 'yolo11n.pt'  # YOLOv11 nano 모델
YOLO_CONFIDENCE_THRESHOLD = 0.05  # 더 많은 객체를 탐지하도록 낮춘 임계값

# DINOv2 설정 (실제 DINOv2 모델 사용)
DINOV2_MODEL_NAME = 'facebook/dinov2-base'  # DINOv2 ViT-Base 모델
DINOV2_SIMILARITY_THRESHOLD = 0.7

# CLIP 설정 (미래 확장용)
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CLIP_DEFECT_THRESHOLD = 0.35  # 더 현실적인 임계값으로 조정

# 메모리 관리 설정
MEMORY_LIMIT_RATIO = 0.85
DINOV2_THRESHOLD_RATIO = 0.75
CROP_BATCH_SIZE = 6

# 비디오 처리 설정
MAX_VIDEO_WIDTH = 1024
DEFAULT_FRAME_INTERVAL = 1
DENSE_ANALYSIS_FPS_RATIO = 4

# 훈련 설정
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMAGE_SIZE = 640

# Flask 설정
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# 디렉터리 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)
