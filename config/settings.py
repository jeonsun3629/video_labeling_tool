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
    'yolo_dinov3': 'models.detectors.yolo_dinov3_hybrid_detector.YOLODINOv3HybridDetector',  # YOLO + DINOv3 하이브리드
    'yolo_clip': 'models.detectors.yolo_clip_hybrid_detector.YOLOCLIPHybridDetector',  # YOLO + CLIP 하이브리드
    # 레거시 호환성
    'yolo_dinov2': 'models.detectors.yolo_dinov3_hybrid_detector.YOLODINOv3HybridDetector'  # DINOv3로 리디렉션
}

AVAILABLE_CLASSIFIERS = {
    'dinov3': 'models.classifiers.dinov3_classifier.DINOv3Classifier',  # 새로운 DINOv3 분류기
    'clip': 'models.classifiers.clip_classifier.CLIPClassifier',
    'universal': 'models.classifiers.universal_classifier.UniversalClassifier',
    # 레거시 호환성
    'dinov2': 'models.classifiers.dinov3_classifier.DINOv3Classifier'  # DINOv3로 리디렉션
}

# 기본 모델 설정
DEFAULT_DETECTOR = 'yolo_dinov3'  # YOLO + DINOv3 하이브리드를 기본으로
DEFAULT_CLASSIFIER = 'dinov3'

# YOLOv11 설정 (업그레이드)
YOLO_MODEL_PATH = 'yolo11n.pt'  # YOLOv11 nano 모델
YOLO_CONFIDENCE_THRESHOLD = 0.15

# DINOv3 설정 (실제 DINOv3 모델 사용)
DINOV3_MODEL_NAME = 'dinov3_vitb16'  # DINOv3 ViT-Base 모델
DINOV3_SIMILARITY_THRESHOLD = 0.7

# 레거시 호환성을 위한 별칭
DINOV2_MODEL_NAME = DINOV3_MODEL_NAME
DINOV2_SIMILARITY_THRESHOLD = DINOV3_SIMILARITY_THRESHOLD

# CLIP 설정 (미래 확장용)
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CLIP_DEFECT_THRESHOLD = 0.8

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
