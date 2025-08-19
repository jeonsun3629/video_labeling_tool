"""
YOLO + DINOv2 하이브리드 탐지기
1단계: YOLO로 객체 탐지 (위치 정보)
2단계: DINOv2로 각 객체의 특징 분석 및 분류
"""
import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from ..base.base_detector import BaseDetector
from .yolo_detector import YOLODetector
from ..classifiers.dinov2_classifier import DINOv2Classifier
from ..classifiers.universal_classifier import UniversalClassifier
from config.settings import YOLO_MODEL_PATH, DINOV2_SIMILARITY_THRESHOLD

class YOLODINOv2HybridDetector(BaseDetector):
    """YOLO + DINOv2 하이브리드 탐지기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_detector = None
        self.dinov2_classifier = None
        self.universal_classifier = None
        self.similarity_threshold = kwargs.get('similarity_threshold', DINOV2_SIMILARITY_THRESHOLD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """YOLO와 DINOv2 모델 모두 로드"""
        try:
            print("📥 Loading YOLO + DINOv2 hybrid detector...")
            
            # 1. YOLO 모델 로드
            self.yolo_detector = YOLODetector()
            yolo_success = self.yolo_detector.load_model()
            
            if not yolo_success:
                print("❌ Failed to load YOLO model")
                return False
            
            # 2. DINOv2 분류기 로드
            self.dinov2_classifier = DINOv2Classifier()
            dinov2_success = self.dinov2_classifier.load_model()
            
            if not dinov2_success:
                print("❌ Failed to load DINOv2 model")
                return False
            
            # 3. Universal 분류기 로드 (패턴 학습용)
            self.universal_classifier = UniversalClassifier()
            self.universal_classifier.load_model()
            self.universal_classifier.set_feature_extractor(self.dinov2_classifier)
            
            print("✅ YOLO + DINOv2 hybrid detector loaded successfully!")
            return True
                
        except Exception as e:
            print(f"❌ Failed to load hybrid detector: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLO + DINOv2 하이브리드 탐지"""
        if not self.is_loaded():
            print("ERROR: Hybrid detector not loaded!")
            return []
        
        try:
            # 1단계: YOLO로 객체 탐지
            yolo_detections = self.yolo_detector.detect(frame)
            
            if not yolo_detections:
                print("No objects detected by YOLO")
                return []
            
            print(f"🎯 YOLO detected {len(yolo_detections)} objects")
            
            # 2단계: 각 탐지된 객체를 DINOv2로 분석
            enhanced_detections = []
            
            for i, detection in enumerate(yolo_detections):
                try:
                    # 객체 영역에서 특징 추출
                    bbox = detection['bbox']
                    features = self.dinov2_classifier.extract_features(frame, bbox)
                    
                    if features is not None:
                        # Universal 분류기로 세밀한 분류 수행
                        enhanced_class = self.universal_classifier.classify_features(
                            features, detection['class_name']
                        )
                        
                        # 기존 탐지 정보에 DINOv2 분석 결과 추가
                        enhanced_detection = detection.copy()
                        enhanced_detection.update({
                            'enhanced_label': enhanced_class,
                            'original_class': detection['class_name'],
                            'has_features': True,
                            'enhanced_by_dinov2': True,
                            'method': 'yolo_dinov2_hybrid'
                        })
                        
                        # 세밀한 분류가 이루어진 경우만 결과에 포함
                        if enhanced_class != detection['class_name']:
                            enhanced_detection['label'] = enhanced_class
                            enhanced_detection['class_name'] = enhanced_class
                            print(f"🧠 DINOv2 enhanced: {detection['class_name']} → {enhanced_class}")
                            enhanced_detections.append(enhanced_detection)
                        else:
                            # 기존 YOLO 클래스와 동일하면 필터링
                            print(f"✅ Basic {detection['class_name']} filtered out (no enhancement)")
                        
                    else:
                        # 특징 추출 실패 시 기존 클래스는 필터링
                        print(f"⚠️ Feature extraction failed for {detection['class_name']}, filtered out")
                    
                except Exception as e:
                    print(f"⚠️ Error analyzing detection {i}: {e}")
                    # 에러 시 원본 탐지 결과 유지
                    enhanced_detections.append(detection)
            
            print(f"🧠 DINOv2 analysis completed for {len(enhanced_detections)} objects")
            return enhanced_detections
            
        except Exception as e:
            print(f"❌ Hybrid detection error: {e}")
            return []
    
    def get_supported_classes(self) -> List[str]:
        """지원하는 클래스 목록 반환"""
        if self.yolo_detector:
            return self.yolo_detector.get_supported_classes()
        return []
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """YOLO 신뢰도 임계값 설정"""
        if self.yolo_detector:
            self.yolo_detector.set_confidence_threshold(threshold)
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """DINOv2 유사도 임계값 설정"""
        self.similarity_threshold = threshold
        if self.dinov2_classifier:
            self.dinov2_classifier.similarity_threshold = threshold
        if self.universal_classifier:
            self.universal_classifier.similarity_threshold = threshold
    
    def learn_patterns(self, video_path: str, annotations: List[Dict[str, Any]]) -> bool:
        """수동 라벨링 데이터에서 패턴 학습"""
        if self.universal_classifier:
            return self.universal_classifier.learn_from_annotations(video_path, annotations)
        return False
    
    def get_learned_patterns_info(self) -> Dict[str, Any]:
        """학습된 패턴 정보 반환"""
        if self.universal_classifier:
            return self.universal_classifier.get_learned_patterns_info()
        return {}
    
    def is_loaded(self) -> bool:
        """하이브리드 탐지기 로드 상태 확인"""
        yolo_loaded = self.yolo_detector is not None and self.yolo_detector.is_loaded()
        dinov2_loaded = self.dinov2_classifier is not None and self.dinov2_classifier.is_loaded()
        return yolo_loaded and dinov2_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """하이브리드 모델 정보 반환"""
        base_info = super().get_model_info()
        base_info.update({
            'hybrid_type': 'yolo_dinov2',
            'yolo_loaded': self.yolo_detector is not None and self.yolo_detector.is_loaded(),
            'dinov2_loaded': self.dinov2_classifier is not None and self.dinov2_classifier.is_loaded(),
            'universal_loaded': self.universal_classifier is not None,
            'similarity_threshold': self.similarity_threshold,
            'supports_pattern_learning': True
        })
        return base_info
