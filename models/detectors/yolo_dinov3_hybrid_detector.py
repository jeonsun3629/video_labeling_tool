"""
YOLO + DINOv3 하이브리드 탐지기
"""
import numpy as np
from typing import List, Dict, Any, Optional
from ..base.base_detector import BaseDetector
from .yolo_detector import YOLODetector
from ..classifiers.dinov3_classifier import DINOv3Classifier
from services.model_manager import ModelManager

class YOLODINOv3HybridDetector(BaseDetector):
    """YOLO + DINOv3 하이브리드 탐지기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_detector = None
        self.dinov3_classifier = None
        
    def load_model(self) -> bool:
        """YOLO + DINOv3 하이브리드 모델 로드"""
        try:
            print("📥 Loading YOLO + DINOv3 hybrid detector...")
            
            # YOLO 탐지기 로드
            self.yolo_detector = YOLODetector()
            yolo_loaded = self.yolo_detector.load_model()
            
            if not yolo_loaded:
                print("❌ Failed to load YOLO detector")
                return False
            
            # DINOv3 분류기 로드
            self.dinov3_classifier = DINOv3Classifier()
            dinov3_loaded = self.dinov3_classifier.load_model()
            
            if not dinov3_loaded:
                print("❌ Failed to load DINOv3 classifier")
                return False
            
            print("✅ YOLO + DINOv3 hybrid detector loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YOLO + DINOv3 hybrid detector: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """하이브리드 탐지기 로드 상태 확인"""
        yolo_loaded = self.yolo_detector is not None and self.yolo_detector.is_loaded()
        dinov3_loaded = self.dinov3_classifier is not None and self.dinov3_classifier.is_loaded()
        return yolo_loaded and dinov3_loaded
    
    def detect(self, frame: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """YOLO + DINOv3 하이브리드 탐지"""
        if not self.is_loaded():
            print("❌ YOLO + DINOv3 hybrid detector not loaded")
            return []
        
        try:
            # 1. YOLO로 객체 탐지
            yolo_detections = self.yolo_detector.detect(frame, **kwargs)
            
            if not yolo_detections:
                return []
            
            print(f"🎯 YOLO detected {len(yolo_detections)} objects")
            
            # 2. 각 탐지된 객체에 대해 DINOv3로 세밀한 분석
            enhanced_detections = []
            
            for i, detection in enumerate(yolo_detections):
                try:
                    # 바운딩 박스 정보 추출
                    bbox = [
                        int(detection['x1']),
                        int(detection['y1']),
                        int(detection['x2']),
                        int(detection['y2'])
                    ]
                    
                    # DINOv3로 특징 추출
                    features = self.dinov3_classifier.extract_features(frame, bbox)
                    
                    if features is not None:
                        # 학습된 패턴과 비교하여 세밀한 분류
                        reference_patterns = self.dinov3_classifier.get_learned_patterns()
                        enhanced_class = self.dinov3_classifier.classify(features, reference_patterns)
                        
                        # 향상된 탐지 결과 생성
                        enhanced_detection = detection.copy()
                        enhanced_detection.update({
                            'enhanced_label': enhanced_class,
                            'original_class': detection['class_name'],
                            'dinov3_features': features.tolist() if isinstance(features, np.ndarray) else features,
                            'has_features': True,
                            'enhanced_by_dinov3': True,
                            'method': 'yolo_dinov3_hybrid'
                        })
                        
                        # 세밀한 분류가 이루어진 경우만 결과에 포함
                        if enhanced_class != detection['class_name']:
                            enhanced_detection['label'] = enhanced_class
                            enhanced_detection['class_name'] = enhanced_class
                            print(f"🧠 DINOv3 enhanced: {detection['class_name']} → {enhanced_class}")
                            enhanced_detections.append(enhanced_detection)
                        else:
                            # 기존 YOLO 클래스와 동일하면 필터링
                            print(f"✅ Basic {detection['class_name']} filtered out (no enhancement)")
                        
                    else:
                        # 특징 추출 실패 시 기존 클래스는 필터링
                        print(f"⚠️ Feature extraction failed for {detection['class_name']}, filtered out")
                    
                except Exception as e:
                    print(f"⚠️ Error analyzing detection {i}: {e}")
                    # 에러 시에도 필터링 (원본 탐지 결과 사용 안함)
                    continue
            
            print(f"🎯 Enhanced detections: {len(enhanced_detections)} objects")
            return enhanced_detections
            
        except Exception as e:
            print(f"❌ YOLO + DINOv3 hybrid detection failed: {e}")
            return []
    
    def learn_pattern(self, frame: np.ndarray, bbox: List[int], pattern_name: str) -> bool:
        """새로운 패턴 학습"""
        if not self.is_loaded():
            return False
        
        try:
            # DINOv3로 특징 추출
            features = self.dinov3_classifier.extract_features(frame, bbox)
            
            if features is not None:
                # 패턴 학습
                success = self.dinov3_classifier.learn_pattern(features, pattern_name)
                if success:
                    print(f"📚 Pattern learned: {pattern_name}")
                return success
            else:
                print(f"❌ Failed to extract features for pattern: {pattern_name}")
                return False
                
        except Exception as e:
            print(f"❌ Pattern learning failed: {e}")
            return False
    
    def get_learned_patterns_info(self) -> Dict[str, Any]:
        """학습된 패턴 정보 반환"""
        if not self.is_loaded():
            return {}
        
        try:
            patterns = self.dinov3_classifier.get_learned_patterns()
            return {
                'total_patterns': len(patterns),
                'pattern_names': list(patterns.keys()),
                'feature_dimension': len(list(patterns.values())[0]) if patterns else 0,
                'model_type': 'DINOv3'
            }
        except Exception as e:
            print(f"⚠️ Error getting patterns info: {e}")
            return {}
    
    def clear_learned_patterns(self) -> bool:
        """학습된 패턴 초기화"""
        if not self.is_loaded():
            return False
        
        try:
            self.dinov3_classifier.features_cache = {}
            print("🧹 All learned patterns cleared")
            return True
        except Exception as e:
            print(f"❌ Failed to clear patterns: {e}")
            return False
    
    def get_supported_classes(self) -> List[str]:
        """지원하는 클래스 목록 반환"""
        if self.yolo_detector and self.yolo_detector.is_loaded():
            return self.yolo_detector.get_supported_classes()
        return []
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.confidence_threshold = threshold
        if self.yolo_detector:
            self.yolo_detector.set_confidence_threshold(threshold)
