"""
모델 관리 서비스
"""
from typing import Dict, Any, Optional, List
from models.base.model_factory import ModelFactory
from models.base.base_detector import BaseDetector
from models.base.base_classifier import BaseClassifier
from config.settings import DEFAULT_DETECTOR, DEFAULT_CLASSIFIER

class ModelManager:
    """모델 생성, 로드, 관리를 담당하는 서비스"""
    
    def __init__(self):
        self.current_detector: Optional[BaseDetector] = None
        self.current_classifier: Optional[BaseClassifier] = None
        self.detector_type: str = DEFAULT_DETECTOR
        self.classifier_type: str = DEFAULT_CLASSIFIER
        
    def initialize_models(self, detector_type: Optional[str] = None, classifier_type: Optional[str] = None) -> Dict[str, Any]:
        """모델 초기화"""
        result = {
            'detector_loaded': False,
            'classifier_loaded': False,
            'errors': []
        }
        
        # 탐지기 초기화
        detector_type = detector_type or self.detector_type
        try:
            self.current_detector = ModelFactory.create_detector(detector_type)
            if self.current_detector:
                result['detector_loaded'] = self.current_detector.load_model()
                self.detector_type = detector_type
            else:
                result['errors'].append(f"Failed to create detector: {detector_type}")
        except Exception as e:
            result['errors'].append(f"Detector initialization error: {str(e)}")
        
        # 분류기 초기화
        classifier_type = classifier_type or self.classifier_type
        try:
            self.current_classifier = ModelFactory.create_classifier(classifier_type)
            if self.current_classifier:
                result['classifier_loaded'] = self.current_classifier.load_model()
                self.classifier_type = classifier_type
                
                # Universal classifier에 특징 추출기 설정
                if hasattr(self.current_classifier, 'set_feature_extractor'):
                    # DINOv2 분류기를 특징 추출기로 사용
                    dinov2_classifier = ModelFactory.create_classifier('dinov2')
                    if dinov2_classifier and dinov2_classifier.load_model():
                        self.current_classifier.set_feature_extractor(dinov2_classifier)
            else:
                result['errors'].append(f"Failed to create classifier: {classifier_type}")
        except Exception as e:
            result['errors'].append(f"Classifier initialization error: {str(e)}")
        
        return result
    
    def switch_detector(self, detector_type: str, **kwargs) -> bool:
        """탐지기 변경"""
        try:
            new_detector = ModelFactory.create_detector(detector_type, **kwargs)
            if new_detector and new_detector.load_model():
                self.current_detector = new_detector
                self.detector_type = detector_type
                print(f"✅ Switched to detector: {detector_type}")
                return True
            else:
                print(f"❌ Failed to switch to detector: {detector_type}")
                return False
        except Exception as e:
            print(f"❌ Error switching detector: {e}")
            return False
    
    def switch_classifier(self, classifier_type: str, **kwargs) -> bool:
        """분류기 변경"""
        try:
            new_classifier = ModelFactory.create_classifier(classifier_type, **kwargs)
            if new_classifier and new_classifier.load_model():
                self.current_classifier = new_classifier
                self.classifier_type = classifier_type
                
                # Universal classifier에 특징 추출기 설정
                if hasattr(self.current_classifier, 'set_feature_extractor'):
                    dinov2_classifier = ModelFactory.create_classifier('dinov2')
                    if dinov2_classifier and dinov2_classifier.load_model():
                        self.current_classifier.set_feature_extractor(dinov2_classifier)
                
                print(f"✅ Switched to classifier: {classifier_type}")
                return True
            else:
                print(f"❌ Failed to switch to classifier: {classifier_type}")
                return False
        except Exception as e:
            print(f"❌ Error switching classifier: {e}")
            return False
    
    def get_detector(self) -> Optional[BaseDetector]:
        """현재 탐지기 반환"""
        return self.current_detector
    
    def get_classifier(self) -> Optional[BaseClassifier]:
        """현재 분류기 반환"""
        return self.current_classifier
    
    def get_available_models(self) -> Dict[str, Any]:
        """사용 가능한 모델 목록 반환"""
        return {
            'detectors': ModelFactory.get_available_detectors(),
            'classifiers': ModelFactory.get_available_classifiers(),
            'current_detector': self.detector_type,
            'current_classifier': self.classifier_type
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 반환"""
        status = {
            'detector': {
                'type': self.detector_type,
                'loaded': self.current_detector is not None and self.current_detector.is_loaded(),
                'info': self.current_detector.get_model_info() if self.current_detector else {}
            },
            'classifier': {
                'type': self.classifier_type,
                'loaded': self.current_classifier is not None and self.current_classifier.is_loaded(),
                'info': self.current_classifier.get_model_info() if self.current_classifier else {}
            }
        }
        return status
    
    def load_custom_detector_model(self, model_path: str) -> bool:
        """커스텀 탐지기 모델 로드"""
        try:
            if self.current_detector:
                # 하이브리드 탐지기인 경우
                if hasattr(self.current_detector, 'yolo_detector'):
                    # YOLO + CLIP 또는 YOLO + DINOv2 하이브리드
                    if self.current_detector.yolo_detector:
                        self.current_detector.yolo_detector.model_path = model_path
                        success = self.current_detector.yolo_detector.load_model()
                        print(f"🔄 Custom model loaded in hybrid detector: {success}")
                        return success
                # 단순 YOLO 탐지기인 경우
                elif hasattr(self.current_detector, 'model_path'):
                    self.current_detector.model_path = model_path
                    success = self.current_detector.load_model()
                    print(f"🔄 Custom model loaded in simple detector: {success}")
                    return success
                else:
                    print(f"⚠️ Detector type {type(self.current_detector)} does not support custom model loading")
            return False
        except Exception as e:
            print(f"❌ Error loading custom detector model: {e}")
            return False
    
    def set_detector_confidence(self, threshold: float) -> bool:
        """탐지기 신뢰도 임계값 설정"""
        try:
            if self.current_detector:
                self.current_detector.set_confidence_threshold(threshold)
                return True
            return False
        except Exception as e:
            print(f"❌ Error setting detector confidence: {e}")
            return False
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.current_detector = None
        self.current_classifier = None
        ModelFactory.clear_cache()
