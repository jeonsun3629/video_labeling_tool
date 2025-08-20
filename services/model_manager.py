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
            # YOLO+CLIP 모델의 경우 UI 설정 처리
            if detector_type == 'yolo_clip' and 'defect_queries' in kwargs:
                print(f"🔧 YOLO+CLIP config received:")
                print(f"   Queries: {len(kwargs['defect_queries'])}")
                print(f"   Threshold: {kwargs.get('defect_threshold', 'default')}")
                
                # 기존 커스텀 모델 경로 보존
                if (self.current_detector and 
                    hasattr(self.current_detector, 'yolo_detector') and 
                    self.current_detector.yolo_detector and
                    hasattr(self.current_detector.yolo_detector, 'is_custom_model') and
                    self.current_detector.yolo_detector.is_custom_model):
                    kwargs['model_path'] = self.current_detector.yolo_detector.model_path
                    print(f"🎯 Preserving custom model: {kwargs['model_path']}")
            
            new_detector = ModelFactory.create_detector(detector_type, **kwargs)
            if new_detector and new_detector.load_model():
                # YOLO+CLIP의 경우 UI 설정 적용
                if detector_type == 'yolo_clip' and hasattr(new_detector, 'update_from_ui_config'):
                    defect_queries = kwargs.get('defect_queries', [])
                    defect_threshold = kwargs.get('defect_threshold', 0.35)
                    new_detector.update_from_ui_config(defect_queries, defect_threshold)
                
                self.current_detector = new_detector
                self.detector_type = detector_type
                print(f"✅ Detector: {detector_type}")
                return True
            else:
                print(f"❌ Failed: {detector_type}")
                return False
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
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
                
                print(f"✅ Classifier: {classifier_type}")
                return True
            else:
                print(f"❌ Failed: {classifier_type}")
                return False
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
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
                print(f"🔄 Loading custom model: {model_path}")
                print(f"🔍 Current detector type: {type(self.current_detector).__name__}")
                
                # 하이브리드 탐지기인 경우 - 새로 생성해서 커스텀 모델 경로 전달
                if hasattr(self.current_detector, 'yolo_detector'):
                    print("🔧 Detected hybrid detector, recreating with custom model...")
                    
                    # 기존 설정 보존
                    existing_kwargs = {}
                    if hasattr(self.current_detector, 'custom_queries'):
                        existing_kwargs['defect_queries'] = self.current_detector.custom_queries
                    if hasattr(self.current_detector, 'defect_threshold'):
                        existing_kwargs['defect_threshold'] = self.current_detector.defect_threshold
                    if hasattr(self.current_detector, 'query_labels'):
                        existing_kwargs['query_labels'] = self.current_detector.query_labels
                    
                    # 커스텀 모델 경로 추가
                    existing_kwargs['model_path'] = model_path
                    
                    # 새 하이브리드 탐지기 생성
                    new_detector = ModelFactory.create_detector(self.detector_type, **existing_kwargs)
                    if new_detector and new_detector.load_model():
                        self.current_detector = new_detector
                        print(f"✅ Hybrid detector recreated with custom model")
                        return True
                    else:
                        print(f"❌ Failed to recreate hybrid detector")
                        return False
                
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
