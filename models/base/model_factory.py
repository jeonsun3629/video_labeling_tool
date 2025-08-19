"""
모델 팩토리 패턴 구현
"""
import importlib
from typing import Dict, Any, Optional
from config.settings import AVAILABLE_DETECTORS, AVAILABLE_CLASSIFIERS
from .base_detector import BaseDetector
from .base_classifier import BaseClassifier

class ModelFactory:
    """모델 생성을 위한 팩토리 클래스"""
    
    _detector_cache = {}
    _classifier_cache = {}
    
    @classmethod
    def create_detector(cls, detector_type: str, **kwargs) -> Optional[BaseDetector]:
        """탐지기 생성
        
        Args:
            detector_type: 탐지기 타입 ('yolo', 'clip' 등)
            **kwargs: 모델별 설정 파라미터
            
        Returns:
            BaseDetector 인스턴스 또는 None
        """
        if detector_type not in AVAILABLE_DETECTORS:
            raise ValueError(f"Unknown detector type: {detector_type}. Available: {list(AVAILABLE_DETECTORS.keys())}")
        
        # 캐시 확인 (해시 가능한 값만 사용)
        try:
            # 리스트나 딕셔너리 등 해시 불가능한 값들을 문자열로 변환
            hashable_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (list, dict)):
                    hashable_kwargs[k] = str(v)
                else:
                    hashable_kwargs[k] = v
            cache_key = f"{detector_type}_{hash(frozenset(hashable_kwargs.items()))}"
        except Exception:
            # 해시 생성 실패시 캐시 사용 안함
            cache_key = f"{detector_type}_no_cache"
        
        if cache_key in cls._detector_cache and not cache_key.endswith('_no_cache'):
            return cls._detector_cache[cache_key]
        
        try:
            # 동적 모듈 로드
            module_path, class_name = AVAILABLE_DETECTORS[detector_type].rsplit('.', 1)
            module = importlib.import_module(module_path)
            detector_class = getattr(module, class_name)
            
            # 인스턴스 생성
            detector = detector_class(**kwargs)
            
            # 캐시에 저장
            cls._detector_cache[cache_key] = detector
            
            return detector
            
        except Exception as e:
            print(f"❌ Failed to create detector {detector_type}: {e}")
            return None
    
    @classmethod
    def create_classifier(cls, classifier_type: str, **kwargs) -> Optional[BaseClassifier]:
        """분류기 생성
        
        Args:
            classifier_type: 분류기 타입 ('dinov2', 'clip', 'universal' 등)
            **kwargs: 모델별 설정 파라미터
            
        Returns:
            BaseClassifier 인스턴스 또는 None
        """
        if classifier_type not in AVAILABLE_CLASSIFIERS:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Available: {list(AVAILABLE_CLASSIFIERS.keys())}")
        
        # 캐시 확인 (해시 가능한 값만 사용)
        try:
            # 리스트나 딕셔너리 등 해시 불가능한 값들을 문자열로 변환
            hashable_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (list, dict)):
                    hashable_kwargs[k] = str(v)
                else:
                    hashable_kwargs[k] = v
            cache_key = f"{classifier_type}_{hash(frozenset(hashable_kwargs.items()))}"
        except Exception:
            # 해시 생성 실패시 캐시 사용 안함
            cache_key = f"{classifier_type}_no_cache"
        
        if cache_key in cls._classifier_cache and not cache_key.endswith('_no_cache'):
            return cls._classifier_cache[cache_key]
        
        try:
            # 동적 모듈 로드
            module_path, class_name = AVAILABLE_CLASSIFIERS[classifier_type].rsplit('.', 1)
            module = importlib.import_module(module_path)
            classifier_class = getattr(module, class_name)
            
            # 인스턴스 생성
            classifier = classifier_class(**kwargs)
            
            # 캐시에 저장
            cls._classifier_cache[cache_key] = classifier
            
            return classifier
            
        except Exception as e:
            print(f"❌ Failed to create classifier {classifier_type}: {e}")
            return None
    
    @classmethod
    def get_available_detectors(cls) -> Dict[str, str]:
        """사용 가능한 탐지기 목록 반환"""
        return AVAILABLE_DETECTORS.copy()
    
    @classmethod
    def get_available_classifiers(cls) -> Dict[str, str]:
        """사용 가능한 분류기 목록 반환"""
        return AVAILABLE_CLASSIFIERS.copy()
    
    @classmethod
    def clear_cache(cls) -> None:
        """캐시 초기화"""
        cls._detector_cache.clear()
        cls._classifier_cache.clear()
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """캐시 정보 반환"""
        return {
            'detector_cache_size': len(cls._detector_cache),
            'classifier_cache_size': len(cls._classifier_cache),
            'cached_detectors': list(cls._detector_cache.keys()),
            'cached_classifiers': list(cls._classifier_cache.keys())
        }
