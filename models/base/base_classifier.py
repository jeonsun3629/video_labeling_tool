"""
추상 분류기 베이스 클래스
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class BaseClassifier(ABC):
    """모든 분류기의 베이스 클래스"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.processor = None
        self.device = None
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        
    @abstractmethod
    def load_model(self) -> bool:
        """모델 로드"""
        pass
    
    @abstractmethod
    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """이미지 크롭에서 특징 추출
        
        Args:
            frame: 입력 프레임
            bbox: 바운딩 박스 [x, y, w, h]
            
        Returns:
            특징 벡터 (numpy array) 또는 None
        """
        pass
    
    @abstractmethod
    def classify_features(self, features: np.ndarray, base_class: str) -> str:
        """특징을 기반으로 분류
        
        Args:
            features: 특징 벡터
            base_class: 기본 클래스 이름
            
        Returns:
            분류된 클래스 이름
        """
        pass
    
    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.model is not None
    
    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """크롭 이미지 전처리 (기본 구현)"""
        return crop
    
    def learn_from_annotations(self, video_path: str, annotations: List[Dict[str, Any]]) -> bool:
        """수동 어노테이션에서 학습 (선택적 구현)"""
        return False
    
    def get_learned_patterns_info(self) -> Dict[str, Any]:
        """학습된 패턴 정보 반환 (선택적 구현)"""
        return {}
    
    def save_patterns(self, filepath: str) -> bool:
        """패턴 저장 (선택적 구현)"""
        return False
    
    def load_patterns(self, filepath: str) -> bool:
        """패턴 로드 (선택적 구현)"""
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_type': self.__class__.__name__,
            'is_loaded': self.is_loaded(),
            'similarity_threshold': self.similarity_threshold,
            'supports_learning': hasattr(self, 'learn_from_annotations')
        }
