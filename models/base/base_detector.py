"""
추상 탐지기 베이스 클래스
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class BaseDetector(ABC):
    """모든 탐지기의 베이스 클래스"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.device = None
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.model_path = kwargs.get('model_path', None)
        
    @abstractmethod
    def load_model(self) -> bool:
        """모델 로드"""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """객체 탐지 수행
        
        Args:
            frame: 입력 프레임 (numpy array)
            
        Returns:
            List of detections with format:
            [
                {
                    'bbox': [x, y, w, h],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """지원하는 클래스 목록 반환"""
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        pass
    
    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.model is not None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 (기본 구현)"""
        return frame
    
    def postprocess_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """탐지 결과 후처리 (기본 구현)"""
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_type': self.__class__.__name__,
            'is_loaded': self.is_loaded(),
            'confidence_threshold': self.confidence_threshold,
            'supported_classes': self.get_supported_classes() if self.is_loaded() else [],
            'model_path': self.model_path
        }
