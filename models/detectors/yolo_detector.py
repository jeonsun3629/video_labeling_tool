"""
YOLO 탐지기 구현
"""
import torch
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from ..base.base_detector import BaseDetector
from config.settings import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD

class YOLODetector(BaseDetector):
    """YOLO 기반 객체 탐지기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = kwargs.get('model_path', YOLO_MODEL_PATH)
        self.confidence_threshold = kwargs.get('confidence_threshold', YOLO_CONFIDENCE_THRESHOLD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 커스텀 모델 여부 확인 (custom_training 폴더의 모델인지 체크)
        self.is_custom_model = 'custom_training' in str(self.model_path)
        print(f"🔧 YOLODetector initialized with YOLOv11 model: {self.model_path}")
        if self.is_custom_model:
            print(f"🎯 Custom model detected: {self.model_path}")
        
    def load_model(self) -> bool:
        """YOLO 모델 로드"""
        try:
            print(f"📥 Loading YOLOv11 model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print(f"✅ YOLOv11 model loaded successfully!")
            print(f"   Available classes: {len(self.model.names)} classes")
            print(f"   Device: {self.device}")
            print(f"   Model type: {type(self.model.model).__name__}")
            return True
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLO로 객체 탐지"""
        if not self.is_loaded():
            print("ERROR: YOLO model is not loaded!")
            return []
        
        try:
            # YOLO 추론 실행
            results = self.model(frame, conf=self.confidence_threshold)
            detections = []
            
            print(f"YOLO inference completed, processing {len(results)} results")
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    print(f"Found {len(boxes)} boxes in this result")
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # 신뢰도 필터링
                        if conf > self.confidence_threshold:
                            class_name = self.model.names[int(cls)]
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # x, y, w, h
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': class_name
                            }
                            detections.append(detection)
                            print(f"Detected: {class_name} with confidence {conf:.3f}")
                else:
                    print("No boxes found in this result")
            
            print(f"Total detections after filtering: {len(detections)}")
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def get_supported_classes(self) -> List[str]:
        """지원하는 클래스 목록 반환"""
        if not self.is_loaded():
            return []
        return list(self.model.names.values())
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        self.confidence_threshold = threshold
        if self.is_loaded() and hasattr(self.model, 'conf'):
            self.model.conf = threshold
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """YOLO용 프레임 전처리"""
        # YOLO는 자체적으로 전처리를 수행하므로 원본 반환
        return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """YOLO 모델 정보 반환"""
        base_info = super().get_model_info()
        if self.is_loaded():
            base_info.update({
                'model_size': getattr(self.model, 'model_size', 'unknown'),
                'input_size': getattr(self.model, 'imgsz', 'unknown'),
                'num_classes': len(self.model.names)
            })
        return base_info
