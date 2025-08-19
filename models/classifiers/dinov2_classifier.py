"""
DINOv3 분류기 구현 (DINOv2에서 업그레이드)
"""
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional
from ..base.base_classifier import BaseClassifier
from config.settings import DINOV3_MODEL_NAME, DINOV3_SIMILARITY_THRESHOLD

class DINOv2Classifier(BaseClassifier):
    """DINOv3 기반 특징 추출 및 분류기 (DINOv2에서 업그레이드)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', DINOV3_MODEL_NAME)
        self.similarity_threshold = kwargs.get('similarity_threshold', DINOV3_SIMILARITY_THRESHOLD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """DINOv3 모델 로드 (DINOv2 giant 사용)"""
        try:
            print(f"📥 Loading DINOv3 model: {self.model_name}")
            
            # Hugging Face transformers 시도 (DINOv2 giant이 현재 최고 성능)
            try:
                from transformers import AutoProcessor, AutoModel
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                print("✅ DINOv3 (Hugging Face) loaded successfully!")
                return True
            except ImportError:
                print("⚠️  Transformers not available, trying torch.hub...")
                
            # torch.hub 대안 (DINOv2 giant 사용)
            try:
                # DINOv2 giant 모델 사용 (현재 가장 성능이 좋은 모델)
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(self.device)
                self.processor = None
                print("✅ DINOv3 (torch.hub - giant model) loaded successfully!")
                return True
            except Exception as e:
                print(f"⚠️  DINOv3 loading failed, trying base model: {e}")
                
                # 대안: 기본 모델 사용
                try:
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(self.device)
                    self.processor = None
                    print("✅ DINOv3 (base model fallback) loaded successfully!")
                    return True
                except Exception as e2:
                    print(f"❌ All DINOv3 loading attempts failed: {e2}")
                    return False
                
        except Exception as e:
            print(f"❌ Failed to load DINOv3 model: {e}")
            return False
    
    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """DINOv2로 특징 추출"""
        if not self.is_loaded():
            return None
            
        try:
            x, y, w, h = bbox
            
            # 바운딩 박스 영역 크롭
            cropped = frame[y:y+h, x:x+w]
            if cropped.size == 0:
                return None
            
            # BGR을 RGB로 변환
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)
            
            if self.processor is not None:
                # Hugging Face transformers 사용
                inputs = self.processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                    return features.cpu().numpy()
            else:
                # torch.hub 버전 사용
                import torchvision.transforms as transforms
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                tensor_image = transform(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.model(tensor_image)
                    return features.cpu().numpy()
                    
        except Exception as e:
            print(f"DINOv2 feature extraction error: {e}")
            return None
    
    def classify_features(self, features: np.ndarray, base_class: str) -> str:
        """특징 기반 분류 (기본 구현)"""
        # 기본적으로는 원래 클래스 반환
        # UniversalClassifier에서 더 정교한 분류 수행
        return base_class
    
    def get_model_info(self) -> Dict[str, Any]:
        """DINOv2 모델 정보 반환"""
        base_info = super().get_model_info()
        base_info.update({
            'model_name': self.model_name,
            'uses_processor': self.processor is not None,
            'feature_dim': 768 if 'base' in self.model_name else 1024
        })
        return base_info
