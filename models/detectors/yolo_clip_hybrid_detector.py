"""
YOLO + CLIP 하이브리드 탐지기
1단계: YOLO로 객체 탐지 (위치 정보)
2단계: CLIP으로 각 객체의 불량 여부 분류
"""
import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from ..base.base_detector import BaseDetector
from .yolo_detector import YOLODetector
from config.settings import YOLO_MODEL_PATH, CLIP_DEFECT_THRESHOLD

class YOLOCLIPHybridDetector(BaseDetector):
    """YOLO + CLIP 하이브리드 탐지기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yolo_detector = None
        self.clip_model = None
        self.clip_processor = None
        self.defect_queries = kwargs.get('defect_queries', [
            "a photo of a defective product",
            "a photo of a damaged item", 
            "a photo of a broken object",
            "a photo of a quality defect",
            "a photo of a normal product",
            "a photo of a good quality item"
        ])
        self.defect_threshold = kwargs.get('defect_threshold', CLIP_DEFECT_THRESHOLD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """YOLO와 CLIP 모델 모두 로드"""
        try:
            print("📥 Loading YOLO + CLIP hybrid detector...")
            
            # 1. YOLO 모델 로드
            self.yolo_detector = YOLODetector()
            yolo_success = self.yolo_detector.load_model()
            
            if not yolo_success:
                print("❌ Failed to load YOLO model")
                return False
            
            # 2. CLIP 모델 로드
            try:
                import clip
                self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=self.device)
                print("✅ YOLO + CLIP hybrid detector loaded successfully!")
                return True
                
            except ImportError:
                print("❌ CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load hybrid detector: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLO + CLIP 하이브리드 탐지"""
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
            
            # 2단계: 각 탐지된 객체를 CLIP으로 분류
            enhanced_detections = []
            
            for i, detection in enumerate(yolo_detections):
                try:
                    # 객체 영역 크롭
                    x, y, w, h = detection['bbox']
                    crop = frame[y:y+h, x:x+w]
                    
                    if crop.size == 0:
                        continue
                    
                    # CLIP으로 불량 여부 분석
                    defect_info = self._analyze_defect_with_clip(crop)
                    
                    # 기존 탐지 정보에 CLIP 분석 결과 추가
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'is_defective': defect_info['is_defective'],
                        'defect_confidence': defect_info['confidence'],
                        'defect_type': defect_info['defect_type'],
                        'clip_analysis': defect_info['analysis'],
                        'method': 'yolo_clip_hybrid'
                    })
                    
                    # 불량품으로 판정된 경우만 결과에 포함 (기존 YOLO 클래스는 제외)
                    if defect_info['is_defective']:
                        enhanced_detection['class_name'] = f"{detection['class_name']}_defective"
                        enhanced_detection['original_class'] = detection['class_name']
                        enhanced_detection['label'] = f"{detection['class_name']}_defective"
                        print(f"🚨 Defective {detection['class_name']} detected with confidence {defect_info['confidence']:.3f}")
                        enhanced_detections.append(enhanced_detection)
                    else:
                        # 정상 제품은 결과에서 제외 (기존 YOLO 클래스 필터링)
                        print(f"✅ Normal {detection['class_name']} filtered out (confidence: {defect_info['confidence']:.3f})")
                    
                except Exception as e:
                    print(f"⚠️ Error analyzing detection {i}: {e}")
                    # 에러 시 원본 탐지 결과 유지
                    enhanced_detections.append(detection)
            
            print(f"🧠 CLIP analysis completed for {len(enhanced_detections)} objects")
            return enhanced_detections
            
        except Exception as e:
            print(f"❌ Hybrid detection error: {e}")
            return []
    
    def _analyze_defect_with_clip(self, crop: np.ndarray) -> Dict[str, Any]:
        """CLIP을 사용하여 크롭된 이미지의 불량 여부 분석"""
        try:
            import clip
            
            # BGR to RGB 변환
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            
            # 이미지 전처리
            image_input = self.clip_processor(pil_image).unsqueeze(0).to(self.device)
            
            # 텍스트 쿼리 토큰화
            text_inputs = clip.tokenize(self.defect_queries).to(self.device)
            
            with torch.no_grad():
                # 특징 추출
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # 유사도 계산
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # 각 쿼리별 점수
                scores = similarity[0].cpu().numpy()
                
                # defect 관련 쿼리들 (처음 4개)와 normal 관련 쿼리들 (나머지 2개) 분리
                defect_scores = scores[:4]  # defective, damaged, broken, quality defect
                normal_scores = scores[4:]  # normal, good quality
                
                max_defect_score = defect_scores.max()
                max_normal_score = normal_scores.max()
                
                # 불량 여부 판정
                is_defective = max_defect_score > max_normal_score and max_defect_score > self.defect_threshold
                
                # 최고 점수 쿼리 찾기
                if is_defective:
                    best_defect_idx = defect_scores.argmax()
                    defect_type = self.defect_queries[best_defect_idx]
                    confidence = float(max_defect_score)
                else:
                    best_normal_idx = normal_scores.argmax()
                    defect_type = self.defect_queries[4 + best_normal_idx]  # normal 쿼리 인덱스
                    confidence = float(max_normal_score)
                
                return {
                    'is_defective': is_defective,
                    'confidence': confidence,
                    'defect_type': defect_type,
                    'analysis': {
                        'defect_scores': defect_scores.tolist(),
                        'normal_scores': normal_scores.tolist(),
                        'all_similarities': scores.tolist()
                    }
                }
                
        except Exception as e:
            print(f"❌ CLIP analysis error: {e}")
            return {
                'is_defective': False,
                'confidence': 0.0,
                'defect_type': 'analysis_failed',
                'analysis': {}
            }
    
    def get_supported_classes(self) -> List[str]:
        """지원하는 클래스 목록 반환"""
        if self.yolo_detector:
            base_classes = self.yolo_detector.get_supported_classes()
            # 각 클래스에 대해 defective 버전도 추가
            extended_classes = base_classes.copy()
            for cls in base_classes:
                extended_classes.append(f"{cls}_defective")
            return extended_classes
        return ["defect", "normal"]
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        if self.yolo_detector:
            self.yolo_detector.set_confidence_threshold(threshold)
    
    def set_defect_threshold(self, threshold: float) -> None:
        """CLIP defect 임계값 설정"""
        self.defect_threshold = threshold
    
    def is_loaded(self) -> bool:
        """하이브리드 탐지기 로드 상태 확인"""
        yolo_loaded = self.yolo_detector is not None and self.yolo_detector.is_loaded()
        clip_loaded = self.clip_model is not None
        return yolo_loaded and clip_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """하이브리드 모델 정보 반환"""
        base_info = super().get_model_info()
        base_info.update({
            'hybrid_type': 'yolo_clip',
            'yolo_loaded': self.yolo_detector is not None and self.yolo_detector.is_loaded(),
            'clip_loaded': self.clip_model is not None,
            'defect_threshold': self.defect_threshold,
            'defect_queries': self.defect_queries
        })
        return base_info
