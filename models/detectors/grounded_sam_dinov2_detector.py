"""
Grounding DINO + SAM2 + DINOv2 하이브리드 탐지기
1단계: Grounding DINO로 텍스트 기반 객체 탐지 (바운딩 박스)
2단계: SAM2로 정교한 마스크 추출
3단계: DINOv2로 각 객체의 특징 분석 및 분류
"""
import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import os
import tempfile
from pathlib import Path

from ..base.base_detector import BaseDetector
from ..classifiers.dinov2_classifier import DINOv2Classifier
from config.settings import GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, SAM2_CHECKPOINT_PATH, SAM2_MODEL_CONFIG, DINOV2_SIMILARITY_THRESHOLD

class GroundedSAMDINOv2Detector(BaseDetector):
    """Grounding DINO + SAM2 + DINOv2 하이브리드 탐지기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grounding_dino_model = None
        self.sam2_predictor = None
        self.dinov2_classifier = None
        
        # 텍스트 프롬프트 설정
        self.text_prompt = kwargs.get('text_prompt', "a screw on the conveyor belt . a washer")
        self.box_threshold = kwargs.get('box_threshold', 0.35)
        self.text_threshold = kwargs.get('text_threshold', 0.25)
        self.similarity_threshold = kwargs.get('similarity_threshold', DINOV2_SIMILARITY_THRESHOLD)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 GroundedSAM+DINOv2 initialized with text prompt: '{self.text_prompt}'")
        print(f"   Box threshold: {self.box_threshold}, Text threshold: {self.text_threshold}")
        
    def load_model(self) -> bool:
        """Grounding DINO, SAM2, DINOv2 모델 모두 로드"""
        try:
            print("📥 Loading Grounding DINO + SAM2 + DINOv2 pipeline...")
            
            # 1. Grounding DINO 모델 로드
            if not self._load_grounding_dino():
                return False
            
            # 2. SAM2 모델 로드
            if not self._load_sam2():
                return False
            
            # 3. DINOv2 분류기 로드
            self.dinov2_classifier = DINOv2Classifier()
            if not self.dinov2_classifier.load_model():
                print("❌ Failed to load DINOv2 classifier")
                return False
            
            print("✅ GroundedSAM + DINOv2 pipeline loaded successfully!")
            print(f"   Device: {self.device}")
            print(f"   Text prompt: '{self.text_prompt}'")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load GroundedSAM pipeline: {e}")
            return False
    
    def _load_grounding_dino(self) -> bool:
        """Grounding DINO 모델 로드 (CPU 호환성 포함)"""
        try:
            print("🔄 Loading Grounding DINO model...")
            
            # CPU 환경에서는 YOLO 기반 텍스트 매칭으로 대체
            if not torch.cuda.is_available():
                print("⚠️ CUDA not available. Using YOLO + text matching fallback...")
                return self._load_yolo_text_fallback()
            
            # Grounding DINO 설치 확인 (GPU 환경에서만)
            try:
                import groundingdino
                from groundingdino.models import build_model
                from groundingdino.util import box_ops
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
                from groundingdino.util.inference import annotate, predict
                
                # 모델 설정 및 체크포인트 경로 확인
                config_path = GROUNDING_DINO_CONFIG_PATH
                checkpoint_path = GROUNDING_DINO_CHECKPOINT_PATH
                
                if not os.path.exists(config_path):
                    print(f"❌ Grounding DINO config not found: {config_path}")
                    return False
                
                if not os.path.exists(checkpoint_path):
                    print(f"❌ Grounding DINO checkpoint not found: {checkpoint_path}")
                    return False
                
                # 모델 빌드 및 로드
                args = SLConfig.fromfile(config_path)
                args.device = self.device
                model = build_model(args)
                
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
                print(f"Grounding DINO load result: {load_res}")
                
                model.eval()
                model = model.to(self.device)
                self.grounding_dino_model = model
                
                # 유틸리티 함수들 저장
                self.grounding_dino_utils = {
                    'predict': predict,
                    'annotate': annotate,
                    'box_ops': box_ops
                }
                
                print("✅ Grounding DINO loaded successfully")
                return True
                
            except ImportError as e:
                print(f"❌ Grounding DINO import error: {e}")
                print("❌ Install Grounding DINO with:")
                print("   git clone https://github.com/IDEA-Research/GroundingDINO.git")
                print("   cd GroundingDINO && pip install -e .")
                return False
                
        except Exception as e:
            print(f"❌ Grounding DINO loading error: {e}")
            return False
    
    def _load_sam2(self) -> bool:
        """SAM2 모델 로드 (CPU 환경에서는 간소화된 버전 사용)"""
        try:
            print("🔄 Loading SAM2 model...")
            
            # CPU 환경에서는 SAM2 생략하고 바운딩 박스만 사용
            if not torch.cuda.is_available():
                print("⚠️ CUDA not available. Skipping SAM2 for CPU compatibility...")
                self.sam2_predictor = None
                return True
            
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                # SAM2 체크포인트 경로 확인
                checkpoint_path = SAM2_CHECKPOINT_PATH
                model_cfg = SAM2_MODEL_CONFIG
                
                if not os.path.exists(checkpoint_path):
                    print(f"❌ SAM2 checkpoint not found: {checkpoint_path}")
                    print(f"   Expected path: {checkpoint_path}")
                    print("   Please download SAM2 checkpoints first")
                    return False
                
                # SAM2 모델 빌드 (GPU에서만)
                sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                
                print("✅ SAM2 loaded successfully")
                return True
                
            except ImportError as e:
                print(f"❌ SAM2 import error: {e}")
                print("❌ Install SAM2 with:")
                print("   git clone https://github.com/facebookresearch/segment-anything-2.git")
                print("   cd segment-anything-2 && pip install -e .")
                return False
                
        except Exception as e:
            print(f"❌ SAM2 loading error: {e}")
            return False
    
    def _load_yolo_text_fallback(self) -> bool:
        """CPU 환경을 위한 YOLO + 텍스트 매칭 대체 방법"""
        try:
            print("🔄 Loading YOLO fallback for CPU environment...")
            
            from ultralytics import YOLO
            from ..detectors.yolo_detector import YOLODetector
            
            # YOLO 모델 로드 (CPU 친화적)
            self.yolo_fallback = YOLODetector()
            if not self.yolo_fallback.load_model():
                print("❌ Failed to load YOLO fallback")
                return False
            
            # 간단한 텍스트 매칭을 위한 키워드 추출
            self.text_keywords = self._extract_keywords_from_prompt()
            
            print("✅ YOLO + text matching fallback loaded successfully")
            print(f"   Extracted keywords: {self.text_keywords}")
            return True
            
        except Exception as e:
            print(f"❌ YOLO fallback loading error: {e}")
            return False
    
    def _extract_keywords_from_prompt(self) -> List[str]:
        """텍스트 프롬프트에서 키워드 추출"""
        import re
        
        # 기본 불용어
        stop_words = {'a', 'an', 'the', 'on', 'in', 'at', 'of', 'for', 'with', 'by'}
        
        # 마침표로 분할하고 각 구문에서 키워드 추출
        phrases = [p.strip() for p in self.text_prompt.split('.') if p.strip()]
        keywords = []
        
        for phrase in phrases:
            # 단어 추출 (알파벳만)
            words = re.findall(r'\b[a-zA-Z]+\b', phrase.lower())
            # 불용어 제거 및 길이 2 이상인 단어만
            phrase_keywords = [w for w in words if w not in stop_words and len(w) > 2]
            keywords.extend(phrase_keywords)
        
        return list(set(keywords))  # 중복 제거
    
    def _detect_with_yolo_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """CPU 환경을 위한 YOLO + 텍스트 매칭 대체 탐지"""
        try:
            print("🔍 Running CPU fallback: YOLO + text matching...")
            
            # YOLO로 기본 객체 탐지
            yolo_detections = self.yolo_fallback.detect(frame)
            
            if not yolo_detections:
                print("ℹ️ No objects detected by YOLO fallback")
                return []
            
            # 텍스트 키워드와 매칭되는 객체 필터링
            matched_detections = []
            
            for detection in yolo_detections:
                class_name = detection.get('class_name', '').lower()
                
                # 키워드 매칭 확인
                matched = False
                for keyword in self.text_keywords:
                    if keyword in class_name or any(k in keyword for k in class_name.split()):
                        matched = True
                        break
                
                if matched:
                    # DINOv2로 특징 분석 (CPU에서도 가능)
                    bbox = detection['bbox']
                    features = None
                    if self.dinov2_classifier:
                        features = self.dinov2_classifier.extract_features(frame, bbox)
                    
                    enhanced_detection = {
                        'bbox': bbox,
                        'confidence': detection['confidence'],
                        'class_name': f"matched_{class_name}",
                        'original_phrase': class_name,
                        'grounding_confidence': detection['confidence'],
                        'has_mask': False,  # CPU 환경에서는 마스크 생성 생략
                        'has_features': features is not None,
                        'enhanced_by_dinov2': features is not None,
                        'method': 'yolo_text_fallback_cpu',
                        'pipeline_step': 'cpu_fallback',
                        'matched_keywords': [k for k in self.text_keywords if k in class_name]
                    }
                    
                    matched_detections.append(enhanced_detection)
                    print(f"✅ CPU Fallback matched: {class_name} → matched_{class_name}")
            
            print(f"🖥️ CPU fallback completed: {len(matched_detections)} objects matched")
            return matched_detections
            
        except Exception as e:
            print(f"❌ CPU fallback error: {e}")
            return []
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """GroundedSAM + DINOv2 파이프라인 실행"""
        if not self.is_loaded():
            print("❌ ERROR: GroundedSAM pipeline not loaded!")
            return []
        
        try:
            print(f"🚀 Starting GroundedSAM pipeline on frame {frame.shape}")
            
            # CPU 환경에서는 YOLO 대체 방법 사용
            if not torch.cuda.is_available() and hasattr(self, 'yolo_fallback'):
                return self._detect_with_yolo_fallback(frame)
            
            # 1단계: Grounding DINO로 텍스트 기반 객체 탐지
            print("🔍 Step 1: Grounding DINO detection...")
            boxes, logits, phrases = self._detect_with_grounding_dino(frame)
            
            if len(boxes) == 0:
                print("ℹ️ No objects detected by Grounding DINO")
                return []
            
            print(f"🎯 Grounding DINO detected {len(boxes)} objects: {phrases}")
            
            # 2단계: SAM2로 정교한 마스크 추출 (GPU에서만)
            masks = []
            if self.sam2_predictor is not None:
                print("🔍 Step 2: SAM2 segmentation...")
                masks = self._segment_with_sam2(frame, boxes)
                
                if len(masks) == 0:
                    print("⚠️ SAM2 segmentation failed, using bounding boxes only")
                    masks = [None] * len(boxes)  # None 마스크로 채움
                else:
                    print(f"✂️ SAM2 generated {len(masks)} masks")
            else:
                print("⚠️ SAM2 not available, using bounding boxes only")
                masks = [None] * len(boxes)  # None 마스크로 채움
            
            # 3단계: DINOv2로 각 객체 분석
            print("🔍 Step 3: DINOv2 feature analysis...")
            final_detections = self._analyze_with_dinov2(frame, boxes, masks, phrases, logits)
            
            print(f"🎯 Final pipeline result: {len(final_detections)} objects detected and analyzed")
            return final_detections
            
        except Exception as e:
            print(f"❌ GroundedSAM pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_with_grounding_dino(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Grounding DINO로 텍스트 기반 객체 탐지"""
        try:
            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Grounding DINO 추론 실행
            predict_func = self.grounding_dino_utils['predict']
            boxes, logits, phrases = predict_func(
                model=self.grounding_dino_model,
                image=image_pil,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            
            # 박스 좌표를 이미지 크기에 맞게 변환
            h, w = frame.shape[:2]
            box_ops = self.grounding_dino_utils['box_ops']
            boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
            
            return boxes.cpu().numpy(), logits.cpu().numpy(), phrases
            
        except Exception as e:
            print(f"❌ Grounding DINO detection error: {e}")
            return np.array([]), np.array([]), []
    
    def _segment_with_sam2(self, frame: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """SAM2로 바운딩 박스를 마스크로 변환"""
        try:
            # SAM2 이미지 설정
            self.sam2_predictor.set_image(frame)
            
            masks = []
            for box in boxes:
                try:
                    # 박스 형태를 SAM2 입력 형식으로 변환 (xyxy)
                    input_box = np.array(box)
                    
                    # SAM2 예측 실행
                    mask, scores, logits = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    
                    if len(mask) > 0:
                        masks.append(mask[0])  # 첫 번째 마스크 사용
                    else:
                        print(f"⚠️ SAM2 failed to generate mask for box {box}")
                        
                except Exception as e:
                    print(f"⚠️ SAM2 segmentation error for box {box}: {e}")
                    continue
            
            return masks
            
        except Exception as e:
            print(f"❌ SAM2 segmentation error: {e}")
            return []
    
    def _analyze_with_dinov2(self, frame: np.ndarray, boxes: np.ndarray, masks: List[np.ndarray], 
                           phrases: List[str], logits: np.ndarray) -> List[Dict[str, Any]]:
        """DINOv2로 각 객체의 특징 분석 및 분류"""
        try:
            detections = []
            
            for i, (box, mask, phrase, logit) in enumerate(zip(boxes, masks, phrases, logits)):
                try:
                    # 바운딩 박스를 정수로 변환
                    x1, y1, x2, y2 = map(int, box)
                    bbox_xywh = [x1, y1, x2-x1, y2-y1]  # x, y, w, h 형식
                    
                    # DINOv2로 특징 추출
                    features = self.dinov2_classifier.extract_features(frame, bbox_xywh)
                    
                    if features is not None:
                        # 마스크를 사용한 추가 분석 (옵션)
                        enhanced_label = self._classify_with_features(features, phrase)
                        
                        detection = {
                            'bbox': bbox_xywh,
                            'confidence': float(logit),
                            'class_name': enhanced_label or phrase,
                            'original_phrase': phrase,
                            'grounding_confidence': float(logit),
                            'has_mask': mask is not None,
                            'mask_area': int(np.sum(mask)) if mask is not None else 0,
                            'has_features': True,
                            'enhanced_by_dinov2': True,
                            'method': 'grounded_sam_dinov2',
                            'pipeline_step': 'complete'
                        }
                        
                        # 마스크 정보 추가 (마스크가 있고 포함 조건을 만족하는 경우)
                        if mask is not None and self._should_include_mask(mask):
                            detection['mask'] = mask.astype(np.uint8)
                        
                        detections.append(detection)
                        print(f"✅ Object {i+1}: {phrase} → {enhanced_label or phrase} (conf: {logit:.3f})")
                        
                    else:
                        # DINOv2 특징 추출 실패 시 기본 정보만 포함
                        detection = {
                            'bbox': bbox_xywh,
                            'confidence': float(logit),
                            'class_name': phrase,
                            'original_phrase': phrase,
                            'grounding_confidence': float(logit),
                            'has_mask': mask is not None,
                            'mask_area': int(np.sum(mask)) if mask is not None else 0,
                            'has_features': False,
                            'enhanced_by_dinov2': False,
                            'method': 'grounded_sam_basic',
                            'pipeline_step': 'partial'
                        }
                        
                        detections.append(detection)
                        print(f"⚠️ Object {i+1}: {phrase} (DINOv2 analysis failed, using basic info)")
                    
                except Exception as e:
                    print(f"⚠️ Error analyzing object {i}: {e}")
                    continue
            
            return detections
            
        except Exception as e:
            print(f"❌ DINOv2 analysis error: {e}")
            return []
    
    def _classify_with_features(self, features: np.ndarray, original_phrase: str) -> Optional[str]:
        """DINOv2 특징을 사용한 분류 (확장 가능)"""
        try:
            # 현재는 기본 구현 - 향후 커스텀 분류 로직 추가 가능
            # 예: 학습된 패턴과의 유사도 비교, 클러스터링 등
            
            # 특징 벡터의 통계 정보를 사용한 간단한 분류
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            
            # 임시 분류 로직 (실제 구현에서는 더 정교한 방법 사용)
            if feature_std > 0.1:
                return f"complex_{original_phrase}"
            elif feature_mean > 0.5:
                return f"bright_{original_phrase}"
            else:
                return f"standard_{original_phrase}"
                
        except Exception as e:
            print(f"⚠️ Feature classification error: {e}")
            return None
    
    def _should_include_mask(self, mask: np.ndarray) -> bool:
        """마스크를 결과에 포함할지 결정 (메모리 최적화)"""
        if mask is None:
            return False
            
        # 마스크 크기가 너무 크거나 작으면 제외
        mask_area = np.sum(mask)
        total_pixels = mask.size
        
        if mask_area < 100 or mask_area > total_pixels * 0.8:
            return False
        
        return True
    
    def set_text_prompt(self, text_prompt: str) -> None:
        """텍스트 프롬프트 설정"""
        self.text_prompt = text_prompt
        print(f"🔧 Updated text prompt to: '{text_prompt}'")
    
    def set_thresholds(self, box_threshold: float = None, text_threshold: float = None, 
                      similarity_threshold: float = None) -> None:
        """임계값 설정"""
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_threshold is not None:
            self.text_threshold = text_threshold
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            if self.dinov2_classifier:
                self.dinov2_classifier.similarity_threshold = similarity_threshold
        
        print(f"🔧 Updated thresholds: box={self.box_threshold}, text={self.text_threshold}, similarity={self.similarity_threshold}")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정 (GroundedSAM은 box_threshold 사용)"""
        self.box_threshold = threshold
        print(f"🔧 Updated GroundedSAM box threshold to: {threshold}")
    
    def get_supported_classes(self) -> List[str]:
        """지원하는 클래스 목록 반환 (텍스트 기반이므로 동적)"""
        # 텍스트 프롬프트에서 객체 목록 추출
        phrases = [phrase.strip() for phrase in self.text_prompt.split('.') if phrase.strip()]
        return phrases or ["text_based_detection"]
    
    def is_loaded(self) -> bool:
        """파이프라인 로드 상태 확인"""
        # CPU 환경에서는 YOLO 대체 방법 사용
        if not torch.cuda.is_available():
            yolo_fallback_loaded = hasattr(self, 'yolo_fallback') and self.yolo_fallback is not None
            dinov2_loaded = self.dinov2_classifier is not None and self.dinov2_classifier.is_loaded()
            return yolo_fallback_loaded and dinov2_loaded
        
        # GPU 환경에서는 전체 파이프라인 확인 (SAM2는 선택사항)
        grounding_loaded = self.grounding_dino_model is not None
        dinov2_loaded = self.dinov2_classifier is not None and self.dinov2_classifier.is_loaded()
        
        # SAM2는 있으면 좋지만 필수는 아님
        return grounding_loaded and dinov2_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        base_info = super().get_model_info()
        base_info.update({
            'pipeline_type': 'grounded_sam_dinov2',
            'grounding_dino_loaded': self.grounding_dino_model is not None,
            'sam2_loaded': self.sam2_predictor is not None,
            'dinov2_loaded': self.dinov2_classifier is not None and self.dinov2_classifier.is_loaded(),
            'text_prompt': self.text_prompt,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold,
            'similarity_threshold': self.similarity_threshold,
            'supports_text_prompts': True,
            'supports_zero_shot': True
        })
        return base_info
    
    def update_from_ui_config(self, text_prompt: str, box_threshold: float, 
                            text_threshold: float, similarity_threshold: float = None) -> None:
        """UI에서 전달된 설정으로 업데이트"""
        print(f"🔧 Updating GroundedSAM from UI config...")
        print(f"   Text prompt: '{text_prompt}'")
        print(f"   Box threshold: {box_threshold}")
        print(f"   Text threshold: {text_threshold}")
        
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            if self.dinov2_classifier:
                self.dinov2_classifier.similarity_threshold = similarity_threshold
        
        print("📝 GroundedSAM configuration updated successfully")
