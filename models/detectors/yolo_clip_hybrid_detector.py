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
        # YOLO 모델 관련 설정 저장
        self.yolo_kwargs = {k: v for k, v in kwargs.items() if k in ['model_path', 'confidence_threshold']}
        # 사용자 정의 CLIP 쿼리 (UI에서 설정 가능)
        self.custom_queries = kwargs.get('defect_queries', [
            # 기본 예시 쿼리들
            "a photo of a person under the car",
            "a person working under a vehicle", 
            "a mechanic under the car",
            "a person lying under the car",
            # 대조군
            "a person standing next to the car",
            "a normal person near the car"
        ])
        
        # 사용자 정의 라벨 매핑 자동 생성
        self.query_labels = kwargs.get('query_labels', self._generate_query_labels())
        
        # 하위 호환성을 위한 별칭
        self.defect_queries = self.custom_queries
        
        print(f"🔧 YOLO+CLIP initialized with {len(self.custom_queries)} custom queries")
        self.defect_threshold = kwargs.get('defect_threshold', CLIP_DEFECT_THRESHOLD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> bool:
        """YOLO와 CLIP 모델 모두 로드"""
        try:
            print("📥 Loading YOLO + CLIP hybrid detector...")
            
            # 1. YOLO 모델 로드 (커스텀 모델 경로 포함)
            self.yolo_detector = YOLODetector(**self.yolo_kwargs)
            yolo_success = self.yolo_detector.load_model()
            
            if not yolo_success:
                print("❌ Failed to load YOLO model")
                return False
            
            # 2. CLIP 모델 로드
            try:
                import clip
                print("🔄 Loading CLIP ViT-B/32 model...")
                self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=self.device)
                self.clip_model.eval()  # 평가 모드로 설정
                print("✅ YOLO + CLIP hybrid detector loaded successfully!")
                print(f"   YOLO classes: {len(self.yolo_detector.get_supported_classes())}")
                print(f"   CLIP device: {self.device}")
                print(f"   Defect threshold: {self.defect_threshold}")
                return True
                
            except ImportError as e:
                print(f"❌ CLIP import error: {e}")
                print("❌ Install CLIP with: pip install git+https://github.com/openai/CLIP.git")
                return False
            except Exception as e:
                print(f"❌ CLIP loading error: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load hybrid detector: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLO + CLIP 하이브리드 탐지"""
        if not self.is_loaded():
            print("❌ ERROR: Hybrid detector not loaded!")
            return []
        
        try:
            # 1단계: YOLO로 객체 탐지
            print("🔍 Running YOLO detection...")
            yolo_detections = self.yolo_detector.detect(frame)
            
            if not yolo_detections:
                print("ℹ️ No objects detected by YOLO")
                return []
            
            print(f"🎯 YOLO detected {len(yolo_detections)} objects")
            
            # 2단계: 각 탐지된 객체를 CLIP으로 분류
            enhanced_detections = []
            successful_analyses = 0
            
            for i, detection in enumerate(yolo_detections):
                try:
                    # 객체 영역 크롭
                    x, y, w, h = detection['bbox']
                    
                    # 경계 확인
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        print(f"⚠️ Invalid bbox for detection {i}: {detection['bbox']}")
                        continue
                    
                    crop = frame[y:y+h, x:x+w]
                    
                    if crop.size == 0:
                        print(f"⚠️ Empty crop for detection {i}")
                        continue
                    
                    # CLIP으로 사용자 정의 상황 분석
                    print(f"🔍 Analyzing object {i+1}/{len(yolo_detections)} with CLIP...")
                    clip_info = self._analyze_with_clip(crop)
                    
                    if not clip_info or clip_info.get('error'):
                        print(f"⚠️ CLIP analysis failed for detection {i}")
                        continue
                    
                    successful_analyses += 1
                    
                    # 기존 탐지 정보에 CLIP 분석 결과 추가
                    enhanced_detection = detection.copy()
                    enhanced_detection.update({
                        'clip_matched': clip_info['matched'],
                        'clip_confidence': clip_info['confidence'],
                        'clip_query': clip_info['matched_query'],
                        'clip_label': clip_info['matched_label'],
                        'clip_analysis': clip_info['analysis'],
                        'method': 'yolo_clip_hybrid'
                    })
                    
                    # CLIP이 사용자 정의 상황을 매칭한 경우
                    if clip_info['matched']:
                        enhanced_detection['class_name'] = clip_info['matched_label']
                        enhanced_detection['original_yolo_class'] = detection['class_name']
                        enhanced_detection['label'] = clip_info['matched_label']
                        print(f"🎯 CLIP matched: {detection['class_name']} → {clip_info['matched_label']} (conf: {clip_info['confidence']:.3f})")
                        print(f"   Query: {clip_info['matched_query']}")
                        enhanced_detections.append(enhanced_detection)
                    else:
                        # 매칭되지 않은 경우 기본 YOLO 클래스로 유지하거나 필터링
                        print(f"❌ No CLIP match for {detection['class_name']} (best: {clip_info['confidence']:.3f})")
                        # 기본 탐지도 포함하려면 아래 주석 해제
                        # enhanced_detections.append(enhanced_detection)
                    
                except Exception as e:
                    print(f"⚠️ Error analyzing detection {i}: {e}")
                    # 에러 시 원본 탐지 결과 유지하지 않음 (하이브리드 모델의 목적에 맞지 않음)
                    continue
            
            print(f"🧠 CLIP analysis completed: {successful_analyses}/{len(yolo_detections)} successful")
            print(f"🎯 Final results: {len(enhanced_detections)} defective objects detected")
            return enhanced_detections
            
        except Exception as e:
            print(f"❌ Hybrid detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _analyze_with_clip(self, crop: np.ndarray) -> Dict[str, Any]:
        """CLIP을 사용하여 크롭된 이미지를 사용자 정의 쿼리와 매칭"""
        try:
            import clip
            
            # 입력 유효성 검사
            if crop is None or crop.size == 0:
                print("⚠️ Invalid crop image for CLIP analysis")
                return {'error': 'invalid_input'}
            
            # 크롭 크기가 너무 작은 경우 처리
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                print(f"⚠️ Crop too small for reliable analysis: {crop.shape}")
                return {'error': 'crop_too_small'}
            
            # BGR to RGB 변환
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            
            # 이미지 전처리
            image_input = self.clip_processor(pil_image).unsqueeze(0).to(self.device)
            
            # 텍스트 쿼리 토큰화
            try:
                text_inputs = clip.tokenize(self.custom_queries).to(self.device)
                if text_inputs.size(0) != len(self.custom_queries):
                    print(f"⚠️ Text tokenization mismatch: {text_inputs.size(0)} vs {len(self.custom_queries)}")
                    return {'error': 'tokenization_mismatch'}
            except Exception as e:
                print(f"⚠️ Text tokenization error: {e}")
                return {'error': 'tokenization_failed'}
            
            with torch.no_grad():
                # 특징 추출
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # 유사도 계산 (정규화)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # 코사인 유사도 계산
                logits = (100.0 * image_features @ text_features.T)
                probs = logits.softmax(dim=-1)
                
                # 각 쿼리별 점수
                scores = probs[0].cpu().numpy()
                print(f"🔍 CLIP scores: {scores}")
                
                # 최고 점수 찾기
                best_idx = scores.argmax()
                best_score = float(scores[best_idx])
                best_query = self.custom_queries[best_idx]
                best_label = self.query_labels[best_idx] if best_idx < len(self.query_labels) else "unknown"
                
                # 임계값 확인 (사용자 정의 상황 매칭 여부)
                matched = best_score > self.defect_threshold
                
                # 디버깅 정보 출력
                print(f"🔍 CLIP Analysis Results:")
                print(f"   Best match: {best_query} (score: {best_score:.3f})")
                print(f"   Threshold: {self.defect_threshold}")
                print(f"   Matched: {'YES' if matched else 'NO'}")
                print(f"   Label: {best_label}")
                
                # 모든 쿼리 점수 출력
                for i, (query, score) in enumerate(zip(self.custom_queries, scores)):
                    label = self.query_labels[i] if i < len(self.query_labels) else "unknown"
                    print(f"   {i+1}. [{label}] {query}: {score:.3f}")
                
                return {
                    'matched': matched,
                    'confidence': best_score,
                    'matched_query': best_query,
                    'matched_label': best_label,
                    'analysis': {
                        'all_scores': scores.tolist(),
                        'all_queries': self.custom_queries,
                        'all_labels': self.query_labels,
                        'best_index': int(best_idx),
                        'threshold': self.defect_threshold
                    }
                }
                
        except Exception as e:
            print(f"❌ CLIP analysis error: {e}")
            return {'error': str(e)}
    
    def _generate_query_labels(self) -> List[str]:
        """사용자 쿼리에서 자동으로 라벨 생성"""
        labels = []
        for query in self.custom_queries:
            # 간단한 키워드 추출로 라벨 생성
            query_lower = query.lower()
            
            if "under" in query_lower and ("car" in query_lower or "vehicle" in query_lower):
                labels.append("person_under_car")
            elif "mechanic" in query_lower:
                labels.append("mechanic_working")
            elif "normal" in query_lower or "standing" in query_lower:
                labels.append("normal_person")
            else:
                # 기본값: 쿼리의 핵심 단어들을 조합
                words = [w for w in query_lower.replace("a photo of", "").replace("a person", "person").split() 
                        if len(w) > 2 and w not in ["the", "and", "with", "under", "near", "next"]]
                label = "_".join(words[:3]) if words else f"custom_{len(labels)}"
                labels.append(label)
        
        return labels
    
    def _get_default_analysis_result(self) -> Dict[str, Any]:
        """기본 분석 결과 반환 (에러 시 사용)"""
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
            # YOLO 기본 클래스 + 사용자 정의 CLIP 라벨들
            extended_classes = base_classes.copy()
            extended_classes.extend(self.query_labels)
            return extended_classes
        return self.query_labels or ["custom_detection"]
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """신뢰도 임계값 설정"""
        if self.yolo_detector:
            self.yolo_detector.set_confidence_threshold(threshold)
    
    def set_defect_threshold(self, threshold: float) -> None:
        """CLIP defect 임계값 설정"""
        self.defect_threshold = threshold
        print(f"🔧 Updated CLIP defect threshold to: {threshold}")
    
    def set_custom_queries(self, queries: List[str], labels: List[str] = None) -> None:
        """사용자 정의 텍스트 쿼리 설정"""
        self.custom_queries = queries
        self.defect_queries = self.custom_queries  # 별칭 업데이트
        self.query_labels = labels or self._generate_query_labels()
        print(f"🔧 Updated custom queries: {len(queries)} total")
        for i, (query, label) in enumerate(zip(self.custom_queries, self.query_labels)):
            print(f"   {i+1}. [{label}] {query}")
    
    def update_from_ui_config(self, defect_queries: List[str], defect_threshold: float) -> None:
        """UI에서 전달된 설정으로 업데이트"""
        print(f"🔧 Updating YOLO+CLIP from UI config...")
        print(f"   Queries: {len(defect_queries)}")
        print(f"   Threshold: {defect_threshold}")
        
        self.custom_queries = defect_queries
        self.defect_queries = self.custom_queries  # 별칭 업데이트
        self.defect_threshold = defect_threshold
        self.query_labels = self._generate_query_labels()
        
        print("📝 Updated queries and labels:")
        for i, (query, label) in enumerate(zip(self.custom_queries, self.query_labels)):
            print(f"   {i+1}. [{label}] {query}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """분석 요약 정보 반환"""
        return {
            'defect_threshold': self.defect_threshold,
            'total_queries': len(self.custom_queries),
            'defect_queries_count': 6,  # 처음 6개가 불량 쿼리
            'normal_queries_count': len(self.custom_queries) - 6,
            'queries': self.custom_queries,
            'device': str(self.device)
        }
    
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
            'defect_queries': self.custom_queries
        })
        return base_info
