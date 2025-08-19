"""
탐지 서비스 - 비디오 처리 및 객체 탐지
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from .model_manager import ModelManager
from .memory_service import MemoryService
from config.settings import MAX_VIDEO_WIDTH, DENSE_ANALYSIS_FPS_RATIO, CROP_BATCH_SIZE

class DetectionService:
    """비디오 처리 및 객체 탐지 서비스"""
    
    def __init__(self, model_manager: ModelManager, memory_service: MemoryService):
        self.model_manager = model_manager
        self.memory_service = memory_service
        self.crop_batch_size = CROP_BATCH_SIZE
        
    def process_video(self, video_path: str, dense_analysis: bool = True) -> List[Dict[str, Any]]:
        """스마트 스트리밍 비디오 처리"""
        detector = self.model_manager.get_detector()
        classifier = self.model_manager.get_classifier()
        
        if not detector or not detector.is_loaded():
            print("ERROR: Detector not loaded!")
            return []
        
        print("🚀 Starting smart streaming video processing")
        print(f"🎯 Detector: {self.model_manager.detector_type}")
        print(f"🧠 Classifier: {self.model_manager.classifier_type}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 스마트 프레임 간격 결정
            if dense_analysis:
                frame_interval = max(1, int(fps // DENSE_ANALYSIS_FPS_RATIO))
            else:
                frame_interval = max(1, total_frames // 80)
            
            print(f"📹 Video: {total_frames} frames, {fps:.1f} fps")
            print(f"⚡ Processing every {frame_interval} frames")
            
            annotations = []
            frame_count = 0
            processed_frames = 0
            crop_batch = []
            crop_metadata = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_time = frame_count / fps
                    print(f"\n🎬 Frame {processed_frames + 1} (time: {frame_time:.3f}s)")
                    
                    # 메모리 효율적인 프레임 크기 조정
                    frame_resized, scale = self._resize_frame(frame)
                    
                    # 객체 탐지
                    detections = detector.detect(frame_resized)
                    print(f"🎯 {self.model_manager.detector_type} found: {len(detections)} detections")
                    
                    if detections:
                        # 분류기 사용 여부 결정
                        use_classifier = self._should_use_classifier(len(detections), processed_frames)
                        print(f"🧠 Classifier usage: {'YES' if use_classifier else 'NO'} (memory: {self.memory_service.get_current_usage():.1f}MB)")
                        
                        if use_classifier and classifier and classifier.is_loaded():
                            # 배치 처리를 위해 크롭 수집
                            for detection in detections:
                                bbox = detection['bbox']
                                
                                # 원본 크기로 bbox 복원
                                bbox_original = [int(x / scale) for x in bbox] if scale != 1.0 else bbox
                                
                                # 크롭 이미지 준비
                                x, y, w, h = bbox
                                cropped = frame_resized[y:y+h, x:x+w]
                                
                                if cropped.size > 0:
                                    crop_batch.append(cropped)
                                    crop_metadata.append({
                                        'detection': detection,
                                        'bbox_original': bbox_original,
                                        'frame_time': frame_time,
                                        'frame': frame
                                    })
                            
                            # 배치가 찼거나 마지막이면 분류기 처리
                            if len(crop_batch) >= self.crop_batch_size or frame_count == total_frames - 1:
                                enhanced_detections = self._process_classifier_batch(crop_batch, crop_metadata)
                                annotations.extend(enhanced_detections)
                                
                                # 배치 초기화
                                crop_batch = []
                                crop_metadata = []
                                
                                # 메모리 정리
                                self.memory_service.optimize_memory()
                        else:
                            # 분류기 없이 탐지 결과만 사용
                            for detection in detections:
                                bbox = detection['bbox']
                                if scale != 1.0:
                                    bbox = [int(x / scale) for x in bbox]
                                
                                annotation = {
                                    'frame': f"{frame_time:.3f}",
                                    'label': detection['class_name'],
                                    'bbox': bbox,
                                    'confidence': detection['confidence'],
                                    'original_class': detection['class_name'],
                                    'enhanced_by_classifier': False,
                                    'source': 'auto'
                                }
                                annotations.append(annotation)
                    
                    processed_frames += 1
                    
                    # 주기적 메모리 정리
                    if processed_frames % 10 == 0:
                        current_memory = self.memory_service.get_current_usage()
                        print(f"🧹 Memory cleanup - Current: {current_memory:.1f}MB")
                        self.memory_service.optimize_memory()
                
                frame_count += 1
            
            cap.release()
            
            # 남은 배치 처리
            if crop_batch:
                enhanced_detections = self._process_classifier_batch(crop_batch, crop_metadata)
                annotations.extend(enhanced_detections)
            
            print(f"\n✅ Smart processing completed!")
            print(f"📊 Total detections: {len(annotations)}")
            print(f"🧠 Final memory usage: {self.memory_service.get_current_usage():.1f}MB")
            
            return annotations
            
        except Exception as e:
            print(f"❌ Smart processing error: {e}")
            return []
    
    def _resize_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """프레임 크기 조정"""
        original_height, original_width = frame.shape[:2]
        if original_width > MAX_VIDEO_WIDTH:
            scale = MAX_VIDEO_WIDTH / original_width
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            print(f"📏 Resized: {original_width}x{original_height} → {new_width}x{new_height}")
            return frame_resized, scale
        else:
            return frame.copy(), 1.0
    
    def _should_use_classifier(self, detection_count: int, processed_frames: int) -> bool:
        """분류기 사용 여부 결정"""
        memory_usage = self.memory_service.get_current_usage()
        
        # 메모리 기반 결정
        if memory_usage < 3000:  # 3GB까지는 무조건 사용
            return True
        elif memory_usage < 4000:  # 4GB까지는 선택적 사용
            return processed_frames % 2 == 0
        else:
            return processed_frames % 4 == 0
    
    def _process_classifier_batch(self, crop_batch: List[np.ndarray], crop_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """분류기 배치 처리"""
        classifier = self.model_manager.get_classifier()
        if not crop_batch or not classifier:
            return []
        
        print(f"🧠 Processing classifier batch: {len(crop_batch)} crops")
        enhanced_detections = []
        
        try:
            # 배치를 더 작은 단위로 나누어 처리
            batch_size = min(4, len(crop_batch))
            
            for i in range(0, len(crop_batch), batch_size):
                mini_batch = crop_batch[i:i+batch_size]
                mini_metadata = crop_metadata[i:i+batch_size]
                
                # 특징 추출 및 분류
                for cropped, metadata in zip(mini_batch, mini_metadata):
                    detection = metadata['detection']
                    frame = metadata['frame']
                    bbox = [0, 0, cropped.shape[1], cropped.shape[0]]  # 크롭된 이미지 전체
                    
                    # 분류기로 특징 추출 및 분류
                    features = classifier.extract_features(frame, metadata['detection']['bbox'])
                    if features is not None:
                        enhanced_label = classifier.classify_features(features, detection['class_name'])
                    else:
                        enhanced_label = detection['class_name']
                    
                    annotation = {
                        'frame': f"{metadata['frame_time']:.3f}",
                        'label': enhanced_label,
                        'bbox': metadata['bbox_original'],
                        'confidence': detection['confidence'],
                        'original_class': detection['class_name'],
                        'enhanced_by_classifier': features is not None,
                        'source': 'auto'
                    }
                    enhanced_detections.append(annotation)
                
                # 미니 배치마다 메모리 정리
                del mini_batch
                self.memory_service.optimize_memory()
        
        except Exception as e:
            print(f"⚠️ Classifier batch processing error: {e}")
            # 에러 시 탐지 결과만 반환
            for metadata in crop_metadata:
                detection = metadata['detection']
                annotation = {
                    'frame': f"{metadata['frame_time']:.3f}",
                    'label': detection['class_name'],
                    'bbox': metadata['bbox_original'],
                    'confidence': detection['confidence'],
                    'original_class': detection['class_name'],
                    'enhanced_by_classifier': False,
                    'source': 'auto'
                }
                enhanced_detections.append(annotation)
        
        return enhanced_detections
