from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import requests
from PIL import Image
import io
import base64
import os
import tempfile
import json
from pathlib import Path
import random
import colorsys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import psutil
import gc
from collections import deque

app = Flask(__name__)
CORS(app)  # CORS 설정으로 프론트엔드와 통신 허용

# 전역 변수로 모델 저장
yolo_model = None
dinov2_model = None
dinov2_processor = None

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 전역 변수 초기화
training_data_accumulator = []  # 누적 학습 데이터
current_custom_model_path = None  # 현재 로드된 커스텀 모델 경로

class UniversalCustomLabelClassifier:
    """DINOv2를 활용한 범용 커스텀 라벨 분류기"""
    
    def __init__(self):
        self.label_features = {}  # 라벨별 특징 데이터베이스
        self.feature_clusters = {}  # 라벨별 클러스터
        self.similarity_threshold = 0.7  # 유사도 임계값
        self.min_samples_for_clustering = 3  # 클러스터링 최소 샘플 수
        
    def learn_from_manual_annotations(self, video_path, manual_annotations, dinov2_analyzer):
        """수동 라벨링 데이터에서 패턴 학습"""
        print("🧠 DINOv2 패턴 학습 시작...")
        
        if not manual_annotations:
            print("⚠️ 수동 라벨링 데이터가 없습니다.")
            return False
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            learned_labels = set()
            
            for ann in manual_annotations:
                label = ann['label']
                frame_time = float(ann['frame'])
                bbox = ann['bbox']
                
                # 해당 프레임으로 이동
                frame_number = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # DINOv2 특징 추출
                features = dinov2_analyzer.analyze_with_dinov2(frame, bbox)
                
                if features is not None:
                    if label not in self.label_features:
                        self.label_features[label] = []
                    
                    self.label_features[label].append(features.flatten())
                    learned_labels.add(label)
            
            cap.release()
            
            # 각 라벨별로 클러스터링 수행
            for label in learned_labels:
                self._create_label_clusters(label)
            
            print(f"✅ 학습 완료: {len(learned_labels)}개 라벨 ({sum(len(features) for features in self.label_features.values())}개 샘플)")
            return True
            
        except Exception as e:
            print(f"❌ 패턴 학습 오류: {e}")
            return False
    
    def _create_label_clusters(self, label):
        """라벨별 특징 클러스터 생성"""
        if label not in self.label_features or len(self.label_features[label]) < self.min_samples_for_clustering:
            print(f"📊 {label}: 샘플 부족으로 클러스터링 건너뜀 ({len(self.label_features.get(label, []))}개)")
            return
        
        try:
            features = np.array(self.label_features[label])
            
            # 적응적 클러스터 수 결정
            n_samples = len(features)
            n_clusters = min(3, max(1, n_samples // 2))  # 최대 3개 클러스터
            
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(features)
                
                self.feature_clusters[label] = {
                    'kmeans': kmeans,
                    'centroids': kmeans.cluster_centers_,
                    'labels': cluster_labels,
                    'features': features
                }
                
                print(f"📊 {label}: {n_clusters}개 클러스터 생성 ({n_samples}개 샘플)")
            else:
                # 클러스터가 1개면 평균 특징만 저장
                self.feature_clusters[label] = {
                    'centroids': [np.mean(features, axis=0)],
                    'features': features
                }
                print(f"📊 {label}: 단일 클러스터 생성 ({n_samples}개 샘플)")
                
        except Exception as e:
            print(f"❌ {label} 클러스터링 오류: {e}")
    
    def classify_with_learned_patterns(self, features, base_class_name):
        """학습된 패턴을 기반으로 세부 분류"""
        if features is None:
            return base_class_name
        
        features_flat = features.flatten()
        best_label = base_class_name
        best_similarity = 0
        
        # 모든 학습된 라벨에 대해 유사도 계산
        for label, cluster_info in self.feature_clusters.items():
            if 'centroids' in cluster_info:
                for centroid in cluster_info['centroids']:
                    similarity = cosine_similarity([features_flat], [centroid])[0][0]
                    
                    if similarity > best_similarity and similarity > self.similarity_threshold:
                        best_similarity = similarity
                        best_label = label
        
        # 유사한 패턴을 찾았으면 더 구체적인 라벨 제안
        if best_label != base_class_name and best_similarity > self.similarity_threshold:
            confidence = best_similarity * 100
            print(f"🎯 DINOv2 세부 분류: {base_class_name} → {best_label} (신뢰도: {confidence:.1f}%)")
            return best_label
        
        return base_class_name
    
    def get_learned_labels_info(self):
        """학습된 라벨 정보 반환"""
        info = {}
        for label, features in self.label_features.items():
            cluster_count = len(self.feature_clusters.get(label, {}).get('centroids', []))
            info[label] = {
                'sample_count': len(features),
                'cluster_count': cluster_count,
                'has_clustering': cluster_count > 1
            }
        return info
    
    def save_learned_patterns(self, filepath):
        """학습된 패턴을 파일로 저장"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'label_features': self.label_features,
                    'feature_clusters': self.feature_clusters,
                    'similarity_threshold': self.similarity_threshold
                }, f)
            return True
        except Exception as e:
            print(f"패턴 저장 오류: {e}")
            return False
    
    def load_learned_patterns(self, filepath):
        """저장된 패턴을 로드"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.label_features = data['label_features']
                self.feature_clusters = data['feature_clusters']
                self.similarity_threshold = data['similarity_threshold']
            return True
        except Exception as e:
            print(f"패턴 로드 오류: {e}")
            return False

class AutoLabelingSystem:
    def __init__(self):
        self.yolo_model = None
        self.dinov2_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.universal_classifier = UniversalCustomLabelClassifier()
        
        # 🚀 DINOv2 최대 활용을 위한 적극적 메모리 관리
        total_memory = self.get_available_memory()
        self.memory_limit = total_memory * 0.85  # 85%까지 적극 활용
        self.dinov2_threshold = total_memory * 0.75  # 75%까지는 DINOv2 무조건 사용
        self.crop_batch_size = 6  # 배치 크기 증가
        
        print(f"🚀 AGGRESSIVE DINOv2 mode enabled!")
        print(f"📊 Total memory: {total_memory:.1f}MB")
        print(f"📊 Memory limit: {self.memory_limit:.1f}MB")
        print(f"📊 DINOv2 safe zone: {self.dinov2_threshold:.1f}MB")
        print(f"Using device: {self.device}")
    
    def get_available_memory(self):
        """사용 가능한 메모리 계산 (MB)"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        return available_mb
    
    def get_current_memory_usage(self):
        """현재 프로세스 메모리 사용량 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def optimize_memory(self):
        """메모리 최적화"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def should_use_dinov2(self, current_detections_count, processed_frames):
        """DINOv2 적극 사용 로직"""
        memory_usage = self.get_current_memory_usage()
        
        # 🚀 메모리 제한을 대폭 완화
        if memory_usage < 3000:  # 3GB까지는 무조건 사용
            return True
        elif memory_usage < 4000:  # 4GB까지는 선택적 사용
            return processed_frames % 2 == 0  # 2프레임마다
        else:
            return processed_frames % 4 == 0  # 4프레임마다
    
    def load_models(self):
        """YOLOv8과 DINOv2 모델 로드"""
        try:
            # YOLOv8 모델 로드 (사전 훈련된 모델)
            print("=" * 50)
            print("🤖 Starting AI model initialization...")
            print("=" * 50)
            
            print("📥 Loading YOLOv8 model...")
            self.yolo_model = YOLO('yolov8n.pt')  # nano 버전 (빠름)
            print("✅ YOLOv8 loaded successfully!")
            print(f"   Available classes: {len(self.yolo_model.names)} classes")
            print(f"   Device: {self.device}")
            
            # DINOv2 모델 로드 (옵셔널)
            print("📥 Loading DINOv2 model...")
            try:
                # DINOv2 모델 로드 (Hugging Face transformers 사용)
                from transformers import AutoProcessor, AutoModel
                self.dinov2_processor = AutoProcessor.from_pretrained('facebook/dinov2-base')
                self.dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base')
                self.dinov2_model = self.dinov2_model.to(self.device)
                print("✅ DINOv2 (Hugging Face) loaded successfully!")
            except ImportError as ie:
                print(f"⚠️  Transformers not available ({ie}), trying alternative...")
                try:
                    # 대안: torch.hub를 통한 로드
                    self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                    if hasattr(self.dinov2_model, 'to'):
                        self.dinov2_model = self.dinov2_model.to(self.device)
                    self.dinov2_processor = None
                    print("✅ DINOv2 (torch.hub) loaded successfully!")
                except Exception as te:
                    print(f"⚠️  DINOv2 loading failed: {te}")
                    print("📝 DINOv2 will be skipped - only YOLO detection will be used")
                    self.dinov2_model = None
                    self.dinov2_processor = None
            
            print("=" * 50)
            print("🎉 Model initialization completed!")
            print(f"✅ YOLO: {'Loaded' if self.yolo_model else 'Failed'}")
            print(f"✅ DINOv2: {'Loaded' if self.dinov2_model else 'Skipped'}")
            print("=" * 50)
                
        except Exception as e:
            print(f"❌ Critical error loading models: {e}")
            print("🔧 Please check your dependencies and model files")
            return False
        return True
    
    def extract_frames(self, video_path, max_frames=None, frame_interval=1):
        """비디오에서 프레임 추출 (개선된 버전)"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_indices = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        # max_frames가 None이면 모든 프레임 처리
        if max_frames is None:
            step = frame_interval  # 기본 1프레임마다
        else:
            step = max(frame_interval, total_frames // max_frames)
        
        print(f"Processing every {step} frames")
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % step == 0:
                frames.append(frame)
                frame_indices.append(frame_count / fps)  # 시간(초) 저장
                processed_count += 1
                
                # 너무 많은 프레임 처리 방지 (메모리 보호)
                if max_frames and processed_count >= max_frames:
                    break
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {len(frames)} frames for processing")
        return frames, frame_indices
    
    def detect_objects_yolo(self, frame):
        """YOLOv8로 객체 탐지"""
        if self.yolo_model is None:
            print("ERROR: YOLO model is not loaded!")
            return []
            
        try:
            results = self.yolo_model(frame, conf=0.15)  # 신뢰도 임계값을 0.15로 대폭 낮춤
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
                        
                        # 더 많은 객체 탐지를 위해 낮은 임계값 사용
                        if conf > 0.2:  # 0.3 -> 0.2로 더 낮춤
                            class_name = self.yolo_model.names[int(cls)]
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
    
    def analyze_with_dinov2(self, frame, bbox):
        """DINOv2로 크롭된 이미지 분석"""
        x, y, w, h = bbox
        
        # 바운딩 박스 영역 크롭
        cropped = frame[y:y+h, x:x+w]
        if cropped.size == 0:
            return None
            
        # BGR을 RGB로 변환
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        
        try:
            if self.dinov2_processor is not None:
                # Hugging Face transformers 사용
                inputs = self.dinov2_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    if hasattr(self.dinov2_model, '__call__'):
                        outputs = self.dinov2_model(**inputs)
                        features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                    else:
                        features = None
            else:
                # torch.hub 버전 사용
                try:
                    import torchvision.transforms as transforms
                    
                    # 올바른 transforms 사용
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    # PIL 이미지를 numpy array로 변환 후 tensor로
                    tensor_image = transform(pil_image).unsqueeze(0).to(self.device)
                    
                except ImportError:
                    print("torchvision not available, skipping DINOv2 analysis")
                    return None
                
                with torch.no_grad():
                    if hasattr(self.dinov2_model, '__call__'):
                        features = self.dinov2_model(tensor_image)
                    else:
                        features = None
            
            if features is not None and hasattr(features, 'cpu'):
                return features.cpu().numpy()
            else:
                return None
            
        except Exception as e:
            print(f"DINOv2 analysis error: {e}")
            return None
    
    def classify_with_features(self, features, class_name):
        """DINOv2 특징을 기반으로 분류 개선 - 범용 분류기 사용"""
        if hasattr(self, 'universal_classifier') and self.universal_classifier:
            return self.universal_classifier.classify_with_learned_patterns(features, class_name)
        else:
            # 기본 분류기가 없으면 원래 클래스 이름 반환
            return class_name
    
    def process_video_smart_streaming(self, video_path, dense_analysis=True):
        """🚀 스마트 스트리밍 비디오 처리 - YOLO + DINOv2 최대 활용"""
        if not self.yolo_model:
            print("ERROR: Models not loaded!")
            return []
        
        print("🚀 Starting SMART streaming video processing")
        print(f"🎯 Target: Maximum YOLO + DINOv2 synergy with memory efficiency")
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 📊 스마트 프레임 간격 결정
            if dense_analysis:
                frame_interval = max(1, int(fps // 4))  # 초당 4프레임
            else:
                frame_interval = max(1, total_frames // 80)  # 최대 80프레임
            
            print(f"📹 Video: {total_frames} frames, {fps:.1f} fps")
            print(f"⚡ Processing every {frame_interval} frames")
            
            annotations = []
            frame_count = 0
            processed_frames = 0
            crop_batch = []  # DINOv2 배치 처리용
            crop_metadata = []  # 크롭 메타데이터
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_time = frame_count / fps
                    print(f"\n🎬 Frame {processed_frames + 1} (time: {frame_time:.3f}s)")
                    
                    # 📏 메모리 효율적인 프레임 크기 조정
                    original_height, original_width = frame.shape[:2]
                    if original_width > 1024:  # 크기 제한
                        scale = 1024 / original_width
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)
                        frame_resized = cv2.resize(frame, (new_width, new_height))
                        print(f"📏 Resized: {original_width}x{original_height} → {new_width}x{new_height}")
                    else:
                        frame_resized = frame.copy()
                        scale = 1.0
                    
                    # 🎯 YOLO 객체 탐지
                    detections = self.detect_objects_yolo(frame_resized)
                    print(f"🎯 YOLO found: {len(detections)} detections")
                    
                    if detections:
                        # 🧠 DINOv2 사용 여부 스마트 결정
                        use_dinov2 = self.should_use_dinov2(len(detections), processed_frames)
                        print(f"🧠 DINOv2 usage: {'YES' if use_dinov2 else 'NO'} (memory: {self.get_current_memory_usage():.1f}MB)")
                        
                        if use_dinov2 and self.dinov2_model:
                            # 🔄 배치 처리를 위해 크롭 수집
                            for detection in detections:
                                bbox = detection['bbox']
                                
                                # 원본 크기로 bbox 복원
                                if scale != 1.0:
                                    bbox_original = [int(x / scale) for x in bbox]
                                else:
                                    bbox_original = bbox
                                
                                # 크롭 이미지 준비
                                x, y, w, h = bbox
                                cropped = frame_resized[y:y+h, x:x+w]
                                
                                if cropped.size > 0:
                                    crop_batch.append(cropped)
                                    crop_metadata.append({
                                        'detection': detection,
                                        'bbox_original': bbox_original,
                                        'frame_time': frame_time
                                    })
                            
                            # 🚀 배치가 찼거나 마지막이면 DINOv2 처리
                            if len(crop_batch) >= self.crop_batch_size or frame_count == total_frames - 1:
                                enhanced_detections = self.process_dinov2_batch(crop_batch, crop_metadata)
                                annotations.extend(enhanced_detections)
                                
                                # 배치 초기화
                                crop_batch = []
                                crop_metadata = []
                                
                                # 메모리 정리
                                self.optimize_memory()
                        else:
                            # DINOv2 없이 YOLO 결과만 사용
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
                                    'enhanced_by_dinov2': False
                                }
                                annotations.append(annotation)
                    
                    processed_frames += 1
                    
                    # 🧹 주기적 메모리 정리
                    if processed_frames % 10 == 0:
                        current_memory = self.get_current_memory_usage()
                        print(f"🧹 Memory cleanup - Current: {current_memory:.1f}MB")
                        self.optimize_memory()
                
                frame_count += 1
            
            cap.release()
            
            # 남은 배치 처리
            if crop_batch:
                enhanced_detections = self.process_dinov2_batch(crop_batch, crop_metadata)
                annotations.extend(enhanced_detections)
            
            print(f"\n✅ Smart processing completed!")
            print(f"📊 Total detections: {len(annotations)}")
            print(f"🧠 Final memory usage: {self.get_current_memory_usage():.1f}MB")
            
            return annotations
            
        except Exception as e:
            print(f"❌ Smart processing error: {e}")
            return []
    
    def process_dinov2_batch(self, crop_batch, crop_metadata):
        """🚀 DINOv2 배치 처리 - 메모리 효율적"""
        if not crop_batch or not self.dinov2_model:
            return []
        
        print(f"🧠 Processing DINOv2 batch: {len(crop_batch)} crops")
        enhanced_detections = []
        
        try:
            # 배치를 더 작은 단위로 나누어 처리
            batch_size = min(4, len(crop_batch))
            
            for i in range(0, len(crop_batch), batch_size):
                mini_batch = crop_batch[i:i+batch_size]
                mini_metadata = crop_metadata[i:i+batch_size]
                
                # DINOv2 특징 추출
                features_batch = []
                for cropped in mini_batch:
                    # BGR to RGB
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cropped_rgb)
                    
                    # DINOv2 분석
                    features = self.analyze_with_dinov2_optimized(pil_image)
                    features_batch.append(features)
                
                # 각 크롭에 대해 결과 생성
                for features, metadata in zip(features_batch, mini_metadata):
                    detection = metadata['detection']
                    
                    # DINOv2 기반 세부 분류
                    if features is not None:
                        enhanced_label = self.classify_with_features(features, detection['class_name'])
                    else:
                        enhanced_label = detection['class_name']
                    
                    annotation = {
                        'frame': f"{metadata['frame_time']:.3f}",
                        'label': enhanced_label,
                        'bbox': metadata['bbox_original'],
                        'confidence': detection['confidence'],
                        'original_class': detection['class_name'],
                        'enhanced_by_dinov2': features is not None
                    }
                    enhanced_detections.append(annotation)
                
                # 미니 배치마다 메모리 정리
                del mini_batch, features_batch
                self.optimize_memory()
        
        except Exception as e:
            print(f"⚠️ DINOv2 batch processing error: {e}")
            # 에러 시 YOLO 결과만 반환
            for metadata in crop_metadata:
                detection = metadata['detection']
                annotation = {
                    'frame': f"{metadata['frame_time']:.3f}",
                    'label': detection['class_name'],
                    'bbox': metadata['bbox_original'],
                    'confidence': detection['confidence'],
                    'original_class': detection['class_name'],
                    'enhanced_by_dinov2': False
                }
                enhanced_detections.append(annotation)
        
        return enhanced_detections
    
    def analyze_with_dinov2_optimized(self, pil_image):
        """최적화된 DINOv2 분석"""
        try:
            if self.dinov2_processor is not None:
                # Hugging Face transformers 사용
                inputs = self.dinov2_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    if hasattr(self.dinov2_model, '__call__'):
                        outputs = self.dinov2_model(**inputs)
                        features = outputs.last_hidden_state.mean(dim=1)
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
                    if hasattr(self.dinov2_model, '__call__'):
                        features = self.dinov2_model(tensor_image)
                        return features.cpu().numpy()
            
            return None
            
        except Exception as e:
            print(f"DINOv2 optimized analysis error: {e}")
            return None

    # 기존 process_video 함수를 교체
    def process_video(self, video_path, dense_analysis=True):
        """메인 비디오 처리 함수 - 스마트 스트리밍 사용"""
        return self.process_video_smart_streaming(video_path, dense_analysis)
    
    def prepare_training_data(self, video_path, annotations, output_dir, accumulated_data=None):
        """수동 라벨링 데이터를 YOLO 학습 포맷으로 변환 (누적 학습 지원)"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 이미지와 라벨 폴더 생성
            images_dir = os.path.join(output_dir, 'images')
            labels_dir = os.path.join(output_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 현재 수동 라벨링 데이터
            current_manual_annotations = [ann for ann in annotations if ann.get('source') == 'manual' or ann.get('source') is None]
            
            # 누적 데이터와 현재 데이터 병합
            all_manual_annotations = current_manual_annotations.copy()
            if accumulated_data:
                all_manual_annotations.extend(accumulated_data)
            
            # 고유 라벨 추출 및 클래스 매핑 생성
            unique_labels = list(set([ann['label'] for ann in all_manual_annotations]))
            class_mapping = {label: i for i, label in enumerate(unique_labels)}
            
            print(f"Total training annotations: {len(all_manual_annotations)}")
            print(f"Classes: {unique_labels}")
            
            # classes.txt 파일 생성
            with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
                for label in unique_labels:
                    f.write(f"{label}\n")
            
            # 현재 비디오의 어노테이션을 시간별로 그룹화
            time_annotations = {}
            for ann in current_manual_annotations:
                frame_time = float(ann['frame'])
                if frame_time not in time_annotations:
                    time_annotations[frame_time] = []
                time_annotations[frame_time].append(ann)
            
            saved_frames = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_count / fps
                
                # 현재 시간에 해당하는 어노테이션 찾기
                matching_annotations = []
                for time_key, anns in time_annotations.items():
                    if abs(time_key - current_time) < (1.0 / fps):
                        matching_annotations.extend(anns)
                
                if matching_annotations:
                    # 프레임 저장
                    frame_filename = f"frame_{saved_frames:06d}.jpg"
                    frame_path = os.path.join(images_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # YOLO 포맷 라벨 파일 생성
                    label_filename = f"frame_{saved_frames:06d}.txt"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    with open(label_path, 'w') as f:
                        for ann in matching_annotations:
                            x, y, w, h = ann['bbox']
                            label = ann['label']
                            
                            if label in class_mapping:
                                # YOLO 포맷으로 변환 (normalized coordinates)
                                x_center = (x + w/2) / width
                                y_center = (y + h/2) / height
                                norm_width = w / width
                                norm_height = h / height
                                
                                class_id = class_mapping[label]
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                    
                    saved_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            # 누적 데이터가 있다면 추가로 처리 (기존 이미지 복사)
            if accumulated_data:
                self._copy_accumulated_data(accumulated_data, images_dir, labels_dir, class_mapping, saved_frames)
            
            # dataset.yaml 파일 생성
            yaml_content = f"""
path: {output_dir}
train: images
val: images

nc: {len(unique_labels)}
names: {unique_labels}
"""
            with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
                f.write(yaml_content)
            
            return {
                'success': True,
                'images_count': saved_frames,
                'total_annotations': len(all_manual_annotations),
                'classes': unique_labels,
                'dataset_path': os.path.join(output_dir, 'dataset.yaml'),
                'current_annotations': current_manual_annotations
            }
            
        except Exception as e:
            print(f"Training data preparation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _copy_accumulated_data(self, accumulated_data, images_dir, labels_dir, class_mapping, start_idx):
        """누적된 학습 데이터를 새 학습 폴더에 복사"""
        try:
            for i, ann in enumerate(accumulated_data):
                if 'image_path' in ann and os.path.exists(ann['image_path']):
                    # 이미지 복사
                    new_image_name = f"accumulated_{start_idx + i:06d}.jpg"
                    new_image_path = os.path.join(images_dir, new_image_name)
                    import shutil
                    shutil.copy2(ann['image_path'], new_image_path)
                    
                    # 라벨 파일 생성
                    label_filename = f"accumulated_{start_idx + i:06d}.txt"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    with open(label_path, 'w') as f:
                        label = ann['label']
                        if label in class_mapping:
                            x, y, w, h = ann['bbox']
                            # 원본 이미지 크기 정보가 있다면 사용, 없으면 현재 비디오 크기 사용
                            img_width = ann.get('image_width', 640)
                            img_height = ann.get('image_height', 480)
                            
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            norm_width = w / img_width
                            norm_height = h / img_height
                            
                            class_id = class_mapping[label]
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        except Exception as e:
            print(f"Error copying accumulated data: {e}")
    
    def train_custom_model(self, dataset_path, epochs=50, imgsz=640):
        """커스텀 YOLO 모델 훈련 (개선된 파라미터)"""
        try:
            from ultralytics import YOLO
            
            print(f"🚀 Enhanced training starting with {epochs} epochs...")
            
            # 베이스 모델에서 시작
            model = YOLO('yolov8n.pt')
            
            # 학습 실행 (성능 향상을 위한 파라미터 조정)
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=8,  # 배치 크기 증가
                lr0=0.001,  # 초기 학습률 조정
                lrf=0.01,  # 최종 학습률
                momentum=0.937,  # 모멘텀
                weight_decay=0.0005,  # 가중치 감쇠
                warmup_epochs=3,  # 웜업 에포크
                warmup_momentum=0.8,  # 웜업 모멘텀
                patience=15,  # 조기 종료 인내심 증가
                close_mosaic=10,  # 마지막 10 에포크에서 모자이크 비활성화
                mixup=0.1,  # MixUp 증강
                copy_paste=0.1,  # Copy-Paste 증강
                degrees=10.0,  # 회전 각도
                translate=0.1,  # 이동
                scale=0.9,  # 스케일 변화
                shear=2.0,  # 전단 변환
                perspective=0.0001,  # 원근 변환
                flipud=0.5,  # 상하 뒤집기
                fliplr=0.5,  # 좌우 뒤집기
                mosaic=1.0,  # 모자이크 증강
                save=True,
                project='custom_training',
                name='custom_model',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',  # 더 나은 옵티마이저
                verbose=True,
                val=True,  # 검증 활성화
                plots=True,  # 플롯 생성
                save_period=10  # 체크포인트 저장 간격
            )
            
            # 학습된 모델 경로
            if hasattr(results, 'save_dir') and results.save_dir:
                best_model_path = results.save_dir / 'weights' / 'best.pt'
                model_path = str(best_model_path)
            else:
                # 기본 경로 사용
                model_path = 'custom_training/custom_model/weights/best.pt'
            
            print(f"✅ Enhanced training completed! Model saved at: {model_path}")
            
            return {
                'success': True,
                'model_path': model_path,
                'results': results
            }
            
        except Exception as e:
            print(f"❌ Enhanced training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_custom_model(self, model_path):
        """커스텀 학습된 모델 로드"""
        try:
            self.yolo_model = YOLO(model_path)
            global current_custom_model_path
            current_custom_model_path = model_path
            print(f"Custom model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"Custom model loading error: {e}")
            return False
    
    def generate_colors(self, num_classes):
        """클래스별 고유한 색상 생성"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # BGR 형태로 변환 (OpenCV 사용)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def create_annotated_video(self, video_path, annotations, output_path):
        """바운딩 박스가 그려진 비디오 생성"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 비디오 속성 가져오기
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 출력 비디오 설정
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 클래스별 색상 생성
            unique_labels = list(set([ann['label'] for ann in annotations]))
            colors = self.generate_colors(len(unique_labels))
            label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_count / fps
                
                # 현재 프레임에 해당하는 어노테이션 찾기
                current_annotations = [
                    ann for ann in annotations 
                    if abs(float(ann['frame']) - current_time) < (1.0 / fps)
                ]
                
                # 바운딩 박스 그리기
                for ann in current_annotations:
                    x, y, w, h = ann['bbox']
                    label = ann['label']
                    confidence = ann.get('confidence', 1.0)
                    source = ann.get('source', 'manual')
                    
                    # 색상 선택
                    color = label_colors.get(label, (0, 255, 0))
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 라벨 텍스트 준비
                    if source == 'auto':
                        text = f"{label} ({confidence:.2f})"
                    else:
                        text = f"{label} (Manual)"
                    
                    # 텍스트 배경 크기 계산
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # 텍스트 배경 그리기
                    cv2.rectangle(
                        frame, 
                        (x, y - text_height - 10), 
                        (x + text_width, y), 
                        color, 
                        -1
                    )
                    
                    # 텍스트 그리기
                    cv2.putText(
                        frame, 
                        text, 
                        (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
                
                # 프레임 저장
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            return True
            
        except Exception as e:
            print(f"Video annotation error: {e}")
            return False

# 전역 자동 라벨링 시스템 인스턴스
auto_labeling = AutoLabelingSystem()

# 시작 시 즉시 모델 로드 시도
print("🚀 Initializing models at startup...")
model_load_success = auto_labeling.load_models()
if model_load_success:
    print("✅ Models loaded successfully at startup!")
else:
    print("⚠️  Some models failed to load at startup. You can try manual initialization via /api/init_models")

@app.route('/')
def index():
    """메인 페이지 서빙"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """정적 파일 서빙"""
    return send_from_directory('.', filename)

@app.route('/api/init_models', methods=['POST'])
def init_models():
    """모델 초기화"""
    try:
        success = auto_labeling.load_models()
        if success:
            return jsonify({"status": "success", "message": "Models loaded successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to load models"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """비디오 업로드"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # 임시 파일 저장
        filename = file.filename or 'uploaded_video.mp4'
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        return jsonify({
            "status": "success", 
            "message": "Video uploaded successfully",
            "video_path": temp_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auto_label', methods=['POST'])
def auto_label_video():
    """자동 라벨링 실행"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        confidence_threshold = data.get('confidence_threshold', 0.2)
        dense_analysis = data.get('dense_analysis', True)
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 400
        
        # 신뢰도 임계값 설정 (YOLO 모델에 적용)
        if auto_labeling.yolo_model and hasattr(auto_labeling.yolo_model, 'conf'):
            auto_labeling.yolo_model.conf = confidence_threshold
        
        # 자동 라벨링 수행
        annotations = auto_labeling.process_video(video_path, dense_analysis)
        
        return jsonify({
            "status": "success",
            "annotations": annotations,
            "total_detections": len(annotations),
            "confidence_threshold": confidence_threshold,
            "dense_analysis": dense_analysis
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """시스템 상태 확인"""
    global current_custom_model_path, training_data_accumulator
    
    return jsonify({
        "yolo_loaded": auto_labeling.yolo_model is not None,
        "dinov2_loaded": auto_labeling.dinov2_model is not None,
        "device": str(auto_labeling.device),
        "custom_model_path": current_custom_model_path,
        "accumulated_training_data": len(training_data_accumulator),
        "is_custom_model": current_custom_model_path is not None
    })

@app.route('/api/create_annotated_video', methods=['POST'])
def create_annotated_video():
    """바운딩 박스가 그려진 비디오 생성"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        annotations = data.get('annotations', [])
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 400
        
        if not annotations:
            return jsonify({"error": "No annotations provided"}), 400
        
        # 출력 파일 경로 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_name}_annotated.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # 어노테이션된 비디오 생성
        success = auto_labeling.create_annotated_video(video_path, annotations, output_path)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Annotated video created successfully",
                "output_path": output_path,
                "download_url": f"/api/download/{output_filename}"
            })
        else:
            return jsonify({"error": "Failed to create annotated video"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """파일 다운로드"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/train_custom_model', methods=['POST'])
def train_custom_model():
    """커스텀 모델 학습"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        annotations = data.get('annotations', [])
        epochs = data.get('epochs', 30)
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 400
        
        # 수동 라벨링 데이터만 필터링 (source가 'manual'이거나 없는 경우도 포함)
        manual_annotations = [ann for ann in annotations if ann.get('source') == 'manual' or ann.get('source') is None]
        
        if not manual_annotations:
            return jsonify({"error": "No manual annotations found for training"}), 400
        
        # 학습 데이터 준비 (누적 데이터 포함)
        training_dir = os.path.join(UPLOAD_FOLDER, 'training_data')
        global training_data_accumulator
        prep_result = auto_labeling.prepare_training_data(video_path, annotations, training_dir, training_data_accumulator)
        
        if not prep_result['success']:
            return jsonify({"error": prep_result['error']}), 500
        
        # 모델 학습
        train_result = auto_labeling.train_custom_model(prep_result['dataset_path'], epochs)
        
        if train_result['success']:
            # 현재 라벨링 데이터를 누적 데이터에 추가
            training_data_accumulator.extend(prep_result['current_annotations'])
            
            # 커스텀 모델 자동 로드
            auto_labeling.load_custom_model(train_result['model_path'])
            
            return jsonify({
                "status": "success",
                "message": "Custom model training completed",
                "model_path": train_result['model_path'],
                "images_count": prep_result['images_count'],
                "total_annotations": prep_result['total_annotations'],
                "classes": prep_result['classes']
            })
        else:
            return jsonify({"error": train_result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/load_custom_model', methods=['POST'])
def load_custom_model():
    """학습된 커스텀 모델 로드"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 400
        
        success = auto_labeling.load_custom_model(model_path)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Custom model loaded successfully"
            })
        else:
            return jsonify({"error": "Failed to load custom model"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/create_custom_model_video', methods=['POST'])
def create_custom_model_video():
    """학습된 커스텀 모델로 바운딩박스 비디오 생성"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 400
        
        if not auto_labeling.yolo_model:
            return jsonify({"error": "No model loaded"}), 400
        
        # 조밀한 분석 옵션 확인
        dense_analysis = data.get('dense_analysis', True)
        
        # 커스텀 모델로 전체 비디오 자동 라벨링
        print(f"Creating annotated video with custom model (dense_analysis: {dense_analysis})...")
        annotations = auto_labeling.process_video(video_path, dense_analysis)
        
        if not annotations:
            return jsonify({"error": "No objects detected in video"}), 400
        
        # 출력 파일 경로 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_name}_custom_model.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # 어노테이션된 비디오 생성
        success = auto_labeling.create_annotated_video(video_path, annotations, output_path)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Custom model video created successfully",
                "output_path": output_path,
                "download_url": f"/api/download/{output_filename}",
                "annotations": annotations,
                "total_detections": len(annotations)
            })
        else:
            return jsonify({"error": "Failed to create annotated video"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/learn_dinov2_patterns', methods=['POST'])
def learn_dinov2_patterns():
    """수동 라벨링 데이터에서 DINOv2 패턴 학습"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        annotations = data.get('annotations', [])
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 400
        
        # 수동 라벨링 데이터만 필터링
        manual_annotations = [ann for ann in annotations if ann.get('source') == 'manual' or ann.get('source') is None]
        
        if not manual_annotations:
            return jsonify({"error": "No manual annotations found for learning"}), 400
        
        # DINOv2 모델이 로드되어 있는지 확인
        if not auto_labeling.dinov2_model:
            return jsonify({"error": "DINOv2 model not loaded"}), 400
        
        # 패턴 학습 실행
        success = auto_labeling.universal_classifier.learn_from_manual_annotations(
            video_path, manual_annotations, auto_labeling
        )
        
        if success:
            # 학습된 패턴 정보 가져오기
            learned_info = auto_labeling.universal_classifier.get_learned_labels_info()
            
            return jsonify({
                "status": "success",
                "message": "DINOv2 patterns learned successfully",
                "learned_labels": learned_info,
                "manual_annotations_count": len(manual_annotations)
            })
        else:
            return jsonify({"error": "Failed to learn DINOv2 patterns"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_dinov2_patterns_info', methods=['GET'])
def get_dinov2_patterns_info():
    """학습된 DINOv2 패턴 정보 조회"""
    try:
        learned_info = auto_labeling.universal_classifier.get_learned_labels_info()
        
        return jsonify({
            "status": "success",
            "learned_labels": learned_info,
            "total_labels": len(learned_info),
            "total_samples": sum(info['sample_count'] for info in learned_info.values()),
            "has_clustering": any(info['has_clustering'] for info in learned_info.values())
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/memory_status', methods=['GET'])
def get_memory_status():
    """메모리 상태 반환"""
    try:
        current_usage = auto_labeling.get_current_memory_usage()
        memory_limit = auto_labeling.memory_limit
        
        return jsonify({
            "memory_usage": current_usage,
            "memory_limit": memory_limit,
            "percentage": (current_usage / memory_limit * 100),
            "dinov2_active": auto_labeling.dinov2_model is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎬 Starting Video Labeling Backend Server...")
    print("=" * 60)
    
    # 모델은 이미 시작 시 로드됨
    if auto_labeling.yolo_model:
        print("🎯 Ready to process videos!")
    else:
        print("⚠️  YOLO model not loaded. Please check the logs above.")
    
    print("🌐 Server starting on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000) 