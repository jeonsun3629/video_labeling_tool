"""
훈련 서비스
"""
import os
import cv2
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from ultralytics import YOLO
from .model_manager import ModelManager
from config.settings import UPLOAD_FOLDER, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, BASE_DIR

class TrainingService:
    """모델 훈련 관련 서비스"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def cleanup_previous_training(self) -> None:
        """이전 훈련 결과 정리"""
        try:
            training_base_dir = Path(BASE_DIR) / 'custom_training'
            
            if training_base_dir.exists():
                print("🧹 이전 훈련 결과 정리 중...")
                
                # custom_training 디렉터리 전체 삭제
                shutil.rmtree(training_base_dir)
                print("✅ 이전 훈련 결과 정리 완료")
            
            # 새로운 디렉터리 생성
            training_base_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            print(f"⚠️ 훈련 결과 정리 중 오류 (계속 진행): {e}")
        
    def train_model(self, video_path: str, annotations: List[Dict[str, Any]], 
                   epochs: int = DEFAULT_EPOCHS, accumulated_data: Optional[List] = None) -> Dict[str, Any]:
        """커스텀 모델 훈련"""
        try:
            # 이전 훈련 결과 정리
            self.cleanup_previous_training()
            
            # 학습 데이터 준비
            training_dir = os.path.join(UPLOAD_FOLDER, 'training_data')
            prep_result = self.prepare_training_data(video_path, annotations, training_dir, accumulated_data)
            
            if not prep_result['success']:
                return {'success': False, 'error': prep_result['error']}
            
            # YOLO 모델 훈련
            train_result = self.train_yolo_model(prep_result['dataset_path'], epochs)
            
            if train_result['success']:
                # 결과에 준비 단계 정보 추가
                train_result.update({
                    'images_count': prep_result['images_count'],
                    'total_annotations': prep_result['total_annotations'],
                    'classes': prep_result['classes'],
                    'current_annotations': prep_result['current_annotations']
                })
            
            return train_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def prepare_training_data(self, video_path: str, annotations: List[Dict[str, Any]], 
                            output_dir: str, accumulated_data: Optional[List] = None) -> Dict[str, Any]:
        """수동 라벨링 데이터를 YOLO 학습 포맷으로 변환"""
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
            
            # 누적 데이터가 있다면 추가로 처리
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
    
    def _copy_accumulated_data(self, accumulated_data: List[Dict[str, Any]], images_dir: str, 
                             labels_dir: str, class_mapping: Dict[str, int], start_idx: int):
        """누적된 학습 데이터를 새 학습 폴더에 복사"""
        try:
            for i, ann in enumerate(accumulated_data):
                if 'image_path' in ann and os.path.exists(ann['image_path']):
                    # 이미지 복사
                    new_image_name = f"accumulated_{start_idx + i:06d}.jpg"
                    new_image_path = os.path.join(images_dir, new_image_name)
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
    
    def train_yolo_model(self, dataset_path: str, epochs: int = DEFAULT_EPOCHS) -> Dict[str, Any]:
        """YOLO 모델 훈련"""
        try:
            print(f"🚀 Enhanced YOLO training starting with {epochs} epochs...")
            
            # 베이스 모델에서 시작
            model = YOLO('yolov8n.pt')
            
            # 학습 실행 (성능 향상을 위한 파라미터 조정)
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=DEFAULT_IMAGE_SIZE,
                batch=DEFAULT_BATCH_SIZE,
                lr0=0.001,  # 초기 학습률
                lrf=0.01,  # 최종 학습률
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                patience=15,
                close_mosaic=10,
                mixup=0.1,
                copy_paste=0.1,
                degrees=10.0,
                translate=0.1,
                scale=0.9,
                shear=2.0,
                perspective=0.0001,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0,
                save=True,
                project='custom_training',
                name='custom_model',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                verbose=True,
                val=True,
                plots=True,
                save_period=10
            )
            
            # 학습된 모델 경로
            if hasattr(results, 'save_dir') and results.save_dir:
                best_model_path = results.save_dir / 'weights' / 'best.pt'
                model_path = str(best_model_path)
            else:
                # 기본 경로 사용
                model_path = 'custom_training/custom_model/weights/best.pt'
            
            print(f"✅ Enhanced YOLO training completed! Model saved at: {model_path}")
            
            # JSON 직렬화 가능한 훈련 결과 정보만 추출
            training_info = {
                'model_path': model_path,
                'epochs_completed': epochs,
                'save_dir': str(results.save_dir) if hasattr(results, 'save_dir') else None
            }
            
            # 훈련 메트릭이 있다면 안전하게 추출
            try:
                if hasattr(results, 'results_dict'):
                    # 기본 메트릭 정보 추출
                    metrics = {}
                    results_dict = results.results_dict
                    for key, value in results_dict.items():
                        if isinstance(value, (int, float, str, bool, list)):
                            metrics[key] = value
                        elif hasattr(value, 'item'):  # torch.Tensor 등
                            try:
                                metrics[key] = value.item()
                            except:
                                metrics[key] = str(value)
                        else:
                            metrics[key] = str(value)
                    training_info['metrics'] = metrics
                
                # 간단한 통계 정보
                if hasattr(results, 'speed'):
                    training_info['training_speed'] = str(results.speed)
                    
            except Exception as e:
                print(f"⚠️ Could not extract training metrics: {e}")
                training_info['metrics_error'] = str(e)
            
            return {
                'success': True,
                'model_path': model_path,
                'training_info': training_info
            }
            
        except Exception as e:
            print(f"❌ Enhanced training error: {e}")
            return {'success': False, 'error': str(e)}
