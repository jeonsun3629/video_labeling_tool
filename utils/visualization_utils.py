"""
시각화 유틸리티
"""
import cv2
import numpy as np
import colorsys
from typing import List, Dict, Any, Tuple

class VisualizationUtils:
    """비디오 시각화 관련 유틸리티"""
    
    @staticmethod
    def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
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
    
    @staticmethod
    def create_annotated_video(video_path: str, annotations: List[Dict[str, Any]], output_path: str) -> bool:
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
            colors = VisualizationUtils.generate_colors(len(unique_labels))
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
                    VisualizationUtils.draw_annotation(frame, ann, label_colors)
                
                # 프레임 저장
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            print(f"✅ Annotated video created: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Video annotation error: {e}")
            return False
    
    @staticmethod
    def draw_annotation(frame: np.ndarray, annotation: Dict[str, Any], label_colors: Dict[str, Tuple[int, int, int]]) -> None:
        """프레임에 어노테이션 그리기"""
        x, y, w, h = annotation['bbox']
        label = annotation['label']
        confidence = annotation.get('confidence', 1.0)
        source = annotation.get('source', 'manual')
        enhanced_by_classifier = annotation.get('enhanced_by_classifier', False)
        
        # 색상 선택
        color = label_colors.get(label, (0, 255, 0))
        
        # 바운딩 박스 그리기
        thickness = 3 if enhanced_by_classifier else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # 라벨 텍스트 준비
        if source == 'auto':
            text = f"{label} ({confidence:.2f})"
            if enhanced_by_classifier:
                text += " ✨"  # 분류기로 향상된 경우 표시
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
    
    @staticmethod
    def create_detection_summary_image(annotations: List[Dict[str, Any]], output_path: str, width: int = 800, height: int = 600) -> bool:
        """탐지 결과 요약 이미지 생성"""
        try:
            # 빈 이미지 생성
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img.fill(255)  # 흰색 배경
            
            # 라벨별 통계 계산
            label_counts = {}
            confidence_sums = {}
            for ann in annotations:
                label = ann['label']
                confidence = ann.get('confidence', 1.0)
                
                if label not in label_counts:
                    label_counts[label] = 0
                    confidence_sums[label] = 0
                
                label_counts[label] += 1
                confidence_sums[label] += confidence
            
            # 평균 신뢰도 계산
            avg_confidences = {
                label: confidence_sums[label] / label_counts[label]
                for label in label_counts
            }
            
            # 제목 그리기
            cv2.putText(img, "Detection Summary", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # 통계 그리기
            y_offset = 80
            for i, (label, count) in enumerate(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)):
                avg_conf = avg_confidences[label]
                text = f"{label}: {count} detections (avg: {avg_conf:.2f})"
                
                cv2.putText(img, text, (20, y_offset + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 총 탐지 수 그리기
            total_detections = len(annotations)
            cv2.putText(img, f"Total: {total_detections} detections", 
                       (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 이미지 저장
            cv2.imwrite(output_path, img)
            print(f"✅ Detection summary image created: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Summary image creation error: {e}")
            return False
