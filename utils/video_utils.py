"""
비디오 처리 유틸리티
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class VideoUtils:
    """비디오 처리 관련 유틸리티"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
        """비디오 정보 추출"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return None
    
    @staticmethod
    def extract_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        """특정 시간의 프레임 추출"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(time_seconds * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            print(f"❌ Error extracting frame: {e}")
            return None
    
    @staticmethod
    def extract_frames_batch(video_path: str, frame_times: List[float]) -> List[Optional[np.ndarray]]:
        """여러 시간의 프레임들을 배치로 추출"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return [None] * len(frame_times)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            
            for time_seconds in frame_times:
                frame_number = int(time_seconds * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                frames.append(frame if ret else None)
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"❌ Error extracting frames batch: {e}")
            return [None] * len(frame_times)
    
    @staticmethod
    def resize_frame(frame: np.ndarray, max_width: int = 1024, max_height: int = 768) -> Tuple[np.ndarray, float]:
        """프레임 크기 조정"""
        height, width = frame.shape[:2]
        
        # 크기 조정이 필요한지 확인
        if width <= max_width and height <= max_height:
            return frame.copy(), 1.0
        
        # 비율 계산
        width_ratio = max_width / width
        height_ratio = max_height / height
        scale = min(width_ratio, height_ratio)
        
        # 새로운 크기 계산
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 크기 조정
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        return resized_frame, scale
    
    @staticmethod
    def calculate_optimal_frame_interval(total_frames: int, target_frames: int = 100) -> int:
        """최적 프레임 간격 계산"""
        if total_frames <= target_frames:
            return 1
        
        return max(1, total_frames // target_frames)
    
    @staticmethod
    def validate_video_file(video_path: str) -> Dict[str, Any]:
        """비디오 파일 유효성 검사"""
        result = {
            'is_valid': False,
            'error': None,
            'info': None
        }
        
        try:
            # 파일 존재 확인
            import os
            if not os.path.exists(video_path):
                result['error'] = "File does not exist"
                return result
            
            # 비디오 정보 추출 시도
            info = VideoUtils.get_video_info(video_path)
            
            if info is None:
                result['error'] = "Cannot read video file"
                return result
            
            # 기본 유효성 검사
            if info['frame_count'] <= 0:
                result['error'] = "Video has no frames"
                return result
            
            if info['fps'] <= 0:
                result['error'] = "Invalid FPS"
                return result
            
            result['is_valid'] = True
            result['info'] = info
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def create_video_thumbnail(video_path: str, output_path: str, time_seconds: float = 1.0) -> bool:
        """비디오 썸네일 생성"""
        try:
            frame = VideoUtils.extract_frame_at_time(video_path, time_seconds)
            
            if frame is None:
                return False
            
            # 썸네일 크기로 조정
            thumbnail, _ = VideoUtils.resize_frame(frame, max_width=320, max_height=240)
            
            # 이미지 저장
            success = cv2.imwrite(output_path, thumbnail)
            
            if success:
                print(f"✅ Thumbnail created: {output_path}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error creating thumbnail: {e}")
            return False
