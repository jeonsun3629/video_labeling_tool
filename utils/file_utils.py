"""
파일 처리 유틸리티
"""
import os
import re
from pathlib import Path
from typing import Optional

class FileUtils:
    """파일 처리 관련 유틸리티"""
    
    @staticmethod
    def secure_filename(filename: str) -> str:
        """안전한 파일명 생성"""
        # 기본 문자만 허용
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # 연속된 언더스코어 제거
        filename = re.sub(r'_+', '_', filename)
        
        # 앞뒤 언더스코어 제거
        filename = filename.strip('_')
        
        # 빈 파일명 처리
        if not filename:
            filename = 'unnamed_file'
        
        return filename
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """디렉터리 존재 확인 및 생성"""
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Failed to create directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: str) -> Optional[int]:
        """파일 크기 반환 (bytes)"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return None
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """파일 확장자 반환"""
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """비디오 파일인지 확인"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        return FileUtils.get_file_extension(file_path) in video_extensions
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """이미지 파일인지 확인"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        return FileUtils.get_file_extension(file_path) in image_extensions
    
    @staticmethod
    def generate_unique_filename(base_path: str, filename: str) -> str:
        """중복되지 않는 파일명 생성"""
        file_path = os.path.join(base_path, filename)
        
        if not os.path.exists(file_path):
            return filename
        
        # 파일명과 확장자 분리
        name, ext = os.path.splitext(filename)
        
        # 중복되지 않는 파일명 찾기
        counter = 1
        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_path = os.path.join(base_path, new_filename)
            
            if not os.path.exists(new_path):
                return new_filename
            
            counter += 1
    
    @staticmethod
    def cleanup_old_files(directory: str, max_files: int = 100, file_pattern: str = "*") -> int:
        """오래된 파일 정리"""
        try:
            path = Path(directory)
            if not path.exists():
                return 0
            
            # 패턴에 맞는 파일들 찾기
            files = list(path.glob(file_pattern))
            
            if len(files) <= max_files:
                return 0
            
            # 수정 시간 기준으로 정렬 (오래된 것부터)
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # 오래된 파일들 삭제
            files_to_delete = files[:-max_files]
            deleted_count = 0
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {file_path}: {e}")
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old files from {directory}")
            
            return deleted_count
            
        except Exception as e:
            print(f"❌ Error during file cleanup: {e}")
            return 0
