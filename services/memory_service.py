"""
메모리 관리 서비스
"""
import gc
import torch
import psutil
from typing import Dict, Any
from config.settings import MEMORY_LIMIT_RATIO, DINOV2_THRESHOLD_RATIO

class MemoryService:
    """메모리 모니터링 및 최적화 서비스"""
    
    def __init__(self):
        total_memory = self._get_available_memory()
        self.memory_limit = total_memory * MEMORY_LIMIT_RATIO
        self.dinov2_threshold = total_memory * DINOV2_THRESHOLD_RATIO
        
        print(f"🚀 Memory service initialized")
        print(f"📊 Total memory: {total_memory:.1f}MB")
        print(f"📊 Memory limit: {self.memory_limit:.1f}MB")
        print(f"📊 DINOv2 safe zone: {self.dinov2_threshold:.1f}MB")
    
    def _get_available_memory(self) -> float:
        """사용 가능한 메모리 계산 (MB)"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        return available_mb
    
    def get_current_usage(self) -> float:
        """현재 프로세스 메모리 사용량 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_system_memory_info(self) -> Dict[str, Any]:
        """시스템 메모리 정보 반환"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024 / 1024,
            'available': memory.available / 1024 / 1024,
            'used': memory.used / 1024 / 1024,
            'percentage': memory.percent,
            'process_usage': self.get_current_usage()
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화 수행"""
        before_usage = self.get_current_usage()
        
        # Python 가비지 컬렉션
        collected = gc.collect()
        
        # PyTorch CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        after_usage = self.get_current_usage()
        freed_mb = before_usage - after_usage
        
        result = {
            'before_usage': before_usage,
            'after_usage': after_usage,
            'freed_mb': freed_mb,
            'collected_objects': collected,
            'cuda_cleared': torch.cuda.is_available()
        }
        
        if freed_mb > 10:  # 10MB 이상 해제된 경우만 로그
            print(f"🧹 Memory optimized: {freed_mb:.1f}MB freed")
        
        return result
    
    def is_memory_available(self, required_mb: float = 500) -> bool:
        """메모리 사용 가능 여부 확인"""
        current_usage = self.get_current_usage()
        return (current_usage + required_mb) < self.memory_limit
    
    def should_use_intensive_processing(self) -> bool:
        """집약적 처리 사용 여부 결정"""
        current_usage = self.get_current_usage()
        return current_usage < self.dinov2_threshold
    
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 반환"""
        current_usage = self.get_current_usage()
        system_info = self.get_system_memory_info()
        
        return {
            'process_usage_mb': current_usage,
            'memory_limit_mb': self.memory_limit,
            'usage_percentage': (current_usage / self.memory_limit) * 100,
            'can_use_intensive_processing': self.should_use_intensive_processing(),
            'system_memory': system_info,
            'thresholds': {
                'memory_limit': self.memory_limit,
                'dinov2_threshold': self.dinov2_threshold
            }
        }
    
    def set_memory_limit(self, limit_mb: float) -> None:
        """메모리 제한 설정"""
        self.memory_limit = limit_mb
        print(f"📊 Memory limit updated to: {limit_mb:.1f}MB")
    
    def force_cleanup(self) -> Dict[str, Any]:
        """강제 메모리 정리"""
        print("🧹 Force memory cleanup initiated...")
        
        # 여러 번 가비지 컬렉션 수행
        results = []
        for i in range(3):
            result = self.optimize_memory()
            results.append(result)
        
        # 최종 결과
        total_freed = sum(r['freed_mb'] for r in results)
        final_usage = self.get_current_usage()
        
        cleanup_result = {
            'cleanup_rounds': len(results),
            'total_freed_mb': total_freed,
            'final_usage_mb': final_usage,
            'individual_results': results
        }
        
        print(f"🧹 Force cleanup completed: {total_freed:.1f}MB freed")
        return cleanup_result
