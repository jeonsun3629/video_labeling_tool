"""
범용 분류기 구현 (기존 UniversalCustomLabelClassifier 리팩토링)
"""
import numpy as np
import cv2
import pickle
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from ..base.base_classifier import BaseClassifier

class UniversalClassifier(BaseClassifier):
    """DINOv2 특징을 활용한 범용 커스텀 라벨 분류기"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_features = {}  # 라벨별 특징 데이터베이스
        self.feature_clusters = {}  # 라벨별 클러스터
        self.min_samples_for_clustering = kwargs.get('min_samples_for_clustering', 3)
        
        # DINOv2 분류기와 연동
        self.feature_extractor = None
        
    def load_model(self) -> bool:
        """모델 로드 (특징 추출기 설정)"""
        # UniversalClassifier는 별도 모델이 아니라 특징 추출기를 사용
        return True
    
    def set_feature_extractor(self, extractor):
        """특징 추출기 설정"""
        self.feature_extractor = extractor
    
    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """특징 추출 (특징 추출기 위임)"""
        if self.feature_extractor:
            return self.feature_extractor.extract_features(frame, bbox)
        return None
    
    def classify_features(self, features: np.ndarray, base_class: str) -> str:
        """학습된 패턴을 기반으로 세부 분류"""
        if features is None:
            return base_class
        
        features_flat = features.flatten()
        best_label = base_class
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
        if best_label != base_class and best_similarity > self.similarity_threshold:
            confidence = best_similarity * 100
            print(f"🎯 Universal classifier: {base_class} → {best_label} (신뢰도: {confidence:.1f}%)")
            return best_label
        
        return base_class
    
    def learn_from_annotations(self, video_path: str, annotations: List[Dict[str, Any]]) -> bool:
        """수동 라벨링 데이터에서 패턴 학습"""
        if not self.feature_extractor:
            print("⚠️ Feature extractor not set")
            return False
            
        print("🧠 Universal classifier 패턴 학습 시작...")
        print("🗑️ 기존 학습된 패턴 초기화...")
        
        # 기존 패턴 데이터 초기화
        self.label_features = {}
        self.feature_clusters = {}
        
        if not annotations:
            print("⚠️ 수동 라벨링 데이터가 없습니다.")
            return False
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            learned_labels = set()
            
            for ann in annotations:
                label = ann['label']
                frame_time = float(ann['frame'])
                bbox = ann['bbox']
                
                # 해당 프레임으로 이동
                frame_number = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # 특징 추출
                features = self.extract_features(frame, bbox)
                
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
    
    def _create_label_clusters(self, label: str):
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
    
    def get_learned_patterns_info(self) -> Dict[str, Any]:
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
    
    def save_patterns(self, filepath: str) -> bool:
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
    
    def load_patterns(self, filepath: str) -> bool:
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
    
    def has_learned_patterns(self) -> bool:
        """학습된 패턴이 있는지 확인"""
        return len(self.label_features) > 0
    
    def classify_as_custom_object(self, features: np.ndarray) -> Optional[str]:
        """특징을 사용하여 학습된 커스텀 객체인지 판별"""
        if features is None:
            print("❌ Features is None, cannot classify")
            return None
            
        if not self.has_learned_patterns():
            print("❌ No learned patterns available")
            return None
        
        features_flat = features.flatten()
        best_label = None
        best_similarity = 0
        all_similarities = []
        
        print(f"🔍 Checking against {len(self.feature_clusters)} learned labels (threshold: {self.similarity_threshold})")
        
        # 모든 학습된 라벨에 대해 유사도 계산
        for label, cluster_info in self.feature_clusters.items():
            if 'centroids' in cluster_info:
                max_sim_for_label = 0
                for i, centroid in enumerate(cluster_info['centroids']):
                    similarity = cosine_similarity([features_flat], [centroid])[0][0]
                    max_sim_for_label = max(max_sim_for_label, similarity)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        if similarity > self.similarity_threshold:
                            best_label = label
                
                all_similarities.append((label, max_sim_for_label))
                print(f"  📊 {label}: {max_sim_for_label:.3f} {'✅' if max_sim_for_label > self.similarity_threshold else '❌'}")
        
        # 임계값을 넘는 유사도를 가진 라벨이 있으면 반환
        if best_label and best_similarity > self.similarity_threshold:
            confidence = best_similarity * 100
            print(f"🎯 Custom object MATCH: {best_label} (신뢰도: {confidence:.1f}%)")
            return best_label
        else:
            print(f"❌ No match found (best: {best_similarity:.3f}, threshold: {self.similarity_threshold})")
            return None
    
    def get_learned_labels(self) -> List[str]:
        """학습된 라벨 목록 반환"""
        return list(self.label_features.keys())