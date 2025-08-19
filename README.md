# 🎯 Video Labeling Tool with AI (YOLOv11 + DINOv2)

동영상에서 객체를 수동으로 라벨링하고, YOLOv11과 DINOv2를 활용한 AI 자동 라벨링 기능을 제공하는 모듈화된 웹 기반 도구입니다.

## ✨ 주요 기능

- 🎥 **웹 기반 비디오 라벨링**: 브라우저에서 직접 동영상을 업로드하고 라벨링
- 🖱️ **직관적인 인터페이스**: 마우스 드래그로 간편한 바운딩 박스 생성
- 🤖 **AI 자동 라벨링**: YOLOv11 + DINOv2를 활용한 정밀한 객체 탐지
- 🧠 **DINOv2 지능형 패턴 학습**: 사용자 정의 라벨의 시각적 패턴을 자동 학습
- 🎯 **범용 커스텀 라벨 지원**: 어떤 라벨이든 자동으로 세부 분류 학습
- ⚡ **실시간 처리**: 프레임별 정밀 제어 및 즉시 결과 확인
- 💾 **표준 포맷 지원**: YOLO 학습에 바로 사용 가능한 JSON 출력

## 🏗️ 시스템 구조

```
Frontend (웹 브라우저)          Backend (모듈화된 Python 서버)
┌─────────────────────┐       ┌─────────────────────────────────┐
│  HTML/CSS/JS        │ ←──→  │  Flask API (app_modular.py)     │
│  - 비디오 플레이어    │       │  ┌─────────────────────────────┐ │
│  - 캔버스 그리기      │       │  │ Services                    │ │
│  - 라벨링 인터페이스   │       │  │ - ModelManager              │ │
└─────────────────────┘       │  │ - DetectionService          │ │
                              │  │ - TrainingService           │ │
                              │  │ - MemoryService             │ │
                              │  └─────────────────────────────┘ │
                              │  ┌─────────────────────────────┐ │
                              │  │ Models                      │ │
                              │  │ - YOLOv11 Detectors         │ │
                              │  │ - DINOv2 Classifiers        │ │
                              │  │ - CLIP Classifiers          │ │
                              │  │ - Hybrid Models             │ │
                              │  └─────────────────────────────┘ │
                              └─────────────────────────────────┘
```

## 🚀 설치 및 실행

### 1. 환경 요구사항

- Python 3.8 이상
- 8GB+ RAM (AI 모델 로딩용)
- GPU 권장 (CUDA 지원시 더 빠른 처리)

### 2. 패키지 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 주요 종속성:
# - YOLOv11: ultralytics>=8.3.180
# - DINOv2: transformers>=4.35.0  
# - 패턴 학습: scikit-learn>=1.3.0
# - 웹 프레임워크: flask>=2.3.0
# - CLIP: git+https://github.com/openai/CLIP.git
```

### 3. 서버 실행

```bash
# 방법 1: 간단 실행
python run_server.py

# 방법 2: 직접 실행
python app_modular.py
```

서버가 시작되면 브라우저에서 `http://localhost:5000`으로 접속하세요.

## 📖 사용 방법

### 🚀 4단계 워크플로우

#### **1단계: 수동 라벨링**
1. **비디오 업로드**: [파일 선택] 버튼으로 동영상 선택
2. **프레임 이동**: ◀ ▶ 버튼으로 원하는 프레임으로 이동
3. **박스 그리기**: 마우스로 드래그하여 객체 주위에 빨간 박스 생성
4. **라벨 입력**: "fallen_person", "first_responder" 등 **어떤 라벨이든** 입력 후 [현재 박스 저장]
5. **통계 확인**: 우측에서 라벨별 개수와 비율 실시간 확인

#### **2단계: 지능형 모델 학습**
1. **DINOv2 패턴 학습**: [🎯 DINOv2 패턴 학습] 버튼으로 사용자 라벨의 시각적 패턴 자동 학습
   - 각 라벨별로 특징 추출 및 클러스터링
   - 라벨별 샘플 수와 클러스터 정보 표시
2. **YOLO 커스텀 학습**: [🚀 모델 학습 시작] 버튼으로 기존 YOLO 모델 fine-tuning

#### **3단계: AI 자동 라벨링 비디오 생성**
1. **기본 모델**: [🤖 기본 YOLO 모델로 비디오 생성] - 사전 훈련된 모델 사용
2. **커스텀 모델**: [🚀 커스텀 모델로 비디오 생성] - 학습된 모델 + DINOv2 지능형 분류
   - DINOv2가 학습한 패턴을 기반으로 더 정확한 세부 분류
   - 사용자 정의 라벨에 대한 고품질 자동 라벨링

#### **4단계: 데이터 검증 및 내보내기**
1. **데이터 검토**: 수동/자동 라벨링 데이터 통합 확인
2. **JSON 내보내기**: [📄 JSON 데이터 내보내기] 버튼으로 `annotations.json` 파일 다운로드

## 📊 출력 데이터 형식

```json
[
  {
    "frame": "2.133",
    "label": "fallen_person",
    "bbox": [120, 80, 200, 150],
    "confidence": 0.85,
    "source": "auto",
    "original_class": "person"
  },
  {
    "frame": "3.450", 
    "label": "car",
    "bbox": [300, 100, 180, 120],
    "source": "manual"
  }
]
```

- `frame`: 시간(초)
- `label`: 최종 라벨
- `bbox`: [x, y, width, height] 바운딩 박스
- `confidence`: AI 신뢰도 (자동 라벨링시만)
- `source`: "auto" (AI) 또는 "manual" (수동)

## 🤖 AI 모델 상세

### YOLOv11 (1차 객체 탐지)
- **역할**: 빠른 객체 탐지 및 바운딩 박스 생성
- **특징**: 실시간 처리 가능, 80개 기본 클래스 지원
- **사용 모델**: `yolo11n.pt` (nano 버전, 빠른 추론)
- **커스텀 학습**: 사용자 수동 라벨링 데이터로 fine-tuning 가능

### DINOv2 (지능형 패턴 학습 및 분류)
- **역할**: 사용자 정의 라벨의 시각적 패턴 학습 및 세부 분류
- **특징**: Self-supervised learning, 범용 특징 추출
- **사용 모델**: `dinov2-base` (Hugging Face)
- **핵심 기능**:
  - 🎯 **범용 라벨 학습**: fallen_person, first_responder 등 어떤 라벨이든 자동 학습
  - 🧠 **패턴 클러스터링**: 라벨별 시각적 특징을 클러스터로 구성
  - 🔍 **지능형 분류**: 코사인 유사도 기반 정밀 세부 분류

### 🚀 **혁신적인 YOLOv11 + DINOv2 하이브리드 시스템**
```
수동 라벨링 → DINOv2 특징 추출 → 지능적 데이터 증강 → YOLO 학습 → 고품질 자동 라벨링
    ↓              ↓                ↓              ↓           ↓
  적은 데이터    풍부한 특징        유사 샘플 발견     더 나은 학습   정밀한 분류
```

#### **하이브리드 모델 아키텍처**
- **YOLOv11 + DINOv2 Hybrid**: 기본 탐지 + 지능형 분류
- **YOLOv11 + CLIP Hybrid**: 기본 탐지 + 텍스트-이미지 매칭

#### **통합 워크플로우**
1. **YOLOv11 탐지**: 기본 객체 탐지 (person, car 등)
2. **DINOv2 특징 추출**: 탐지된 바운딩 박스에서 고차원 특징 벡터 추출
3. **패턴 학습**: 사용자 라벨별로 특징 클러스터링 및 패턴 저장
4. **지능형 분류**: 새로운 객체에 대해 학습된 패턴과 유사도 비교하여 정확한 라벨 할당

## 📁 파일 구조

```
video_labeling_tool/
├── index.html              # 프론트엔드 메인 페이지
├── style.css               # 스타일시트
├── script.js               # 프론트엔드 로직
├── app_modular.py          # 모듈화된 Flask 백엔드 서버
├── app_old.py              # 기존 단일 파일 서버 (레거시)
├── run_server.py           # 서버 실행 스크립트
├── requirements.txt        # Python 종속성
├── README.md               # 이 파일
├── config/                 # 설정 파일들
│   └── settings.py         # 시스템 설정
├── models/                 # AI 모델 모듈
│   ├── base/               # 베이스 클래스
│   │   ├── base_detector.py
│   │   ├── base_classifier.py
│   │   └── model_factory.py
│   ├── detectors/          # 탐지 모델들
│   │   ├── yolo_detector.py
│   │   ├── yolo_dinov2_hybrid_detector.py
│   │   └── yolo_clip_hybrid_detector.py
│   ├── classifiers/        # 분류 모델들
│   │   ├── dinov2_classifier.py
│   │   ├── clip_classifier.py
│   │   └── universal_classifier.py
│   └── processors/         # 데이터 처리기 (향후 확장)
├── services/               # 비즈니스 로직 서비스
│   ├── model_manager.py    # 모델 관리
│   ├── detection_service.py # 탐지 서비스
│   ├── training_service.py  # 학습 서비스
│   └── memory_service.py    # 메모리 관리
├── utils/                  # 유틸리티 함수들
│   ├── file_utils.py       # 파일 처리
│   ├── video_utils.py      # 비디오 처리
│   └── visualization_utils.py # 시각화
├── uploads/                # 업로드된 비디오 저장 폴더
└── custom_training/        # 커스텀 모델 학습 데이터
```

## 🔧 개발자 정보

### API 엔드포인트

#### **기본 기능**
- `GET /`: 메인 페이지 서빙
- `GET /<path:filename>`: 정적 파일 서빙
- `POST /api/upload_video`: 비디오 업로드
- `GET /api/status`: 시스템 상태 확인
- `POST /api/init_models`: 모델 초기화
- `POST /api/switch_model`: 모델 전환 (detector/classifier)

#### **라벨링 및 학습**
- `POST /api/auto_label`: AI 자동 라벨링 실행
- `POST /api/train_custom_model`: 커스텀 YOLO 모델 학습
- `POST /api/load_custom_model`: 학습된 모델 로드

#### **🧠 하이브리드 모델 패턴 학습**
- `POST /api/learn_patterns`: 수동 라벨링에서 패턴 학습 (YOLO+DINOv2 전용)
- `GET /api/get_patterns_info`: 학습된 패턴 정보 조회

#### **비디오 생성 및 다운로드**
- `POST /api/create_annotated_video`: 바운딩 박스 비디오 생성
- `POST /api/create_custom_model_video`: 커스텀 모델 비디오 생성
- `GET /api/download/<filename>`: 파일 다운로드

#### **메모리 관리**
- `GET /api/memory_status`: 메모리 상태 조회
- `POST /api/optimize_memory`: 메모리 최적화 실행

### 확장 가능성

1. **✅ 구현 완료**:
   - **모듈화된 아키텍처**: 서비스, 모델, 유틸리티 분리
   - **하이브리드 탐지 시스템**: YOLOv11 + DINOv2, YOLOv11 + CLIP
   - **범용 DINOv2 패턴 학습**: 사용자 정의 라벨 지능형 분류
   - **커스텀 YOLO 모델 학습**: 사용자 데이터로 fine-tuning
   - **메모리 관리 서비스**: 자동 메모리 최적화
   - **모델 전환 시스템**: 런타임 모델 변경 가능

2. **🚀 향후 개발 계획**:
   - **지능적 데이터 증강**: DINOv2로 유사 샘플 자동 발견
   - **실시간 처리**: 웹캠 실시간 라벨링 + 하이브리드 분류
   - **데이터베이스 연동**: 학습된 패턴 영구 저장 및 공유
   - **다중 사용자**: 협업 라벨링 + 패턴 공유 기능
   - **고급 모델**: SAM, GroundingDINO 등 최신 모델 통합

## 🐛 문제 해결

### 일반적인 문제들

1. **"서버에 연결할 수 없습니다"**
   - Python 백엔드 서버가 실행중인지 확인
   - `python run_server.py` 또는 `python app_modular.py` 실행

2. **"AI 모델 로딩 중..."**
   - 첫 실행시 YOLOv11, DINOv2 모델 다운로드로 시간 소요
   - 인터넷 연결 및 충분한 저장공간 확인 (5GB+ 권장)

3. **메모리 부족 오류**
   - 시스템 RAM 확인 (8GB+ 권장, 16GB 최적)
   - `/api/optimize_memory` 엔드포인트로 메모리 정리
   - 다른 프로그램 종료

4. **모델 전환 실패**
   - 사용 가능한 모델 목록: `yolo_dinov2`, `yolo_clip`
   - `/api/status`로 현재 모델 상태 확인

5. **CUDA 오류**
   - GPU 드라이버 및 CUDA 설치 확인
   - CPU 모드로도 동작 가능 (느림)
   - PyTorch CUDA 호환성 확인

## 📄 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🙏 기여

버그 리포트, 기능 제안, 코드 기여를 환영합니다! 