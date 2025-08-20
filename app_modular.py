"""
Video Labeling Tool - 모듈화된 Flask 애플리케이션
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

# 모듈화된 컴포넌트 import
from services import ModelManager, DetectionService, TrainingService, MemoryService
from utils import VisualizationUtils, FileUtils
from config.settings import UPLOAD_FOLDER, DEBUG, HOST, PORT

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 서비스 초기화
memory_service = MemoryService()
model_manager = ModelManager()
detection_service = DetectionService(model_manager, memory_service)
training_service = TrainingService(model_manager)

# 전역 상태
current_custom_model_path = None
training_data_accumulator = []

# 시작 시 모델 초기화
print("🚀 Initializing AI models...")
init_result = model_manager.initialize_models()
if init_result['detector_loaded'] and init_result['classifier_loaded']:
    print("✅ Models ready!")
elif init_result['errors']:
    print(f"⚠️  Partial loading: {', '.join(init_result['errors'][:2])}")
else:
    print("❌ Model loading failed")

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
        data = request.get_json() or {}
        detector_type = data.get('detector_type')
        classifier_type = data.get('classifier_type')
        
        result = model_manager.initialize_models(detector_type, classifier_type)
        
        if result['detector_loaded'] or result['classifier_loaded']:
            return jsonify({
                "status": "success", 
                "message": "Models initialized",
                "detector_loaded": result['detector_loaded'],
                "classifier_loaded": result['classifier_loaded'],
                "errors": result['errors']
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to load models",
                "errors": result['errors']
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """모델 전환"""
    try:
        data = request.get_json()
        model_type = data.get('model_type')  # 'detector' or 'classifier'
        model_name = data.get('model_name')
        model_config = data.get('config', {})
        
        if model_type == 'detector':
            success = model_manager.switch_detector(model_name, **model_config)
        elif model_type == 'classifier':
            success = model_manager.switch_classifier(model_name, **model_config)
        else:
            return jsonify({"error": "Invalid model_type. Use 'detector' or 'classifier'"}), 400
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Switched to {model_type}: {model_name}",
                "model_status": model_manager.get_model_status()
            })
        else:
            return jsonify({"error": f"Failed to switch {model_type} to {model_name}"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """비디오 업로드"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # 파일 저장
        filename = FileUtils.secure_filename(file.filename or 'uploaded_video.mp4')
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
        
        # 신뢰도 임계값 설정
        model_manager.set_detector_confidence(confidence_threshold)
        
        # 자동 라벨링 수행
        annotations = detection_service.process_video(video_path, dense_analysis)
        
        return jsonify({
            "status": "success",
            "annotations": annotations,
            "total_detections": len(annotations),
            "confidence_threshold": confidence_threshold,
            "dense_analysis": dense_analysis,
            "model_info": model_manager.get_model_status()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """시스템 상태 확인"""
    global current_custom_model_path, training_data_accumulator
    
    model_status = model_manager.get_model_status()
    memory_status = memory_service.get_memory_status()
    
    return jsonify({
        "models": model_status,
        "memory": memory_status,
        "custom_model_path": current_custom_model_path,
        "accumulated_training_data": len(training_data_accumulator),
        "available_models": model_manager.get_available_models()
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
        success = VisualizationUtils.create_annotated_video(video_path, annotations, output_path)
        
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
        
        # 수동 라벨링 데이터만 필터링
        manual_annotations = [ann for ann in annotations if ann.get('source') == 'manual' or ann.get('source') is None]
        
        if not manual_annotations:
            return jsonify({"error": "No manual annotations found for training"}), 400
        
        # 학습 실행
        global training_data_accumulator, current_custom_model_path
        result = training_service.train_model(video_path, annotations, epochs, training_data_accumulator)
        
        if result['success']:
            # 누적 데이터 업데이트
            training_data_accumulator.extend(result.get('current_annotations', []))
            
            # 커스텀 모델 자동 로드
            model_path = result['model_path']
            model_manager.load_custom_detector_model(model_path)
            current_custom_model_path = model_path
            
            # JSON 직렬화 안전한 응답 생성
            response_data = {
                "status": "success",
                "message": "Custom model training completed",
                "model_path": model_path,
                "training_info": result.get('training_info', {}),
                "images_count": result.get('images_count', 0),
                "total_annotations": result.get('total_annotations', 0),
                "classes": result.get('classes', [])
            }
            
            return jsonify(response_data)
        else:
            return jsonify({"error": result['error']}), 500
            
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
        
        success = model_manager.load_custom_detector_model(model_path)
        
        if success:
            global current_custom_model_path
            current_custom_model_path = model_path
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
        dense_analysis = data.get('dense_analysis', True)
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 400
        
        detector = model_manager.get_detector()
        if not detector or not detector.is_loaded():
            return jsonify({"error": "No detector loaded"}), 400
        
        # 현재 모델로 전체 비디오 자동 라벨링
        model_type = model_manager.detector_type
        print(f"Creating annotated video with {model_type} model (dense_analysis: {dense_analysis})...")
        annotations = detection_service.process_video(video_path, dense_analysis)
        
        if not annotations:
            return jsonify({"error": "No objects detected in video"}), 400
        
        # 출력 파일 경로 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        model_type = model_manager.detector_type
        output_filename = f"{video_name}_{model_type}_model.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # 어노테이션된 비디오 생성
        success = VisualizationUtils.create_annotated_video(video_path, annotations, output_path)
        
        if success:
            # 모델 타입에 따른 메시지 생성
            model_descriptions = {
                'yolo_clip': 'CLIP 불량품 탐지',
                'yolo_dinov2': '커스텀 DINOv2',
                'yolo': '기본 YOLO'
            }
            model_desc = model_descriptions.get(model_type, model_type.upper())
            
            return jsonify({
                "status": "success",
                "message": f"{model_desc} 모델 비디오가 성공적으로 생성되었습니다",
                "output_path": output_path,
                "download_url": f"/api/download/{output_filename}",
                "annotations": annotations,
                "total_detections": len(annotations),
                "model_type": model_type
            })
        else:
            return jsonify({"error": "Failed to create annotated video"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/learn_patterns', methods=['POST'])
def learn_patterns():
    """수동 라벨링 데이터에서 패턴 학습 (YOLO + DINOv2 하이브리드 전용)"""
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
        
        # 하이브리드 탐지기 확인
        detector = model_manager.get_detector()
        if not detector:
            return jsonify({"error": "No detector loaded"}), 400
        
        # YOLO + DINOv2 하이브리드인지 확인
        if hasattr(detector, 'learn_patterns'):
            # 패턴 학습 실행
            success = detector.learn_patterns(video_path, manual_annotations)
            
            if success:
                # 학습된 패턴 정보 가져오기
                learned_info = detector.get_learned_patterns_info()
                
                return jsonify({
                    "status": "success",
                    "message": "Patterns learned successfully",
                    "learned_labels": learned_info,
                    "manual_annotations_count": len(manual_annotations),
                    "detector_type": model_manager.detector_type
                })
            else:
                return jsonify({"error": "Failed to learn patterns"}), 500
        else:
            return jsonify({"error": "Current detector does not support pattern learning. Use 'yolo_dinov2' detector."}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_patterns_info', methods=['GET'])
def get_patterns_info():
    """학습된 패턴 정보 조회 (YOLO + DINOv2 하이브리드 전용)"""
    try:
        detector = model_manager.get_detector()
        if not detector:
            return jsonify({"error": "No detector loaded"}), 400
        
        # YOLO + DINOv2 하이브리드인지 확인
        if hasattr(detector, 'get_learned_patterns_info'):
            learned_info = detector.get_learned_patterns_info()
            
            return jsonify({
                "status": "success",
                "learned_labels": learned_info,
                "total_labels": len(learned_info),
                "total_samples": sum(info.get('sample_count', 0) for info in learned_info.values()),
                "has_clustering": any(info.get('has_clustering', False) for info in learned_info.values()),
                "detector_type": model_manager.detector_type
            })
        else:
            return jsonify({"error": "Current detector does not support pattern learning. Use 'yolo_dinov2' detector."}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/memory_status', methods=['GET'])
def get_memory_status():
    """메모리 상태 반환"""
    try:
        return jsonify(memory_service.get_memory_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize_memory', methods=['POST'])
def optimize_memory():
    """메모리 최적화 실행"""
    try:
        force_cleanup = request.get_json().get('force', False)
        
        if force_cleanup:
            result = memory_service.force_cleanup()
        else:
            result = memory_service.optimize_memory()
        
        return jsonify({
            "status": "success",
            "message": "Memory optimization completed",
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🎬 Video Labeling Server Starting...")
    
    # 모델 상태 확인
    detector = model_manager.get_detector()
    classifier = model_manager.get_classifier()
    
    if detector and detector.is_loaded():
        print(f"🎯 {model_manager.detector_type}")
    if classifier and classifier.is_loaded():
        print(f"🧠 {model_manager.classifier_type}")
    
    print(f"🌐 http://localhost:{PORT}")
    print("-" * 40)
    app.run(debug=DEBUG, host=HOST, port=PORT)
