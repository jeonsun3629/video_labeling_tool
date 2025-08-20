// HTML 요소들을 변수에 할당
const videoUpload = document.getElementById('videoUpload');
const videoPlayer = document.getElementById('videoPlayer');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const prevFrameBtn = document.getElementById('prevFrameBtn');
const nextFrameBtn = document.getElementById('nextFrameBtn');
const frameInfo = document.getElementById('frameInfo');

// 비디오 타임라인 컨트롤 요소들
const videoSlider = document.getElementById('videoSlider');
const playPauseBtn = document.getElementById('playPauseBtn');
const currentTimeSpan = document.getElementById('currentTime');
const totalDurationSpan = document.getElementById('totalDuration');
const speedSelect = document.getElementById('speedSelect');

// 1단계: 수동 라벨링 요소들
const labelText = document.getElementById('labelText');
const saveAnnotationBtn = document.getElementById('saveAnnotationBtn');
const manualStats = document.getElementById('manualStats');

// 2단계: 모델 학습 요소들
const trainModelBtn = document.getElementById('trainModelBtn');
const epochsInput = document.getElementById('epochsInput');
const trainingStatus = document.getElementById('trainingStatus');

// DINOv2 패턴 학습 요소들
const learnDinov2PatternsBtn = document.getElementById('learnDinov2PatternsBtn');
const getDinov2InfoBtn = document.getElementById('getDinov2InfoBtn');
const dinov2Status = document.getElementById('dinov2Status');
const dinov2PatternsInfo = document.getElementById('dinov2PatternsInfo');

// 3단계: AI 자동 라벨링 비디오 생성 요소들
const createBaseVideoBtn = document.getElementById('createBaseVideoBtn');
const createCustomVideoBtn = document.getElementById('createCustomVideoBtn');
const videoGenerationStatus = document.getElementById('videoGenerationStatus');
const denseAnalysisCheckbox = document.getElementById('denseAnalysisCheckbox');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const autoStats = document.getElementById('autoStats');

// 4단계: 데이터 검증 요소들
const exportBtn = document.getElementById('exportBtn');
const clearDataBtn = document.getElementById('clearDataBtn');
const annotationsOutput = document.getElementById('annotationsOutput');
const manualCount = document.getElementById('manualCount');
const autoCount = document.getElementById('autoCount');
const totalCount = document.getElementById('totalCount');

// 시스템 상태 요소들
const systemStatus = document.getElementById('systemStatus');
const statusMessage = document.getElementById('statusMessage');
const currentModelSpan = document.getElementById('currentModel');

// 모델 선택 요소들
const modelCards = document.querySelectorAll('.model-card');
const switchModelBtn = document.getElementById('switchModelBtn');
const updateClipSettingsBtn = document.getElementById('updateClipSettingsBtn');
const defectQueriesText = document.getElementById('defectQueriesText');
const defectThresholdSlider = document.getElementById('defectThresholdSlider');
const defectThresholdValue = document.getElementById('defectThresholdValue');

// 상태 변수 초기화
let annotations = []; // 모든 라벨링 데이터를 저장할 배열
let currentRect = {}; // 현재 그리고 있는 사각형 정보
let isDrawing = false; // 현재 그림을 그리고 있는지 여부
let currentVideoFile = null; // 현재 업로드된 비디오 파일
let currentVideoPath = null; // 서버에 업로드된 비디오 경로
let isCustomModelTrained = false; // 커스텀 모델 학습 완료 여부
let isSliderDragging = false; // 슬라이더를 드래그 중인지 여부
let currentModelType = 'yolo_dinov2'; // 현재 선택된 모델 타입
let selectedModelType = 'yolo_dinov2'; // 사용자가 선택한 모델 타입

// 워크플로우 상태 관리 (4단계로 변경)
const workflowState = {
    step1_manual: false,    // 수동 라벨링 완료
    step2_training: false,  // 모델 학습 완료
    step3_video: false,     // AI 자동 라벨링 비디오 생성 완료
    step4_data: false       // 데이터 검증 완료
};

// =================
// 워크플로우 진행 상태 관리
// =================

function updateWorkflowProgress() {
    // 각 단계별 진행 상태 업데이트
    updateStepStatus('progress-step1', workflowState.step1_manual);
    updateStepStatus('progress-step2', workflowState.step2_training);
    updateStepStatus('progress-step3', workflowState.step3_video);
    updateStepStatus('progress-step4', workflowState.step4_data);
    
    // 섹션별 활성화 상태 업데이트
    updateSectionActivation();
    
    // 버튼 활성화 상태 업데이트
    updateButtonStates();
}

function updateStepStatus(stepId, isCompleted) {
    const stepElement = document.getElementById(stepId);
    const statusElement = stepElement.querySelector('.step-status');
    
    stepElement.classList.remove('active', 'completed');
    
    if (isCompleted) {
        stepElement.classList.add('completed');
        statusElement.textContent = '완료';
    } else {
        // 현재 활성 단계 결정
        const currentStep = getCurrentActiveStep();
        if (stepId === `progress-${currentStep}`) {
            stepElement.classList.add('active');
            statusElement.textContent = '진행 중';
        } else {
            statusElement.textContent = '대기';
        }
    }
}

function getCurrentActiveStep() {
    if (!workflowState.step1_manual) return 'step1';
    if (!workflowState.step2_training) return 'step2';
    if (!workflowState.step3_video) return 'step3';
    if (!workflowState.step4_data) return 'step4';
    return 'step4'; // 모든 단계 완료
}

function updateSectionActivation() {
    // 각 섹션별 활성화 상태 CSS 클래스 적용
    const sections = ['step1-section', 'step2-section', 'step3-section', 'step4-section'];
    const currentStep = getCurrentActiveStep();
    
    sections.forEach((sectionId, index) => {
        const section = document.getElementById(sectionId);
        const stepNum = index + 1;
        const stepKey = `step${stepNum}`;
        
        section.classList.remove('active', 'completed');
        
        if (workflowState[`step${stepNum}_${stepKey.slice(4)}`] || 
            (stepNum === 1 && workflowState.step1_manual) ||
            (stepNum === 2 && workflowState.step2_training) ||
            (stepNum === 3 && workflowState.step3_video) ||
            (stepNum === 4 && workflowState.step4_data)) {
            section.classList.add('completed');
        } else if (currentStep === stepKey) {
            section.classList.add('active');
        }
    });
}

function updateButtonStates() {
    // 1단계: 수동 라벨링은 항상 가능
    saveAnnotationBtn.disabled = !currentVideoPath || !labelText.value.trim();
    
    // 2단계: 모델 학습 - 수동 라벨링 데이터가 있어야 함
    const manualAnnotations = annotations.filter(ann => ann.source === 'manual' || !ann.source);
    trainModelBtn.disabled = manualAnnotations.length === 0;
    
    // DINOv2 패턴 학습 버튼들
    if (learnDinov2PatternsBtn) {
        learnDinov2PatternsBtn.disabled = !currentVideoPath || manualAnnotations.length === 0;
    }
    if (getDinov2InfoBtn) {
        getDinov2InfoBtn.disabled = false; // 정보 조회는 항상 가능
    }
    
    // 3단계: AI 자동 라벨링 비디오 생성 - 비디오가 업로드되어야 함
    createBaseVideoBtn.disabled = !currentVideoPath;
    
    // 커스텀 비디오 버튼: 모델 타입에 따라 다른 조건
    if (currentModelType === 'yolo_clip') {
        // YOLO+CLIP 모델: YOLO 학습 없이도 CLIP으로 불량품 탐지 가능
        createCustomVideoBtn.disabled = !currentVideoPath;
        createCustomVideoBtn.textContent = '🎯 CLIP 불량품 탐지 비디오 생성';
    } else {
        // YOLO+DINOv2 모델: 커스텀 학습이 필요함
        createCustomVideoBtn.disabled = !currentVideoPath || !isCustomModelTrained;
        createCustomVideoBtn.textContent = '🎯 커스텀 객체 탐지 비디오 생성';
    }
    
    // 4단계: 데이터 내보내기 - 데이터가 있어야 함
    exportBtn.disabled = annotations.length === 0;
}

function updateDataStatistics() {
    const manualAnnotations = annotations.filter(ann => ann.source === 'manual' || !ann.source);
    const autoAnnotations = annotations.filter(ann => ann.source === 'auto' || ann.source === 'custom_model');
    
    // 라벨별 통계 계산
    const labelStats = calculateLabelStatistics(manualAnnotations);
    
    // 개별 통계 업데이트 (라벨별 상세 정보 포함)
    updateManualStatsDisplay(manualAnnotations.length, labelStats);
    autoStats.textContent = `자동 라벨링 데이터: ${autoAnnotations.length}개`;
    
    // 요약 통계 업데이트
    manualCount.textContent = `${manualAnnotations.length}개`;
    autoCount.textContent = `${autoAnnotations.length}개`;
    totalCount.textContent = `${annotations.length}개`;
    
    // 워크플로우 상태 업데이트
    workflowState.step1_manual = manualAnnotations.length > 0;
    // step2_training은 학습 완료 시에만 true로 설정 (별도 함수에서 처리)
    // step3_video는 비디오 생성 완료 시에만 true로 설정 (별도 함수에서 처리)  
    workflowState.step4_data = autoAnnotations.length > 0; // 자동 라벨링 데이터가 추가될 때 완료
    
    updateWorkflowProgress();
}

function calculateLabelStatistics(manualAnnotations) {
    const labelCounts = {};
    
    manualAnnotations.forEach(ann => {
        const label = ann.label;
        if (labelCounts[label]) {
            labelCounts[label]++;
        } else {
            labelCounts[label] = 1;
        }
    });
    
    return labelCounts;
}

function updateManualStatsDisplay(totalCount, labelStats) {
    if (totalCount === 0) {
        manualStats.innerHTML = `
            <div class="stats-header">수동 라벨링 데이터: 0개</div>
            <div class="stats-detail">아직 라벨링된 데이터가 없습니다.</div>
        `;
        return;
    }
    
    // 라벨별 통계 HTML 생성
    const labelStatsHTML = Object.entries(labelStats)
        .sort(([,a], [,b]) => b - a) // 개수 순으로 정렬
        .map(([label, count]) => {
            const percentage = ((count / totalCount) * 100).toFixed(1);
            return `
                <div class="label-stat-item">
                    <span class="label-name">${label}</span>
                    <span class="label-count">${count}개 (${percentage}%)</span>
                </div>
            `;
        }).join('');
    
    manualStats.innerHTML = `
        <div class="stats-header">수동 라벨링 데이터: ${totalCount}개</div>
        <div class="stats-detail">
            <div class="label-stats-container">
                ${labelStatsHTML}
            </div>
        </div>
    `;
}

// =================
// 1단계: 동영상 업로드 및 수동 라벨링
// =================

videoUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        currentVideoFile = file;
        const url = URL.createObjectURL(file);
        videoPlayer.src = url;
        
        // 서버에 비디오 업로드
        await uploadVideoToServer(file);
    }
});

videoPlayer.addEventListener('loadedmetadata', () => {
    console.log(`Video loaded: ${videoPlayer.videoWidth}x${videoPlayer.videoHeight}`);
    // 짧은 지연 후 캔버스 크기 조정 (브라우저가 비디오 크기를 완전히 계산할 시간 제공)
    setTimeout(() => {
        resizeCanvasToMatchVideo();
    }, 100);
});

videoPlayer.addEventListener('canplay', () => {
    videoPlayer.pause();
    updateFrameInfo();
    updateTimelineDisplay();
    // canplay 시점에도 다시 한번 캔버스 크기 확인
    resizeCanvasToMatchVideo();
});

videoPlayer.addEventListener('resize', () => {
    resizeCanvasToMatchVideo();
});

window.addEventListener('resize', () => {
    if (videoPlayer.videoWidth && videoPlayer.videoHeight) {
        resizeCanvasToMatchVideo();
    }
});

function resizeCanvasToMatchVideo() {
    if (!videoPlayer.videoWidth || !videoPlayer.videoHeight) {
        return; // 비디오가 아직 로드되지 않음
    }
    
    const containerWidth = videoPlayer.offsetWidth;
    const containerHeight = videoPlayer.offsetHeight;
    
    // 비디오의 원본 aspect ratio 계산
    const videoAspectRatio = videoPlayer.videoWidth / videoPlayer.videoHeight;
    const containerAspectRatio = containerWidth / containerHeight;
    
    let actualVideoWidth, actualVideoHeight;
    let offsetX = 0, offsetY = 0;
    
    // 비디오가 실제로 표시되는 크기와 위치 계산 (letterbox/pillarbox 고려)
    if (videoAspectRatio > containerAspectRatio) {
        // 비디오가 더 넓음 -> 상하에 letterbox
        actualVideoWidth = containerWidth;
        actualVideoHeight = containerWidth / videoAspectRatio;
        offsetX = 0;
        offsetY = (containerHeight - actualVideoHeight) / 2;
    } else {
        // 비디오가 더 높음 -> 좌우에 pillarbox
        actualVideoWidth = containerHeight * videoAspectRatio;
        actualVideoHeight = containerHeight;
        offsetX = (containerWidth - actualVideoWidth) / 2;
        offsetY = 0;
    }
    
    // 캔버스를 비디오의 실제 표시 영역과 정확히 일치시키기
    canvas.width = actualVideoWidth;
    canvas.height = actualVideoHeight;
    canvas.style.width = actualVideoWidth + 'px';
    canvas.style.height = actualVideoHeight + 'px';
    canvas.style.left = offsetX + 'px';
    canvas.style.top = offsetY + 'px';
    
    // 전역 변수로 저장하여 좌표 변환시 사용
    window.videoDisplayInfo = {
        actualWidth: actualVideoWidth,
        actualHeight: actualVideoHeight,
        offsetX: offsetX,
        offsetY: offsetY,
        scaleX: videoPlayer.videoWidth / actualVideoWidth,
        scaleY: videoPlayer.videoHeight / actualVideoHeight
    };
    
    console.log(`Video display info:`, window.videoDisplayInfo);
    console.log(`Canvas resized to: ${actualVideoWidth}x${actualVideoHeight} at (${offsetX}, ${offsetY})`);
}

function getRelativeCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    
    // 캔버스 내부 좌표 계산 (픽셀 좌표)
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    
    // 캔버스 크기 대비 비율로 정규화
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: canvasX * scaleX,
        y: canvasY * scaleY
    };
}

// 캔버스 드래그 이벤트
canvas.addEventListener('mousedown', (e) => {
    const relativeCoords = getRelativeCoordinates(e);
    currentRect.startX = relativeCoords.x;
    currentRect.startY = relativeCoords.y;
    isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    
    const relativeCoords = getRelativeCoordinates(e);
    const endX = relativeCoords.x;
    const endY = relativeCoords.y;

    drawCurrentFrameAnnotations();
    
    ctx.strokeStyle = '#f39c12'; // 대기 중 색상
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(currentRect.startX, currentRect.startY, endX - currentRect.startX, endY - currentRect.startY);
    ctx.setLineDash([]);
});

canvas.addEventListener('mouseup', (e) => {
    isDrawing = false;
    const relativeCoords = getRelativeCoordinates(e);
    currentRect.endX = relativeCoords.x;
    currentRect.endY = relativeCoords.y;
    
    currentRect.x = Math.min(currentRect.startX, currentRect.endX);
    currentRect.y = Math.min(currentRect.startY, currentRect.endY);
    currentRect.width = Math.abs(currentRect.endX - currentRect.startX);
    currentRect.height = Math.abs(currentRect.endY - currentRect.startY);

    drawCurrentFrameAnnotations();
    
    if (currentRect.width > 5 && currentRect.height > 5) {
        ctx.strokeStyle = '#f39c12';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(currentRect.x, currentRect.y, currentRect.width, currentRect.height);
        
        // 좌표 정보 표시
        ctx.fillStyle = '#f39c12';
        ctx.font = '14px Arial';
        
        if (window.videoDisplayInfo) {
            const { scaleX, scaleY } = window.videoDisplayInfo;
            const originalX = Math.round(currentRect.x * scaleX);
            const originalY = Math.round(currentRect.y * scaleY);
            const originalW = Math.round(currentRect.width * scaleX);
            const originalH = Math.round(currentRect.height * scaleY);
            
            ctx.fillText(`라벨 입력 후 저장 [${originalX},${originalY},${originalW},${originalH}]`, currentRect.x, currentRect.y - 5);
        } else {
            ctx.fillText('라벨을 입력하고 저장하세요', currentRect.x, currentRect.y - 5);
        }
    }

    // 저장 버튼 활성화 상태 업데이트
    updateButtonStates();
});

// 라벨 입력 실시간 검증
labelText.addEventListener('input', () => {
    updateButtonStates();
});

// 라벨링 데이터 저장
saveAnnotationBtn.addEventListener('click', () => {
    if (!labelText.value.trim()) {
        alert('라벨을 입력해주세요.');
        return;
    }
    if (!currentRect.width || !currentRect.height) {
        alert('먼저 바운딩 박스를 그려주세요.');
        return;
    }

    // 정확한 좌표 변환을 위해 저장된 비디오 표시 정보 사용
    if (!window.videoDisplayInfo) {
        alert('비디오 표시 정보를 찾을 수 없습니다. 페이지를 새로고침 해주세요.');
        return;
    }
    
    const { scaleX, scaleY } = window.videoDisplayInfo;
    
    // 캔버스 좌표를 원본 비디오 좌표로 변환
    const originalX = currentRect.x * scaleX;
    const originalY = currentRect.y * scaleY;
    const originalWidth = currentRect.width * scaleX;
    const originalHeight = currentRect.height * scaleY;

    const annotation = {
        frame: videoPlayer.currentTime.toFixed(3),
        label: labelText.value.trim(),
        bbox: [
            Math.round(originalX),
            Math.round(originalY),
            Math.round(originalWidth),
            Math.round(originalHeight)
        ],
        video_resolution: {
            width: videoPlayer.videoWidth,
            height: videoPlayer.videoHeight
        },
        display_resolution: {
            width: canvas.width,
            height: canvas.height
        },
        source: 'manual'
    };

    annotations.push(annotation);
    updateAnnotationsOutput();
    updateDataStatistics();
    
    // 입력 필드와 박스 초기화
    labelText.value = '';
    currentRect = {};
    drawCurrentFrameAnnotations();
    
    console.log('수동 라벨링 저장:', annotation);
});

// =================
// 2단계: 커스텀 모델 학습
// =================

trainModelBtn.addEventListener('click', async () => {
    await trainCustomModel();
});

// DINOv2 패턴 학습 이벤트 핸들러
if (learnDinov2PatternsBtn) {
    learnDinov2PatternsBtn.addEventListener('click', async () => {
        await learnDinov2Patterns();
    });
}

if (getDinov2InfoBtn) {
    getDinov2InfoBtn.addEventListener('click', async () => {
        await getDinov2PatternsInfo();
    });
}



async function trainCustomModel() {
    if (!currentVideoPath) {
        alert('먼저 비디오를 업로드해주세요.');
        return;
    }
    
    const manualAnnotations = annotations.filter(ann => ann.source === 'manual' || !ann.source);
    if (manualAnnotations.length === 0) {
        alert('수동 라벨링 데이터가 없습니다. 먼저 수동으로 몇 개의 객체를 라벨링해주세요.');
        return;
    }
    
    const epochs = parseInt(epochsInput.value);
    if (epochs < 10 || epochs > 200) {
        alert('학습 횟수는 10~200 사이로 설정해주세요.');
        return;
    }
    
    try {
        updateTrainingStatus(`🚀 커스텀 모델 학습 시작...\n- 학습 데이터: ${manualAnnotations.length}개\n- 학습 횟수: ${epochs} epochs`, 'info');
        trainModelBtn.disabled = true;
        
        const response = await fetch('http://localhost:5000/api/train_custom_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_path: currentVideoPath,
                annotations: annotations,
                epochs: epochs
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isCustomModelTrained = true;
            workflowState.step2_training = true;
            
            updateTrainingStatus(
                `✅ 모델 학습 완료!\n` +
                `- 학습 이미지: ${data.images_count}개\n` + 
                `- 총 누적 데이터: ${data.total_annotations}개\n` +
                `- 클래스: ${data.classes.join(', ')}\n` +
                `🚀 커스텀 모델이 자동으로 활성화되었습니다!`, 
                'success'
            );
            
            updateWorkflowProgress();
            await checkServerStatus();
            
        } else {
            updateTrainingStatus('❌ 모델 학습 실패: ' + data.error, 'error');
        }
    } catch (error) {
        updateTrainingStatus('❌ 학습 오류: ' + error.message, 'error');
        console.error('Training error:', error);
    } finally {
        trainModelBtn.disabled = false;
    }
}

async function learnDinov2Patterns() {
    // DINOv2 패턴 학습 실행
    if (!currentVideoPath) {
        alert('먼저 비디오를 업로드해주세요.');
        return;
    }
    
    const manualAnnotations = annotations.filter(ann => ann.source === 'manual' || !ann.source);
    if (manualAnnotations.length === 0) {
        alert('수동 라벨링 데이터가 없습니다. 먼저 수동으로 몇 개의 객체를 라벨링해주세요.');
        return;
    }
    
    try {
        updateDinov2Status(`🧠 DINOv2 패턴 학습 시작...\n- 학습 데이터: ${manualAnnotations.length}개\n- 라벨 종류: ${new Set(manualAnnotations.map(ann => ann.label)).size}개`, 'info');
        
        learnDinov2PatternsBtn.disabled = true;
        
        const response = await fetch('http://localhost:5000/api/learn_dinov2_patterns', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_path: currentVideoPath,
                annotations: annotations
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateDinov2Status(
                `✅ DINOv2 패턴 학습 완료!\n` +
                `- 학습된 라벨: ${Object.keys(data.learned_labels).length}개\n` +
                `- 총 샘플: ${Object.values(data.learned_labels).reduce((sum, info) => sum + info.sample_count, 0)}개\n` +
                `🎯 이제 자동 라벨링에서 더 정확한 분류가 가능합니다!`,
                'success'
            );
            
            // 학습된 패턴 정보 표시
            displayDinov2PatternsInfo(data.learned_labels);
            
        } else {
            updateDinov2Status('❌ DINOv2 패턴 학습 실패: ' + data.error, 'error');
        }
    } catch (error) {
        updateDinov2Status('❌ 패턴 학습 오류: ' + error.message, 'error');
        console.error('DINOv2 learning error:', error);
    } finally {
        learnDinov2PatternsBtn.disabled = false;
        updateButtonStates();
    }
}

async function getDinov2PatternsInfo() {
    // 학습된 DINOv2 패턴 정보 조회
    try {
        const response = await fetch('http://localhost:5000/api/get_dinov2_patterns_info');
        const data = await response.json();
        
        if (data.status === 'success') {
            if (data.total_labels === 0) {
                updateDinov2Status('📊 아직 학습된 패턴이 없습니다.\n먼저 "DINOv2 패턴 학습"을 실행해주세요.', 'info');
                dinov2PatternsInfo.innerHTML = '';
            } else {
                updateDinov2Status(
                    `📊 학습된 패턴 정보:\n` +
                    `- 총 라벨: ${data.total_labels}개\n` +
                    `- 총 샘플: ${data.total_samples}개\n` +
                    `- 클러스터링: ${data.has_clustering ? '활성화됨' : '비활성화됨'}`,
                    'success'
                );
                
                displayDinov2PatternsInfo(data.learned_labels);
            }
        } else {
            updateDinov2Status('❌ 패턴 정보 조회 실패: ' + data.error, 'error');
        }
    } catch (error) {
        updateDinov2Status('❌ 서버 연결 오류: ' + error.message, 'error');
        console.error('DINOv2 info error:', error);
    }
}

function displayDinov2PatternsInfo(learnedLabels) {
    // 학습된 패턴 정보를 시각적으로 표시
    if (!learnedLabels || Object.keys(learnedLabels).length === 0) {
        dinov2PatternsInfo.innerHTML = '<div class="no-patterns">학습된 패턴이 없습니다.</div>';
        return;
    }
    
    const labelEntries = Object.entries(learnedLabels).sort(([,a], [,b]) => b.sample_count - a.sample_count);
    
    const patternsHTML = labelEntries.map(([label, info]) => {
        const clusterInfo = info.has_clustering ? 
            `${info.cluster_count}개 클러스터` : 
            '단일 패턴';
        
        return `
            <div class="pattern-item">
                <div class="pattern-label">${label}</div>
                <div class="pattern-details">
                    <span class="pattern-samples">${info.sample_count}개 샘플</span>
                    <span class="pattern-clusters">${clusterInfo}</span>
                </div>
            </div>
        `;
    }).join('');
    
    dinov2PatternsInfo.innerHTML = `
        <div class="patterns-container">
            <h5>🎯 학습된 라벨별 패턴</h5>
            ${patternsHTML}
        </div>
    `;
}

function updateDinov2Status(message, type) {
    if (dinov2Status) {
        dinov2Status.textContent = message;
        dinov2Status.className = `status-display status-${type}`;
    }
}

// =================
// DINOv2 패턴 학습 함수들
// =================

async function learnDinov2Patterns() {
    // DINOv2 패턴 학습 실행
    if (!currentVideoPath) {
        alert('먼저 비디오를 업로드해주세요.');
        return;
    }
    
    const manualAnnotations = annotations.filter(ann => ann.source === 'manual' || !ann.source);
    if (manualAnnotations.length === 0) {
        alert('수동 라벨링 데이터가 없습니다. 먼저 수동으로 몇 개의 객체를 라벨링해주세요.');
        return;
    }
    
    try {
        updateDinov2Status(`🧠 DINOv2 커스텀 패턴 학습 시작...\n- 학습 데이터: ${manualAnnotations.length}개\n- 라벨 종류: ${new Set(manualAnnotations.map(ann => ann.label)).size}개\n- 모드: 커스텀 객체 전용`, 'info');
        
        learnDinov2PatternsBtn.disabled = true;
        
        const response = await fetch('http://localhost:5000/api/learn_patterns', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_path: currentVideoPath,
                annotations: annotations
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // 2단계 완료 표시
            workflowState.step2_training = true;
            
            updateDinov2Status(
                `✅ DINOv2 커스텀 패턴 학습 완료!\n` +
                `- 학습된 라벨: ${Object.keys(data.learned_labels || {}).length}개\n` +
                `- 총 샘플: ${Object.values(data.learned_labels || {}).reduce((sum, info) => sum + (info.sample_count || 0), 0)}개\n` +
                `🎯 이제 해당 객체들만 정확히 탐지할 수 있습니다!`,
                'success'
            );
            
            // 학습된 패턴 정보 표시
            displayDinov2PatternsInfo(data.learned_labels || {});
            
            // 워크플로우 진행 상태 업데이트
            updateWorkflowProgress();
            
        } else {
            updateDinov2Status('❌ DINOv2 패턴 학습 실패: ' + (data.error || '알 수 없는 오류'), 'error');
        }
    } catch (error) {
        updateDinov2Status('❌ 패턴 학습 오류: ' + error.message, 'error');
        console.error('DINOv2 learning error:', error);
    } finally {
        if (learnDinov2PatternsBtn) {
            learnDinov2PatternsBtn.disabled = false;
        }
        updateButtonStates();
    }
}

async function getDinov2PatternsInfo() {
    // 학습된 DINOv2 패턴 정보 조회
    try {
        const response = await fetch('http://localhost:5000/api/get_patterns_info');
        const data = await response.json();
        
        if (data.status === 'success') {
            if ((data.total_labels || 0) === 0) {
                updateDinov2Status('📊 아직 학습된 패턴이 없습니다.\n먼저 "DINOv2 패턴 학습"을 실행해주세요.', 'info');
                if (dinov2PatternsInfo) {
                    dinov2PatternsInfo.innerHTML = '';
                }
            } else {
                updateDinov2Status(
                    `📊 학습된 패턴 정보:\n` +
                    `- 총 라벨: ${data.total_labels || 0}개\n` +
                    `- 총 샘플: ${data.total_samples || 0}개\n` +
                    `- 모드: 커스텀 객체 전용`,
                    'success'
                );
                
                displayDinov2PatternsInfo(data.learned_labels || {});
            }
        } else {
            updateDinov2Status('❌ 패턴 정보 조회 실패: ' + (data.error || '알 수 없는 오류'), 'error');
        }
    } catch (error) {
        updateDinov2Status('❌ 서버 연결 오류: ' + error.message, 'error');
        console.error('DINOv2 info error:', error);
    }
}

function displayDinov2PatternsInfo(learnedLabels) {
    // 학습된 DINOv2 패턴 정보를 시각적으로 표시
    if (!dinov2PatternsInfo) return;
    
    if (!learnedLabels || Object.keys(learnedLabels).length === 0) {
        dinov2PatternsInfo.innerHTML = '<div class="no-patterns">학습된 패턴이 없습니다.</div>';
        return;
    }
    
    const labelEntries = Object.entries(learnedLabels).sort(([,a], [,b]) => (b.sample_count || 0) - (a.sample_count || 0));
    
    const patternsHTML = labelEntries.map(([label, info]) => {
        return `
            <div class="pattern-item">
                <div class="pattern-label">${label}</div>
                <div class="pattern-details">
                    <span class="pattern-samples">${info.sample_count || 0}개 샘플</span>
                    <span class="pattern-mode">커스텀 전용</span>
                </div>
            </div>
        `;
    }).join('');
    
    dinov2PatternsInfo.innerHTML = `
        <div class="patterns-container">
            <h5>🎯 학습된 커스텀 객체 패턴</h5>
            ${patternsHTML}
        </div>
    `;
}

function updateDinov2Status(message, type) {
    if (dinov2Status) {
        dinov2Status.textContent = message;
        dinov2Status.className = `status-display status-${type}`;
    }
}



// =================
// 3단계: AI 자동 라벨링 비디오 생성
// =================

createBaseVideoBtn.addEventListener('click', async () => {
    await createAILabeledVideo(false); // 기본 YOLO 모델 사용
});

createCustomVideoBtn.addEventListener('click', async () => {
    await createAILabeledVideo(true); // 커스텀 모델 사용
});

async function createAILabeledVideo(useCustomModel = false) {
    if (!currentVideoPath) {
        alert('먼저 비디오를 업로드해주세요.');
        return;
    }
    
    try {
        const modelType = useCustomModel ? '커스텀 모델' : '기본 YOLO 모델';
        updateVideoGenerationStatus(`🚀 ${modelType}로 AI 자동 라벨링 비디오 생성 중...\n전체 비디오를 분석하고 라벨링된 비디오를 생성합니다.`, 'info');
        
        createBaseVideoBtn.disabled = true;
        createCustomVideoBtn.disabled = true;
        
        const response = await fetch('http://localhost:5000/api/create_custom_model_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_path: currentVideoPath,
                dense_analysis: denseAnalysisCheckbox.checked,
                confidence_threshold: parseFloat(confidenceSlider.value),
                use_custom_model: useCustomModel
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateVideoGenerationStatus(
                `✅ ${modelType} 비디오 생성 완료!\n` +
                `- 탐지된 객체: ${data.total_detections}개\n` +
                `- 사용된 모델: ${modelType}\n` +
                `- 다운로드 중...`,
                'success'
            );
            
            // 자동 다운로드
            const downloadUrl = `http://localhost:5000${data.download_url}`;
            const downloadLink = document.createElement('a');
            downloadLink.href = downloadUrl;
            downloadLink.download = data.output_path.split('/').pop();
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            // 새로운 어노테이션 데이터 추가
            const sourceType = useCustomModel ? 'custom_model' : 'auto';
            const newAnnotations = data.annotations.map(ann => ({
                ...ann,
                source: sourceType
            }));
            annotations = [...annotations, ...newAnnotations];
            updateAnnotationsOutput();
            updateDataStatistics();
            
            workflowState.step3_video = true;
            updateWorkflowProgress();
            
            setTimeout(() => {
                updateVideoGenerationStatus(`🎉 ${modelType} 라벨링 비디오 다운로드 완료!\n탐지 데이터가 자동으로 추가되었습니다.`, 'success');
            }, 1000);
            
        } else {
            updateVideoGenerationStatus(`❌ ${modelType} 비디오 생성 실패: ${data.error}`, 'error');
        }
    } catch (error) {
        updateVideoGenerationStatus(`❌ 비디오 생성 오류: ${error.message}`, 'error');
        console.error('Video creation error:', error);
    } finally {
        createBaseVideoBtn.disabled = false;
        createCustomVideoBtn.disabled = false;
        updateButtonStates();
    }
}

// 신뢰도 슬라이더 이벤트
confidenceSlider.addEventListener('input', (e) => {
    confidenceValue.textContent = e.target.value;
});

// =================
// 4단계: 데이터 검증 및 내보내기
// =================

exportBtn.addEventListener('click', () => {
    if (annotations.length === 0) {
        alert('저장된 데이터가 없습니다.');
        return;
    }
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(annotations, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "annotations.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
    
    workflowState.step4_data = true;
    updateWorkflowProgress();
    
    console.log('데이터 내보내기 완료');
});

clearDataBtn.addEventListener('click', () => {
    if (confirm('모든 라벨링 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
        annotations = [];
        updateAnnotationsOutput();
        updateDataStatistics();
        
        // 워크플로우 상태 초기화
        workflowState.step1_manual = false;
        workflowState.step3_video = false;
        workflowState.step4_data = false;
        updateWorkflowProgress();
        
        console.log('모든 데이터 삭제됨');
    }
});



// =================
// 유틸리티 함수들
// =================

function updateFrameInfo() {
    const currentFrame = Math.round(videoPlayer.currentTime * 30);
    const totalFrames = Math.round(videoPlayer.duration * 30);
    frameInfo.textContent = `프레임: ${currentFrame} / ${totalFrames || 0}`;
    
    if (!isDrawing) {
        drawCurrentFrameAnnotations();
    }
}

// =================
// 비디오 타임라인 컨트롤 기능
// =================

function formatTime(seconds) {
    if (isNaN(seconds)) return '00:00';
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function updateTimelineDisplay() {
    if (!videoPlayer.duration) return;
    
    // 슬라이더가 드래그 중이 아닐 때만 업데이트
    if (!isSliderDragging) {
        const progress = (videoPlayer.currentTime / videoPlayer.duration) * 100;
        videoSlider.value = progress;
    }
    
    currentTimeSpan.textContent = formatTime(videoPlayer.currentTime);
    totalDurationSpan.textContent = formatTime(videoPlayer.duration);
}

function updatePlayPauseButton() {
    if (videoPlayer.paused) {
        playPauseBtn.textContent = '▶ 재생';
    } else {
        playPauseBtn.textContent = '⏸ 일시정지';
    }
}

// 비디오 타임라인 이벤트 핸들러들
videoSlider.addEventListener('mousedown', () => {
    isSliderDragging = true;
});

videoSlider.addEventListener('mouseup', () => {
    isSliderDragging = false;
});

videoSlider.addEventListener('input', () => {
    if (videoPlayer.duration) {
        const newTime = (videoSlider.value / 100) * videoPlayer.duration;
        videoPlayer.currentTime = newTime;
        updateFrameInfo();
        currentTimeSpan.textContent = formatTime(newTime);
    }
});

playPauseBtn.addEventListener('click', () => {
    if (videoPlayer.paused) {
        videoPlayer.play();
    } else {
        videoPlayer.pause();
    }
    updatePlayPauseButton();
});

speedSelect.addEventListener('change', () => {
    videoPlayer.playbackRate = parseFloat(speedSelect.value);
});

// 비디오 플레이어 이벤트에 타임라인 업데이트 추가
videoPlayer.addEventListener('timeupdate', () => {
    updateFrameInfo();
    updateTimelineDisplay();
});

videoPlayer.addEventListener('loadedmetadata', () => {
    videoSlider.max = 100;
    videoSlider.value = 0;
    updateTimelineDisplay();
    updatePlayPauseButton();
});

videoPlayer.addEventListener('play', updatePlayPauseButton);
videoPlayer.addEventListener('pause', updatePlayPauseButton);

videoPlayer.addEventListener('ended', () => {
    updatePlayPauseButton();
    videoSlider.value = 100;
});

function drawCurrentFrameAnnotations() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const currentTime = videoPlayer.currentTime;
    const tolerance = 0.1;
    
    const currentAnnotations = annotations.filter(ann => {
        const frameTime = parseFloat(ann.frame);
        return Math.abs(frameTime - currentTime) <= tolerance;
    });
    
    currentAnnotations.forEach((ann, index) => {
        const bbox = ann.bbox;
        
        // 정확한 좌표 변환을 위해 저장된 비디오 표시 정보 사용
        if (!window.videoDisplayInfo) {
            return; // 표시 정보가 없으면 스킵
        }
        
        const { scaleX, scaleY } = window.videoDisplayInfo;
        
        // 원본 비디오 좌표를 캔버스 좌표로 변환
        const x = bbox[0] / scaleX;
        const y = bbox[1] / scaleY;
        const width = bbox[2] / scaleX;
        const height = bbox[3] / scaleY;
        
        // 소스별 색상 설정
        let color;
        switch(ann.source) {
            case 'manual': color = '#e74c3c'; break;
            case 'auto': color = '#27ae60'; break;
            case 'custom_model': color = '#9b59b6'; break;
            default: color = '#e74c3c';
        }
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        
        ctx.fillStyle = color;
        ctx.font = '14px Arial';
        
        // 라벨과 좌표 정보 표시
        let labelText = ann.label;
        if (ann.confidence) {
            labelText += ` (${(ann.confidence * 100).toFixed(1)}%)`;
        }
        
        // 디버깅을 위한 원본 좌표 정보 추가
        const originalCoords = `[${bbox[0]},${bbox[1]},${bbox[2]},${bbox[3]}]`;
        labelText += ` ${originalCoords}`;
        
        ctx.fillText(labelText, x, y - 5);
    });
}

function updateAnnotationsOutput() {
    if (annotations.length === 0) {
        annotationsOutput.textContent = '데이터가 없습니다.';
    } else {
        annotationsOutput.textContent = JSON.stringify(annotations, null, 2);
    }
    drawCurrentFrameAnnotations();
}

async function uploadVideoToServer(file) {
    try {
        updateSystemStatus('비디오 업로드 중...', 'info');
        
        const formData = new FormData();
        formData.append('video', file);
        
        const response = await fetch('http://localhost:5000/api/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentVideoPath = data.video_path;
            updateSystemStatus('✅ 비디오 업로드 완료! 수동 라벨링을 시작할 수 있습니다.', 'success');
            updateButtonStates();
        } else {
            updateSystemStatus(`❌ 비디오 업로드 실패: ${data.message}`, 'error');
        }
    } catch (error) {
        updateSystemStatus(`❌ 서버 연결 오류: ${error.message}`, 'error');
        console.error('Upload error:', error);
    }
}

async function checkServerStatus() {
    try {
        const response = await fetch('http://localhost:5000/api/status');
        const data = await response.json();
        
        let statusText = '🤖 AI 모델 준비 완료';
        
        // 현재 모델 타입 표시
        const modelTypeNames = {
            'yolo_dinov2': 'YOLO + DINOv2',
            'yolo_clip': 'YOLO + CLIP (불량검사)'
        };
        
        const currentModelName = modelTypeNames[currentModelType] || currentModelType;
        currentModelSpan.textContent = `현재 모델: ${currentModelName}`;
        
        if (data.is_custom_model) {
            statusText += '\n🚀 커스텀 모델 활성화됨';
            isCustomModelTrained = true;
            workflowState.step2_training = true;
        }
        
        if (data.accumulated_training_data > 0) {
            statusText += `\n📊 누적 학습 데이터: ${data.accumulated_training_data}개`;
        }
        
        // DINOv2 패턴 학습 상태 확인
        if (data.learned_patterns_count > 0) {
            statusText += `\n🎯 학습된 패턴: ${data.learned_patterns_count}개`;
            workflowState.step2_training = true;
        }
        
        updateSystemStatus(statusText, 'success');
        updateWorkflowProgress();
        
    } catch (error) {
        updateSystemStatus('❌ 서버에 연결할 수 없습니다. Python 백엔드를 시작해주세요.', 'error');
        currentModelSpan.textContent = `현재 모델: ${currentModelType} (연결 안됨)`;
    }
}

function updateSystemStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-${type}`;
}

function updateTrainingStatus(message, type) {
    trainingStatus.textContent = message;
    trainingStatus.className = `status-display status-${type}`;
}

function updateVideoGenerationStatus(message, type) {
    videoGenerationStatus.textContent = message;
    videoGenerationStatus.className = `status-display status-${type}`;
}

// =================
// 모델 선택 및 전환 기능
// =================

function initializeModelSelection() {
    // 모델 카드 클릭 이벤트
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            const modelType = this.parentElement.dataset.model;
            selectModel(modelType);
        });
    });
    
    // 모델 전환 버튼
    if (switchModelBtn) {
        switchModelBtn.addEventListener('click', switchToSelectedModel);
    }
    
    // CLIP 설정 업데이트 버튼
    if (updateClipSettingsBtn) {
        updateClipSettingsBtn.addEventListener('click', updateClipSettings);
    }
    
    // Defect threshold 슬라이더
    if (defectThresholdSlider) {
        defectThresholdSlider.addEventListener('input', function() {
            if (defectThresholdValue) {
                defectThresholdValue.textContent = this.value;
            }
        });
    }
    
    // 초기 모델 상태 설정
    selectModel(currentModelType);
    updateCurrentModelDisplay();
}

function selectModel(modelType) {
    selectedModelType = modelType;
    
    // 모든 카드에서 active 클래스 제거
    modelCards.forEach(card => {
        card.classList.remove('active');
    });
    
    // 선택된 카드에 active 클래스 추가
    const selectedCard = document.querySelector(`[data-model="${modelType}"] .model-card`);
    if (selectedCard) {
        selectedCard.classList.add('active');
    }
    
    // 모델별 섹션 표시/숨김
    updateModelSpecificSections(modelType);
    
    // 전환 버튼 활성화 (현재 모델과 다른 경우)
    if (switchModelBtn) {
        switchModelBtn.disabled = (modelType === currentModelType);
    }
}

function updateModelSpecificSections(modelType) {
    const modelSpecificSections = document.querySelectorAll('.model-specific');
    
    modelSpecificSections.forEach(section => {
        const sectionModel = section.dataset.model;
        if (sectionModel === modelType) {
            section.style.display = 'block';
        } else {
            section.style.display = 'none';
        }
    });
}

async function switchToSelectedModel() {
    if (selectedModelType === currentModelType) {
        return;
    }
    
    if (switchModelBtn) {
        switchModelBtn.disabled = true;
        switchModelBtn.textContent = '모델 전환 중...';
    }
    
    try {
        let config = {};
        
        // YOLO + CLIP 모델의 경우 설정 추가
        if (selectedModelType === 'yolo_clip') {
            const defectQueries = defectQueriesText ? defectQueriesText.value.split('\n').filter(q => q.trim()) : [];
            const defectThreshold = defectThresholdSlider ? parseFloat(defectThresholdSlider.value) : 0.7;
            
            config = {
                defect_queries: defectQueries,
                defect_threshold: defectThreshold
            };
        }
        
        const response = await fetch('/api/switch_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_type: 'detector',
                model_name: selectedModelType,
                config: config
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            currentModelType = selectedModelType;
            updateCurrentModelDisplay();
            updateSystemStatus(`✅ 모델이 ${getModelDisplayName(selectedModelType)}로 전환되었습니다.`, 'success');
            
            // 모델별 버튼 상태 업데이트
            updateButtonStates();
            
            // YOLO+CLIP 모델로 전환 시 추가 안내
            if (selectedModelType === 'yolo_clip') {
                updateSystemStatus(
                    `✅ 모델이 ${getModelDisplayName(selectedModelType)}로 전환되었습니다.\n` +
                    `🎯 CLIP 기반 불량품 탐지가 활성화되었습니다.\n` +
                    `📹 커스텀 비디오 생성이 바로 가능합니다.`, 
                    'success'
                );
            }
        } else {
            throw new Error(result.error || '모델 전환 실패');
        }
        
    } catch (error) {
        console.error('Model switch error:', error);
        updateSystemStatus(`❌ 모델 전환 실패: ${error.message}`, 'error');
        
        // 선택을 원래대로 되돌림
        selectModel(currentModelType);
    } finally {
        if (switchModelBtn) {
            switchModelBtn.disabled = false;
            switchModelBtn.textContent = '모델 전환';
        }
    }
}

async function updateClipSettings() {
    if (currentModelType !== 'yolo_clip') {
        updateSystemStatus('❌ YOLO + CLIP 모델이 활성화되어 있을 때만 설정을 변경할 수 있습니다.', 'error');
        return;
    }
    
    if (updateClipSettingsBtn) {
        updateClipSettingsBtn.disabled = true;
        updateClipSettingsBtn.textContent = '설정 업데이트 중...';
    }
    
    try {
        const defectQueries = defectQueriesText ? defectQueriesText.value.split('\n').filter(q => q.trim()) : [];
        const defectThreshold = defectThresholdSlider ? parseFloat(defectThresholdSlider.value) : 0.7;
        
        const response = await fetch('/api/switch_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_type: 'detector',
                model_name: 'yolo_clip',
                config: {
                    defect_queries: defectQueries,
                    defect_threshold: defectThreshold
                }
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            updateSystemStatus('✅ CLIP 설정이 업데이트되었습니다.', 'success');
            
            // CLIP 상태 업데이트
            const clipStatus = document.getElementById('clipStatus');
            if (clipStatus) {
                clipStatus.innerHTML = `
                    <div class="clip-settings-summary">
                        <strong>현재 설정:</strong><br>
                        • 검사 쿼리: ${defectQueries.length}개<br>
                        • 임계값: ${defectThreshold}<br>
                        • 업데이트: ${new Date().toLocaleTimeString()}
                    </div>
                `;
            }
        } else {
            throw new Error(result.error || 'CLIP 설정 업데이트 실패');
        }
        
    } catch (error) {
        console.error('CLIP settings update error:', error);
        updateSystemStatus(`❌ CLIP 설정 업데이트 실패: ${error.message}`, 'error');
    } finally {
        if (updateClipSettingsBtn) {
            updateClipSettingsBtn.disabled = false;
            updateClipSettingsBtn.textContent = '🔧 CLIP 설정 업데이트';
        }
    }
}

function getModelDisplayName(modelType) {
    const modelNames = {
        'yolo_dinov2': 'YOLO + DINOv2',
        'yolo_clip': 'YOLO + CLIP (불량검사)'
    };
    return modelNames[modelType] || modelType;
}

function updateCurrentModelDisplay() {
    if (currentModelSpan) {
        const displayName = getModelDisplayName(currentModelType);
        currentModelSpan.textContent = `현재 모델: ${displayName}`;
    }
}

// =================
// 프레임 이동 기능
// =================

const FRAME_RATE = 1 / 30;

nextFrameBtn.addEventListener('click', () => {
    videoPlayer.currentTime += FRAME_RATE;
    updateFrameInfo();
    if (Object.keys(currentRect).length > 0) {
        currentRect = {};
    }
});

prevFrameBtn.addEventListener('click', () => {
    videoPlayer.currentTime -= FRAME_RATE;
    updateFrameInfo();
    if (Object.keys(currentRect).length > 0) {
        currentRect = {};
    }
});

videoPlayer.addEventListener('seeked', drawCurrentFrameAnnotations);

// =================
// 초기화
// =================

// 페이지 로드 시 초기화
window.addEventListener('load', async () => {
    // 모델 선택 기능 초기화
    initializeModelSelection();
    
    await checkServerStatus();
    updateWorkflowProgress();
    updateDataStatistics();
    console.log('🚀 4단계 워크플로우 시스템 초기화 완료');
}); 