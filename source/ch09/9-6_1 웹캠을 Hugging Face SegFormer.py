# ============================================================
# 9-6_1 웹캠을 Hugging Face SegFormer.py (수정본)
# 웹캠 실시간 시맨틱 분할 (PixelLib -> Hugging Face SegFormer)
# 필요 패키지: pip install transformers torch numpy opencv-python
# ============================================================

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F
import torch
import numpy as np
import cv2 as cv
from pathlib import Path

# 1. 모델 로드 및 디바이스(CPU/GPU) 설정
print("Hugging Face SegFormer 모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"추론 디바이스: {device}")

model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)

# 2. 색상 팔레트 설정 (ADE20K 150개 클래스용 랜덤 컬러)
np.random.seed(42)
palette = np.random.randint(0, 255, size=(150, 3), dtype=np.uint8)

# 3. 웹캠 캡처 초기화 (기존 cv.VideoCapture(0) 유지)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")

# 4. 비디오 저장 설정 (기존 output_video.mp4 기능 구현)
output_name = 'output_video.mp4'
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# 코덱 설정 (mp4v) 및 저장 FPS 지정
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps = 5.0 # CPU 환경의 추론 속도를 고려해 5 프레임으로 설정
out = cv.VideoWriter(output_name, fourcc, fps, (width, height))

print("\n실시간 분할을 시작합니다. 화면을 종료하려면 키보드의 'q'를 누르세요.")

# 5. 실시간 프레임 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다. 종료합니다.")
        break

    # OpenCV 프레임(BGR)을 RGB 형식으로 변환하여 모델 입력용으로 준비
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 결과 후처리: 원본 웹캠 해상도로 리사이즈
    logits = outputs.logits
    upsampled_logits = F.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    # 확률이 가장 높은 클래스 선택 및 컬러 맵 변환
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    color_mask = palette[predicted_mask]
    color_mask_bgr = cv.cvtColor(color_mask, cv.COLOR_RGB2BGR)

    # 원본 프레임과 컬러 마스크 오버레이 합성 (기존 overlay=True 구현)
    overlay_frame = cv.addWeighted(frame, 0.6, color_mask_bgr, 0.4, 0)

    # 화면 출력 및 비디오 파일에 기록
    cv.imshow('Pixellib (SegFormer)', overlay_frame)
    out.write(overlay_frame)

    # 'q' 키를 누르면 루프 탈출
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("\n사용자에 의해 종료되었습니다.")
        break

# 6. 자원 해제 (기존 닫기 로직 유지)
cap.release()
out.release()
cv.destroyAllWindows()
print(f"비디오 저장이 완료되었습니다: {output_name}")