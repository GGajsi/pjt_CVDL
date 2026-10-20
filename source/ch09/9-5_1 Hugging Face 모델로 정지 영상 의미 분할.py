# ============================================================
# PixelLib 대체: Hugging Face SegFormer 기반 시맨틱 분할
# 필요 패키지: pip install transformers torch pillow opencv-python
# ============================================================

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
import cv2 as cv
from pathlib import Path

# 1. 파일 설정
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "busy_street.jpg"

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"이미지 파일이 없습니다: {IMAGE_PATH}")

# OpenCV(BGR)와 PIL(RGB) 형식으로 각각 읽기
img_cv = cv.imread(str(IMAGE_PATH))
img_pil = Image.open(IMAGE_PATH).convert("RGB")

# 2. 최신 모델 로드 (ADE20K로 학습된 NVIDIA SegFormer-B0)
print("Hugging Face SegFormer 모델 로딩 중 (최초 실행 시 다운로드 필요)...")
model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id)

# 3. 모델 추론 실행
print("시맨틱 분할 추론 중...")
inputs = processor(images=img_pil, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 4. 결과 후처리 (원본 해상도로 복구)
logits = outputs.logits  # shape: (batch, classes, height/4, width/4)
logits = F.interpolate(
    logits,
    size=img_pil.size[::-1], # (height, width)
    mode="bilinear",
    align_corners=False
)

# 픽셀별로 가장 확률이 높은 클래스 인덱스(0~149) 추출
predicted_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

# 5. 컬러 마스크 생성 및 오버레이 합성
# ADE20K의 150개 클래스를 구분하기 위한 랜덤 컬러 팔레트 생성
np.random.seed(42)  # 색상 고정
palette = np.random.randint(0, 255, size=(150, 3), dtype=np.uint8)

# 예측된 마스크 값에 컬러 매핑 후 OpenCV BGR 형식으로 변환
color_mask = palette[predicted_mask]
color_mask_bgr = cv.cvtColor(color_mask, cv.COLOR_RGB2BGR)

# 원본 이미지와 6:4 비율로 합성 (오버레이)
img_overlay = cv.addWeighted(img_cv, 0.6, color_mask_bgr, 0.4, 0)

# 6. 이미지 출력 (화면에 맞게 크기 조절)
height, width = 400, 600
windows = [
    ("1. Original", img_cv),
    ("2. Semantic Mask", color_mask_bgr),
    ("3. Overlayed", img_overlay)
]

for title, img in windows:
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, width, height)
    cv.imshow(title, img)

print("\n3개의 이미지 창이 열렸습니다. 창에서 아무 키나 누르면 종료됩니다.")
cv.waitKey(0)
cv.destroyAllWindows()