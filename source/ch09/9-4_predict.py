from pathlib import Path
import numpy as np
import cv2 as cv
from tensorflow import keras

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / 'oxford_seg.h5'
IMAGE_DIR = BASE_DIR / 'datasets' / 'oxford_pets' / 'images'
LABEL_DIR = BASE_DIR / 'datasets' / 'oxford_pets' / 'annotations' / 'trimaps'

# 저장된 모델 로딩
model = keras.models.load_model(MODEL_PATH, compile=False)

# 이미지 한 장 선택
image_paths = sorted(IMAGE_DIR.glob('*.jpg'))
if not image_paths:
    raise FileNotFoundError('이미지를 찾을 수 없습니다.')

image_path = image_paths[0]
label_path = LABEL_DIR / (image_path.stem + '.png')

# 원본 이미지 읽기
original = cv.imread(str(image_path))
if original is None:
    raise FileNotFoundError(image_path)

# 학습 입력과 동일하게 160x160 RGB로 변환
resized = cv.resize(original, (160, 160))
rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
x = np.expand_dims(rgb.astype(np.float32), axis=0)

# AI 예측
pred = model.predict(x, verbose=0)
pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

# 실제 정답 마스크
ground_truth = cv.imread(str(label_path), cv.IMREAD_GRAYSCALE)
if ground_truth is None:
    raise FileNotFoundError(label_path)

ground_truth = cv.resize(
    ground_truth, (160, 160),
    interpolation=cv.INTER_NEAREST
)

# 1,2,3 -> 0,1,2
ground_truth = ground_truth - 1

# 화면 표시용 변환
gt_display = (ground_truth * 127).astype(np.uint8)
pred_display = (pred_mask * 127).astype(np.uint8)

print('예측 이미지:', image_path.name)
print('예측 배열 크기:', pred.shape)

cv.imshow('Original Image', resized)
cv.imshow('Ground Truth Mask', gt_display)
cv.imshow('AI Prediction Mask', pred_display)

cv.waitKey(0)
cv.destroyAllWindows()