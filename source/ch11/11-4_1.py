from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import cv2 as cv
import torch

# 1. 이미지 로드
img = Image.open('BSDS_361010.jpg')

# 2. 전처리기 및 모델 로드 (경고 해결: FeatureExtractor -> ImageProcessor)
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# 3. 모델 추론 진행 (PyTorch 텐서 사용)
inputs = processor(images=img, return_tensors='pt')
outputs = model(**inputs)

# 4. 바운딩 박스 후처리 (Hugging Face 내장 함수 사용하여 좌표 계산 자동화)
target_sizes = torch.tensor([img.size[::-1]]) # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

# 5. OpenCV로 결과 그리기 (RGB -> BGR 색상 교정 완벽 적용)
im = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
colors = np.random.uniform(0, 255, size=(100, 3))

# 감지된 객체들을 순회하며 박스와 라벨 그리기
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x_min, y_min, x_max, y_max = map(int, box)

    class_name = model.config.id2label[label.item()]
    prob = round(score.item() * 100, 2) # 경고 해결: .item()으로 텐서 값 안전하게 추출

    # 바운딩 박스 그리기
    color = colors[label.item() % 100]
    cv.rectangle(im, (x_min, y_min), (x_max, y_max), color, 2)

    # 텍스트 라벨 시각화 (가독성 개선)
    text = f"{class_name}: {prob}%"
    cv.putText(im, text, (x_min, max(y_min - 10, 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 6. 최종 결과 창 출력
cv.imshow('DETR Object Detection', im)
cv.waitKey(0)
cv.destroyAllWindows()