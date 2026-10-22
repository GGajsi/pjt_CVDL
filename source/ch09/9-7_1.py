# ============================================================
# YOLOv8 인스턴스 분할 5단계 독립 시각화 (원본/분할/포커스/블러/크롭)
# 필요 패키지: pip install ultralytics opencv-python
# ============================================================

from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import cv2 as cv
import numpy as np

# 1. 파일 및 디렉토리 경로 설정
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "busy_street.jpg"

CARS_DIR = BASE_DIR / "cars"
CARS_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"이미지 파일이 없습니다: {IMAGE_PATH}")

# 원본 이미지 불러오기
img_original = cv.imread(str(IMAGE_PATH))

# 2. YOLOv8 Small 분할 모델 로드
print("YOLOv8 Small Segmentation 모델 로딩 중...")
model = YOLO("yolov8s-seg.pt")

# 3. 이미지 추론 실행
print("\n이미지 분할 및 효과 적용 중...")
results = model.predict(source=img_original, conf=0.3, save=False)
result = results[0]

# [윈도우 2] 기본 YOLOv8 분할 이미지 (바운딩 박스 + 컬러 마스크)
img_standard_seg = result.plot()

# 효과용 사전 작업 (흑백, 가우시안 블러)
gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
img_blurred = cv.GaussianBlur(img_original, (51, 51), 0)

# 캔버스(마스크) 준비
mask_focus = np.zeros(img_original.shape[:2], dtype=np.uint8)  # 포커스용 (사람, 자전거)
mask_blur = np.zeros(img_original.shape[:2], dtype=np.uint8)   # 블러용 (사람, 가방류)
car_crops = [] # 크롭된 자동차 이미지를 담을 리스트
car_count = 1

detected_classes = []

# 4. 탐지된 객체 순회 및 마스크/크롭 처리
if result.boxes is not None and result.masks is not None:
    boxes = result.boxes
    masks = result.masks.xy

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        class_name = model.names[cls_id]
        detected_classes.append(class_name)

        polygon = masks[i].astype(np.int32)

        # [기능 3을 위한 마스크] 사람과 자전거는 컬러 유지
        if class_name in ['person', 'bicycle']:
            cv.fillPoly(mask_focus, [polygon], 255)

        # [기능 4를 위한 마스크] 사람과 가방류는 프라이버시 보호(블러)
        if class_name in ['person', 'backpack', 'handbag']:
            cv.fillPoly(mask_blur, [polygon], 255)

        # [기능 5를 위한 작업] 자동차 바운딩 박스 크롭
        if class_name == 'car':
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            car_crop = img_original[y1:y2, x1:x2]

            # 1) 파일로 자동 저장
            crop_path = CARS_DIR / f"car_{car_count}.jpg"
            cv.imwrite(str(crop_path), car_crop)
            car_count += 1

            # 2) 화면 출력을 위해 리스트에 보관 (높이를 250px로 통일하여 가로로 붙이기 위함)
            h, w = car_crop.shape[:2]
            ratio = 250 / h
            resized_crop = cv.resize(car_crop, (int(w * ratio), 250))
            car_crops.append(resized_crop)

# 5. 각 창에 띄울 최종 이미지 생성
# [윈도우 3] 포커스 이미지: 배경은 흑백, 대상(마스크)은 원본 컬러
img_focus = gray_bgr.copy()
img_focus[mask_focus == 255] = img_original[mask_focus == 255]

# [윈도우 4] 블러 이미지: 배경은 원본 컬러, 대상(마스크)은 블러
img_blur_only = img_original.copy()
img_blur_only[mask_blur == 255] = img_blurred[mask_blur == 255]

# [윈도우 5] 크롭된 자동차 콜라주 이미지 생성
if car_crops:
    img_cropped_cars = cv.hconcat(car_crops) # 리스트에 담긴 이미지를 가로로 이어 붙임
else:
    # 탐지된 차가 없을 경우 빈 화면 생성
    img_cropped_cars = np.zeros((250, 500, 3), dtype=np.uint8)
    cv.putText(img_cropped_cars, "No cars detected", (100, 125), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# 6. 콘솔 결과 출력
object_counts = Counter(detected_classes)
print("\n========== 탐지 및 작업 결과 ==========")
print(f"추출되어 저장된 자동차(car) 이미지 수: {len(car_crops)}개")
print("\n========== 객체 종류별 개수 ==========")
if object_counts:
    for name, count in object_counts.most_common():
        print(f"{name:15s}: {count}개")


# 7. 5개의 이미지 창 동시 출력
# 창이 5개이므로 모니터를 가득 채우지 않도록 창 크기를 조정합니다.
win_w, win_h = 450, 300

window_configs = [
    ("1. Original", img_original),
    ("2. Standard Seg", img_standard_seg),
    ("3. Focus Effect", img_focus),
    ("4. Blur Effect", img_blur_only),
    ("5. Cropped Cars", img_cropped_cars)
]

for title, img in window_configs:
    cv.namedWindow(title, cv.WINDOW_NORMAL)

    # 크롭 이미지는 가로로 길 수 있으므로 원본 비율을 존중하여 창 크기 적용
    if title == "5. Cropped Cars":
        cv.resizeWindow(title, img.shape[1], img.shape[0])
    else:
        cv.resizeWindow(title, win_w, win_h)

    cv.imshow(title, img)

print("\n5개의 이미지 창이 열렸습니다. 창에서 아무 키나 누르면 종료됩니다.")
cv.waitKey(0)
cv.destroyAllWindows()