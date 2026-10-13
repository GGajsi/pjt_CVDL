import cv2 as cv
from pathlib import Path

# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parents[2]

# 이미지 폴더
IMG_DIR = (
        BASE_DIR
        / "datasets"
        / "MOT17"
        / "train"
        / "MOT17-13-SDP"
        / "img1"
)

# 이미지 목록
images = sorted(IMG_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError(f"이미지가 없습니다: {IMG_DIR}")

print("전체 프레임 수:", len(images))

# 순서대로 이미지 출력
for image_path in images:
    img = cv.imread(str(image_path))

    if img is None:
        continue

    cv.imshow("MOT17 Video", img)

    # 약 14FPS로 재생
    key = cv.waitKey(71)

    if key == 27:  # ESC
        break

cv.destroyAllWindows()