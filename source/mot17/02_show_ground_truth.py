import cv2 as cv
import pandas as pd
from pathlib import Path

# 1. 프로젝트 및 데이터 경로
BASE_DIR = Path(__file__).resolve().parents[2]

SEQ_DIR = (
        BASE_DIR
        / "datasets"
        / "MOT17"
        / "train"
        / "MOT17-13-SDP"
)

IMG_DIR = SEQ_DIR / "img1"
GT_FILE = SEQ_DIR / "gt" / "gt.txt"

# 2. 정답 데이터 읽기
columns = [
    "frame", "id", "x", "y",
    "w", "h", "mark", "class", "visibility"
]

gt = pd.read_csv(
    GT_FILE,
    header=None,
    names=columns
)

# 3. 실제 사람(class=1) 중 유효한 정답만 선택
gt = gt[
    (gt["mark"] == 1) &
    (gt["class"] == 1)
    ]

# 4. 이미지 목록
images = sorted(IMG_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError("MOT17 이미지가 없습니다.")

# 5. 프레임별 정답 표시
for image_path in images:

    frame_no = int(image_path.stem)

    img = cv.imread(str(image_path))

    if img is None:
        continue

    frame_gt = gt[gt["frame"] == frame_no]

    for _, obj in frame_gt.iterrows():

        x = int(obj["x"])
        y = int(obj["y"])
        w = int(obj["w"])
        h = int(obj["h"])
        person_id = int(obj["id"])

        # 사람 영역 표시
        cv.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # 사람 ID 표시
        cv.putText(
            img,
            f"ID {person_id}",
            (x, max(20, y - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # 현재 프레임 표시
    cv.putText(
        img,
        f"Frame: {frame_no}",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv.imshow("MOT17 Ground Truth", img)

    if cv.waitKey(71) == 27:
        break

cv.destroyAllWindows()