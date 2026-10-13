
import cv2 as cv
from ultralytics import YOLO

from mot17_common import (
    get_images, get_fps, color_for_id
)

# 사전학습된 객체 검출 모델
# 첫 실행 시 모델 가중치 다운로드 필요
model = YOLO("yolo11n.pt")

images = get_images()
fps = get_fps()

cv.namedWindow("YOLO ByteTrack", cv.WINDOW_NORMAL)

for path in images:
    img = cv.imread(str(path))
    if img is None:
        continue

    # YOLO 검출 + ByteTrack 추적
    results = model.track(
        img,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],   # COCO의 사람 클래스
        conf=0.25,
        verbose=False
    )

    result = results[0]
    count = 0

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()

        for box, tid in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)
            color = color_for_id(tid)

            cv.rectangle(
                img, (x1, y1), (x2, y2), color, 2
            )

            cv.putText(
                img, f"Person ID: {tid}",
                (x1, max(20, y1 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

            count += 1

    cv.putText(
        img, f"Tracked people: {count}",
        (20, 40), cv.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 255), 2
    )

    cv.imshow("YOLO ByteTrack", img)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
