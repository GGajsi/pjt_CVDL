
import cv2 as cv
import numpy as np

from mot17_common import (
    get_images, get_fps, load_detections,
    get_boxes, bottom_center
)

images = get_images()
fps = get_fps()
groups = load_detections(0.5)

heat = None

for path in images:
    frame_no = int(path.stem)
    img = cv.imread(str(path))
    if img is None:
        continue

    h, w = img.shape[:2]

    if heat is None:
        heat = np.zeros((h, w), dtype=np.float32)

    boxes = get_boxes(groups.get(frame_no))

    # 검출된 사람의 발 위치를 누적
    for box in boxes:
        x, y = bottom_center(box)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        heat[y, x] += 1

    # 주변으로 밀집도 확산
    blurred = cv.GaussianBlur(heat, (0, 0), sigmaX=25)

    # 현재 누적 최대값 기준으로 정규화
    normalized = cv.normalize(
        blurred, None, 0, 255, cv.NORM_MINMAX
    ).astype(np.uint8)

    colored = cv.applyColorMap(
        normalized, cv.COLORMAP_JET
    )

    result = cv.addWeighted(
        img, 0.55, colored, 0.45, 0
    )

    cv.putText(result, f"Frame: {frame_no}",
               (20, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2)

    cv.imshow("Pedestrian Heatmap", result)

    if cv.waitKey(round(1000 / fps)) & 0xFF == 27:
        break

cv.destroyAllWindows()
