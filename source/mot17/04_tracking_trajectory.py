
import cv2 as cv
from collections import defaultdict, deque

from mot17_common import (
    get_images, get_fps, load_detections,
    get_boxes, Tracker, draw_tracks,
    bottom_center, color_for_id
)

images = get_images()
fps = get_fps()
groups = load_detections(0.5)

tracker = Tracker()

# 각 사람의 최근 위치 50개 저장
histories = defaultdict(lambda: deque(maxlen=50))

for path in images:
    frame_no = int(path.stem)
    img = cv.imread(str(path))
    if img is None:
        continue

    boxes = get_boxes(groups.get(frame_no))
    tracks = tracker.update(boxes)

    draw_tracks(img, tracks)

    for tid, box in tracks.items():
        center = bottom_center(box)
        histories[tid].append(center)

        points = list(histories[tid])
        color = color_for_id(tid)

        # 과거 위치를 이어서 궤적 그리기
        for i in range(1, len(points)):
            cv.line(img, points[i - 1], points[i],
                    color, 3)

        cv.circle(img, center, 5, (0, 0, 255), -1)

    cv.putText(img, f"Tracking: {len(tracks)}",
               (20, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 255), 2)

    cv.imshow("Person Trajectory", img)

    if cv.waitKey(round(1000 / fps)) & 0xFF == 27:
        break

cv.destroyAllWindows()
