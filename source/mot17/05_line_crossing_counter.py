
import cv2 as cv

from mot17_common import (
    get_images, get_fps, load_detections,
    get_boxes, Tracker, draw_tracks, bottom_center
)

images = get_images()
fps = get_fps()
groups = load_detections(0.5)
tracker = Tracker()

previous_y = {}
crossed_down = set()
crossed_up = set()

count_down = 0
count_up = 0

# 영상의 50% 높이에 가상선 설정
LINE_RATIO = 0.5

for path in images:
    frame_no = int(path.stem)
    img = cv.imread(str(path))
    if img is None:
        continue

    h, w = img.shape[:2]
    line_y = int(h * LINE_RATIO)

    tracks = tracker.update(
        get_boxes(groups.get(frame_no))
    )

    draw_tracks(img, tracks)

    cv.line(img, (0, line_y), (w, line_y),
            (0, 255, 255), 3)

    for tid, box in tracks.items():
        x, y = bottom_center(box)

        if tid in previous_y:
            old_y = previous_y[tid]

            # 위에서 아래로 통과
            if old_y < line_y <= y:
                if tid not in crossed_down:
                    count_down += 1
                    crossed_down.add(tid)

            # 아래에서 위로 통과
            if old_y > line_y >= y:
                if tid not in crossed_up:
                    count_up += 1
                    crossed_up.add(tid)

        previous_y[tid] = y
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)

    cv.putText(img, f"DOWN: {count_down}",
               (20, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.9, (0, 255, 0), 2)

    cv.putText(img, f"UP: {count_up}",
               (20, 70), cv.FONT_HERSHEY_SIMPLEX,
               0.9, (0, 255, 255), 2)

    cv.imshow("People Counter", img)

    if cv.waitKey(round(1000 / fps)) & 0xFF == 27:
        break

cv.destroyAllWindows()

print("아래쪽 통과:", count_down)
print("위쪽 통과:", count_up)
