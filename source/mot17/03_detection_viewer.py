
import cv2 as cv
from mot17_common import get_images, get_fps, load_detections

images = get_images()
fps = get_fps()
groups = load_detections(conf=0.5)

for path in images:
    frame_no = int(path.stem)
    img = cv.imread(str(path))
    if img is None:
        continue

    rows = groups.get(frame_no)
    count = 0

    if rows is not None:
        for r in rows.itertuples(index=False):
            x1, y1 = int(r.x), int(r.y)
            x2 = int(r.x + r.w)
            y2 = int(r.y + r.h)

            cv.rectangle(img, (x1, y1), (x2, y2),
                         (0, 255, 0), 2)
            cv.putText(img, f"{r.conf:.2f}",
                       (x1, max(15, y1 - 5)),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
            count += 1

    cv.putText(img, f"Detections: {count}",
               (20, 35), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 255), 2)

    cv.imshow("Detection Viewer", img)
    if cv.waitKey(round(1000 / fps)) & 0xFF == 27:
        break

cv.destroyAllWindows()
