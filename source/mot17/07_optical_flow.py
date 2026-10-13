
import cv2 as cv
import numpy as np

from mot17_common import get_images, get_fps

images = get_images()
fps = get_fps()

previous_gray = None

for path in images:
    frame = cv.imread(str(path))
    if frame is None:
        continue

    # 계산 속도를 위해 축소
    frame = cv.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if previous_gray is None:
        previous_gray = gray
        continue

    # 두 프레임 사이의 Optical Flow 계산
    flow = cv.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = gray.shape
    result = frame.copy()

    # 일정한 간격으로 화살표 표시
    step = 20

    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):

            dx, dy = flow[y, x]

            magnitude = np.hypot(dx, dy)

            # 작은 움직임은 무시
            if magnitude < 1.0:
                continue

            end_x = int(x + dx * 3)
            end_y = int(y + dy * 3)

            cv.arrowedLine(
                result,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    cv.imshow("Optical Flow", result)
    previous_gray = gray

    if cv.waitKey(round(1000 / fps)) & 0xFF == 27:
        break

cv.destroyAllWindows()
