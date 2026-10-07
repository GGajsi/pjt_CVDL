import cv2 as cv
import numpy as np

# 영상 불러오기 및 그레이스케일 변환
img = cv.imread('apples.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 허프 원 변환을 통한 원 검출
# 해상도 비율: 1, 원 중심 간 최소 거리: 200, 캐니 에지 임곗값: 150, 중심 검출 임곗값: 20, 반지름 범위: 50~120
apples = cv.HoughCircles(
    gray,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=200,
    param1=150,
    param2=20,
    minRadius=50,
    maxRadius=120
)

# 검출된 원 시각화
if apples is not None:
    apples = np.uint16(np.around(apples))
    for circle in apples[0]:
        center = (int(circle[0]), int(circle[1]))
        radius = int(circle[2])
        # 원 그리기 (BGR: 청색, 두께 2)
        cv.circle(img, center, radius, (255, 0, 0), 2)

cv.imshow('Apple detection', img)
cv.waitKey()
cv.destroyAllWindows()