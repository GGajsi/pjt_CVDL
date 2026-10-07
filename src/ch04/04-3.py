import cv2 as cv
import numpy as np

# 1. 영상 로드 및 그레이스케일 변환
img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. 캐니 에지 검출 (Tlow=100, Thigh=200)
canny = cv.Canny(gray, 100, 200)

# 3. 에지 맵에서 경계선 좌표 검출
contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# 4. 길이가 100을 초과하는 윤곽선만 필터링
long_contours = [c for c in contours if len(c) > 100]

# 5. 원본 영상에 윤곽선 덧그리기 (BGR 기준 초록색, 두께 3)
cv.drawContours(img, long_contours, -1, (0, 255, 0), 3)

# 6. 결과 출력
cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)

cv.waitKey()
cv.destroyAllWindows()