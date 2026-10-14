import cv2 as cv
import sys
from pathlib import Path

img = cv.imread(str(Path(__file__).resolve().parent / "girl_laughing.jpg"))

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

ix, iy = -1, -1

def draw(event, x, y, flags, param):
    global ix, iy

    if event == cv.EVENT_LBUTTONDOWN:         # 마우스 왼쪽 버튼을 클릭했을 때 초기 위치 저장
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP and ix >= 0:  # 마우스 왼쪽 버튼을 놓았을 때 직사각형 그리기
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        ix, iy = -1, -1

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break