import cv2 as cv
import sys
from pathlib import Path

img = cv.imread(str(Path(__file__).resolve().parent / "soccer.jpg"))

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('Image Display', img)  # 윈도우에 영상 표시

cv.waitKey()
cv.destroyAllWindows()