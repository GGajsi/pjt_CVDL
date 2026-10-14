import cv2 as cv
import matplotlib.pyplot as plt
import sys
from pathlib import Path

img = cv.imread(str(Path(__file__).resolve().parents[1] / "ch02" / "soccer.jpg"))

# 이미지를 정상적으로 불러왔는지 확인하는 예외 처리
if img is None:
    sys.exit('파일을 찾을 수 없습니다. 파일 경로를 다시 확인해주세요.')

h = cv.calcHist([img], [2], None, [256], [0, 256])  # 2번 채널인 R 채널 히스토그램 계산
plt.plot(h, color='r', linewidth=1)
plt.show()