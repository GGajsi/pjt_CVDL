import cv2 as cv
import numpy as np
from pathlib import Path

# 1. 이미지 로드 및 캔버스 준비
img = cv.imread(str(Path(__file__).resolve().parents[1] / "ch02" / "soccer.jpg"))
img_show = img.copy()

# 2. 마스크 및 브러시 파라미터 초기화
# GC_PR_BGD(아마도 배경일 영역)로 초기화
mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask[:] = cv.GC_PR_BGD

brush_size = 9
color_fg = (255, 0, 0)  # 파란색 (전경 지정)
color_bg = (0, 0, 255)  # 빨간색 (배경 지정)

# 3. 마우스 콜백 함수 정의
def on_mouse(event, x, y, flags, param):
    # 좌클릭 / 좌클릭 드래그: 전경(물체) 표시
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON)):
        cv.circle(img_show, (x, y), brush_size, color_fg, -1)
        cv.circle(mask, (x, y), brush_size, cv.GC_FGD, -1)

    # 우클릭 / 우클릭 드래그: 배경 표시
    elif event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_RBUTTON)):
        cv.circle(img_show, (x, y), brush_size, color_bg, -1)
        cv.circle(mask, (x, y), brush_size, cv.GC_BGD, -1)

    cv.imshow('Painting', img_show)

# 4. 인터랙티브 드로잉 창 설정
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', on_mouse)
cv.imshow('Painting', img_show)

print("마우스 좌클릭(물체)/우클릭(배경)으로 표시한 뒤 'q'를 누르면 GrabCut이 실행됩니다.")
while True:
    if (cv.waitKey(1) & 0xFF) == ord('q'):
        break

# 5. GrabCut 알고리즘 적용
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)

cv.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_MASK)

# 6. 마스크 필터링: 확실한 배경(GC_BGD) 및 배경 추정(GC_PR_BGD)은 0, 나머지는 1
fg_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
result_cut = img * fg_mask[:, :, np.newaxis]

# 7. 결과 출력
cv.imshow('Grab cut image', result_cut)
cv.waitKey()
cv.destroyAllWindows()