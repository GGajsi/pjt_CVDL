import cv2 as cv
import numpy as np
import skimage

# scikit-image 내장 예제 영상 불러오기
img = skimage.data.coffee()
cv.imshow('Coffee image', cv.cvtColor(img, cv.COLOR_RGB2BGR))

# 1. 컴팩트도(compactness) 20 설정 후 슈퍼 픽셀 분할 및 경계선 표시
slic1 = skimage.segmentation.slic(img, compactness=20, n_segments=600)
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)
sp_img1 = np.uint8(sp_img1 * 255.0)

# 2. 컴팩트도(compactness) 40 설정 후 슈퍼 픽셀 분할 및 경계선 표시
slic2 = skimage.segmentation.slic(img, compactness=40, n_segments=600)
sp_img2 = skimage.segmentation.mark_boundaries(img, slic2)
sp_img2 = np.uint8(sp_img2 * 255.0)

# 결과 출력 (OpenCV 규격에 맞게 RGB -> BGR 변환)
cv.imshow('Super pixels (compact 20)', cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR))
cv.imshow('Super pixels (compact 40)', cv.cvtColor(sp_img2, cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()