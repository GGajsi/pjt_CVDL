import cv2 as cv

# 1. 이미지 읽기
img = cv.imread('mot_color70.jpg')

if img is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")

# 2. 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3. SIFT 객체 생성
sift = cv.SIFT_create()

# 4. 특징점 검출 및 기술자 계산
kp, des = sift.detectAndCompute(gray, None)

# 5. 검출 결과 확인
print("검출된 특징점 개수:", len(kp))

if des is not None:
    print("기술자 배열 크기:", des.shape)

# 6. 특징점 그리기 (원본 컬러 이미지 사용)
result = cv.drawKeypoints(
    img,
    kp,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 7. 이미지 출력
cv.imshow('Original', img)
cv.imshow('SIFT', result)

if len(kp) > 0:
    print("첫 번째 특징점 위치:", kp[0].pt)
    print("첫 번째 특징점 크기:", kp[0].size)
    print("첫 번째 특징점 방향:", kp[0].angle)

if des is not None:
    print("첫 번째 기술자:", des[0])

cv.waitKey(0)
cv.destroyAllWindows()