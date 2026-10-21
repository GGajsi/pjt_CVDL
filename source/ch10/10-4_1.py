# ============================================================
# 10-4_opencv.py (MediaPipe 대체본)
# OpenCV 내장 알고리즘을 이용한 얼굴 검출
# ============================================================

import cv2 as cv
import sys
from pathlib import Path

# 파일 경로 설정 및 방어 코드 (원본 로직 유지)
BASE_DIR = Path(__file__).resolve().parent #[cite: 13]
IMG_PATH = BASE_DIR / 'BSDS_376001.jpg' #[cite: 13]

if not IMG_PATH.is_file(): #[cite: 13]
    sys.exit(f"에러: 이미지 파일을 찾을 수 없습니다. 경로를 확인해 주세요.\n예상 경로: {IMG_PATH}") #[cite: 13]

img = cv.imread(str(IMG_PATH)) #[cite: 13]

print("OpenCV 얼굴 검출을 시작합니다...")

# 1. OpenCV는 흑백 이미지에서 객체를 훨씬 빠르고 정확하게 찾습니다.
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. OpenCV에 기본 내장된 정면 얼굴 검출 AI 모델(XML) 로드
cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(cascade_path)

# 3. 얼굴 검출 실행
# scaleFactor: 이미지 피라미드 비율, minNeighbors: 신뢰도 임계값
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 4. 검출 결과 시각화
if len(faces) == 0:
    print('얼굴 검출에 실패했습니다. (사진에 얼굴이 없거나 인식할 수 없습니다.)') #[cite: 13]
else:
    print(f"총 {len(faces)}개의 얼굴이 검출되었습니다.") #[cite: 13]

    # x, y 좌표와 너비(w), 높이(h)를 이용해 초록색 바운딩 박스 그리기
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('Face detection by OpenCV', img) #[cite: 13]
    print("창에서 아무 키나 누르면 종료됩니다.") #[cite: 13]
    cv.waitKey(0) #[cite: 13]
    cv.destroyAllWindows() #[cite: 13]