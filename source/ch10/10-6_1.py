# ============================================================
# 10-6_current_env.py (현재 환경 맞춤형 AR 증강현실 코드)
# MediaPipe를 걷어내고 현재 작동 중인 OpenCV YuNet으로 눈 위치를 찾습니다.
# ============================================================
import cv2 as cv
import os
import sys
import urllib.request

# 1. AR 장신구(dice.png) 로드 및 방어 코드
img_path = 'dice.png'
if not os.path.exists(img_path):
    sys.exit(f"에러: '{img_path}' 파일이 없습니다. 교재의 주사위 이미지를 폴더에 넣어주세요.")

# 투명도(Alpha)를 포함하여 이미지 읽기
dice = cv.imread(img_path, cv.IMREAD_UNCHANGED)
dice = cv.resize(dice, dsize=(0, 0), fx=0.1, fy=0.1)
w, h = dice.shape[1], dice.shape[0]

if dice.shape[2] != 4:
    sys.exit("에러: 주사위 이미지는 배경이 투명한 PNG(4채널)여야 합니다.")

# 2. 현재 환경에서 가장 안정적인 OpenCV YuNet 모델 준비
model_path = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(model_path):
    # 이전 실습에서 받아둔 파일이 없다면 자동으로 다운로드
    print("OpenCV 얼굴 검출 모델을 준비합니다...")
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(url, model_path)

# 윈도우 환경 웹캠 딜레이 및 에러 방지를 위해 CAP_DSHOW 옵션 추가
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("웹캠을 열 수 없습니다.")

# 초기 프레임 크기 확인 및 카메라 로딩 실패 방어 코드
ret, frame = cap.read()
if not ret or frame is None:
    sys.exit("카메라가 다른 프로그램에서 사용 중이거나 응답하지 않습니다. 모든 터미널을 종료 후 재실행해 주세요.")

frame_h, frame_w = frame.shape[:2]

detector = cv.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(frame_w, frame_h),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

print("현재 환경 맞춤형 AR 모드를 시작합니다. 창을 닫으려면 'q' 키를 누르세요.")

# 3. 실시간 합성 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 직관적인 조작을 위해 화면 좌우 반전(거울 모드)
    frame = cv.flip(frame, 1)

    # 얼굴 및 이목구비 검출 실행
    retval, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # YuNet 모델에서 face[4], face[5]는 '오른쪽 눈'의 x, y 좌표입니다.
            cx, cy = int(face[4]), int(face[5])

            # 주사위가 들어갈 합성 영역(x1, y1, x2, y2) 계산
            x1, x2 = cx - w // 2, cx + w // 2
            y1, y2 = cy - h // 2, cy + h // 2

            # 합성 영역이 화면 밖으로 삐져나가지 않을 때만 안전하게 합성
            if x1 > 0 and y1 > 0 and x2 < frame_w and y2 < frame_h:
                # PNG 투명도 값을 이용하여 자연스러운 배경 합성(Alpha Blending)
                alpha = dice[:, :, 3:] / 255.0
                frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1.0 - alpha) + dice[:, :, :3] * alpha

    # 화면 출력
    cv.imshow('AR Face overlay (Current Env)', frame)

    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
