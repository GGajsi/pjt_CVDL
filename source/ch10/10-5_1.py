# ============================================================
# 10-5_opencv_dnn.py (최종 Vibe Coding: 안정성 및 인식률 최우선)
# OpenCV 공식 딥러닝(DNN) 모듈을 이용한 고성능 얼굴 검출
# ============================================================
import cv2 as cv
import urllib.request
import os
import sys

# 1. OpenCV 공식 딥러닝 얼굴 검출 모델 다운로드 (안정적인 공식 GitHub 서버)
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print("OpenCV 공식 딥러닝 얼굴 검출 모델을 다운로드합니다 (최초 1회)...")
    try:
        # 네트워크 구조 파일 (Prototxt)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
        # 훈련된 가중치 파일 (Caffe Model)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_path
        )
        print("다운로드 완료!\n")
    except Exception as e:
        sys.exit(f"다운로드 실패. 인터넷 연결을 확인해 주세요: {e}")

# 2. 딥러닝 모델 네트워크 로드
print("OpenCV DNN Face Detector 로딩 중...")
net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

# 3. 웹캠 캡처 초기화
cap = cv.VideoCapture(0)
if not cap.isOpened():
    sys.exit("웹캠을 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")

print("\n딥러닝 기반 실시간 얼굴 탐지를 시작합니다. 창을 닫으려면 'q' 키를 누르세요.")

# 4. 실시간 프레임 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득에 실패하여 루프를 나갑니다.")
        break

    (h, w) = frame.shape[:2]

    # 5. 프레임을 딥러닝 모델이 인식할 수 있는 블롭(Blob) 형태로 변환
    # 300x300으로 리사이징 후 평균값(104.0, 177.0, 123.0)을 빼서 정규화 수행
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 6. 네트워크 추론 실행
    net.setInput(blob)
    detections = net.forward()

    # 7. 검출 결과 시각화
    # detections.shape[2]는 검출된 객체의 총개수를 의미합니다.
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 신뢰도가 50% 이상인 경우에만 얼굴로 인정 (오탐지 방지)
        if confidence > 0.5:
            # 바운딩 박스 좌표 계산 (정규화된 비율을 실제 픽셀 크기로 복원)
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # 얼굴 영역에 박스 그리고 신뢰도 텍스트 출력
            text = f"Face: {confidence * 100:.2f}%"
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # 텍스트가 화면 밖으로 나가지 않도록 Y좌표 보정
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 8. 화면 출력
    cv.imshow("Face Detection by OpenCV DNN", frame)

    if cv.waitKey(5) == ord('q'):
        print("\n사용자에 의해 종료되었습니다.")
        break

# 9. 자원 해제
cap.release()
cv.destroyAllWindows()