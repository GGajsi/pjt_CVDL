# ============================================================
# 10-3_1.py (YOLOv8 내장 트래커 사용)
# yolov3.weights, yolov3.cfg, sort.py가 전혀 필요 없습니다.
# ============================================================


from ultralytics import YOLO
import cv2 as cv
import sys

# 1. YOLOv8 객체 탐지 모델 로드 (가볍고 빠른 nano 모델)
print("YOLOv8 모델 로딩 중...")
model = YOLO("yolov8n.pt")

# 2. 웹캠 캡처 초기화
cap = cv.VideoCapture(0)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

print("\n실시간 사람 추적을 시작합니다. 멈추려면 'q' 키를 누르세요.")

# 3. 실시간 프레임 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    # YOLOv8 내장 트래킹 실행 (sort.py, filterpy, cfg, weights 모두 불필요)
    # classes=[0]: 사람만 타겟팅 / persist=True: 프레임 간 추적 ID 유지
    results = model.track(
        source=frame,
        classes=[0],
        persist=True,
        tracker="botsort.yaml",  # BoT-SORT 알고리즘 사용 (기본값)
        verbose=False            # 터미널 로그 숨김
    )

    # 바운딩 박스와 추적 ID(Track ID)가 자동으로 그려진 프레임 추출
    annotated_frame = results[0].plot()

    # 4. 화면 출력
    cv.imshow('Person tracking by YOLOv8', annotated_frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

    # 5. 자원 해제
cap.release()
cv.destroyAllWindows()