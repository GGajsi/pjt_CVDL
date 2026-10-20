# ============================================================
# 9-8_1 사람과 책만 선택해 웹캠 인스턴스 분할.py (수정본)
# 웹캠 실시간 특정 객체(사람, 책) 인스턴스 분할 (YOLOv8)
# 필요 패키지: pip install ultralytics opencv-python
# ============================================================

from ultralytics import YOLO
import cv2 as cv

# 1. YOLOv8 분할 모델 로드 (속도와 정확도의 밸런스가 좋은 Small 모델 추천)
print("YOLOv8 Segmentation 모델 로딩 중...")
model = YOLO("yolov8s-seg.pt")

# 2. 타겟 클래스 설정 (기존 person=True, book=True 기능)
# YOLOv8(COCO 데이터셋)의 클래스 ID: 0 = person(사람), 73 = book(책)
target_classes = [0, 73]

# 3. 웹캠 캡처 초기화
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")

print("\n실시간 인스턴스 분할을 시작합니다. 화면을 종료하려면 키보드의 'q'를 누르세요.")

# 4. 실시간 프레임 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다. 종료합니다.")
        break

    # 이미지 추론 실행
    # classes 파라미터에 리스트를 전달하여 원하는 객체만 필터링 (사람, 책)
    results = model.predict(
        source=frame,
        classes=target_classes,
        conf=0.4,       # 신뢰도 임계값 (필요에 따라 조절)
        save=False,
        verbose=False   # 터미널에 프레임마다 로그가 찍히는 것을 방지
    )

    # 5. 결과 시각화 (바운딩 박스 및 컬러 마스크가 입혀진 프레임)
    annotated_frame = results[0].plot()

    # 6. 화면 출력 (기존 frame_name='Pixellib' 대체)
    cv.imshow('YOLOv8 Instance Segmentation (Person & Book)', annotated_frame)

    # 'q' 키를 누르면 루프 탈출
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("\n사용자에 의해 종료되었습니다.")
        break

# 7. 자원 해제
cap.release()
cv.destroyAllWindows()