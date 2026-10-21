# ============================================================
# 10-7_mediapipe_mesh.py (촘촘한 468 포인트 페이스 메시)
# ============================================================
import os
# 보기 싫은 oneDNN C++ 경고 메시지 강제 숨김
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2 as cv
import mediapipe as mp

# 1. MediaPipe Face Mesh 초기화
mp_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# FaceMesh 객체 생성 (얼굴 윤곽 및 눈동자 추적 활성화)
mesh = mp_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 초기화 (윈도우 환경 딜레이 및 에러 방어)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("MediaPipe 페이스 메시를 시작합니다. 창을 닫으려면 'q' 키를 누르세요.")

# 3. 실시간 추론 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # 직관적인 조작을 위해 AI 모델에 넣기 전 프레임을 먼저 좌우 반전
    frame = cv.flip(frame, 1)

    # BGR 이미지를 RGB로 변환하여 MediaPipe에 전달
    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # 4. 얼굴 랜드마크가 검출된 경우 촘촘한 그물망 그리기
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            # (1) 얼굴 전체 그물망 (Tesselation: 촘촘한 다각형 메시)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            # (2) 눈썹, 입술 등의 주요 윤곽선 (Contours)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
            # (3) 눈동자 홍채 추적 (Irises)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
            )

    cv.imshow('Dense Face Mesh by MediaPipe', frame)

    if cv.waitKey(5) == ord('q'):
        print("사용자에 의해 종료되었습니다.")
        break

# 5. 자원 해제
cap.release()
cv.destroyAllWindows()