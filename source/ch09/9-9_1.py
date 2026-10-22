# ============================================================
# 9-9_1.py (수정본)
# PyQt5 GUI + YOLOv8 실시간 배경 변환 (GUI 프리징 해결)
# ============================================================

# [핵심 해결] PyTorch 기반 모듈을 가장 먼저 import 합니다!
from ultralytics import YOLO

# 그 다음 나머지 모듈들을 import 합니다.
import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer

class VideoSpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('배경을 내 맘대로 (YOLOv8 버전)')
        self.setGeometry(200, 200, 450, 100)

        # 1. YOLOv8 모델 로드 (초기화 단계에서 미리 로드하여 딜레이 방지)
        print("YOLOv8 모델 로딩 중...")
        self.model = YOLO("yolov8n-seg.pt")

        # 2. UI 컴포넌트 구성
        videoButton = QPushButton('배경 내 맘대로 켜기', self)
        self.pickCombo = QComboBox(self)
        self.pickCombo.addItems(['원래 영상', '흐릿(조금)', '흐릿(중간)', '흐릿(많이)', '빨강', '녹색', '파랑'])
        quitButton = QPushButton('나가기', self)

        videoButton.setGeometry(10, 10, 150, 30)
        self.pickCombo.setGeometry(170, 10, 120, 30)
        quitButton.setGeometry(300, 10, 100, 30)

        videoButton.clicked.connect(self.start_video)
        quitButton.clicked.connect(self.quitFunction)

        # 3. 비디오 처리를 위한 타이머 및 캡처 객체 초기화
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_video(self):
        # 웹캠 켜기 및 타이머 시작 (while True 루프 대체)
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "에러", "카메라 연결 실패")
                return

        # 약 33ms 마다(초당 약 30프레임) update_frame 함수 호출
        self.timer.start(33)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        pick_effect = self.pickCombo.currentIndex()

        if pick_effect == 0:
            special_img = frame
        else:
            # YOLOv8 추론: 사람(class 0)만 탐지
            results = self.model.predict(source=frame, classes=[0], conf=0.5, verbose=False)
            result = results[0]

            # 사람 영역 마스크 생성
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            if result.masks is not None:
                for poly in result.masks.xy:
                    cv.fillPoly(mask, [poly.astype(np.int32)], 255)

            # 마스크 차원 확장 (합성을 위함)
            mask_3d = mask[:, :, np.newaxis]

            if pick_effect in [1, 2, 3]:
                # 배경 블러 처리
                k = {1: 15, 2: 31, 3: 51}[pick_effect] # 커널 크기 매핑
                bg = cv.GaussianBlur(frame, (k, k), 0)
                # np.where: 마스크가 255(사람)이면 원본, 아니면 배경(블러) 적용
                special_img = np.where(mask_3d == 255, frame, bg)

            elif pick_effect in [4, 5, 6]:
                # 단색 배경 처리 (OpenCV는 BGR 포맷 사용)
                bg = np.zeros_like(frame)
                if pick_effect == 4: bg[:] = (0, 0, 255)   # 빨강
                elif pick_effect == 5: bg[:] = (0, 255, 0) # 녹색
                elif pick_effect == 6: bg[:] = (255, 0, 0) # 파랑
                special_img = np.where(mask_3d == 255, frame, bg)

        cv.imshow('Special effect', special_img)

    def quitFunction(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        cv.destroyAllWindows()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = VideoSpecialEffect()
    win.show()
    sys.exit(app.exec_())