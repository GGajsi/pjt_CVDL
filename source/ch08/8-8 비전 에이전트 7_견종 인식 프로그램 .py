
import sys
import pickle
import pathlib

import cv2 as cv
import numpy as np
import tensorflow as tf
import winsound

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel
)


# ============================================
# 1. 모델과 견종 이름 불러오기
# ============================================

BASE_DIR = pathlib.Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / 'cnn_for_stanford_dogs.keras'
NAMES_PATH = BASE_DIR / 'dog_species_names.pkl'

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"학습 모델이 없습니다: {MODEL_PATH}\n"
        "먼저 8-7.py를 실행하세요."
    )

if not NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"견종 이름 파일이 없습니다: {NAMES_PATH}"
    )

cnn = tf.keras.models.load_model(str(MODEL_PATH))

with open(NAMES_PATH, 'rb') as f:
    dog_species = pickle.load(f)

print("모델 로딩 완료")
print("인식 가능한 견종:", len(dog_species))


# ============================================
# 2. PyQt5 프로그램
# ============================================

class DogSpeciesRecognition(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('AI 강아지 품종 인식')
        self.setGeometry(200, 200, 650, 150)

        self.img = None

        # 사진 열기 버튼
        fileButton = QPushButton('강아지 사진 열기', self)
        fileButton.setGeometry(20, 20, 150, 40)

        # 품종 인식 버튼
        recognitionButton = QPushButton('품종 인식', self)
        recognitionButton.setGeometry(190, 20, 150, 40)

        # 종료 버튼
        quitButton = QPushButton('나가기', self)
        quitButton.setGeometry(360, 20, 150, 40)

        # 상태 표시
        self.statusLabel = QLabel('사진을 선택해 주세요.', self)
        self.statusLabel.setGeometry(20, 80, 600, 35)

        # 버튼 이벤트 연결
        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)


    # ========================================
    # 3. 강아지 사진 불러오기
    # ========================================

    def pictureOpenFunction(self):

        fname, _ = QFileDialog.getOpenFileName(
            self,
            '강아지 사진 선택',
            str(BASE_DIR),
            'Images (*.jpg *.jpeg *.png *.bmp)'
        )

        if not fname:
            return

        # 한글 파일 경로도 읽을 수 있도록 처리
        raw = np.fromfile(fname, dtype=np.uint8)
        self.img = cv.imdecode(raw, cv.IMREAD_COLOR)

        if self.img is None:
            QMessageBox.warning(
                self,
                '오류',
                '이미지 파일을 읽을 수 없습니다.'
            )
            return

        cv.imshow('Dog image', self.img)

        self.statusLabel.setText('사진을 불러왔습니다.')


    # ========================================
    # 4. 강아지 품종 인식
    # ========================================

    def recognitionFunction(self):

        if self.img is None:
            QMessageBox.warning(
                self,
                '안내',
                '먼저 강아지 사진을 선택하세요.'
            )
            return

        # OpenCV BGR -> RGB 변환
        img_rgb = cv.cvtColor(
            self.img,
            cv.COLOR_BGR2RGB
        )

        # 모델 입력 크기 224 x 224
        x = cv.resize(img_rgb, (224, 224))

        # 배치 차원 추가
        x = np.reshape(x, (1, 224, 224, 3))

        # 실수형 변환
        x = x.astype(np.float32)

        # 학습 모델로 예측
        # 전처리는 저장된 모델 내부에서 수행
        res = cnn.predict(x, verbose=0)[0]

        # 예측 확률 상위 5개
        top5 = np.argsort(res)[::-1][:5]

        print("\n===== 견종 인식 결과 =====")

        # 원본 이미지 복사
        result_img = self.img.copy()

        for i, idx in enumerate(top5):

            species = dog_species[idx]

            # n02085620-Chihuahua -> Chihuahua
            name = species.split('-', 1)[-1]

            probability = res[idx] * 100

            print(
                f"{i+1}위: {name} "
                f"({probability:.2f}%)"
            )

            text = f"{name}: {probability:.1f}%"

            cv.putText(
                result_img,
                text,
                (10, 30 + i * 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

        cv.imshow('Recognition result', result_img)

        best_species = dog_species[top5[0]].split('-', 1)[-1]

        self.statusLabel.setText(
            f"예측 품종: {best_species}"
        )

        winsound.Beep(1000, 300)


    # ========================================
    # 5. 프로그램 종료
    # ========================================

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()


# ============================================
# 6. 프로그램 실행
# ============================================

app = QApplication(sys.argv)

win = DogSpeciesRecognition()
win.show()

sys.exit(app.exec_())
