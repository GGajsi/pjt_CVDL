# 9장 인식 — 프로그램별 쉬운 설명서 (수정본 반영)

> 주 교재: 오일석, 『컴퓨터 비전과 딥러닝』 **9장 인식(363쪽부터)**  
> 적용 소스: **9-1.py~9-4.py는 기존 소스**, **9-4-수정.py는 학습/모델 재사용을 지원하는 개선 소스**, **9-4_predict.py는 저장 모델 예측을 확인하는 별도 추가 소스**, **9-5_1.py~9-9_1.py는 PixelLib를 대체하여 수정한 소스**입니다.  
> 이 문서의 페이지는 **교재의 해당 원본 프로그램 위치**를 가리키며, 수정본 소스 자체가 교재에 그대로 실려 있다는 뜻은 아닙니다.

## 1. 프로그램별 학습 주제

| 프로그램 | 학습 주제 | 교재 원본 위치 | 필수 요소 / 준비물 | 적용 버전 |
|---|---|---:|---|---|
| 9-1.py | YOLOv3로 사진 속 물체 검출 | 387쪽 | `yolov3.cfg`, `yolov3.weights`, `coco_names.txt`, `soccer.jpg` | 기존 |
| 9-2.py | YOLOv3로 웹캠 영상 속 물체 검출 | 391쪽 | YOLOv3 모델 파일, 웹캠 | 기존 |
| 9-3.py | YOLOv3 웹캠 물체 검출의 처리 속도(FPS) 측정 | 392쪽 | YOLOv3 모델 파일, 웹캠 | 기존 |
| 9-4.py | Oxford Pets 영상과 정답 마스크로 픽셀별 영역 분할 학습 | 405쪽 | Oxford Pets `images`, `trimaps`, TensorFlow/Keras | 기존 |
| **9-4-수정.py** | **Oxford Pets 분할 모델을 새로 학습하거나 기존 모델을 재사용하고, 4개 화면과 픽셀 정확도로 비교** | **405쪽 관련 수정 실습** | Oxford Pets 이미지·trimaps, TensorFlow/Keras, OpenCV, NumPy, 기존 모델 또는 새 학습 | **수정** |
| **9-4_predict.py** | **저장된 Oxford Pets 분할 모델로 예측하고 정답 마스크와 비교** | **405쪽 관련 추가 실습** | `oxford_seg.h5`, Oxford Pets `images`, `trimaps`, TensorFlow/Keras, OpenCV, NumPy | **추가** |
| 9-5_1.py | **SegFormer**로 정지 영상의 의미 분할 | 411쪽(원본 9-5) | `busy_street.jpg`, Transformers, PyTorch, Pillow, OpenCV, 인터넷(최초 모델 다운로드) | **수정** |
| 9-6_1.py | **SegFormer**로 웹캠 의미 분할 및 MP4 저장 | 412쪽(원본 9-6) | 웹캠, Transformers, PyTorch, OpenCV, MP4 코덱 | **수정** |
| 9-7_1.py | **YOLOv8s-seg**로 사진 분할, 컬러 포커스, 블러, 자동차 크롭 | 414쪽(원본 9-7) | `busy_street.jpg`, Ultralytics, OpenCV | **수정** |
| 9-8_1.py | **YOLOv8s-seg**로 웹캠에서 사람과 책만 인스턴스 분할 | 415쪽(원본 9-8) | 웹캠, Ultralytics, OpenCV | **수정** |
| 9-9_1.py | **PyQt5 + YOLOv8n-seg**로 사람은 유지하고 배경 흐림·단색 변경 | 417쪽(원본 9-9) | 웹캠, PyQt5, Ultralytics, OpenCV, NumPy | **수정** |

※ 교재 페이지는 원본 실습을 찾아보기 위한 안내입니다. 패키지 대체와 코드 개선은 **최신 보충 내용**입니다.

## 2. 오늘의 학습 목표

1. 영상 **분류·검출·의미 분할·인스턴스 분할**의 차이를 설명한다.
2. YOLOv3가 사진과 웹캠에서 물체를 찾는 흐름을 이해하고 FPS를 계산한다.
3. Oxford Pets의 **원본 이미지와 정답 마스크**로 픽셀 분할 학습이 가능함을 이해한다.
4. **9-4-수정.py**의 `TRAIN_NEW_MODEL`로 새 학습과 기존 모델 불러오기를 선택하고, 4개 결과 및 픽셀 정확도를 확인한다.
5. **9-4_predict.py**에서 학습된 모델의 **예측 마스크와 정답 마스크**를 나란히 비교할 수 있다.
6. SegFormer의 픽셀별 예측 결과를 색상 마스크로 표현한다.
7. YOLOv8이 검출한 **개별 객체 마스크**를 이용해 포커스·블러·크롭 기능을 만든다.
8. PyQt5의 `QTimer`와 OpenCV를 이용해 배경 변경 프로그램을 구성한다.

## 3. 권장 학습 순서

1. **9-1 → 9-2 → 9-3:** 정지 영상 물체 검출 → 웹캠 검출 → FPS 측정
2. **9-4 → 9-4-수정 → 9-4_predict:** 교재 원본 구조 이해 → 수정본에서 새 학습 또는 모델 재사용·4개 결과 확인 → 기존 `.h5` 모델을 별도 예측 프로그램으로 비교. 단, `.keras` 파일만 있을 때 `9-4_predict.py`의 모델 경로를 수정해야 합니다.
3. **9-5_1 → 9-6_1:** SegFormer를 활용해 사진·웹캠의 **의미 분할** 체험
4. **9-7_1 → 9-8_1:** YOLOv8을 활용해 사진의 **5가지 효과**와 웹캠의 **사람·책 인스턴스 분할** 체험
5. **9-9_1:** PyQt5에서 선택한 효과로 **사람을 제외한 배경** 변경하기

### 전체 처리 흐름

```text
영상 입력(사진/웹캠)
       │
       ├─ YOLOv3 (9-1~9-3) ──> 물체 종류 + 바운딩 박스 + FPS
       │
       ├─ Oxford Pets (9-4) ──> 정답 마스크로 분할 모델 학습 → oxford_seg.h5 저장
       │                  ├─ 9-4-수정 → 학습/모델 재사용 → 색상 마스크·오버레이·픽셀 정확도
       │                  │             (새 학습 시 models/oxford_seg.keras 저장)
       │                  └─ 9-4_predict → 기존 .h5 모델로 원본·정답·AI 예측 3창 비교
       │
       ├─ SegFormer (9-5_1, 9-6_1) ──> 픽셀별 의미 분할 → 색상 마스크/오버레이
       │
       └─ YOLOv8-seg (9-7_1~9-9_1) ──> 객체별 마스크
                                            ├─ 컬러 포커스·블러·자동차 크롭
                                            ├─ 사람·책 필터링
                                            └─ PyQt5 배경 효과
```

## 4. 꼭 구분할 개념

| 개념 | 쉬운 설명 | 결과 예시 |
|---|---|---|
| **분류(Classification)** | 사진 **전체**가 무엇인지 판단 | '강아지 사진' |
| **검출(Object Detection)** | 물체의 **종류와 위치**를 네모 상자로 표시 | 사람 2명, 자동차 1대의 박스 |
| **의미 분할(Semantic Segmentation)** | **픽셀마다** 어느 종류인지 표시 | 모든 사람 영역을 같은 종류로 표시 |
| **인스턴스 분할(Instance Segmentation)** | 같은 종류라도 **각 물체를 따로** 구분 | 사람 A와 사람 B를 각각 구분 |
| **마스크(Mask)** | 효과를 적용할 픽셀의 위치를 표시한 지도 | 사람 영역=255, 나머지=0 |
| **오버레이(Overlay)** | 원본 영상 위에 색상 마스크를 섞어 표시 | 원본+분할 색상 |
| **크롭(Crop)** | 물체가 있는 영역만 잘라내기 | 자동차 사진 별도 저장 |

### 핵심 식 — FPS

\[
FPS = \frac{N}{T}
\]

- **FPS:** 초당 처리한 프레임 수
- **N:** 처리한 프레임 수
- **T:** 경과 시간(초)

예: 100장을 20초에 처리했다면 `FPS = 100 / 20 = 5`입니다.

## 5. 실행 환경과 설치

Windows 11 / IntelliJ / Python 가상환경(`.venv`)을 가정합니다. **프로젝트의 가상환경을 활성화한 터미널**에서 설치하세요.

```powershell
python -m pip install numpy opencv-python matplotlib
python -m pip install tensorflow
python -m pip install torch transformers pillow
python -m pip install ultralytics PyQt5
```

- **9-1~9-3:** OpenCV와 YOLOv3 모델 파일이 필요합니다.
- **9-4:** TensorFlow/Keras와 Oxford Pets 데이터셋이 필요합니다.
- **9-4-수정:** `TRAIN_NEW_MODEL=False`가 기본으로, `models/oxford_seg.keras`를 우선 로드하고 없으면 **현재 작업 디렉터리의** `oxford_seg.h5`를 찾습니다. `True`이면 최대 30에포크 학습하며 최고 검증 손실 모델을 `models/oxford_seg.keras`에 저장합니다.
- **9-4_predict:** 9-4 학습 후 저장된 **`oxford_seg.h5`** 파일이 필요하며, 학습 없이 저장 모델로 예측만 수행합니다. `9-4_predict.py`는 프로젝트 루트를 `Path(__file__).resolve().parents[2]`로 계산하므로 **`source/ch09/` 아래에 배치하는 구성**을 전제로 합니다.
- **9-5_1~9-6_1:** Hugging Face의 `nvidia/segformer-b0-finetuned-ade-512-512`를 첫 실행 시 다운로드합니다.
- **9-7_1~9-8_1:** `yolov8s-seg.pt`를 사용합니다.
- **9-9_1:** `yolov8n-seg.pt`를 사용합니다.
- **수정본 9-5_1~9-9_1에는 PixelLib를 설치하거나 기존 PixelLib용 `.h5`/`.pb` 파일을 준비할 필요가 없습니다.**

※ 설치 성공 여부는 Python 및 기존 패키지 버전에 따라 달라질 수 있습니다. 패키지 충돌이 발생하면 새 가상환경에서 재현해 보세요.

## 6. 프로그램별 핵심 코드 읽기

| 소스 | 핵심 코드 | 역할 |
|---|---|---|
| 9-1 | `cv.dnn.readNet(...)` | YOLOv3 모델 읽기 |
| 9-2 | `cv.VideoCapture(0, cv.CAP_DSHOW)` | Windows 웹캠 열기 |
| 9-3 | `n_frame / (end-start)` | 처리 FPS 계산 |
| 9-4 | `model.fit(train_gen, ...)` | 정답 분할 마스크로 학습 |
| **9-4-수정** | `TRAIN_NEW_MODEL = False` | 새 학습 없이 저장된 모델을 로드할지 결정 |
| **9-4-수정** | `pairs = list(zip(img_paths, label_paths))` | 사진과 정답 마스크의 대응을 보존한 채 섞기 |
| **9-4-수정** | `ModelCheckpoint(...)`, `EarlyStopping(...)` | 검증 기준으로 모델 저장 및 조기 종료 |
| **9-4-수정** | `np.mean(true_mask == pred_mask) * 100` | 선택된 이미지 한 장의 픽셀 정확도 |
| **9-4_predict** | `keras.models.load_model(MODEL_PATH, compile=False)` | 학습된 분할 모델 로딩(재학습하지 않음) |
| **9-4_predict** | `cv.cvtColor(..., cv.COLOR_BGR2RGB)`, `np.expand_dims(...)` | 160×160 RGB 입력을 배치 차원과 함께 구성 |
| **9-4_predict** | `np.argmax(pred[0], axis=-1)` | 픽셀마다 3개 부류 중 점수가 가장 높은 부류 선택 |
| **9-4_predict** | `cv.INTER_NEAREST`, `ground_truth - 1` | 정답 마스크의 부류 값을 유지해 크기 변경하고 1·2·3을 0·1·2로 변환 |
| 9-5_1 | `SegformerForSemanticSegmentation.from_pretrained(...)` | 사전 학습 의미 분할 모델 읽기 |
| 9-5_1 | `logits.argmax(dim=1)` | 픽셀마다 점수가 가장 큰 종류 선택 |
| 9-6_1 | `cv.VideoWriter(...)`, `out.write(...)` | 의미 분할 결과 동영상 저장 |
| 9-7_1 | `result.masks.xy`, `cv.fillPoly(...)` | 객체 윤곽을 마스크로 변환 |
| 9-7_1 | `img_original[y1:y2, x1:x2]` | 자동차 영역 크롭 |
| 9-8_1 | `model.predict(..., classes=[0,73])` | 사람(0), 책(73)만 추론 대상으로 지정 |
| 9-9_1 | `QTimer.timeout.connect(...)` | 반복 프레임 처리 요청 |
| 9-9_1 | `np.where(mask_3d == 255, frame, bg)` | 사람은 원본으로 두고 배경만 바꾸기 |

### 9-4-수정.py 추가 실습 — 학습과 예측을 한 프로그램에서 실행

**선수 지식:** 9-4.py의 분할 모델과 Oxford Pets의 이미지·정답 마스크가 무엇인지 알고 있어야 합니다. 이 소스는 원본 9-4의 구조를 바탕으로 실행 경로, 이미지-정답 연결, 저장 및 시각화를 개선한 버전입니다.

**쉬운 비유:** 학습은 시험을 위해 새로 공부하는 과정이고, 기존 모델 불러오기는 이미 공부한 학생에게 시험 문제만 다시 풀게 하는 과정입니다. `TRAIN_NEW_MODEL`은 둘 중 하나를 선택하는 스위치입니다.

**실제 처리 흐름**

```text
9-4-수정.py 실행 (source/ch09 배치 가정)
  ├─ 프로젝트 루트에서 Oxford Pets 이미지/trimaps 찾기
  ├─ 같은 파일명(stem)의 사진 + 마스크만 쌍으로 묶기
  ├─ 쌍을 함께 섞고 훈련 90% / 검증 10% 분리
  ├─ TRAIN_NEW_MODEL ?
  │     ├─ True  → 모델 생성·학습(최대 30 epoch)
  │     │          → 최적 가중치 models/oxford_seg.keras 저장·로드
  │     └─ False → models/oxford_seg.keras 우선 로드
  │                → 없으면 작업 디렉터리의 oxford_seg.h5 로드
  ├─ 검증 첫 배치에서 예측 수행
  ├─ 픽셀별 클래스 argmax → 정답과 비교해 픽셀 정확도 계산
  └─ 4개 창: 원본 / 정답 컬러 마스크 / 예측 컬러 마스크 / 오버레이
```

**경로 구조 및 주의 사항**

```text
pjt_CVDL/
├─ datasets/oxford_pets/
│  ├─ images/*.jpg
│  └─ annotations/trimaps/*.png
├─ models/
│  └─ oxford_seg.keras           # 9-4-수정.py 새 학습 시 저장
└─ source/ch09/
   ├─ 9-4.py
   ├─ 9-4-수정.py
   └─ 9-4_predict.py
```

`9-4-수정.py`에서 프로젝트 루트는 `Path(__file__).resolve().parents[2]`입니다. 반면 **기존 `.h5` 모델 경로는 `Path('oxford_seg.h5').resolve()`이므로 소스 위치가 아니라 IntelliJ의 작업 디렉터리(Working Directory)를 기준**으로 합니다. `9-4_predict.py`는 별도로 `source/ch09/oxford_seg.h5`를 기대하므로, 두 프로그램의 `.h5` 탐색 위치는 다를 수 있습니다.

**핵심 코드 — 단계별 확인**

1. 실행 모드 선택:

   ```python
   TRAIN_NEW_MODEL = False  # 이미 학습된 모델 사용
   # TRAIN_NEW_MODEL = True  # 새로 학습하고 .keras 저장
   ```

2. 이미지와 레이블 짝 유지:

   ```python
   pairs = list(zip(img_paths, label_paths))
   random.Random(1).shuffle(pairs)
   ```

   앞부분에서는 원본 사진 이름과 동일한 `.png`가 있는 경우만 사용합니다. 이를 쌍으로 섞어 서로 다른 사진과 정답 마스크가 잘못 연결되는 문제를 막습니다.

3. 정답 마스크 변환:

   ```python
   img = load_img(path, target_size=self.img_size,
                  color_mode='grayscale', interpolation='nearest')
   y[j] = np.expand_dims(np.asarray(img), 2)
   y[j] -= 1
   ```

   정답 값 1·2·3을 0·1·2로 바꾸고, 최근접 보간을 이용해 클래스 번호가 섞이지 않도록 합니다.

4. 예측과 정확도 계산:

   ```python
   x_batch, y_batch = test_gen[0]
   preds = model.predict(x_batch, verbose=0)
   pred_mask = np.argmax(preds[0], axis=-1).astype(np.uint8)
   true_mask = y_batch[0, :, :, 0].astype(np.uint8)
   pixel_accuracy = np.mean(true_mask == pred_mask) * 100
   ```

   `pixel_accuracy`는 **첫 번째 검증 이미지 한 장**에서 두 마스크의 픽셀 값이 같은 비율입니다. 검증 집합 전체 정확도나 IoU와는 다릅니다.

5. 결과 표시: 클래스 0(동물)·1(배경)·2(경계)을 각각 초록·회색·빨강의 BGR 색상으로 표시하고, `cv.addWeighted()`로 원본과 예측 색상 마스크를 절반씩 혼합합니다. 원본·정답·예측·오버레이 **4개 창**이 열립니다.

**학습 조건:** `TRAIN_NEW_MODEL=True`이면 `ModelCheckpoint(monitor='val_loss', save_best_only=True)`와 `EarlyStopping(patience=5, restore_best_weights=True)`를 사용합니다. `epochs=30`은 최대 횟수이며 조기 종료할 수 있습니다. **`False`인 경우에는 새로운 모델을 학습하지 않습니다.**

**주의할 점:** `test_gen[0]`은 검증 집합에서 최소 1배치(32장)가 만들어져야 사용할 수 있습니다. `__len__()`에서 나머지 배치를 버리므로 데이터가 아주 적으면 인덱스 오류가 발생할 수 있습니다. 또한 현 소스는 이미지-마스크 이름 대응을 검증하지만, 학습된 모델의 정확도/환경 호환성을 별도로 검증한 것은 아닙니다.

**9-4_predict.py와의 차이**

| 구분 | 9-4-수정.py | 9-4_predict.py |
|---|---|---|
| 모델 학습 | `TRAIN_NEW_MODEL=True`이면 가능 | 불가, 불러오기만 수행 |
| 모델 경로 | `models/oxford_seg.keras` 우선, 없으면 작업 디렉터리의 `.h5` | 소스 폴더의 `oxford_seg.h5` |
| 비교 이미지 | 검증 데이터 첫 배치의 첫 장 | 파일명순 첫 번째 이미지 |
| 결과 창 | 4개(원본·정답·예측·오버레이) | 3개(원본·정답·예측) |
| 수치 출력 | 이미지 한 장 픽셀 정확도 | 이미지 이름·예측 배열 크기 |

---

### 9-4_predict.py 추가 실습 — 저장 모델의 예측 결과 확인

**선수 지식:** 9-4.py에서 입력 이미지와 정답 마스크를 이용해 모델을 학습한 경험이 필요합니다.

**비유:** 9-4.py가 정답지를 보며 공부하는 과정이라면, 9-4_predict.py는 공부가 끝난 AI에게 사진을 보여 주고 **AI의 답안**을 **정답지**와 나란히 비교하는 과정입니다.

**실제 소스 처리 순서**

```text
source/ch09/9-4_predict.py 실행
        │
        ├─ source/ch09/oxford_seg.h5 로드
        ├─ datasets/oxford_pets/images/*.jpg 중 정렬상 첫 번째 사진 선택
        ├─ 같은 파일명의 datasets/oxford_pets/annotations/trimaps/*.png 선택
        ├─ 원본 → 160×160 → BGR에서 RGB로 변환 → (1,160,160,3)
        ├─ model.predict(x) → 픽셀별 3개 부류 점수
        ├─ np.argmax(..., axis=-1) → (160,160) 예측 마스크
        └─ 원본 / 정답 / 예측 3개 OpenCV 창 표시
```

**파일 경로:** 소스는 `Path(__file__).resolve().parents[2]`를 프로젝트 루트로 사용합니다. 따라서 소스가 `source/ch09/9-4_predict.py`에 있을 때 다음 구조를 기대합니다.

```text
pjt_CVDL/
├─ datasets/
│  └─ oxford_pets/
│     ├─ images/                # 사진 .jpg
│     └─ annotations/
│        └─ trimaps/            # 동일한 이름의 정답 .png
└─ source/
   └─ ch09/
      ├─ 9-4.py
      ├─ 9-4_predict.py
      └─ oxford_seg.h5          # 9-4 학습 결과
```

**핵심 코드(첨부 소스 발췌)**

```python
model = keras.models.load_model(MODEL_PATH, compile=False)
image_path = image_paths[0]
label_path = LABEL_DIR / (image_path.stem + '.png')

resized = cv.resize(original, (160, 160))
rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
x = np.expand_dims(rgb.astype(np.float32), axis=0)

pred = model.predict(x, verbose=0)
pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

ground_truth = cv.resize(ground_truth, (160, 160), interpolation=cv.INTER_NEAREST)
ground_truth = ground_truth - 1
```

- **`compile=False`:** 저장된 모델의 추론에 컴파일된 손실·지표를 요구하지 않습니다. 모델 구조와 가중치는 여전히 로드합니다.
- **`(1, 160, 160, 3)`:** 이미지 1장, 높이 160, 너비 160, RGB 채널 3개를 의미합니다.
- **`np.argmax`**: 각 화소에 대해 출력 점수가 가장 높은 부류의 번호를 고릅니다. 확률 자체나 정확도를 산출하는 함수는 아닙니다.
- **`INTER_NEAREST`:** 정답 마스크의 클래스 번호를 보간 과정에서 섞지 않도록 최근접 이웃 보간을 사용합니다.
- **`ground_truth - 1`:** 원본의 값 **1·2·3을 0·1·2로 이동**합니다. 단, 원본에 0이 포함되어 있으면 `uint8`에서 255로 바뀔 수 있으므로 레이블 값 확인이 필요합니다.
- **`* 127`:** 0·1·2의 레이블을 0·127·254의 밝기로 바꿔 사람이 구분하기 쉽게 표시합니다.

**출력:** `Original Image`, `Ground Truth Mask`, `AI Prediction Mask`의 **3개 창**을 표시하고 콘솔에 이미지 이름 및 모델 출력 배열 크기를 출력합니다. **픽셀 정확도나 IoU 수치를 계산하는 코드는 포함되지 않습니다.**

**실습 주의:** 이 소스는 정렬한 사진 중 첫 장(`image_paths[0]`)을 선택합니다. 정답 비교 화면이 실제로 얼마나 일치하는지 살펴보되, 한 장의 시각화만으로 모델 전체 정확도를 판단하지 마세요. `oxford_seg.h5`는 9-4.py 학습 시 저장한 모델이므로 실행 전에 반드시 존재해야 합니다.

---

## 7. 원본 교재 대비 수정 사항 — 최신 보충 내용

| 원본 | 수정본 | 바뀐 이유·기능 |
|---|---|---|
| 9-4 분할 모델 학습 | **9-4-수정.py 개선** | 학습/재사용 스위치, 경로 개선, `.keras` 체크포인트, 4개 결과 창 및 한 장 픽셀 정확도 |
| 9-4 분할 모델 학습 | **9-4_predict.py 추가** | 학습된 `oxford_seg.h5` 불러오기 및 원본·정답·예측 비교 |
| 9-5 PixelLib + ADE20K DeepLabv3 | **9-5_1 SegFormer** | Hugging Face 모델로 정지 영상 의미 분할 |
| 9-6 PixelLib 카메라 의미 분할 | **9-6_1 SegFormer + VideoWriter** | 웹캠 마스크·오버레이와 MP4 저장 |
| 9-7 PixelLib Mask R-CNN | **9-7_1 YOLOv8s-seg** | 객체 분할 및 **5개 결과 창**, 자동차 잘라 저장 |
| 9-8 PixelLib Mask R-CNN 카메라 | **9-8_1 YOLOv8s-seg** | 사람과 책만 선택해 웹캠 인스턴스 분할 |
| 9-9 PixelLib `alter_bg` + `while True` | **9-9_1 YOLOv8n-seg + QTimer** | 사람 마스크를 이용해 배경 흐림/색상 변경 |

> **구분:** 9-1~9-4의 기존 교재 기반 내용, 9-4-수정의 개선 실습, 9-4_predict의 추가 실습, 9-5_1~9-9_1의 최신 대체 구현을 혼동하지 마세요. 수정본은 새로운 모델을 사용하므로 추론 결과나 성능이 원본과 동일하지 않습니다.

## 8. 실행 전 점검 및 자주 발생하는 문제

| 현상 | 원인 확인 | 수정·대응 방향 |
|---|---|---|
| `9-4-수정: 학습된 모델 없음` | `models/oxford_seg.keras`와 작업 디렉터리의 `oxford_seg.h5` 둘 다 없는지 확인 | 기존 모델 경로를 확인하거나 `TRAIN_NEW_MODEL=True`로 학습 |
| `9-4-수정: test_gen[0] IndexError` | 검증 데이터 크기와 `batch_siz=32`, `__len__`의 몫 계산 확인 | 검증 이미지 32장 이상 확보하거나 배치 처리 코드 조정 |
| `9-4_predict: oxford_seg.h5 없음` | 9-4.py 학습과 체크포인트 저장 완료 여부 및 `source/ch09/` 경로 확인 | 학습된 모델을 해당 위치에 놓은 뒤 실행 |
| `9-4_predict: 이미지를 찾을 수 없습니다` | `datasets/oxford_pets/images/*.jpg` 경로와 압축 해제 구조 확인 | 실제 이미지 폴더를 확인하고 경로 수정 |
| `9-4_predict: 정답 마스크 없음` | `annotations/trimaps`에 같은 이름의 `.png`가 있는지 확인 | 데이터셋 라벨 위치 확인 |
| `9-4_predict: 모델 로딩 오류` | Keras/TensorFlow 버전과 `.h5` 모델 저장 형식 확인 | 학습한 환경과 호환되는 버전으로 로드 여부 점검 |
| `FileNotFoundError: busy_street.jpg` | `9-5_1.py`, `9-7_1.py`와 이미지 위치 확인 | 소스와 같은 폴더에 사진을 배치 |
| 모델 다운로드 실패 | 인터넷 연결, Hugging Face/GitHub 접근 여부 확인 | 네트워크와 다운로드 캐시 확인 |
| `ModuleNotFoundError` | 실행 인터프리터와 설치 패키지 확인 | 해당 `.venv`에서 패키지 설치 |
| 웹캠이 열리지 않음 | 카메라 앱 점유, 웹캠 번호, 권한 확인 | 카메라 앱 종료 후 `VideoCapture(0)` 재시도 |
| CUDA가 사용되지 않음 | `torch.cuda.is_available()` 확인 | CPU 실행 또는 별도 GPU 지원 환경 준비 |
| 9-6_1의 MP4 재생 속도가 실제와 다름 | 저장 FPS가 **고정 5.0**이고 실제 추론 FPS를 측정하지 않음 | 영상 저장 FPS와 실제 프레임 처리 속도를 구분하여 검토 |
| 9-7_1의 자동차 크롭 오류 | 박스가 영상 밖으로 벗어났거나 크롭 높이가 0인지 확인 | 좌표를 영상 범위로 제한하고 빈 크롭 건너뛰기 |
| 9-9_1 GUI가 느리거나 멈춘 듯 보임 | `QTimer` 콜백에서 YOLO 추론이 오래 걸릴 수 있음 | 해상도를 줄이거나 추론을 별도 스레드로 분리 |

### 프로그램별 파일 생성·표시

- **9-4_predict:** 저장된 `oxford_seg.h5`를 읽고 원본 사진·정답 마스크·AI 예측 마스크를 **3개 창**에 출력합니다. 새로운 모델은 저장하지 않습니다.

- **9-5_1:** 원본·컬러 마스크·오버레이 **3개 창**을 출력합니다.
- **9-6_1:** 분할 영상을 출력하고 `output_video.mp4`를 **현재 작업 디렉터리**에 저장합니다.
- **9-7_1:** 원본·기본 분할·포커스·블러·자동차 크롭 **5개 창**을 출력하며, 자동차를 소스 옆 `cars/car_1.jpg` 등으로 저장합니다.
- **9-8_1:** 웹캠에서 사람·책만 분할하여 표시합니다. 기본 소스는 동영상을 저장하지 않습니다.
- **9-9_1:** GUI의 효과 선택값으로 사람 외의 배경을 변화시킵니다. 기본 소스는 결과 동영상을 저장하지 않습니다.

## 9. 확인 문제 3개 (힌트만 제공)

1. **의미 분할과 인스턴스 분할은 무엇이 다를까요?**  
   *힌트: 사람 3명이 나오는 사진을 생각해 보세요.*
2. **9-7_1에서 사람과 자전거만 컬러로 남기려면 어떤 자료를 이용할까요?**  
   *힌트: `mask_focus`, `cv.fillPoly`를 살펴보세요.*
3. **9-9_1에서 배경만 바꿀 수 있는 이유는 무엇일까요?**  
   *힌트: `np.where()`의 조건과 `mask_3d` 값을 보세요.*

## 10. 미니 과제

**과제: 9-7_1.py의 포커스 대상에 자동차도 추가하기**

`if class_name in ['person', 'bicycle']:` 구문을 변경하여 자동차도 컬러로 남기고, 실행 전·후 화면을 비교해 보세요. 실행 결과에서 **원본·포커스 화면 2개를 캡처**하고 변경한 코드를 설명합니다.

## 11. 오늘의 핵심 요약

9장 원본은 **물체 검출 → 영역 분할 → 비전 에이전트**로 확장됩니다. **9-4_predict.py를 추가하여 9-4의 학습 결과를 저장 모델로 다시 예측하고 정답과 비교할 수 있습니다.** 기존 9-1~9-4는 그대로 학습하고, 9-5~9-9는 **SegFormer 및 YOLOv8으로 대체된 `*_1.py`**를 사용합니다. 의미 분할은 픽셀마다 종류를 구분하고, 인스턴스 분할은 같은 종류의 개별 객체를 구분합니다. 각 프로그램의 **모델·입력·출력·파일 저장 위치**를 확인하면서 학습하세요.

---

**자료 상태:** 첨부된 소스의 정적 코드 분석에 기초한 학습 안내서입니다. 이 문서 작성 과정에서 모든 프로그램의 실제 실행 성공과 모델 정확도를 검증한 것은 아닙니다.
