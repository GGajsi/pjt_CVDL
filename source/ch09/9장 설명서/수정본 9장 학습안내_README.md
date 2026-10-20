# 9장 수정본 프로그램 학습 안내서 (README)

> **교재:** 『컴퓨터 비전과 딥러닝』, 9장 **인식**  
> **대상:** 컴퓨터 비전 입문 학생 · Python/OpenCV 실습  
> **기준 소스:** `9-5_1.py`, `9-6_1.py`, `9-7_1.py`, `9-8_1.py`, `9-9_1.py` (총 5개)  
> **주의:** 아래 프로그램들은 교재의 PixelLib 기반 예제를 **SegFormer/YOLOv8로 대체한 수정본**입니다. 교재에 직접 수록된 코드·실행 결과와 동일하다는 뜻은 아닙니다. 해당 변경은 **최신 보충 내용**으로 구분합니다.

---

## 1. 선수 지식 확인

실습 전에 다음 질문에 답할 수 있는지 확인하세요.

- 이미지를 `cv.imread()`로 읽고 `cv.imshow()`로 출력할 수 있나요?
- `cv.VideoCapture(0)`으로 웹캠 프레임을 가져오는 방법을 아나요?
- AI가 **물체를 분류**하는 것과 **물체 영역을 분할**하는 것의 차이를 알고 있나요?
- Python 리스트, `for`, `if`, NumPy 배열을 사용할 수 있나요?

아직 어렵다면 2장(OpenCV 기초), 7~8장(딥러닝 및 CNN)을 먼저 복습하세요.

## 2. 9장 수정본 학습 목표

1. 의미 분할과 인스턴스 분할의 차이를 말할 수 있다.
2. 이미 학습된 SegFormer/YOLOv8 모델을 불러와 예측할 수 있다.
3. 마스크를 이용해 원본 이미지에서 특정 영역만 강조하거나 흐리게 만들 수 있다.
4. 사진과 웹캠 영상을 처리하고, 결과 영상을 저장할 수 있다.
5. PyQt5의 버튼·콤보박스·`QTimer`로 간단한 비전 프로그램을 만들 수 있다.

## 3. 제일 먼저 이해할 개념

### 3-1. 일상적인 비유

- **분류(Classification):** 사진 전체를 보고 “강아지 사진”이라고 말하기
- **객체 검출(Detection):** 사진 속 강아지가 **어디에 있는지** 네모 상자 그리기
- **의미 분할(Semantic Segmentation):** 강아지에 해당하는 **모든 픽셀**을 같은 색으로 칠하기
- **인스턴스 분할(Instance Segmentation):** 강아지가 여러 마리면 **각 강아지를 따로 구분해서** 칠하기

### 3-2. 처리 흐름

```text
사진 또는 웹캠 프레임
       ↓
AI 모델(SegFormer 또는 YOLOv8)
       ↓
픽셀별 마스크 / 객체별 마스크
       ↓
색상 합성 · 객체 강조 · 블러 · 크롭
       ↓
화면 출력 또는 동영상 저장
```

### 3-3. 중요한 용어

| 용어 | 쉽게 설명 |
|---|---|
| 마스크(mask) | 선택할 픽셀을 표시한 지도. 보통 0은 배경, 255는 선택 영역 |
| 클래스(class) | `person`, `car`처럼 객체의 종류 |
| 바운딩 박스(bounding box) | 객체를 둘러싼 사각형 |
| 신뢰도(`conf`) | 모델이 예측한 객체를 얼마나 확신하는지에 관한 점수 |
| 오버레이(overlay) | 원본 사진 위에 색 마스크를 반투명하게 겹친 결과 |
| 추론(inference) | 기존 모델로 입력 영상을 분석하는 단계. 이 프로그램들은 새 모델을 훈련하지 않음 |

## 4. 5개 프로그램 한눈에 비교

| 프로그램 | 주요 기능 | AI 모델 | 입력 | 결과 |
|---|---|---|---|---|
| **9-5_1** | 정지 이미지 **의미 분할** | SegFormer-B0, ADE20K | `busy_street.jpg` | 원본·컬러 마스크·오버레이 **3개 창** |
| **9-6_1** | 웹캠 **의미 분할** | SegFormer-B0, ADE20K | 웹캠 | 오버레이 화면·`output_video.mp4` 저장 |
| **9-7_1** | 정지 이미지 **인스턴스 분할 + 5가지 효과** | YOLOv8s-seg | `busy_street.jpg` | 원본·분할·포커스·블러·자동차 크롭 **5개 창**, `cars/` 저장 |
| **9-8_1** | 웹캠에서 **사람과 책**만 인스턴스 분할 | YOLOv8s-seg, COCO | 웹캠 | 사람·책의 마스크와 박스 표시 |
| **9-9_1** | 사람을 남기고 **배경 변경**하는 GUI | YOLOv8n-seg | 웹캠 | 원본 / 3단계 블러 / 빨강·초록·파랑 배경 |

**권장 학습 순서:** `9-5_1 → 9-6_1 → 9-7_1 → 9-8_1 → 9-9_1`

## 5. IntelliJ / Windows 실습 환경 준비

프로젝트 예시:

```text
pjt_CVDL/
├── .venv/
└── source/
    └── ch09/
        ├── busy_street.jpg          # 9-5_1, 9-7_1의 입력 이미지
        ├── 9-5_1.py
        ├── 9-6_1.py
        ├── 9-7_1.py
        ├── 9-8_1.py
        ├── 9-9_1.py
        ├── cars/                    # 9-7_1 실행 시 자동 생성
        └── 9장_수정본_프로그램_설명서/
            ├── 9-5_1_설명서.md
            ├── 9-6_1_설명서.md
            ├── 9-7_1_설명서.md
            ├── 9-8_1_설명서.md
            ├── 9-9_1_설명서.md
            └── 9장_수정본_README.md
```

> `busy_street.jpg` 파일은 실제로 소스 파일과 **같은 폴더**에 있어야 합니다. `9-6_1.py`의 `output_video.mp4`는 **실행 작업 디렉터리**에 저장됩니다(반드시 `ch09`는 아님).

### 5-1. 가상환경 확인

IntelliJ Terminal(Windows PowerShell)에서 프로젝트 루트 기준:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
```

### 5-2. 필요한 패키지

```powershell
python -m pip install ultralytics opencv-python numpy matplotlib
python -m pip install transformers torch pillow
python -m pip install PyQt5
```

- `9-5_1`, `9-6_1`: `transformers`, `torch`, `pillow`, `opencv-python`, `numpy`
- `9-7_1`, `9-8_1`: `ultralytics`, `opencv-python`, `numpy`
- `9-9_1`: `ultralytics`, `opencv-python`, `numpy`, `PyQt5`

패키지는 **현재 프로젝트의 `.venv` 인터프리터**에 설치해야 합니다. 최초 실행에는 사전 학습 가중치 다운로드 때문에 인터넷이 필요할 수 있습니다.

### 5-3. 실행 방법

```powershell
python .\source\ch09\9-5_1.py
python .\source\ch09\9-6_1.py
python .\source\ch09\9-7_1.py
python .\source\ch09\9-8_1.py
python .\source\ch09\9-9_1.py
```

IntelliJ 실행 버튼을 사용해도 됩니다. 웹캠은 다른 앱에서 사용 중이면 열리지 않을 수 있습니다.

## 6. 프로그램별 쉬운 핵심 설명

### 6-1. `9-5_1.py` — 거리 사진을 픽셀별로 분류

```text
busy_street.jpg → SegFormer → 픽셀별 클래스 번호(0~149)
                              ↓
                    색상표 적용 → 마스크
                              ↓
                 원본+마스크 합성 → 오버레이
```

핵심 코드:

```python
predicted_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
color_mask = palette[predicted_mask]
img_overlay = cv.addWeighted(img_cv, 0.6, color_mask_bgr, 0.4, 0)
```

- `argmax`: 각 픽셀에 가장 높은 점수를 받은 클래스 선택
- `palette[...]`: 클래스 번호를 눈에 띄는 색으로 바꾸기
- `addWeighted`: 원본 60% + 마스크 40%를 섞어 표시하기

**확인:** 같은 종류의 물체는 대체로 같은 색을 갖지만, 동일한 클래스의 개별 객체를 분리해서 세는 코드는 아닙니다.

### 6-2. `9-6_1.py` — 웹캠 프레임을 연속 분할하고 저장

```text
웹캠 읽기 → BGR→RGB → SegFormer 추론
                        ↓
           원본 크기로 마스크 복원
                        ↓
           오버레이 표시 + MP4 저장
```

핵심 코드:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
out = cv.VideoWriter(output_name, fourcc, fps, (width, height))
out.write(overlay_frame)
```

- CUDA를 PyTorch에서 사용할 수 있으면 GPU, 아니면 CPU로 추론합니다.
- `fps = 5.0`은 **저장 동영상의 재생 FPS 설정값**입니다. 실제 카메라 처리 속도가 정확히 5 FPS라는 뜻은 아닙니다.
- 종료 시 `cap.release()`와 `out.release()`로 자원을 해제해야 합니다.

### 6-3. `9-7_1.py` — 인스턴스 분할 결과로 5가지 효과 만들기

```text
busy_street.jpg → YOLOv8s-seg
                    ├─ 원본 출력
                    ├─ 탐지 박스+객체별 마스크
                    ├─ 사람·자전거만 컬러 강조
                    ├─ 사람·가방류만 블러
                    └─ 자동차 박스 크롭 → cars/car_*.jpg
```

핵심 코드:

```python
polygon = masks[i].astype(np.int32)
cv.fillPoly(mask_focus, [polygon], 255)
img_focus[mask_focus == 255] = img_original[mask_focus == 255]
```

**핵심 원리:** `mask_focus`의 255 위치에만 원본 컬러 픽셀을 복사하고, 나머지는 흑백으로 유지합니다.

자동차 영역은 `boxes.xyxy` 좌표를 사용해 자릅니다. 자동차가 검출되지 않으면 `No cars detected` 화면이 표시됩니다. 자동차 저장 파일은 같은 이름으로 재실행할 경우 덮어쓸 수 있습니다.

### 6-4. `9-8_1.py` — 사람·책만 실시간 분할

```python
target_classes = [0, 73]  # COCO: 사람, 책
results = model.predict(source=frame, classes=target_classes, conf=0.4)
annotated_frame = results[0].plot()
```

- `classes=[0,73]`: 원본 소스에서 **사람(person)**, **책(book)** 클래스만 표시
- `conf=0.4`: 점수가 기준보다 낮은 검출 결과는 제외
- `plot()`: 객체별 색상 마스크와 박스를 합친 화면 생성

`q`를 누르면 종료합니다. 이 코드 자체에는 별도의 동영상 저장 기능이 없습니다.

### 6-5. `9-9_1.py` — GUI 버튼으로 배경을 바꾸는 비전 에이전트

```text
PyQt5 버튼 클릭 → 웹캠 시작 → QTimer 주기적 호출
                                   ↓
                             사람 마스크 생성
                                   ↓
                원본 / 배경 블러 / 단색 배경 합성
                                   ↓
                              OpenCV 창 출력
```

핵심 코드:

```python
self.timer.timeout.connect(self.update_frame)
self.timer.start(33)
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
special_img = np.where(mask_3d == 255, frame, bg)
```

- `QTimer`: 무한 반복문 대신 주기적으로 다음 프레임 처리 함수를 호출
- `np.where`: **사람이면 원본 픽셀**, 아니면 새 배경 픽셀 선택
- 콤보박스: 원본, 블러 3단계, 배경색 3종류 선택

**주의:** 33ms는 타이머 요청 간격이며 **30FPS를 보장하지 않습니다**. `model.predict()`가 메인 GUI 스레드에서 오래 걸리면 창 반응이 느려질 수 있습니다. QTimer 사용은 기존 `while True`보다 이벤트 처리에 유리할 수 있지만, 완전한 비동기 추론 구조는 아닙니다.

## 7. 주요 코드/함수 빠른 사전

| 함수/속성 | 의미 | 주로 등장 |
|---|---|---|
| `from_pretrained()` | Hugging Face 사전 학습 모델 다운로드·불러오기 | 9-5_1, 9-6_1 |
| `model.predict()` | YOLO 모델 추론 | 9-7_1, 9-8_1, 9-9_1 |
| `F.interpolate()` | 예측 결과를 원본 해상도로 확대 | 9-5_1, 9-6_1 |
| `argmax()` | 각 픽셀에서 가장 높은 클래스 점수 선택 | 9-5_1, 9-6_1 |
| `result.masks.xy` | 각 객체의 다각형 외곽점 좌표 | 9-7_1, 9-9_1 |
| `cv.fillPoly()` | 마스크 다각형 내부 채우기 | 9-7_1, 9-9_1 |
| `cv.GaussianBlur()` | 이미지 부드럽게 흐리기 | 9-7_1, 9-9_1 |
| `cv.hconcat()` | 이미지 여러 장을 가로로 연결 | 9-7_1 |
| `cv.VideoWriter()` | 동영상 파일 기록 | 9-6_1 |
| `QTimer` | GUI 이벤트 루프 안에서 정기적으로 함수 호출 | 9-9_1 |

## 8. 자주 발생하는 문제: 원인 → 확인 → 해결

### 8-1. `FileNotFoundError: busy_street.jpg`

- **원인:** 사진이 소스 파일과 같은 폴더에 없음
- **확인:** `print(Path(__file__).resolve().parent)`와 탐색기 위치 비교
- **해결:** `busy_street.jpg`를 해당 폴더에 배치하거나 `IMAGE_PATH`를 실제 위치로 수정

### 8-2. `ModuleNotFoundError: ultralytics / transformers / PyQt5`

- **원인:** 실행 중인 `.venv`에 패키지가 없음
- **확인:** `python -m pip show ultralytics transformers PyQt5`
- **해결:** IntelliJ에서 현재 프로젝트 인터프리터를 확인한 후 해당 패키지 설치

### 8-3. 웹캠이 열리지 않음

- **원인:** 장치 번호가 다르거나 다른 앱이 카메라 사용 중
- **확인:** Windows 카메라 앱에서 동작 확인 후 종료
- **해결:** `cv.VideoCapture(0)`의 번호를 `1`로 시험하거나, Windows에서는 `cv.VideoCapture(0, cv.CAP_DSHOW)`로 시험

### 8-4. AI 모델이 느리거나 화면이 끊김

- **원인:** CPU 추론 시간이 프레임 간격보다 오래 걸림
- **확인:** CPU/GPU 사용률과 프레임 처리 시간 측정
- **해결:** 웹캠 해상도 축소, 추론 대상 프레임 수 조절, 소형 모델 사용. 필요 시 추론을 별도 스레드로 분리

### 8-5. 모델 다운로드 실패

- **원인:** 처음 다운로드 시 네트워크, 프록시, 저장 공간 등의 문제
- **확인:** 콘솔의 다운로드 주소와 예외 메시지 확인
- **해결:** 인터넷 연결 및 디스크 공간 확인 후 재실행, 방화벽 환경에서는 승인된 모델 파일을 로컬 경로로 사용

### 8-6. `output_video.mp4`가 재생되지 않거나 속도가 어색함

- **원인:** 비디오 코덱 지원, `VideoWriter` 열기 실패, 실제 추론 속도와 저장 FPS 불일치
- **확인:** `print(out.isOpened())`, 파일 크기와 플레이어 재생 확인
- **해결:** 코덱을 점검하고 실제 처리 속도에 맞는 저장 FPS 설정. 녹화 종료 후 `out.release()` 호출

## 9. 확인 문제 3개 (정답 대신 힌트)

1. **의미 분할과 인스턴스 분할은 무엇이 다를까요?**  
   *힌트:* 같은 사진 안에 자동차가 3대 있다면, 각각을 따로 번호 붙일 수 있는지 생각해 보세요.
2. **`cv.fillPoly(mask, [polygon], 255)`는 무엇을 하나요?**  
   *힌트:* 검은 종이에 특정 모양만 하얗게 칠하는 장면을 떠올려 보세요.
3. **`9-9_1.py`에서 `QTimer`를 사용하는 이유는 무엇일까요?**  
   *힌트:* 화면 갱신 중에도 버튼과 콤보박스 이벤트를 처리해야 합니다. 다만 추론 자체는 메인 스레드에서 실행됩니다.

## 10. 미니 과제

**과제명: 내 주변 물체 강조하기**

`9-7_1.py`에서 컬러로 남기는 객체를 `person`과 `bicycle`에서 `person`과 `car`로 바꾸어 보세요.

수정할 곳:

```python
if class_name in ['person', 'car']:
    cv.fillPoly(mask_focus, [polygon], 255)
```

**제출물:** 수정 코드, 실행 화면 캡처, 이전 결과와 달라진 점을 설명한 3문장.

## 11. 오늘의 핵심 요약

- **SegFormer** 수정본은 픽셀별로 종류를 나누는 **의미 분할**을 실습합니다.
- **YOLOv8-seg** 수정본은 객체마다 개별 마스크를 얻는 **인스턴스 분할**을 실습합니다.
- 마스크를 활용하면 색상 강조·흐림·배경 교체 같은 실용적인 효과를 만들 수 있습니다.
- 정지 사진 → 웹캠 → 객체별 효과 → PyQt5 GUI 순서로 실습하면 난이도를 점진적으로 높일 수 있습니다.
- 본 README는 **첨부된 수정 소스의 구조 분석**을 바탕으로 작성했으며, 5개 프로그램의 실제 실행 성공을 보증하는 문서는 아닙니다.

---

## 12. 개별 설명서 바로가기

- [9-5_1_설명서.md](수정본%209-5_1_설명서.md) — SegFormer 정지 이미지 의미 분할
- [9-6_1_설명서.md](수정본%209-6_1_설명서.md) — SegFormer 웹캠 의미 분할·MP4 저장
- [9-7_1_설명서.md](./9-7_1_설명서.md) — YOLOv8 5가지 효과와 자동차 크롭
- [9-8_1_설명서.md](수정본%209-8_1_설명서.md) — YOLOv8 사람·책 인스턴스 분할
- [9-9_1_설명서.md](수정본%209-9_1_설명서.md) — PyQt5+YOLOv8 배경 변경 GUI
