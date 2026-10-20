# 9-5.py PixelLib 이미지 분할 실습 환경 구축 및 실행 설명서

> **실습 결과:** IntelliJ IDEA에서 별도 Python `venv` 가상환경을 구성하고, TensorFlow 2.7.0 + PixelLib 0.7.1을 이용하여 **DeepLabV3+ ADE20K 이미지 분할을 CPU로 정상 실행**했습니다.
>
> 이 문서는 실제로 수행한 명령어, 발생한 오류, 해결 과정 및 최종 결과를 시간 순서로 정리한 재현용 기록입니다. **Anaconda와 GPU용 CUDA/cuDNN은 사용하지 않습니다.**

## 1. 실습 목표와 기존 환경

교재의 `9-5.py`는 PixelLib의 `semantic_segmentation()`과 사전 학습된 `deeplabv3_xception65_ade20k.h5`를 사용하여 거리 사진을 픽셀 단위로 분할하는 프로그램입니다. `9-4.py`처럼 새 모델을 30 Epoch 학습하는 프로그램이 **아닙니다**.

실습 당시 주요 환경:

| 구분 | 기존 프로젝트 환경 | 교재 전용 환경 |
|---|---|---|
| IDE | IntelliJ IDEA | IntelliJ IDEA |
| 경로 | `K:\LSK\pjt_CVDL\.venv` | `K:\LSK\pjt_CVDL\.venv_cvbook` |
| Python | 3.11 계열 | 3.9 계열 |
| TensorFlow | 2.21.0 | **2.7.0** |
| Keras | 3.15.1 | **2.7.0** |
| NumPy | 2.4.6 | **1.22.1** |
| OpenCV | 5.0.0.93 | **4.5.5.62** |
| PixelLib | 미설치 | **0.7.1** |
| GPU | Windows의 최신 TF 환경에서 미인식 | **GPU 미설치·CPU 실행** |

교재에 제시된 환경은 Python 3.9.7, TensorFlow 2.7.0, NumPy 1.22.1, OpenCV 4.5.5.62, PyQt 5.9.2, Spyder 5.1.5입니다. 이번 실습에서는 **Spyder·Anaconda 대신 IntelliJ IDEA + 표준 `venv`**를 사용했습니다. 교재 스크린샷에는 PixelLib 버전이 나타나지 않았으므로 PixelLib 0.7.1은 이번 실습에서 별도로 선택하고 검증한 버전입니다.

## 2. 기존 환경을 변경하지 않은 이유

처음에는 기존 `.venv`에서 `9-5.py`의 다음 import를 인식하지 못했습니다.

```python
from pixellib.semantic import semantic_segmentation
```

확인 명령:

```powershell
pip show pixellib
```

결과:

```text
WARNING: Package(s) not found: pixellib
```

기존 `.venv`의 TensorFlow 2.21.0 / Keras 3 계열과 구형 PixelLib 사이에 API·의존성 호환성 위험이 있으므로, 기존 환경에 설치하지 않고 **`.venv_cvbook`**를 생성했습니다.

## 3. 별도 가상환경 생성

### 3.1 설치된 Python 확인

IntelliJ IDEA 하단 **Terminal**에서:

```powershell
py -0p
```

당시 Python 3.14, 독립 설치 Python 3.11, Anaconda의 Python 3.9가 조회되었습니다. 사용자는 Anaconda를 쓰지 않기로 결정했으므로 **별도 설치한 Python 3.9**를 사용합니다.

> 주의: 아래 `py -3.9` 명령을 사용하기 전에 `py -3.9 --version`으로 **Anaconda가 아닌 독립 Python 3.9**가 실행되는지 확인합니다. Python 3.9는 지원이 종료된 구버전이므로 교재 재현용 격리 환경으로만 사용합니다.

```powershell
py -3.9 --version
```

### 3.2 가상환경 생성 및 활성화

```powershell
cd K:\LSK\pjt_CVDL
py -3.9 -m venv .venv_cvbook
.\.venv_cvbook\Scripts\Activate.ps1
```

정상적으로 활성화되면 프롬프트가 다음처럼 표시됩니다.

```text
(.venv_cvbook) PS K:\LSK\pjt_CVDL>
```

프로젝트 구조:

```text
K:\LSK\pjt_CVDL\
├── .venv\                  # 기존 TensorFlow 2.21 환경: 수정하지 않음
├── .venv_cvbook\           # 교재 전용 Python 3.9 가상환경
├── datasets\
└── source\
    └── ch09\
        ├── 9-5.py
        ├── busy_street.jpg
        └── deeplabv3_xception65_ade20k.h5
```

가상환경 활성화가 안 되면 `K:\LSK\pjt_CVDL\.venv_cvbook\Scripts\python.exe`를 명령에 직접 지정할 수도 있습니다.

## 4. 교재 핵심 패키지 설치

실제로 구성한 주요 버전:

```powershell
python -m pip install "numpy==1.22.1"
python -m pip install "tensorflow==2.7.0"
python -m pip install "opencv-python==4.5.5.62"
```

설치된 버전 확인:

```powershell
python -m pip show tensorflow keras numpy opencv-python
```

실제 확인 결과는 TensorFlow **2.7.0**, Keras **2.7.0**, NumPy **1.22.1**, OpenCV **4.5.5.62**였습니다.

> 설치된 패키지끼리 서로 버전을 다시 변경할 수 있으므로 설치 직후 `pip show`와 `pip check`로 검증합니다. 오래된 환경에 최신 패키지들을 무조건 일괄 설치하지 않습니다.

## 5. TensorFlow import 오류와 해결 과정

### 5.1 첫 번째 오류: Protobuf 버전 충돌

최초 `import tensorflow` 시 다음 오류가 발생했습니다.

```text
TypeError: Descriptors cannot be created directly.
```

당시 확인한 상태:

- `tensorflow==2.7.0`
- `protobuf==6.33.6` (**구버전 TensorFlow와 호환되지 않음**)

해결:

```powershell
python -m pip install --force-reinstall "protobuf==3.19.6"
```

Protobuf 3.19.6으로 변경한 뒤에는 기존 TensorBoard 2.21.0이 새 Protobuf와 충돌했습니다.

### 5.2 두 번째 오류: TensorBoard 버전 충돌

실행했던 명령:

```powershell
python -m pip install --force-reinstall --no-deps "tensorboard==2.7.0"
```

이때 `--no-deps`는 TensorBoard 변경 시 다른 핵심 패키지의 자동 변경을 피하기 위해 사용했습니다. 그 결과 추가 의존성이 아직 설치되지 않은 상태가 되어 `pip check`에서 다음 항목들이 나타났습니다.

- `google-auth` 미설치
- `google-auth-oauthlib` 미설치
- `requests` 미설치
- `tensorboard-plugin-wit` 미설치
- `tensorboard-data-server==0.7.2`는 TensorBoard 2.7.0의 요구 버전과 불일치

해결 명령:

```powershell
python -m pip install --force-reinstall "tensorboard-data-server==0.6.1"
python -m pip install "google-auth>=1.6.3,<3" "google-auth-oauthlib>=0.4.1,<0.5" "requests>=2.21,<3" "tensorboard-plugin-wit>=1.6.0"
python -m pip check
```

최종 출력:

```text
No broken requirements found.
```

이 메시지는 **패키지 메타데이터 기준 의존성 충돌이 없다는 뜻**입니다. 개별 기능의 정상 실행 여부는 별도 시험이 필요합니다.

### 5.3 TensorFlow 실행 확인

```powershell
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

실제 핵심 결과:

```text
TensorFlow: 2.7.0
GPU: []
```

`cudart64_110.dll`, `cublas64_11.dll`, `cudnn64_8.dll` 등을 찾을 수 없다는 **CUDA 경고**도 함께 나왔습니다. 이번 실습은 GPU를 설정하지 않고 CPU로 실행하기로 했으므로 별도 CUDA/cuDNN을 설치하지 않았습니다. TensorFlow import와 CPU 추론이 정상 수행되는 한 이 경고는 실습 실패가 아닙니다.

## 6. PixelLib 설치와 추가 오류 해결

### 6.1 설치 가능한 버전 확인

```powershell
python -m pip index versions pixellib
```

목록에서 `0.7.1`이 확인되었습니다.

### 6.2 기존 환경 스냅샷 저장

PixelLib 설치 전에:

```powershell
cd K:\LSK\pjt_CVDL
python -m pip freeze > requirements_cvbook_before_pixellib.txt
```

이 파일은 변경 이전 패키지 버전 **목록**입니다. 가상환경 전체 백업이나 모델·데이터 파일 백업은 아닙니다.

### 6.3 PixelLib 설치

기존 TensorFlow/Keras/NumPy의 자동 교체를 막기 위해:

```powershell
python -m pip install --no-deps "pixellib==0.7.1"
python -m pip show pixellib
```

실제 출력에서 `Successfully installed pixellib-0.7.1` 및 `Version: 0.7.1`을 확인했습니다.

`--no-deps`를 사용했으므로 `pip check`에는 `black`, `cloudpickle`, `cython`, `future`, `fvcore`, `hydra-core`, `imantics`, `imgaug`, `iopath`, `matplotlib`, `omegaconf`, `pyqt5`, `scikit-image` 등을 비롯한 **미설치 의존성**이 표시되었습니다. 이는 PixelLib 전체 의존성이 설치되었다는 의미가 **아닙니다**.

### 6.4 PixelLib import 오류: imantics 없음

첫 import 테스트:

```powershell
python -c "from pixellib.semantic import semantic_segmentation; print('PixelLib import 성공')"
```

오류:

```text
ModuleNotFoundError: No module named 'imantics'
```

해결:

```powershell
python -m pip install "imantics==0.1.12"
```

실제로 `imantics-0.1.12`, `lxml-6.1.3`, `xmljson-0.2.1`이 설치되었습니다. PixelLib의 다른 미설치 의존성 경고는 일부 남아 있었지만, 필요한 기능을 먼저 검사하기로 했습니다.

### 6.5 import 재시험 성공

```powershell
python -c "from pixellib.semantic import semantic_segmentation; print('PixelLib import 성공')"
```

최종 출력:

```text
PixelLib import 성공
```

CUDA 관련 경고 뒤에 이 문구가 출력되었으므로 **semantic 모듈 import까지 성공**한 것입니다.

> **재현 시 유의:** 이번 기록은 `--no-deps` 설치 후 실제로 필요한 기능을 단계별 테스트한 과정입니다. PixelLib의 모든 부가기능이나 `pip check` 전체 통과까지 증명한 것은 아닙니다. `9-5.py`는 실제 정상 실행까지 확인했습니다.

## 7. 9-5.py 코드와 준비 파일

프로젝트의 `source/ch09/` 폴더에 다음 파일이 있어야 합니다.

```text
source\ch09\
├── 9-5.py
├── busy_street.jpg
└── deeplabv3_xception65_ade20k.h5
```

실습 코드:

```python
from pixellib.semantic import semantic_segmentation
import cv2 as cv

seg = semantic_segmentation()
seg.load_ade20k_model('deeplabv3_xception65_ade20k.h5')

img_fname = 'busy_street.jpg'
seg.segmentAsAde20k(img_fname, output_image_name='image_new.jpg')
info1, img_segmented1 = seg.segmentAsAde20k(img_fname)
info2, img_segmented2 = seg.segmentAsAde20k(img_fname, overlay=True)

cv.imshow('Image original', cv.imread(img_fname))
cv.imshow('Image segmentation', img_segmented1)
cv.imshow('Image segmentation overlayed', img_segmented2)

cv.waitKey()
cv.destroyAllWindows()
```

핵심 기능:

| 코드 | 의미 |
|---|---|
| `semantic_segmentation()` | PixelLib 이미지 분할 객체 생성 |
| `load_ade20k_model(...)` | ADE20K 사전 학습 모델 로딩 |
| `segmentAsAde20k(...)` | 픽셀별 장면 클래스 분할 수행 |
| `output_image_name='image_new.jpg'` | 분할 결과 파일 저장 |
| `overlay=True` | 원본 사진과 분할 영역 겹치기 |
| `cv.imshow(...)` | 원본·분할·오버레이 창 표시 |
| `cv.waitKey()` | 키 입력까지 창 유지 |

분할 모델은 사전 학습된 가중치를 활용하므로 **`fit()`이나 Epoch 학습이 없습니다**.

## 8. IntelliJ IDEA에서 실행하기

### 방법 A: Terminal에서 실행 (이번 실습에서 성공한 방법)

```powershell
cd K:\LSK\pjt_CVDL\source\ch09
python .\9-5.py
```

실제 프롬프트:

```text
(.venv_cvbook) PS K:\LSK\pjt_CVDL\source\ch09>
```

필요하면 인터프리터를 완전한 경로로 지정합니다.

```powershell
& "K:\LSK\pjt_CVDL\.venv_cvbook\Scripts\python.exe" .\9-5.py
```

### 방법 B: IntelliJ의 Run 버튼 사용

1. `Run → Edit Configurations...`에서 `9-5.py`용 실행 설정을 생성하거나 선택합니다.
2. Python Interpreter를 `K:\LSK\pjt_CVDL\.venv_cvbook\Scripts\python.exe`로 선택합니다.
3. **Working directory**를 `K:\LSK\pjt_CVDL\source\ch09`로 설정합니다.
4. 초록색 Run 버튼을 누릅니다.

IDE 버전에 따라 메뉴 표기가 다를 수 있습니다. **실행 인터프리터와 작업 폴더**가 중요합니다.

## 9. 최종 실행 결과

실제 터미널 명령:

```powershell
(.venv_cvbook) PS K:\LSK\pjt_CVDL\source\ch09> python .\9-5.py
```

출력 끝부분:

```text
Skipping registering GPU devices...
Processed Image saved successfuly in your current working directory.
```

실행 화면에서 **세 개의 OpenCV 창**을 확인했습니다.

1. **Image original** — 거리 원본 사진
2. **Image segmentation** — 모델이 분류한 픽셀별 영역
3. **Image segmentation overlayed** — 원본 사진 위에 분할 색상을 겹친 결과

차량 영역 등이 분할 색상으로 표시되었으며, 실제 이미지 처리와 창 출력이 동작했습니다. 결과 파일은 실행 작업 폴더의 `image_new.jpg`입니다.

파일 확인:

```powershell
Get-Item K:\LSK\pjt_CVDL\source\ch09\image_new.jpg
```

창을 닫을 때는 OpenCV 이미지 창이 활성화된 상태에서 아무 키나 누릅니다.

## 10. 오류별 빠른 점검표

| 증상 | 원인 또는 점검 | 조치 |
|---|---|---|
| `Unresolved reference pixellib` | IntelliJ가 다른 인터프리터를 사용하거나 PixelLib 미설치 | `.venv_cvbook` 인터프리터 지정, `pip show pixellib` |
| `Descriptors cannot be created directly` | 구형 TF와 최신 Protobuf 충돌 | `protobuf==3.19.6` |
| TensorBoard 의존성 충돌 | TF 2.7에 TensorBoard 2.21 혼재 | `tensorboard==2.7.0`, 관련 의존성 조정 |
| `No module named imantics` | PixelLib 추가 의존성 없음 | `imantics==0.1.12` 설치 |
| `cudart64_110.dll not found` | GPU용 DLL 없음 | **CPU 사용 시 별도 설치하지 않음** |
| `.h5` 파일을 찾지 못함 | 모델 경로 또는 작업 폴더 불일치 | `source/ch09`에서 실행하고 파일 위치 확인 |
| `busy_street.jpg` 오류 | 이미지 경로 불일치 | 파일 존재·작업 폴더 확인 |
| IntelliJ Run에서만 실패 | 터미널과 Run 인터프리터·작업 디렉터리가 다름 | 실행 설정 확인 |

## 11. 환경 보존 및 다음 실습

실습 완료 후 프로젝트 루트에서:

```powershell
cd K:\LSK\pjt_CVDL
python -m pip freeze > requirements_cvbook.txt
```

검증 명령:

```powershell
python -m pip show tensorflow keras numpy opencv-python pixellib
python -m pip check
```

> 주의: PixelLib을 `--no-deps`로 설치했으므로 마지막 `pip check`는 미설치 부가기능 의존성을 계속 보고할 수 있습니다. 실제로 검증된 범위는 **`9-5.py`의 PixelLib ADE20K 분할 실행 성공**입니다. `pip freeze`를 저장했다고 해서 패키지 설치 파일 자체가 백업되는 것은 아닙니다.

### 최종 정리

- **기존 `.venv`**는 TensorFlow 2.21 실습을 위해 유지합니다.
- **새 `.venv_cvbook`**는 Python 3.9 / TensorFlow 2.7 / PixelLib 0.7.1 교재 실습에 사용합니다.
- **Anaconda 사용 안 함, GPU/CUDA/cuDNN 추가 설치 안 함.**
- 오류 해결 순서: **Protobuf → TensorBoard 의존성 → PixelLib → imantics → import 성공 → `9-5.py` 실행 성공**.
- 최종 확인: 원본·분할·오버레이 3개 이미지 창 출력 및 처리 이미지 파일 저장 메시지 확인.

