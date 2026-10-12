# pjt_CVDL — 컴퓨터 비전과 딥러닝 실습

『컴퓨터 비전과 딥러닝』 교재의 OpenCV 및 객체 추적 실습을 위한 Python 프로젝트입니다. 특히 5장 MOT17 다중 객체 추적(Multi-Object Tracking) 데이터셋의 배치와 사용 방법을 정리했습니다.

## 1. 개발 환경

- IDE: IntelliJ IDEA
- 운영체제: Windows (PowerShell)
- Python: 3.11 계열 (가상환경 `.venv` 사용; 설치 로그의 `cp311` 기준)
- 주요 패키지: OpenCV, NumPy, pandas 등 (정확한 버전은 `requirements.txt` 참조)
- 데이터셋: [MOTChallenge MOT17](https://motchallenge.net/data/MOT17)

> 프로젝트 예시 위치: `D:\LSK\pjt_CVDL` (다른 PC에서는 경로가 달라도 됩니다.)

## 2. 프로젝트 폴더 구조

```text
pjt_CVDL/
├── .gitignore
├── .venv/                  # 로컬 Python 가상환경 (Git에서 제외 권장)
├── datasets/               # 다운로드한 대용량 데이터 (Git에서 제외)
│   └── MOT17/
│       ├── train/
│       │   └── MOT17-02-DPM/
│       │       ├── det/
│       │       │   └── det.txt
│       │       ├── gt/
│       │       │   └── gt.txt
│       │       ├── img1/
│       │       │   ├── 000001.jpg
│       │       │   └── ...
│       │       └── seqinfo.ini
│       └── test/
│           └── MOT17-01-FRCNN/
│               ├── det/
│               ├── img1/
│               └── seqinfo.ini
├── images/                 # 교재 실습용 이미지
├── source/
│   ├── ch02/
│   ├── ch03/
│   ├── ch04/
│   ├── ch05/
│   └── mot17/
│       ├── 01_show_frames.py
│       ├── 02_show_ground_truth.py
│       ├── 03_paly_video.py
│       └── 04_object_tracking.py
└── requirements.txt
```

위 구조는 대화에서 확인된 일부 폴더를 예시로 작성한 것으로, 전체 MOT17 시퀀스 목록을 나타내지는 않습니다. `03_paly_video.py` 파일명은 IntelliJ 화면에 표시된 이름을 그대로 기재했습니다.

## 3. MOT17 데이터셋 준비

1. 공식 사이트 [MOT17 데이터 페이지](https://motchallenge.net/data/MOT17)에서 데이터셋을 받습니다.
2. 압축을 풀어 `datasets/MOT17/train`, `datasets/MOT17/test` 구조로 배치합니다.
3. `train` 아래 시퀀스에 `img1/`, `det/`, `gt/`가 있는지 확인합니다.

| 항목 | 설명 |
| --- | --- |
| `img1/` | 연속 영상 프레임 (`000001.jpg` 등) |
| `det/det.txt` | 객체 검출 결과 |
| `gt/gt.txt` | 실제 객체 위치와 ID 등 정답(Ground Truth) |
| `seqinfo.ini` | 영상의 프레임 수, 해상도, FPS 등의 메타데이터 |

**`train`과 `test`의 차이:** 공식 MOT17 테스트 세트에는 평가용 정답 `gt.txt`가 일반적으로 공개되지 않습니다. 따라서 `test`에 `gt/` 폴더가 없더라도 이상이 아닙니다.

### MOT17Labels.zip을 따로 넣어야 하나요?

현재 화면에서 `datasets/MOT17/train/MOT17-02-DPM/gt/gt.txt`가 확인되었습니다. 즉 이 시퀀스에는 정답이 이미 들어 있으므로 **`MOT17Labels.zip`을 별도로 합칠 필요가 없습니다.** 다른 시퀀스가 누락되었다면 압축파일 내부 구조를 먼저 확인한 다음 해당 시퀀스에만 병합하세요. 기존 이미지 파일을 덮어쓰지 않도록 주의합니다.

PowerShell에서 정답 파일 존재 여부 확인:

```powershell
Test-Path .\datasets\MOT17\train\MOT17-02-DPM\gt\gt.txt
```

`True`이면 파일이 있습니다. 학습 세트 안의 모든 정답 파일을 열거하려면:

```powershell
Get-ChildItem .\datasets\MOT17\train -Recurse -Filter gt.txt |
    Select-Object FullName
```

## 4. `.gitignore`로 대용량 데이터 제외

프로젝트 **최상위 폴더**의 `.gitignore`에 아래 규칙을 추가합니다.

```gitignore
# 대용량 데이터셋 (GitHub 업로드 제외)
/datasets/

# Python 가상환경 및 캐시
/.venv/
__pycache__/
*.pyc

# IntelliJ 로컬 설정
/.idea/
```

`/datasets/`는 프로젝트 루트의 `datasets` 폴더 전체를 제외합니다. 소스 코드와 `requirements.txt`는 Git으로 관리합니다.

해당 파일이 Git에서 무시되는지 확인:

```powershell
git check-ignore -v datasets/MOT17/train/MOT17-02-DPM/img1/000001.jpg
```

**이미 Git이 데이터셋을 추적 중인 경우에만** 다음을 사용합니다.

```powershell
git rm -r --cached datasets/
git add .gitignore
git commit -m "Exclude datasets from Git"
git push
```

`--cached`는 로컬 파일을 지우지 않지만, 커밋을 push하면 GitHub 저장소의 현재 상태에서는 추적 파일이 제거됩니다. 과거 커밋의 대용량 데이터까지 자동 삭제되는 것은 아닙니다.

## 5. 라이브러리 설치와 `requirements.txt` 관리

IntelliJ 터미널에서 해당 프로젝트의 `.venv`가 활성화되어 있는지 확인한 뒤 설치합니다.

```powershell
pip install pandas
pip freeze > requirements.txt  --> pandas 설치후 라이브러리를 requirements.txt update 함
```

대화에서 확인한 설치 로그:

- `pandas==3.0.6` 설치 성공
- 기존 `numpy==2.4.6` 사용
- `tzdata==2026.5` 추가 설치
- 이후 `pip freeze > requirements.txt` 실행

`pip freeze`는 **직접 설치한 패키지뿐 아니라 의존성 패키지도 모두 기록**합니다. 설치·업데이트·삭제 후 다시 실행하면 현재 가상환경의 목록으로 파일이 덮어써집니다.

설치 상태 검사:

```powershell
python -m pip check
python -m pip show pandas
```

다른 PC에서 프로젝트를 내려받으면 별도 가상환경을 만든 뒤 복원합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1 > pip install -r requirements.txt
```

> Python 버전, 운영체제, GPU/CUDA 유무에 따라 일부 패키지 설치 또는 실행에 차이가 날 수 있습니다. 
> 데이터셋 자체는 `requirements.txt`에 포함되지 않으므로 별도로 다운로드해야 합니다.



## 6. MOT17 실습 파일

| 파일 | 학습 내용 |
| --- | --- |
| `01_show_frames.py` | 프레임 이미지 표시 |
| `02_show_ground_truth.py` | 정답 바운딩 박스 및 객체 ID 확인 |
| `03_paly_video.py` | 연속 프레임 영상처럼 재생 |
| `04_object_tracking.py` | 객체 추적 알고리즘 실습 |

파일 이름과 기능은 대화에 나타난 실습 구성 기준입니다. 실제 실행 방식과 추가 의존성은 각 파일의 소스를 확인하세요.

### 실행 위치에 영향을 덜 받는 데이터 경로

`source/mot17/`에 놓인 Python 스크립트라면 다음처럼 `pathlib`를 사용할 수 있습니다.

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "datasets" / "MOT17"
SEQ_DIR = DATASET_DIR / "train" / "MOT17-02-DPM"
IMG_DIR = SEQ_DIR / "img1"
GT_FILE = SEQ_DIR / "gt" / "gt.txt"

print("이미지 폴더:", IMG_DIR)
print("정답 파일:", GT_FILE)
print("정답 파일 존재:", GT_FILE.exists())
```

`parents[2]`는 파일 위치가 `pjt_CVDL/source/mot17/스크립트.py`일 때 프로젝트 루트를 가리킵니다. 다른 위치에 소스를 두었다면 상위 폴더 단계 수를 조정해야 합니다.

## 7. 점검 체크리스트

- [ ] `datasets/MOT17/train`과 `test`가 있다.
- [ ] 학습 시퀀스의 `img1/`에 JPG 프레임이 있다.
- [ ] 학습 시퀀스의 `gt/gt.txt`가 있다.
- [ ] `.gitignore`에 `/datasets/`를 등록했다.
- [ ] `.venv` 환경에서 필요한 패키지를 설치했다.
- [ ] `requirements.txt`를 최신화했다.
- [ ] `01_show_frames.py`부터 순서대로 실습한다.

---

**참고:** [MOTChallenge MOT17 공식 데이터 페이지](https://motchallenge.net/data/MOT17)
