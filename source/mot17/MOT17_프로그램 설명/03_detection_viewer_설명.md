# 03. 공개 Detection 파일 시각화

> 대상 소스: `03_detection_viewer.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
MOT17의 `det/det.txt`에 포함된 검출 후보를 읽고 신뢰도 기준 이상인 박스를 표시합니다. YOLO를 직접 실행하거나 사람 ID를 추적하는 예제는 아닙니다.

## 2. 학습목표
- 객체 검출(Detection)과 추적(Tracking)을 구별할 수 있습니다.
- 검출 신뢰도 임계값에 따른 결과 변화를 확인할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `numpy`, `pandas`, `scipy`(공통 모듈 전체 기능을 사용할 때)
- `mot17_common.py`가 같은 import 경로에 있어야 합니다.
- **공통 설정 `MOT17/train/MOT17-11-SDP`의 `img1`, `det/det.txt`, `seqinfo.ini` 필요**

## 4. 처리 순서
1. `get_images()`와 `get_fps()` 호출
2. `load_detections(conf=0.5)`로 검출 신뢰도 필터링
3. 프레임별 검출 후보 조회
4. 박스 및 신뢰도 표시
5. 화면에 그 프레임의 검출 건수 표시

## 5. 핵심 코드와 설명
```python
groups = load_detections(conf=0.5)
rows = groups.get(frame_no)
cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.putText(img, f"{r.conf:.2f}", ...)
```
- `groups`: 프레임 번호별 검출 행을 묶은 딕셔너리
- `(x,y,w,h)`에서 `(x1,y1,x2,y2)`로 변환하여 박스 표시
- 검출 수는 화면에 표시된 박스 개수이며 **실제 사람 수를 보장하지 않습니다**.

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/03_detection_viewer.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 녹색 박스와 점수, 좌측 상단 검출 건수가 표시됩니다.

## 7. 주의할 점과 한계
- `det.txt`의 신뢰도는 해당 검출 파일이 제공한 값으로, 일반적으로 YOLO 확률과 직접 비교하면 안 됩니다.
- 소스 자체는 사람 클래스 분류를 수행하지 않습니다.
- 공통 모듈이 기본적으로 `MOT17-11-SDP`를 바라봅니다.

## 8. 실습 과제
1. 임계값을 `0.3`, `0.7`로 바꾸고 검출 건수를 비교하세요.
2. 1개 프레임을 이미지 파일로 저장해 보세요.
3. GT 시각화 결과와 차이점을 기록하세요.

## 9. 한 줄 요약
이미 만들어진 검출 결과를 확인하는 프로그램입니다.
