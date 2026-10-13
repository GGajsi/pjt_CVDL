# 04. 사람 ID 추적과 이동 궤적

> 대상 소스: `04_tracking_trajectory.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
공개 검출 박스를 단순 IoU 기반 추적기에 연결한 후 사람별 최근 50개 위치를 선으로 이은 이동 궤적을 표시합니다.

## 2. 학습목표
- 프레임 사이 동일 객체에 ID를 붙이는 목적을 설명할 수 있습니다.
- deque로 이동 이력을 저장하고 궤적을 시각화할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `pandas`, `numpy`, `scipy`
- `mot17_common.py` 및 **MOT17-11-SDP train** 이미지·검출 데이터·설정 파일 필요

## 4. 처리 순서
1. 검출 파일 로드(`conf=0.5`)
2. 각 프레임 검출을 `get_boxes`로 변환
3. `Tracker.update()`로 ID 할당
4. `bottom_center()`로 사람 박스의 하단 중앙점 계산
5. ID별 최근 50개 위치 저장 및 선 연결
6. 현재 추적 대상 수 표시

## 5. 핵심 코드와 설명
```python
tracker = Tracker()
histories = defaultdict(lambda: deque(maxlen=50))
tracks = tracker.update(boxes)
center = bottom_center(box)
histories[tid].append(center)
cv.line(img, points[i - 1], points[i], color, 3)
```
- `deque(maxlen=50)`: 오래된 위치부터 자동 삭제
- `tracks`: `{추적 ID: 박스}` 형태
- `bottom_center`: 사람의 발 주변 위치를 대표값으로 사용
- `color_for_id`: ID별 일관된 색상 생성

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/04_tracking_trajectory.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 사람 주위에 ID 박스, 하단 중앙의 빨간 점, 이동 경로가 표시됩니다.

## 7. 주의할 점과 한계
- 이 `Tracker`는 **IoU + 헝가리안 매칭** 기반의 교육용 추적기로 ByteTrack이 아닙니다.
- 가림, 급격한 이동, 카메라 이동 시 ID가 바뀔 수 있습니다.
- ID별 이력은 삭제하지 않으므로 긴 영상에서는 메모리 사용량이 증가할 수 있습니다.

## 8. 실습 과제
1. `maxlen=20`과 `100`을 비교하세요.
2. 발 위치 대신 박스 중심 `center_of`를 사용해 보세요.
3. 가림 현상에서 ID가 바뀌는 장면을 찾으세요.

## 9. 한 줄 요약
검출 위치를 시간에 따라 연결해 사람의 이동 경로를 그립니다.
