# 공통 모듈: 경로·검출·IoU 추적 함수

> 대상 소스: `mot17_common.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
03~08번 프로그램의 공통 데이터 경로, FPS, 검출 파일 읽기, 박스 변환, IoU 기반 ID 매칭, 표시 함수를 모아 둔 보조 모듈입니다. 단독 실행용 메인 프로그램이 아닙니다.

## 2. 학습목표
- 여러 프로그램에서 공통 함수를 분리하는 이유를 설명할 수 있습니다.
- IoU 기반 간단 추적기의 동작과 한계를 이해할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `pandas`, `numpy`, `scipy` (추적기 실행 시)
- 기본 `SEQ_DIR`: **MOT17/train/MOT17-11-SDP**
- `img1`, `det/det.txt`, `seqinfo.ini`는 각각 사용하는 함수에 따라 필요합니다.

## 4. 처리 순서
1. `BASE_DIR`와 `SEQ_DIR` 정의
2. `get_images`: 프레임 JPG 목록 반환
3. `get_fps`: `seqinfo.ini`의 `frameRate`를 읽음(없으면 fallback 30)
4. `load_detections`: `det.txt` 읽고 신뢰도 필터 적용, 프레임별 그룹화
5. `get_boxes`: `(x,y,w,h)`를 `(x1,y1,x2,y2)`로 변환
6. `Tracker.update`: 이전 박스와 현재 박스의 IoU 행렬을 만들고 헝가리안 매칭을 수행, 매칭되지 않은 객체는 잠시 유지하거나 새 ID 부여
7. 시각화 함수로 표시

## 5. 핵심 코드와 설명
| 함수/클래스 | 역할 |
|---|---|
| `get_images()` | 이미지 파일 목록 수집 |
| `get_fps()` | 시퀀스 재생 fps 조회 |
| `load_detections(conf)` | 검출 후보 필터 및 프레임별 그룹 |
| `get_boxes(group)` | 박스 좌표 형식 변환 |
| `box_iou(a,b)` | 두 박스의 Intersection over Union 계산 |
| `Tracker(threshold=0.3, max_missing=15)` | IoU/최적 할당으로 ID 연결 |
| `color_for_id(tid)` | ID별 색상 계산 |
| `center_of(box)` | 박스 중앙 좌표 |
| `bottom_center(box)` | 박스 하단 중앙 좌표 |
| `draw_tracks(frame,tracks)` | 박스와 ID 그리기 |

```python
cost = np.array([[1 - box_iou(self.tracks[i]["box"], b) for b in boxes]
                 for i in ids])
rows, cols = linear_sum_assignment(cost)
```
- IoU는 겹친 넓이를 합집합 넓이로 나눈 값입니다(0~1).
- 비용 `1-IoU`가 작을수록 비슷한 위치의 박스입니다.
- 매칭 후 IoU가 `threshold` 미만이면 같은 사람으로 인정하지 않습니다.
- 일정 기간 놓친 추적은 `max_missing` 초과 시 제거합니다.

## 6. 실행 방법 및 화면 확인
이 파일은 직접 실행해서 창을 띄우는 형태가 아닙니다. 동일 폴더에 두고 `03_detection_viewer.py` 등의 `from mot17_common import ...`에서 사용합니다.

## 7. 주의할 점과 한계
- `get_fps()`의 fallback=30은 설정 파일 전체가 없을 때 항상 안전하게 동작한다는 뜻은 아닙니다. 정상적인 `seqinfo.ini` 구조를 준비하세요.
- Tracker에는 외형(Re-ID) 특징, 운동 예측, 가림 복구 등이 없고 ByteTrack이 아닙니다.
- `max_missing`은 초 단위가 아닌 **프레임 갱신 횟수**입니다.
- `SEQ_DIR`를 바꾸면 03~08번이 사용하는 시퀀스가 함께 바뀝니다.

## 8. 실습 과제
1. `SEQ_DIR`를 `MOT17-13-SDP`로 바꾸고 03번을 실행해 보세요.
2. `threshold=0.1`과 `0.6`에서 추적 ID 변화를 비교하세요.
3. `box_iou` 함수에 겹치는 박스와 겹치지 않는 박스를 입력해 결과를 확인하세요.

## 9. 한 줄 요약
여러 실습이 공유하는 데이터 로더와 교육용 객체 추적기를 제공합니다.
