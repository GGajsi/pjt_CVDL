# 08. YOLO11 + ByteTrack 사람 추적

> 대상 소스: `08_yolo_bytetrack.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
사전 학습된 YOLO11n 모델로 사람을 검출하고, Ultralytics의 ByteTrack 알고리즘으로 프레임 간 ID를 유지하는 예제입니다. 03~06번과 달리 MOT17 `det.txt`를 사용하지 않고 원본 이미지에서 직접 검출합니다.

## 2. 학습목표
- YOLO 객체 검출과 ByteTrack 객체 추적의 역할을 구분할 수 있습니다.
- 모델 예측 박스와 추적 ID를 영상에 표시할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `ultralytics` 및 설치 과정에서 요구되는 관련 패키지(`lap` 등)
- `mot17_common.py`, **MOT17-11-SDP train** 이미지와 seqinfo.ini 필요
- 첫 실행 시 `yolo11n.pt` 모델 가중치 다운로드에 인터넷이 필요할 수 있습니다.

## 4. 처리 순서
1. `YOLO("yolo11n.pt")` 모델 로드
2. 매 프레임 이미지를 입력
3. `model.track(... persist=True, tracker="bytetrack.yaml", classes=[0])` 실행
4. `result.boxes.id`가 있을 때 박스와 ID 획득
5. 사람별 박스 및 현재 ID 개수 표시

## 5. 핵심 코드와 설명
```python
model = YOLO("yolo11n.pt")
results = model.track(img, persist=True, tracker="bytetrack.yaml",
                      classes=[0], conf=0.25, verbose=False)
result = results[0]
if result.boxes is not None and result.boxes.id is not None:
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()
```
- `classes=[0]`: COCO의 사람 클래스만 대상
- `persist=True`: 프레임 사이 추적 상태 유지
- `conf=0.25`: YOLO의 검출 신뢰도 임계값
- `boxes.id`: 검출과 추적이 연결되어 ID가 부여된 경우에 존재

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/08_yolo_bytetrack.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 화면에 `Person ID`와 `Tracked people`이 표시됩니다. 첫 실행에는 모델 다운로드 또는 종속성 자동 설치 메시지가 나타날 수 있습니다.

## 7. 주의할 점과 한계
- YOLO 추적 결과는 GT 정답이 아니며, 가림·중복 검출·작은 객체에 따라 달라집니다.
- `cv.waitKey(1)`만 사용하므로 영상은 처리 성능에 따라 재생 속도가 달라집니다.
- 모델이 다운로드되는 기본 위치는 실행 작업 디렉터리의 영향을 받습니다.
- **08번의 ByteTrack은 04~05번의 교육용 IoU Tracker와 다른 알고리즘**입니다.

## 8. 실습 과제
1. `conf=0.25`와 `0.5`를 비교하세요.
2. 04번 궤적 프로그램과 ID 안정성을 비교해 보세요.
3. 추적 ID별 통과 카운터를 확장해 보세요.

## 9. 한 줄 요약
AI가 사람을 직접 찾아서 ByteTrack으로 연속 추적합니다.
