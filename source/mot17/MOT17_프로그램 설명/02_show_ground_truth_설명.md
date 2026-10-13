# 02. 정답(Ground Truth) 박스와 ID 표시

> 대상 소스: `02_show_ground_truth.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
MOT17의 `gt/gt.txt`에 기록된 사람 위치와 정답 ID를 각 이미지 위에 그립니다. 학습 모델을 실행하는 것이 아니라 **사람이 제공한 정답 주석을 시각화**하는 프로그램입니다.

## 2. 학습목표
- Ground Truth와 예측 검출을 구분할 수 있습니다.
- 바운딩 박스 `(x,y,w,h)` 및 정답 ID의 의미를 설명할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `pandas`
- **MOT17/train/MOT17-13-SDP/img1**, **gt/gt.txt** 필요
- **test에는 보통 공개 GT가 없으므로 train 데이터로 실행**하세요.

## 4. 처리 순서
1. 이미지·GT 경로 생성
2. `gt.txt`를 9개 열(`frame,id,x,y,w,h,mark,class,visibility`)로 로드
3. `mark==1` 및 `class==1`(보행자)만 선택
4. 파일 이름에서 프레임 번호 추출
5. 같은 프레임의 정답 상자와 사람 ID, 프레임 번호 표시

## 5. 핵심 코드와 설명
```python
gt = pd.read_csv(GT_FILE, header=None, names=columns)
gt = gt[(gt["mark"] == 1) & (gt["class"] == 1)]
frame_no = int(image_path.stem)
frame_gt = gt[gt["frame"] == frame_no]
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```
- `mark`: 평가용으로 유효한 객체인지 판별
- `class==1`: MOT17에서 pedestrian 클래스
- `stem`: `000123.jpg` → `000123`, 정수로 바꿔 123번 프레임 조회
- 녹색 사각형과 `ID n`은 **GT에 있는 식별자**입니다.

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/02_show_ground_truth.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 녹색 박스, `ID`, `Frame` 문구가 나타납니다.

## 7. 주의할 점과 한계
- GT의 `visibility` 열은 읽지만 필터 조건에는 사용하지 않습니다.
- GT ID는 실제 모델이 예측한 결과가 아닙니다.
- 이미지나 GT 경로가 다르면 수정해야 합니다.

## 8. 실습 과제
1. `visibility >= 0.5` 필터를 추가해 보세요.
2. 프레임별 보행자 수를 화면에 출력해 보세요.
3. 정답 ID마다 서로 다른 색을 부여해 보세요.

## 9. 한 줄 요약
모델의 성능을 비교할 기준인 정답 데이터를 영상에 표시합니다.
