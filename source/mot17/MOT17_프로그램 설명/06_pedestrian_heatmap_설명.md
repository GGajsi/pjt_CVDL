# 06. 보행자 위치 누적 히트맵

> 대상 소스: `06_pedestrian_heatmap.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
각 프레임에서 검출된 보행자의 바운딩 박스 하단 중앙점을 2차원 배열에 누적하고, 가우시안 블러와 컬러맵으로 활동 밀집도를 시각화합니다.

## 2. 학습목표
- 누적 빈도 히트맵의 의미를 설명할 수 있습니다.
- `GaussianBlur`, `normalize`, `applyColorMap`을 활용할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `numpy`, `pandas`
- 공통 모듈 및 **MOT17-11-SDP train** 데이터 필요

## 4. 처리 순서
1. 첫 프레임 크기로 0으로 채워진 배열 생성
2. 검출된 각 객체의 하단 중앙 좌표 확보
3. 해당 화소에 1씩 누적
4. Gaussian 블러로 주변에 밀집도 확산
5. 0~255 정규화 후 JET 컬러맵 적용
6. 원본과 55:45 비율로 합성

## 5. 핵심 코드와 설명
```python
heat = np.zeros((h, w), dtype=np.float32)
heat[y, x] += 1
blurred = cv.GaussianBlur(heat, (0, 0), sigmaX=25)
normalized = cv.normalize(blurred, None, 0, 255, cv.NORM_MINMAX)
colored = cv.applyColorMap(normalized.astype(np.uint8), cv.COLORMAP_JET)
result = cv.addWeighted(img, 0.55, colored, 0.45, 0)
```
- `heat[y,x]`: 화면상의 위치가 관측된 횟수
- `sigmaX=25`: 퍼짐 정도
- 최대값 기준으로 **매 프레임 재정규화**하므로 서로 다른 시점의 색상 강도를 절대값으로 비교하면 안 됩니다.

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/06_pedestrian_heatmap.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 사람이 자주 검출된 장소에 누적 색상이 나타납니다.

## 7. 주의할 점과 한계
- **순간 인구 밀도**가 아니라 **누적 검출 위치의 열지도**입니다. 같은 사람이 오래 머물면 값이 증가합니다.
- 추적 ID를 사용하지 않고 매 프레임의 검출 결과를 누적합니다.
- 카메라가 움직이는 시퀀스에서는 픽셀 좌표 기준 히트맵 해석이 어려워질 수 있습니다.

## 8. 실습 과제
1. `sigmaX=10`, `50`을 비교하세요.
2. 누적값 최대 위치의 좌표를 출력하세요.
3. 최종 히트맵 이미지를 파일로 저장하세요.

## 9. 한 줄 요약
어느 화면 위치에서 검출이 자주 일어났는지 누적해서 보여 줍니다.
