# 07. Optical Flow 움직임 벡터

> 대상 소스: `07_optical_flow.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
연속 프레임 사이 픽셀의 겉보기 움직임을 Farnebäck 밀집 옵티컬 플로우로 계산하여 화살표로 표시합니다. 사람 검출·식별 없이 전체 영상의 움직임을 분석합니다.

## 2. 학습목표
- Optical Flow의 의미를 설명할 수 있습니다.
- 두 연속 프레임에서 이동 벡터를 계산하고 시각화할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`, `numpy`
- 공통 모듈 `get_images`, `get_fps` 및 **MOT17-11-SDP train** 이미지·seqinfo.ini 필요
- `det.txt`는 사용하지 않습니다.

## 4. 처리 순서
1. 이미지 크기를 원본의 절반으로 축소
2. 회색조 이미지 생성
3. 처음 프레임을 이전 프레임으로 저장
4. 다음 프레임과 Farnebäck Optical Flow 계산
5. 20픽셀 간격으로 벡터 추출
6. 이동 크기 1.0 이상인 위치에 화살표 그리기

## 5. 핵심 코드와 설명
```python
flow = cv.calcOpticalFlowFarneback(previous_gray, gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
dx, dy = flow[y, x]
magnitude = np.hypot(dx, dy)
if magnitude >= 1.0:
    cv.arrowedLine(result, (x, y), (int(x + dx*3), int(y + dy*3)), (0,255,0), 1)
```
- `(dx,dy)`: 해당 픽셀의 두 프레임 사이 겉보기 이동량
- `magnitude`: 벡터 크기
- `*3`: 시각화를 위해 화살표를 세 배 늘림(실제 이동 거리 아님)

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/07_optical_flow.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 녹색 화살표들이 영상의 움직임 방향과 크기를 나타냅니다.

## 7. 주의할 점과 한계
- 카메라 자체 움직임도 화살표에 포함됩니다.
- Optical Flow는 사람 ID를 추적하거나 사람만 구분하지 않습니다.
- 매우 빠른 움직임이나 큰 가림은 부정확할 수 있습니다.

## 8. 실습 과제
1. `step=10`과 `40`을 비교하세요.
2. `magnitude < 1.0`의 기준을 바꿔 보세요.
3. 프레임당 평균 이동 크기를 화면에 표시하세요.

## 9. 한 줄 요약
영상 전체에서 픽셀의 움직임을 화살표로 표시합니다.
