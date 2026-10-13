# 01. MOT17 연속 프레임 재생

> 대상 소스: `01_show_frames.py`  
> 대상: 컴퓨터 비전·객체 추적 입문 학습자  
> 데이터: MOT17 (실제 소스 기준)

## 1. 프로그램 개요
MOT17 이미지 시퀀스(`img1/*.jpg`)를 순서대로 읽어 동영상처럼 재생하는 가장 기초적인 예제입니다. 객체 검출이나 추적은 수행하지 않습니다.

## 2. 학습목표
- MOT17 동영상이 JPG 프레임의 연속으로 저장되는 방식을 설명할 수 있습니다.
- `pathlib`로 경로를 만들고 `OpenCV`로 프레임을 재생할 수 있습니다.

## 3. 준비 사항
- Python, `opencv-python`
- **MOT17/train/MOT17-13-SDP/img1**에 JPG 이미지가 있어야 합니다.
- 이 파일은 `mot17_common.py`를 사용하지 않습니다.

## 4. 처리 순서
1. 소스 위치에서 두 단계 상위 폴더를 `BASE_DIR`로 설정
2. `MOT17-13-SDP/img1`에서 JPG 파일을 이름순 정렬
3. 이미지 장수를 출력
4. 이미지를 한 장씩 읽어 화면에 표시
5. ESC 입력 또는 마지막 프레임에서 종료

## 5. 핵심 코드와 설명
```python
images = sorted(IMG_DIR.glob("*.jpg"))
img = cv.imread(str(image_path))
cv.imshow("MOT17 Video", img)
key = cv.waitKey(71)
```
- `glob('*.jpg')`: JPG 프레임 목록 수집
- `sorted(...)`: `000001.jpg`처럼 0을 채운 이름 기준 순서 유지
- `cv.waitKey(71)`: 각 프레임에서 약 71ms 대기. 대략 14fps 수준이지만 처리 시간 때문에 정확한 재생 속도 보장은 아닙니다.
- `cv.destroyAllWindows()`: OpenCV 창 닫기

## 6. 실행 방법 및 화면 확인
Windows PowerShell에서 프로젝트의 가상환경을 활성화한 후 프로젝트 루트에서 실행합니다. 예: `python source/mot17/01_show_frames.py`. 실제 파일 위치가 다르면 경로를 수정합니다. **ESC** 키로 종료합니다. 데이터는 `datasets/MOT17` 아래에 배치하고, `BASE_DIR = Path(__file__).resolve().parents[2]`가 프로젝트 루트를 가리키는지 확인합니다. 영상이 재생되고 콘솔에 전체 프레임 수가 표시됩니다.

## 7. 주의할 점과 한계
- 이미지가 없으면 `FileNotFoundError`가 발생합니다.
- 파일명과 폴더명이 정확해야 합니다.
- 소스에 정해진 71ms는 `seqinfo.ini`의 실제 프레임 레이트를 읽지 않습니다.

## 8. 실습 과제
1. `cv.waitKey(71)`을 `33`, `100`으로 바꾸고 재생 속도를 비교하세요.
2. 영상에 현재 프레임 번호를 표시해 보세요.
3. `MOT17-13-SDP`를 다른 train 시퀀스로 바꿔 보세요.

## 9. 한 줄 요약
정지 이미지 여러 장을 동영상처럼 보여 주는 기초 실습입니다.
