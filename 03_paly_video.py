import cv2 as cv
from pathlib import Path
import configparser

# ==========================================
# 1. 데이터셋 경로 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

SEQ_DIR = (
        BASE_DIR
        / "datasets"
        / "MOT17"
        / "test"
        / "MOT17-14-SDP"
)

IMG_DIR = SEQ_DIR / "img1"
INFO_FILE = SEQ_DIR / "seqinfo.ini"

# ==========================================
# 2. 경로 확인
# ==========================================
if not IMG_DIR.exists():
    raise FileNotFoundError(f"이미지 폴더가 없습니다: {IMG_DIR}")

# ==========================================
# 3. 영상 정보 읽기
# ==========================================
config = configparser.ConfigParser()

fps = 30  # 기본 FPS

if INFO_FILE.exists():
    config.read(INFO_FILE, encoding="utf-8-sig")
    fps = config.getint(
        "Sequence",
        "frameRate",
        fallback=30
    )

delay = max(1, round(1000 / fps))

# ==========================================
# 4. 이미지 목록 가져오기
# ==========================================
images = sorted(IMG_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError(f"이미지가 없습니다: {IMG_DIR}")

print("=" * 50)
print("MOT17 영상 재생")
print("=" * 50)
print("시퀀스:", SEQ_DIR.name)
print("이미지 경로:", IMG_DIR)
print("전체 프레임:", len(images))
print("FPS:", fps)
print("ESC: 종료")
print("SPACE: 일시정지 / 재개")
print("=" * 50)

# ==========================================
# 5. 동영상 재생
# ==========================================
paused = False

for index, image_path in enumerate(images, start=1):

    # 이미지 읽기
    img = cv.imread(str(image_path))

    if img is None:
        print("이미지 읽기 실패:", image_path)
        continue

    # 화면 출력용 복사
    frame = img.copy()

    # 프레임 정보 표시
    cv.putText(
        frame,
        f"Frame: {index}/{len(images)}",
        (20, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # 시퀀스 이름 표시
    cv.putText(
        frame,
        "MOT17-14-SDP",
        (20, 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # 영상 출력
    cv.imshow("MOT17 Video Player", frame)

    # ======================================
    # 6. 키보드 입력 처리
    # ======================================
    while True:

        key = cv.waitKey(
            0 if paused else delay
        ) & 0xFF

        # ESC : 종료
        if key == 27:
            cv.destroyAllWindows()
            raise SystemExit

        # SPACE : 일시정지 / 재개
        elif key == 32:
            paused = not paused

        if not paused:
            break

# ==========================================
# 7. 종료
# ==========================================
cv.destroyAllWindows()

print("영상 재생 완료")