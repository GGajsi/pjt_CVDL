
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import configparser
import time


# ==========================================
# 1. 기본 환경 설정
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
DET_FILE = SEQ_DIR / "det" / "det.txt"
INFO_FILE = SEQ_DIR / "seqinfo.ini"

# 추적 설정
CONF_THRESHOLD = 0.5    # 검출 신뢰도 기준
IOU_THRESHOLD = 0.3     # 같은 사람으로 판단하는 IoU 기준
MAX_MISSING = 15        # 추적 정보 유지 프레임 수

# ==========================================
# 2. IoU 계산 함수
# ==========================================
def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ==========================================
# 3. 다중 객체 추적 클래스
# ==========================================
class SimpleTracker:

    def __init__(self, iou_threshold=0.3, max_missing=15):

        self.tracks = {}
        self.next_id = 1

        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, detections):

        track_ids = list(self.tracks.keys())

        # 기존 객체가 없는 경우
        if len(track_ids) == 0:

            for box in detections:
                self.add_track(box)

            return self.get_visible_tracks()

        # IoU 비용 행렬 생성
        cost_matrix = np.ones(
            (len(track_ids), len(detections)),
            dtype=float
        )

        for i, track_id in enumerate(track_ids):
            track_box = self.tracks[track_id]["box"]

            for j, det_box in enumerate(detections):

                iou = calculate_iou(track_box, det_box)
                cost_matrix[i, j] = 1.0 - iou

        matched_tracks = set()
        matched_detections = set()

        # 헝가리안 알고리즘으로 최적 매칭
        if len(detections) > 0:

            rows, cols = linear_sum_assignment(cost_matrix)

            for row, col in zip(rows, cols):

                iou = 1.0 - cost_matrix[row, col]

                if iou < self.iou_threshold:
                    continue

                track_id = track_ids[row]

                self.tracks[track_id]["box"] = detections[col]
                self.tracks[track_id]["missing"] = 0

                matched_tracks.add(track_id)
                matched_detections.add(col)

        # 매칭 실패한 기존 객체 처리
        for track_id in track_ids:

            if track_id not in matched_tracks:

                self.tracks[track_id]["missing"] += 1

        # 새로운 객체 등록
        for j, box in enumerate(detections):

            if j not in matched_detections:
                self.add_track(box)

        # 오래 사라진 객체 제거
        for track_id in list(self.tracks.keys()):

            if self.tracks[track_id]["missing"] > self.max_missing:
                del self.tracks[track_id]

        return self.get_visible_tracks()

    def add_track(self, box):

        self.tracks[self.next_id] = {
            "box": box,
            "missing": 0
        }

        self.next_id += 1

    def get_visible_tracks(self):

        return {
            track_id: data["box"]
            for track_id, data in self.tracks.items()
            if data["missing"] == 0
        }


# ==========================================
# 4. 데이터 확인 및 불러오기
# ==========================================
if not IMG_DIR.is_dir():
    raise FileNotFoundError(f"이미지 폴더 없음: {IMG_DIR}")

if not DET_FILE.is_file():
    raise FileNotFoundError(f"검출 데이터 없음: {DET_FILE}")

config = configparser.ConfigParser()
fps = 30

if INFO_FILE.exists():
    config.read(INFO_FILE, encoding="utf-8-sig")
    fps = config.getint("Sequence", "frameRate", fallback=30)

delay = max(1, round(1000 / fps))

images = sorted(IMG_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError("이미지가 없습니다.")

# 검출 결과 열 이름
columns = [
    "frame", "id", "x", "y",
    "w", "h", "confidence",
    "unused1", "unused2", "unused3"
]

det = pd.read_csv(
    DET_FILE,
    header=None,
    names=columns
)

# 신뢰도 필터링
det = det[
    det["confidence"] >= CONF_THRESHOLD
    ]

# 프레임별 검출 결과를 미리 분류
det_groups = {
    frame: group
    for frame, group in det.groupby("frame")
}

tracker = SimpleTracker(
    iou_threshold=IOU_THRESHOLD,
    max_missing=MAX_MISSING
)

print("=" * 50)
print("MOT17 객체 추적 시작")
print("시퀀스:", SEQ_DIR.name)
print("전체 프레임:", len(images))
print("원본 FPS:", fps)
print("ESC: 종료 / SPACE: 일시정지")
print("=" * 50)


# ==========================================
# 5. 추적 결과를 화면에 표시
# ==========================================
def draw_tracks(frame, tracks):

    for track_id, box in tracks.items():

        x1, y1, x2, y2 = map(int, box)

        # ID에 따라 일정한 색상 부여
        color = (
            (track_id * 37) % 256,
            (track_id * 97) % 256,
            (track_id * 157) % 256
        )

        # 바운딩 박스
        cv.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # 객체 ID
        cv.putText(
            frame,
            f"ID {track_id}",
            (x1, max(20, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


# ==========================================
# 6. 영상 재생 및 객체 추적
# ==========================================
paused = False
previous_time = time.perf_counter()

for image_path in images:

    frame_no = int(image_path.stem)

    img = cv.imread(str(image_path))

    if img is None:
        continue

    # 현재 프레임의 검출 결과
    frame_det = det_groups.get(frame_no)

    detections = []

    if frame_det is not None:

        for row in frame_det.itertuples(index=False):

            x1 = float(row.x)
            y1 = float(row.y)

            x2 = x1 + float(row.w)
            y2 = y1 + float(row.h)

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append([x1, y1, x2, y2])

    # 추적 실행
    tracks = tracker.update(detections)

    # 화면에 추적 결과 표시
    draw_tracks(img, tracks)

    # FPS 계산
    current_time = time.perf_counter()
    elapsed = current_time - previous_time

    current_fps = 1.0 / elapsed if elapsed > 0 else 0
    previous_time = current_time

    # 상태 정보
    cv.putText(
        img,
        f"Frame: {frame_no}",
        (20, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv.putText(
        img,
        f"Tracking: {len(tracks)}",
        (20, 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv.putText(
        img,
        f"FPS: {current_fps:.1f}",
        (20, 105),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv.imshow("MOT17 Object Tracking", img)

    # 키보드 제어
    while True:

        key = cv.waitKey(
            0 if paused else delay
        ) & 0xFF

        if key == 27:
            cv.destroyAllWindows()
            raise SystemExit

        elif key == 32:
            paused = not paused
            previous_time = time.perf_counter()

        if not paused:
            break

cv.destroyAllWindows()

print("객체 추적 완료")
