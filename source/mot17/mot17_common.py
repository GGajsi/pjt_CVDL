
from pathlib import Path
from collections import defaultdict
import configparser

import cv2 as cv
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
SEQ_DIR = BASE_DIR / "datasets" / "MOT17" / "train" / "MOT17-11-SDP"
IMG_DIR = SEQ_DIR / "img1"
DET_FILE = SEQ_DIR / "det" / "det.txt"

def get_images():
    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"이미지 없음: {IMG_DIR}")
    return images

def get_fps():
    info = configparser.ConfigParser()
    info.read(SEQ_DIR / "seqinfo.ini", encoding="utf-8-sig")
    return max(1, info.getint("Sequence", "frameRate", fallback=30))

def load_detections(conf=0.5):
    columns = [
        "frame", "id", "x", "y", "w", "h",
        "conf", "u1", "u2", "u3"
    ]
    df = pd.read_csv(DET_FILE, header=None, names=columns)
    df = df[df["conf"] >= conf]
    return {int(k): v for k, v in df.groupby("frame")}

def get_boxes(group):
    if group is None:
        return []
    boxes = []
    for row in group.itertuples(index=False):
        if row.w <= 0 or row.h <= 0:
            continue
        boxes.append([
            float(row.x), float(row.y),
            float(row.x + row.w), float(row.y + row.h)
        ])
    return boxes

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    a2 = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (a1 + a2 - inter + 1e-9)

class Tracker:
    def __init__(self, threshold=0.3, max_missing=15):
        self.tracks = {}
        self.next_id = 1
        self.threshold = threshold
        self.max_missing = max_missing

    def update(self, boxes):
        from scipy.optimize import linear_sum_assignment

        ids = list(self.tracks)
        matched_ids = set()
        matched_boxes = set()

        if ids and boxes:
            cost = np.array([
                [1 - box_iou(self.tracks[i]["box"], b) for b in boxes]
                for i in ids
            ])
            rows, cols = linear_sum_assignment(cost)

            for r, c in zip(rows, cols):
                if 1 - cost[r, c] < self.threshold:
                    continue
                tid = ids[r]
                self.tracks[tid] = {"box": boxes[c], "missing": 0}
                matched_ids.add(tid)
                matched_boxes.add(c)

        for tid in ids:
            if tid not in matched_ids:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]

        for j, box in enumerate(boxes):
            if j not in matched_boxes:
                self.tracks[self.next_id] = {"box": box, "missing": 0}
                self.next_id += 1

        return {
            tid: obj["box"]
            for tid, obj in self.tracks.items()
            if obj["missing"] == 0
        }

def color_for_id(tid):
    return (
        (tid * 37) % 256,
        (tid * 97) % 256,
        (tid * 157) % 256
    )

def center_of(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int((y1+y2)/2))

def bottom_center(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int(y2))

def draw_tracks(frame, tracks):
    for tid, box in tracks.items():
        x1, y1, x2, y2 = map(int, box)
        color = color_for_id(tid)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"ID {tid}", (x1, max(20, y1-7)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
