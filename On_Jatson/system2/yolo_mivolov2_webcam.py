#!/usr/bin/env python3
"""
Jetson Orin Nano webcam presentation demo

- YOLO detects every person, dog and cat at a configurable frame interval.
- Dog and cat are displayed with one label: animal.
- Every detected person is assigned a simple tracking ID.
- MiVOLO TorchScript estimates age on CUDA for one tracked person at a time.
- Age values are cached and median-smoothed per person.
- A background webcam thread keeps only the newest camera frame to reduce delay.
- Uses the same presentation UI as the prerecorded demo-video version.

Controls
--------
Q / ESC : quit
R       : reset person IDs and cached ages
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from alert_client import AlertClient


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class PersonTrack:
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float
    missed_frames: int = 0
    raw_age: Optional[float] = None
    smoothed_age: Optional[float] = None
    face_box: Optional[tuple[int, int, int, int]] = None
    age_history: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def update_age(self, predicted_age: float, window_size: int) -> None:
        window_size = max(1, int(window_size))

        if self.age_history.maxlen != window_size:
            old_values = list(self.age_history)
            self.age_history = deque(old_values[-window_size:], maxlen=window_size)

        self.raw_age = float(predicted_age)
        self.age_history.append(float(predicted_age))
        self.smoothed_age = float(
            np.median(np.asarray(self.age_history, dtype=np.float32))
        )


class ThreadedWebcam:
    """Capture frames in the background and retain only the newest frame."""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        fourcc: str,
    ) -> None:
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        fourcc_text = (fourcc.upper() + "    ")[:4]

        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*fourcc_text),
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))

        fourcc_value = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        self.actual_fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF)
            for index in range(4)
        )

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedWebcam":
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while self._running:
            success, frame = self.capture.read()

            if not success or frame is None:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None

            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self.capture.release()



class PersonTracker:
    """Simple IoU tracker for a fixed in-cabin webcam demonstration."""

    def __init__(
        self,
        iou_threshold: float,
        max_missed_frames: int,
        age_window: int,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.age_window = max(1, int(age_window))
        self.next_track_id = 1
        self.tracks: dict[int, PersonTrack] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def update(self, detections: list[Detection]) -> list[PersonTrack]:
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                score = calculate_iou(track.box, detection.box)

                if score >= self.iou_threshold:
                    candidates.append((score, track_id, detection_index))

        candidates.sort(key=lambda item: item[0], reverse=True)

        for _score, track_id, detection_index in candidates:
            if track_id not in unmatched_track_ids:
                continue

            if detection_index not in unmatched_detection_indices:
                continue

            detection = detections[detection_index]
            track = self.tracks[track_id]
            track.box = detection.box
            track.confidence = detection.confidence
            track.missed_frames = 0

            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed_frames += 1

            if track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        for detection_index in sorted(unmatched_detection_indices):
            detection = detections[detection_index]

            self.tracks[self.next_track_id] = PersonTrack(
                track_id=self.next_track_id,
                box=detection.box,
                confidence=detection.confidence,
                age_history=deque(maxlen=self.age_window),
            )
            self.next_track_id += 1

        visible_tracks = [
            track
            for track in self.tracks.values()
            if track.missed_frames == 0
        ]
        visible_tracks.sort(key=lambda track: track.track_id)
        return visible_tracks


class AbandonmentMonitor:
    """시동 OFF + 아동/반려동물 감지 상태를 시간에 따라 1~3단계로 승격시키는
    아주 단순한 상태 머신.

    - 조건(시동 OFF 이고 차량 내 아동 또는 동물이 1명/1마리 이상)이 처음 성립한
      시각을 기록해두고, 그 이후 경과 시간을 각 단계 임계값과 비교한다.
    - 조건이 깨지면(시동이 켜지거나, 아무도 남아있지 않으면) 즉시 0단계로
      리셋되고 타이머도 초기화된다. -> 오탐 누적을 막기 위함.
    """

    def __init__(
        self,
        stage1_seconds: float,
        stage2_seconds: float,
        stage3_seconds: float,
    ) -> None:
        self.stage1_seconds = float(stage1_seconds)
        self.stage2_seconds = float(stage2_seconds)
        self.stage3_seconds = float(stage3_seconds)
        self.abandon_start_time: Optional[float] = None

    def update(
        self,
        engine_on: bool,
        occupants_present: bool,
        now: Optional[float] = None,
    ) -> tuple[int, float]:
        """현재 (stage, elapsed_seconds)를 반환한다."""
        now = time.monotonic() if now is None else now
        condition = (not engine_on) and occupants_present

        if not condition:
            self.abandon_start_time = None
            return 0, 0.0

        if self.abandon_start_time is None:
            self.abandon_start_time = now

        elapsed = now - self.abandon_start_time

        if elapsed >= self.stage3_seconds:
            stage = 3
        elif elapsed >= self.stage2_seconds:
            stage = 2
        elif elapsed >= self.stage1_seconds:
            stage = 1
        else:
            stage = 0

        return stage, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO multi-person + MiVOLO age + animal display"
    )

    parser.add_argument("--mivolo", required=True)
    parser.add_argument("--detector", default="./yolo11n.pt")

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")

    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Show the presentation window in fullscreen mode.",
    )

    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.35)
    parser.add_argument(
        "--yolo-every",
        type=int,
        default=2,
        help=(
            "Run YOLO every N frames and reuse the latest detections "
            "between inference frames."
        ),
    )

    parser.add_argument(
        "--person-labels",
        default="person,adult,child,baby",
    )
    parser.add_argument(
        "--animal-labels",
        default="dog,cat",
    )

    parser.add_argument(
        "--age-interval",
        type=float,
        default=1.0,
        help=(
            "Seconds between MiVOLO calls. "
            "Only one visible person is processed per call."
        ),
    )
    parser.add_argument(
        "--age-window",
        type=int,
        default=5,
        help="Median smoothing window for each tracked person.",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--mivolo-fp32",
        action="store_true",
        help=(
            "Run MiVOLO TorchScript in FP32 instead of the default FP16. "
            "Use this only if FP16 causes an unsupported-operation error."
        ),
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=5,
        help=(
            "Maximum number of people to track and process. "
            "If more people are detected, the highest-confidence detections "
            "are selected first."
        ),
    )

    parser.add_argument("--tracker-iou", type=float, default=0.30)
    parser.add_argument("--tracker-max-missed", type=int, default=15)
    parser.add_argument("--no-display-crops", action="store_true")

    # --- 방치 감지 / 알림 서버 연동 -------------------------------------
    parser.add_argument(
        "--alert-server-url",
        default="http://127.0.0.1:8000",
        help="alert_server.py 주소. 웹 대시보드 + 텔레그램 알림을 담당한다.",
    )
    parser.add_argument(
        "--no-alert-server",
        action="store_true",
        help="알림 서버로 전송하지 않고 로컬 창에만 표시(오프라인 테스트용).",
    )
    parser.add_argument(
        "--engine-key",
        default="e",
        help="시동 ON/OFF를 시뮬레이션하는 키(데모용). 기본 'e'.",
    )
    parser.add_argument(
        "--engine-start-on",
        action="store_true",
        default=True,
        help="프로그램 시작 시 시동이 켜진 상태로 시작(기본값).",
    )
    parser.add_argument(
        "--stage1-seconds",
        type=float,
        default=10.0,
        help="시동 꺼짐 + 아동/동물 감지 후 1단계(저소음 알림)까지의 시간(초). "
        "데모: 10초 / 실서비스 제안: 300초(5분).",
    )
    parser.add_argument(
        "--stage2-seconds",
        type=float,
        default=15.0,
        help="2단계(보호자 텔레그램 알림)까지의 시간(초). "
        "데모: 15초 / 실서비스 제안: 600초(10분).",
    )
    parser.add_argument(
        "--stage3-seconds",
        type=float,
        default=20.0,
        help="3단계(공조 가동)까지의 시간(초). "
        "데모: 20초 / 실서비스 제안: 1200초(20분).",
    )

    return parser.parse_args()


def normalize_label_set(text: str) -> set[str]:
    return {
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    }


def clamp_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(1, min(image_width, x2))
    y2 = max(1, min(image_height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_image(
    image: np.ndarray,
    box: Optional[tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    if box is None:
        return None

    x1, y1, x2, y2 = box
    result = image[y1:y2, x1:x2]

    if result.size == 0:
        return None

    return result


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    union_area = box_area(box_a) + box_area(box_b) - intersection_area

    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


def get_class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def resize_for_display(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return frame

    source_height, source_width = frame.shape[:2]
    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=interpolation,
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8,
    )

    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def draw_box(
    frame: np.ndarray,
    box: Optional[tuple[int, int, int, int]],
    text: str,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 3,
) -> None:
    """Presentation-friendly detection box with a filled label tag."""
    if box is None:
        return

    x1, y1, x2, y2 = box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.68
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        font_thickness,
    )

    tag_y1 = max(0, y1 - text_h - baseline - 14)
    tag_y2 = max(text_h + baseline + 14, y1)
    tag_x2 = min(frame.shape[1] - 1, x1 + text_w + 18)

    cv2.rectangle(
        frame,
        (x1, tag_y1),
        (tag_x2, tag_y2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 9, tag_y2 - baseline - 6),
        font,
        font_scale,
        (15, 15, 15),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_translucent_rect(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.72,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        top_left,
        bottom_right,
        color,
        -1,
    )
    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0.0,
        frame,
    )


def draw_presentation_ui(
    frame: np.ndarray,
    *,
    engine_on: bool,
    child_count: int,
    adult_count: int,
    animal_count: int,
    alert_stage: int,
    alert_elapsed: float,
    fps: float,
    engine_key: str,
) -> None:
    """Compact presentation UI: title, engine state, counts, alert stage."""
    h, w = frame.shape[:2]

    # BGR colors
    white = (245, 245, 245)
    muted = (185, 185, 185)
    dark = (18, 22, 28)
    panel = (24, 29, 36)
    green = (95, 205, 105)
    blue = (235, 170, 70)
    amber = (35, 190, 255)
    orange = (20, 125, 255)
    red = (45, 55, 235)

    # --------------------------------------------------------
    # Top title bar
    # --------------------------------------------------------
    draw_translucent_rect(
        frame,
        (0, 0),
        (w, 64),
        dark,
        0.82,
    )

    cv2.putText(
        frame,
        "CabinGuard",
        (22, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.93,
        white,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Residual Occupant Detection",
        (175, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        muted,
        1,
        cv2.LINE_AA,
    )

    engine_text = "ENGINE ON" if engine_on else "ENGINE OFF"
    engine_color = green if engine_on else red
    (eng_w, eng_h), _ = cv2.getTextSize(
        engine_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        2,
    )
    eng_x1 = max(10, w - eng_w - 42)
    cv2.rectangle(
        frame,
        (eng_x1, 14),
        (w - 18, 50),
        engine_color,
        -1,
    )
    cv2.putText(
        frame,
        engine_text,
        (eng_x1 + 12, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Bottom-left occupant summary panel
    # --------------------------------------------------------
    panel_w = min(520, max(430, int(w * 0.42)))
    panel_h = 160
    px1 = 18
    py2 = h - 18
    py1 = max(76, py2 - panel_h)
    px2 = min(w - 18, px1 + panel_w)

    draw_translucent_rect(
        frame,
        (px1, py1),
        (px2, py2),
        panel,
        0.82,
    )

    cv2.putText(
        frame,
        "CABIN STATUS",
        (px1 + 18, py1 + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        muted,
        1,
        cv2.LINE_AA,
    )

    count_text = (
        f"Child {child_count}    "
        f"Adult {adult_count}    "
        f"Animal {animal_count}"
    )
    cv2.putText(
        frame,
        count_text,
        (px1 + 18, py1 + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        white,
        2,
        cv2.LINE_AA,
    )

    residual_detected = (
        (not engine_on)
        and (child_count > 0 or animal_count > 0)
    )

    if engine_on:
        state_text = "MONITORING"
        stage_text = "No alert"
        stage_color = blue
    elif not residual_detected:
        state_text = "CABIN CLEAR"
        stage_text = "No residual occupant"
        stage_color = green
    elif alert_stage <= 0:
        state_text = "RESIDUAL OCCUPANT DETECTED"
        stage_text = f"Monitoring  {alert_elapsed:.1f}s"
        stage_color = amber
    elif alert_stage == 1:
        state_text = "WARNING"
        stage_text = f"Stage 1  |  {alert_elapsed:.1f}s"
        stage_color = amber
    elif alert_stage == 2:
        state_text = "GUARDIAN ALERT"
        stage_text = f"Stage 2  |  {alert_elapsed:.1f}s"
        stage_color = orange
    else:
        state_text = "CRITICAL"
        stage_text = f"Stage 3  |  HVAC action  |  {alert_elapsed:.1f}s"
        stage_color = red

    cv2.circle(
        frame,
        (px1 + 24, py1 + 98),
        7,
        stage_color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        state_text,
        (px1 + 42, py1 + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        stage_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        stage_text,
        (px1 + 18, py1 + 137),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        white,
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Bottom-right lightweight technical info
    # --------------------------------------------------------
    info = f"{fps:.1f} FPS"
    cv2.putText(
        frame,
        info,
        (max(20, w - 120), h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        white,
        1,
        cv2.LINE_AA,
    )

    controls = (
        f"{engine_key.upper()} Engine   R Reset   Q Exit"
    )
    (ctrl_w, _), _ = cv2.getTextSize(
        controls,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1,
    )
    cv2.putText(
        frame,
        controls,
        (max(20, w - ctrl_w - 20), h - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        muted,
        1,
        cv2.LINE_AA,
    )


def run_yolo(
    model: YOLO,
    frame: np.ndarray,
    confidence: float,
    image_size: int,
) -> list[Detection]:
    results = model.predict(
        source=frame,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    boxes = results[0].boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    image_height, image_width = frame.shape[:2]
    detections: list[Detection] = []

    for box_values, score, class_id in zip(xyxy, scores, class_ids):
        valid_box = clamp_box(
            box_values,
            image_width,
            image_height,
        )

        if valid_box is None:
            continue

        detections.append(
            Detection(
                label=get_class_name(model.names, int(class_id)).lower(),
                confidence=float(score),
                box=valid_box,
            )
        )

    return detections


def detect_face_inside_person(
    frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
    person_box: tuple[int, int, int, int],
) -> Optional[tuple[int, int, int, int]]:
    person_crop = crop_image(frame, person_box)

    if person_crop is None:
        return None

    person_x1, person_y1, _, _ = person_box
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(
        faces,
        key=lambda item: item[2] * item[3],
    )

    margin_x = int(width * 0.12)
    margin_y = int(height * 0.12)
    frame_height, frame_width = frame.shape[:2]

    return clamp_box(
        (
            x + person_x1 - margin_x,
            y + person_y1 - margin_y,
            x + person_x1 + width + margin_x,
            y + person_y1 + height + margin_y,
        ),
        frame_width,
        frame_height,
    )


def letterbox_black(image: np.ndarray, size: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    scale = min(size / image_height, size / image_width)

    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_mivolo_torch(
    image: Optional[np.ndarray],
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    MiVOLO preprocessing:
    - black letterbox to 384x384
    - BGR -> RGB
    - [0, 1] scaling
    - ImageNet normalization
    - NCHW tensor
    - transfer to CUDA
    """
    if image is None:
        rgb = np.zeros(
            (size, size, 3),
            dtype=np.float32,
        )
    else:
        processed = letterbox_black(
            image,
            size,
        )
        rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)
        rgb /= 255.0

    normalized = (
        rgb - IMAGENET_MEAN
    ) / IMAGENET_STD

    array = np.ascontiguousarray(
        normalized.transpose(2, 0, 1)[None, ...],
        dtype=np.float32,
    )

    return torch.from_numpy(array).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )


def load_mivolo_torchscript(
    model_path: str,
    use_fp32: bool,
) -> tuple[
    torch.jit.ScriptModule,
    torch.device,
    torch.dtype,
]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch. "
            "MiVOLO TorchScript GPU execution cannot start."
        )

    device = torch.device("cuda:0")
    dtype = (
        torch.float32
        if use_fp32
        else torch.float16
    )

    model = torch.jit.load(
        model_path,
        map_location=device,
    )
    model.eval()
    model = (
        model.float()
        if use_fp32
        else model.half()
    )

    torch.backends.cudnn.benchmark = True

    # Warm up CUDA once so the first visible inference does not include
    # all initialization overhead.
    dummy_face = torch.zeros(
        (1, 3, 384, 384),
        device=device,
        dtype=dtype,
    )
    dummy_body = torch.zeros(
        (1, 3, 384, 384),
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        _ = model(
            dummy_face,
            dummy_body,
        )

    torch.cuda.synchronize(device)

    print(
        "MiVOLO TorchScript device:",
        device,
    )
    print(
        "MiVOLO TorchScript dtype :",
        dtype,
    )
    print(
        "MiVOLO GPU:",
        torch.cuda.get_device_name(0),
    )

    return model, device, dtype


def infer_mivolo_age_torchscript(
    model: torch.jit.ScriptModule,
    device: torch.device,
    dtype: torch.dtype,
    face_crop: Optional[np.ndarray],
    body_crop: Optional[np.ndarray],
) -> float:
    face_tensor = preprocess_mivolo_torch(
        face_crop,
        size=384,
        device=device,
        dtype=dtype,
    )
    body_tensor = preprocess_mivolo_torch(
        body_crop,
        size=384,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        age_output = model(
            face_tensor,
            body_tensor,
        )

    # The provided export wrapper returns one age tensor. Keep a fallback
    # for a TorchScript module that returns a tuple/list.
    if isinstance(age_output, (tuple, list)):
        age_output = age_output[0]

    if not isinstance(age_output, torch.Tensor):
        raise RuntimeError(
            "MiVOLO TorchScript output is not a tensor."
        )

    if age_output.numel() == 0:
        raise RuntimeError(
            "MiVOLO TorchScript returned an empty age output."
        )

    return float(
        age_output
        .detach()
        .float()
        .reshape(-1)[0]
        .item()
    )


def main() -> int:
    args = parse_args()
    mivolo_path = Path(args.mivolo)

    if not mivolo_path.is_file():
        print(
            f"MiVOLO TorchScript file not found: {mivolo_path}",
            file=sys.stderr,
        )
        return 2

    person_labels = normalize_label_set(args.person_labels)
    animal_labels = normalize_label_set(args.animal_labels)

    print("Loading YOLO detector:", args.detector)
    detector = YOLO(
        args.detector,
        task="detect",
    )

    print(
        "Loading MiVOLO TorchScript:",
        args.mivolo,
    )

    try:
        (
            age_model,
            age_device,
            age_dtype,
        ) = load_mivolo_torchscript(
            args.mivolo,
            use_fp32=args.mivolo_fp32,
        )
    except Exception as error:
        print(
            f"Could not load MiVOLO TorchScript: {error}",
            file=sys.stderr,
        )
        return 3

    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)

    if face_cascade.empty():
        print(f"Could not load Haar face cascade: {haar_path}", file=sys.stderr)
        return 3

    # ------------------------------------------------------------
    # Webcam input
    # ------------------------------------------------------------
    try:
        webcam = ThreadedWebcam(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
        ).start()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 4

    print(
        "Input source:",
        f"camera:{args.camera}",
    )
    print(
        "Capture:",
        f"{webcam.actual_width}x{webcam.actual_height}",
        f"{webcam.actual_fps:.1f} FPS",
        f"FOURCC={webcam.actual_fourcc!r}",
    )

    if (
        webcam.actual_width != args.width
        or webcam.actual_height != args.height
    ):
        print(
            "WARNING: Camera did not accept the requested "
            f"{args.width}x{args.height} resolution.",
            file=sys.stderr,
        )

    tracker = PersonTracker(
        iou_threshold=args.tracker_iou,
        max_missed_frames=args.tracker_max_missed,
        age_window=args.age_window,
    )

    # --- 방치 감지 상태 머신 + 알림 서버 클라이언트 ------------------------
    abandonment_monitor = AbandonmentMonitor(
        stage1_seconds=args.stage1_seconds,
        stage2_seconds=args.stage2_seconds,
        stage3_seconds=args.stage3_seconds,
    )
    engine_on = bool(args.engine_start_on)
    engine_key = ord(args.engine_key.lower()[:1]) if args.engine_key else None

    alert_client: Optional[AlertClient] = None
    if not args.no_alert_server:
        alert_client = AlertClient(args.alert_server_url).start()
        print(f"Alert server: {args.alert_server_url}  (키 '{args.engine_key}' = 시동 ON/OFF 토글)")
    else:
        print("Alert server 비활성화(--no-alert-server). 로컬 오버레이만 표시합니다.")

    window_name = "CabinGuard - Residual Occupant Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.fullscreen:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    else:
        cv2.resizeWindow(
            window_name,
            args.display_width,
            args.display_height,
        )

    frame_index = 0
    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    # Cached YOLO detections reduce detector calls.
    cached_detections: list[Detection] = []

    # Round-robin MiVOLO scheduling state.
    age_track_cursor = 0
    last_age_time = 0.0

    print("Controls: Q/ESC=quit, R=reset tracking, E=toggle engine")

    try:
        while True:
            success, frame = webcam.read()

            if not success or frame is None:
                time.sleep(0.005)
                continue

            frame_index += 1

            # Run YOLO only every N frames. Reuse its most recent
            # detections on intermediate frames.
            yolo_interval = max(1, args.yolo_every)

            if (
                frame_index == 1
                or frame_index % yolo_interval == 0
            ):
                cached_detections = run_yolo(
                    detector,
                    frame,
                    confidence=args.detector_conf,
                    image_size=args.detector_imgsz,
                )

            detections = cached_detections

            all_person_detections = [
                detection
                for detection in detections
                if detection.label in person_labels
            ]

            # Track and process at most --max-people people.
            # Confidence is the primary criterion; box area is used as a
            # secondary criterion when confidence values are similar.
            person_detections = sorted(
                all_person_detections,
                key=lambda detection: (
                    detection.confidence,
                    box_area(detection.box),
                ),
                reverse=True,
            )[: max(1, args.max_people)]

            animal_detections = [
                detection
                for detection in detections
                if detection.label in animal_labels
            ]

            visible_tracks = tracker.update(person_detections)

            # MiVOLO is expensive, especially with CPUExecutionProvider.
            # Process only one visible person at each interval.
            current_age_time = time.perf_counter()

            age_candidate_tracks = sorted(
                visible_tracks,
                key=lambda track: track.track_id,
            )

            should_run_age = (
                len(age_candidate_tracks) > 0
                and (
                    current_age_time - last_age_time
                    >= max(0.05, args.age_interval)
                )
            )

            if should_run_age:
                selected_track = age_candidate_tracks[
                    age_track_cursor
                    % len(age_candidate_tracks)
                ]

                age_track_cursor += 1
                last_age_time = current_age_time

                body_crop = crop_image(
                    frame,
                    selected_track.box,
                )
                face_box = detect_face_inside_person(
                    frame,
                    face_cascade,
                    selected_track.box,
                )
                face_crop = crop_image(
                    frame,
                    face_box,
                )
                selected_track.face_box = face_box

                if (
                    body_crop is not None
                    or face_crop is not None
                ):
                    try:
                        predicted_age = (
                            infer_mivolo_age_torchscript(
                                age_model,
                                age_device,
                                age_dtype,
                                face_crop,
                                body_crop,
                            )
                        )
                        selected_track.update_age(
                            predicted_age,
                            args.age_window,
                        )
                    except Exception as error:
                        print(
                            f"MiVOLO error for person "
                            f"{selected_track.track_id}: {error}",
                            file=sys.stderr,
                        )

            child_box_color = (35, 190, 255)
            adult_box_color = (235, 170, 70)
            person_pending_color = (210, 210, 210)
            animal_box_color = (190, 95, 235)

            for track in visible_tracks:
                if track.smoothed_age is None:
                    label = (
                        f"PERSON #{track.track_id} | analyzing..."
                    )
                    box_color = person_pending_color
                elif track.smoothed_age <= args.age_threshold:
                    label = (
                        f"CHILD | {track.smoothed_age:.1f}y"
                    )
                    box_color = child_box_color
                else:
                    label = (
                        f"ADULT | {track.smoothed_age:.1f}y"
                    )
                    box_color = adult_box_color

                draw_box(
                    frame,
                    track.box,
                    label,
                    color=box_color,
                )

            for animal in animal_detections:
                draw_box(
                    frame,
                    animal.box,
                    f"ANIMAL | {animal.confidence:.2f}",
                    color=animal_box_color,
                )

            under_threshold_count = sum(
                1
                for track in visible_tracks
                if (
                    track.smoothed_age is not None
                    and track.smoothed_age <= args.age_threshold
                )
            )

            # --- 방치 감지: 단계 계산 + 알림 서버로 상태 전송 -----------------
            occupants_present = under_threshold_count > 0 or len(animal_detections) > 0

            alert_stage, alert_elapsed = abandonment_monitor.update(
                engine_on=engine_on,
                occupants_present=occupants_present,
            )

            if alert_client is not None:
                occupants_payload = [
                    {"type": "child", "age": track.smoothed_age}
                    for track in visible_tracks
                    if (
                        track.smoothed_age is not None
                        and track.smoothed_age <= args.age_threshold
                    )
                ] + [
                    {"type": "animal", "age": None}
                    for _ in animal_detections
                ]

                alert_client.send(
                    {
                        "engine_on": engine_on,
                        "stage": alert_stage,
                        "elapsed_seconds": alert_elapsed,
                        "person_count": len(visible_tracks),
                        "child_count": under_threshold_count,
                        "animal_count": len(animal_detections),
                        "occupants": occupants_payload,
                    }
                )

            current_time = time.perf_counter()
            frame_time = max(current_time - previous_time, 1e-6)
            instant_fps = 1.0 / frame_time

            if smoothed_fps == 0.0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = 0.9 * smoothed_fps + 0.1 * instant_fps

            previous_time = current_time

            adult_count = sum(
                1
                for track in visible_tracks
                if (
                    track.smoothed_age is not None
                    and track.smoothed_age > args.age_threshold
                )
            )

            draw_presentation_ui(
                frame,
                engine_on=engine_on,
                child_count=under_threshold_count,
                adult_count=adult_count,
                animal_count=len(animal_detections),
                alert_stage=alert_stage,
                alert_elapsed=alert_elapsed,
                fps=smoothed_fps,
                engine_key=args.engine_key,
            )

            display_frame = resize_for_display(
                frame,
                args.display_width,
                args.display_height,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("r"), ord("R")):
                tracker.reset()
                cached_detections = []
                age_track_cursor = 0
                last_age_time = 0.0
                print("Person tracking and cached ages reset.")

            if engine_key is not None and key in (engine_key, engine_key - 32):
                engine_on = not engine_on
                print(f"[DEMO] Engine toggled -> {'ON' if engine_on else 'OFF'}")

    finally:
        webcam.stop()
        cv2.destroyAllWindows()

        if alert_client is not None:
            alert_client.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
