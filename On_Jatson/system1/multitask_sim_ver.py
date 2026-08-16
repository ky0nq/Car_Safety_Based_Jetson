#!/usr/bin/env python3
"""
One-checkpoint Multi-task Drowsiness Video/Webcam Inference

Input:
  - Eye ROI  -> Eye head
  - Face ROI -> Yawn head + Head-pose head

Output from ONE .pth checkpoint:
  - P(OPEN), P(CLOSED)
  - P(NO_YAWN), P(YAWN)
  - Pitch / Yaw / Roll

Example:
python3 drowsiness_multitask_video_demo.py \
    --model ./mobilevit_xxs_multitask_best.pth \
    --source ./demo.mp4

Webcam:
python3 drowsiness_multitask_video_demo.py \
    --model ./mobilevit_xxs_multitask_best.pth \
    --source 0

Press q to quit.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MODEL_MAP = {
    "mobilenetv2": "mobilenetv2_100",
    "resnet18": "resnet18",
    "mobilevit_xxs": "mobilevit_xxs",
}


class MultiTaskDrowsinessModel(nn.Module):
    def __init__(self, model_key, pretrained=False, dropout=0.2):
        super().__init__()

        if model_key not in MODEL_MAP:
            raise ValueError(f"Unsupported model_key: {model_key}")

        self.model_key = model_key

        self.backbone = timm.create_model(
            MODEL_MAP[model_key],
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features

        self.eye_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2),
        )

        self.yawn_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2),
        )

        self.pose_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 3),
            nn.Tanh(),
        )

    def forward_multitask(self, eye_x, face_x):
        eye_feat = self.backbone(eye_x)
        face_feat = self.backbone(face_x)

        eye_logits = self.eye_head(eye_feat)
        yawn_logits = self.yawn_head(face_feat)
        pose_norm = self.pose_head(face_feat)

        return eye_logits, yawn_logits, pose_norm


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument(
        "--source",
        default="0",
        help="Video path or webcam index (e.g. ./demo.mp4 or 0)",
    )
    # 이전 실행 명령과의 호환용. --source가 우선입니다.
    p.add_argument("--camera", type=int, default=None)
    p.add_argument(
        "--loop",
        action="store_true",
        help="Video source를 끝까지 재생한 뒤 처음부터 반복",
    )
    p.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Video 재생 속도. 1.0=원본 속도",
    )
    # Webcam capture resolution
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)

    # OpenCV display window initial size
    # WINDOW_NORMAL을 사용하므로 실행 중 마우스로 자유롭게 확대/축소 가능
    p.add_argument("--window-width", type=int, default=1200)
    p.add_argument("--window-height", type=int, default=540)

    p.add_argument("--infer-every", type=int, default=1)

    p.add_argument("--closed-threshold", type=float, default=0.5)
    p.add_argument("--yawn-threshold", type=float, default=0.5)

    # Eye / Head는 최근 몇 초 비율로 계산
    p.add_argument("--window-sec", type=float, default=3.0)

    # Yawn은 프레임 비율이 아니라 이벤트 횟수로 계산
    p.add_argument(
        "--yawn-window-sec",
        type=float,
        default=60.0,
        help="최근 몇 초 동안의 하품 횟수를 사용할지",
    )
    p.add_argument(
        "--min-yawn-sec",
        type=float,
        default=0.7,
        help="P(yawn)이 threshold 이상으로 이 시간 이상 지속되면 하품 1회",
    )
    p.add_argument(
        "--yawns-for-full-score",
        type=int,
        default=2,
        help="window 내 몇 회 하품이면 yawn_score=1.0인지",
    )

    p.add_argument("--calibration-sec", type=float, default=2.0)
    p.add_argument("--head-down-deg", type=float, default=15.0)
    p.add_argument(
        "--head-down-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
    )

    p.add_argument("--warning-threshold", type=float, default=0.40)
    p.add_argument("--danger-threshold", type=float, default=0.70)

    p.add_argument(
        "--face-cascade",
        default=None,
        help="Optional haarcascade_frontalface_default.xml path",
    )

    return p.parse_args()


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_key = ckpt["model_key"]

    print("Checkpoint :", checkpoint_path)
    print("Model      :", model_key)
    print("Tasks      :", ckpt.get("tasks"))
    print("Input size :", ckpt.get("input_size", 256))
    print("Best epoch :", ckpt.get("best_epoch"))

    model = MultiTaskDrowsinessModel(
        model_key=model_key,
        pretrained=False,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, ckpt


def build_transform(ckpt):
    size = int(
        ckpt.get(
            "input_size",
            256,
        )
    )

    mean = tuple(
        ckpt.get(
            "imagenet_mean",
            (0.485, 0.456, 0.406),
        )
    )

    std = tuple(
        ckpt.get(
            "imagenet_std",
            (0.229, 0.224, 0.225),
        )
    )

    return transforms.Compose([
        transforms.Resize(
            int(size / 0.875),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def bgr_to_tensor(image, transform, device):
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    pil = Image.fromarray(rgb)

    return (
        transform(pil)
        .unsqueeze(0)
        .to(device)
    )


def create_face_detector(path=None):
    if path is None:
        if not hasattr(cv2, "data"):
            raise RuntimeError(
                "cv2.data.haarcascades가 없습니다. "
                "--face-cascade로 XML 경로를 지정하세요."
            )

        path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

    detector = cv2.CascadeClassifier(path)

    if detector.empty():
        raise RuntimeError(
            f"Face detector load failed: {path}"
        )

    print("Face detector:", path)

    return detector


def detect_face(frame, detector):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
    )

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda r: r[2] * r[3],
    )


def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(x1 + 1, min(int(x2), W))
    y2 = max(y1 + 1, min(int(y2), H))

    return x1, y1, x2, y2


def get_face_crop(frame, face):
    x, y, w, h = [
        int(v)
        for v in face
    ]

    H, W = frame.shape[:2]

    scale = 1.12

    cx = x + w / 2
    cy = y + h / 2

    nw = w * scale
    nh = h * scale

    box = clamp_box(
        cx - nw / 2,
        cy - nh / 2,
        cx + nw / 2,
        cy + nh / 2,
        W,
        H,
    )

    x1, y1, x2, y2 = box

    return (
        frame[y1:y2, x1:x2],
        box,
    )


def get_eye_crops(frame, face):
    """
    Face bbox 비율로 좌/우 눈 ROI를 근사합니다.

    MRL eye dataset처럼 eye crop을 입력으로 학습했기 때문에
    얼굴 전체가 아니라 각 눈 영역을 별도 입력합니다.
    """

    x, y, w, h = [
        int(v)
        for v in face
    ]

    H, W = frame.shape[:2]

    y1 = y + 0.20 * h
    y2 = y + 0.52 * h

    left_box = clamp_box(
        x + 0.08 * w,
        y1,
        x + 0.50 * w,
        y2,
        W,
        H,
    )

    right_box = clamp_box(
        x + 0.50 * w,
        y1,
        x + 0.92 * w,
        y2,
        W,
        H,
    )

    lx1, ly1, lx2, ly2 = left_box
    rx1, ry1, rx2, ry2 = right_box

    left = frame[
        ly1:ly2,
        lx1:lx2,
    ]

    right = frame[
        ry1:ry2,
        rx1:rx2,
    ]

    return (
        left,
        right,
        left_box,
        right_box,
    )


@torch.inference_mode()
def infer_multitask(
    model,
    left_eye,
    right_eye,
    face_crop,
    transform,
    device,
    pose_scale,
):
    """
    효율적인 실시간 추론:
      - 좌/우 Eye ROI는 batch=2로 backbone 한 번
      - Face ROI는 backbone 한 번
      - 하나의 model instance에서 3 task 출력

    즉 매 inference 시 backbone forward는 총 2회입니다.
    """

    left_x = bgr_to_tensor(
        left_eye,
        transform,
        device,
    )

    right_x = bgr_to_tensor(
        right_eye,
        transform,
        device,
    )

    eye_x = torch.cat(
        [left_x, right_x],
        dim=0,
    )

    face_x = bgr_to_tensor(
        face_crop,
        transform,
        device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    eye_feat = model.backbone(
        eye_x
    )

    face_feat = model.backbone(
        face_x
    )

    eye_logits = model.eye_head(
        eye_feat
    )

    yawn_logits = model.yawn_head(
        face_feat
    )

    pose_norm = model.pose_head(
        face_feat
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    infer_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    eye_prob = torch.softmax(
        eye_logits,
        dim=1,
    )

    # 좌/우 눈의 CLOSED 확률 평균
    p_open = float(
        eye_prob[:, 0].mean().item()
    )

    p_closed = float(
        eye_prob[:, 1].mean().item()
    )

    yawn_prob = torch.softmax(
        yawn_logits,
        dim=1,
    )[0]

    pose_deg = (
        pose_norm[0]
        * pose_scale
    )

    return {
        "p_open": p_open,
        "p_closed": p_closed,
        "p_no_yawn": float(
            yawn_prob[0].item()
        ),
        "p_yawn": float(
            yawn_prob[1].item()
        ),
        "pitch": float(
            pose_deg[0].item()
        ),
        "yaw": float(
            pose_deg[1].item()
        ),
        "roll": float(
            pose_deg[2].item()
        ),
        "infer_ms": infer_ms,
    }

def clean_history(history, now, window_sec):
    cutoff = now - window_sec

    while (
        history
        and history[0]["time"] < cutoff
    ):
        history.popleft()


def safe_mean(values):
    values = list(values)

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def get_status(
    risk,
    ready,
    warning_th,
    danger_th,
):
    if not ready:
        return "WARMUP"

    if risk >= danger_th:
        return "DANGER"

    if risk >= warning_th:
        return "WARNING"

    return "NORMAL"


def status_color(status):
    if status == "NORMAL":
        return (80, 220, 120)

    if status == "WARNING":
        return (0, 190, 255)

    if status == "DANGER":
        return (50, 50, 255)

    return (255, 180, 70)


def put_text(
    frame,
    text,
    x,
    y,
    scale=0.6,
    color=(255, 255, 255),
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _overlay_rect(image, pt1, pt2, color, alpha=0.75):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def _draw_metric_bar(canvas, x, y, width, value, color, label, value_text):
    value = float(np.clip(value, 0.0, 1.0))
    put_text(canvas, label, x, y, 0.42, (175, 180, 188), 1)
    put_text(canvas, value_text, x + width - 86, y, 0.42, (238, 240, 244), 1)

    bar_y = y + 10
    cv2.rectangle(canvas, (x, bar_y), (x + width, bar_y + 7), (55, 58, 64), -1)
    fill_w = int(width * value)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, bar_y), (x + fill_w, bar_y + 7), color, -1)


def draw_corner_box(frame, box, color, thickness=2, corner_ratio=0.22):
    if box is None:
        return

    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cw = max(10, int(w * corner_ratio))
    ch = max(10, int(h * corner_ratio))

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + cw, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + ch), color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - cw, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + ch), color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + cw, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - ch), color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cw, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - ch), color, thickness, cv2.LINE_AA)


def draw_ui(
    frame,
    result,
    closed_ratio,
    yawn_count,
    yawn_score,
    head_ratio,
    pitch_delta,
    risk,
    status,
    fps,
    face_found=True,
    calibrated=True,
):
    """Return a presentation-friendly canvas with metrics in a side panel.

    The camera image itself is not darkened or covered, so the driver's face
    stays fully visible. All monitoring information is rendered outside the
    video area in a compact dashboard on the right.
    """

    H, W = frame.shape[:2]
    panel_w = 340

    canvas = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
    canvas[:, :W] = frame
    panel_x = W
    canvas[:, panel_x:] = (20, 22, 27)

    cv2.line(canvas, (W, 0), (W, H), (72, 76, 84), 1, cv2.LINE_AA)

    color = status_color(status)
    px = panel_x + 20
    right = panel_x + panel_w - 20
    content_w = panel_w - 40

    # ---------- Header ----------
    put_text(canvas, "DROWSINESS MONITOR", px, 28, 0.56, (242, 243, 246), 2)
    put_text(canvas, "ON-DEVICE AI", px, 49, 0.34, (135, 142, 152), 1)
    cv2.line(canvas, (px, 59), (right, 59), (55, 58, 64), 1)

    # ---------- Status ----------
    _overlay_rect(canvas, (px, 69), (right, 116), (34, 36, 42), 0.96)
    cv2.rectangle(canvas, (px, 69), (px + 5, 116), color, -1)
    put_text(canvas, "STATUS", px + 16, 88, 0.34, (145, 150, 160), 1)
    put_text(canvas, status, px + 16, 109, 0.63, color, 2)

    if not face_found:
        note = "FACE NOT FOUND"
        note_color = (50, 50, 255)
    elif not calibrated:
        note = "LOOK FORWARD - CALIBRATING"
        note_color = (0, 190, 255)
    else:
        note = "DRIVER TRACKING ACTIVE"
        note_color = (100, 220, 120)
    put_text(canvas, note, px, 139, 0.36, note_color, 1)

    # ---------- Eye ----------
    put_text(canvas, "EYE STATE", px, 166, 0.43, (228, 230, 235), 2)
    _draw_metric_bar(
        canvas, px, 185, content_w,
        result["p_closed"],
        (70, 150, 255),
        "CLOSED",
        f"{result['p_closed']*100:5.1f}%",
    )

    # ---------- Yawn ----------
    put_text(canvas, "YAWN", px, 223, 0.43, (228, 230, 235), 2)
    _draw_metric_bar(
        canvas, px, 242, content_w,
        result["p_yawn"],
        (70, 150, 255),
        "PROBABILITY",
        f"{result['p_yawn']*100:5.1f}%",
    )
    put_text(canvas, f"Events {yawn_count}", px, 275, 0.36, (170, 176, 185), 1)

    # ---------- Head pose ----------
    put_text(canvas, "HEAD POSE", px, 303, 0.43, (228, 230, 235), 2)
    delta = "N/A" if pitch_delta is None else f"{pitch_delta:.1f}"
    put_text(canvas, f"Pitch {result['pitch']:5.1f}", px, 326, 0.37, (205, 208, 214), 1)
    put_text(canvas, f"Yaw {result['yaw']:5.1f}", px + 150, 326, 0.37, (205, 208, 214), 1)
    put_text(canvas, f"Roll  {result['roll']:5.1f}", px, 347, 0.37, (205, 208, 214), 1)
    put_text(canvas, f"Delta {delta}", px + 150, 347, 0.37, color, 1)

    # ---------- Recent-window summary ----------
    cv2.line(canvas, (px, 360), (right, 360), (55, 58, 64), 1)
    put_text(canvas, "RECENT WINDOW", px, 380, 0.39, (180, 185, 194), 1)
    put_text(canvas, f"Eye Closed  {closed_ratio*100:3.0f}%", px, 402, 0.37, (210, 212, 218), 1)
    put_text(canvas, f"Head Down   {head_ratio*100:3.0f}%", px + 150, 402, 0.37, (210, 212, 218), 1)

    # ---------- Risk card ----------
    risk_y1 = 414
    risk_y2 = min(H - 32, 455)
    if risk_y2 > risk_y1 + 20:
        _overlay_rect(canvas, (px, risk_y1), (right, risk_y2), (34, 36, 42), 0.96)
        put_text(canvas, "RISK", px + 12, risk_y1 + 21, 0.37, (155, 160, 170), 1)
        put_text(canvas, f"{risk:.2f}", right - 62, risk_y1 + 22, 0.56, color, 2)
        bar_y = risk_y2 - 10
        cv2.rectangle(canvas, (px + 12, bar_y), (right - 12, bar_y + 5), (55, 58, 64), -1)
        risk_w = int((content_w - 24) * float(np.clip(risk, 0.0, 1.0)))
        if risk_w > 0:
            cv2.rectangle(canvas, (px + 12, bar_y), (px + 12 + risk_w, bar_y + 5), color, -1)

    # Compact footer. No video current-time / total-length information.
    put_text(
        canvas,
        f"{result['infer_ms']:.1f} ms   |   FPS {fps:.1f}",
        px,
        H - 10,
        0.33,
        (130, 136, 145),
        1,
    )

    return canvas

def main():
    args = parse_args()

    if args.infer_every < 1:
        raise ValueError(
            "--infer-every must be >= 1"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model, ckpt = load_model(
        args.model,
        device,
    )

    transform = build_transform(
        ckpt
    )

    pose_scale = float(
        ckpt.get(
            "pose_scale",
            90.0,
        )
    )

    detector = create_face_detector(
        args.face_cascade
    )

    # --------------------------------------------------
    # Input source
    #   --source 0            -> webcam
    #   --source ./demo.mp4   -> video file
    # --------------------------------------------------
    source_arg = args.source
    if args.camera is not None and args.source == "0":
        source_arg = str(args.camera)

    if str(source_arg).lstrip("-").isdigit():
        source = int(source_arg)
        is_video_file = False
    else:
        source = str(source_arg)
        is_video_file = True

    if args.playback_speed <= 0:
        raise ValueError("--playback-speed must be > 0")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    if not is_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0

    print("Source     :", source)
    if is_video_file:
        print("Video FPS  :", f"{source_fps:.2f}")

    result = {
        "p_open": 1.0,
        "p_closed": 0.0,
        "p_no_yawn": 1.0,
        "p_yawn": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "infer_ms": 0.0,
    }

    history = deque()

    # Yawn event count state
    yawn_events = deque()
    yawn_candidate_start = None
    yawn_latched = False

    baseline_pitch = None
    calibration_start = None
    calibration_samples = []

    frame_idx = 0

    prev_t = time.perf_counter()
    fps_ema = 0.0

    print()
    print("Press q to quit.")

    # Resizable OpenCV window
    window_name = "Multi-task Drowsiness Demo"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )

    # 초기 창 크기만 지정합니다.
    # 이후에는 마우스로 창 모서리를 드래그해 자유롭게 확대/축소할 수 있습니다.
    cv2.resizeWindow(
        window_name,
        args.window_width,
        args.window_height,
    )

    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()

            if not ok:
                if is_video_file and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    # 새 반복 재생은 새로운 시연으로 간주하여 상태 초기화
                    history.clear()
                    yawn_events.clear()
                    yawn_candidate_start = None
                    yawn_latched = False
                    baseline_pitch = None
                    calibration_start = None
                    calibration_samples = []
                    frame_idx = 0
                    result = {
                        "p_open": 1.0,
                        "p_closed": 0.0,
                        "p_no_yawn": 1.0,
                        "p_yawn": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "roll": 0.0,
                        "infer_ms": 0.0,
                    }
                    continue

                break

            now = time.monotonic()

            face = detect_face(
                frame,
                detector,
            )

            face_box = None
            left_box = None
            right_box = None

            face_found = (
                face is not None
            )

            if face_found:
                face_crop, face_box = (
                    get_face_crop(
                        frame,
                        face,
                    )
                )

                (
                    left_eye,
                    right_eye,
                    left_box,
                    right_box,
                ) = get_eye_crops(
                    frame,
                    face,
                )

                if (
                    frame_idx
                    % args.infer_every
                    == 0
                ):
                    result = infer_multitask(
                        model,
                        left_eye,
                        right_eye,
                        face_crop,
                        transform,
                        device,
                        pose_scale,
                    )

                # Head pose baseline calibration
                if baseline_pitch is None:
                    if calibration_start is None:
                        calibration_start = now
                        calibration_samples = []

                        print(
                            "Head pitch calibration started."
                        )

                    calibration_samples.append(
                        result["pitch"]
                    )

                    if (
                        now - calibration_start
                        >= args.calibration_sec
                    ):
                        baseline_pitch = float(
                            np.median(
                                calibration_samples
                            )
                        )

                        print(
                            "Baseline pitch:",
                            f"{baseline_pitch:.2f} deg"
                        )

            if face_found:
                eye_closed = int(
                    result["p_closed"]
                    >= args.closed_threshold
                )

                yawn_active = (
                    result["p_yawn"]
                    >= args.yawn_threshold
                )

                if baseline_pitch is None:
                    pitch_delta = None
                    head_down = None

                else:
                    pitch_delta = (
                        args.head_down_sign
                        * (
                            result["pitch"]
                            - baseline_pitch
                        )
                    )

                    head_down = int(
                        pitch_delta
                        >= args.head_down_deg
                    )

            else:
                eye_closed = None
                yawn_active = False
                pitch_delta = None
                head_down = None

            # --------------------------------------------------
            # Yawn event counting
            #
            # P(yawn) >= threshold 상태가 min_yawn_sec 이상 지속되면
            # 하품 1회로 기록합니다.
            # yawn_latched로 한 번의 긴 하품이 여러 번 카운트되는 것을
            # 방지합니다.
            # --------------------------------------------------
            if yawn_active:
                if yawn_candidate_start is None:
                    yawn_candidate_start = now

                yawn_duration = (
                    now - yawn_candidate_start
                )

                if (
                    yawn_duration >= args.min_yawn_sec
                    and not yawn_latched
                ):
                    yawn_events.append(now)
                    yawn_latched = True

            else:
                # 하품 상태가 끝나야 다음 하품을 새 이벤트로 인식
                yawn_candidate_start = None
                yawn_latched = False

            # 최근 yawn_window_sec 밖의 이벤트 제거
            yawn_cutoff = (
                now - args.yawn_window_sec
            )

            while (
                yawn_events
                and yawn_events[0] < yawn_cutoff
            ):
                yawn_events.popleft()

            yawn_count = len(yawn_events)

            # 기본: 0회 -> 0.0, 1회 -> 0.5, 2회 이상 -> 1.0
            yawn_score = min(
                yawn_count
                / max(
                    args.yawns_for_full_score,
                    1,
                ),
                1.0,
            )

            history.append({
                "time": now,
                "eye_closed": eye_closed,
                "head_down": head_down,
            })

            clean_history(
                history,
                now,
                args.window_sec,
            )

            closed_ratio = safe_mean(
                x["eye_closed"]
                for x in history
                if x["eye_closed"] is not None
            )

            head_ratio = safe_mean(
                x["head_down"]
                for x in history
                if x["head_down"] is not None
            )

            # 기본 fusion score
            # Risk Score
            # Eye  : 최근 window-sec 동안 감긴 프레임 비율
            # Yawn : 최근 yawn-window-sec 동안 하품 횟수 기반 점수
            # Head : 최근 window-sec 동안 고개 숙임 비율
            risk = (
                0.45 * closed_ratio
                + 0.25 * yawn_score
                + 0.30 * head_ratio
            )

            ready = (
                baseline_pitch is not None
                and len(history) >= 2
                and (
                    history[-1]["time"]
                    - history[0]["time"]
                ) >= min(
                    2.0,
                    args.window_sec,
                )
            )

            status = get_status(
                risk,
                ready,
                args.warning_threshold,
                args.danger_threshold,
            )

            current_t = time.perf_counter()
            dt = current_t - prev_t
            prev_t = current_t

            if dt > 0:
                fps_now = 1.0 / dt

                if fps_ema == 0.0:
                    fps_ema = fps_now
                else:
                    fps_ema = (
                        0.9 * fps_ema
                        + 0.1 * fps_now
                    )

            # Cleaner ROI visualization: corner brackets only.
            if face_box is not None:
                draw_corner_box(frame, face_box, status_color(status), 2, 0.18)

            for box in [left_box, right_box]:
                if box is not None:
                    draw_corner_box(frame, box, (255, 185, 75), 1, 0.30)

            display = draw_ui(
                frame,
                result,
                closed_ratio,
                yawn_count,
                yawn_score,
                head_ratio,
                pitch_delta,
                risk,
                status,
                fps_ema,
                face_found=face_found,
                calibrated=(baseline_pitch is not None),
            )

            cv2.imshow(
                window_name,
                display,
            )

            # Video file은 원본 FPS에 가깝게 재생.
            # 추론/렌더링에 이미 사용한 시간을 제외한 만큼만 기다립니다.
            if is_video_file:
                target_sec = 1.0 / (source_fps * args.playback_speed)
                used_sec = time.perf_counter() - frame_start
                wait_ms = max(1, int((target_sec - used_sec) * 1000.0))
            else:
                wait_ms = 1

            key = (
                cv2.waitKey(wait_ms)
                & 0xFF
            )

            if key == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()