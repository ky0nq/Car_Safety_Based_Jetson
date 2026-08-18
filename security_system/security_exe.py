from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from anti_spoof import MiniFASLiveness
from capture_manager import CaptureManager
from config import CONFIG
from face_database import FaceDatabase
from face_detector import YuNetFaceDetector
from face_recognizer import FaceMatcher, SFaceEmbedder
from registration import register_from_camera
from security_policy import PolicyStatus, UnknownOnlyCapturePolicy
from web_alarm import (
    publish_capture_event,
    start_web_server,
    trigger_web_alarm,
    update_live_status,
)

def send_web_alarm(
    message: str,
    event_dir=None,
    event_type: str = "SECURITY_ALARM",
) -> int:
    """Send a high-priority visual alert to connected PC/phone browsers."""
    return trigger_web_alarm(
        message=message,
        event_dir=event_dir,
        event_type=event_type,
    )


def send_web_capture(
    event_dir,
    message: str,
    event_type: str = "UNKNOWN_CAPTURE",
) -> int:
    """Send a captured event card to the mobile dashboard."""
    return publish_capture_event(
        event_dir=event_dir,
        message=message,
        event_type=event_type,
    )


WINDOW_NAME = "Vehicle Security Camera"
PANEL_WIDTH = 300
UI_MIN_HEIGHT = 600


@dataclass
class FaceObservation:
    face: np.ndarray
    bbox: tuple[int, int, int, int]
    role: str
    name: str
    score: float
    person_id: str | None
    crop: np.ndarray


@dataclass
class UIButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


class SecurityCameraUI:
    """OpenCV-based security camera UI with mouse-click controls."""

    def __init__(self) -> None:
        self.buttons: list[UIButton] = []
        self.pending_action: str | None = None
        self.toast_text = ""
        self.toast_until = 0.0
        self.confirm_action: str | None = None
        self.confirm_until = 0.0

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.contains(x, y):
                self.pending_action = button.action
                break

    def pop_action(self) -> str | None:
        action = self.pending_action
        self.pending_action = None
        return action

    def toast(self, text: str, seconds: float = 1.5) -> None:
        self.toast_text = text
        self.toast_until = time.monotonic() + seconds

    def confirm_or_arm(self, action: str, seconds: float = 4.0) -> bool:
        """Return True only when the same destructive action is clicked twice within the timeout."""
        now = time.monotonic()
        if self.confirm_action == action and now <= self.confirm_until:
            self.confirm_action = None
            self.confirm_until = 0.0
            return True
        self.confirm_action = action
        self.confirm_until = now + seconds
        return False

    def confirmation_active(self, action: str) -> bool:
        now = time.monotonic()
        if now > self.confirm_until:
            self.confirm_action = None
            return False
        return self.confirm_action == action

    @staticmethod
    def _overlay_rect(
        image: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        overlay = image.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        org: tuple[int, int],
        scale: float = 0.55,
        color: tuple[int, int, int] = (235, 235, 235),
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_button(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        label: str,
        action: str,
        emphasis: str = "normal",
    ) -> None:
        x1, y1, x2, y2 = rect
        if emphasis == "danger":
            fill = (45, 45, 125)
            border = (90, 90, 245)
        elif emphasis == "reset":
            fill = (70, 70, 70)
            border = (210, 210, 210)
        else:
            fill = (48, 48, 48)
            border = (105, 105, 105)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)

        text_scale = 0.48
        max_text_width = max(1, x2 - x1 - 12)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        while tw > max_text_width and text_scale > 0.30:
            text_scale -= 0.02
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        self._put_text(canvas, label, (tx, ty), text_scale, (245, 245, 245), 1)
        self.buttons.append(UIButton(label, action, rect))

    def draw(
        self,
        frame: np.ndarray,
        observations: list[FaceObservation],
        status: PolicyStatus,
        unknown_seconds: float,
        alarm_seconds: float,
        spoof_elapsed: float,
        spoof_seconds: float,
        spoof_triggered: bool,
        registered_counts: tuple[int, int],
        source_label: str,
    ) -> np.ndarray:
        camera = frame.copy()
        draw_observations(camera, observations)
        source_height, source_width = camera.shape[:2]
        if source_height < UI_MIN_HEIGHT:
            scale = UI_MIN_HEIGHT / source_height
            camera = cv2.resize(
                camera,
                (max(1, int(round(source_width * scale))), UI_MIN_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        height, width = camera.shape[:2]
        canvas = np.zeros((height, width + PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[:, :width] = camera
        canvas[:, width:] = (24, 24, 24)
        self.buttons = []

        # CCTV header overlay.
        self._overlay_rect(canvas, (0, 0), (width, 56), (15, 15, 15), 0.78)
        cv2.circle(canvas, (20, 27), 6, (0, 0, 255), -1)
        self._put_text(canvas, "REC", (34, 33), 0.58, (245, 245, 245), 2)
        self._put_text(canvas, "CABIN SECURITY", (95, 33), 0.62, (245, 245, 245), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        self._put_text(canvas, timestamp, (max(10, width - tw - 14), 32), 0.48, (215, 215, 215), 1)

        owner_visible = sum(obs.role == "OWNER" for obs in observations)
        guest_visible = sum(obs.role == "GUEST" for obs in observations)
        unknown_visible = sum(obs.role == "UNKNOWN" for obs in observations)
        spoof_visible = sum(obs.role == "SPOOF" for obs in observations)

        # Security status banner at bottom of camera image.
        # SPOOF has the highest priority because it is treated as an attack event
        # even when an OWNER/GUEST is also visible.
        banner_y = max(0, height - 54)
        if spoof_visible > 0:
            if spoof_triggered:
                banner_text = "SPOOF ATTACK  |  CAPTURED + WEB ALARM SENT"
                banner_color = (120, 20, 160)
            else:
                spoof_remaining = max(0.0, spoof_seconds - spoof_elapsed)
                banner_text = f"SPOOF DETECTED  |  SECURITY EVENT IN {spoof_remaining:04.1f}s"
                banner_color = (100, 20, 135)
        elif status.state == "ALARM":
            banner_text = f"WARNING ALARM  |  UNKNOWN PRESENT {status.unknown_presence_elapsed:04.1f}s"
            banner_color = (20, 20, 210)
        elif status.state == "UNKNOWN_TIMER":
            remaining = max(0.0, unknown_seconds - status.unknown_elapsed)
            alarm_remaining = max(0.0, alarm_seconds - status.unknown_presence_elapsed)
            banner_text = (
                f"UNAUTHORIZED PERSON  |  CAPTURE IN {remaining:04.1f}s"
                f"  |  ALERT IN {alarm_remaining:04.1f}s"
            )
            banner_color = (35, 35, 150)
        elif status.state == "CAPTURED":
            alarm_remaining = max(0.0, alarm_seconds - status.unknown_presence_elapsed)
            banner_text = f"SECURITY EVENT CAPTURED  |  ALERT IN {alarm_remaining:04.1f}s"
            banner_color = (35, 35, 150)
        elif status.state == "REGISTERED_PRESENT":
            banner_text = "AUTHORIZED PERSON PRESENT  |  CAPTURE / ALARM DISABLED"
            banner_color = (35, 105, 35)
        else:
            banner_text = "MONITORING  |  NO SECURITY EVENT"
            banner_color = (50, 50, 50)

        self._overlay_rect(canvas, (0, banner_y), (width, height), banner_color, 0.82)
        self._put_text(canvas, banner_text, (16, banner_y + 33), 0.55, (250, 250, 250), 2)

        sx = width
        self._put_text(canvas, "SYSTEM STATUS", (sx + 18, 32), 0.56, (220, 220, 220), 2)
        cv2.line(canvas, (sx + 16, 45), (sx + PANEL_WIDTH - 16, 45), (75, 75, 75), 1)

        state_label = {
            "UNKNOWN_TIMER": "ALERT / WATCHING",
            "CAPTURED": "EVENT CAPTURED",
            "ALARM": "WARNING ALARM",
            "REGISTERED_PRESENT": "AUTHORIZED",
            "EMPTY": "MONITORING",
        }.get(status.state, status.state)
        state_color = (80, 220, 80) if status.state == "REGISTERED_PRESENT" else (
            (80, 80, 255) if status.state in {"UNKNOWN_TIMER", "CAPTURED", "ALARM"} else (200, 200, 200)
        )
        self._put_text(canvas, state_label, (sx + 18, 72), 0.53, state_color, 2)

        source_scale = 0.40
        self._put_text(canvas, "Source", (sx + 18, 98), source_scale, (220, 220, 220), 1)
        (source_tw, _), _ = cv2.getTextSize(source_label, cv2.FONT_HERSHEY_SIMPLEX, source_scale, 1)
        self._put_text(
            canvas,
            source_label,
            (sx + PANEL_WIDTH - 18 - source_tw, 98),
            source_scale,
            (220, 220, 220),
            1,
        )

        if status.unknown_presence_elapsed > 0:
            alarm_remaining = max(0.0, alarm_seconds - status.unknown_presence_elapsed)
            alarm_text = "ALERT ACTIVE" if status.alarm_triggered else f"Alert in {alarm_remaining:04.1f}s"
            self._put_text(canvas, alarm_text, (sx + 18, 119), 0.38, (90, 90, 245), 1)

        self._put_text(canvas, "VISIBLE NOW", (sx + 18, 145), 0.42, (165, 165, 165), 1)

        def draw_panel_row(label: str, value: str, y: int, color: tuple[int, int, int]) -> None:
            row_scale = 0.44
            self._put_text(canvas, label, (sx + 24, y), row_scale, color, 1)
            (value_tw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, row_scale, 1)
            self._put_text(
                canvas,
                value,
                (sx + PANEL_WIDTH - 22 - value_tw, y),
                row_scale,
                color,
                1,
            )

        draw_panel_row("OWNER", str(owner_visible), 169, (100, 235, 100))
        draw_panel_row("GUEST", str(guest_visible), 191, (235, 190, 100))
        draw_panel_row("UNKNOWN", str(unknown_visible), 213, (100, 100, 245))
        draw_panel_row("SPOOF", str(spoof_visible), 235, (220, 100, 230))

        button_h = 30 if height >= 540 else 28
        gap = 5 if height >= 540 else 4
        margin_bottom = 12
        button_rows = 5
        total_button_height = button_rows * button_h + (button_rows - 1) * gap
        button_top = height - margin_bottom - total_button_height

        db_owners, db_guests = registered_counts
        db_title_y = min(278, button_top - 68)
        db_owner_y = db_title_y + 23
        db_guest_y = db_title_y + 45

        if spoof_visible > 0 and db_title_y >= 272:
            spoof_text = "SPOOF ALARM SENT" if spoof_triggered else f"Spoof event in {max(0.0, spoof_seconds - spoof_elapsed):04.1f}s"
            self._put_text(canvas, spoof_text, (sx + 18, 255), 0.36, (220, 100, 230), 1)

        self._put_text(canvas, "REGISTERED DB", (sx + 18, db_title_y), 0.42, (165, 165, 165), 1)
        draw_panel_row("OWNER", f"{db_owners}/{CONFIG.max_owners}", db_owner_y, (220, 220, 220))
        draw_panel_row("GUEST", f"{db_guests}/{CONFIG.max_guests}", db_guest_y, (220, 220, 220))

        x1 = sx + 18
        x2 = sx + PANEL_WIDTH - 18
        y_owner = button_top
        y_guest = y_owner + button_h + gap
        y_list = y_guest + button_h + gap
        y_reset = y_list + button_h + gap
        y_quit = y_reset + button_h + gap

        self._draw_button(canvas, (x1, y_owner, x2, y_owner + button_h), "+ REGISTER OWNER", "OWNER")
        self._draw_button(canvas, (x1, y_guest, x2, y_guest + button_h), "+ REGISTER GUEST", "GUEST")
        self._draw_button(canvas, (x1, y_list, x2, y_list + button_h), "REGISTERED LIST", "LIST")

        mid = (x1 + x2) // 2
        owner_label = "CONFIRM OWNER" if self.confirmation_active("RESET_OWNER") else "RESET OWNERS"
        guest_label = "CONFIRM GUEST" if self.confirmation_active("RESET_GUEST") else "RESET GUESTS"
        self._draw_button(canvas, (x1, y_reset, mid - 3, y_reset + button_h), owner_label, "RESET_OWNER", "reset")
        self._draw_button(canvas, (mid + 3, y_reset, x2, y_reset + button_h), guest_label, "RESET_GUEST", "reset")
        self._draw_button(canvas, (x1, y_quit, x2, y_quit + button_h), "QUIT", "QUIT", "danger")

        # Temporary toast message.
        if self.toast_text and time.monotonic() < self.toast_until:
            toast_w = min(width - 40, 400)
            tx1 = max(20, (width - toast_w) // 2)
            ty1 = 74
            tx2 = tx1 + toast_w
            ty2 = ty1 + 46
            self._overlay_rect(canvas, (tx1, ty1), (tx2, ty2), (20, 20, 20), 0.88)
            cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (150, 150, 150), 1)
            (tw, th), _ = cv2.getTextSize(self.toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            self._put_text(
                canvas,
                self.toast_text,
                (tx1 + max(8, (toast_w - tw) // 2), ty1 + 28 + th // 2),
                0.52,
                (245, 245, 245),
                1,
            )

        return canvas

class ResponsiveWindow:
    def __init__(self, title: str, ui: SecurityCameraUI) -> None:
        self.ui = ui
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1200x680")
        self.root.minsize(900, 520)
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.photo = None
        self.closed = False
        self.base_width = 1
        self.base_height = 1
        self.image_x = 0
        self.image_y = 0
        self.image_width = 1
        self.image_height = 1
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas.bind("<Button-1>", self._on_click)
        self.root.bind("<KeyPress>", self._on_key)
        self.root.update_idletasks()
        self.root.update()

    def _on_click(self, event) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            return
        if not (
            self.image_x <= event.x < self.image_x + self.image_width
            and self.image_y <= event.y < self.image_y + self.image_height
        ):
            return
        x = int((event.x - self.image_x) * self.base_width / self.image_width)
        y = int((event.y - self.image_y) * self.base_height / self.image_height)
        for button in self.ui.buttons:
            if button.contains(x, y):
                self.ui.pending_action = button.action
                break

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in {"q", "escape"}:
            self.ui.pending_action = "QUIT"
        elif key == "l":
            self.ui.pending_action = "LIST"
        elif key == "o":
            self.ui.pending_action = "OWNER"
        elif key == "g":
            self.ui.pending_action = "GUEST"

    def show(self, display: np.ndarray) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        height, width = display.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(display, (canvas_width, canvas_height), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_x = 0
        self.image_y = 0
        self.image_width = canvas_width
        self.image_height = canvas_height
        self.base_width = width
        self.base_height = height
        self.canvas.coords(self.image_id, 0, 0)
        self.canvas.itemconfig(self.image_id, image=self.photo)

    def close(self) -> None:
        self.closed = True
        self.ui.pending_action = "QUIT"

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

def safe_display_name(name: str, role: str) -> str:
    ascii_name = name.encode("ascii", errors="ignore").decode().strip()
    return ascii_name or role


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle cabin face security camera / video demo")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video path or camera index. Example: --source ./demo.mp4 or --source 0",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Webcam index used when --source is omitted (backward compatible)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--process-every", type=int, default=2, help="Run models every N frames")
    parser.add_argument("--loop", action="store_true", help="Loop a video file when it reaches the end")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Video playback speed. 1.0=original, 0.5=half speed, 2.0=double speed",
    )
    parser.add_argument("--recognition-threshold", type=float, default=CONFIG.face_recognition_threshold)
    parser.add_argument("--live-threshold", type=float, default=CONFIG.live_threshold)
    parser.add_argument("--unknown-seconds", type=float, default=CONFIG.unknown_capture_seconds)
    parser.add_argument(
        "--unknown-reset-seconds",
        type=float,
        default=CONFIG.unknown_reset_seconds,
        help="Reset UNKNOWN-only timer after this many continuous seconds",
    )
    parser.add_argument(
        "--unknown-alarm-seconds",
        type=float,
        default=CONFIG.unknown_alarm_seconds,
        help="Send a high-priority web alert after this many uninterrupted UNKNOWN-only seconds",
    )
    parser.add_argument(
        "--spoof-seconds",
        type=float,
        default=CONFIG.spoof_event_seconds,
        help="Capture and send a high-priority web alert after continuous SPOOF detection for this many seconds",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=5000,
        help="Mobile dashboard web server port",
    )
    parser.add_argument("--device", type=str, default=None, help="MiniFAS torch device, e.g. cuda:0 or cpu")
    return parser.parse_args()


def open_input(args: argparse.Namespace):
    """Open either a webcam index or a video file and return playback metadata."""
    source = args.source if args.source is not None else str(args.camera)

    # Numeric source -> webcam index. Non-numeric source -> video file.
    is_camera = source.strip().lstrip("+-").isdigit()

    if is_camera:
        camera_index = int(source)
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        return cap, True, f"/dev/video{camera_index}", 0.0

    video_path = Path(source).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3:
        fps = 30.0

    return cap, False, video_path.name, fps


def draw_observations(frame: np.ndarray, observations: list[FaceObservation]) -> None:
    colors = {
        "OWNER": (0, 220, 0),
        "GUEST": (255, 180, 0),
        "UNKNOWN": (0, 0, 255),
        "SPOOF": (180, 0, 255),
    }
    for obs in observations:
        x, y, w, h = obs.bbox
        color = colors.get(obs.role, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if obs.role in {"OWNER", "GUEST"}:
            label = f"{obs.role}:{safe_display_name(obs.name, obs.role)} {obs.score:.2f}"
        else:
            label = f"{obs.role} {obs.score:.2f}"

        # Dark label background for CCTV readability.
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        label_y = max(58, y)
        cv2.rectangle(
            frame,
            (x, label_y - th - baseline - 7),
            (x + tw + 8, label_y),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x + 4, label_y - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )


def print_people(database: FaceDatabase) -> None:
    rows = database.list_people()
    print("\n[REGISTERED PEOPLE]")
    if not rows:
        print("  (empty)")
    for row in rows:
        print(
            f"  {row['person_id']} | {row['role']:<5} | {row['name']} | "
            f"last_seen={row['last_seen_at']} | seen={row['seen_count']}"
        )
    print()


def registered_counts(database: FaceDatabase) -> tuple[int, int]:
    owners = 0
    guests = 0
    for row in database.list_people():
        if row["role"] == "OWNER":
            owners += 1
        elif row["role"] == "GUEST":
            guests += 1
    return owners, guests


def main() -> int:
    args = parse_args()
    if args.process_every < 1:
        raise ValueError("--process-every must be at least 1")
    if args.playback_speed <= 0:
        raise ValueError("--playback-speed must be > 0")

    detector = YuNetFaceDetector(CONFIG.yunet_model, CONFIG.face_detection_threshold)
    embedder = SFaceEmbedder(CONFIG.sface_model)
    liveness = MiniFASLiveness(CONFIG.minifas_model, threshold=args.live_threshold, device=args.device)
    database = FaceDatabase(CONFIG.database_path, CONFIG.max_owners, CONFIG.max_guests)
    matcher = FaceMatcher(database.load_profiles(), threshold=args.recognition_threshold)
    policy = UnknownOnlyCapturePolicy(
        capture_after_seconds=args.unknown_seconds,
        reset_after_seconds=args.unknown_reset_seconds,
        alarm_after_seconds=args.unknown_alarm_seconds,
    )
    capture_manager = CaptureManager(CONFIG.event_dir)
    ui = SecurityCameraUI()

    # Start the mobile dashboard web server on the Jetson.
    # Open http://<JETSON_IP>:<web_port> from a PC on the same network.
    start_web_server(host="0.0.0.0", port=args.web_port, event_root=CONFIG.event_dir)

    cap, is_camera_source, source_label, source_fps = open_input(args)
    if is_camera_source:
        print(f"[INPUT] Webcam: {source_label}")
    else:
        print(f"[INPUT] Video : {source_label} ({source_fps:.2f} FPS)")
        print(f"[VIDEO] Loop={args.loop}  Playback speed={args.playback_speed:.2f}x")

    window = ResponsiveWindow(WINDOW_NAME, ui)

    print("Mouse: OWNER / GUEST / LIST / RESET OWNERS / RESET GUESTS / QUIT")
    print("Reset buttons require a second click within 4 seconds to confirm.")
    print("Keyboard fallback: O=register OWNER, G=register GUEST, L=list, Q=quit")

    frame_index = 0
    observations: list[FaceObservation] = []
    policy_status = PolicyStatus("EMPTY", 0.0, 0.0, False, False, False)
    visible_person_ids: set[str] = set()
    last_seen_write = 0.0
    db_counts = registered_counts(database)
    last_unknown_event_dir = None

    # SPOOF is tracked separately from UNKNOWN.
    # Continuous SPOOF for --spoof-seconds triggers one capture + one web alarm.
    spoof_start_time: float | None = None
    spoof_elapsed = 0.0
    spoof_triggered = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if not is_camera_source and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    policy.reset()
                    observations = []
                    visible_person_ids = set()
                    policy_status = PolicyStatus("EMPTY", 0.0, 0.0, False, False, False)
                    spoof_start_time = None
                    spoof_elapsed = 0.0
                    spoof_triggered = False
                    last_unknown_event_dir = None
                    frame_index = 0
                    print("[VIDEO] Restart from beginning.")
                    continue

                print("[INFO] Input stream ended.")
                break

            if frame_index % args.process_every == 0:
                observations = []
                faces = detector.detect(frame, max_faces=CONFIG.max_faces)

                for face in faces:
                    bbox = detector.bbox_xywh(face)
                    crop = detector.crop_face(frame, face)
                    try:
                        live = liveness.predict(frame, bbox)
                    except Exception as exc:
                        print(f"[WARN] Anti-spoof inference failed: {exc}")
                        continue

                    if not live.is_live:
                        observations.append(FaceObservation(face, bbox, "SPOOF", "SPOOF", live.score, None, crop))
                        continue

                    try:
                        embedding = embedder.extract(frame, face)
                        match = matcher.match(embedding)
                    except Exception as exc:
                        print(f"[WARN] Face embedding failed: {exc}")
                        continue

                    observations.append(
                        FaceObservation(face, bbox, match.role, match.name, match.score, match.person_id, crop)
                    )

                owner_exists = any(obs.role == "OWNER" for obs in observations)
                guest_exists = any(obs.role == "GUEST" for obs in observations)
                unknown_observations = [obs for obs in observations if obs.role == "UNKNOWN"]
                spoof_observations = [obs for obs in observations if obs.role == "SPOOF"]
                visible_person_ids = {
                    obs.person_id
                    for obs in observations
                    if obs.person_id is not None and obs.role in {"OWNER", "GUEST"}
                }

                now = time.monotonic()

                # Separate SPOOF attack timer. It is independent of OWNER/GUEST/UNKNOWN.
                # A SPOOF that stays visible continuously for 3 seconds (default) is
                # captured and immediately sent to the mobile web dashboard.
                if spoof_observations:
                    if spoof_start_time is None:
                        spoof_start_time = now
                        spoof_triggered = False
                    spoof_elapsed = max(0.0, now - spoof_start_time)

                    if spoof_elapsed >= args.spoof_seconds and not spoof_triggered:
                        event_dir = capture_manager.save_spoof_event(
                            frame=frame,
                            spoof_crops=[obs.crop for obs in spoof_observations],
                            spoof_duration=spoof_elapsed,
                        )
                        send_web_alarm(
                            f"SPOOF attack remained continuously for {spoof_elapsed:.1f}s",
                            event_dir=event_dir,
                            event_type="SPOOF_ATTACK",
                        )
                        spoof_triggered = True
                        print(f"[SPOOF CAPTURE] {event_dir}")
                        print(f"[WEB ALERT] SPOOF attack confirmed after {spoof_elapsed:.1f}s")
                        ui.toast("SPOOF CAPTURED + WEB ALERT SENT", 3.0)
                else:
                    # A new SPOOF episode may trigger again after the old one disappears.
                    spoof_start_time = None
                    spoof_elapsed = 0.0
                    spoof_triggered = False

                if visible_person_ids and now - last_seen_write >= CONFIG.seen_update_interval_seconds:
                    database.update_seen_many(visible_person_ids)
                    last_seen_write = now

                policy_status = policy.update(
                    owner_exists=owner_exists,
                    guest_exists=guest_exists,
                    unknown_exists=bool(unknown_observations),
                    now=now,
                )
                if policy_status.should_capture:
                    event_dir = capture_manager.save_unknown_event(
                        frame=frame,
                        unknown_crops=[obs.crop for obs in unknown_observations],
                        unknown_duration=policy_status.unknown_elapsed,
                    )
                    last_unknown_event_dir = event_dir
                    send_web_capture(
                        event_dir=event_dir,
                        message=(
                            f"UNKNOWN person captured after "
                            f"{policy_status.unknown_elapsed:.1f}s"
                        ),
                        event_type="UNKNOWN_CAPTURE",
                    )
                    print(f"[CAPTURE] {event_dir}")
                    ui.toast("SECURITY EVENT CAPTURED + SENT TO WEB", 2.0)

                if policy_status.should_alarm:
                    send_web_alarm(
                        f"UNKNOWN person remained continuously for "
                        f"{policy_status.unknown_presence_elapsed:.1f}s",
                        event_dir=last_unknown_event_dir,
                        event_type="UNKNOWN_ALARM",
                    )
                    print(
                        f"[WEB ALERT] UNKNOWN remained continuously for "
                        f"{policy_status.unknown_presence_elapsed:.1f}s"
                    )
                    ui.toast(f"WEB ALERT SENT - UNKNOWN {args.unknown_alarm_seconds:.0f}s", 3.0)

            update_live_status(
                owner_count=sum(obs.role == "OWNER" for obs in observations),
                guest_count=sum(obs.role == "GUEST" for obs in observations),
                unknown_count=sum(obs.role == "UNKNOWN" for obs in observations),
                spoof_count=sum(obs.role == "SPOOF" for obs in observations),
                unknown_cycle_seconds=policy_status.unknown_elapsed,
                unknown_presence_seconds=policy_status.unknown_presence_elapsed,
            )

            display = ui.draw(
                frame=frame,
                observations=observations,
                status=policy_status,
                unknown_seconds=args.unknown_seconds,
                alarm_seconds=args.unknown_alarm_seconds,
                spoof_elapsed=spoof_elapsed,
                spoof_seconds=args.spoof_seconds,
                spoof_triggered=spoof_triggered,
                registered_counts=db_counts,
                source_label=source_label,
            )
            window.show(display)

            if is_camera_source:
                wait_ms = 1
            else:
                wait_ms = max(1, int(round(1000.0 / (source_fps * args.playback_speed))))

            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)

            action = ui.pop_action()
            if window.closed:
                action = "QUIT"

            if action == "QUIT":
                break

            if action == "LIST":
                print_people(database)
                ui.toast("REGISTERED LIST PRINTED TO TERMINAL")

            elif action in {"RESET_OWNER", "RESET_GUEST"}:
                role = "OWNER" if action == "RESET_OWNER" else "GUEST"
                if not ui.confirm_or_arm(action):
                    ui.toast(f"CLICK RESET {role} AGAIN TO CONFIRM", 3.5)
                    print(f"[RESET] Click RESET {role} again within 4 seconds to clear that registration list.")
                else:
                    removed = database.clear_role(role)
                    matcher.reload(database.load_profiles())
                    db_counts = registered_counts(database)

                    # Reset only transient recognition state after changing the DB.
                    policy.reset()
                    observations = []
                    visible_person_ids = set()
                    policy_status = PolicyStatus("EMPTY", 0.0, 0.0, False, False, False)
                    spoof_start_time = None
                    spoof_elapsed = 0.0
                    spoof_triggered = False
                    last_seen_write = time.monotonic()

                    ui.toast(f"{role} LIST CLEARED ({removed})", 2.0)
                    print(f"[RESET] Cleared {removed} registered {role}(s). Saved capture events were preserved.")

            elif action in {"OWNER", "GUEST"}:
                role = action
                print(f"\n[UI] {role} registration selected.")
                name = input(f"Enter {role} name: ").strip()
                result = register_from_camera(
                    cap=cap,
                    name=name,
                    role=role,
                    detector=detector,
                    liveness=liveness,
                    embedder=embedder,
                    database=database,
                    recognition_threshold=args.recognition_threshold,
                    sample_count=CONFIG.registration_samples,
                    visible_person_ids=visible_person_ids,
                )
                print(f"[REGISTER] {result.message}")
                ui.toast("REGISTRATION COMPLETE" if result.success else "REGISTRATION FAILED", 2.0)
                if result.success:
                    matcher.reload(database.load_profiles())
                    db_counts = registered_counts(database)
                policy.reset()
                observations = []
                policy_status = PolicyStatus("EMPTY", 0.0, 0.0, False, False, False)
                spoof_start_time = None
                spoof_elapsed = 0.0
                spoof_triggered = False

            frame_index += 1
    finally:
        cap.release()
        window.destroy()
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())