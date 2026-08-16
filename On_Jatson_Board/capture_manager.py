from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np


class CaptureManager:
    def __init__(self, event_root: Path | str) -> None:
        self.event_root = Path(event_root)
        self.event_root.mkdir(parents=True, exist_ok=True)

    def save_unknown_event(
        self,
        frame: np.ndarray,
        unknown_crops: list[np.ndarray],
        unknown_duration: float,
    ) -> Path:
        now = datetime.now()
        event_dir = self.event_root / now.strftime("%Y%m%d") / now.strftime("event_%H%M%S_%f")
        event_dir.mkdir(parents=True, exist_ok=False)

        full_frame_path = event_dir / "full_frame.jpg"
        if not cv2.imwrite(str(full_frame_path), frame):
            raise IOError(f"Failed to save {full_frame_path}")

        saved_crops: list[str] = []
        for index, crop in enumerate(unknown_crops, start=1):
            if crop.size == 0:
                continue
            crop_name = f"unknown_{index:02d}.jpg"
            if cv2.imwrite(str(event_dir / crop_name), crop):
                saved_crops.append(crop_name)

        metadata = {
            "event_type": "UNKNOWN_ONLY",
            "captured_at": now.isoformat(timespec="milliseconds"),
            "unknown_duration_seconds": round(unknown_duration, 3),
            "unknown_count": len(saved_crops),
            "full_frame": full_frame_path.name,
            "face_crops": saved_crops,
        }
        (event_dir / "event.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return event_dir
    def save_spoof_event(
        self,
        frame: np.ndarray,
        spoof_crops: list[np.ndarray],
        spoof_duration: float,
    ) -> Path:
        """Save a confirmed anti-spoofing attack event."""
        now = datetime.now()
        event_dir = (
            self.event_root
            / now.strftime("%Y%m%d")
            / now.strftime("spoof_%H%M%S_%f")
        )
        event_dir.mkdir(parents=True, exist_ok=False)

        full_frame_path = event_dir / "full_frame.jpg"
        if not cv2.imwrite(str(full_frame_path), frame):
            raise IOError(f"Failed to save {full_frame_path}")

        saved_crops: list[str] = []
        for index, crop in enumerate(spoof_crops, start=1):
            if crop.size == 0:
                continue
            crop_name = f"spoof_{index:02d}.jpg"
            if cv2.imwrite(str(event_dir / crop_name), crop):
                saved_crops.append(crop_name)

        metadata = {
            "event_type": "SPOOF_ATTACK",
            "captured_at": now.isoformat(timespec="milliseconds"),
            "spoof_duration_seconds": round(spoof_duration, 3),
            "spoof_count": len(saved_crops),
            "full_frame": full_frame_path.name,
            "face_crops": saved_crops,
        }
        (event_dir / "event.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return event_dir

