from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    # Model paths
    yunet_model: Path = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
    sface_model: Path = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
    minifas_model: Path = BASE_DIR / "models" / "2.7_80x80_MiniFASNetV2.pth"

    # Persistent data
    database_path: Path = BASE_DIR / "database" / "face_database.sqlite3"
    event_dir: Path = BASE_DIR / "security_events"

    # Registration policy
    max_owners: int = 5
    max_guests: int = 5
    registration_samples: int = 15

    # Recognition / liveness thresholds. Calibrate with the actual webcam.
    face_detection_threshold: float = 0.85
    face_recognition_threshold: float = 0.45
    live_threshold: float = 0.60

    # Security policy
    unknown_capture_seconds: float = 10.0
    unknown_reset_seconds: float = 20.0
    unknown_alarm_seconds: float = 30.0
    spoof_event_seconds: float = 3.0
    max_faces: int = 6
    seen_update_interval_seconds: float = 5.0


CONFIG = AppConfig()
