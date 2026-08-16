from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class YuNetFaceDetector:
    """OpenCV YuNet wrapper.

    Each returned row has this layout:
    x, y, w, h, five facial landmarks (10 values), confidence.
    """

    def __init__(
        self,
        model_path: Path | str,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YuNet model not found: {path}")

        self._detector = cv2.FaceDetectorYN.create(
            model=str(path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self._last_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray, max_faces: int = 6) -> np.ndarray:
        height, width = frame.shape[:2]
        input_size = (width, height)
        if input_size != self._last_size:
            self._detector.setInputSize(input_size)
            self._last_size = input_size

        _, faces = self._detector.detect(frame)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)

        # Higher-confidence faces first.
        faces = faces[np.argsort(faces[:, -1])[::-1]]
        return faces[:max_faces]

    @staticmethod
    def bbox_xywh(face: np.ndarray) -> tuple[int, int, int, int]:
        x, y, w, h = face[:4]
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def crop_face(frame: np.ndarray, face: np.ndarray, margin: float = 0.15) -> np.ndarray:
        x, y, w, h = YuNetFaceDetector.bbox_xywh(face)
        frame_h, frame_w = frame.shape[:2]
        mx, my = int(w * margin), int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(frame_w, x + w + mx)
        y2 = min(frame_h, y + h + my)
        return frame[y1:y2, x1:x2].copy()
