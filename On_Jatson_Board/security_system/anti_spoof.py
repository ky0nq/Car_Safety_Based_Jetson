from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from minifasnet import MiniFASNetV2


@dataclass(frozen=True)
class LivenessResult:
    is_live: bool
    label: str
    score: float
    probabilities: np.ndarray


def _expanded_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], scale: float = 2.7) -> np.ndarray:
    """Crop a scale-expanded face patch, matching the official MiniFAS preprocessing."""
    x, y, box_w, box_h = bbox
    src_h, src_w = frame.shape[:2]
    if box_w <= 0 or box_h <= 0:
        raise ValueError("Invalid face bounding box")

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x + box_w / 2.0, y + box_h / 2.0

    x1, y1 = cx - new_w / 2.0, cy - new_h / 2.0
    x2, y2 = cx + new_w / 2.0, cy + new_h / 2.0

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > src_w - 1:
        x1 -= x2 - src_w + 1
        x2 = src_w - 1
    if y2 > src_h - 1:
        y1 -= y2 - src_h + 1
        y2 = src_h - 1

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(src_w - 1, int(x2)), min(src_h - 1, int(y2))
    patch = frame[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        raise ValueError("MiniFAS crop is empty")
    return cv2.resize(patch, (80, 80), interpolation=cv2.INTER_LINEAR)


class MiniFASLiveness:
    """Single-model MiniFASNetV2 inference using the official pretrained .pth weight.

    The official three-class weight uses class index 1 for a real/live face.
    """

    def __init__(self, model_path: Path | str, threshold: float = 0.60, device: str | None = None) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"MiniFASNetV2 model not found: {path}")

        self.threshold = threshold
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        # 80x80 input produces a 5x5 final spatial map, hence conv6 kernel 5x5.
        self.model = MiniFASNetV2(conv6_kernel=(5, 5)).to(self.device)

        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # Older PyTorch
            state_dict = torch.load(path, map_location=self.device)

        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> LivenessResult:
        patch = _expanded_crop(frame, bbox, scale=2.7)
        # The official implementation keeps OpenCV BGR order and raw 0..255 float values.
        tensor = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
        tensor_t = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = F.softmax(self.model(tensor_t), dim=1)[0].detach().cpu().numpy()

        predicted_class = int(np.argmax(probabilities))
        real_score = float(probabilities[1])
        is_live = predicted_class == 1 and real_score >= self.threshold
        return LivenessResult(
            is_live=is_live,
            label="LIVE" if is_live else "SPOOF",
            score=real_score,
            probabilities=probabilities,
        )
