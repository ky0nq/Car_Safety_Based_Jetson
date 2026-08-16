from __future__ import annotations

import sys

import cv2
import numpy as np
import torch

from anti_spoof import MiniFASLiveness
from config import CONFIG
from face_detector import YuNetFaceDetector
from face_recognizer import SFaceEmbedder


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    missing = [path for path in (CONFIG.yunet_model, CONFIG.sface_model, CONFIG.minifas_model) if not path.exists()]
    if missing:
        for path in missing:
            print(f"MISSING: {path}")
        print("Run: python3 download_models.py")
        return 1

    detector = YuNetFaceDetector(CONFIG.yunet_model)
    _ = SFaceEmbedder(CONFIG.sface_model)
    liveness = MiniFASLiveness(CONFIG.minifas_model, device="cpu")

    # MiniFAS smoke test with a synthetic frame and a centered bbox.
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = liveness.predict(frame, (110, 70, 100, 100))
    print(f"MiniFAS smoke output: {result.probabilities.tolist()}")
    print("Model loading succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
