from __future__ import annotations

import argparse

import cv2

from anti_spoof import MiniFASLiveness
from config import CONFIG
from face_database import FaceDatabase
from face_detector import YuNetFaceDetector
from face_recognizer import SFaceEmbedder
from registration import register_from_camera


def main() -> int:
    parser = argparse.ArgumentParser(description="Register one OWNER or GUEST")
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True, choices=["OWNER", "GUEST", "owner", "guest"])
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--samples", type=int, default=CONFIG.registration_samples)
    parser.add_argument("--live-threshold", type=float, default=CONFIG.live_threshold)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    detector = YuNetFaceDetector(CONFIG.yunet_model, CONFIG.face_detection_threshold)
    embedder = SFaceEmbedder(CONFIG.sface_model)
    liveness = MiniFASLiveness(CONFIG.minifas_model, args.live_threshold, args.device)
    database = FaceDatabase(CONFIG.database_path, CONFIG.max_owners, CONFIG.max_guests)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    try:
        result = register_from_camera(
            cap,
            args.name,
            args.role.upper(),
            detector,
            liveness,
            embedder,
            database,
            CONFIG.face_recognition_threshold,
            args.samples,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
