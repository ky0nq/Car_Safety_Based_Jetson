from __future__ import annotations

import time
from typing import Iterable

import cv2
import numpy as np

from anti_spoof import MiniFASLiveness
from face_database import FaceDatabase, RegistrationResult
from face_detector import YuNetFaceDetector
from face_recognizer import FaceMatcher, SFaceEmbedder, normalize_embedding


def _safe_name_for_window(name: str) -> str:
    ascii_name = name.encode("ascii", errors="ignore").decode().strip()
    return ascii_name or "PERSON"


def register_from_camera(
    cap: cv2.VideoCapture,
    name: str,
    role: str,
    detector: YuNetFaceDetector,
    liveness: MiniFASLiveness,
    embedder: SFaceEmbedder,
    database: FaceDatabase,
    recognition_threshold: float,
    sample_count: int = 15,
    visible_person_ids: set[str] | None = None,
) -> RegistrationResult:
    """Collect embeddings from the already-open camera.

    SPACE saves one sample. Q cancels. Only a single live face is accepted.
    """
    embeddings: list[np.ndarray] = []
    last_capture_time = 0.0
    window = "Register Face"
    display_name = _safe_name_for_window(name)

    print(f"[REGISTER] {name} / {role}: move your head and press SPACE {sample_count} times. Q cancels.")

    while len(embeddings) < sample_count:
        ok, frame = cap.read()
        if not ok:
            cv2.destroyWindow(window)
            return RegistrationResult(False, "Camera frame could not be read.")

        faces = detector.detect(frame, max_faces=2)
        status = "Show exactly one face"
        valid_face = None
        live_result = None

        if len(faces) == 1:
            valid_face = faces[0]
            bbox = detector.bbox_xywh(valid_face)
            try:
                live_result = liveness.predict(frame, bbox)
                status = f"{live_result.label} {live_result.score:.2f}"
            except Exception as exc:
                status = f"Anti-spoof error: {type(exc).__name__}"

            x, y, w, h = bbox
            color = (0, 200, 0) if live_result and live_result.is_live else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        elif len(faces) > 1:
            status = "Multiple faces: registration blocked"

        cv2.putText(frame, f"{role}: {display_name}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(embeddings)}/{sample_count}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE: capture   Q: cancel", (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            return RegistrationResult(False, "Registration cancelled.")

        if key == ord(" ") and valid_face is not None and live_result and live_result.is_live:
            if time.monotonic() - last_capture_time < 0.25:
                continue
            try:
                embeddings.append(embedder.extract(frame, valid_face))
                last_capture_time = time.monotonic()
                print(f"[REGISTER] captured {len(embeddings)}/{sample_count}")
            except Exception as exc:
                print(f"[REGISTER] embedding failed: {exc}")

    cv2.destroyWindow(window)

    # Prevent the same face from being registered under multiple names.
    existing_profiles = database.load_profiles()
    if existing_profiles:
        mean_embedding = normalize_embedding(np.mean(np.stack(embeddings), axis=0))
        duplicate = FaceMatcher(existing_profiles, threshold=recognition_threshold).match(mean_embedding)
        if duplicate.person_id is not None:
            return RegistrationResult(
                False,
                f"This face already matches '{duplicate.name}' ({duplicate.role}, score={duplicate.score:.3f}).",
            )

    return database.add_person(
        name=name,
        role=role,
        embeddings=embeddings,
        visible_person_ids=visible_person_ids or set(),
    )
