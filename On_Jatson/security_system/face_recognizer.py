from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    person_id: str | None
    name: str
    role: str
    score: float


class SFaceEmbedder:
    def __init__(self, model_path: Path | str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"SFace model not found: {path}")
        self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def extract(self, frame: np.ndarray, yunet_face: np.ndarray) -> np.ndarray:
        aligned = self._recognizer.alignCrop(frame, yunet_face)
        feature = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-12:
            raise ValueError("SFace returned a zero-length embedding")
        return feature / norm


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length embedding")
    return vector / norm


class FaceMatcher:
    """Matches an SFace embedding against cached database profiles."""

    def __init__(self, profiles: Iterable[object], threshold: float = 0.45) -> None:
        self.threshold = threshold
        self.reload(profiles)

    def reload(self, profiles: Iterable[object]) -> None:
        self._profiles = list(profiles)

    def match(self, embedding: np.ndarray) -> MatchResult:
        query = normalize_embedding(embedding)
        best_profile = None
        best_score = -1.0

        for profile in self._profiles:
            matrix = np.asarray(profile.embeddings, dtype=np.float32)
            if matrix.size == 0:
                continue
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            scores = matrix @ query

            # Average the strongest few samples to reduce one-sample false matches.
            top_count = min(3, scores.size)
            score = float(np.mean(np.partition(scores, -top_count)[-top_count:]))
            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile is None or best_score < self.threshold:
            return MatchResult(None, "UNKNOWN", "UNKNOWN", max(best_score, 0.0))

        return MatchResult(
            person_id=best_profile.person_id,
            name=best_profile.name,
            role=best_profile.role,
            score=best_score,
        )
