from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from face_recognizer import normalize_embedding


VALID_ROLES = {"OWNER", "GUEST"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PersonProfile:
    person_id: str
    name: str
    role: str
    created_at: str
    last_seen_at: str
    seen_count: int
    embeddings: list[np.ndarray]


@dataclass(frozen=True)
class RegistrationResult:
    success: bool
    message: str
    person_id: str | None = None
    removed_guest_name: str | None = None


class FaceDatabase:
    def __init__(self, path: Path | str, max_owners: int = 5, max_guests: int = 5) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_owners = max_owners
        self.max_guests = max_guests
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL CHECK(role IN ('OWNER', 'GUEST')),
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                );
                """
            )

    def count_role(self, role: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM persons WHERE role = ?", (role,)).fetchone()
        return int(row["count"])

    def add_person(
        self,
        name: str,
        role: str,
        embeddings: Iterable[np.ndarray],
        visible_person_ids: set[str] | None = None,
    ) -> RegistrationResult:
        name = name.strip()
        role = role.upper().strip()
        visible_person_ids = visible_person_ids or set()

        if not name:
            return RegistrationResult(False, "Name cannot be empty.")
        if role not in VALID_ROLES:
            return RegistrationResult(False, f"Invalid role: {role}")

        vectors = [normalize_embedding(vector) for vector in embeddings]
        if not vectors:
            return RegistrationResult(False, "No valid face embeddings were collected.")

        removed_guest_name = None
        with self._connect() as conn:
            duplicate = conn.execute("SELECT 1 FROM persons WHERE name = ?", (name,)).fetchone()
            if duplicate:
                return RegistrationResult(False, f"'{name}' is already registered.")

            role_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM persons WHERE role = ?", (role,)).fetchone()["count"]
            )

            if role == "OWNER" and role_count >= self.max_owners:
                return RegistrationResult(False, f"OWNER limit reached ({self.max_owners}). Delete one manually first.")

            if role == "GUEST" and role_count >= self.max_guests:
                guests = conn.execute(
                    "SELECT person_id, name FROM persons WHERE role = 'GUEST' ORDER BY last_seen_at ASC, created_at ASC"
                ).fetchall()
                removable = next((row for row in guests if row["person_id"] not in visible_person_ids), None)
                if removable is None:
                    return RegistrationResult(False, "All registered guests are currently visible; registration cancelled.")
                removed_guest_name = str(removable["name"])
                conn.execute("DELETE FROM persons WHERE person_id = ?", (removable["person_id"],))

            person_id = f"{role.lower()}_{uuid.uuid4().hex[:10]}"
            now = utc_now_iso()
            conn.execute(
                "INSERT INTO persons(person_id, name, role, created_at, last_seen_at, seen_count) VALUES (?, ?, ?, ?, ?, 0)",
                (person_id, name, role, now, now),
            )
            conn.executemany(
                "INSERT INTO embeddings(person_id, vector, dimension) VALUES (?, ?, ?)",
                [(person_id, vector.astype(np.float32).tobytes(), int(vector.size)) for vector in vectors],
            )

        message = f"Registered {name} as {role}."
        if removed_guest_name:
            message = f"Removed least-recently-seen guest '{removed_guest_name}', then {message}"
        return RegistrationResult(True, message, person_id, removed_guest_name)

    def load_profiles(self) -> list[PersonProfile]:
        profiles: list[PersonProfile] = []
        with self._connect() as conn:
            persons = conn.execute(
                "SELECT person_id, name, role, created_at, last_seen_at, seen_count FROM persons ORDER BY role, name"
            ).fetchall()
            for person in persons:
                rows = conn.execute(
                    "SELECT vector, dimension FROM embeddings WHERE person_id = ? ORDER BY embedding_id",
                    (person["person_id"],),
                ).fetchall()
                vectors = [np.frombuffer(row["vector"], dtype=np.float32, count=row["dimension"]).copy() for row in rows]
                profiles.append(
                    PersonProfile(
                        person_id=str(person["person_id"]),
                        name=str(person["name"]),
                        role=str(person["role"]),
                        created_at=str(person["created_at"]),
                        last_seen_at=str(person["last_seen_at"]),
                        seen_count=int(person["seen_count"]),
                        embeddings=vectors,
                    )
                )
        return profiles

    def update_seen_many(self, person_ids: Iterable[str]) -> None:
        ids = sorted(set(person_ids))
        if not ids:
            return
        now = utc_now_iso()
        with self._connect() as conn:
            conn.executemany(
                "UPDATE persons SET last_seen_at = ?, seen_count = seen_count + 1 WHERE person_id = ?",
                [(now, person_id) for person_id in ids],
            )

    def clear_role(self, role: str) -> int:
        """Delete all registered people for one role. Embeddings cascade automatically."""
        role = role.upper().strip()
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM persons WHERE role = ?", (role,))
        return int(cursor.rowcount)

    def delete_person(self, identifier: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM persons WHERE person_id = ? OR name = ?",
                (identifier, identifier),
            )
        return cursor.rowcount > 0

    def list_people(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT person_id, name, role, created_at, last_seen_at, seen_count FROM persons ORDER BY role, name"
            ).fetchall()
