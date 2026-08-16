from __future__ import annotations

import argparse

from config import CONFIG
from face_database import FaceDatabase


def print_people(database: FaceDatabase) -> None:
    rows = database.list_people()
    if not rows:
        print("No registered people.")
        return
    print("person_id | role | name | last_seen | seen_count")
    for row in rows:
        print(
            f"{row['person_id']} | {row['role']} | {row['name']} | "
            f"{row['last_seen_at']} | {row['seen_count']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="List or delete registered people")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    delete_parser = sub.add_parser("delete")
    delete_parser.add_argument("identifier", help="person_id or exact name")
    args = parser.parse_args()

    database = FaceDatabase(CONFIG.database_path, CONFIG.max_owners, CONFIG.max_guests)
    if args.command == "list":
        print_people(database)
        return 0

    deleted = database.delete_person(args.identifier)
    print("Deleted." if deleted else "No matching person.")
    return 0 if deleted else 1


if __name__ == "__main__":
    raise SystemExit(main())
