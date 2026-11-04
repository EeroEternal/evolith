from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]


def write_json(path: Path, value: dict) -> None:
    """Atomically publish a complete report; never serialize NaN as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, report TEXT NOT NULL)"
            )

    def record(self, run_id: str, kind: str, report: dict) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?)",
                (run_id, kind, json.dumps(report, allow_nan=False)),
            )

    def list(self) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            return [
                {"run_id": r[0], "kind": r[1], "report": json.loads(r[2])}
                for r in conn.execute("SELECT run_id, kind, report FROM runs ORDER BY rowid")
            ]

    def recent(self, kind: str, limit: int = 32) -> list[dict]:
        if not 1 <= limit <= 256:
            raise ValueError("History limit must be between 1 and 256.")
        with sqlite3.connect(self.path) as conn:
            return [json.loads(row[0]) for row in conn.execute(
                "SELECT report FROM runs WHERE kind = ? ORDER BY rowid DESC LIMIT ?",
                (kind, limit),
            )]
