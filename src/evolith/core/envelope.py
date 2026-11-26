"""First-class Envelope Failure Log (`envelope_gaps`).

Implements Evolith's governing principle (`AGENTS.md`):
'Keep capability boundaries explicit. An unsupported bottleneck is a capability
gap, not a successful optimization.'

Whenever a measured bottleneck or candidate patch hits a pinned wire contract,
hardware resource ceiling, or hot-reload boundary, it is recorded in an
append-only `EnvelopeFailureLog` rather than silently hacked around.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import sqlite3
import uuid

from pydantic import BaseModel, Field

from evolith.models import EvolutionPlane


class EnvelopeGapCategory(str, Enum):
    PINNED_CONTRACT_BOUNDARY = "pinned_contract_boundary"
    RESOURCE_BUDGET_CEILING = "resource_budget_ceiling"
    NATIVE_EXTENSION_RELOAD_LIMIT = "native_extension_reload_limit"
    TOPOLOGY_OR_COLLECTIVE_CHANGE = "topology_or_collective_change"
    UNSUPPORTED_BOTTLENECK_OP = "unsupported_bottleneck_op"
    FORMAL_MODEL_COVERAGE_GAP = "formal_model_coverage_gap"


class EnvelopeGap(BaseModel):
    """Append-only record of an optimization blocked by an explicit capability boundary."""

    gap_id: str = Field(default_factory=lambda: f"gap-{uuid.uuid4().hex[:12]}")
    candidate_id: str = ""
    target_engine: str = "zstep"
    plane: EvolutionPlane = EvolutionPlane.L1_KERNEL
    category: EnvelopeGapCategory
    blocked_by_boundary: str
    attempted_optimization: str
    observed_bottleneck_share: float | None = None
    required_rung: str = "restart_engine"
    files_involved: tuple[str, ...] = ()
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class EnvelopeFailureLog:
    """Append-only SQLite store for capability/envelope gaps."""

    def __init__(self, db_path: Path | str, *, conn: sqlite3.Connection | None = None) -> None:
        self._owns_conn = conn is None
        if conn is not None:
            self._conn = conn
        else:
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, timeout=30.0)
            self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS envelope_gaps (
                gap_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL DEFAULT '',
                target_engine TEXT NOT NULL DEFAULT 'zstep',
                plane TEXT NOT NULL DEFAULT 'l1_kernel',
                category TEXT NOT NULL,
                blocked_by_boundary TEXT NOT NULL,
                attempted_optimization TEXT NOT NULL,
                observed_bottleneck_share REAL,
                required_rung TEXT NOT NULL DEFAULT 'restart_engine',
                files_involved TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(self, gap: EnvelopeGap) -> EnvelopeGap:
        if gap.observed_bottleneck_share is not None and not (
            0.0 <= gap.observed_bottleneck_share <= 1.0
        ):
            raise ValueError("observed_bottleneck_share must lie in [0.0, 1.0]")
        self._conn.execute(
            """
            INSERT INTO envelope_gaps (
                gap_id, candidate_id, target_engine, plane, category,
                blocked_by_boundary, attempted_optimization, observed_bottleneck_share,
                required_rung, files_involved, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gap.gap_id,
                gap.candidate_id,
                gap.target_engine,
                gap.plane.value,
                gap.category.value,
                gap.blocked_by_boundary,
                gap.attempted_optimization,
                gap.observed_bottleneck_share,
                gap.required_rung,
                ",".join(gap.files_involved),
                gap.created_at,
            ),
        )
        self._conn.commit()
        return gap

    def list_gaps(
        self,
        *,
        target_engine: str | None = None,
        category: EnvelopeGapCategory | None = None,
    ) -> list[EnvelopeGap]:
        clauses: list[str] = []
        params: list[object] = []
        if target_engine is not None:
            clauses.append("target_engine = ?")
            params.append(target_engine)
        if category is not None:
            clauses.append("category = ?")
            params.append(category.value)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM envelope_gaps {where} ORDER BY rowid ASC",
            params,
        ).fetchall()
        out: list[EnvelopeGap] = []
        for row in rows:
            files_raw = str(row["files_involved"] or "")
            files = tuple(f for f in files_raw.split(",") if f)
            out.append(
                EnvelopeGap(
                    gap_id=str(row["gap_id"]),
                    candidate_id=str(row["candidate_id"]),
                    target_engine=str(row["target_engine"]),
                    plane=EvolutionPlane(str(row["plane"])),
                    category=EnvelopeGapCategory(str(row["category"])),
                    blocked_by_boundary=str(row["blocked_by_boundary"]),
                    attempted_optimization=str(row["attempted_optimization"]),
                    observed_bottleneck_share=(
                        float(row["observed_bottleneck_share"])
                        if row["observed_bottleneck_share"] is not None
                        else None
                    ),
                    required_rung=str(row["required_rung"]),
                    files_involved=files,
                    created_at=str(row["created_at"]),
                )
            )
        return out

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()
