from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from evolith.controller.archive import SteppingStonesArchive
from evolith.models import EvolutionPlane, LineageNode
from evolith.zstep.task import CandidateStatus

_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("plane", "TEXT NOT NULL DEFAULT 'l2_scheduler'"),
    ("niche_key", "TEXT NOT NULL DEFAULT '(plane=auto,target=deepseek_v41)'"),
    ("changed_lines", "INTEGER NOT NULL DEFAULT 0"),
    ("changed_files", "TEXT NOT NULL DEFAULT '[]'"),
    ("hypothesis", "TEXT NOT NULL DEFAULT ''"),
    ("patch_summary", "TEXT NOT NULL DEFAULT ''"),
    ("metric_delta", "TEXT NOT NULL DEFAULT ''"),
    ("screen_gain", "REAL NOT NULL DEFAULT 0"),
    ("net_gate_ratio", "REAL NOT NULL DEFAULT 1"),
    ("stage", "TEXT NOT NULL DEFAULT 'propose'"),
    ("baseline_hash", "TEXT NOT NULL DEFAULT ''"),
    ("upstream_sha", "TEXT NOT NULL DEFAULT ''"),
)


_NOISE_TAIL_TOKENS: tuple[str, ...] = (
    "{",
    "}",
    "[",
    "]",
    "},",
    "],",
)
_EXCEPTION_KEYWORDS: tuple[str, ...] = (
    "traceback (most recent call last)",
    "runtimeerror:",
    "typeerror:",
    "valueerror:",
    "assertionerror",
    "indexerror:",
    "keyerror:",
    "attributeerror:",
    "cuda error",
    "panicked at",
    "error[e",
    "error:",
)
_ROOT_CAUSE_KEYWORDS: tuple[str, ...] = (
    "exact_token_match_rate",
    "violated cross-mode invariance",
    "logit drift",
    "mismatch",
    "regressed",
    "leak",
    "failed",
)


def _is_noise_tail_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped in _NOISE_TAIL_TOKENS:
        return True
    if stripped.startswith("TRITON_RECOMPILATIONS=0"):
        return True
    lower = stripped.lower()
    if re.match(r'^"[^"\n]+"\s*:', stripped) and not any(
        kw in lower for kw in _EXCEPTION_KEYWORDS
    ):
        return True
    return False


def _summarize_failure_reason(raw: str, max_chars: int = 360) -> str:
    """
    Distills multi-kilobyte remote compiler/progress logs into a concise single-line
    root cause preserving diagnostic exception/assertion lines instead of trailing
    JSON braces or `TRITON_RECOMPILATIONS=0` teardown banners (full log stays in `failure.json`).
    """
    text = raw.strip()
    if not text:
        return "None"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 2 and len(text) <= max_chars:
        return " | ".join(lines)
    head = (
        lines[0].split(":", 1)[0].strip()
        if lines[0].startswith("Remote benchmark")
        else lines[0][:100]
    )
    body = lines[1:]
    diagnostic_hits: list[str] = []
    for kw_group in (_EXCEPTION_KEYWORDS, _ROOT_CAUSE_KEYWORDS):
        for line in reversed(body):
            if _is_noise_tail_line(line):
                continue
            lower = line.lower()
            if any(kw in lower for kw in kw_group) and line not in diagnostic_hits:
                diagnostic_hits.append(line)
            if len(diagnostic_hits) >= 2:
                break
        if diagnostic_hits:
            break

    if diagnostic_hits:
        tail = " ; ".join(reversed(diagnostic_hits))
    else:
        fallback_candidates: list[str] = []
        for line in reversed(body):
            if _is_noise_tail_line(line):
                continue
            if line not in fallback_candidates:
                fallback_candidates.append(line)
            if len(fallback_candidates) >= 3:
                break
        if fallback_candidates:
            tail = " ; ".join(reversed(fallback_candidates))
        else:
            tail = " ; ".join(body[-2:])
    combined = f"{head}: {tail}" if tail and tail != head else head
    return combined[:max_chars]


class ZStepArchive:
    """
    Persistent zstep candidate archive integrated directly with Evolith's
    SteppingStonesArchive (DGM / MAP-Elites + AIDE^2 16x 4-Tuple Lineage Memory).
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self.stepping_stones = SteppingStonesArchive(db_path=db_path)
        self.stepping_stones._conn.close()
        self.stepping_stones._conn = self._conn

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                parent_id TEXT,
                patch BLOB NOT NULL,
                source_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                evidence_dir TEXT NOT NULL DEFAULT '',
                failure_reason TEXT NOT NULL DEFAULT '',
                geomean_gain REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        for col_name, col_ddl in _EXTRA_COLUMNS:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE candidates ADD COLUMN {col_name} {col_ddl}"
                )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def _log_event(
        self,
        candidate_id: str,
        event_type: str,
        *,
        stage: str = "",
        status: str = "",
        detail: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO candidate_events (candidate_id, event_type, stage, status, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            (candidate_id, event_type, stage, status, detail[:500]),
        )

    def list_events(self, candidate_id: str | None = None) -> list[dict[str, object]]:
        if candidate_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM candidate_events WHERE candidate_id = ? ORDER BY event_id ASC",
                (candidate_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM candidate_events ORDER BY event_id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ZStepArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_attempt(
        self,
        candidate_id: str,
        generation: int,
        parent_id: str | None,
        evidence_dir: str = "",
    ) -> None:
        """Records a candidate slot immediately before running the proposer so failed proposals never leave unindexed ghost directories."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO candidates (
                candidate_id, generation, parent_id, patch, source_hash, status, evidence_dir, stage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'propose')
            """,
            (
                candidate_id,
                generation,
                parent_id,
                b"",
                "",
                CandidateStatus.PROPOSED.value,
                evidence_dir,
            ),
        )
        self._log_event(
            candidate_id,
            "register_attempt",
            stage="propose",
            status=CandidateStatus.PROPOSED.value,
        )
        self._conn.commit()

    def start_candidate(
        self,
        candidate_id: str,
        generation: int,
        parent_id: str | None,
        patch: bytes,
        source_hash: str,
        *,
        plane: EvolutionPlane = EvolutionPlane.L2_SCHEDULER,
        niche_key: str = "(plane=auto,target=deepseek_v41)",
        changed_lines: int = 0,
        changed_files: tuple[str, ...] | list[str] = (),
        hypothesis: str = "",
        patch_summary: str = "",
        evidence_dir: str = "",
        baseline_hash: str = "",
        upstream_sha: str = "",
    ) -> None:
        files_json = json.dumps(list(changed_files))
        self._conn.execute(
            """
            INSERT INTO candidates (
                candidate_id, generation, parent_id, patch, source_hash, status,
                evidence_dir, plane, niche_key, changed_lines, changed_files,
                hypothesis, patch_summary, stage, baseline_hash, upstream_sha
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'local', ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                patch = excluded.patch,
                source_hash = excluded.source_hash,
                status = excluded.status,
                evidence_dir = CASE WHEN excluded.evidence_dir != '' THEN excluded.evidence_dir ELSE candidates.evidence_dir END,
                plane = excluded.plane,
                niche_key = excluded.niche_key,
                changed_lines = excluded.changed_lines,
                changed_files = excluded.changed_files,
                hypothesis = excluded.hypothesis,
                patch_summary = excluded.patch_summary,
                stage = 'local',
                baseline_hash = CASE WHEN excluded.baseline_hash != '' THEN excluded.baseline_hash ELSE candidates.baseline_hash END,
                upstream_sha = CASE WHEN excluded.upstream_sha != '' THEN excluded.upstream_sha ELSE candidates.upstream_sha END
            """,
            (
                candidate_id,
                generation,
                parent_id,
                patch,
                source_hash,
                CandidateStatus.PROPOSED.value,
                evidence_dir,
                plane.value,
                niche_key,
                changed_lines,
                files_json,
                hypothesis,
                patch_summary,
                baseline_hash,
                upstream_sha,
            ),
        )
        self._log_event(
            candidate_id,
            "start_candidate",
            stage="local",
            status=CandidateStatus.PROPOSED.value,
            detail=niche_key,
        )
        self._conn.commit()

    def update_stage(
        self,
        candidate_id: str,
        stage: str,
        *,
        screen_gain: float | None = None,
        metric_delta: str | None = None,
    ) -> None:
        fields = ["stage = ?"]
        params: list[object] = [stage]
        if screen_gain is not None:
            fields.append("screen_gain = ?")
            params.append(screen_gain)
        if metric_delta is not None:
            fields.append("metric_delta = ?")
            params.append(metric_delta)
        params.append(candidate_id)
        self._conn.execute(
            f"UPDATE candidates SET {', '.join(fields)} WHERE candidate_id = ?",
            params,
        )
        self._log_event(
            candidate_id,
            "update_stage",
            stage=stage,
            detail=metric_delta or "",
        )
        self._conn.commit()

    def _sync_stepping_stone(self, candidate_id: str) -> None:
        row = self._conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            return
        status = CandidateStatus(str(row["status"]))
        gain = float(row["geomean_gain"] or row["screen_gain"] or 0.0)
        speedup = max(0.0, 1.0 + gain) if status in (
            CandidateStatus.ACCEPTED,
            CandidateStatus.VERIFY_REJECTED,
            CandidateStatus.SCREEN_REJECTED,
        ) else 0.0
        plane_raw = str(row["plane"] or EvolutionPlane.L2_SCHEDULER.value)
        try:
            plane = EvolutionPlane(plane_raw)
        except ValueError:
            plane = EvolutionPlane.L2_SCHEDULER

        patch_bytes = bytes(row["patch"] or b"")
        patch_text = patch_bytes.decode("utf-8", errors="replace")[:4000]
        is_accepted = status == CandidateStatus.ACCEPTED
        is_stepping = (
            status == CandidateStatus.VERIFY_REJECTED and gain > 0.005
        )
        hypothesis = str(row["hypothesis"] or "").strip() or "Unannotated source mutation"
        patch_summary = str(row["patch_summary"] or "").strip()
        if not patch_summary:
            files = json.loads(str(row["changed_files"] or "[]"))
            patch_summary = (
                f"{', '.join(files)} ({int(row['changed_lines'] or 0)} LOC)"
                if files
                else f"status={status.value}"
            )
        metric_delta = str(row["metric_delta"] or "").strip() or (
            f"geomean={gain:+.2%}" if gain else status.value
        )
        failure_reason = _summarize_failure_reason(str(row["failure_reason"] or ""))

        node = LineageNode(
            node_id=str(row["candidate_id"]),
            parent_id=str(row["parent_id"]) if row["parent_id"] else None,
            generation=int(row["generation"]),
            plane=plane,
            target_name="zstep_deepseek_v41",
            niche_key=str(row["niche_key"] or "(plane=auto,target=deepseek_v41)"),
            source_code=patch_text or f"# {candidate_id}",
            loc=int(row["changed_lines"] or 0),
            score=float(row["net_gate_ratio"] or speedup),
            speedup=round(speedup, 4),
            accepted_by_gate=is_accepted,
            is_stepping_stone=is_stepping,
            hypothesis=hypothesis,
            patch_summary=patch_summary,
            metric_delta=metric_delta,
            failure_root_cause=failure_reason,
            metadata={
                "status": status.value,
                "geomean_gain": float(row["geomean_gain"] or 0.0),
                "screen_gain": float(row["screen_gain"] or 0.0),
                "evidence_dir": str(row["evidence_dir"] or ""),
            },
        )
        self.stepping_stones.record_node(node)

    def finish_candidate(
        self,
        candidate_id: str,
        status: CandidateStatus,
        evidence_dir: str,
        failure_reason: str,
        *,
        geomean_gain: float | None = None,
        net_gate_ratio: float | None = None,
        metric_delta: str | None = None,
    ) -> None:
        compact_reason = (
            _summarize_failure_reason(failure_reason) if failure_reason.strip() else ""
        )
        fields = ["status = ?", "evidence_dir = ?", "failure_reason = ?"]
        params: list[object] = [status.value, evidence_dir, compact_reason]
        if geomean_gain is not None:
            fields.append("geomean_gain = ?")
            params.append(geomean_gain)
        if net_gate_ratio is not None:
            fields.append("net_gate_ratio = ?")
            params.append(net_gate_ratio)
        if metric_delta is not None:
            fields.append("metric_delta = ?")
            params.append(metric_delta)
        params.append(candidate_id)
        self._conn.execute(
            f"UPDATE candidates SET {', '.join(fields)} WHERE candidate_id = ?",
            params,
        )
        self._log_event(
            candidate_id,
            "finish_candidate",
            status=status.value,
            detail=compact_reason,
        )
        self._conn.commit()
        self._sync_stepping_stone(candidate_id)

    def accept_candidate(
        self,
        candidate_id: str,
        evidence_dir: str,
        geomean_gain: float,
        *,
        net_gate_ratio: float = 1.0,
        metric_delta: str = "",
    ) -> None:
        self._conn.execute(
            """
            UPDATE candidates
            SET status = ?, evidence_dir = ?, geomean_gain = ?, net_gate_ratio = ?,
                metric_delta = CASE WHEN ? != '' THEN ? ELSE metric_delta END,
                failure_reason = '', stage = 'done'
            WHERE candidate_id = ?
            """,
            (
                CandidateStatus.ACCEPTED.value,
                evidence_dir,
                geomean_gain,
                net_gate_ratio,
                metric_delta,
                metric_delta,
                candidate_id,
            ),
        )
        self._log_event(
            candidate_id,
            "accept_candidate",
            stage="done",
            status=CandidateStatus.ACCEPTED.value,
            detail=metric_delta,
        )
        self._conn.commit()
        self._sync_stepping_stone(candidate_id)

    def get_candidate(self, candidate_id: str) -> dict[str, object]:
        row = self._conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return dict(row)

    def accepted_patches(self, upstream_sha: str | None = None) -> list[bytes]:
        if upstream_sha:
            rows = self._conn.execute(
                """
                SELECT patch FROM candidates
                WHERE status = ? AND upstream_sha = ?
                ORDER BY generation ASC, candidate_id ASC
                """,
                (CandidateStatus.ACCEPTED.value, upstream_sha),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT patch FROM candidates
                WHERE status = ? ORDER BY generation ASC, candidate_id ASC
                """,
                (CandidateStatus.ACCEPTED.value,),
            ).fetchall()
        return [bytes(row["patch"]) for row in rows if bytes(row["patch"])]

    def current_parent(self, upstream_sha: str | None = None) -> str | None:
        if upstream_sha:
            row = self._conn.execute(
                """
                SELECT candidate_id FROM candidates
                WHERE status = ? AND upstream_sha = ?
                ORDER BY generation DESC, candidate_id DESC LIMIT 1
                """,
                (CandidateStatus.ACCEPTED.value, upstream_sha),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT candidate_id FROM candidates
                WHERE status = ? ORDER BY generation DESC, candidate_id DESC LIMIT 1
                """,
                (CandidateStatus.ACCEPTED.value,),
            ).fetchone()
        return None if row is None else str(row["candidate_id"])

    def next_generation(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(generation), 0) + 1 AS next_generation FROM candidates"
        ).fetchone()
        assert row is not None
        return int(row["next_generation"])

    def list_candidates(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT * FROM candidates ORDER BY generation ASC, candidate_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def evidence_dir(self, candidate_id: str) -> str:
        row = self._conn.execute(
            "SELECT evidence_dir FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return str(row["evidence_dir"])

    def compressed_lineage_memory(
        self,
        limit: int = 8,
        *,
        hidden_tier_labels: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """
        Formats historical zstep candidates as AIDE^2 16x compressed 4-tuples
        ([Hypothesis, PatchSummary, MetricDelta, FailureRootCause]).
        Falls back to candidate rows if any legacy rows were never synced into lineage_nodes.
        """
        nodes = [
            n
            for n in self.stepping_stones.list_nodes(target_name="zstep_deepseek_v41")
        ][-limit:]
        synced_ids = {n.node_id for n in nodes}
        lines = [n.to_compressed_4tuple() for n in nodes]
        for row in self.list_candidates()[-limit:]:
            cid = str(row["candidate_id"])
            if cid in synced_ids:
                continue
            status = str(row["status"])
            hyp = str(row.get("hypothesis") or "legacy candidate")
            summary = str(row.get("patch_summary") or "unannotated patch")
            delta = str(row.get("metric_delta") or f"gain={float(row['geomean_gain']):+.2%}")
            reason = str(row.get("failure_reason") or "None")
            lines.append(
                f"[{cid} gen={row['generation']} {status.upper()}] "
                f"Hyp: {hyp} | Patch: {summary} | Delta: {delta} | RootCause: {reason}"
            )
        rendered = "\n".join(lines[-limit:]) if lines else "None"
        if hidden_tier_labels and rendered != "None":
            pattern = re.compile(
                r"\b(?:" + "|".join(re.escape(lbl) for lbl in sorted(hidden_tier_labels)) + r")\b"
            )
            rendered = pattern.sub("verify_tier", rendered)
        return rendered

    def blocked_niches(
        self,
        strike_limit: int = 2,
        parent_id: str | None = None,
        exclude_candidate_id: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Implements vstack's 'Attack the Premise' 2-strike rule: tracks consecutive
        gate rejections per explicit `niche_key` under the active `parent_id`.
        Infrastructure failures (`failed`) and empty-patch proposer crashes do not
        count as premise strikes.
        """
        rows = self._conn.execute(
            """
            SELECT candidate_id, niche_key, status, failure_reason, patch
            FROM candidates
            WHERE parent_id IS ?
            ORDER BY generation ASC, candidate_id ASC
            """,
            (parent_id,),
        ).fetchall()
        streaks: dict[str, list[str]] = {}
        rejected_statuses = {
            CandidateStatus.LOCAL_REJECTED.value,
            CandidateStatus.SCREEN_REJECTED.value,
            CandidateStatus.VERIFY_REJECTED.value,
        }
        for row in rows:
            cid = str(row["candidate_id"])
            if exclude_candidate_id and cid == exclude_candidate_id:
                continue
            niche = str(row["niche_key"] or "").strip()
            if not niche or niche == "(plane=auto,target=deepseek_v41)":
                continue
            status = str(row["status"])
            if status == CandidateStatus.ACCEPTED.value:
                streaks[niche] = []
            elif status in rejected_statuses and bytes(row["patch"] or b""):
                reason = str(row["failure_reason"] or status).strip()
                streaks.setdefault(niche, []).append(reason)
        return {
            niche: reasons
            for niche, reasons in streaks.items()
            if len(reasons) >= strike_limit
        }
