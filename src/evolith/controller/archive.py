from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from evolith.models import EvolutionPlane, LineageNode


class SteppingStonesArchive:
    """
    Darwin Gödel Machine (DGM) / MAP-Elites Stepping-Stones Archive + AIDE^2 16x Compressed Memory.
    Persists all evolved nodes in SQLite, maintains niche elites, and computes the 3D Pareto Front.
    """

    def __init__(self, db_path: Path | str = Path(".evolith_archive.db")) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage_nodes (
                node_id TEXT PRIMARY KEY,
                parent_id TEXT,
                generation INTEGER,
                plane TEXT,
                target_name TEXT,
                niche_key TEXT,
                source_code TEXT,
                loc INTEGER,
                score REAL,
                speedup REAL,
                pareto_goodput REAL,
                pareto_p99_ttft_ms REAL,
                pareto_p99_tpot_ms REAL,
                accepted_by_gate INTEGER,
                is_stepping_stone INTEGER,
                hypothesis TEXT,
                patch_summary TEXT,
                metric_delta TEXT,
                failure_root_cause TEXT,
                metadata_json TEXT
            )
            """
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM lineage_nodes")
        self._conn.commit()

    def record_node(self, node: LineageNode) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO lineage_nodes VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                node.node_id,
                node.parent_id,
                node.generation,
                node.plane.value,
                node.target_name,
                node.niche_key,
                node.source_code,
                node.loc,
                node.score,
                node.speedup,
                node.pareto_goodput,
                node.pareto_p99_ttft_ms,
                node.pareto_p99_tpot_ms,
                1 if node.accepted_by_gate else 0,
                1 if node.is_stepping_stone else 0,
                node.hypothesis,
                node.patch_summary,
                node.metric_delta,
                node.failure_root_cause,
                json.dumps(node.metadata),
            ),
        )
        self._conn.commit()

    def list_nodes(
        self,
        plane: EvolutionPlane | None = None,
        target_name: str | None = None,
    ) -> list[LineageNode]:
        query = "SELECT * FROM lineage_nodes WHERE 1=1"
        params: list[object] = []
        if plane is not None:
            query += " AND plane = ?"
            params.append(plane.value)
        if target_name is not None:
            query += " AND target_name = ?"
            params.append(target_name)
        query += " ORDER BY generation ASC, node_id ASC"

        rows = self._conn.execute(query, params).fetchall()

        nodes: list[LineageNode] = []
        for r in rows:
            nodes.append(
                LineageNode(
                    node_id=r[0],
                    parent_id=r[1],
                    generation=int(r[2]),
                    plane=EvolutionPlane(r[3]),
                    target_name=r[4],
                    niche_key=r[5],
                    source_code=r[6],
                    loc=int(r[7]),
                    score=float(r[8]),
                    speedup=float(r[9]),
                    pareto_goodput=float(r[10]),
                    pareto_p99_ttft_ms=float(r[11]),
                    pareto_p99_tpot_ms=float(r[12]),
                    accepted_by_gate=bool(r[13]),
                    is_stepping_stone=bool(r[14]),
                    hypothesis=r[15],
                    patch_summary=r[16],
                    metric_delta=r[17],
                    failure_root_cause=r[18],
                    metadata=json.loads(r[19] or "{}"),
                )
            )
        return nodes

    def map_elites_grid(
        self, plane: EvolutionPlane, target_name: str
    ) -> dict[str, LineageNode]:
        """Returns the highest-scoring valid node per structural niche (MAP-Elites / DGM)."""
        grid: dict[str, LineageNode] = {}
        for node in self.list_nodes(plane=plane, target_name=target_name):
            if not (node.accepted_by_gate or node.is_stepping_stone):
                continue
            prev = grid.get(node.niche_key)
            if prev is None or node.score > prev.score:
                grid[node.niche_key] = node
        return grid

    def pareto_front_l2(self) -> list[LineageNode]:
        """
        Computes the non-dominated 3D Pareto Front across [Goodput ↑, P99_TTFT ↓, P99_TPOT ↓]
        for Plane L2 Scheduler Policies (GEPA).
        """
        candidates = [
            n
            for n in self.list_nodes(plane=EvolutionPlane.L2_SCHEDULER)
            if n.accepted_by_gate or n.is_stepping_stone
        ]
        front: list[LineageNode] = []
        for a in candidates:
            dominated = False
            for b in candidates:
                if a.node_id == b.node_id:
                    continue
                better_or_eq = (
                    b.pareto_goodput >= a.pareto_goodput
                    and b.pareto_p99_ttft_ms <= a.pareto_p99_ttft_ms
                    and b.pareto_p99_tpot_ms <= a.pareto_p99_tpot_ms
                )
                strictly_better = (
                    b.pareto_goodput > a.pareto_goodput
                    or b.pareto_p99_ttft_ms < a.pareto_p99_ttft_ms
                    or b.pareto_p99_tpot_ms < a.pareto_p99_tpot_ms
                )
                if better_or_eq and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(a)
        return front

    def format_compressed_memory(
        self, plane: EvolutionPlane | None = None, limit: int = 12
    ) -> tuple[str, float]:
        """
        Formats historical nodes using AIDE^2 16x Prompt-Compression 4-tuples.
        Returns (compressed_text, compression_ratio).
        """
        nodes = self.list_nodes(plane=plane)[-limit:]
        if not nodes:
            return ("(No historical evolutions recorded yet.)", 1.0)
        raw_chars = sum(len(n.source_code) + 450 for n in nodes)
        lines = [n.to_compressed_4tuple() for n in nodes]
        compressed_text = "\n".join(lines)
        ratio = raw_chars / float(max(1, len(compressed_text)))
        return (compressed_text, round(ratio, 1))
