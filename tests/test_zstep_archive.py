from pathlib import Path
import sqlite3

import pytest

from evolith.models import EvolutionPlane
from evolith.zstep.archive import ZStepArchive
from evolith.zstep.task import CandidateStatus


def test_archive_restores_accepted_patch_stack_and_syncs_stepping_stones(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite3"
    archive = ZStepArchive(db_path)
    archive.start_candidate(
        "g0001-a",
        1,
        None,
        b"patch-a",
        "sha-a",
        plane=EvolutionPlane.L1_KERNEL,
        niche_key="(plane=l1_kernel,focus=moe_epilogue)",
        changed_lines=24,
        changed_files=("worker/zstep_worker/kernels_sm120/b12x_dsv41.py",),
        hypothesis="Fuse MoE epilogue scale+add to cut SM120 global memory traffic",
        patch_summary="Fused epilogue in b12x_dsv41.py (24 LOC)",
    )
    archive.accept_candidate(
        "g0001-a",
        "evidence/g0001-a",
        0.042,
        net_gate_ratio=1.042,
        metric_delta="geomean=+4.20% [in512_c16:+3.8%, in512_c128:+4.6%]",
    )
    archive.start_candidate(
        "g0002-b",
        2,
        "g0001-a",
        b"patch-b",
        "sha-b",
        changed_lines=15,
        changed_files=("worker/zstep_worker/models/deepseek_v41.py",),
        hypothesis="Pack host decode metadata into single H2D copy",
        patch_summary="Packed decode I/O in deepseek_v41.py (15 LOC)",
    )
    archive.finish_candidate(
        "g0002-b",
        CandidateStatus.VERIFY_REJECTED,
        "evidence/g0002-b",
        "geomean throughput gain -0.08% below required minimum",
        geomean_gain=-0.0008,
        metric_delta="geomean=-0.08%",
    )

    assert archive.accepted_patches() == [b"patch-a"]
    assert archive.current_parent() == "g0001-a"
    assert archive.next_generation() == 3

    memory = archive.compressed_lineage_memory()
    assert "Fuse MoE epilogue scale+add" in memory
    assert "Pack host decode metadata" in memory
    assert "geomean=+4.20%" in memory

    nodes = archive.stepping_stones.list_nodes(target_name="zstep_deepseek_v41")
    assert len(nodes) == 2
    assert nodes[0].accepted_by_gate is True
    assert nodes[1].accepted_by_gate is False
    archive.close()

    restored = ZStepArchive(db_path)
    assert restored.accepted_patches() == [b"patch-a"]
    assert restored.evidence_dir("g0002-b") == "evidence/g0002-b"
    restored.close()


def test_archive_migrates_legacy_schema_seamlessly(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE candidates (
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
    conn.execute(
        "INSERT INTO candidates (candidate_id, generation, patch, source_hash, status) VALUES ('g0001-old', 1, ?, 'sha', 'verify_rejected')",
        (b"diff",),
    )
    conn.commit()
    conn.close()

    with ZStepArchive(db_path) as archive:
        entries = archive.list_candidates()
        assert len(entries) == 1
        assert entries[0]["candidate_id"] == "g0001-old"
        assert "g0001-old" in archive.compressed_lineage_memory()
        with pytest.raises(KeyError):
            archive.evidence_dir("missing")


def test_blocked_niches_tracks_two_strikes_and_ignores_infra_failures(
    tmp_path: Path,
) -> None:
    with ZStepArchive(tmp_path / "strikes.sqlite3") as archive:
        # Infra failure (GPU busy) does not count as a premise strike
        archive.start_candidate(
            "g0001", 1, None, b"diff-1", "s1", niche_key="(plane=l2_scheduler,focus=h2d_pack)"
        )
        archive.finish_candidate("g0001", CandidateStatus.FAILED, "ev1", "GPU 0 occupied")
        assert archive.blocked_niches(strike_limit=2, parent_id=None) == {}

        # Strike 1 + Strike 2 on the same niche under parent=None
        for gen, cid in ((2, "g0002"), (3, "g0003")):
            archive.start_candidate(
                cid, gen, None, b"diff", f"s{gen}", niche_key="(plane=l2_scheduler,focus=h2d_pack)"
            )
            archive.finish_candidate(
                cid, CandidateStatus.VERIFY_REJECTED, f"ev{gen}", "gain +0.1% below min"
            )

        blocked = archive.blocked_niches(strike_limit=2, parent_id=None)
        assert "(plane=l2_scheduler,focus=h2d_pack)" in blocked
        assert len(blocked["(plane=l2_scheduler,focus=h2d_pack)"]) == 2
        # Advancing to a new accepted parent starts with a clean slate
        assert archive.blocked_niches(strike_limit=2, parent_id="g0004-accepted") == {}


def test_summarize_failure_reason_preserves_diagnostic_over_json_and_teardown_noise(
    tmp_path: Path,
) -> None:
    filler = "x" * 290
    noisy_log = (
        "Remote benchmark exited 1:\n"
        f"  {filler}\n"
        "Error: exact_token_match_rate 0.8125 < 1.0 on prime_squares_table\n"
        f"  \"verbose_tail\": \"{filler}\"\n"
        "  ]\n"
        "}\n"
        "TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5\n"
    )
    with ZStepArchive(tmp_path / "diag.sqlite3") as archive:
        archive.start_candidate("g0006", 6, None, b"diff", "s6")
        archive.finish_candidate("g0006", CandidateStatus.SCREEN_REJECTED, "ev6", noisy_log)
        row = archive.get_candidate("g0006")
        reason = str(row["failure_reason"])
        assert "exact_token_match_rate 0.8125 < 1.0" in reason
        assert "TRITON_RECOMPILATIONS=0" not in reason


def test_archive_redacts_hidden_verify_tier_labels_and_records_events(tmp_path: Path) -> None:
    with ZStepArchive(tmp_path / "events.sqlite3") as archive:
        archive.start_candidate("g0007", 7, None, b"diff", "s7", hypothesis="Split PXB ring")
        archive.update_stage("g0007", "verify")
        archive.finish_candidate(
            "g0007",
            CandidateStatus.VERIFY_REJECTED,
            "ev7",
            "in1024_c128 p99 ITL regressed +7.2%",
            metric_delta="geomean=+1.2% [in1024_c128:-4.1%, in512_c16:+2.0%]",
        )
        redacted = archive.compressed_lineage_memory(
            hidden_tier_labels=("in1024_c128", "in512_c16")
        )
        assert "in1024_c128" not in redacted
        assert "in512_c16" not in redacted
        assert "verify_tier" in redacted
        assert "p99 ITL regressed" in redacted

        events = archive.list_events("g0007")
        assert [e["event_type"] for e in events] == [
            "start_candidate",
            "update_stage",
            "finish_candidate",
        ]
        assert len(archive.list_events()) == 3
