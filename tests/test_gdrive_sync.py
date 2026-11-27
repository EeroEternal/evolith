from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from evolith.zstep.gdrive_sync import (
    GDriveBenchmarkSync,
    export_archive_summary_csv,
    export_evidence_ledger_json,
)


class _FakeGDriveBackend:
    """In-memory mock of `gdrive readonly ls` and `gdrive mutate mkdir/upload/update`."""

    def __init__(self, *, transient_failures: int = 0) -> None:
        self.transient_failures = transient_failures
        self.calls: list[list[str]] = []
        self.children: dict[str, list[dict[str, Any]]] = {"root123": []}
        self._counter = 0

    def __call__(self, args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        if self.transient_failures > 0:
            self.transient_failures -= 1
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="searching: googleapi: got HTTP response code 502 with body: Backend error: EOF",
            )
        ns, cmd = args[1], args[2]
        if ns == "readonly" and cmd == "ls":
            folder_id = args[3]
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=json.dumps(self.children.get(folder_id, [])),
                stderr="",
            )
        if ns == "mutate" and cmd == "mkdir":
            name = args[3]
            parent_id = args[args.index("--parent") + 1]
            self._counter += 1
            fid = f"folder_{self._counter}"
            self.children.setdefault(parent_id, []).append(
                {
                    "id": fid,
                    "name": name,
                    "mimeType": "application/vnd.google-apps.folder",
                }
            )
            self.children[fid] = []
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"Created {fid}", stderr="")
        if ns == "mutate" and cmd == "upload":
            fpath = Path(args[3])
            parent_id = args[args.index("--parent") + 1]
            self._counter += 1
            fid = f"file_{self._counter}"
            self.children.setdefault(parent_id, []).append(
                {
                    "id": fid,
                    "name": fpath.name,
                    "mimeType": "application/json",
                }
            )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"Uploaded {fid}", stderr="")
        if ns == "mutate" and cmd == "update":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="Updated", stderr="")
        raise AssertionError(f"Unexpected gdrive invocation: {args}")


def test_export_archive_summary_csv_preserves_failed_and_accepted_trials(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "g0001-deadbeef",
            "generation": 1,
            "parent_id": None,
            "status": "screen_rejected",
            "stage": "screen",
            "plane": "l1_kernel",
            "niche_key": "(plane=l1_kernel,focus=dsa)",
            "changed_lines": 204,
            "upstream_sha": "f2c23b1",
            "source_hash": "abc",
            "hypothesis": "DSA ragged prefill speedup",
            "patch_summary": "piecewise K>=2048",
            "notes": "screen geomean -0.31%",
            "created_at": "2026-11-27T00:00:00Z",
            "updated_at": "2026-11-27T00:10:00Z",
        },
        {
            "candidate_id": "g0009-fe01dea3",
            "generation": 9,
            "parent_id": None,
            "status": "accepted",
            "stage": "verify",
            "plane": "l2_scheduler",
            "niche_key": "(plane=l2_scheduler,focus=tbo)",
            "changed_lines": 125,
            "upstream_sha": "f2c23b1",
            "source_hash": "def",
            "hypothesis": "256-aligned pre-warmed TBO",
            "patch_summary": "16B comp_bt_gpu + TBO warmup",
            "notes": "geomean +0.59%",
            "created_at": "2026-11-27T01:00:00Z",
            "updated_at": "2026-11-27T02:00:00Z",
        },
    ]
    out_csv = export_archive_summary_csv(rows, tmp_path / "lineage.csv")
    with out_csv.open(encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert len(parsed) == 2
    assert parsed[0]["status"] == "screen_rejected"
    assert parsed[1]["candidate_id"] == "g0009-fe01dea3"


def test_gdrive_sync_candidate_idempotent_and_skips_tarballs(tmp_path: Path) -> None:
    ev_dir = tmp_path / "evidence" / "g0009-fe01dea3"
    ev_dir.mkdir(parents=True)
    (ev_dir / "decision.json").write_text('{"accepted": true}', encoding="utf-8")
    (ev_dir / "screen-metrics.json").write_text('{"tiers": []}', encoding="utf-8")
    (ev_dir / "candidate.tar.gz").write_bytes(b"large-binary-tarball")
    patch_file = tmp_path / "patches" / "g0009-fe01dea3.patch"
    patch_file.parent.mkdir(parents=True)
    patch_file.write_text("diff --git a/x.py b/x.py\n", encoding="utf-8")

    sleeps: list[float] = []
    backend = _FakeGDriveBackend(transient_failures=2)
    syncer = GDriveBenchmarkSync(
        "root123",
        runner=backend,
        sleep_fn=sleeps.append,
        base_backoff_sec=0.25,
        max_workers=1,
    )

    first = syncer.sync_candidate_evidence(
        "g0009-fe01dea3",
        ev_dir,
        campaign_folder_name="sm120_longctx",
        patch_path=patch_file,
        archive_rows=[{"candidate_id": "g0009-fe01dea3", "status": "verify_rejected"}],
    )
    assert len(sleeps) == 2
    assert set(first.uploaded_files) == {
        "evidence/g0009-fe01dea3/decision.json",
        "evidence/g0009-fe01dea3/screen-metrics.json",
        "evidence/g0009-fe01dea3/g0009-fe01dea3.patch",
        "lineage_summary.csv",
    }
    assert first.updated_files == ()

    # Second run updates existing files in place instead of creating duplicates
    second = syncer.sync_candidate_evidence(
        "g0009-fe01dea3",
        ev_dir,
        campaign_folder_name="sm120_longctx",
        patch_path=patch_file,
        archive_rows=[{"candidate_id": "g0009-fe01dea3", "status": "verify_rejected"}],
    )
    assert second.uploaded_files == ()
    assert len(second.updated_files) == 4
    # Never invokes `share`
    assert not any("share" in call for call in backend.calls)


def test_gdrive_sync_rejects_empty_folder_id() -> None:
    with pytest.raises(ValueError, match="root_folder_id"):
        GDriveBenchmarkSync("   ")


def test_export_evidence_ledger_consolidates_json_artifacts(tmp_path: Path) -> None:
    ev = tmp_path / "evidence" / "g0002-e1ad0d86"
    ev.mkdir(parents=True)
    (ev / "decision.json").write_text('{"accepted": false, "geomean_gain": 0.0045}', encoding="utf-8")
    (ev / "gdrive-sync.json").write_text('{"ignored": true}', encoding="utf-8")
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "g0002-e1ad0d86.patch").write_text("line1\nline2\n", encoding="utf-8")

    out = export_evidence_ledger_json(tmp_path, tmp_path / "evidence_ledger.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "g0002-e1ad0d86" in data
    assert data["g0002-e1ad0d86"]["decision"]["geomean_gain"] == 0.0045
    assert "gdrive-sync" not in data["g0002-e1ad0d86"]
    assert data["g0002-e1ad0d86"]["patch_lines"] == 2

