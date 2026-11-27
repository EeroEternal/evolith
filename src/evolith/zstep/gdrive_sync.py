from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

_STATISTICAL_SUFFIXES = frozenset({".json", ".csv", ".tsv", ".md", ".patch", ".txt"})
_TRANSIENT_MARKERS = (
    "500",
    "502",
    "503",
    "504",
    "eof",
    "timeout",
    "timed out",
    "connection reset",
    "temporarily unavailable",
    "rate limit",
    "quota exceeded",
)


@dataclass(frozen=True)
class GDriveSyncReport:
    root_folder_id: str
    folder_url: str
    uploaded_files: tuple[str, ...] = ()
    updated_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_folder_id": self.root_folder_id,
            "folder_url": self.folder_url,
            "uploaded_count": len(self.uploaded_files),
            "updated_count": len(self.updated_files),
            "uploaded_files": list(self.uploaded_files),
            "updated_files": list(self.updated_files),
            "errors": list(self.errors),
        }


CommandRunner = Callable[[list[str], int], subprocess.CompletedProcess[str]]


def _default_runner(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def export_archive_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Export append-only lineage rows (including rejected and failed trials) to CSV."""
    headers = [
        "candidate_id",
        "generation",
        "parent_id",
        "status",
        "stage",
        "plane",
        "niche_key",
        "changed_lines",
        "upstream_sha",
        "source_hash",
        "hypothesis",
        "patch_summary",
        "notes",
        "created_at",
        "updated_at",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(col, "") for col in headers])
    return output_path


class GDriveBenchmarkSync:
    """Idempotent Google Drive archiver for Evolith statistical benchmark artifacts.

    Uses bare `gdrive` (`readonly` / `mutate` subcommands) and never widens file
    sharing permissions. Source tarballs (`*.tar.gz`) are skipped by default so
    only structured statistical telemetry, parity reports, CSV summaries, and
    patches are synced.
    """

    def __init__(
        self,
        root_folder_id: str,
        *,
        gdrive_bin: str = "gdrive",
        max_retries: int = 4,
        base_backoff_sec: float = 1.0,
        timeout_sec: int = 120,
        runner: CommandRunner | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        cleaned = root_folder_id.strip()
        if not cleaned:
            raise ValueError("root_folder_id must be non-empty")
        self.root_folder_id = cleaned
        self.gdrive_bin = gdrive_bin
        self.max_retries = max(1, max_retries)
        self.base_backoff_sec = max(0.0, base_backoff_sec)
        self.timeout_sec = max(5, timeout_sec)
        self._runner = runner or _default_runner
        self._sleep = sleep_fn

    @property
    def root_folder_url(self) -> str:
        return f"https://drive.google.com/drive/folders/{self.root_folder_id}"

    def _run_gdrive(self, subargs: list[str]) -> str:
        cmd = [self.gdrive_bin, *subargs]
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                proc = self._runner(cmd, self.timeout_sec)
                if proc.returncode == 0:
                    return (proc.stdout or "").strip()
                last_err = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
            except subprocess.TimeoutExpired as exc:
                last_err = f"timeout after {self.timeout_sec}s: {exc}"
            lowered = last_err.lower()
            is_transient = any(marker in lowered for marker in _TRANSIENT_MARKERS)
            if not is_transient or attempt + 1 >= self.max_retries:
                break
            self._sleep(min(12.0, self.base_backoff_sec * (2**attempt)))
        raise RuntimeError(f"gdrive {' '.join(subargs[:3])} failed: {last_err}")

    def list_children(self, folder_id: str) -> list[dict[str, Any]]:
        out = self._run_gdrive(["readonly", "ls", folder_id, "--max", "500", "--json"])
        if not out:
            return []
        parsed = json.loads(out)
        return parsed if isinstance(parsed, list) else []

    def ensure_subfolder(self, name: str, parent_id: str) -> str:
        for item in self.list_children(parent_id):
            if (
                item.get("name") == name
                and item.get("mimeType") == "application/vnd.google-apps.folder"
            ):
                return str(item["id"])
        self._run_gdrive(["mutate", "mkdir", name, "--parent", parent_id])
        for item in self.list_children(parent_id):
            if (
                item.get("name") == name
                and item.get("mimeType") == "application/vnd.google-apps.folder"
            ):
                return str(item["id"])
        raise RuntimeError(f"Created Drive subfolder `{name}` under `{parent_id}` could not be resolved")

    @staticmethod
    def is_statistical_artifact(path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.lower()
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            return False
        return path.suffix.lower() in _STATISTICAL_SUFFIXES

    def sync_directory(
        self,
        local_dir: Path,
        target_folder_id: str,
        *,
        extra_files: tuple[Path, ...] = (),
    ) -> tuple[list[str], list[str]]:
        existing = {
            str(item["name"]): str(item["id"])
            for item in self.list_children(target_folder_id)
            if item.get("mimeType") != "application/vnd.google-apps.folder"
            and item.get("name")
            and item.get("id")
        }
        candidates: list[Path] = []
        if local_dir.is_dir():
            candidates.extend(
                sorted(p for p in local_dir.iterdir() if self.is_statistical_artifact(p))
            )
        for extra in extra_files:
            if self.is_statistical_artifact(extra):
                candidates.append(extra)

        uploaded: list[str] = []
        updated: list[str] = []
        seen_names: set[str] = set()
        for fpath in candidates:
            if fpath.name in seen_names:
                continue
            seen_names.add(fpath.name)
            if fpath.name in existing:
                self._run_gdrive(["mutate", "update", existing[fpath.name], str(fpath)])
                updated.append(fpath.name)
            else:
                self._run_gdrive(["mutate", "upload", str(fpath), "--parent", target_folder_id])
                uploaded.append(fpath.name)
        return uploaded, updated

    def sync_candidate_evidence(
        self,
        candidate_id: str,
        evidence_dir: Path,
        *,
        campaign_folder_name: str = "evolith_rsi_runs",
        patch_path: Path | None = None,
        archive_rows: list[dict[str, Any]] | None = None,
    ) -> GDriveSyncReport:
        uploaded_all: list[str] = []
        updated_all: list[str] = []
        campaign_id = self.ensure_subfolder(campaign_folder_name, self.root_folder_id)
        evidence_root_id = self.ensure_subfolder("evidence", campaign_id)
        cand_folder_id = self.ensure_subfolder(candidate_id, evidence_root_id)

        extras = (patch_path,) if patch_path is not None and patch_path.is_file() else ()
        up, upd = self.sync_directory(evidence_dir, cand_folder_id, extra_files=extras)
        uploaded_all.extend(f"evidence/{candidate_id}/{x}" for x in up)
        updated_all.extend(f"evidence/{candidate_id}/{x}" for x in upd)

        if archive_rows is not None:
            csv_path = evidence_dir.parent / "lineage_summary.csv"
            export_archive_summary_csv(archive_rows, csv_path)
            l_up, l_upd = self.sync_directory(Path("/nonexistent"), campaign_id, extra_files=(csv_path,))
            uploaded_all.extend(l_up)
            updated_all.extend(l_upd)

        return GDriveSyncReport(
            root_folder_id=self.root_folder_id,
            folder_url=self.root_folder_url,
            uploaded_files=tuple(uploaded_all),
            updated_files=tuple(updated_all),
        )

    def sync_state_and_benchmarks(
        self,
        state_dir: Path,
        *,
        archive_rows: list[dict[str, Any]] | None = None,
        benchmark_dirs: tuple[Path, ...] = (),
        campaign_folder_name: str = "evolith_rsi_runs",
    ) -> GDriveSyncReport:
        uploaded_all: list[str] = []
        updated_all: list[str] = []
        errors: list[str] = []

        campaign_id = self.ensure_subfolder(campaign_folder_name, self.root_folder_id)
        if archive_rows is not None:
            csv_path = state_dir / "lineage_summary.csv"
            export_archive_summary_csv(archive_rows, csv_path)
            up, upd = self.sync_directory(Path("/nonexistent"), campaign_id, extra_files=(csv_path,))
            uploaded_all.extend(up)
            updated_all.extend(upd)

        baseline_cache = state_dir / "baseline-cache"
        if baseline_cache.is_dir():
            b_id = self.ensure_subfolder("baseline-cache", campaign_id)
            up, upd = self.sync_directory(baseline_cache, b_id)
            uploaded_all.extend(f"baseline-cache/{x}" for x in up)
            updated_all.extend(f"baseline-cache/{x}" for x in upd)

        evidence_root = state_dir / "evidence"
        if evidence_root.is_dir():
            ev_root_id = self.ensure_subfolder("evidence", campaign_id)
            for cand_dir in sorted(p for p in evidence_root.iterdir() if p.is_dir()):
                try:
                    c_id = self.ensure_subfolder(cand_dir.name, ev_root_id)
                    patch_file = state_dir / "patches" / f"{cand_dir.name}.patch"
                    extras = (patch_file,) if patch_file.is_file() else ()
                    up, upd = self.sync_directory(cand_dir, c_id, extra_files=extras)
                    uploaded_all.extend(f"evidence/{cand_dir.name}/{x}" for x in up)
                    updated_all.extend(f"evidence/{cand_dir.name}/{x}" for x in upd)
                except Exception as exc:
                    errors.append(f"{cand_dir.name}: {exc}")

        for bdir in benchmark_dirs:
            if not bdir.is_dir():
                continue
            b_folder_id = self.ensure_subfolder(bdir.name, self.root_folder_id)
            up, upd = self.sync_directory(bdir, b_folder_id)
            uploaded_all.extend(f"{bdir.name}/{x}" for x in up)
            updated_all.extend(f"{bdir.name}/{x}" for x in upd)

        return GDriveSyncReport(
            root_folder_id=self.root_folder_id,
            folder_url=self.root_folder_url,
            uploaded_files=tuple(uploaded_all),
            updated_files=tuple(updated_all),
            errors=tuple(errors),
        )
