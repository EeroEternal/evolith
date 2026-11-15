from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from evolith.models import EvolutionPlane
from evolith.zstep.task import PatchStats, ZStepRSITask

_EXCLUDED_PARTS = {".git", "target", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def classify_plane(changed_files: list[str] | tuple[str, ...]) -> EvolutionPlane:
    for path in changed_files:
        if path.startswith("worker/zstep_worker/kernels_sm120"):
            return EvolutionPlane.L1_KERNEL
    return EvolutionPlane.L2_SCHEDULER


class ZStepWorkspace:
    def __init__(self, task: ZStepRSITask) -> None:
        self.task = task

    def prepare_upstream(self) -> None:
        if self.task.upstream_dir.exists():
            return
        self.task.upstream_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                str(self.task.source_repo),
                str(self.task.upstream_dir),
            ]
        )
        _run(
            ["git", "checkout", "--detach", self.task.baseline_sha],
            self.task.upstream_dir,
        )

    def create_candidate(self, candidate_id: str, accepted_patches: list[bytes]) -> Path:
        self.prepare_upstream()
        destination = self.task.candidates_dir / candidate_id
        if destination.exists():
            raise FileExistsError(destination)
        self.task.candidates_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                str(self.task.upstream_dir),
                str(destination),
            ]
        )
        # Detach origin remote so candidate clones carry no path reference back into evolith
        subprocess.run(
            ["git", "remote", "remove", "origin"],
            cwd=destination,
            check=False,
            capture_output=True,
        )
        for patch in accepted_patches:
            self.apply_patch(destination, patch)
        return destination

    @staticmethod
    def apply_patch(target_dir: Path, patch: bytes) -> None:
        subprocess.run(
            ["git", "apply", "-"],
            cwd=target_dir,
            input=patch,
            check=True,
            capture_output=True,
        )

    @contextmanager
    def isolated_proposer_sandbox(self, candidate_dir: Path) -> Iterator[Path]:
        """
        Stages a temporary git clone outside `project_root` (under /tmp by default)
        for the untrusted coding agent so traversing parent directories (`../../..`)
        cannot expose Evolith's `task.py` (hidden `verify_tiers`) or `.evolith/` state.
        Only a validated allowlist patch is ever applied back to `candidate_dir`.
        """
        base_root = self.task.sandbox_root
        if base_root is not None:
            base_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="evolith-zstep-propose-",
            dir=str(base_root) if base_root is not None else None,
        ) as tmp_root:
            sandbox_dir = Path(tmp_root) / "zstep"
            shutil.copytree(
                candidate_dir,
                sandbox_dir,
                ignore=shutil.ignore_patterns("target", "__pycache__", ".pytest_cache", ".ruff_cache"),
            )
            yield sandbox_dir

    def _is_allowed(self, path: str) -> bool:
        candidate = PurePosixPath(path)
        for root in self.task.allowed_roots:
            allowed = PurePosixPath(root)
            if candidate == allowed or allowed in candidate.parents:
                return True
        return False

    def capture_patch_stats(self, candidate_dir: Path) -> PatchStats:
        changed = set(
            _run(["git", "diff", "--name-only"], candidate_dir)
            .stdout.decode()
            .splitlines()
        )
        untracked = set(
            _run(["git", "ls-files", "--others", "--exclude-standard"], candidate_dir)
            .stdout.decode()
            .splitlines()
        )
        all_files = tuple(sorted(changed | untracked))
        for path in all_files:
            if not self._is_allowed(path):
                raise ValueError(f"Candidate modifies forbidden path: {path}")
        if untracked:
            _run(["git", "add", "-N", "--", *sorted(untracked)], candidate_dir)

        numstat = _run(["git", "diff", "--numstat"], candidate_dir).stdout.decode()
        changed_lines = 0
        for line in numstat.splitlines():
            if not line.strip():
                continue
            added, deleted, _ = line.split("\t", 2)
            changed_lines += (0 if added == "-" else int(added)) + (
                0 if deleted == "-" else int(deleted)
            )
        if changed_lines > self.task.max_changed_lines:
            raise ValueError(
                f"Candidate has {changed_lines} changed lines; "
                f"limit is {self.task.max_changed_lines}"
            )

        patch = _run(["git", "diff", "--binary", "--full-index"], candidate_dir).stdout
        if not patch.strip():
            raise ValueError("Candidate produced no source diff")
        return PatchStats(
            patch=patch,
            changed_lines=changed_lines,
            changed_files=all_files,
            plane=classify_plane(all_files),
        )

    def capture_patch(self, candidate_dir: Path) -> bytes:
        return self.capture_patch_stats(candidate_dir).patch

    def freeze_source(self, source_dir: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            for path in sorted(source_dir.rglob("*")):
                relative = path.relative_to(source_dir)
                if _EXCLUDED_PARTS & set(relative.parts) or not path.is_file():
                    continue
                archive.add(path, arcname=str(relative), recursive=False)
        return destination
