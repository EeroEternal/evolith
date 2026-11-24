from __future__ import annotations

from pathlib import Path
import subprocess
import tarfile

import pytest

from evolith.models import EvolutionPlane
from evolith.zstep.task import ZStepRSITask
from evolith.zstep.workspace import ZStepWorkspace


def run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_source_repo(tmp_path: Path) -> Path:
    source = tmp_path / "original-zstep"
    source.mkdir()
    run(["git", "init", "-q"], source)
    run(["git", "config", "user.email", "tests@example.invalid"], source)
    run(["git", "config", "user.name", "Evolith Tests"], source)
    write(source / "worker/zstep_worker/runtime_loop.py", "# original\n")
    write(source / "worker/zstep_worker/kernels_sm120/op.py", "# kernel\n")
    write(source / "Cargo.toml", "[workspace]\n")
    run(["git", "add", "."], source)
    run(["git", "commit", "-qm", "baseline"], source)
    return source


def make_task(tmp_path: Path, source: Path, **overrides: object) -> ZStepRSITask:
    values: dict[str, object] = {
        "project_root": tmp_path / "evolith-project",
        "source_repo": source,
        "baseline_sha": "HEAD",
    }
    values.update(overrides)
    return ZStepRSITask(**values)  # type: ignore[arg-type]


def test_candidate_is_isolated_and_patch_stack_is_replayed(tmp_path: Path) -> None:
    source = make_source_repo(tmp_path)
    manager = ZStepWorkspace(make_task(tmp_path, source))
    manager.prepare_upstream()
    first = manager.create_candidate("g0001-test", [])
    write(first / "worker/zstep_worker/runtime_loop.py", "# candidate one\n")

    stats = manager.capture_patch_stats(first)
    second = manager.create_candidate("g0002-test", [stats.patch])

    assert stats.patch
    assert stats.changed_lines == 2
    assert stats.plane == EvolutionPlane.L2_SCHEDULER
    assert (second / "worker/zstep_worker/runtime_loop.py").read_text() == "# candidate one\n"
    assert (source / "worker/zstep_worker/runtime_loop.py").read_text() == "# original\n"

    # Verify incremental patch capture + sandbox replay on top of an accepted patch layer
    with manager.isolated_proposer_sandbox(second) as sandbox2:
        write(sandbox2 / "worker/zstep_worker/kernels_sm120/op.py", "# kernel two\n")
        stats2 = manager.capture_patch_stats(sandbox2)
    manager.apply_patch(second, stats2.patch)
    assert stats2.changed_files == ("worker/zstep_worker/kernels_sm120/op.py",)
    assert stats2.changed_lines == 2
    assert (second / "worker/zstep_worker/kernels_sm120/op.py").read_text() == "# kernel two\n"


def test_isolated_proposer_sandbox_lives_outside_project_root_and_classifies_l1(
    tmp_path: Path,
) -> None:
    source = make_source_repo(tmp_path)
    task = make_task(tmp_path, source, sandbox_root=tmp_path / "external-sandbox")
    # Write a fake secret file inside evolith project_root to verify sandbox parent isolation
    write(task.project_root / "secret_verify_tiers.txt", "hidden")
    manager = ZStepWorkspace(task)
    candidate = manager.create_candidate("g0001-sandbox", [])

    with manager.isolated_proposer_sandbox(candidate) as sandbox_dir:
        assert task.project_root not in sandbox_dir.parents
        write(sandbox_dir / "worker/zstep_worker/kernels_sm120/op.py", "# fused kernel\n")
        stats = manager.capture_patch_stats(sandbox_dir)

    assert stats.plane == EvolutionPlane.L1_KERNEL
    assert (candidate / "worker/zstep_worker/kernels_sm120/op.py").read_text() == "# kernel\n"
    manager.apply_patch(candidate, stats.patch)
    assert (candidate / "worker/zstep_worker/kernels_sm120/op.py").read_text() == "# fused kernel\n"


def test_forbidden_manifest_change_is_rejected(tmp_path: Path) -> None:
    manager = ZStepWorkspace(make_task(tmp_path, make_source_repo(tmp_path)))
    candidate = manager.create_candidate("g0001-forbidden", [])
    write(candidate / "Cargo.toml", "[workspace]\nmembers = []\n")

    with pytest.raises(ValueError, match="forbidden path"):
        manager.capture_patch(candidate)


def test_oversized_patch_is_rejected(tmp_path: Path) -> None:
    manager = ZStepWorkspace(
        make_task(tmp_path, make_source_repo(tmp_path), max_changed_lines=1)
    )
    candidate = manager.create_candidate("g0001-large", [])
    write(candidate / "worker/zstep_worker/runtime_loop.py", "# one\n# two\n")

    with pytest.raises(ValueError, match="changed lines"):
        manager.capture_patch(candidate)


def test_freeze_source_excludes_git_and_build_outputs(tmp_path: Path) -> None:
    manager = ZStepWorkspace(make_task(tmp_path, make_source_repo(tmp_path)))
    candidate = manager.create_candidate("g0001-archive", [])
    write(candidate / "target/debug/generated.txt", "build")
    write(candidate / "worker/__pycache__/cache.pyc", "cache")

    archive = manager.freeze_source(candidate, tmp_path / "evidence/candidate.tar.gz")

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "worker/zstep_worker/runtime_loop.py" in names
    assert not any(name.startswith((".git", "target", "worker/__pycache__")) for name in names)

    # Identical source content frozen across two clones produces byte-identical SHA-256
    candidate_two = manager.create_candidate("g0002-archive", [])
    archive_two = manager.freeze_source(candidate_two, tmp_path / "evidence/candidate2.tar.gz")
    assert archive.read_bytes() == archive_two.read_bytes()


def test_prepare_upstream_syncs_when_baseline_sha_advances(tmp_path: Path) -> None:
    source = make_source_repo(tmp_path)
    sha1 = (
        subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=source, check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )
    ws1 = ZStepWorkspace(make_task(tmp_path, source, baseline_sha=sha1))
    assert ws1.upstream_head_sha() == sha1

    write(source / "worker/zstep_worker/env_config.py", "# env\n")
    run(["git", "add", "."], source)
    run(["git", "commit", "-qm", "merged pr"], source)
    sha2 = (
        subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=source, check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )
    assert sha2 != sha1

    ws2 = ZStepWorkspace(make_task(tmp_path, source, baseline_sha=sha2))
    assert ws2.upstream_head_sha() == sha2
    cand = ws2.create_candidate("g0013-rebased", [])
    assert (cand / "worker/zstep_worker/env_config.py").read_text() == "# env\n"
