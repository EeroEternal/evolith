from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evolith.cli import app
from evolith.zstep.archive import ZStepArchive

runner = CliRunner()


def test_zstep_evolve_constructs_isolated_task(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = []

    class FakeLoop:
        def __init__(self, task, *, archive) -> None:
            tasks.append(task)
            self.archive = archive

        def run(self) -> list[object]:
            from evolith.zstep.task import CandidateStatus

            return [CandidateStatus.ACCEPTED]

    monkeypatch.setattr("evolith.zstep.loop.ZStepRSILoop", FakeLoop)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "zstep-evolve",
            "--source",
            str(tmp_path / "zstep"),
            "--generations",
            "2",
            "--local-timeout",
            "1800",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(tasks) == 1
    assert tasks[0].project_root == tmp_path
    assert tasks[0].source_repo == tmp_path / "zstep"
    assert tasks[0].max_generations == 2
    assert tasks[0].local_timeout_sec == 1800
    assert "accepted" in result.stdout


def test_zstep_evolve_defaults_to_one_generation(tmp_path: Path, monkeypatch) -> None:
    tasks = []

    class FakeLoop:
        def __init__(self, task, *, archive) -> None:
            tasks.append(task)

        def run(self) -> list[str]:
            return []

    monkeypatch.setattr("evolith.zstep.loop.ZStepRSILoop", FakeLoop)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["zstep-evolve", "--help"])
    assert result.exit_code == 0
    assert "deployment" in result.stdout

    result = runner.invoke(app, ["zstep-evolve"])
    assert result.exit_code == 0
    assert tasks[0].max_generations == 1


def test_zstep_resume_status_and_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    empty = runner.invoke(app, ["zstep-status"])
    assert empty.exit_code == 0
    assert "No zstep candidates" in empty.stdout

    with ZStepArchive(tmp_path / ".evolith/zstep-rsi/archive.sqlite3") as archive:
        archive.start_candidate(
            "g0001-test",
            1,
            None,
            b"patch",
            "sha",
            changed_lines=12,
            hypothesis="Fuse epilogue",
        )
        archive.accept_candidate("g0001-test", "evidence/g0001-test", 0.04)

    status = runner.invoke(app, ["zstep-status"])
    evidence = runner.invoke(app, ["zstep-evidence", "g0001-test"])
    missing = runner.invoke(app, ["zstep-evidence", "missing"])
    resumed = runner.invoke(app, ["zstep-resume", "g0001-test"])

    assert "g0001-test" in status.stdout
    assert "accepted" in status.stdout
    assert "Fuse epilogue" in status.stdout
    assert "evidence/g0001-test" in evidence.stdout
    assert missing.exit_code == 1
    assert resumed.exit_code == 0
    assert "accepted" in resumed.stdout
