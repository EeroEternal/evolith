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
            "--host",
            "gpu-server",
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

    result = runner.invoke(app, ["zstep-evolve", "--host", "gpu-server"])
    assert result.exit_code == 0
    assert tasks[0].max_generations == 1
    assert tasks[0].local_timeout_sec == 600


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
    resumed = runner.invoke(app, ["zstep-resume", "g0001-test", "--host", "gpu-server"])

    assert "g0001-test" in status.stdout
    assert "accepted" in status.stdout
    assert "Fuse epilogue" in status.stdout
    assert "evidence/g0001-test" in evidence.stdout
    assert missing.exit_code == 1
    assert resumed.exit_code == 0
    assert "accepted" in resumed.stdout


def test_zstep_campaigns_and_propose_only_flags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    campaigns_out = runner.invoke(app, ["zstep-campaigns"])
    assert campaigns_out.exit_code == 0, campaigns_out.stdout
    assert "t1-1-device-decode" in campaigns_out.stdout
    assert "t1-2-dispatch-buckets" in campaigns_out.stdout
    assert "NEXT (auto)" in campaigns_out.stdout

    captured_tasks = []

    class FakeLoop:
        def __init__(self, task, *, archive) -> None:
            captured_tasks.append(task)

        def run(self) -> list[object]:
            from evolith.zstep.task import CandidateStatus

            return [CandidateStatus.PROPOSED]

    monkeypatch.setattr("evolith.zstep.loop.ZStepRSILoop", FakeLoop)
    res = runner.invoke(
        app,
        [
            "zstep-evolve",
            "--campaign",
            "t2-3",
            "--propose-only",
            "--proposer",
            "openai",
            "--proposer-api-base",
            "http://127.0.0.1:43537/v1",
            "--proposer-model",
            "Qwen3.5-35B-A3B",
            "--gpu-smoke",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert captured_tasks[0].campaign == "t2-3"
    assert captured_tasks[0].propose_only is True
    assert captured_tasks[0].proposer_backend == "openai"
    assert captured_tasks[0].proposer_api_base == "http://127.0.0.1:43537/v1"
    assert captured_tasks[0].proposer_model == "Qwen3.5-35B-A3B"
    assert captured_tasks[0].enable_gpu_smoke is True
    assert "proposed" in res.stdout


def test_zstep_submit_injects_patch_proposer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    patch_file = tmp_path / "candidate.patch"
    patch_file.write_text("diff --git a/x b/x\n", encoding="utf-8")

    captured = {}

    class FakeLoop:
        def __init__(self, task, *, archive, proposer=None) -> None:
            captured["task"] = task
            captured["proposer"] = proposer

        def run(self) -> list[object]:
            from evolith.zstep.task import CandidateStatus

            return [CandidateStatus.PROPOSED]

    monkeypatch.setattr("evolith.zstep.loop.ZStepRSILoop", FakeLoop)
    res = runner.invoke(
        app,
        [
            "zstep-submit",
            "--patch",
            str(patch_file),
            "--hypothesis",
            "Device-resident decode + intermediate buckets",
            "--niche-key",
            "(plane=l2_scheduler,focus=dsv41_device_token_overlap)",
            "--summary",
            "int64 index_copy_ + buckets",
            "--campaign",
            "t1-1",
            "--propose-only",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert captured["task"].max_generations == 1
    assert captured["task"].campaign == "t1-1"
    assert captured["task"].propose_only is True
    assert captured["proposer"].patch_path == patch_file.resolve()
    assert captured["proposer"].niche_key == "(plane=l2_scheduler,focus=dsv41_device_token_overlap)"
    assert "proposed" in res.stdout
