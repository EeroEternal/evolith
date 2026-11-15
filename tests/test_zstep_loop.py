from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolith.zstep.archive import ZStepArchive
from evolith.zstep.loop import ZStepRSILoop
from evolith.zstep.task import CandidateStatus, ZStepRSITask


class FakeWorkspace:
    def __init__(self, task: ZStepRSITask) -> None:
        self.task = task
        self.received_patches: list[list[bytes]] = []

    def create_candidate(self, candidate_id: str, patches: list[bytes]) -> Path:
        self.received_patches.append(patches)
        path = self.task.candidates_dir / candidate_id
        path.mkdir(parents=True)
        (path / "source.txt").write_text(candidate_id, encoding="utf-8")
        return path

    def capture_patch(self, candidate_dir: Path) -> bytes:
        return f"patch-{candidate_dir.name}".encode()

    def freeze_source(self, source_dir: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_dir / "source.txt").read_bytes())
        return destination


class FakeProposer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
        self.prompts.append(prompt)
        return (
            'EVOLITH_PROPOSAL_JSON: {"hypothesis": "Fuse decode norm", '
            '"niche_key": "(plane=l1_kernel,focus=norm)", "patch_summary": "Fused norm"}'
        )


class FakeRemote:
    def __init__(
        self,
        candidate_throughputs: list[float] | None = None,
        screen_throughput: float = 90.0,
        screen_recompilations: int = 0,
    ) -> None:
        self.candidate_throughputs = candidate_throughputs or [105.0]
        self.screen_throughput = screen_throughput
        self.screen_recompilations = screen_recompilations
        self.screen_calls = 0
        self.verify_calls = 0

    def preflight(self) -> None:
        return

    def screen_baseline(self, baseline_hash: str, baseline_archive: Path) -> Path:
        path = baseline_archive.parent / "base-screen.json"
        return self._benchmark(path, 90.0, label="in512_c1")

    def screen(self, candidate_id: str, source: Path, evidence: Path) -> Path:
        self.screen_calls += 1
        self._parity(evidence / "screen-parity.json", self.screen_recompilations)
        return self._benchmark(
            evidence / "screen-metrics.json", self.screen_throughput, label="in512_c1"
        )

    def verify(
        self,
        candidate_id: str,
        baseline_archive: Path,
        candidate_archive: Path,
        evidence: Path,
    ) -> tuple[Path, Path, Path]:
        index = min(self.verify_calls, len(self.candidate_throughputs) - 1)
        self.verify_calls += 1
        throughput = self.candidate_throughputs[index]
        candidate = self._benchmark(evidence / "candidate.json", throughput)
        baseline = self._benchmark(evidence / "baseline.json", 100.0)
        parity = self._parity(evidence / "parity.json", 0)
        return candidate, baseline, parity

    @staticmethod
    def _parity(path: Path, recompilations: int) -> Path:
        path.write_text(
            json.dumps(
                {
                    "triton_recompilations": recompilations,
                    "cases": [
                        {
                            "name": "prime_squares_table",
                            "single_match": {"exact_token_match_rate": 1.0},
                            "batched_match": {"exact_token_match_rate": 1.0},
                            "chunked_match": {"exact_token_match_rate": 1.0},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _benchmark(path: Path, throughput: float, label: str = "hidden") -> Path:
        path.write_text(
            json.dumps(
                {
                    "tiers": [
                        {
                            "label": label,
                            "repeat": 0,
                            "output_throughput_tok_s": throughput,
                            "median_ttft_ms": 100.0,
                            "mean_tpot_ms": 10.0,
                            "p99_itl_ms": 20.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path


def make_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generations: int = 1,
    remote: FakeRemote | None = None,
) -> tuple[ZStepRSILoop, ZStepArchive, FakeProposer, FakeRemote]:
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)
    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "source",
        max_generations=generations,
    )
    archive = ZStepArchive(task.state_dir / "archive.sqlite3")
    proposer = FakeProposer()
    remote = remote or FakeRemote()
    loop = ZStepRSILoop(
        task,
        workspace=FakeWorkspace(task),
        archive=archive,
        proposer=proposer,
        remote=remote,
    )
    return loop, archive, proposer, remote


def test_accepted_patch_becomes_parent_and_prompt_includes_allowlist_and_4tuple_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, archive, proposer, _ = make_loop(
        tmp_path,
        monkeypatch,
        generations=3,
        remote=FakeRemote([105.0, 95.0, 95.0]),
    )

    statuses = loop.run()

    assert statuses == [
        CandidateStatus.ACCEPTED,
        CandidateStatus.VERIFY_REJECTED,
        CandidateStatus.VERIFY_REJECTED,
    ]
    first_patch = archive.list_candidates()[0]["patch"]
    assert archive.accepted_patches() == [first_patch]
    assert "worker/zstep_worker/models/deepseek_v41.py" in proposer.prompts[0]
    assert "SM120 Hardware & Bottleneck Reflection Card" in proposer.prompts[0]
    assert proposer.prompts[1].startswith("Parent: g0001")
    assert "Fuse decode norm" in proposer.prompts[1]
    assert "geomean throughput gain" in proposer.prompts[2]


def test_screen_rejection_never_runs_expensive_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = FakeRemote(screen_throughput=75.0)
    loop, archive, _, _ = make_loop(tmp_path, monkeypatch, remote=remote)

    statuses = loop.run()

    assert statuses == [CandidateStatus.SCREEN_REJECTED]
    assert remote.screen_calls == 1
    assert remote.verify_calls == 0
    assert "screen in512_c1 throughput regressed" in str(
        archive.list_candidates()[0]["failure_reason"]
    )


def test_proposer_failure_is_recorded_in_archive_without_ghost_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, archive, _, remote = make_loop(tmp_path, monkeypatch)

    class FailingProposer:
        def propose(self, *args: object) -> str:
            raise RuntimeError("kimi timed out")

    loop.proposer = FailingProposer()
    statuses = loop.run()

    assert statuses == [CandidateStatus.LOCAL_REJECTED]
    assert remote.screen_calls == 0
    rows = archive.list_candidates()
    assert len(rows) == 1
    assert rows[0]["status"] == CandidateStatus.LOCAL_REJECTED.value
    assert "kimi timed out" in str(rows[0]["failure_reason"])


def test_local_rejection_never_uses_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, archive, _, remote = make_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "evolith.zstep.loop.run_local_gates",
        lambda *args: (_ for _ in ()).throw(RuntimeError("cargo failed")),
    )

    statuses = loop.run()

    assert statuses == [CandidateStatus.LOCAL_REJECTED]
    assert remote.screen_calls == 0
    assert remote.verify_calls == 0


def test_preflight_failure_is_recorded_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BusyRemote(FakeRemote):
        def preflight(self) -> None:
            raise RuntimeError("GPU occupied")

    loop, archive, _, remote = make_loop(
        tmp_path, monkeypatch, generations=2, remote=BusyRemote()
    )

    statuses = loop.run()

    assert statuses == [CandidateStatus.FAILED]
    assert remote.screen_calls == 0
    assert len(archive.list_candidates()) == 1


def test_resume_completes_interrupted_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, archive, _, remote = make_loop(
        tmp_path, monkeypatch, remote=FakeRemote([106.0])
    )
    evidence = loop.task.state_dir / "evidence/g0003-resume"
    evidence.mkdir(parents=True)
    base_tar = evidence / "baseline.tar.gz"
    cand_tar = evidence / "candidate.tar.gz"
    base_tar.write_bytes(b"frozen-base")
    cand_tar.write_bytes(b"frozen-cand")
    cand_sha = hashlib.sha256(b"frozen-cand").hexdigest()
    archive.start_candidate(
        "g0003-resume",
        3,
        None,
        b"patch",
        cand_sha,
        changed_lines=10,
        hypothesis="Overlap shared experts",
        evidence_dir=str(evidence),
    )

    status = loop.resume("g0003-resume")

    assert status == CandidateStatus.ACCEPTED
    assert archive.current_parent() == "g0003-resume"
    assert remote.verify_calls == 1


def test_tampered_frozen_candidate_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TamperingRemote(FakeRemote):
        def verify(self, *args: object) -> tuple[Path, Path, Path]:
            candidate_archive = args[2]
            assert isinstance(candidate_archive, Path)
            candidate_archive.write_bytes(b"tampered")
            return super().verify(*args)  # type: ignore[arg-type]

    loop, archive, _, _ = make_loop(
        tmp_path, monkeypatch, remote=TamperingRemote()
    )

    statuses = loop.run()

    assert statuses == [CandidateStatus.FAILED]
    assert archive.accepted_patches() == []
    assert "source hash" in str(archive.list_candidates()[0]["failure_reason"])


def test_premise_two_strike_circuit_breaker_blocks_third_attempt_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = FakeRemote(candidate_throughputs=[100.5, 100.4, 108.0])
    loop, archive, proposer, _ = make_loop(
        tmp_path, monkeypatch, generations=3, remote=remote
    )

    statuses = loop.run()

    assert statuses == [
        CandidateStatus.VERIFY_REJECTED,
        CandidateStatus.VERIFY_REJECTED,
        CandidateStatus.LOCAL_REJECTED,
    ]
    # Attempt 3 is blocked locally by Premise 2-Strike without calling remote screen/verify!
    assert remote.screen_calls == 2
    assert remote.verify_calls == 2
    assert "BLOCKED `(plane=l1_kernel,focus=norm)`" in proposer.prompts[2]
    assert "Premise 2-strike circuit breaker tripped" in str(
        archive.list_candidates()[2]["failure_reason"]
    )


def test_redteam_gate_blocks_unsafe_stream_patch_in_loop_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = FakeRemote()
    loop, archive, _, _ = make_loop(tmp_path, monkeypatch, generations=1, remote=remote)

    class UnsafeWorkspace(FakeWorkspace):
        def capture_patch(self, candidate_dir: Path) -> bytes:
            return (
                b"diff --git a/worker/zstep_worker/runtime_loop.py b/worker/zstep_worker/runtime_loop.py\n"
                b"--- a/worker/zstep_worker/runtime_loop.py\n"
                b"+++ b/worker/zstep_worker/runtime_loop.py\n"
                b"@@ -1,1 +1,2 @@\n"
                b"+self.bg_stream = torch.cuda.Stream()\n"
            )

    loop.workspace = UnsafeWorkspace(loop.task)
    statuses = loop.run()

    assert statuses == [CandidateStatus.LOCAL_REJECTED]
    assert remote.screen_calls == 0
    row = archive.list_candidates()[0]
    assert "Red-Team gate rejected patch" in str(row["failure_reason"])
    assert (Path(str(row["evidence_dir"])) / "redteam-report.json").is_file()


def test_propose_only_campaign_freezes_candidate_and_resumes_with_campaign_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)
    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "source",
        campaign="t1-1",
        propose_only=True,
    )
    archive = ZStepArchive(task.state_dir / "archive.sqlite3")
    proposer = FakeProposer()
    remote = FakeRemote(candidate_throughputs=[103.5])
    preflight_called = False

    def fail_if_called_in_propose_only() -> None:
        nonlocal preflight_called
        preflight_called = True
        raise AssertionError("remote.preflight() must not be called in propose_only mode")

    class LargeCampaignWorkspace(FakeWorkspace):
        def capture_patch(self, candidate_dir: Path) -> bytes:
            # 160-line patch: under default 120 free LOC a +3.5% gain drops to ~2.67% net
            # and is rejected, whereas under T1-1 campaign (180 free LOC) it passes!
            return ("\n".join(f"+line_{i}" for i in range(160))).encode("utf-8")

    monkeypatch.setattr(remote, "preflight", fail_if_called_in_propose_only)
    loop = ZStepRSILoop(
        task,
        workspace=LargeCampaignWorkspace(task),
        archive=archive,
        proposer=proposer,
        remote=remote,
    )

    statuses = loop.run()
    assert statuses == [CandidateStatus.PROPOSED]
    assert preflight_called is False
    assert remote.screen_calls == 0
    assert "[Active Campaign T1-1 Blueprint" in proposer.prompts[0]
    assert "Maximum changed lines (added + deleted): 320 LOC" in proposer.prompts[0]

    row = archive.list_candidates()[0]
    candidate_id = str(row["candidate_id"])
    evidence_dir = Path(str(row["evidence_dir"]))
    assert row["stage"] == "preflight"
    assert row["changed_lines"] == 160
    assert (evidence_dir / "campaign.json").is_file()
    assert (evidence_dir / "dispatch-verification.json").is_file()
    assert (evidence_dir / "candidate.tar.gz").is_file()

    resume_task = ZStepRSITask(project_root=tmp_path, source_repo=tmp_path / "source")
    resume_remote = FakeRemote(candidate_throughputs=[103.5])
    resumed_loop = ZStepRSILoop(
        resume_task,
        workspace=FakeWorkspace(resume_task),
        archive=archive,
        proposer=proposer,
        remote=resume_remote,
    )
    final_status = resumed_loop.resume(candidate_id)
    assert final_status == CandidateStatus.ACCEPTED
    assert archive.current_parent() == candidate_id


def test_auto_campaign_rotates_from_t1_1_to_t1_2_after_two_strikes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)
    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "source",
        campaign="auto",
        max_generations=3,
    )
    archive = ZStepArchive(task.state_dir / "archive.sqlite3")

    class CampaignAwareProposer:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
            self.prompts.append(prompt)
            if "T1-1 Blueprint" in prompt:
                niche = "(plane=l2_scheduler,focus=dsv41_device_token_overlap)"
            else:
                niche = "(plane=l1_kernel,focus=sm120_dispatch_buckets)"
            return (
                f'EVOLITH_PROPOSAL_JSON: {{"hypothesis": "Campaign step", '
                f'"niche_key": "{niche}", "patch_summary": "Focused mutation"}}'
            )

    proposer = CampaignAwareProposer()
    remote = FakeRemote(candidate_throughputs=[100.5, 100.5, 105.0])
    loop = ZStepRSILoop(
        task,
        workspace=FakeWorkspace(task),
        archive=archive,
        proposer=proposer,
        remote=remote,
    )

    statuses = loop.run()
    assert statuses == [
        CandidateStatus.VERIFY_REJECTED,
        CandidateStatus.VERIFY_REJECTED,
        CandidateStatus.ACCEPTED,
    ]
    assert "T1-1 Blueprint" in proposer.prompts[0]
    assert "T1-1 Blueprint" in proposer.prompts[1]
    assert "T1-2 Blueprint" in proposer.prompts[2]
