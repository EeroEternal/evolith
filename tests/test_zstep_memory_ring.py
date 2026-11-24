from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evolith.cli import app
from evolith.zstep.archive import ZStepArchive
from evolith.zstep.loop import ZStepRSILoop
from evolith.zstep.memory_ring import load_memory_ring
from evolith.zstep.task import ZStepRSITask


class _StubWorkspace:
    def __init__(self, task: ZStepRSITask) -> None:
        self.task = task

    def create_candidate(self, candidate_id: str, patches: list[bytes]) -> Path:
        path = self.task.candidates_dir / candidate_id
        path.mkdir(parents=True)
        (path / "source.txt").write_bytes(b"baseline")
        return path

    def capture_patch(self, candidate_dir: Path) -> bytes:
        (candidate_dir / "source.txt").write_text(candidate_dir.name, encoding="utf-8")
        return f"patch-{candidate_dir.name}".encode()

    def freeze_source(self, source_dir: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_dir / "source.txt").read_bytes())
        return destination


class _StubProposer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
        self.prompts.append(prompt)
        return (
            'EVOLITH_PROPOSAL_JSON: {"hypothesis": "Use PDL", '
            '"niche_key": "(plane=l1_kernel,focus=pdl)", "patch_summary": "PDL"}'
        )


def test_load_memory_ring_filters_facts_by_allowed_roots_and_extracts_active_directives(
    tmp_path: Path,
) -> None:
    assert load_memory_ring(tmp_path) == ""

    facts = tmp_path / "memory/facts"
    facts.mkdir(parents=True)
    (facts / "01-moe.md").write_text(
        "<!-- roots: worker/zstep_worker/kernels_sm120/b12x_dsv41.py -->\n"
        "# MoE Facts\n- Zero-pad I=288->320 for N64.\n",
        encoding="utf-8",
    )
    (facts / "02-sched.md").write_text(
        "<!-- roots: crates/zs-scheduler/src -->\n"
        "# Scheduler Facts\n- Clamp mixed_cap <= 3072.\n",
        encoding="utf-8",
    )
    (tmp_path / "memory/directives.md").write_text(
        "# Directives\n\n## Active Directives\n- Combine g0021 and g0025.\n\n## Archived\n- Old directive.\n",
        encoding="utf-8",
    )
    (tmp_path / "memory/stepping-stones.md").write_text(
        "# Stones\n\n## 1. Accepted\n- g0018\n\n## 2. Near-Miss\n- g0025 (+2.88%)\n",
        encoding="utf-8",
    )

    # Scoped to crates/zs-scheduler/src -> includes 02-sched, excludes 01-moe
    sched_mem = load_memory_ring(
        tmp_path, allowed_roots=("crates/zs-scheduler/src/serving.rs",)
    )
    assert "Clamp mixed_cap <= 3072" in sched_mem
    assert "Zero-pad I=288->320" not in sched_mem
    assert "Combine g0021 and g0025" in sched_mem
    assert "Old directive" not in sched_mem
    assert "g0025 (+2.88%)" in sched_mem
    assert "<!-- roots:" not in sched_mem

    # Scoped to kernels_sm120 -> includes 01-moe, excludes 02-sched
    moe_mem = load_memory_ring(
        tmp_path, allowed_roots=("worker/zstep_worker/kernels_sm120",)
    )
    assert "Zero-pad I=288->320" in moe_mem
    assert "Clamp mixed_cap <= 3072" not in moe_mem


def test_zstep_loop_injects_local_memory_ring_before_dynamic_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)
    (tmp_path / "memory/facts").mkdir(parents=True)
    (tmp_path / "memory/facts/ar.md").write_text(
        "<!-- roots: worker/zstep_worker/kernels_sm120 -->\n# AR Fact\n- Keep 2-slot PXB ring.\n",
        encoding="utf-8",
    )
    (tmp_path / "memory/directives.md").write_text(
        "## Active Directives\n- Test single-stream PDL overlap.\n",
        encoding="utf-8",
    )

    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "source",
        campaign="t1-2",
        propose_only=True,
    )
    archive = ZStepArchive(task.state_dir / "archive.sqlite3")
    proposer = _StubProposer()
    loop = ZStepRSILoop(
        task,
        workspace=_StubWorkspace(task),
        archive=archive,
        proposer=proposer,
        remote=object(),
    )
    loop.run()

    prompt = proposer.prompts[0]
    fact_pos = prompt.index("Keep 2-slot PXB ring.")
    dir_pos = prompt.index("Test single-stream PDL overlap.")
    lineage_pos = prompt.index("AIDE^2 Compressed Lineage Memory")
    assert fact_pos < dir_pos < lineage_pos


def test_cli_zstep_memory_command_renders_scoped_real_repo_memory() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["zstep-memory", "--campaign", "t1-3"])
    assert res.exit_code == 0
    assert "t1-3-chunked-prefill" in res.stdout
    assert "Full-Lifecycle Inference Optimization Laws" in res.stdout
    assert "Amdahl" in res.stdout
    assert "Active Human & Scout Directives" in res.stdout
    assert "FlashInfer Autotune" in res.stdout
    assert "SM120 MoE" not in res.stdout
    assert "...[truncated]" not in res.stdout
