from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from evolith.zstep.proposer import KimiCodeProposer, parse_proposal_metadata
from evolith.zstep.task import ZStepRSITask


def make_task(tmp_path: Path, **overrides: object) -> ZStepRSITask:
    values: dict[str, object] = {
        "project_root": tmp_path,
        "source_repo": tmp_path / "source",
    }
    values.update(overrides)
    return ZStepRSITask(**values)  # type: ignore[arg-type]


def fake_kimi(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "kimi"
    executable.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    executable.chmod(0o755)
    return executable


def test_proposer_runs_in_candidate_and_scrubs_credentials_and_ssh_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = fake_kimi(
        tmp_path,
        """import json
import os
from pathlib import Path
import sys
Path("invocation.json").write_text(json.dumps({
    "cwd": os.getcwd(),
    "args": sys.argv[1:],
    "git_prompt": os.environ.get("GIT_TERMINAL_PROMPT"),
    "credentials": [key for key in os.environ if key in {
        "SSHPASS", "SSH_AUTH_SOCK", "SSH_AGENT_PID", "GIT_SSH_COMMAND",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MY_TOKEN", "MY_SECRET"
    }],
}))
Path("worker/zstep_worker").mkdir(parents=True)
Path("worker/zstep_worker/runtime_loop.py").write_text("# candidate\\n")
print("candidate ready")
""",
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in (
        "SSHPASS",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GIT_SSH_COMMAND",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MY_TOKEN",
        "MY_SECRET",
    ):
        monkeypatch.setenv(name, "do-not-forward")

    output = KimiCodeProposer(make_task(tmp_path), str(executable)).propose(
        candidate, "improve zstep", tmp_path / "evidence"
    )

    invocation = json.loads((candidate / "invocation.json").read_text())
    assert output.strip() == "candidate ready"
    assert invocation["cwd"] == str(candidate)
    assert invocation["args"] == ["-p", "improve zstep"]
    assert invocation["git_prompt"] == "0"
    assert invocation["credentials"] == []
    assert "candidate ready" in (tmp_path / "evidence/proposer.log").read_text()


def test_parse_proposal_metadata_handles_structured_and_fallback_outputs() -> None:
    structured = parse_proposal_metadata(
        'Done.\nEVOLITH_PROPOSAL_JSON: {"hypothesis": "Fuse AllReduce+RMSNorm", "niche_key": "(plane=l1_kernel,focus=ar_norm)", "patch_summary": "Fused AR+Norm"}',
        changed_files=("worker/zstep_worker/kernels_sm120/ar.py",),
        changed_lines=42,
    )
    assert structured.hypothesis == "Fuse AllReduce+RMSNorm"
    assert structured.niche_key == "(plane=l1_kernel,focus=ar_norm)"
    assert structured.patch_summary == "Fused AR+Norm"

    fallback = parse_proposal_metadata(
        "• Eliminate redundant host synchronization in chunked prefill step.",
        changed_files=("worker/zstep_worker/runtime_loop.py",),
        changed_lines=8,
    )
    assert "Eliminate redundant host synchronization" in fallback.hypothesis
    assert "runtime_loop.py (8 LOC)" in fallback.patch_summary


def test_proposer_rejects_nonzero_exit(tmp_path: Path) -> None:
    executable = fake_kimi(tmp_path, "raise SystemExit(7)\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    with pytest.raises(RuntimeError, match="exited 7"):
        KimiCodeProposer(make_task(tmp_path), str(executable)).propose(
            candidate, "prompt", tmp_path / "evidence"
        )


def test_proposer_enforces_timeout(tmp_path: Path) -> None:
    executable = fake_kimi(tmp_path, "import time\ntime.sleep(2)\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    with pytest.raises(subprocess.TimeoutExpired):
        KimiCodeProposer(make_task(tmp_path, local_timeout_sec=1), str(executable)).propose(
            candidate, "prompt", tmp_path / "evidence"
        )
