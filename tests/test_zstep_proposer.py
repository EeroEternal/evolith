from __future__ import annotations

import json
from pathlib import Path
import subprocess

import httpx
import pytest

from evolith.zstep.proposer import (
    KimiCodeProposer,
    OpenAICompatibleProposer,
    PatchProposer,
    build_proposer,
    extract_unified_diff,
    parse_proposal_metadata,
)
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


def test_patch_proposer_applies_unified_diff_and_emits_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    target = repo / "worker/zstep_worker/runtime_loop.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    target.write_text("x = 2\n", encoding="utf-8")
    diff_bytes = subprocess.run(
        ["git", "diff"], cwd=repo, check=True, capture_output=True
    ).stdout
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, check=True)

    patch_file = tmp_path / "opt.patch"
    patch_file.write_bytes(diff_bytes)

    out = PatchProposer(
        patch_file,
        hypothesis="Overlap device tokens",
        niche_key="(plane=l2_scheduler,focus=dsv41_device_token_overlap)",
        patch_summary="Enable overlap",
    ).propose(repo, "prompt body", tmp_path / "evidence")

    assert target.read_text(encoding="utf-8") == "x = 2\n"
    meta = parse_proposal_metadata(out)
    assert meta.hypothesis == "Overlap device tokens"
    assert meta.niche_key == "(plane=l2_scheduler,focus=dsv41_device_token_overlap)"
    assert meta.patch_summary == "Enable overlap"
    assert (tmp_path / "evidence/prompt.txt").read_text(encoding="utf-8") == "prompt body"


def test_openai_compatible_proposer_sends_apc_prefix_and_applies_diff(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    target = repo / "worker/zstep_worker/runtime_loop.py"
    target.parent.mkdir(parents=True)
    target.write_text("MAX_STEPS = 4\n\n", encoding="utf-8")
    big_first = repo / "crates/zs-scheduler/src/serving.rs"
    big_first.parent.mkdir(parents=True)
    big_first.write_text("fn step() {}\n" * 11_500, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    target.write_text("MAX_STEPS = 8\n\n", encoding="utf-8")
    diff_text = subprocess.run(
        ["git", "diff"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    assert diff_text.endswith(" \n"), "expected trailing blank context line in diff"
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, check=True)

    captured_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "json": json.loads(request.content.decode("utf-8")),
            }
        )
        completion = (
            "Applying speculative graph width increase:\n"
            f"```diff\n{diff_text}```\n"
            'EVOLITH_PROPOSAL_JSON: {"hypothesis": "Self-hosted zstep proposal", '
            '"niche_key": "(plane=l2_scheduler,focus=spec_decode_verify_graph)", '
            '"patch_summary": "Bump MAX_STEPS"}\n'
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": completion}}]},
        )

    task = make_task(
        tmp_path,
        proposer_backend="openai",
        proposer_api_base="http://zstep.local:43537/v1",
        proposer_model="DeepSeek-V4.1-Flash",
    )
    built = build_proposer(task)
    assert isinstance(built, OpenAICompatibleProposer)

    proposer = OpenAICompatibleProposer(
        task,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    out = proposer.propose(repo, "Optimize runtime_loop", tmp_path / "ev-openai")

    assert target.read_text(encoding="utf-8") == "MAX_STEPS = 8\n\n"
    assert len(captured_requests) == 1
    assert captured_requests[0]["url"] == "http://zstep.local:43537/v1/chat/completions"
    assert captured_requests[0]["auth"] == "Bearer test-key"
    req_json = captured_requests[0]["json"]
    assert isinstance(req_json, dict)
    msgs = req_json["messages"]
    assert "[Parent Source Snapshot — Deterministic APC Prefix]" in msgs[1]["content"]
    assert "[TRUNCATED AT LINE" in msgs[1]["content"]
    assert "=== FILE: worker/zstep_worker/runtime_loop.py ===" in msgs[1]["content"]
    meta = parse_proposal_metadata(out)
    assert meta.hypothesis == "Self-hosted zstep proposal"
    assert extract_unified_diff(out).startswith(b"diff --git ")
