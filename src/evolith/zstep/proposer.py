from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

from evolith.zstep.task import ProposalMetadata, ZStepRSITask

_BLOCKED_EXACT_ENV = frozenset(
    {
        "SSHPASS",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "SSH_CONNECTION",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
)
_BLOCKED_SUFFIXES = (
    "_PASSWORD",
    "_TOKEN",
    "_API_KEY",
    "_SECRET",
    "_CREDENTIAL",
    "_CREDENTIALS",
)
_PROPOSAL_JSON_RE = re.compile(r"EVOLITH_PROPOSAL_JSON\s*:\s*(\{.*\})")


def _is_credential(name: str) -> bool:
    upper = name.upper()
    return upper in _BLOCKED_EXACT_ENV or upper.endswith(_BLOCKED_SUFFIXES)


def parse_proposal_metadata(
    output: str,
    changed_files: tuple[str, ...] = (),
    changed_lines: int = 0,
) -> ProposalMetadata:
    default_summary = (
        f"{', '.join(changed_files)} ({changed_lines} LOC)"
        if changed_files
        else "Source patch"
    )
    matches = _PROPOSAL_JSON_RE.findall(output)
    if matches:
        try:
            raw = json.loads(matches[-1])
            if isinstance(raw, dict):
                return ProposalMetadata(
                    hypothesis=str(raw.get("hypothesis") or "").strip()
                    or "Focused runtime/kernel throughput optimization",
                    niche_key=str(
                        raw.get("niche_key")
                        or raw.get("niche")
                        or "(plane=auto,target=deepseek_v41)"
                    ).strip(),
                    patch_summary=str(
                        raw.get("patch_summary") or raw.get("summary") or default_summary
                    ).strip(),
                )
        except ValueError:
            pass

    # Fallback: extract first concise meaningful line from proposer output
    hypothesis = "Focused runtime/kernel throughput optimization"
    for line in output.splitlines():
        cleaned = line.strip().lstrip("•*- ").strip()
        if 15 <= len(cleaned) <= 220 and not cleaned.startswith(("EVOLITH_", "{", "}")):
            hypothesis = cleaned
            break
    return ProposalMetadata(
        hypothesis=hypothesis,
        niche_key="(plane=auto,target=deepseek_v41)",
        patch_summary=default_summary,
    )


class KimiCodeProposer:
    def __init__(self, task: ZStepRSITask, executable: str = "kimi") -> None:
        self.task = task
        self.executable = executable

    def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
        env = {key: value for key, value in os.environ.items() if not _is_credential(key)}
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            [self.executable, "-p", prompt],
            cwd=candidate_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.task.local_timeout_sec,
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "proposer.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(f"Kimi proposer exited {result.returncode}")
        return result.stdout
