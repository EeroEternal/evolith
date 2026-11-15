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
        import signal

        env = {key: value for key, value in os.environ.items() if not _is_credential(key)}
        env["GIT_TERMINAL_PROMPT"] = "0"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        log_path = evidence_dir / "proposer.log"
        # Write directly to proposer.log instead of a pipe so backgrounded hook helpers
        # (e.g. terminal badge daemons) inheriting FDs cannot block pipe EOF after kimi exits.
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [self.executable, "-p", prompt],
                cwd=candidate_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                returncode = proc.wait(timeout=self.task.local_timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait(timeout=5)
                raise
        output = log_path.read_text(encoding="utf-8", errors="replace")
        if returncode:
            raise RuntimeError(f"Kimi proposer exited {returncode}")
        return output


class PatchProposer:
    """Direct patch-submission proposer without spawning external CLI agents.

    Applies a unified git diff inside Evolith's isolated candidate sandbox so
    every submitted patch still passes 100% of Coordinator allowlist, Spec-Pin,
    RedTeam, Exhaustive Dispatch Verifier, local Cargo/Ruff/Pytest, SHA-256
    freeze, 8xGPU screen, and 3-round interleaved verify gates.
    """

    def __init__(
        self,
        patch_path: Path,
        *,
        hypothesis: str,
        niche_key: str = "(plane=auto,target=deepseek_v41)",
        patch_summary: str = "",
    ) -> None:
        self.patch_path = Path(patch_path).expanduser().resolve()
        self.hypothesis = hypothesis.strip()
        self.niche_key = niche_key.strip() or "(plane=auto,target=deepseek_v41)"
        self.patch_summary = patch_summary.strip()

    def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
        if not self.patch_path.is_file():
            raise FileNotFoundError(f"Patch file not found: {self.patch_path}")
        patch_bytes = self.patch_path.read_bytes()
        if not patch_bytes.strip():
            raise ValueError(f"Patch file is empty: {self.patch_path}")

        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        check = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=patch_bytes,
            cwd=candidate_dir,
            capture_output=True,
        )
        if check.returncode != 0:
            err = check.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git apply --check failed: {err}")

        applied = subprocess.run(
            ["git", "apply", "-"],
            input=patch_bytes,
            cwd=candidate_dir,
            capture_output=True,
        )
        if applied.returncode != 0:
            err = applied.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git apply failed: {err}")

        payload = {
            "hypothesis": self.hypothesis,
            "niche_key": self.niche_key,
            "patch_summary": self.patch_summary,
        }
        output = (
            f"Direct PatchProposer applied {self.patch_path.name} ({len(patch_bytes)} bytes)\n"
            f"EVOLITH_PROPOSAL_JSON: {json.dumps(payload, ensure_ascii=False)}\n"
        )
        (evidence_dir / "proposer.log").write_text(output, encoding="utf-8")
        return output
