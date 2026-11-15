from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import httpx

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


_FENCED_DIFF_RE = re.compile(
    r"```(?:diff|patch)?\s*\n(diff --git .*?)```",
    flags=re.DOTALL,
)
_RAW_DIFF_RE = re.compile(
    r"(^diff --git .*?)(?=^EVOLITH_PROPOSAL_JSON\s*:|\Z)",
    flags=re.DOTALL | re.MULTILINE,
)

_SELF_HOSTED_SYSTEM_PROMPT = (
    "You are Evolith's Self-Referential Inference Engine Optimizer. "
    "Output a clean, minimal unified git diff (`diff --git a/... b/...`) inside a ```diff``` "
    "fenced block that applies cleanly with `git apply`, followed on the final line by:\n"
    'EVOLITH_PROPOSAL_JSON: {"hypothesis": "...", "niche_key": "...", "patch_summary": "..."}\n'
    "Never modify files outside the allowlisted roots."
)


def extract_unified_diff(response_text: str) -> bytes:
    """Extracts a `git apply`-ready unified diff without stripping blank context spaces."""
    fenced = _FENCED_DIFF_RE.findall(response_text)
    raw_diff = fenced[-1] if fenced else ""
    if not raw_diff:
        raw_match = _RAW_DIFF_RE.search(response_text)
        if raw_match:
            raw_diff = raw_match.group(1)
    # Preserve trailing single-space lines (`" \n"`) that represent unchanged blank lines
    cleaned = raw_diff.lstrip("\r\n").rstrip("\r\n")
    if not cleaned.strip():
        raise ValueError("Model completion did not contain a `diff --git` unified patch")
    return (cleaned + "\n").encode("utf-8")


def _collect_apc_aligned_context(
    candidate_dir: Path,
    allowed_roots: tuple[str, ...],
    max_file_chars: int = 64_000,
    max_total_chars: int = 160_000,
) -> str:
    """
    Serializes allowlisted parent source files in deterministic sorted order so
    consecutive RSI generations share an identical token prefix on `zstep` APC
    (`64`/`128`-token nodes) and Nebula `NEBULA_KV_HASH_SEED` affinity routing.
    """
    chunks: list[str] = ["[Parent Source Snapshot — Deterministic APC Prefix]"]
    used = 0
    files: list[Path] = []
    for rel in sorted(allowed_roots):
        target = candidate_dir / rel
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(
                sorted(
                    p
                    for p in target.rglob("*")
                    if p.is_file() and p.suffix in {".py", ".rs"}
                )
            )
    seen: set[str] = set()
    for path in files:
        rel_str = path.relative_to(candidate_dir).as_posix()
        if rel_str in seen or rel_str.startswith("worker/tests/"):
            continue
        seen.add(rel_str)
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_file_chars:
            cut_idx = text.rfind("\n", 0, max_file_chars)
            prefix = text[: cut_idx + 1] if cut_idx >= 0 else text[:max_file_chars]
            kept_lines = prefix.count("\n")
            total_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
            snippet = (
                f"{prefix}# ... [TRUNCATED AT LINE {kept_lines} OF {total_lines}] ...\n"
            )
        else:
            snippet = text
        block = f"=== FILE: {rel_str} ===\n{snippet}\n"
        if used + len(block) > max_total_chars:
            continue
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


class OpenAICompatibleProposer:
    """
    Self-Referential Ignition proposer targeting a live OpenAI-compatible endpoint
    such as `zstep serve` (`--enable-prefix-caching --speculative-method dspark|mtp|ngram`)
    or `nebula-gateway` (`/v1/chat/completions`).
    """

    def __init__(
        self,
        task: ZStepRSITask,
        *,
        api_base: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.task = task
        self.api_base = (api_base or task.proposer_api_base).rstrip("/")
        self.model = model or task.proposer_model
        self.api_key = (
            api_key
            if api_key is not None
            else os.environ.get("EVOLITH_PROPOSER_API_KEY", "")
        )
        self.max_tokens = max_tokens
        self.transport = transport

    def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        static_snapshot = _collect_apc_aligned_context(
            candidate_dir, self.task.allowed_roots
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": _SELF_HOSTED_SYSTEM_PROMPT},
                {"role": "user", "content": f"{static_snapshot}\n\n{prompt}"},
            ],
        }
        url = f"{self.api_base}/chat/completions"
        with httpx.Client(
            timeout=float(self.task.local_timeout_sec),
            transport=self.transport,
        ) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            body = resp.json()

        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError("OpenAI proposer response missing `choices[0]`")
        content = str((choices[0].get("message") or {}).get("content") or "")
        (evidence_dir / "proposer.log").write_text(content, encoding="utf-8")

        patch_bytes = extract_unified_diff(content)
        check = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=patch_bytes,
            cwd=candidate_dir,
            capture_output=True,
        )
        if check.returncode != 0:
            err = check.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"OpenAI proposer diff failed `git apply --check`: {err}")

        applied = subprocess.run(
            ["git", "apply", "-"],
            input=patch_bytes,
            cwd=candidate_dir,
            capture_output=True,
        )
        if applied.returncode != 0:
            err = applied.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"OpenAI proposer diff failed `git apply`: {err}")

        return content


def build_proposer(task: ZStepRSITask) -> KimiCodeProposer | OpenAICompatibleProposer:
    backend = task.proposer_backend.strip().lower()
    if backend == "kimi":
        return KimiCodeProposer(task)
    if backend == "openai":
        return OpenAICompatibleProposer(task)
    raise ValueError(
        f"Unknown proposer_backend `{task.proposer_backend}`; expected 'kimi' or 'openai'"
    )
