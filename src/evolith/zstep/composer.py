"""Orthogonal Stepping-Stone Composer, Scout Probe Guard & Evening Gate Analyzer.

Implements Phase-3 autonomous evolution capabilities for Evolith × zstep:
1. ``find_orthogonal_stepping_stone_pairs`` & ``compose_patch_pair`` (`AV-2a`):
   Discovers near-miss positive candidates (`+1.5% ~ +2.99%` Verify gain, zero parity
   drift) under the same `(upstream_sha, parent_id)` with disjoint `changed_files`,
   and cleanly composes their unified git diffs in an isolated worktree.
2. ``extract_and_validate_scout_probe`` (`AV-2b`):
   Extracts an optional Stage-1 read-only microbenchmark script (`EVOLITH_SCOUT_PROBE`)
   from proposer output and AST-audits it before running ~35s `zstep-probe`.
3. ``cluster_remote_failures`` & ``validate_monotonic_gate`` (`AV-4`):
   Clusters recurring remote failures for Evening Gate promotion and enforces the
   strict monotonic hardening rule (100% catch on targeted failures ∧ 0% false
   positives on all historical `accepted` patches).
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable

from evolith.zstep.archive import ZStepArchive
from evolith.zstep.task import CandidateStatus

_PARITY_OR_CRASH_TOKENS: tuple[str, ...] = (
    "exact match is not 100%",
    "violated cross-mode invariance",
    "logit drift",
    "triton recompilations",
    "cuda error",
    "runtimeerror",
    "panicked at",
    "leak",
)

_FORBIDDEN_PROBE_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "requests",
        "httpx",
        "paramiko",
        "pty",
    }
)
_FORBIDDEN_PROBE_CALLS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnv",
        "spawnve",
        "execv",
        "execve",
        "execl",
        "rmtree",
    }
)


@dataclass(frozen=True)
class SteppingStonePair:
    left_id: str
    right_id: str
    parent_id: str | None
    upstream_sha: str
    left_gain: float
    right_gain: float
    additive_gain_estimate: float
    combined_loc: int
    left_files: tuple[str, ...]
    right_files: tuple[str, ...]
    left_hypothesis: str
    right_hypothesis: str


@dataclass(frozen=True)
class FailureCluster:
    category: str
    count: int
    candidate_ids: tuple[str, ...]
    sample_reason: str


def _parse_changed_files(raw: object, patch_bytes: bytes) -> tuple[str, ...]:
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return tuple(sorted({str(x) for x in parsed if str(x).strip()}))
        except ValueError:
            pass
    files: set[str] = set()
    for line in patch_bytes.decode("utf-8", errors="replace").splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
    return tuple(sorted(files))


def _has_parity_or_crash_failure(reason: str) -> bool:
    low = reason.lower()
    return any(tok in low for tok in _PARITY_OR_CRASH_TOKENS)


def find_orthogonal_stepping_stone_pairs(
    archive: ZStepArchive,
    *,
    upstream_sha: str = "",
    parent_id: str | None = None,
    min_gain: float = 0.015,
    max_gain: float = 0.0299,
) -> list[SteppingStonePair]:
    """Return ranked disjoint-file near-miss positive candidate pairs from `archive`."""
    eligible: list[dict[str, object]] = []
    for row in archive.list_candidates():
        if str(row.get("status")) != CandidateStatus.VERIFY_REJECTED.value:
            continue
        row_up = str(row.get("upstream_sha") or "")
        if upstream_sha and row_up and row_up != upstream_sha:
            continue
        row_parent = row.get("parent_id") or None
        if row_parent != parent_id:
            continue
        gain = float(row.get("geomean_gain") or 0.0)
        if not (min_gain <= gain <= max_gain):
            continue
        reason = str(row.get("failure_reason") or "")
        if _has_parity_or_crash_failure(reason):
            continue
        patch_bytes = bytes(row.get("patch") or b"")
        if not patch_bytes:
            continue
        files = _parse_changed_files(row.get("changed_files"), patch_bytes)
        if not files:
            continue
        eligible.append({**row, "_files": files, "_gain": gain})

    pairs: list[SteppingStonePair] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a = eligible[i]
            b = eligible[j]
            fa = set(a["_files"])  # type: ignore[arg-type]
            fb = set(b["_files"])  # type: ignore[arg-type]
            if not fa.isdisjoint(fb):
                continue
            ga = float(a["_gain"])  # type: ignore[arg-type]
            gb = float(b["_gain"])  # type: ignore[arg-type]
            est = (1.0 + ga) * (1.0 + gb) - 1.0
            pairs.append(
                SteppingStonePair(
                    left_id=str(a["candidate_id"]),
                    right_id=str(b["candidate_id"]),
                    parent_id=parent_id,
                    upstream_sha=upstream_sha or str(a.get("upstream_sha") or ""),
                    left_gain=ga,
                    right_gain=gb,
                    additive_gain_estimate=est,
                    combined_loc=int(a.get("changed_lines") or 0) + int(b.get("changed_lines") or 0),
                    left_files=tuple(a["_files"]),  # type: ignore[arg-type]
                    right_files=tuple(b["_files"]),  # type: ignore[arg-type]
                    left_hypothesis=str(a.get("hypothesis") or ""),
                    right_hypothesis=str(b.get("hypothesis") or ""),
                )
            )
    pairs.sort(key=lambda p: p.additive_gain_estimate, reverse=True)
    return pairs


def compose_patch_pair(base_dir: Path, patch_a: bytes, patch_b: bytes) -> bytes:
    """Apply two unified diffs in an isolated temporary git tree and return the combined diff."""
    git_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "evolith",
        "GIT_AUTHOR_EMAIL": "evolith@local",
        "GIT_COMMITTER_NAME": "evolith",
        "GIT_COMMITTER_EMAIL": "evolith@local",
    }
    with tempfile.TemporaryDirectory(prefix="evolith_compose_") as td:
        work = Path(td) / "tree"
        shutil.copytree(
            base_dir,
            work,
            ignore=shutil.ignore_patterns(".git", "target", "__pycache__", ".pytest_cache"),
        )
        subprocess.run(["git", "init", "-q"], cwd=work, env=git_env, check=True)
        subprocess.run(["git", "add", "-A"], cwd=work, env=git_env, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=work, env=git_env, check=True)

        for label, p_bytes in (("left", patch_a), ("right", patch_b)):
            chk = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=p_bytes,
                cwd=work,
                env=git_env,
                capture_output=True,
                check=False,
            )
            if chk.returncode != 0:
                err = chk.stderr.decode("utf-8", errors="replace").strip()
                raise ValueError(f"Patch composition conflict on {label} patch: {err}")
            subprocess.run(["git", "apply", "-"], input=p_bytes, cwd=work, env=git_env, check=True)

        subprocess.run(["git", "add", "-N", "."], cwd=work, env=git_env, check=True)
        diff_res = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        if not diff_res.stdout.strip():
            raise ValueError("Composed patch produced an empty diff")
        return diff_res.stdout


_SCOUT_BLOCK_RE = re.compile(
    r"EVOLITH_SCOUT_PROBE_BEGIN\s*\n(.*?)\nEVOLITH_SCOUT_PROBE_END",
    re.S,
)


def extract_and_validate_scout_probe(proposer_output: str) -> str | None:
    """Extract and AST-audit an optional read-only GPU microbenchmark script."""
    m = _SCOUT_BLOCK_RE.search(proposer_output)
    if m is None:
        return None
    raw_body = m.group(1).strip()
    # Strip optional markdown ```python fences inside block
    if raw_body.startswith("```"):
        lines = raw_body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_body = "\n".join(lines).strip()

    if not raw_body:
        raise ValueError("Empty EVOLITH_SCOUT_PROBE script body")
    if "--output" not in raw_body and "sys.argv" not in raw_body:
        raise ValueError("Scout probe script must accept `--output <json_path>`")

    try:
        tree = ast.parse(raw_body, filename="scout_probe.py")
    except SyntaxError as exc:
        raise ValueError(f"SyntaxError in scout probe script: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".", 1)[0]
                if root_mod in _FORBIDDEN_PROBE_MODULES:
                    raise ValueError(f"Forbidden module `{alias.name}` in scout probe")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_mod = node.module.split(".", 1)[0]
            if root_mod in _FORBIDDEN_PROBE_MODULES:
                raise ValueError(f"Forbidden module `{node.module}` in scout probe")
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_PROBE_CALLS:
                raise ValueError(f"Forbidden call `{node.attr}` in scout probe")
            if node.attr.startswith("__") and node.attr not in ("__name__", "__main__", "__init__"):
                raise ValueError(f"Forbidden dunder attribute `{node.attr}` in scout probe")
    return raw_body + "\n"


def cluster_remote_failures(archive: ZStepArchive) -> list[FailureCluster]:
    """Cluster historical remote Screen/Verify failures into recurring root-cause buckets."""
    buckets: dict[str, list[tuple[str, str]]] = {}
    for row in archive.list_candidates():
        st = str(row.get("status") or "")
        if st not in (
            CandidateStatus.SCREEN_REJECTED.value,
            CandidateStatus.VERIFY_REJECTED.value,
            CandidateStatus.LOCAL_REJECTED.value,
        ):
            continue
        cid = str(row.get("candidate_id") or "")
        reason = str(row.get("failure_reason") or "").strip()
        if not reason:
            continue
        low = reason.lower()
        if "out of memory" in low or "oom" in low or "pxb_ring" in low:
            cat = "hbm_oom_or_ring_budget"
        elif "triton recompilations" in low:
            cat = "triton_jit_recompilation"
        elif "exact match" in low or "invariance" in low or "logit drift" in low:
            cat = "greedy_parity_or_invariance_drift"
        elif "swa pool" in low or "swa_pool" in low:
            cat = "swa_pool_desync"
        elif "divisibility" in low or "eff_blocks" in low or "fused_ar" in low:
            cat = "fused_allreduce_guard_mismatch"
        elif "geomean" in low or "throughput" in low:
            cat = "insufficient_throughput_gain"
        else:
            cat = "other_runtime_rejection"
        buckets.setdefault(cat, []).append((cid, reason))

    counts = Counter({k: len(v) for k, v in buckets.items()})
    return [
        FailureCluster(
            category=cat,
            count=cnt,
            candidate_ids=tuple(cid for cid, _ in buckets[cat]),
            sample_reason=buckets[cat][-1][1],
        )
        for cat, cnt in counts.most_common()
    ]


def validate_monotonic_gate(
    archive: ZStepArchive,
    base_dir: Path,
    gate_fn: Callable[[Path], bool],
    *,
    must_reject_ids: tuple[str, ...],
) -> tuple[bool, str]:
    """
    Verify monotonic Evening Gate discipline (`AGENTS.md`):
    1. `gate_fn(tree)` MUST return `True` on `base_dir` and 100% of historical `accepted` trees;
    2. `gate_fn(tree)` MUST return `False` on 100% of `must_reject_ids`.
    """
    if not gate_fn(base_dir):
        return False, "Gate rejected clean baseline tree"

    by_id = {str(r["candidate_id"]): r for r in archive.list_candidates()}
    git_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }

    def _apply_and_check(patch_bytes: bytes) -> bool | None:
        with tempfile.TemporaryDirectory(prefix="evolith_monogate_") as td:
            work = Path(td) / "tree"
            shutil.copytree(
                base_dir,
                work,
                ignore=shutil.ignore_patterns(".git", "target", "__pycache__", ".pytest_cache"),
            )
            chk = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=patch_bytes,
                cwd=work,
                env=git_env,
                capture_output=True,
                check=False,
            )
            if chk.returncode != 0:
                return None
            subprocess.run(["git", "apply", "-"], input=patch_bytes, cwd=work, env=git_env, check=True)
            return bool(gate_fn(work))

    for row in archive.list_candidates():
        if str(row.get("status")) != CandidateStatus.ACCEPTED.value:
            continue
        pbytes = bytes(row.get("patch") or b"")
        if not pbytes:
            continue
        verdict = _apply_and_check(pbytes)
        if verdict is False:
            return False, f"False positive: gate rejected historical accepted candidate {row['candidate_id']}"

    for bad_id in must_reject_ids:
        if bad_id not in by_id:
            return False, f"Unknown target failure candidate {bad_id}"
        pbytes = bytes(by_id[bad_id].get("patch") or b"")
        verdict = _apply_and_check(pbytes)
        if verdict is not False:
            return False, f"False negative: gate failed to reject targeted bad candidate {bad_id}"

    return True, "Monotonic gate verified (0% accepted false-positives, 100% targeted kills)"
