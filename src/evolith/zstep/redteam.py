from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from evolith.zstep.task import PatchStats, RedTeamFinding, RedTeamReport, ZStepRSITask

SEVEN_BOUNDARY_CATEGORIES: tuple[str, ...] = (
    "indices_offsets",
    "value_ranges",
    "overwrite_war",
    "execution_context",
    "async_lifetime",
    "aliases_views",
    "initial_values",
)


class RedTeamRejectedError(RuntimeError):
    pass


class PremiseStrikeError(RuntimeError):
    pass


def _hash_pinned_file(path: Path) -> str:
    if path.is_symlink():
        return f"__symlink__:{path.readlink()}"
    if not path.is_file():
        return "__missing__"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_spec_pins(root_dir: Path, pinned_files: tuple[str, ...]) -> dict[str, str]:
    """Computes coordinator-owned SHA-256 pins before an untrusted proposer runs."""
    return {rel: _hash_pinned_file(root_dir / rel) for rel in pinned_files}


def verify_spec_pins(
    candidate_dir: Path, expected_pins: dict[str, str]
) -> list[RedTeamFinding]:
    findings: list[RedTeamFinding] = []
    for rel, expected in sorted(expected_pins.items()):
        actual = _hash_pinned_file(candidate_dir / rel)
        if actual != expected:
            findings.append(
                RedTeamFinding(
                    category="anti_cheat",
                    severity="blocking",
                    file_path=rel,
                    message=(
                        f"Coordinator-pinned spec file `{rel}` hash changed "
                        f"(expected {expected[:12]}, got {actual[:12]})"
                    ),
                )
            )
    return findings


def _parse_added_lines_by_file(patch_bytes: bytes) -> dict[str, list[str]]:
    patch_text = patch_bytes.decode("utf-8", errors="replace")
    added_by_file: dict[str, list[str]] = {}
    current_file: str | None = None
    for raw_line in patch_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[6:].strip()
            added_by_file.setdefault(current_file, [])
        elif raw_line.startswith("+++ "):
            current_file = None
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            if current_file is not None:
                added_by_file[current_file].append(raw_line[1:])
    return added_by_file


_TRANSIENT_ASYNC_H2D_RE = re.compile(
    r"torch\.(?:tensor|as_tensor)\([^)]*\)\s*\.\s*(?:to|cuda)\([^)]*non_blocking\s*=\s*True"
)
_COPY_TRANSIENT_ASYNC_RE = re.compile(
    r"\.copy_\(\s*torch\.(?:tensor|as_tensor)\([^)]*\)\s*,\s*non_blocking\s*=\s*True"
)
_STREAM_CTOR_RE = re.compile(r"\b(?:torch\.)?cuda\.Stream\s*\(")
_STREAM_SYNC_RE = re.compile(
    r"\b(?:wait_stream|record_event|wait_event|cuda\.Event|torch\.cuda\.Event|synchronize)\b"
)
_RUST_UNCHECKED_RE = re.compile(
    r"\b(?:get_unchecked|get_unchecked_mut|unreachable_unchecked|from_raw_parts|from_raw_parts_mut)\b"
)
_SPOOF_RECOMP_RE = re.compile(
    r"triton_recompilations\s*[:=]\s*(?:Some\(\s*0\s*\)|0\b)"
)


def scan_patch_boundaries(patch_stats: PatchStats) -> list[RedTeamFinding]:
    """
    Deterministic Red-Team pass modeled on vstack's 7-category cross-boundary
    Consumer Scan + anti-reward-hacking rules. Inspects newly added diff hunks
    before burning 8xGPU remote time.
    """
    findings: list[RedTeamFinding] = []
    added_by_file = _parse_added_lines_by_file(patch_stats.patch)
    all_added_blob = "\n".join(
        line for lines in added_by_file.values() for line in lines
    )

    for file_path, added_lines in sorted(added_by_file.items()):
        if not added_lines or file_path.startswith("worker/tests/"):
            continue
        joined = "\n".join(added_lines)

        # Anti-cheat: spoofing Triton recompilation counters or skipping greedy gates
        if _SPOOF_RECOMP_RE.search(joined) and (
            "main.rs" in file_path or "runtime_loop.py" in file_path
        ):
            findings.append(
                RedTeamFinding(
                    category="anti_cheat",
                    severity="blocking",
                    file_path=file_path,
                    message="Patch hardcodes `triton_recompilations` to 0 instead of reporting real JIT counter",
                )
            )

        # Category 1: indices_offsets (unchecked Rust indexing in scheduler/server)
        if file_path.endswith(".rs") and _RUST_UNCHECKED_RE.search(joined):
            findings.append(
                RedTeamFinding(
                    category="indices_offsets",
                    severity="blocking",
                    file_path=file_path,
                    message="Unchecked Rust indexing/slice construction (`get_unchecked` / `from_raw_parts`) in scheduler/server risks silent KV/slot memory corruption",
                )
            )

        # Category 3 & 4: overwrite_war + execution_context (multi-stream WAR hazard, VS-003/VS-012)
        if file_path.startswith("worker/zstep_worker/") and _STREAM_CTOR_RE.search(joined):
            if not _STREAM_SYNC_RE.search(all_added_blob):
                findings.append(
                    RedTeamFinding(
                        category="overwrite_war",
                        severity="blocking",
                        file_path=file_path,
                        message="Auxiliary `cuda.Stream` introduced without cross-stream `Event`/`wait_stream` synchronization (cross-stream WAR hazard)",
                    )
                )

        # Category 4: execution_context (runtime Triton autotuning / recompilation hazard)
        if file_path.startswith("worker/zstep_worker/kernels_sm120/"):
            if "@triton.autotune" in joined:
                findings.append(
                    RedTeamFinding(
                        category="execution_context",
                        severity="blocking",
                        file_path=file_path,
                        message="Unguarded `@triton.autotune` in SM120 kernel path triggers runtime Triton JIT recompilations",
                    )
                )

        # Category 5: async_lifetime (ephemeral pageable tensor with non_blocking=True)
        if file_path.endswith(".py") and (
            _TRANSIENT_ASYNC_H2D_RE.search(joined)
            or _COPY_TRANSIENT_ASYNC_RE.search(joined)
        ):
            findings.append(
                RedTeamFinding(
                    category="async_lifetime",
                    severity="blocking",
                    file_path=file_path,
                    message="Transient inline `torch.tensor(...)` copied with `non_blocking=True` without persistent pinned buffer risks async lifetime/pageable hazard",
                )
            )

        # Category 6: aliases_views (audit warning on raw stride views)
        if file_path.endswith(".py") and "as_strided(" in joined:
            findings.append(
                RedTeamFinding(
                    category="aliases_views",
                    severity="warning",
                    file_path=file_path,
                    message="Raw `as_strided` view introduced; verify consumer mutation / contiguity assumptions across batch buckets",
                )
            )

        # Category 7: initial_values (audit warning on slot/page recycle mutations)
        if ("free_slots" in joined or "engram" in joined.lower()) and not any(
            tok in joined for tok in ("zero_", "fill_", "fill(", "clear(")
        ):
            findings.append(
                RedTeamFinding(
                    category="initial_values",
                    severity="warning",
                    file_path=file_path,
                    message="Slot/Engram state touched without explicit zero/reset in hunk; verify recycled slots cannot bleed across requests",
                )
            )

    return findings


def _check_tp8_device_token_agreement(
    candidate_dir: Path, patch_stats: PatchStats
) -> list[RedTeamFinding]:
    """
    vstack Pillar 4 (Agreement): if `deepseek_v41.py` consumes `STEP_FLAG_DEVICE_TOKENS`
    from `out_tokens_gpu`, all 8 TP ranks must hold identical sampled tokens — either by
    removing the `if (rank == 0)` guard in `shm_allreduce.py`'s `tp8_argmax_gather_shm`
    kernel or via an explicit cross-rank broadcast.
    """
    dsv41_rel = "worker/zstep_worker/models/deepseek_v41.py"
    shm_rel = "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    dsv41_path = candidate_dir / dsv41_rel
    shm_path = candidate_dir / shm_rel
    if not dsv41_path.is_file() or not shm_path.is_file():
        return []
    dsv41_src = dsv41_path.read_text(encoding="utf-8", errors="replace")
    if "STEP_FLAG_DEVICE_TOKENS" not in dsv41_src:
        return []
    if "dist.broadcast" in dsv41_src or "all_gather" in dsv41_src:
        return []
    shm_src = shm_path.read_text(encoding="utf-8", errors="replace")
    if re.search(
        r"if\s*\(\s*rank\s*==\s*0\s*\)\s*\{[^}]*out_top2_tokens\[",
        shm_src,
        flags=re.DOTALL,
    ):
        return [
            RedTeamFinding(
                category="agreement",
                severity="blocking",
                file_path=dsv41_rel,
                message=(
                    "`deepseek_v41.py` enables `STEP_FLAG_DEVICE_TOKENS`, but "
                    "`shm_allreduce.py` still gates `out_top2_tokens` behind `if (rank == 0)` "
                    "without TP8 broadcast — ranks 1..7 would feed zero/stale token IDs"
                ),
            )
        ]
    return []


def run_redteam_gate(
    candidate_dir: Path,
    patch_stats: PatchStats,
    task: ZStepRSITask,
    evidence_dir: Path,
    expected_pins: dict[str, str] | None = None,
) -> RedTeamReport:
    """
    Runs Coordinator Spec-Pin verification + 7-category Red-Team boundary scan,
    persists `redteam-report.json` (and `spec-pins.json`), and raises
    `RedTeamRejectedError` on any blocking finding.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    findings: list[RedTeamFinding] = []
    pins_ok = True
    if expected_pins is not None:
        (evidence_dir / "spec-pins.json").write_text(
            json.dumps(expected_pins, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pin_findings = verify_spec_pins(candidate_dir, expected_pins)
        if pin_findings:
            pins_ok = False
            findings.extend(pin_findings)

    findings.extend(scan_patch_boundaries(patch_stats))
    findings.extend(_check_tp8_device_token_agreement(candidate_dir, patch_stats))
    blocking = [f for f in findings if f.severity == "blocking"]
    report = RedTeamReport(
        passed=not blocking,
        spec_pins_verified=pins_ok,
        checked_categories=SEVEN_BOUNDARY_CATEGORIES,
        findings=findings,
    )
    (evidence_dir / "redteam-report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if blocking:
        summary = "; ".join(
            f"[{item.category}] {item.file_path}: {item.message}" for item in blocking
        )
        raise RedTeamRejectedError(f"Red-Team gate rejected patch: {summary}")
    return report
