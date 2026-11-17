from __future__ import annotations

import ast
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
_DISABLE_GRAPH_SABOTAGE_RE = re.compile(
    r"\b(?:use_cuda_graph\s*=\s*False|disable_cuda_graph\s*=\s*True|max_graph_bucket\s*=\s*0)\b"
)
_TORCH_WHERE_SCALAR_OUT_RE = re.compile(
    r"\bwhere\s*\([^;\n,]+,\s*(?:-?\d+(?:\.\d+)?\s*,\s*[^,\n]+|[^,\n]+,\s*-?\d+(?:\.\d+)?)\s*,\s*out\s*="
)
_INT32_INDEX_TENSORS_RE = re.compile(
    r"\.(?:index_copy_|index_select)\s*\(\s*\d+\s*,\s*"
    r"(?:self\.)?(?:comp_bt_gpu|out_tokens_gpu|slot_ids_gpu|input_ids_gpu)\b"
    r"(?:\s*\[[^\]]*\])*"
    r"(?!\s*\[|\s*\.\s*(?:to\s*\(\s*(?:dtype\s*=\s*)?torch\.(?:int64|long)|long\s*\(\s*\)))"
)
_HOTPATH_FN_PREFIXES: tuple[str, ...] = (
    "execute",
    "_execute_step",
    "launch",
    "finish",
    "step",
    "run_step",
    "forward_step",
    "_forward",
    "_sample",
    "_speculative",
    "_run_decode",
    "_run_draft",
    "_run_verify",
    "draft",
    "_draft",
    "verify",
    "_verify",
)
_HOST_BUFFER_SUFFIXES: tuple[str, ...] = ("_cpu", "_np", "_host")
_HOTPATH_SYNC_ATTRS: frozenset[str] = frozenset({"item", "cpu", "tolist"})
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _parse_added_line_numbers_by_file(patch_bytes: bytes) -> dict[str, set[int]]:
    patch_text = patch_bytes.decode("utf-8", errors="replace")
    added_lines: dict[str, set[int]] = {}
    current_file: str | None = None
    new_lineno: int | None = None
    for raw_line in patch_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[6:].strip()
            added_lines.setdefault(current_file, set())
            new_lineno = None
        elif raw_line.startswith("+++ "):
            current_file = None
            new_lineno = None
        elif raw_line.startswith("@@ "):
            match = _HUNK_HEADER_RE.match(raw_line)
            new_lineno = int(match.group(1)) if match else None
        elif current_file is not None and new_lineno is not None:
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                added_lines[current_file].add(new_lineno)
                new_lineno += 1
            elif raw_line.startswith(" "):
                new_lineno += 1
    return added_lines


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
            "main.rs" in file_path or "lib.rs" in file_path or "runtime_loop.py" in file_path
        ):
            findings.append(
                RedTeamFinding(
                    category="anti_cheat",
                    severity="blocking",
                    file_path=file_path,
                    message="Patch hardcodes `triton_recompilations` to 0 instead of reporting real JIT counter",
                )
            )

        # Anti-cheat: degrading default CUDA Graph path to inflate relative speculative gains
        if (
            file_path.startswith("worker/zstep_worker/models/")
            or "runtime_loop.py" in file_path
        ) and _DISABLE_GRAPH_SABOTAGE_RE.search(joined):
            findings.append(
                RedTeamFinding(
                    category="anti_cheat",
                    severity="blocking",
                    file_path=file_path,
                    message="Patch disables CUDA Graphs (`use_cuda_graph=False` / `max_graph_bucket=0`) on model runner path (baseline-sabotage hazard)",
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

        # Category 1b: indices_offsets (PyTorch int32 index tensor passed to index_copy_/index_select, g0004)
        if file_path.endswith(".py") and _INT32_INDEX_TENSORS_RE.search(joined):
            findings.append(
                RedTeamFinding(
                    category="indices_offsets",
                    severity="blocking",
                    file_path=file_path,
                    message="`index_copy_`/`index_select` called with `int32` device buffer without `.to(torch.int64)` / `.long()` cast (raises CUDA RuntimeError)",
                )
            )

        # Category 2: value_ranges (PyTorch torch.where(..., out=...) scalar overload crash, g0005)
        if file_path.endswith(".py") and _TORCH_WHERE_SCALAR_OUT_RE.search(joined):
            findings.append(
                RedTeamFinding(
                    category="value_ranges",
                    severity="blocking",
                    file_path=file_path,
                    message="`torch.where(..., out=...)` called with Python numeric scalar operand; `out=` overload requires Tensor operands on CUDA",
                )
            )

        # Category 3 & 4: overwrite_war + execution_context (multi-stream WAR hazard, VS-003/VS-012)
        if file_path.startswith("worker/zstep_worker/") and _STREAM_CTOR_RE.search(joined):
            if not _STREAM_SYNC_RE.search(joined):
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

        # Category 7b: initial_values (zero-init SHM recurrent slot sentinel collision)
        if (
            ("compressor_slot_ids" in joined or "restore_slot" in joined)
            and ">= 0" in joined
            and "num_recurrent_checkpoints" not in all_added_blob
        ):
            findings.append(
                RedTeamFinding(
                    category="initial_values",
                    severity="warning",
                    file_path=file_path,
                    message="Zero-initialized SHM `compressor_slot_ids`/`restore_slot >= 0` checked without `num_recurrent_checkpoints > 0` guard",
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


def _is_hotpath_function_name(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}_")
        for prefix in _HOTPATH_FN_PREFIXES
    )


def _is_host_buffer_expr(expr: ast.AST) -> bool:
    cur = expr
    while isinstance(cur, ast.Subscript):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id.endswith(_HOST_BUFFER_SUFFIXES)
    if isinstance(cur, ast.Attribute):
        return cur.attr.endswith(_HOST_BUFFER_SUFFIXES)
    return False


def _check_hotpath_ast_sync_and_collectives(
    candidate_dir: Path, patch_stats: PatchStats
) -> list[RedTeamFinding]:
    """
    AST-scans newly added lines inside step/draft/verify hot-path methods to block:
      1. Host synchronization calls (`.item()`, `.cpu()`, `.tolist()`) on GPU tensors
         that break CUDAGraph capture and stall GPU execution;
      2. PCIe `dist.broadcast` / `torch.distributed.broadcast` regressions in steady-state step paths;
      3. `torch.where(..., out=...)` multi-line calls with Python numeric literal operands.
    """
    findings: list[RedTeamFinding] = []
    added_lines_map = _parse_added_line_numbers_by_file(patch_stats.patch)
    for rel_path, added_linenos in sorted(added_lines_map.items()):
        if not added_linenos or not rel_path.endswith(".py"):
            continue
        if not (
            rel_path.startswith("worker/zstep_worker/models/")
            or rel_path.startswith("worker/zstep_worker/kernels_sm120/")
            or rel_path
            in {
                "worker/zstep_worker/spec_decode.py",
                "worker/zstep_worker/runtime_loop.py",
            }
        ):
            continue
        full_path = candidate_dir / rel_path
        if not full_path.is_file():
            continue
        try:
            tree = ast.parse(full_path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue

        # Multi-line torch.where(..., out=...) check across the file's added lines
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or getattr(node, "lineno", 0) not in added_linenos:
                continue
            func = node.func
            is_where = (isinstance(func, ast.Attribute) and func.attr == "where") or (
                isinstance(func, ast.Name) and func.id == "where"
            )
            if is_where and any(kw.arg == "out" for kw in node.keywords) and len(node.args) >= 3:
                for operand in node.args[1:3]:
                    is_num = isinstance(operand, ast.Constant) and isinstance(
                        operand.value, (int, float)
                    )
                    is_neg_num = (
                        isinstance(operand, ast.UnaryOp)
                        and isinstance(operand.op, (ast.USub, ast.UAdd))
                        and isinstance(operand.operand, ast.Constant)
                        and isinstance(operand.operand.value, (int, float))
                    )
                    if is_num or is_neg_num:
                        if not any(
                            f.category == "value_ranges" and f.file_path == rel_path
                            for f in findings
                        ):
                            findings.append(
                                RedTeamFinding(
                                    category="value_ranges",
                                    severity="blocking",
                                    file_path=rel_path,
                                    message=f"Line {node.lineno}: `where(..., out=...)` called with numeric literal; `out=` overload requires Tensor operands",
                                )
                            )

        # Hot-path function body check for .item()/.cpu()/.tolist() and dist.broadcast
        for fn_node in ast.walk(tree):
            if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_hotpath_function_name(fn_node.name):
                continue
            for sub in ast.walk(fn_node):
                lineno = getattr(sub, "lineno", 0)
                if lineno not in added_linenos or not isinstance(sub, ast.Call):
                    continue
                func = sub.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in _HOTPATH_SYNC_ATTRS
                    and not _is_host_buffer_expr(func.value)
                ):
                    findings.append(
                        RedTeamFinding(
                            category="execution_context",
                            severity="blocking",
                            file_path=rel_path,
                            message=(
                                f"Line {lineno} in `{fn_node.name}`: host sync `.{func.attr}()` "
                                "introduced on GPU step/draft/verify hot path (breaks CUDAGraph & stalls TP8)"
                            ),
                        )
                    )
                elif isinstance(func, ast.Attribute) and func.attr == "broadcast":
                    owner_name = getattr(func.value, "id", "") or getattr(
                        func.value, "attr", ""
                    )
                    if owner_name in {"dist", "distributed"}:
                        findings.append(
                            RedTeamFinding(
                                category="execution_context",
                                severity="blocking",
                                file_path=rel_path,
                                message=(
                                    f"Line {lineno} in `{fn_node.name}`: `dist.broadcast` introduced "
                                    "on non-P2P step hot path; use SHM all-rank reduction instead"
                                ),
                            )
                        )
    return findings


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
    findings.extend(_check_hotpath_ast_sync_and_collectives(candidate_dir, patch_stats))
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
