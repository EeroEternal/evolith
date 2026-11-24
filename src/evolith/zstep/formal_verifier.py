"""vstack-style Formal Verification & Source-Calibrated Gate for Evolith × zstep.

Implements strict fail-stop (fail-closed) static & SMT verification anchored to
real candidate source files (supporting both split ``csrc/`` files in ``g0013+``
and legacy inline ``_CPP_SOURCE`` / ``_CUDA_SOURCE`` strings in ``g0001``–``g0012``),
plus honest capability boundaries for design-level protocol state machines:

1. ``STATIC_PXB_RING_BYTES_BOUND`` (Exact AST evaluation + consensus check, enforced):
   Evaluates ``HostMappedShmAllReduce._init_pxb_ipc`` directly from Python AST
   with zero silent fallback defaults. Verifies 4 KiB alignment, ``dist.all_gather``
   TP=8 allocation/open consensus before ``self._pxb_enabled = True``, and static
   PXB IPC ring footprint ``<= 180 MiB`` (catching historical ``g0001`` 320.125 MiB
   and ``g0002`` 240.125 MiB Rank-6 OOMs). Note: bounds the static IPC ring buffer
   only, not dynamic model activation allocations.
2. ``Z3_FUSED_AR_3WAY_SOUNDNESS`` (Z3 SMT-LIB2 cross-language proof, enforced):
   Extracts exact guards across Python ``can_fuse_add_rmsnorm`` (``.py``),
   C++ ``launch_shm_v5_[pxb_]allreduce_add_rmsnorm`` (``.cpp``), and CUDA
   ``launch_shm_v5_allreduce_add_rmsnorm_cuda`` + ``launch_shm_v5_allreduce_cuda``
   (``.cu``). Fails closed if any expected function or guard expression cannot be
   parsed. Proves over all valid ``(rows, hidden, blocks, threads, world_size, pxb)``:
   - ``PyCanFuse => CppPrecond`` (catching ``g0004`` / ``g0009`` ``eff_blocks % rows`` drift),
   - ``PyCanFuse => (py_eff_blocks == cu_fused_launch_blocks)``, and
   - ``cu_ar_small_pxb <=> cu_fused_small_pxb`` when dual-grid ``small_pxb`` is active
     on both launchers (catching ``g0010`` ``10240 vs 40960`` ``control`` lockstep crash).
3. ``SM_DOUBLE_BUFFER_RING_MODEL`` & ``SM_KV_PREFIX_SPEC_MODEL`` (Explicit-state BFS,
   ``MODEL_ONLY``, non-gating):
   Reference protocol state machines for async double-buffer slot parity and
   APC/speculative KV page conservation. Explicitly marked ``enforced=False`` and
   ``verdict="MODEL_ONLY"`` because candidate Rust/Python write sites are not yet
   writer-census anchored to state transitions.
"""

from __future__ import annotations

import ast
from collections import deque
import copy
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_SAFE_PXB_HBM_BYTES: int = 180 * 1024 * 1024  # 180 MiB static PXB IPC ring ceiling

# Ground-truth outcomes on real historical candidate trees in targets/zstep/candidates/
HISTORICAL_CANDIDATE_EXPECTATIONS: dict[str, tuple[bool, str]] = {
    "g0001-03408986": (False, "Rank 6 OOM: 4-slot PXB ring = 320.125 MiB (> 180 MiB)"),
    "g0002-a989db09": (False, "Rank 6 OOM: 4-slot PXB ring = 240.125 MiB (> 180 MiB)"),
    "g0003-b96b840f": (True, "Accepted (+3.72%): 2-slot PXB ring = 122.625 MiB"),
    "g0004-1363b6ec": (True, "Campaign-1 non-PXB candidate (shm_v5 without _init_pxb_ipc)"),
    "g0004-eac6c3f7": (False, "GPU smoke crash: Python eff_blocks=8 vs C++ raw blocks=32 divisibility"),
    "g0005-6b87fc7b": (False, "Propose-only tree carrying unfixed C++ raw blocks=32 divisibility bug"),
    "g0008-67b2dd20": (True, "Accepted (+5.67%, Campaign-1 non-PXB)"),
    "g0009-9da0950d": (False, "GPU smoke crash: C++ raw blocks=32 divisibility + CUDA 10240 drift"),
    "g0010-47b2aec5": (False, "GPU smoke crash: main lockstep broken (CUDA AR 10240 vs Fused 40960)"),
    "g0011-bcaef1a3": (True, "Accepted (+5.19%): 3-way 40960 small_pxb alignment"),
    "g0012-243da173": (True, "Accepted (+5.93%): 2-slot 162.625 MiB PXB ring"),
    "g0015-7a608d48": (True, "Accepted (+7.76%): 32..56 blocks range aligned across PY/CPP/CU"),
    "g0018-dc4581de": (True, "Accepted (+7.83%): pxb_bfp=0 default 40960 lockstep preserved"),
}


class FormalVerificationError(RuntimeError):
    pass


class ClauseResult(BaseModel):
    clause_id: str
    pillar: str
    method: str
    enforced: bool = True
    verdict: str = "PASS"
    passed: bool
    anchored_sites: list[str] = Field(default_factory=list)
    explored_states: int = 0
    detail: str | None = None
    counterexample: str | None = None


class MutantCalibrationRow(BaseModel):
    mutant_id: str
    mutation_kind: str = "real_source_edit"
    expected_clause: str
    model_verdict: str
    killed: bool
    violated_clauses: list[str] = Field(default_factory=list)
    witness: str | None = None


class HistoricalCandidateRow(BaseModel):
    candidate_id: str
    historical_note: str
    expected_pass: bool
    actual_pass: bool
    matched_expectation: bool
    violated_clauses: list[str] = Field(default_factory=list)
    witness: str | None = None


class FormalVerificationReport(BaseModel):
    passed: bool
    gate_verdict: Literal["CLOSED", "REJECTED", "INCONCLUSIVE", "NOT_APPLICABLE"] = "CLOSED"
    weakest_link: str = ""
    z3_available: bool
    z3_binary: str | None = None
    clauses: list[ClauseResult] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class MutationCalibrationReport(BaseModel):
    passed: bool
    baseline_passed: bool
    total_mutants: int
    killed_mutants: int
    kill_rate: float
    rows: list[MutantCalibrationRow] = Field(default_factory=list)
    historical_checked: int = 0
    historical_matched: int = 0
    historical_rows: list[HistoricalCandidateRow] = Field(default_factory=list)


def find_z3_binary() -> str | None:
    found = shutil.which("z3")
    if found:
        return found
    for fallback in ("/opt/homebrew/bin/z3", "/usr/local/bin/z3", "/usr/bin/z3"):
        if Path(fallback).is_file():
            return fallback
    return None


def _run_z3_smt2(
    assertions_smt2: str,
    z3_bin: str,
    *,
    get_value_expr: str = "",
    timeout_sec: float = 5.0,
) -> tuple[str, str]:
    """Run ``z3 -smt2 -in``; queries ``(get-value ...)`` only when ``sat``."""
    check_script = assertions_smt2.rstrip() + "\n(check-sat)\n"
    proc = subprocess.run(
        [z3_bin, "-smt2", "-in"],
        input=check_script,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    lines = [ln.strip() for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if ln.strip()]
    status = next((ln for ln in lines if ln in ("unsat", "sat", "unknown")), "error")
    if status == "sat" and get_value_expr:
        model_script = check_script + f"(get-value {get_value_expr})\n"
        mproc = subprocess.run(
            [z3_bin, "-smt2", "-in"],
            input=model_script,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        mlines = [
            ln.strip()
            for ln in mproc.stdout.splitlines()
            if ln.strip() and ln.strip() != "sat"
        ]
        return "sat", " ".join(mlines)
    return status, " ".join(lines)


# ---------------------------------------------------------------------------
# Strict Fail-Stop AST / C++ / CUDA Source Extractors (Split & Legacy Inline)
# ---------------------------------------------------------------------------


def _eval_int_expr(node: ast.AST, env: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _eval_int_expr(node.operand, env)
        return -inner if inner is not None else None
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left, env)
        right = _eval_int_expr(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_align_up"
        and len(node.args) == 2
        and not node.keywords
    ):
        val = _eval_int_expr(node.args[0], env)
        align = _eval_int_expr(node.args[1], env)
        if val is not None and align is not None and align > 0:
            return ((val + align - 1) // align) * align
    return None


def _extract_c_fn_body(src: str, fn_name: str) -> tuple[str, int] | None:
    """Extract function definition ``void <fn_name>(...) { ... }`` (ignoring ``;`` prototypes)."""
    pattern = re.compile(rf"void\s+{re.escape(fn_name)}\s*\([^;{{}}]*\)\s*\{{")
    m = pattern.search(src)
    if m is None:
        return None
    start_brace = src.find("{", m.start())
    depth = 0
    for idx in range(start_brace, len(src)):
        ch = src[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                lineno = src[: m.start()].count("\n") + 1
                return src[start_brace + 1 : idx], lineno
    return None


def _resolve_shm_sources(candidate_dir: Path) -> dict[str, Any]:
    """Resolve ``shm_allreduce.py`` plus split ``csrc/`` or inline ``_CPP_SOURCE``/``_CUDA_SOURCE``."""
    rel_py = "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    py_path = candidate_dir / rel_py
    py_text = py_path.read_text(encoding="utf-8")
    tree = ast.parse(py_text, filename=rel_py)

    module_consts: dict[str, int] = {}
    const_lines: dict[str, int] = {}
    inline_strings: dict[str, tuple[str, int]] = {}

    for node in tree.body:
        target_name: str | None = None
        val_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            val_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target_name = node.target.id
            val_node = node.value
        if target_name and val_node is not None:
            int_val = _eval_int_expr(val_node, module_consts)
            if int_val is not None:
                module_consts[target_name] = int_val
                const_lines[target_name] = node.lineno
            elif isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                inline_strings[target_name] = (val_node.value, node.lineno)

    rel_cpp_split = "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce.cpp"
    cpp_split_path = candidate_dir / rel_cpp_split
    if cpp_split_path.is_file():
        cpp_src: str | None = cpp_split_path.read_text(encoding="utf-8")
        rel_cpp = rel_cpp_split
        cpp_line_offset = 0
    elif "_CPP_SOURCE" in inline_strings:
        cpp_src, cpp_assign_line = inline_strings["_CPP_SOURCE"]
        rel_cpp = rel_py
        cpp_line_offset = cpp_assign_line - 1
    else:
        cpp_src = None
        rel_cpp = rel_cpp_split
        cpp_line_offset = 0

    rel_cu_split = "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu"
    cu_split_path = candidate_dir / rel_cu_split
    if cu_split_path.is_file():
        cu_src: str | None = cu_split_path.read_text(encoding="utf-8")
        rel_cu = rel_cu_split
        cu_line_offset = 0
    elif "_CUDA_SOURCE" in inline_strings:
        cu_src, cu_assign_line = inline_strings["_CUDA_SOURCE"]
        rel_cu = rel_py
        cu_line_offset = cu_assign_line - 1
    else:
        cu_src = None
        rel_cu = rel_cu_split
        cu_line_offset = 0

    shm_cls: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "HostMappedShmAllReduce":
            shm_cls = node
            break

    return {
        "py_path": py_path,
        "rel_py": rel_py,
        "py_text": py_text,
        "tree": tree,
        "shm_cls": shm_cls,
        "module_consts": module_consts,
        "const_lines": const_lines,
        "cpp_src": cpp_src,
        "rel_cpp": rel_cpp,
        "cpp_line_offset": cpp_line_offset,
        "cu_src": cu_src,
        "rel_cu": rel_cu,
        "cu_line_offset": cu_line_offset,
    }


def _parse_py_can_fuse_strict(py_text: str, shm_cls: ast.ClassDef, rel_py: str) -> tuple[dict[str, Any] | None, str | None]:
    fn_node: ast.FunctionDef | None = None
    has_pxb_init = False
    for item in shm_cls.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == "_init_pxb_ipc":
                has_pxb_init = True
            elif item.name == "can_fuse_add_rmsnorm":
                fn_node = item

    if fn_node is None:
        return None, f"{rel_py}:{shm_cls.lineno} missing HostMappedShmAllReduce.can_fuse_add_rmsnorm"

    fn_src = ast.get_source_segment(py_text, fn_node) or ""
    anchor = f"{rel_py}:{fn_node.lineno}"

    m_vpr = re.search(r"launch_threads\s+or\s+vecs_per_row\s*>\s*(\d+)", fn_src)
    if m_vpr is None:
        return None, f"{anchor} cannot parse max vecs_per_row cap in can_fuse_add_rmsnorm"
    max_vecs_per_row = int(m_vpr.group(1))

    has_small_pxb = False
    blocks_lo = -1
    blocks_hi = -1
    small_vecs_cap = -1
    small_eff_blocks = -1

    if "small_pxb" in fn_src or "eff_blocks" in fn_src:
        m_range = re.search(
            r"small_pxb\s*=\s*\(?self\._pxb_enabled\s+and\s+(\d+)\s*<=\s*self\.blocks\s*<=\s*(\d+)\s+and\s+num_vecs\s*<=\s*(\d+)\)?",
            fn_src,
        )
        m_eq = re.search(
            r"small_pxb\s*=\s*\(?self\._pxb_enabled\s+and\s+self\.blocks\s*==\s*(\d+)\s+and\s+num_vecs\s*<=\s*(\d+)\)?",
            fn_src,
        )
        m_eff = re.search(
            r"eff_blocks\s*=\s*(\d+)\s+if\s+small_pxb\s+else\s+self\.blocks",
            fn_src,
        )
        if m_eff is None or (m_range is None and m_eq is None):
            return None, f"{anchor} unparseable small_pxb/eff_blocks expression in can_fuse_add_rmsnorm"
        has_small_pxb = True
        small_eff_blocks = int(m_eff.group(1))
        if m_range is not None:
            blocks_lo = int(m_range.group(1))
            blocks_hi = int(m_range.group(2))
            small_vecs_cap = int(m_range.group(3))
        else:
            assert m_eq is not None
            blocks_lo = int(m_eq.group(1))
            blocks_hi = blocks_lo
            small_vecs_cap = int(m_eq.group(2))

    div_var: str | None = None
    if re.search(r"rows\s*<\s*eff_blocks\s+and\s*\(?\s*eff_blocks\s*%\s*rows\s*\)?\s*!=\s*0", fn_src):
        div_var = "py_eff_blocks"
    elif re.search(r"rows\s*<\s*self\.blocks\s+and\s*\(?\s*self\.blocks\s*%\s*rows\s*\)?\s*!=\s*0", fn_src):
        div_var = "blocks"
    elif re.search(r"%\s*rows\b|\brows\s*%", fn_src):
        return None, f"{anchor} unrecognized rows divisibility guard in can_fuse_add_rmsnorm"

    has_world_mod = bool(
        re.search(
            r"(?:num_vecs|rows\s*\*\s*vecs_per_row)\s*>\s*self\.one_shot_max_vecs\s+and\s*\(?\s*vecs_per_row\s*%\s*self\.world_size\s*\)?\s*!=\s*0",
            fn_src,
        )
    )

    return (
        {
            "anchor": anchor,
            "has_pxb_init": has_pxb_init,
            "has_small_pxb": has_small_pxb,
            "blocks_lo": blocks_lo,
            "blocks_hi": blocks_hi,
            "small_vecs_cap": small_vecs_cap,
            "small_eff_blocks": small_eff_blocks,
            "max_vecs_per_row": max_vecs_per_row,
            "div_var": div_var,
            "has_world_mod": has_world_mod,
        },
        None,
    )


def _parse_cpp_fused_ar_strict(
    cpp_src: str | None,
    *,
    has_pxb_init: bool,
    rel_cpp: str,
    line_offset: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if cpp_src is None:
        return None, f"{rel_cpp}:1 missing C++ AllReduce launcher source"
    fn_name = (
        "launch_shm_v5_pxb_allreduce_add_rmsnorm"
        if has_pxb_init
        else "launch_shm_v5_allreduce_add_rmsnorm"
    )
    extracted = _extract_c_fn_body(cpp_src, fn_name)
    if extracted is None:
        return None, f"{rel_cpp}:1 missing C++ function {fn_name}"
    fn_body, rel_line = extracted
    anchor = f"{rel_cpp}:{line_offset + rel_line}"

    m_vpr = re.search(r"TORCH_CHECK\(\s*vecs_per_row\s*<=\s*(\d+)", fn_body)
    if m_vpr is None:
        return None, f"{anchor} missing TORCH_CHECK(vecs_per_row <= ...) in {fn_name}"
    max_vecs_per_row = int(m_vpr.group(1))

    has_small_pxb = False
    blocks_lo = -1
    blocks_hi = -1
    small_vecs_cap = -1
    small_eff_blocks = -1

    if "eff_blocks" in fn_body:
        m_range = re.search(
            r"eff_blocks\s*=\s*\(\s*my_hbm_ptr\s*!=\s*0\s*&&\s*blocks\s*>=\s*(\d+)\s*&&\s*blocks\s*<=\s*(\d+)\s*&&\s*(?:rows\s*\*\s*vecs_per_row|num_vecs)\s*<=\s*(\d+)\s*\)\s*\?\s*(\d+)\s*:\s*blocks",
            fn_body,
        )
        m_eq = re.search(
            r"eff_blocks\s*=\s*\(\s*my_hbm_ptr\s*!=\s*0\s*&&\s*blocks\s*==\s*(\d+)\s*&&\s*(?:rows\s*\*\s*vecs_per_row|num_vecs)\s*<=\s*(\d+)\s*\)\s*\?\s*(\d+)\s*:\s*blocks",
            fn_body,
        )
        if m_range is None and m_eq is None:
            return None, f"{anchor} unparseable eff_blocks definition in {fn_name}"
        has_small_pxb = True
        if m_range is not None:
            blocks_lo = int(m_range.group(1))
            blocks_hi = int(m_range.group(2))
            small_vecs_cap = int(m_range.group(3))
            small_eff_blocks = int(m_range.group(4))
        else:
            assert m_eq is not None
            blocks_lo = int(m_eq.group(1))
            blocks_hi = blocks_lo
            small_vecs_cap = int(m_eq.group(2))
            small_eff_blocks = int(m_eq.group(3))

    if re.search(r"if\s*\(\s*rows\s*<\s*eff_blocks\s*\)\s*\{\s*TORCH_CHECK\(\s*eff_blocks\s*%\s*rows\s*==\s*0", fn_body):
        div_var: str | None = "cpp_eff_blocks"
    elif re.search(r"if\s*\(\s*rows\s*<\s*blocks\s*\)\s*\{\s*TORCH_CHECK\(\s*blocks\s*%\s*rows\s*==\s*0", fn_body):
        div_var = "blocks"
    elif re.search(r"if\s*\(\s*rows\s*<\s*eff_blocks\s*\)\s*\{\s*TORCH_CHECK\(\s*blocks\s*%\s*rows\s*==\s*0", fn_body):
        div_var = "blocks"
    elif "% rows" in fn_body:
        return None, f"{anchor} unrecognized rows divisibility TORCH_CHECK in {fn_name}"
    else:
        div_var = None

    requires_world_mod = bool(
        re.search(
            r"if\s*\(\s*(?:rows\s*\*\s*vecs_per_row|num_vecs)\s*>\s*one_shot_max_vecs\s*\)\s*\{\s*TORCH_CHECK\(\s*vecs_per_row\s*%\s*world_size\s*==\s*0",
            fn_body,
        )
    )

    return (
        {
            "anchor": anchor,
            "fn_name": fn_name,
            "has_small_pxb": has_small_pxb,
            "blocks_lo": blocks_lo,
            "blocks_hi": blocks_hi,
            "small_vecs_cap": small_vecs_cap,
            "small_eff_blocks": small_eff_blocks,
            "max_vecs_per_row": max_vecs_per_row,
            "div_var": div_var,
            "requires_world_mod": requires_world_mod,
        },
        None,
    )


def _parse_cu_launcher_strict(
    cu_src: str | None,
    fn_name: str,
    *,
    rel_cu: str,
    line_offset: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse ``small_pxb`` and ``launch_blocks`` (under default ``pxb_bfp=0``, ``ar_cta_done_ptr=0``)."""
    if cu_src is None:
        return None, f"{rel_cu}:1 missing CUDA launcher source"
    extracted = _extract_c_fn_body(cu_src, fn_name)
    if extracted is None:
        return None, f"{rel_cu}:1 missing CUDA function {fn_name}"
    fn_body, rel_line = extracted
    anchor = f"{rel_cu}:{line_offset + rel_line}"

    if "small_pxb" not in fn_body and "launch_blocks" not in fn_body:
        return (
            {
                "anchor": anchor,
                "fn_name": fn_name,
                "has_small_pxb": False,
                "blocks_lo": -1,
                "blocks_hi": -1,
                "small_vecs_cap": -1,
                "small_launch_blocks": -1,
            },
            None,
        )

    small_lim_default: int | None = None
    m_lim = re.search(
        r"small_lim\s*=\s*\(\s*pxb_bfp\s*==\s*1\s*\)\s*\?\s*(\d+)\s*:\s*(\d+)\s*;",
        fn_body,
    )
    if m_lim is not None:
        small_lim_default = int(m_lim.group(2))

    m_sp_range = re.search(
        r"small_pxb\s*=\s*\(\s*(?:ar_cta_done_ptr\s*==\s*0\s*&&\s*)?my_hbm_ptr\s*!=\s*0\s*&&\s*blocks\s*>=\s*(\d+)\s*&&\s*blocks\s*<=\s*(\d+)\s*&&\s*num_vecs\s*<=\s*(\d+|small_lim)\s*\)\s*;",
        fn_body,
    )
    m_sp_eq = re.search(
        r"small_pxb\s*=\s*\(\s*(?:ar_cta_done_ptr\s*==\s*0\s*&&\s*)?my_hbm_ptr\s*!=\s*0\s*&&\s*blocks\s*==\s*(\d+)\s*&&\s*num_vecs\s*<=\s*(\d+|small_lim)\s*\)\s*;",
        fn_body,
    )
    m_lb = re.search(
        r"launch_blocks\s*=\s*small_pxb\s*\?\s*(\d+)\s*:\s*blocks\s*;",
        fn_body,
    )
    if m_lb is None or (m_sp_range is None and m_sp_eq is None):
        return None, f"{anchor} unparseable small_pxb/launch_blocks in {fn_name}"

    if m_sp_range is not None:
        blocks_lo = int(m_sp_range.group(1))
        blocks_hi = int(m_sp_range.group(2))
        cap_tok = m_sp_range.group(3)
    else:
        assert m_sp_eq is not None
        blocks_lo = int(m_sp_eq.group(1))
        blocks_hi = blocks_lo
        cap_tok = m_sp_eq.group(2)

    if cap_tok == "small_lim":
        if small_lim_default is None:
            return None, f"{anchor} small_lim referenced without parseable definition in {fn_name}"
        small_vecs_cap = small_lim_default
    else:
        small_vecs_cap = int(cap_tok)

    return (
        {
            "anchor": anchor,
            "fn_name": fn_name,
            "has_small_pxb": True,
            "blocks_lo": blocks_lo,
            "blocks_hi": blocks_hi,
            "small_vecs_cap": small_vecs_cap,
            "small_launch_blocks": int(m_lb.group(1)),
        },
        None,
    )


# ---------------------------------------------------------------------------
# Clause 1: Static PXB Ring Byte Bound & TP=8 Consensus (Exact AST Eval)
# ---------------------------------------------------------------------------


def verify_static_pxb_ring_bytes_bound(
    sources: dict[str, Any],
    *,
    max_hbm_bytes: int = MAX_SAFE_PXB_HBM_BYTES,
) -> ClauseResult:
    rel_py: str = sources["rel_py"]
    shm_cls: ast.ClassDef | None = sources["shm_cls"]
    module_consts: dict[str, int] = sources["module_consts"]
    const_lines: dict[str, int] = sources["const_lines"]

    if shm_cls is None:
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Conservation",
            method="ast_exact_eval",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_py}:1"],
            counterexample="HostMappedShmAllReduce class missing in shm_allreduce.py",
        )

    init_pxb_fn: ast.FunctionDef | None = None
    for item in shm_cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "_init_pxb_ipc":
            init_pxb_fn = item
            break

    if init_pxb_fn is None:
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Conservation",
            method="ast_exact_eval",
            enforced=True,
            verdict="NOT_APPLICABLE",
            passed=True,
            anchored_sites=[f"{rel_py}:{shm_cls.lineno}"],
            detail="Non-PXB HostMappedShmAllReduce (_init_pxb_ipc absent)",
        )

    anchors = [f"{rel_py}:{init_pxb_fn.lineno}"]
    if "DEFAULT_PXB_MAX_ELEMENTS" not in module_consts:
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Conservation",
            method="ast_exact_eval",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=anchors,
            counterexample="DEFAULT_PXB_MAX_ELEMENTS missing or non-constant in shm_allreduce.py",
        )
    anchors.append(f"{rel_py}:{const_lines['DEFAULT_PXB_MAX_ELEMENTS']}")
    if "NUM_PXB_RING_SLOTS" in const_lines:
        anchors.append(f"{rel_py}:{const_lines['NUM_PXB_RING_SLOTS']}")
    elif "NUM_RING_SLOTS" in const_lines:
        anchors.append(f"{rel_py}:{const_lines['NUM_RING_SLOTS']}")

    local_env = dict(module_consts)
    local_env["pxb_max_elements"] = module_consts["DEFAULT_PXB_MAX_ELEMENTS"]
    pxb_total_bytes: int | None = None
    pxb_bytes_line: int | None = None

    for stmt in init_pxb_fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            var_name = stmt.targets[0].id
            val = _eval_int_expr(stmt.value, local_env)
            if val is not None:
                local_env[var_name] = val
                if var_name == "pxb_total_bytes":
                    pxb_total_bytes = val
                    pxb_bytes_line = stmt.lineno
                    break
            elif var_name == "pxb_total_bytes":
                return ClauseResult(
                    clause_id="STATIC_PXB_RING_BYTES_BOUND",
                    pillar="Conservation",
                    method="ast_exact_eval",
                    enforced=True,
                    verdict="INCONCLUSIVE_PARSE_ERROR",
                    passed=False,
                    anchored_sites=[*anchors, f"{rel_py}:{stmt.lineno}"],
                    counterexample="Cannot statically evaluate pxb_total_bytes in _init_pxb_ipc",
                )

    if pxb_total_bytes is None or pxb_bytes_line is None:
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Conservation",
            method="ast_exact_eval",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=anchors,
            counterexample="pxb_total_bytes assignment not found in _init_pxb_ipc",
        )
    anchors.append(f"{rel_py}:{pxb_bytes_line}")

    # Verify TP=8 dist.all_gather consensus precedes self._pxb_enabled = True
    allgather_lines: list[int] = []
    pxb_enable_line: int | None = None
    for node in ast.walk(init_pxb_fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "all_gather"
        ):
            allgather_lines.append(node.lineno)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                and t.attr == "_pxb_enabled"
                for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ):
            pxb_enable_line = node.lineno

    if not allgather_lines or (pxb_enable_line is not None and min(allgather_lines) > pxb_enable_line):
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Agreement",
            method="ast_exact_eval",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=anchors,
            counterexample=(
                "_init_pxb_ipc sets self._pxb_enabled = True without prior dist.all_gather "
                "TP=8 rank consensus"
            ),
        )
    anchors.append(f"{rel_py}:{min(allgather_lines)}")

    mib = pxb_total_bytes / (1024 * 1024)
    cap_mib = max_hbm_bytes / (1024 * 1024)
    if pxb_total_bytes <= 0 or (pxb_total_bytes % 4096) != 0:
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Conservation",
            method="ast_exact_eval",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=anchors,
            counterexample=f"pxb_total_bytes={pxb_total_bytes} is non-positive or not 4 KiB page-aligned",
        )
    if pxb_total_bytes > max_hbm_bytes:
        return ClauseResult(
            clause_id="STATIC_PXB_RING_BYTES_BOUND",
            pillar="Conservation",
            method="ast_exact_eval",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=anchors,
            counterexample=(
                f"Static PXB IPC ring = {pxb_total_bytes} bytes ({mib:.3f} MiB) exceeds "
                f"ceiling {max_hbm_bytes} bytes ({cap_mib:.1f} MiB)"
            ),
        )
    return ClauseResult(
        clause_id="STATIC_PXB_RING_BYTES_BOUND",
        pillar="Conservation",
        method="ast_exact_eval",
        enforced=True,
        verdict="PASS",
        passed=True,
        anchored_sites=anchors,
        detail=f"pxb_total_bytes={pxb_total_bytes} ({mib:.3f} MiB <= {cap_mib:.1f} MiB) + all_gather consensus",
    )


# ---------------------------------------------------------------------------
# Clause 2: Seeded Differential Check Against Real Python Function (veriserve)
# ---------------------------------------------------------------------------


class _DummyTensor2D:
    __slots__ = ("ndim", "shape")

    def __init__(self, rows: int, hidden: int) -> None:
        self.ndim = 2
        self.shape = (rows, hidden)


class _DummyShmSelf:
    __slots__ = (
        "_use_shm_v5",
        "_pxb_enabled",
        "blocks",
        "threads",
        "one_shot_max_vecs",
        "world_size",
    )

    def __init__(
        self,
        *,
        pxb: bool,
        blocks: int,
        threads: int,
        one_shot_max_vecs: int,
        world_size: int = 8,
    ) -> None:
        self._use_shm_v5 = True
        self._pxb_enabled = pxb
        self.blocks = blocks
        self.threads = threads
        self.one_shot_max_vecs = one_shot_max_vecs
        self.world_size = world_size


def _eval_py_spec_descriptor(
    py_spec: dict[str, Any],
    *,
    rows: int,
    hidden: int,
    blocks: int,
    threads: int,
    one_shot_max_vecs: int,
    pxb: bool,
    world_size: int = 8,
) -> bool:
    """Pure Python evaluation of the exact SMT-LIB2 ``py_can_fuse`` descriptor."""
    if rows <= 0 or hidden <= 0 or (hidden % 8) != 0:
        return False
    vecs_per_row = hidden // 8
    num_vecs = rows * vecs_per_row
    py_small_pxb = (
        bool(
            pxb
            and py_spec["blocks_lo"] <= blocks <= py_spec["blocks_hi"]
            and num_vecs <= py_spec["small_vecs_cap"]
        )
        if py_spec["has_small_pxb"]
        else False
    )
    py_eff_blocks = py_spec["small_eff_blocks"] if py_small_pxb else blocks
    py_capped_threads = 512 if threads > 512 else threads
    py_eff_threads = 256 if py_small_pxb else py_capped_threads
    py_vecs_per_cta = (num_vecs + py_eff_blocks - 1) // py_eff_blocks
    py_launch_threads = (
        512
        if (py_eff_threads == 256 and (py_vecs_per_cta > 256 or vecs_per_row > 256))
        else py_eff_threads
    )
    if vecs_per_row > 3 * py_launch_threads or vecs_per_row > py_spec["max_vecs_per_row"]:
        return False
    div_var = py_spec["div_var"]
    if div_var == "py_eff_blocks":
        if rows < py_eff_blocks and (py_eff_blocks % rows) != 0:
            return False
    elif div_var == "blocks":
        if rows < blocks and (blocks % rows) != 0:
            return False
    if py_spec["has_world_mod"]:
        if num_vecs > one_shot_max_vecs and (vecs_per_row % world_size) != 0:
            return False
    return True


def _compile_real_can_fuse_fn(shm_cls: ast.ClassDef, rel_py: str) -> tuple[Any | None, str | None]:
    fn_node: ast.FunctionDef | None = None
    for item in shm_cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "can_fuse_add_rmsnorm":
            fn_node = item
            break
    if fn_node is None:
        return None, f"{rel_py}:{shm_cls.lineno} missing can_fuse_add_rmsnorm"

    fn_copy = copy.deepcopy(fn_node)
    fn_copy.decorator_list = []
    fn_copy.returns = None
    for sub in ast.walk(fn_copy):
        if isinstance(sub, ast.arg):
            sub.annotation = None
        elif isinstance(sub, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return (
                None,
                f"{rel_py}:{fn_node.lineno} forbidden statement `{type(sub).__name__}` inside can_fuse_add_rmsnorm",
            )
        elif isinstance(sub, ast.Attribute) and sub.attr.startswith("__"):
            return (
                None,
                f"{rel_py}:{fn_node.lineno} dunder attribute `{sub.attr}` forbidden inside can_fuse_add_rmsnorm",
            )
        elif isinstance(sub, ast.Name) and sub.id.startswith("__"):
            return (
                None,
                f"{rel_py}:{fn_node.lineno} dunder identifier `{sub.id}` forbidden inside can_fuse_add_rmsnorm",
            )

    mod = ast.Module(body=[fn_copy], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, Any] = {}
    try:
        code_obj = compile(mod, filename=rel_py, mode="exec")
        exec(code_obj, {"__builtins__": {"int": int, "bool": bool, "min": min, "max": max}}, ns)
    except Exception as exc:
        return None, f"{rel_py}:{fn_node.lineno} failed to compile can_fuse_add_rmsnorm: {exc}"
    return ns.get("can_fuse_add_rmsnorm"), None


def _run_py_diff_worker(
    py_path_str: str,
    rel_py: str,
    py_spec: dict[str, Any],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    """Executed inside an isolated ``python -I`` child process with a hard wall-clock timeout."""
    py_text = Path(py_path_str).read_text(encoding="utf-8")
    tree = ast.parse(py_text, filename=rel_py)
    shm_cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HostMappedShmAllReduce"),
        None,
    )
    if shm_cls is None:
        return {"ok": False, "kind": "parse", "error": "HostMappedShmAllReduce missing"}

    real_fn, fn_err = _compile_real_can_fuse_fn(shm_cls, rel_py)
    if fn_err or real_fn is None:
        return {"ok": False, "kind": "parse", "error": fn_err or "Failed to compile can_fuse_add_rmsnorm"}

    edge_rows = (1, 2, 3, 4, 7, 8, 9, 12, 16, 17, 24, 32, 48, 56, 64, 96, 128, 256, 512, 1024, 4096)
    hiddens = (2048, 4096, 5120, 7168, 8192, 12288, 16384)
    block_choices = (8, 16, 24, 32, 40, 48, 56, 64)
    thread_choices = (128, 256, 512, 1024)
    oneshot_choices = (64, 512, 4096, 16384, 65536)
    pxb_choices = (False, True) if py_spec["has_pxb_init"] else (False,)

    def _run_batch(spec_to_test: dict[str, Any], n_trials: int, batch_seed: int) -> tuple[int, str | None]:
        rng = random.Random(batch_seed)
        mismatches = 0
        first_witness: str | None = None
        for i in range(n_trials):
            rows = edge_rows[i % len(edge_rows)] if (i % 2 == 0) else rng.randint(1, 4096)
            hidden = rng.choice(hiddens)
            blocks = rng.choice(block_choices)
            threads = rng.choice(thread_choices)
            one_shot = rng.choice(oneshot_choices)
            pxb = rng.choice(pxb_choices)
            dummy_self = _DummyShmSelf(
                pxb=pxb,
                blocks=blocks,
                threads=threads,
                one_shot_max_vecs=one_shot,
            )
            dummy_in = _DummyTensor2D(rows, hidden)
            real_out = bool(real_fn(dummy_self, dummy_in))
            desc_out = _eval_py_spec_descriptor(
                spec_to_test,
                rows=rows,
                hidden=hidden,
                blocks=blocks,
                threads=threads,
                one_shot_max_vecs=one_shot,
                pxb=pxb,
            )
            if real_out != desc_out:
                mismatches += 1
                if first_witness is None:
                    first_witness = (
                        f"trial={i} (rows={rows}, hidden={hidden}, blocks={blocks}, "
                        f"threads={threads}, pxb={pxb}): real_py={real_out} != descriptor={desc_out}"
                    )
        return mismatches, first_witness

    try:
        bad_count, bad_witness = _run_batch(py_spec, trials, seed)
    except Exception as exc:
        return {
            "ok": False,
            "kind": "exec",
            "error": f"Runtime error executing real can_fuse_add_rmsnorm: {exc}",
        }

    if bad_count > 0:
        return {
            "ok": False,
            "kind": "mismatch",
            "bad_count": bad_count,
            "bad_witness": bad_witness,
        }

    pert_vpr = {**py_spec, "max_vecs_per_row": 512 if py_spec["max_vecs_per_row"] != 512 else 1536}
    pert_div = {
        **py_spec,
        "div_var": None if py_spec["div_var"] is not None else "blocks",
    }
    m_vpr, _ = _run_batch(pert_vpr, 300, seed + 1)
    m_div, _ = _run_batch(pert_div, 300, seed + 2)
    return {
        "ok": True,
        "m_vpr": m_vpr,
        "m_div": m_div,
    }


def verify_py_descriptor_differential(
    sources: dict[str, Any],
    *,
    trials: int = 3000,
    seed: int = 20261123,
    timeout_sec: float = 5.0,
) -> ClauseResult:
    """
    veriserve ``diff_test.py`` Gate (Subprocess-Isolated & Timeout-Bounded):
    1. Executes the candidate's REAL ``can_fuse_add_rmsnorm`` Python function inside
       an isolated ``python -I`` subprocess (with dunder/import AST ban and ``timeout_sec``)
       across ``trials`` (default 3,000) seeded configurations.
    2. If the descriptor diverges from real Python behavior, classifies as
       ``INCONCLUSIVE_MODEL_MISMATCH`` (blocking promotion without falsely claiming
       the candidate itself has a bug).
    3. Deliberately perturbs 2 descriptor parameters and requires ``mismatches > 0``
       on both so the differential harness itself cannot pass vacuously.
    """
    rel_py: str = sources["rel_py"]
    shm_cls: ast.ClassDef | None = sources["shm_cls"]
    if shm_cls is None:
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_py}:1"],
            counterexample="HostMappedShmAllReduce class missing in shm_allreduce.py",
        )

    py_spec, py_err = _parse_py_can_fuse_strict(sources["py_text"], shm_cls, rel_py)
    if py_err or py_spec is None:
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_py}:{shm_cls.lineno}"],
            counterexample=py_err or "Failed to parse can_fuse_add_rmsnorm descriptor",
        )

    # Pre-check AST dunder/import bans in parent before spawning subprocess
    _, ast_err = _compile_real_can_fuse_fn(shm_cls, rel_py)
    if ast_err is not None:
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"]],
            counterexample=ast_err,
        )

    src_root = str(Path(__file__).resolve().parents[2])
    payload = json.dumps(
        {
            "py_path": str(sources["py_path"]),
            "rel_py": rel_py,
            "py_spec": py_spec,
            "trials": trials,
            "seed": seed,
        }
    )
    child_code = (
        "import json, sys\n"
        f"sys.path.insert(0, {src_root!r})\n"
        "from evolith.zstep.formal_verifier import _run_py_diff_worker\n"
        "cfg = json.loads(sys.stdin.read())\n"
        "res = _run_py_diff_worker(cfg['py_path'], cfg['rel_py'], cfg['py_spec'], cfg['trials'], cfg['seed'])\n"
        "print(json.dumps(res))\n"
    )
    scrubbed_env = {"PATH": os.environ.get("PATH", "")}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", child_code],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=scrubbed_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_TIMEOUT",
            passed=False,
            anchored_sites=[py_spec["anchor"]],
            counterexample=(
                f"Candidate can_fuse_add_rmsnorm timed out after {timeout_sec:.1f}s "
                f"in isolated subprocess differential check"
            ),
        )

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["subprocess error"]
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"]],
            counterexample=f"Differential check subprocess failed: {err_tail[0]}",
        )

    out = json.loads(proc.stdout.strip())
    if not out.get("ok"):
        if out.get("kind") == "mismatch":
            bad_count = int(out["bad_count"])
            bad_witness = str(out["bad_witness"])
            return ClauseResult(
                clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
                pillar="Integrity",
                method="seeded_diff_3000",
                enforced=True,
                verdict="INCONCLUSIVE_MODEL_MISMATCH",
                passed=False,
                anchored_sites=[py_spec["anchor"]],
                explored_states=trials,
                counterexample=(
                    f"Verifier modeling mismatch (not a proven candidate bug): extracted SMT "
                    f"descriptor diverged from candidate can_fuse_add_rmsnorm in "
                    f"{bad_count}/{trials} trials; first: {bad_witness}"
                ),
            )
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"]],
            counterexample=str(out.get("error", "Differential check failed")),
        )

    m_vpr = int(out["m_vpr"])
    m_div = int(out["m_div"])
    if m_vpr == 0 or m_div == 0:
        return ClauseResult(
            clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
            pillar="Integrity",
            method="seeded_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"]],
            explored_states=trials,
            counterexample=(
                f"Perturbed-descriptor self-check failed to produce mismatches "
                f"(m_vpr={m_vpr}, m_div={m_div})"
            ),
        )

    return ClauseResult(
        clause_id="DIFF_PY_DESCRIPTOR_EQUIVALENCE",
        pillar="Integrity",
        method="seeded_diff_3000",
        enforced=True,
        verdict="CLOSED_DIFF",
        passed=True,
        anchored_sites=[py_spec["anchor"]],
        explored_states=trials,
        detail=(
            f"0/{trials} mismatches vs real Python fn (isolated subprocess); "
            f"perturbed descriptor self-check detected ({m_vpr}, {m_div}) mismatches"
        ),
    )


# ---------------------------------------------------------------------------
# Clause 3: 3-Way Cross-Boundary Z3 Soundness (.py + .cpp + .cu)
# ---------------------------------------------------------------------------


def _smt_small_pxb_expr(spec: dict[str, Any]) -> str:
    if not spec["has_small_pxb"]:
        return "false"
    return (
        f"(and pxb (>= blocks {spec['blocks_lo']}) "
        f"(<= blocks {spec['blocks_hi']}) (<= num_vecs {spec['small_vecs_cap']}))"
    )


def verify_z3_fused_ar_3way_soundness(
    sources: dict[str, Any],
    z3_bin: str | None,
    *,
    require_z3: bool = True,
) -> ClauseResult:
    rel_py: str = sources["rel_py"]
    shm_cls: ast.ClassDef | None = sources["shm_cls"]
    if shm_cls is None:
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_py}:1"],
            counterexample="HostMappedShmAllReduce class missing in shm_allreduce.py",
        )

    py_spec, py_err = _parse_py_can_fuse_strict(sources["py_text"], shm_cls, rel_py)
    if py_err or py_spec is None:
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_py}:{shm_cls.lineno}"],
            counterexample=py_err or "Failed to parse can_fuse_add_rmsnorm",
        )

    cpp_spec, cpp_err = _parse_cpp_fused_ar_strict(
        sources["cpp_src"],
        has_pxb_init=py_spec["has_pxb_init"],
        rel_cpp=sources["rel_cpp"],
        line_offset=sources["cpp_line_offset"],
    )
    if cpp_err or cpp_spec is None:
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"]],
            counterexample=cpp_err or "Failed to parse C++ launcher",
        )

    cu_fused_spec, cu_fused_err = _parse_cu_launcher_strict(
        sources["cu_src"],
        "launch_shm_v5_allreduce_add_rmsnorm_cuda",
        rel_cu=sources["rel_cu"],
        line_offset=sources["cu_line_offset"],
    )
    if cu_fused_err or cu_fused_spec is None:
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"], cpp_spec["anchor"]],
            counterexample=cu_fused_err or "Failed to parse CUDA fused launcher",
        )

    cu_ar_spec, cu_ar_err = _parse_cu_launcher_strict(
        sources["cu_src"],
        "launch_shm_v5_allreduce_cuda",
        rel_cu=sources["rel_cu"],
        line_offset=sources["cu_line_offset"],
    )
    if cu_ar_err or cu_ar_spec is None:
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[py_spec["anchor"], cpp_spec["anchor"], cu_fused_spec["anchor"]],
            counterexample=cu_ar_err or "Failed to parse CUDA AllReduce launcher",
        )

    anchors = [
        py_spec["anchor"],
        cpp_spec["anchor"],
        cu_fused_spec["anchor"],
        cu_ar_spec["anchor"],
    ]

    if z3_bin is None:
        if require_z3:
            raise FormalVerificationError(
                "Z3 binary is required (`require_z3=True`) for Z3_FUSED_AR_3WAY_SOUNDNESS"
            )
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=False,
            verdict="INCONCLUSIVE_NO_Z3",
            passed=True,
            anchored_sites=anchors,
            detail="Z3 binary not installed; SMT proof inconclusive (require_z3=False)",
        )

    pxb_domain_constraint = "" if py_spec["has_pxb_init"] else "(assert (not pxb))"

    py_small_smt = _smt_small_pxb_expr(py_spec)
    py_eff_smt = (
        f"(ite py_small_pxb {py_spec['small_eff_blocks']} blocks)"
        if py_spec["has_small_pxb"]
        else "blocks"
    )
    py_div_smt = (
        f"(=> (< rows {py_spec['div_var']}) (= (mod {py_spec['div_var']} rows) 0))"
        if py_spec["div_var"] is not None
        else "true"
    )
    py_wmod_smt = (
        "(=> (> num_vecs one_shot_max_vecs) (= (mod vecs_per_row world_size) 0))"
        if py_spec["has_world_mod"]
        else "true"
    )

    cpp_small_smt = _smt_small_pxb_expr(cpp_spec)
    cpp_eff_smt = (
        f"(ite cpp_small_pxb {cpp_spec['small_eff_blocks']} blocks)"
        if cpp_spec["has_small_pxb"]
        else "blocks"
    )
    cpp_div_smt = (
        f"(=> (< rows {cpp_spec['div_var']}) (= (mod {cpp_spec['div_var']} rows) 0))"
        if cpp_spec["div_var"] is not None
        else "true"
    )
    cpp_wmod_smt = (
        "(=> (> num_vecs one_shot_max_vecs) (= (mod vecs_per_row world_size) 0))"
        if cpp_spec["requires_world_mod"]
        else "true"
    )

    cu_fused_small_smt = _smt_small_pxb_expr(cu_fused_spec)
    cu_fused_lb_smt = (
        f"(ite cu_fused_small_pxb {cu_fused_spec['small_launch_blocks']} blocks)"
        if cu_fused_spec["has_small_pxb"]
        else "blocks"
    )
    cu_ar_small_smt = _smt_small_pxb_expr(cu_ar_spec)

    # Once both CUDA launchers implement dual-grid `small_pxb` (`flag_base = small_pxb ? ... : 0`),
    # default `pxb_bfp=0` calls MUST route identical shapes to `control[0..blocks)` vs `control[blocks..blocks+8)`!
    check_dual_lockstep = (
        "(= cu_ar_small_pxb cu_fused_small_pxb)"
        if (cu_ar_spec["has_small_pxb"] and cu_fused_spec["has_small_pxb"])
        else "true"
    )

    smt2 = f"""
(declare-const rows Int)
(declare-const hidden Int)
(declare-const blocks Int)
(declare-const threads Int)
(declare-const world_size Int)
(declare-const one_shot_max_vecs Int)
(declare-const pxb Bool)

; Valid runtime configuration domain on TP=8 SM120
(assert (and (>= rows 1) (<= rows 4096)))
(assert (or (= hidden 2048) (= hidden 4096) (= hidden 5120) (= hidden 7168) (= hidden 8192)))
(assert (and (>= blocks 8) (<= blocks 64)))
(assert (or (= threads 128) (= threads 256) (= threads 512) (= threads 1024)))
(assert (= world_size 8))
(assert (and (>= one_shot_max_vecs 64) (<= one_shot_max_vecs 65536) (= (mod one_shot_max_vecs 8) 0)))
{pxb_domain_constraint}

(define-fun vecs_per_row () Int (div hidden 8))
(define-fun num_vecs () Int (* rows vecs_per_row))

; 1. Python HostMappedShmAllReduce.can_fuse_add_rmsnorm
(define-fun py_small_pxb () Bool {py_small_smt})
(define-fun py_eff_blocks () Int {py_eff_smt})
(define-fun py_capped_threads () Int (ite (> threads 512) 512 threads))
(define-fun py_eff_threads () Int (ite py_small_pxb 256 py_capped_threads))
(define-fun py_vecs_per_cta () Int (div (+ num_vecs (- py_eff_blocks 1)) py_eff_blocks))
(define-fun py_launch_threads () Int
  (ite (and (= py_eff_threads 256) (or (> py_vecs_per_cta 256) (> vecs_per_row 256))) 512 py_eff_threads))

(define-fun py_can_fuse () Bool
  (and (<= vecs_per_row (* 3 py_launch_threads))
       (<= vecs_per_row {py_spec['max_vecs_per_row']})
       {py_div_smt}
       {py_wmod_smt}))

; 2. C++ launch_shm_v5_[pxb_]allreduce_add_rmsnorm TORCH_CHECK preconditions
(define-fun cpp_small_pxb () Bool {cpp_small_smt})
(define-fun cpp_eff_blocks () Int {cpp_eff_smt})
(define-fun cpp_precond () Bool
  (and (<= vecs_per_row {cpp_spec['max_vecs_per_row']})
       {cpp_div_smt}
       {cpp_wmod_smt}))

; 3. CUDA launch_shm_v5_allreduce_add_rmsnorm_cuda & launch_shm_v5_allreduce_cuda
(define-fun cu_fused_small_pxb () Bool {cu_fused_small_smt})
(define-fun cu_fused_launch_blocks () Int {cu_fused_lb_smt})
(define-fun cu_ar_small_pxb () Bool {cu_ar_small_smt})

(define-fun fused_sound () Bool
  (=> py_can_fuse
      (and cpp_precond
           (= py_eff_blocks cu_fused_launch_blocks)
           (=> (< rows cu_fused_launch_blocks) (= (mod cu_fused_launch_blocks rows) 0)))))

(define-fun dual_grid_lockstep () Bool {check_dual_lockstep})

; Negation of 3-way soundness & lockstep equivalence (unsat proves total soundness)
(assert (not (and fused_sound dual_grid_lockstep)))
"""
    status, witness = _run_z3_smt2(
        smt2,
        z3_bin,
        get_value_expr=(
            "(rows hidden blocks threads pxb py_eff_blocks cpp_eff_blocks "
            "cu_fused_launch_blocks cu_ar_small_pxb cu_fused_small_pxb)"
        ),
    )
    if status == "unsat":
        return ClauseResult(
            clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
            pillar="Integrity",
            method="z3_smt2",
            enforced=True,
            verdict="PROVED",
            passed=True,
            anchored_sites=anchors,
            detail="3-way .py/.cpp/.cu guard implication + dual-grid lockstep proved UNSAT",
        )
    return ClauseResult(
        clause_id="Z3_FUSED_AR_3WAY_SOUNDNESS",
        pillar="Integrity",
        method="z3_smt2",
        enforced=True,
        verdict="VIOLATION",
        passed=False,
        anchored_sites=anchors,
        counterexample=f"Z3 status={status}: {witness}",
    )


# ---------------------------------------------------------------------------
# Reference Protocol Models (Explicitly Non-Gating / MODEL_ONLY on Candidate Code)
# ---------------------------------------------------------------------------


def model_check_double_buffer_ring(
    *,
    num_slots: int = 2,
    max_inflight_steps: int = 2,
    enforce_slot_wait: bool = True,
    enforce_allgather_pxb_consensus: bool = True,
    max_steps: int = 5,
    enforced: bool = False,
    anchored_sites: list[str] | None = None,
) -> ClauseResult:
    """Reference explicit-state BFS protocol model for double-buffer parity & TP8 consensus."""
    init_slots = tuple(("free", -1) for _ in range(num_slots))
    init_states = []
    for all_ranks_alloc_ok in (True, False):
        if enforce_allgather_pxb_consensus:
            rank0_pxb = all_ranks_alloc_ok
            rank1_pxb = all_ranks_alloc_ok
        else:
            rank0_pxb = True
            rank1_pxb = all_ranks_alloc_ok
        init_states.append((0, 0, 0, init_slots, (rank0_pxb, rank1_pxb)))

    visited: set[tuple[Any, ...]] = set(init_states)
    queue: deque[tuple[tuple[Any, ...], tuple[str, ...]]] = deque((st, ()) for st in init_states)

    while queue:
        state, trace = queue.popleft()
        n_launch, n_gpu, n_fin, slots, (r0_pxb, r1_pxb) = state

        if r0_pxb != r1_pxb:
            return ClauseResult(
                clause_id="SM_DOUBLE_BUFFER_RING_MODEL",
                pillar="Agreement",
                method="explicit_bfs_model",
                enforced=enforced,
                verdict="VIOLATION",
                passed=False,
                anchored_sites=anchored_sites or [],
                explored_states=len(visited),
                counterexample=(
                    f"TP8 Agreement violated: rank0_pxb={r0_pxb} != rank1_pxb={r1_pxb} "
                    f"after trace={list(trace)}"
                ),
            )

        within_window = ((n_launch - n_fin) < max_inflight_steps) if enforce_slot_wait else True
        if n_launch < max_steps and within_window:
            s_idx = n_launch % num_slots
            slot_state, prev_step = slots[s_idx]
            can_acquire = (slot_state == "free") if enforce_slot_wait else True
            if can_acquire:
                if slot_state != "free":
                    return ClauseResult(
                        clause_id="SM_DOUBLE_BUFFER_RING_MODEL",
                        pillar="Isolation",
                        method="explicit_bfs_model",
                        enforced=enforced,
                        verdict="VIOLATION",
                        passed=False,
                        anchored_sites=anchored_sites or [],
                        explored_states=len(visited),
                        counterexample=(
                            f"WAR hazard on slot[{s_idx}]: launch(step={n_launch}) overwrote "
                            f"{slot_state}(step={prev_step}); trace={[*trace, f'launch({n_launch})']}"
                        ),
                    )
                next_slots = list(slots)
                next_slots[s_idx] = ("inflight_gpu", n_launch)
                nxt = (n_launch + 1, n_gpu, n_fin, tuple(next_slots), (r0_pxb, r1_pxb))
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, (*trace, f"launch({n_launch})")))

        if n_gpu < n_launch:
            s_idx = n_gpu % num_slots
            slot_state, owner_step = slots[s_idx]
            if slot_state == "inflight_gpu" and owner_step == n_gpu:
                next_slots = list(slots)
                next_slots[s_idx] = ("await_finish", n_gpu)
                nxt = (n_launch, n_gpu + 1, n_fin, tuple(next_slots), (r0_pxb, r1_pxb))
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, (*trace, f"gpu_done({n_gpu})")))

        if n_fin < n_gpu:
            s_idx = n_fin % num_slots
            slot_state, owner_step = slots[s_idx]
            if slot_state == "await_finish" and owner_step == n_fin:
                next_slots = list(slots)
                next_slots[s_idx] = ("free", -1)
                nxt = (n_launch, n_gpu, n_fin + 1, tuple(next_slots), (r0_pxb, r1_pxb))
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, (*trace, f"finish({n_fin})")))

    return ClauseResult(
        clause_id="SM_DOUBLE_BUFFER_RING_MODEL",
        pillar="Isolation",
        method="explicit_bfs_model",
        enforced=enforced,
        verdict="PASS" if enforced else "MODEL_ONLY",
        passed=True,
        anchored_sites=anchored_sites or [],
        explored_states=len(visited),
        detail="Design-level protocol model (not writer-census anchored to candidate code; non-gating)",
    )


def model_check_kv_prefix_spec_lifecycle(
    *,
    total_pages: int = 4,
    trim_kv_leases_on_spec_rewind: bool = True,
    guard_zero_init_restore_slot: bool = True,
    release_checkpoint_on_cancel: bool = True,
    enforced: bool = False,
    anchored_sites: list[str] | None = None,
) -> ClauseResult:
    """Reference explicit-state BFS protocol model for APC + speculative KV page leases."""
    init_states = [
        ("idle", False, 0, 0, 0, total_pages, 0, 0),
        ("idle", True, 0, 0, 0, total_pages, 0, -1),
    ]
    visited: set[tuple[Any, ...]] = set(init_states)
    queue: deque[tuple[tuple[Any, ...], tuple[str, ...]]] = deque((st, ()) for st in init_states)

    while queue:
        st, trace = queue.popleft()
        phase, apc_on, p_len, pref_p, kv_p, free_p, ckpt, raw_slot = st

        if p_len != pref_p + kv_p:
            return ClauseResult(
                clause_id="SM_KV_PREFIX_SPEC_MODEL",
                pillar="Conservation",
                method="explicit_bfs_model",
                enforced=enforced,
                verdict="VIOLATION",
                passed=False,
                anchored_sites=anchored_sites or [],
                explored_states=len(visited),
                counterexample=(
                    f"Page lease mismatch: page_ids_len={p_len} != prefix_lease({pref_p}) + "
                    f"kv_leases({kv_p}); trace={list(trace)}"
                ),
            )

        if free_p + pref_p + kv_p != total_pages:
            return ClauseResult(
                clause_id="SM_KV_PREFIX_SPEC_MODEL",
                pillar="Conservation",
                method="explicit_bfs_model",
                enforced=enforced,
                verdict="VIOLATION",
                passed=False,
                anchored_sites=anchored_sites or [],
                explored_states=len(visited),
                counterexample=(
                    f"Physical KV pool leak/overcommit: free({free_p}) + prefix({pref_p}) + "
                    f"kv({kv_p}) != total({total_pages}); trace={list(trace)}"
                ),
            )

        would_restore = (apc_on and raw_slot >= 0) if guard_zero_init_restore_slot else (raw_slot >= 0)
        if would_restore and not apc_on:
            return ClauseResult(
                clause_id="SM_KV_PREFIX_SPEC_MODEL",
                pillar="Isolation",
                method="explicit_bfs_model",
                enforced=enforced,
                verdict="VIOLATION",
                passed=False,
                anchored_sites=anchored_sites or [],
                explored_states=len(visited),
                counterexample=(
                    f"Zero-init SHM sentinel collision: restore triggered at raw_slot={raw_slot} "
                    f"with apc_enabled=False; trace={list(trace)}"
                ),
            )

        if phase == "retired" and (p_len != 0 or free_p != total_pages or ckpt != 0):
            return ClauseResult(
                clause_id="SM_KV_PREFIX_SPEC_MODEL",
                pillar="Finality",
                method="explicit_bfs_model",
                enforced=enforced,
                verdict="VIOLATION",
                passed=False,
                anchored_sites=anchored_sites or [],
                explored_states=len(visited),
                counterexample=(
                    f"Post-retirement leak: p_len={p_len}, free_pages={free_p}/{total_pages}, "
                    f"ckpt={ckpt}; trace={list(trace)}"
                ),
            )

        transitions: list[tuple[str, tuple[Any, ...]]] = []
        if phase == "idle" and free_p >= 2:
            transitions.append(
                (
                    "admit_prefill_2pages",
                    ("prefilled", apc_on, 2, 0, 2, free_p - 2, 1 if apc_on else 0, 0),
                )
            )
        elif phase == "prefilled":
            if apc_on and kv_p >= 1:
                transitions.append(
                    (
                        "promote_completed_prefix",
                        ("promoted", apc_on, p_len, pref_p + 1, kv_p - 1, free_p, ckpt, raw_slot),
                    )
                )
            rem_ckpt = 0 if release_checkpoint_on_cancel else ckpt
            transitions.append(
                ("cancel_prefill", ("retired", apc_on, 0, 0, 0, total_pages, rem_ckpt, -1))
            )
        elif phase == "promoted":
            if free_p >= 1:
                transitions.append(
                    (
                        "spec_draft_cross_page",
                        ("drafted", apc_on, p_len + 1, pref_p, kv_p + 1, free_p - 1, ckpt, raw_slot),
                    )
                )
            transitions.append(
                (
                    "compact_context_rewrite",
                    ("prefilled", apc_on, 1, 0, 1, total_pages - 1, ckpt, raw_slot),
                )
            )
            rem_ckpt = 0 if release_checkpoint_on_cancel else ckpt
            transitions.append(
                ("cancel_promoted", ("retired", apc_on, 0, 0, 0, total_pages, rem_ckpt, -1))
            )
        elif phase == "drafted":
            new_kv_p = (kv_p - 1) if trim_kv_leases_on_spec_rewind else kv_p
            new_free_p = (free_p + 1) if trim_kv_leases_on_spec_rewind else free_p
            transitions.append(
                (
                    "spec_reject_cross_page_rewind",
                    ("promoted", apc_on, p_len - 1, pref_p, new_kv_p, new_free_p, ckpt, raw_slot),
                )
            )
            rem_ckpt = 0 if release_checkpoint_on_cancel else ckpt
            transitions.append(
                ("cancel_drafted", ("retired", apc_on, 0, 0, 0, total_pages, rem_ckpt, -1))
            )

        for act_name, nxt in transitions:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, (*trace, act_name)))

    return ClauseResult(
        clause_id="SM_KV_PREFIX_SPEC_MODEL",
        pillar="Conservation",
        method="explicit_bfs_model",
        enforced=enforced,
        verdict="PASS" if enforced else "MODEL_ONLY",
        passed=True,
        anchored_sites=anchored_sites or [],
        explored_states=len(visited),
        detail="Design-level protocol model (not writer-census anchored to candidate code; non-gating)",
    )


# ---------------------------------------------------------------------------
# Top-Level Candidate Gate & Real Source / Historical Calibration
# ---------------------------------------------------------------------------


def run_formal_verification(
    candidate_dir: Path,
    evidence_dir: Path | None = None,
    *,
    require_z3: bool = False,
    expect_shm_allreduce: bool = False,
    diff_timeout_sec: float = 5.0,
) -> FormalVerificationReport:
    """Run fail-stop AST + subprocess-isolated differential + 3-way Z3 verification."""
    z3_bin = find_z3_binary()
    if z3_bin is None and require_z3:
        raise FormalVerificationError(
            "Formal verification INCONCLUSIVE: Z3 binary is required (`require_z3=True`) but not found"
        )

    clauses: list[ClauseResult] = []
    shm_py = candidate_dir / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"

    if shm_py.is_file():
        try:
            sources = _resolve_shm_sources(candidate_dir)
        except SyntaxError as exc:
            clauses.append(
                ClauseResult(
                    clause_id="STATIC_PXB_RING_BYTES_BOUND",
                    pillar="Integrity",
                    method="ast_exact_eval",
                    enforced=True,
                    verdict="INCONCLUSIVE_PARSE_ERROR",
                    passed=False,
                    anchored_sites=["worker/zstep_worker/kernels_sm120/shm_allreduce.py:1"],
                    counterexample=f"SyntaxError in shm_allreduce.py: {exc}",
                )
            )
        else:
            clauses.append(verify_static_pxb_ring_bytes_bound(sources))
            clauses.append(
                verify_py_descriptor_differential(sources, timeout_sec=diff_timeout_sec)
            )
            clauses.append(
                verify_z3_fused_ar_3way_soundness(
                    sources,
                    z3_bin,
                    require_z3=require_z3,
                )
            )
    elif expect_shm_allreduce:
        # Fail-stop: if shm_allreduce.py existed in baseline (or CLI explicitly verifies a zstep tree)
        # and is now missing, NEVER silently pass!
        clauses.append(
            ClauseResult(
                clause_id="STATIC_PXB_RING_BYTES_BOUND",
                pillar="Integrity",
                method="ast_exact_eval",
                enforced=True,
                verdict="INCONCLUSIVE_MISSING_FILE",
                passed=False,
                anchored_sites=["worker/zstep_worker/kernels_sm120/shm_allreduce.py:1"],
                counterexample="Missing expected file worker/zstep_worker/kernels_sm120/shm_allreduce.py",
            )
        )

    # Non-gating protocol design models (explicitly reported as MODEL_ONLY, enforced=False)
    clauses.append(model_check_double_buffer_ring(enforced=False))
    clauses.append(model_check_kv_prefix_spec_lifecycle(enforced=False))

    enforced_clauses = [c for c in clauses if c.enforced]
    has_violation = any(c.verdict == "VIOLATION" for c in enforced_clauses)
    has_enforced_inconclusive = any(
        c.verdict.startswith("INCONCLUSIVE") for c in enforced_clauses
    )
    has_optional_no_z3 = any(c.verdict == "INCONCLUSIVE_NO_Z3" for c in clauses)
    if not enforced_clauses and not has_optional_no_z3:
        gate_verdict: Literal["CLOSED", "REJECTED", "INCONCLUSIVE", "NOT_APPLICABLE"] = (
            "NOT_APPLICABLE"
        )
    elif has_violation:
        gate_verdict = "REJECTED"
    elif has_enforced_inconclusive or has_optional_no_z3:
        gate_verdict = "INCONCLUSIVE"
    else:
        gate_verdict = "CLOSED"

    weakest_link = (
        "Static PXB bound checks IPC ring bytes only (not dynamic activations); "
        "Python guard is 3000-trial differentially checked against real fn, while C++/CUDA guards "
        "are strict regex-extracted; SM_*_MODEL state machines are non-gating design models."
    )

    violations = [
        f"{c.clause_id} [{c.verdict}] ({c.pillar}): {c.counterexample}"
        for c in enforced_clauses
        if not c.passed
    ]
    report = FormalVerificationReport(
        passed=not violations,
        gate_verdict=gate_verdict,
        weakest_link=weakest_link,
        z3_available=z3_bin is not None,
        z3_binary=z3_bin,
        clauses=clauses,
        violations=violations,
    )
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "formal-verification.json").write_text(
            json.dumps(report.model_dump(), indent=2),
            encoding="utf-8",
        )
    if violations:
        raise FormalVerificationError(
            f"Formal verification {gate_verdict}: " + "; ".join(violations)
        )
    return report


def _copy_shm_tree(reference_dir: Path, dst_root: Path) -> None:
    rel_files = (
        "worker/zstep_worker/kernels_sm120/shm_allreduce.py",
        "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce.cpp",
        "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu",
    )
    for rel in rel_files:
        src = reference_dir / rel
        if src.is_file():
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _apply_exact_replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise FormalVerificationError(f"Calibration source pattern {old!r} not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


_FAILURE_CLAUSE_RE = re.compile(
    r"([A-Z0-9_]+)\s+\[(VIOLATION|INCONCLUSIVE_[A-Z0-9_]+|UNVERIFIED_[A-Z0-9_]+)\]"
)


def calibrate_historical_candidates(
    candidates_dir: Path,
    upstream_dir: Path | None = None,
) -> list[HistoricalCandidateRow]:
    """Run ``run_formal_verification(require_z3=True)`` against real historical candidate trees."""
    rows: list[HistoricalCandidateRow] = []
    targets: list[tuple[str, Path, bool, str]] = []
    if upstream_dir is not None and upstream_dir.is_dir():
        targets.append(("upstream", upstream_dir, True, "Current upstream zstep tree"))

    if candidates_dir.is_dir():
        for cand_id, (expected_pass, note) in sorted(HISTORICAL_CANDIDATE_EXPECTATIONS.items()):
            cand_path = candidates_dir / cand_id
            if cand_path.is_dir():
                targets.append((cand_id, cand_path, expected_pass, note))

    for cid, cpath, exp_pass, note in targets:
        try:
            rep = run_formal_verification(cpath, require_z3=True)
            actual_pass = rep.gate_verdict == "CLOSED"
            violated: list[str] = []
            witness: str | None = None
        except FormalVerificationError as exc:
            actual_pass = False
            msg = str(exc)
            violated = [m.group(1) for m in _FAILURE_CLAUSE_RE.finditer(msg)]
            witness = re.sub(r"^Formal verification (?:REJECTED|INCONCLUSIVE|failed):\s*", "", msg)
        rows.append(
            HistoricalCandidateRow(
                candidate_id=cid,
                historical_note=note,
                expected_pass=exp_pass,
                actual_pass=actual_pass,
                matched_expectation=(actual_pass == exp_pass),
                violated_clauses=violated,
                witness=witness,
            )
        )
    return rows


def calibrate_formal_mutations(
    reference_dir: Path,
    *,
    candidates_dir: Path | None = None,
) -> MutationCalibrationReport:
    """
    veriserve / vstack Source-Level & Differential Anti-Vacuity Gate:
    1. Verifies unmutated ``reference_dir`` achieves ``gate_verdict == "CLOSED"``.
    2. Applies 7 REAL source-file edits to copied candidate trees (missing pattern
       raises ``FormalVerificationError`` rather than counting as killed) and verifies
       end-to-end rejection / fail-closed ``INCONCLUSIVE`` on the predicted clause.
    3. When ``candidates_dir`` (or ``reference_dir.parent / 'candidates'``) exists,
       verifies ground-truth agreement across real historical candidate trees.
    """
    if find_z3_binary() is None:
        raise FormalVerificationError(
            "Z3 binary is required to run full formal source-mutation calibration"
        )

    base_report = run_formal_verification(reference_dir, require_z3=True)
    baseline_closed = base_report.passed and base_report.gate_verdict == "CLOSED"
    rows: list[MutantCalibrationRow] = []

    rel_py = "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    rel_cpp = "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce.cpp"
    rel_cu = "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu"
    is_split = (reference_dir / rel_cpp).is_file()

    def _run_source_mutant(
        mutant_id: str,
        expected_clause: str,
        expected_verdict: str,
        mutator: Any,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            mut_root = Path(td) / mutant_id
            _copy_shm_tree(reference_dir, mut_root)
            mutator(mut_root)
            try:
                rep = run_formal_verification(mut_root, require_z3=True)
                violated: list[str] = []
                verdict = rep.gate_verdict
                witness = None
                killed = False
            except FormalVerificationError as exc:
                msg = str(exc)
                matches = list(_FAILURE_CLAUSE_RE.finditer(msg))
                violated = [m.group(1) for m in matches]
                matching_verdicts = [m.group(2) for m in matches if m.group(1) == expected_clause]
                verdict = matching_verdicts[0] if matching_verdicts else (matches[0].group(2) if matches else "VIOLATION")
                witness = re.sub(r"^Formal verification (?:REJECTED|INCONCLUSIVE|failed):\s*", "", msg)
                killed = (expected_clause in violated) and (verdict == expected_verdict)
            rows.append(
                MutantCalibrationRow(
                    mutant_id=mutant_id,
                    mutation_kind="real_source_edit",
                    expected_clause=expected_clause,
                    model_verdict=verdict,
                    killed=killed,
                    violated_clauses=violated,
                    witness=witness,
                )
            )

    # M1: Real source edit in shm_allreduce.py -> 4-slot PXB ring (> 180 MiB OOM)
    _run_source_mutant(
        "M1_src_pxb_4slot_hbm_oom",
        "STATIC_PXB_RING_BYTES_BOUND",
        "VIOLATION",
        lambda root: _apply_exact_replace(
            root / rel_py, "NUM_PXB_RING_SLOTS = 2", "NUM_PXB_RING_SLOTS = 4"
        ),
    )

    # M2: Real source edit in shm_allreduce.py -> drop TP=8 all_gather consensus in _init_pxb_ipc
    def _drop_allgather(root: Path) -> None:
        p = root / rel_py
        txt = p.read_text(encoding="utf-8")
        if "dist.all_gather(" not in txt:
            raise FormalVerificationError(f"Calibration source pattern 'dist.all_gather(' not found in {p}")
        p.write_text(txt.replace("dist.all_gather(", "_noop_gather("), encoding="utf-8")

    _run_source_mutant(
        "M2_src_drop_pxb_allgather_consensus",
        "STATIC_PXB_RING_BYTES_BOUND",
        "VIOLATION",
        _drop_allgather,
    )

    # M3: Real source edit in C++ launcher -> g0004/g0009 `blocks % rows == 0` divisibility drift
    _run_source_mutant(
        "M3_src_g0004_cpp_raw_blocks_div",
        "Z3_FUSED_AR_3WAY_SOUNDNESS",
        "VIOLATION",
        lambda root: _apply_exact_replace(
            root / (rel_cpp if is_split else rel_py),
            "eff_blocks % rows == 0",
            "blocks % rows == 0",
        ),
    )

    # M4: Real source edit in Python can_fuse_add_rmsnorm -> omit divisibility guard
    def _drop_py_div(root: Path) -> None:
        p = root / rel_py
        txt = p.read_text(encoding="utf-8")
        new_txt, n = re.subn(
            r"if\s+rows\s*<\s*eff_blocks\s+and\s*\(?\s*eff_blocks\s*%\s*rows\s*\)?\s*!=\s*0:\s*\n\s*return\s+False",
            "pass",
            txt,
            count=1,
        )
        if n != 1:
            raise FormalVerificationError("Could not locate Python divisibility guard for M4")
        p.write_text(new_txt, encoding="utf-8")

    _run_source_mutant(
        "M4_src_drop_py_divisibility_guard",
        "Z3_FUSED_AR_3WAY_SOUNDNESS",
        "VIOLATION",
        _drop_py_div,
    )

    # M5: Real source edit in CUDA launch_shm_v5_allreduce_cuda -> g0010 10240 vs 40960 lockstep drift
    def _inject_g0010_cuda_drift(root: Path) -> None:
        p = root / (rel_cu if is_split else rel_py)
        txt = p.read_text(encoding="utf-8")
        if "small_lim = (pxb_bfp == 1) ? 20480 : 40960;" in txt:
            txt = txt.replace(
                "small_lim = (pxb_bfp == 1) ? 20480 : 40960;",
                "small_lim = (pxb_bfp == 1) ? 20480 : 10240;",
                1,
            )
        else:
            m = re.search(r"(void\s+launch_shm_v5_allreduce_cuda\s*\([^{]*\{.*?num_vecs\s*<=\s*)40960", txt, re.S)
            if m is None:
                raise FormalVerificationError("Could not locate CUDA allreduce 40960 cap for M5")
            txt = txt[: m.start()] + m.group(1) + "10240" + txt[m.end() :]
        p.write_text(txt, encoding="utf-8")

    _run_source_mutant(
        "M5_src_g0010_cuda_ar_10240_lockstep_drift",
        "Z3_FUSED_AR_3WAY_SOUNDNESS",
        "VIOLATION",
        _inject_g0010_cuda_drift,
    )

    # M6: Real source edit -> unmodeled 2 * launch_threads Python change MUST be classified as
    # INCONCLUSIVE_MODEL_MISMATCH (modeling boundary, NOT a false VIOLATION claim against candidate!)
    _run_source_mutant(
        "M6_src_unmodeled_py_2x_inconclusive_mismatch",
        "DIFF_PY_DESCRIPTOR_EQUIVALENCE",
        "INCONCLUSIVE_MODEL_MISMATCH",
        lambda root: _apply_exact_replace(
            root / rel_py,
            "vecs_per_row > 3 * launch_threads",
            "vecs_per_row > 2 * launch_threads",
        ),
    )

    # M7: Real source edit -> opaque dynamic small_pxb guard MUST fail closed as INCONCLUSIVE_PARSE_ERROR
    def _inject_unparseable_guard(root: Path) -> None:
        p = root / rel_py
        txt = p.read_text(encoding="utf-8")
        new_txt, n = re.subn(
            r"small_pxb\s*=\s*self\._pxb_enabled[^\n]+",
            "small_pxb = bool(getattr(self, '_opaque_runtime_hook', False))",
            txt,
            count=1,
        )
        if n != 1:
            raise FormalVerificationError("Could not locate Python small_pxb line for M7")
        p.write_text(new_txt, encoding="utf-8")

    _run_source_mutant(
        "M7_src_fail_stop_inconclusive_parse",
        "Z3_FUSED_AR_3WAY_SOUNDNESS",
        "INCONCLUSIVE_PARSE_ERROR",
        _inject_unparseable_guard,
    )

    resolved_cands = candidates_dir
    if resolved_cands is None:
        for cand_loc in (reference_dir.parent / "candidates", Path.cwd() / "targets/zstep/candidates"):
            if cand_loc.is_dir():
                resolved_cands = cand_loc
                break
    hist_rows = (
        calibrate_historical_candidates(resolved_cands, reference_dir)
        if resolved_cands is not None and resolved_cands.is_dir()
        else []
    )
    hist_matched = sum(1 for r in hist_rows if r.matched_expectation)

    killed = sum(1 for r in rows if r.killed)
    total = len(rows)
    kill_rate = killed / total if total else 0.0
    hist_ok = (hist_matched == len(hist_rows))
    return MutationCalibrationReport(
        passed=(baseline_closed and killed == total and hist_ok),
        baseline_passed=baseline_closed,
        total_mutants=total,
        killed_mutants=killed,
        kill_rate=kill_rate,
        rows=rows,
        historical_checked=len(hist_rows),
        historical_matched=hist_matched,
        historical_rows=hist_rows,
    )
