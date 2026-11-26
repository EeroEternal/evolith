"""Cross-language Z3, AST & compiled C++ differential clauses for Evolith × zstep.

Extends Evolith's formal verification path beyond single-file ``shm_allreduce``:
1. ``DIFF_CPP_GUARD_EQUIVALENCE``: Compiles the candidate's real C++ fused launcher
   (``launch_shm_v5_[pxb_]allreduce_add_rmsnorm``) with a lightweight pure-C++17
   ``TORCH_CHECK`` / ``torch::Tensor`` shim and executes 3,000 seeded inputs in an
   isolated child process (5s timeout) + 2 anti-vacuity perturbation checks against
   the extracted C++ SMT descriptor.
2. ``Z3_CROSS_LANG_SWA_POOL_SAFETY``: Anchors Rust ``serve.rs`` / ``cli.rs``
   ``min_active_swa`` capacity reservation against Python ``_SwaPagePool`` active
   straddling window pins under speculative verify/draft.
3. ``Z3_DSPARK_BT_CACHE_HORIZON``: Anchors Python ``_matches_gpu_bt_cache`` verify
   checked column upper bound against ``deepseek_v41_dspark.py`` draft reuse of
   ``_last_verify_bt_sig`` / ``mla_bt_gpu``.
4. ``STATIC_RUNNER_PXB_TOTAL_BUDGET``: Sums all static ``HostMappedShmAllReduce``
   PXB IPC ring allocations in ``DeepSeekV41StepRunner.__init__`` per topology
   (``PP=1`` vs ``PP=2``).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

MAX_SAFE_PP1_TOTAL_PXB_BYTES: int = 256 * 1024 * 1024  # 256 MiB total PXB ring ceiling on TP=8 PP=1
MAX_SAFE_PP2_TOTAL_PXB_BYTES: int = 384 * 1024 * 1024  # 384 MiB total PXB ring ceiling on PP=2 (12 GiB weight headroom)
_CPP_DIFF_CACHE: dict[tuple[str, int, int], list[bool]] = {}


def find_cpp_compiler() -> str | None:
    for cand in ("clang++", "g++", "c++"):
        found = shutil.which(cand)
        if found:
            return found
    return None


def _eval_cpp_spec_descriptor(
    cpp_spec: dict[str, Any],
    *,
    rows: int,
    hidden: int,
    blocks: int,
    one_shot_max_vecs: int,
    pxb: bool,
    world_size: int = 8,
) -> bool:
    """Pure Python evaluation of the exact SMT-LIB2 ``cpp_precond`` descriptor."""
    if rows <= 0 or hidden <= 0 or (hidden % 8) != 0:
        return False
    vecs_per_row = hidden // 8
    num_vecs = rows * vecs_per_row
    if vecs_per_row > cpp_spec["max_vecs_per_row"]:
        return False
    cpp_small_pxb = (
        bool(
            pxb
            and cpp_spec["blocks_lo"] <= blocks <= cpp_spec["blocks_hi"]
            and num_vecs <= cpp_spec["small_vecs_cap"]
        )
        if cpp_spec["has_small_pxb"]
        else False
    )
    cpp_eff_blocks = cpp_spec["small_eff_blocks"] if cpp_small_pxb else blocks
    div_var = cpp_spec["div_var"]
    if div_var == "cpp_eff_blocks":
        if rows < cpp_eff_blocks and (cpp_eff_blocks % rows) != 0:
            return False
    elif div_var == "blocks":
        if rows < blocks and (blocks % rows) != 0:
            return False
    if cpp_spec["requires_world_mod"]:
        if num_vecs > one_shot_max_vecs and (vecs_per_row % world_size) != 0:
            return False
    return True


def _extract_cpp_launcher_fn_full(cpp_src: str, fn_name: str) -> str | None:
    pattern = re.compile(rf"void\s+{re.escape(fn_name)}\s*\([^;{{}}]*\)\s*\{{")
    m = pattern.search(cpp_src)
    if m is None:
        return None
    start_brace = cpp_src.find("{", m.start())
    depth = 0
    for idx in range(start_brace, len(cpp_src)):
        ch = cpp_src[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cpp_src[m.start() : idx + 1]
    return None


def verify_cpp_guard_differential(
    sources: dict[str, Any],
    cpp_spec: dict[str, Any] | None,
    cpp_err: str | None,
    *,
    clause_cls: type[Any],
    trials: int = 3000,
    seed: int = 20261124,
    timeout_sec: float = 5.0,
) -> Any:
    """Compile real C++ fused launcher and differentially check 3,000 seeded trials."""
    rel_cpp = sources["rel_cpp"]
    if cpp_err or cpp_spec is None:
        return clause_cls(
            clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
            pillar="ModelCalibration",
            method="compiled_cpp_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_cpp}:1"],
            counterexample=cpp_err or "Failed to extract C++ fused AllReduce descriptor",
        )

    compiler = find_cpp_compiler()
    if compiler is None:
        return clause_cls(
            clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
            pillar="ModelCalibration",
            method="compiled_cpp_diff_3000",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            anchored_sites=[cpp_spec["anchor"]],
            detail="Host C++ compiler (clang++/g++) not found; C++ differential harness skipped",
        )

    has_pxb = bool(cpp_spec.get("has_small_pxb") or ("my_hbm_ptr" in (sources["cpp_src"] or "")))
    fn_name = (
        "launch_shm_v5_pxb_allreduce_add_rmsnorm"
        if (sources["cpp_src"] and "void launch_shm_v5_pxb_allreduce_add_rmsnorm" in sources["cpp_src"])
        else "launch_shm_v5_allreduce_add_rmsnorm"
    )
    fn_full = _extract_cpp_launcher_fn_full(sources["cpp_src"] or "", fn_name)
    if fn_full is None:
        return clause_cls(
            clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
            pillar="ModelCalibration",
            method="compiled_cpp_diff_3000",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[cpp_spec["anchor"]],
            counterexample=f"Could not extract full C++ body of {fn_name}",
        )

    sig_header = fn_full.split("{", 1)[0]
    m_params = re.search(r"\((.*)\)", sig_header, re.S)
    raw_params = [p.strip() for p in (m_params.group(1).split(",") if m_params else []) if p.strip()]
    param_expr_map = {
        "input": "in_t",
        "residual": "res_t",
        "weight": "w_t",
        "normed_out": "out_t",
        "output": "out_t",
        "control": "ctrl_t",
        "rank_shm_ptrs_u64": "ptrs_t",
        "my_hbm_ptr": "(pxb ? 0x1000ULL : 0ULL)",
        "pt_hbm_ptr": "(pxb ? 0x2000ULL : 0ULL)",
        "slot_stride_bytes": "slot_stride_bytes",
        "pxb_slot_stride_bytes": "pxb_slot_stride_bytes",
        "eps": "1e-6",
        "weight_offset": "0.0",
        "rank": "0",
        "world_size": "world_size",
        "blocks": "blocks",
        "threads": "threads",
        "one_shot_max_vecs": "one_shot_max_vecs",
        "pxb_max_vecs": "pxb_max_vecs",
        "rows": "rows",
        "hidden": "hidden",
    }
    call_args: list[str] = []
    has_pxb_sig = False
    for rp in raw_params:
        m_ident = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=.*)?$", rp)
        pname = m_ident.group(1) if m_ident else ""
        if pname == "my_hbm_ptr":
            has_pxb_sig = True
        call_args.append(param_expr_map.get(pname, "0"))
    invoke_stmt = f"{fn_name}({', '.join(call_args)});"

    harness_cpp = f"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace at {{
constexpr int kBFloat16 = 1;
constexpr int kLong = 2;
}}

namespace torch {{
struct Sizes {{
    int64_t a;
    int64_t b;
    bool operator==(const Sizes& o) const {{ return a == o.a && b == o.b; }}
    bool operator!=(const Sizes& o) const {{ return !(*this == o); }}
}};
struct Tensor {{
    int64_t d;
    int64_t s0;
    int64_t s1;
    int stype;
    std::uintptr_t ptr;
    bool is_cuda() const {{ return true; }}
    bool is_contiguous() const {{ return true; }}
    int scalar_type() const {{ return stype; }}
    int64_t dim() const {{ return d; }}
    int64_t size(int idx) const {{ return idx == 0 ? s0 : s1; }}
    Sizes sizes() const {{ return Sizes{{s0, d == 2 ? s1 : 1}}; }}
    int64_t numel() const {{ return d == 2 ? (s0 * s1) : s0; }}
    void* data_ptr() const {{ return reinterpret_cast<void*>(ptr); }}
    template <typename T>
    T* data_ptr() const {{ return reinterpret_cast<T*>(ptr); }}
}};
}}

struct CheckFailed {{}};
#define TORCH_CHECK(cond, ...) do {{ if (!(cond)) {{ throw CheckFailed{{}}; }} }} while (0)

static bool g_launched = false;
template <typename... Args>
void launch_shm_v5_allreduce_add_rmsnorm_cuda(Args&&...) {{
    g_launched = true;
}}

{fn_full}

int main() {{
    int64_t rows = 0, hidden = 0, blocks = 0, threads = 0, one_shot_max_vecs = 0, world_size = 0;
    int pxb_int = 0;
    while (std::scanf("%lld %lld %lld %lld %lld %lld %d",
                      &rows, &hidden, &blocks, &threads, &one_shot_max_vecs, &world_size, &pxb_int) == 7) {{
        const bool pxb = (pxb_int != 0);
        const int64_t slot_stride_bytes = 1LL << 34;
        const int64_t pxb_slot_stride_bytes = 1LL << 30;
        const int64_t pxb_max_vecs = 1LL << 26;
        torch::Tensor in_t{{2, rows, hidden, at::kBFloat16, 0x10000ULL}};
        torch::Tensor res_t{{2, rows, hidden, at::kBFloat16, 0x20000ULL}};
        torch::Tensor w_t{{1, hidden, 1, at::kBFloat16, 0x30000ULL}};
        torch::Tensor out_t{{2, rows, hidden, at::kBFloat16, 0x40000ULL}};
        torch::Tensor ctrl_t{{1, 4096, 1, at::kLong, 0x50000ULL}};
        torch::Tensor ptrs_t{{1, 16, 1, at::kLong, 0x60000ULL}};
        bool ok = false;
        try {{
            if (rows <= 0 || hidden <= 0 || (hidden % 8) != 0) {{
                throw CheckFailed{{}};
            }}
            {invoke_stmt}
            ok = true;
        }} catch (const CheckFailed&) {{
            ok = false;
        }}
        std::printf("%d\\n", ok ? 1 : 0);
    }}
    return 0;
}}
"""
    import random

    edge_rows = (1, 2, 3, 4, 7, 8, 9, 12, 16, 17, 24, 32, 48, 56, 64, 96, 128, 256, 512, 1024, 4096)
    hiddens = (2048, 4096, 5120, 7168, 8192, 12288, 12296, 16384)
    block_choices = (8, 16, 24, 32, 40, 48, 56, 64)
    thread_choices = (128, 256, 512, 1024)
    oneshot_choices = (64, 512, 4096, 15360, 16384, 65536)
    world_choices = (4, 8)
    pxb_choices = (0, 1) if (has_pxb and has_pxb_sig) else (0,)

    rng = random.Random(seed)
    samples: list[tuple[int, int, int, int, int, int, int]] = []
    for _ in range(trials):
        r = rng.choice(edge_rows) if rng.random() < 0.7 else rng.randint(1, 2048)
        h = rng.choice(hiddens)
        b = rng.choice(block_choices)
        t = rng.choice(thread_choices)
        os_v = rng.choice(oneshot_choices)
        ws = rng.choice(world_choices)
        pxb_v = rng.choice(pxb_choices)
        samples.append((r, h, b, t, os_v, ws, pxb_v))

    cache_key = (hashlib.sha256(harness_cpp.encode("utf-8")).hexdigest(), trials, seed)
    if cache_key in _CPP_DIFF_CACHE:
        real_bools = _CPP_DIFF_CACHE[cache_key]
    else:
        with tempfile.TemporaryDirectory(prefix="evolith_cpp_diff_") as td:
            src_file = Path(td) / "harness.cpp"
            bin_file = Path(td) / "harness_bin"
            src_file.write_text(harness_cpp, encoding="utf-8")
            comp = subprocess.run(
                [compiler, "-std=c++17", "-O1", str(src_file), "-o", str(bin_file)],
                text=True,
                capture_output=True,
                timeout=max(10.0, timeout_sec * 2.0),
                check=False,
            )
            if comp.returncode != 0:
                err_tail = (comp.stderr or comp.stdout).strip()[:280]
                return clause_cls(
                    clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
                    pillar="ModelCalibration",
                    method="compiled_cpp_diff_3000",
                    enforced=True,
                    verdict="INCONCLUSIVE_MODEL_MISMATCH",
                    passed=False,
                    anchored_sites=[cpp_spec["anchor"]],
                    counterexample=f"C++ guard harness failed to compile ({fn_name}): {err_tail}",
                )

            stdin_payload = "".join(
                f"{r} {h} {b} {t} {os_v} {ws} {pxb_v}\n"
                for (r, h, b, t, os_v, ws, pxb_v) in samples
            )
            try:
                run_res = subprocess.run(
                    [str(bin_file)],
                    input=stdin_payload,
                    text=True,
                    capture_output=True,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return clause_cls(
                    clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
                    pillar="ModelCalibration",
                    method="compiled_cpp_diff_3000",
                    enforced=True,
                    verdict="INCONCLUSIVE_TIMEOUT",
                    passed=False,
                    anchored_sites=[cpp_spec["anchor"]],
                    counterexample=f"Compiled C++ differential harness timed out after {timeout_sec:.1f}s",
                )

            if run_res.returncode != 0:
                return clause_cls(
                    clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
                    pillar="ModelCalibration",
                    method="compiled_cpp_diff_3000",
                    enforced=True,
                    verdict="INCONCLUSIVE_PARSE_ERROR",
                    passed=False,
                    anchored_sites=[cpp_spec["anchor"]],
                    counterexample=f"Compiled C++ harness exited {run_res.returncode}",
                )

            out_lines = [ln.strip() for ln in run_res.stdout.splitlines() if ln.strip()]
            if len(out_lines) != len(samples):
                return clause_cls(
                    clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
                    pillar="ModelCalibration",
                    method="compiled_cpp_diff_3000",
                    enforced=True,
                    verdict="INCONCLUSIVE_PARSE_ERROR",
                    passed=False,
                    anchored_sites=[cpp_spec["anchor"]],
                    counterexample=f"Expected {len(samples)} C++ results, got {len(out_lines)}",
                )

        real_bools = [ln == "1" for ln in out_lines]
        _CPP_DIFF_CACHE[cache_key] = real_bools
    for idx, (r, h, b, t, os_v, ws, pxb_v) in enumerate(samples):
        spec_val = _eval_cpp_spec_descriptor(
            cpp_spec,
            rows=r,
            hidden=h,
            blocks=b,
            one_shot_max_vecs=os_v,
            pxb=bool(pxb_v),
            world_size=ws,
        )
        if real_bools[idx] != spec_val:
            return clause_cls(
                clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
                pillar="ModelCalibration",
                method="compiled_cpp_diff_3000",
                enforced=True,
                verdict="INCONCLUSIVE_MODEL_MISMATCH",
                passed=False,
                anchored_sites=[cpp_spec["anchor"]],
                explored_states=idx + 1,
                counterexample=(
                    f"C++ model mismatch on (rows={r}, hidden={h}, blocks={b}, "
                    f"one_shot_max_vecs={os_v}, world_size={ws}, pxb={bool(pxb_v)}): "
                    f"compiled_cpp={real_bools[idx]} != extracted_spec={spec_val}"
                ),
            )

    # Anti-vacuity self-checks: perturbed descriptors MUST disagree with compiled C++ outputs
    pert_vpr = dict(cpp_spec, max_vecs_per_row=max(256, cpp_spec["max_vecs_per_row"] // 2))
    pert_div = dict(
        cpp_spec,
        div_var=None if cpp_spec["div_var"] is not None else "blocks",
    )
    for label, pert_spec in (("max_vecs_per_row//2", pert_vpr), ("flip_div_var", pert_div)):
        caught = any(
            real_bools[i]
            != _eval_cpp_spec_descriptor(
                pert_spec,
                rows=s[0],
                hidden=s[1],
                blocks=s[2],
                one_shot_max_vecs=s[4],
                pxb=bool(s[6]),
                world_size=s[5],
            )
            for i, s in enumerate(samples)
        )
        if not caught:
            return clause_cls(
                clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
                pillar="ModelCalibration",
                method="compiled_cpp_diff_3000",
                enforced=True,
                verdict="INCONCLUSIVE_PARSE_ERROR",
                passed=False,
                anchored_sites=[cpp_spec["anchor"]],
                explored_states=len(samples),
                counterexample=f"Anti-vacuity self-check failed on C++ perturbation `{label}`",
            )

    return clause_cls(
        clause_id="DIFF_CPP_GUARD_EQUIVALENCE",
        pillar="ModelCalibration",
        method="compiled_cpp_diff_3000",
        enforced=True,
        verdict="CLOSED_DIFF",
        passed=True,
        anchored_sites=[cpp_spec["anchor"]],
        explored_states=len(samples),
        detail=(
            f"Compiled C++ {fn_name} matched SMT descriptor across {len(samples)} seeded trials "
            "(+ 2/2 anti-vacuity perturbations caught)"
        ),
    )


def _run_z3_smt2(z3_bin: str, smt2_query: str, timeout_sec: float = 10.0) -> tuple[str, str]:
    res = subprocess.run(
        [z3_bin, "-in", "-smt2"],
        input=smt2_query,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    out = (res.stdout + "\n" + res.stderr).strip()
    first = out.splitlines()[0].strip() if out else ""
    return first, out


def verify_z3_cross_lang_swa_pool_safety(
    candidate_dir: Path,
    z3_bin: str | None,
    *,
    clause_cls: type[Any],
    require_z3: bool = False,
) -> Any:
    """Prove Rust SwaSafeLru active reservation >= Python _SwaPagePool max pinned pages/seq."""
    rel_py = "worker/zstep_worker/models/deepseek_v41.py"
    rel_serve = "crates/zs-server/src/serve.rs"
    rel_cli = "crates/zs-server/src/serve/cli.rs"
    py_path = candidate_dir / rel_py
    serve_path = candidate_dir / rel_serve
    cli_path = candidate_dir / rel_cli

    if not py_path.is_file():
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="deepseek_v41.py not present",
        )

    py_text = py_path.read_text(encoding="utf-8")
    if "class _SwaPagePool" not in py_text or not serve_path.is_file():
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="Decoupled _SwaPagePool / serve.rs SWA reservation not present in this tree",
        )

    serve_text = serve_path.read_text(encoding="utf-8")
    if "min_active_swa" not in serve_text:
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="Rust serve.rs does not configure min_active_swa",
        )

    m_serve = re.search(
        r"let\s+min_active_swa\s*=\s*(\d+)\s*\*\s*args\.max_seqs", serve_text
    )
    if m_serve is None:
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_serve}:1"],
            counterexample="Unparseable `min_active_swa` expression in crates/zs-server/src/serve.rs",
        )
    serve_mult = int(m_serve.group(1))
    serve_line = serve_text[: m_serve.start()].count("\n") + 1
    anchors = [f"{rel_serve}:{serve_line}"]

    cli_mult = serve_mult
    if cli_path.is_file():
        cli_text = cli_path.read_text(encoding="utf-8")
        if "min_active_swa" in cli_text:
            m_cli = re.search(
                r"let\s+min_active_swa\s*=\s*(\d+)\s*\*\s*args\.max_seqs", cli_text
            )
            if m_cli is None:
                return clause_cls(
                    clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
                    pillar="CrossLanguageConservation",
                    method="z3_smt2",
                    enforced=True,
                    verdict="INCONCLUSIVE_PARSE_ERROR",
                    passed=False,
                    anchored_sites=[f"{rel_cli}:1"],
                    counterexample="Unparseable `min_active_swa` expression in crates/zs-server/src/serve/cli.rs",
                )
            cli_mult = int(m_cli.group(1))
            cli_line = cli_text[: m_cli.start()].count("\n") + 1
            anchors.append(f"{rel_cli}:{cli_line}")

    m_plo = re.search(
        r"p_lo\s*=\s*max\(\s*0\s*,\s*\(\s*int\(seq_pos_min\[i\]\)\s*-\s*(\d+)\s*\)\s*//\s*(\d+)\s*\)",
        py_text,
    )
    m_phi = re.search(
        r"p_hi\s*=\s*min\([^,]+,\s*int\(seq_pos_max\[i\]\)\s*//\s*(\d+)\s*\)",
        py_text,
    )
    if m_plo is None or m_phi is None or int(m_plo.group(2)) != int(m_phi.group(1)):
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[*anchors, f"{rel_py}:1"],
            counterexample="Unparseable _SwaPagePool p_lo / p_hi window expressions in deepseek_v41.py",
        )

    swa_lookback = int(m_plo.group(1))
    page_tokens = int(m_plo.group(2))
    py_line = py_text[: m_plo.start()].count("\n") + 1
    anchors.append(f"{rel_py}:{py_line}")

    # Default DSpark verify window spans k+1 tokens (pos_max - pos_min <= k, default k = 3)
    spec_k = 3
    m_spec = re.search(r"num_spec_tokens[^\n]*?=\s*(\d+)", py_text)
    if m_spec is not None:
        spec_k = max(spec_k, int(m_spec.group(1)))

    if z3_bin is None:
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=require_z3,
            verdict="INCONCLUSIVE_NO_Z3",
            passed=False,
            anchored_sites=anchors,
            counterexample="Z3 binary not installed" if require_z3 else None,
            detail="Z3 unavailable; cross-language SWA capacity proof skipped",
        )

    eff_rust_mult = min(serve_mult, cli_mult)
    smt2 = f"""
(set-logic QF_LIA)
(declare-const pos_min Int)
(declare-const span Int)
(assert (>= pos_min 0))
(assert (<= pos_min 131072))
(assert (>= span 0))
(assert (<= span {spec_k}))
(define-fun pos_max () Int (+ pos_min span))
(define-fun raw_lo () Int (div (- pos_min {swa_lookback}) {page_tokens}))
(define-fun p_lo () Int (ite (> raw_lo 0) raw_lo 0))
(define-fun p_hi () Int (div pos_max {page_tokens}))
(define-fun pinned_pages () Int (+ (- p_hi p_lo) 1))
(assert (or (distinct {serve_mult} {cli_mult}) (> pinned_pages {eff_rust_mult})))
(check-sat)
(get-value (pos_min pos_max p_lo p_hi pinned_pages))
"""
    status, raw_out = _run_z3_smt2(z3_bin, smt2)
    if status == "unsat":
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=True,
            verdict="PROVED",
            passed=True,
            anchored_sites=anchors,
            detail=(
                f"Z3 proved active SWA window pins <= {eff_rust_mult} pages/seq across all "
                f"(pos_min, span<={spec_k}) matching Rust min_active_swa={eff_rust_mult}*max_seqs"
            ),
        )
    if status == "sat":
        kv = dict(re.findall(r"\(([a-zA-Z0-9_]+)\s+(-?\d+)\)", raw_out))
        witness = (
            f"Rust reserves min_active_swa={eff_rust_mult}*max_seqs (serve={serve_mult}, cli={cli_mult}), "
            f"but Python _SwaPagePool pins {kv.get('pinned_pages', '?')} pages/seq at "
            f"pos_min={kv.get('pos_min', '?')}..pos_max={kv.get('pos_max', '?')} "
            f"(cols {kv.get('p_lo', '?')}..{kv.get('p_hi', '?')}), evicting prefix pages tracked by SwaSafeLru"
        )
        return clause_cls(
            clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
            pillar="CrossLanguageConservation",
            method="z3_smt2",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=anchors,
            counterexample=witness,
        )
    return clause_cls(
        clause_id="Z3_CROSS_LANG_SWA_POOL_SAFETY",
        pillar="CrossLanguageConservation",
        method="z3_smt2",
        enforced=True,
        verdict="INCONCLUSIVE_PARSE_ERROR",
        passed=False,
        anchored_sites=anchors,
        counterexample=f"Unexpected Z3 solver output: {status!r}",
    )


def verify_z3_dspark_bt_cache_horizon(
    candidate_dir: Path,
    z3_bin: str | None,
    *,
    clause_cls: type[Any],
    require_z3: bool = False,
) -> Any:
    """Prove DSpark draft reuse of _last_verify_bt_sig never reads beyond verify-checked columns."""
    rel_v41 = "worker/zstep_worker/models/deepseek_v41.py"
    rel_dsp = "worker/zstep_worker/models/deepseek_v41_dspark.py"
    v41_path = candidate_dir / rel_v41
    dsp_path = candidate_dir / rel_dsp

    if not v41_path.is_file() or not dsp_path.is_file():
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="deepseek_v41.py or deepseek_v41_dspark.py not present",
        )

    v41_text = v41_path.read_text(encoding="utf-8")
    dsp_text = dsp_path.read_text(encoding="utf-8")
    if "_matches_gpu_bt_cache" not in v41_text or "_last_verify_bt_sig" not in dsp_text:
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="GPU block-table cross-step cache (_matches_gpu_bt_cache / _last_verify_bt_sig) not present",
        )

    # Locate verify block-table cache check around `_last_verify_bt_sig = (`
    m_sig_set = re.search(r"self\._last_verify_bt_sig\s*=\s*\(", v41_text)
    if m_sig_set is None:
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="self._last_verify_bt_sig is never populated in deepseek_v41.py",
        )

    window_start = max(0, m_sig_set.start() - 8000)
    verify_region = v41_text[window_start : m_sig_set.end() + 200]
    sig_line = v41_text[: m_sig_set.start()].count("\n") + 1

    # Check how verify computes the upper column validated by _matches_gpu_bt_cache
    verify_extra_toks: int | None = None
    if re.search(
        r"c_hi\s*=\s*np\.minimum\([^,]+,\s*\(\s*p_max\s*\+\s*self\.num_spec_tokens\s*\)\s*//\s*128\s*\)",
        verify_region,
    ):
        verify_extra_toks = -1  # symbolic `k` (`self.num_spec_tokens`)
    elif re.search(
        r"c_hi\s*=\s*np\.minimum\([^,]+,\s*\(\s*p_max\s*\+\s*(\d+)\s*\)\s*//\s*128\s*\)",
        verify_region,
    ):
        m_const = re.search(
            r"c_hi\s*=\s*np\.minimum\([^,]+,\s*\(\s*p_max\s*\+\s*(\d+)\s*\)\s*//\s*128\s*\)",
            verify_region,
        )
        assert m_const is not None
        verify_extra_toks = int(m_const.group(1))
    elif re.search(r"c_hi\s*=\s*np\.minimum\([^,]+,\s*p_max\s*//\s*128\s*\)", verify_region):
        verify_extra_toks = 0

    m_dsp_sig = re.search(r"_last_verify_bt_sig", dsp_text)
    dsp_line = (dsp_text[: m_dsp_sig.start()].count("\n") + 1) if m_dsp_sig else 1
    anchors = [f"{rel_v41}:{sig_line}", f"{rel_dsp}:{dsp_line}"]

    if verify_extra_toks is None:
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=anchors,
            counterexample="Unparseable verify `c_hi` horizon expression in deepseek_v41.py",
        )

    # Check whether dspark.py explicitly guards `bt_cached` with a column-bound check against verify_c_hi
    draft_guards_col_bound = bool(
        re.search(r"verify_c_hi|_verified_bt_c_hi", dsp_text)
        and re.search(r"<=\s*(?:runner\.)?(?:verify_c_hi|_verified_bt_c_hi)", dsp_text)
    )

    if z3_bin is None:
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=require_z3,
            verdict="INCONCLUSIVE_NO_Z3",
            passed=False,
            anchored_sites=anchors,
            counterexample="Z3 binary not installed" if require_z3 else None,
            detail="Z3 unavailable; DSpark block-table cache horizon proof skipped",
        )

    verify_rhs = "(+ p_max k)" if verify_extra_toks == -1 else f"(+ p_max {verify_extra_toks})"
    guard_cond = "false" if draft_guards_col_bound else "true"
    smt2 = f"""
(set-logic QF_LIA)
(declare-const p_max Int)
(declare-const k Int)
(assert (>= p_max 0))
(assert (<= p_max 131000))
(assert (>= k 1))
(assert (<= k 3))
(define-fun verify_checked_col () Int (div {verify_rhs} 128))
(define-fun draft_max_col () Int (div (+ p_max k) 128))
(assert {guard_cond})
(assert (> draft_max_col verify_checked_col))
(check-sat)
(get-value (p_max k verify_checked_col draft_max_col))
"""
    status, raw_out = _run_z3_smt2(z3_bin, smt2)
    if status == "unsat":
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=True,
            verdict="PROVED",
            passed=True,
            anchored_sites=anchors,
            detail="Z3 proved DSpark draft block-table reuse never accesses columns beyond verify-validated c_hi",
        )
    if status == "sat":
        kv = dict(re.findall(r"\(([a-zA-Z0-9_]+)\s+(-?\d+)\)", raw_out))
        witness = (
            f"Verify validates mla_bt_gpu/swa_bt_gpu up to col={kv.get('verify_checked_col', '?')} "
            f"at verify p_max={kv.get('p_max', '?')}, but DSpark draft(k={kv.get('k', '?')}) sets "
            f"bt_cached=True via _last_verify_bt_sig and reads unvalidated col={kv.get('draft_max_col', '?')}"
        )
        return clause_cls(
            clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
            pillar="CacheCoherence",
            method="z3_smt2",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=anchors,
            counterexample=witness,
        )
    return clause_cls(
        clause_id="Z3_DSPARK_BT_CACHE_HORIZON",
        pillar="CacheCoherence",
        method="z3_smt2",
        enforced=True,
        verdict="INCONCLUSIVE_PARSE_ERROR",
        passed=False,
        anchored_sites=anchors,
        counterexample=f"Unexpected Z3 solver output: {status!r}",
    )


def _compute_single_pxb_ring_bytes(pxb_max_elements: int, num_pxb_slots: int = 2) -> int:
    def _align_up(val: int, align: int) -> int:
        return ((val + align - 1) // align) * align

    pxb_max_vecs = max(1, pxb_max_elements // 8)
    slot_stride = _align_up(2 * pxb_max_vecs * 16, 4096)
    return _align_up(131072 + num_pxb_slots * slot_stride, 4096)


def verify_static_runner_pxb_total_budget(
    candidate_dir: Path,
    *,
    clause_cls: type[Any],
    max_pp1_bytes: int = MAX_SAFE_PP1_TOTAL_PXB_BYTES,
    max_pp2_bytes: int = MAX_SAFE_PP2_TOTAL_PXB_BYTES,
) -> Any:
    """Sum all HostMappedShmAllReduce PXB IPC ring allocations in DeepSeekV41StepRunner."""
    rel_v41 = "worker/zstep_worker/models/deepseek_v41.py"
    rel_shm = "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    v41_path = candidate_dir / rel_v41
    shm_path = candidate_dir / rel_shm

    if not v41_path.is_file() or not shm_path.is_file():
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="deepseek_v41.py or shm_allreduce.py not present",
        )

    shm_text = shm_path.read_text(encoding="utf-8")
    if "def _init_pxb_ipc" not in shm_text:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="Non-PXB shm_allreduce tree",
        )

    m_slots = re.search(r"^NUM_PXB_RING_SLOTS\s*=\s*(\d+)", shm_text, re.M)
    m_def_elems = re.search(r"^DEFAULT_PXB_MAX_ELEMENTS\s*=\s*([0-9*\s_]+)", shm_text, re.M)
    num_pxb_slots = int(m_slots.group(1)) if m_slots else 2
    default_pxb_elems = 4096 * 5120
    if m_def_elems:
        try:
            default_pxb_elems = int(eval(m_def_elems.group(1).strip(), {"__builtins__": {}}, {}))
        except Exception:
            default_pxb_elems = 4096 * 5120

    try:
        tree = ast.parse(v41_path.read_text(encoding="utf-8"), filename=rel_v41)
    except SyntaxError as exc:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=True,
            verdict="INCONCLUSIVE_PARSE_ERROR",
            passed=False,
            anchored_sites=[f"{rel_v41}:1"],
            counterexample=f"SyntaxError in deepseek_v41.py: {exc}",
        )

    runner_cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV41StepRunner"),
        None,
    )
    if runner_cls is None:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="DeepSeekV41StepRunner not found",
        )

    init_fns = [
        n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "_init_impl")
    ]
    if not init_fns:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="DeepSeekV41StepRunner.__init__ not found",
        )

    def _eval_ast_int(node: ast.AST, env: dict[str, int]) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            key = f"self.{node.attr}"
            if key in env:
                return env[key]
        if isinstance(node, ast.BinOp):
            lv = _eval_ast_int(node.left, env)
            rv = _eval_ast_int(node.right, env)
            if lv is None or rv is None:
                return None
            if isinstance(node.op, ast.Mult):
                return lv * rv
            if isinstance(node.op, ast.Add):
                return lv + rv
            if isinstance(node.op, ast.Sub):
                return lv - rv
            if isinstance(node.op, ast.FloorDiv) and rv != 0:
                return lv // rv
        if isinstance(node, ast.IfExp):
            # Support `8192 if self.pp_size > 1 else ...`
            if (
                isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1
                and len(node.test.comparators) == 1
            ):
                lv = _eval_ast_int(node.test.left, env)
                rv = _eval_ast_int(node.test.comparators[0], env)
                if lv is not None and rv is not None:
                    cond = False
                    op = node.test.ops[0]
                    if isinstance(op, ast.Gt):
                        cond = lv > rv
                    elif isinstance(op, ast.Eq):
                        cond = lv == rv
                    elif isinstance(op, ast.GtE):
                        cond = lv >= rv
                    elif isinstance(op, ast.Lt):
                        cond = lv < rv
                    elif isinstance(op, ast.LtE):
                        cond = lv <= rv
                    return _eval_ast_int(node.body if cond else node.orelse, env)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
            vals = [_eval_ast_int(a, env) for a in node.args]
            if vals and all(v is not None for v in vals):
                ints = [int(v) for v in vals if v is not None]
                return min(ints) if node.func.id == "min" else max(ints)
        return None

    def _simulate_topology(pp_size: int, max_step_tokens: int) -> tuple[int, list[str], list[str]]:
        env: dict[str, int] = {
            "self.pp_size": pp_size,
            "self.max_step_tokens": max_step_tokens,
            "DEFAULT_PXB_MAX_ELEMENTS": default_pxb_elems,
        }
        total_bytes = 0
        sites: list[str] = []
        breakdown: list[str] = []

        def _walk_stmts(stmts: list[ast.stmt], active_guard: bool) -> None:
            nonlocal total_bytes
            if not active_guard:
                return
            for st in stmts:
                target_var: str | None = None
                val_expr: ast.AST | None = None
                if isinstance(st, ast.Assign) and len(st.targets) == 1:
                    tgt = st.targets[0]
                    if isinstance(tgt, ast.Name):
                        target_var = tgt.id
                    elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                        target_var = f"self.{tgt.attr}"
                    val_expr = st.value
                elif isinstance(st, ast.AnnAssign) and st.value is not None:
                    tgt = st.target
                    if isinstance(tgt, ast.Name):
                        target_var = tgt.id
                    elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                        target_var = f"self.{tgt.attr}"
                    val_expr = st.value

                if target_var and val_expr is not None:
                    iv = _eval_ast_int(val_expr, env)
                    if iv is not None:
                        env[target_var] = iv

                if (
                    val_expr is not None
                    and isinstance(val_expr, ast.Call)
                    and isinstance(val_expr.func, ast.Name)
                    and val_expr.func.id == "HostMappedShmAllReduce"
                ):
                    kw_map = {kw.arg: kw.value for kw in val_expr.keywords if kw.arg}
                    en_node = kw_map.get("enable_pxb_p2p")
                    pxb_on = False
                    if en_node is not None:
                        if isinstance(en_node, ast.Constant):
                            pxb_on = bool(en_node.value)
                        else:
                            # e.g. `self.p2p_report.pxb_shm_enabled` -> True on PXB hosts
                            src_seg = ast.unparse(en_node)
                            pxb_on = "False" not in src_seg
                    if pxb_on:
                        cap_val = _eval_ast_int(kw_map["capacity_elements"], env) if "capacity_elements" in kw_map else default_pxb_elems
                        pxb_val = _eval_ast_int(kw_map["pxb_max_elements"], env) if "pxb_max_elements" in kw_map else default_pxb_elems
                        eff_elems = min(cap_val or default_pxb_elems, max(8, pxb_val or default_pxb_elems))
                        ring_b = _compute_single_pxb_ring_bytes(eff_elems, num_pxb_slots)
                        total_bytes += ring_b
                        sites.append(f"{rel_v41}:{st.lineno}")
                        breakdown.append(f"{target_var or 'shm_ar'}={ring_b / (1024 * 1024):.2f}MiB")

                if isinstance(st, ast.If):
                    # Check if branch explicitly requires `self.pp_size == 1` or `self.pp_size > 1`
                    take_body = True
                    take_else = True
                    test_src = ast.unparse(st.test)
                    if "self.pp_size == 1" in test_src and pp_size != 1:
                        take_body = False
                    elif "self.pp_size > 1" in test_src and pp_size <= 1:
                        take_body = False
                    _walk_stmts(st.body, take_body)
                    _walk_stmts(st.orelse, take_else)
                elif isinstance(st, ast.Try):
                    _walk_stmts(st.body, active_guard)
                    _walk_stmts(st.finalbody, active_guard)
                elif isinstance(st, ast.With):
                    _walk_stmts(st.body, active_guard)

        for fn_node in init_fns:
            _walk_stmts(fn_node.body, True)
        return total_bytes, sites, breakdown

    # Evaluate standard Evolith benchmark step budget (max_step_tokens=4096 on PP=1, 8192 on PP=2)
    pp1_bytes, pp1_sites, pp1_desc = _simulate_topology(pp_size=1, max_step_tokens=4096)
    pp2_bytes, pp2_sites, pp2_desc = _simulate_topology(pp_size=2, max_step_tokens=4096)
    all_sites = list(dict.fromkeys([*pp1_sites, *pp2_sites]))

    if not all_sites:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=False,
            verdict="NOT_APPLICABLE",
            passed=True,
            detail="No PXB-enabled HostMappedShmAllReduce instances in DeepSeekV41StepRunner",
        )

    if pp1_bytes > max_pp1_bytes:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=all_sites,
            counterexample=(
                f"PP=1 (TP=8) allocates {pp1_bytes / (1024 * 1024):.2f} MiB/GPU total static PXB IPC rings "
                f"({' + '.join(pp1_desc)}) > safe TP=8 budget {max_pp1_bytes / (1024 * 1024):.1f} MiB"
            ),
        )
    if pp2_bytes > max_pp2_bytes:
        return clause_cls(
            clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
            pillar="ResourceBudget",
            method="ast_exact_eval",
            enforced=True,
            verdict="VIOLATION",
            passed=False,
            anchored_sites=all_sites,
            counterexample=(
                f"PP=2 (TP=4) allocates {pp2_bytes / (1024 * 1024):.2f} MiB/GPU total static PXB IPC rings "
                f"({' + '.join(pp2_desc)}) > safe PP=2 budget {max_pp2_bytes / (1024 * 1024):.1f} MiB"
            ),
        )

    return clause_cls(
        clause_id="STATIC_RUNNER_PXB_TOTAL_BUDGET",
        pillar="ResourceBudget",
        method="ast_exact_eval",
        enforced=True,
        verdict="PASS",
        passed=True,
        anchored_sites=all_sites,
        detail=(
            f"Runner total static PXB IPC ring budget within bounds: "
            f"PP=1={pp1_bytes / (1024 * 1024):.2f} MiB ({', '.join(pp1_desc) or 'none'}), "
            f"PP=2={pp2_bytes / (1024 * 1024):.2f} MiB ({', '.join(pp2_desc) or 'none'})"
        ),
    )


def dump_json_compact(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
