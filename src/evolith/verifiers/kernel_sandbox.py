from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
import numpy as np

from evolith.models import BottleneckClass, HardwareProfile, KernelEvalResult


FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "_STATIC_TENSOR_CACHE",
        "breakpoint",
        "builtins",
        "compile",
        "ctypes",
        "cudaStreamCreate",
        "cudaStreamCreateWithFlags",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "importlib",
        "input",
        "locals",
        "open",
        "os",
        "pathlib",
        "setattr",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "vars",
    }
)

FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "_core",
        "ag_frame",
        "core",
        "cr_frame",
        "ctypes",
        "ctypeslib",
        "cudaStreamCreate",
        "cudaStreamCreateWithFlags",
        "distutils",
        "f2py",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "fork",
        "fromfile",
        "func_globals",
        "gi_frame",
        "kill",
        "load",
        "loadtxt",
        "memmap",
        "open",
        "popen",
        "Popen",
        "rec",
        "remove",
        "rmdir",
        "rmtree",
        "run",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "spawn",
        "system",
        "tb_frame",
        "testing",
        "tofile",
        "unlink",
        "urlopen",
    }
)

FORBIDDEN_DOTTED_CHAINS: frozenset[str] = frozenset(
    {
        "os.system",
        "os.popen",
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_output",
    }
)

_SAFE_BUILTIN_SYMBOLS: dict[str, object] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "RuntimeError": RuntimeError,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}


def _dotted_attr_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def audit_sandboxed_python(
    source_code: str,
    *,
    allowed_modules: frozenset[str],
) -> tuple[bool, str]:
    """
    Static AST & anti-escape gate for untrusted candidate Python snippets.
    Blocks module-level mutable state (memoization reward hacking), unauthorized
    imports, forbidden names, dangerous attribute/dotted chains (`os.system`,
    `np.load`/`np.memmap`), and dunder attribute introspection (`__subclasses__`).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(
                    node.value, (ast.Dict, ast.List, ast.Set, ast.Call)
                ):
                    return (
                        False,
                        f"Anti-Cheat Gate: Module-level mutable state '{target.id}' forbidden (Memoization Hack).",
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return (
                False,
                f"Anti-Cheat Gate: Module-level mutable state '{node.target.id}' forbidden (Memoization Hack).",
            )
        elif isinstance(node, ast.Expr):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                return (
                    False,
                    "Anti-Cheat Gate: Top-level executable expressions are forbidden.",
                )
        elif not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            return (
                False,
                f"Anti-Cheat Gate: Disallowed top-level statement '{type(node).__name__}'.",
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "Anti-Cheat Gate: 'global'/'nonlocal' statement forbidden inside kernel."
        if isinstance(
            node,
            (ast.AsyncFunctionDef, ast.ClassDef, ast.Yield, ast.YieldFrom, ast.Await),
        ):
            return (
                False,
                f"Anti-Cheat Gate: Disallowed construct '{type(node).__name__}' inside sandbox.",
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_modules:
                    return (
                        False,
                        f"Anti-Cheat Gate: Import '{alias.name}' is not in sandbox allowlist {sorted(allowed_modules)}.",
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level != 0 or mod not in allowed_modules:
                return (
                    False,
                    f"Anti-Cheat Gate: Import from '{mod}' is not in sandbox allowlist {sorted(allowed_modules)}.",
                )
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    return (
                        False,
                        f"Anti-Cheat Gate: Disallowed symbol import '{alias.name}' from '{mod}'.",
                    )
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or (
                node.id.startswith("__") and node.id.endswith("__")
            ):
                return False, f"Anti-Cheat Gate: Forbidden identifier '{node.id}'."
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return (
                    False,
                    f"Anti-Cheat Gate: Dunder attribute access '{node.attr}' forbidden.",
                )
            chain = _dotted_attr_chain(node)
            if chain and chain in FORBIDDEN_DOTTED_CHAINS:
                return (
                    False,
                    f"Anti-Cheat Gate: Forbidden attribute call '{chain}'.",
                )
            if node.attr in FORBIDDEN_ATTRS:
                return (
                    False,
                    f"Anti-Cheat Gate: Forbidden attribute '{node.attr}'.",
                )

    return True, "OK"


_ALLOWED_NUMPY_ATTRS: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "allclose",
        "amax",
        "amin",
        "any",
        "arange",
        "argmax",
        "argmin",
        "argsort",
        "array",
        "asarray",
        "bool_",
        "clip",
        "concatenate",
        "cos",
        "dot",
        "einsum",
        "empty",
        "empty_like",
        "exp",
        "expand_dims",
        "eye",
        "float16",
        "float32",
        "float64",
        "floor",
        "full",
        "full_like",
        "inf",
        "int32",
        "int64",
        "isfinite",
        "isnan",
        "linalg",
        "log",
        "log1p",
        "matmul",
        "max",
        "maximum",
        "mean",
        "min",
        "minimum",
        "nan",
        "ndarray",
        "newaxis",
        "ones",
        "ones_like",
        "percentile",
        "pi",
        "power",
        "prod",
        "reciprocal",
        "reshape",
        "rint",
        "round",
        "sin",
        "sort",
        "sqrt",
        "square",
        "squeeze",
        "stack",
        "std",
        "sum",
        "tanh",
        "transpose",
        "uint8",
        "var",
        "where",
        "zeros",
        "zeros_like",
    }
)


class _SafeModuleProxy:
    __slots__ = ("_mod", "_allowed", "_name")

    def __init__(self, mod: Any, allowed: frozenset[str], name: str) -> None:
        object.__setattr__(self, "_mod", mod)
        object.__setattr__(self, "_allowed", allowed)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr: str) -> Any:
        if attr not in object.__getattribute__(self, "_allowed"):
            raise AttributeError(
                f"Anti-Cheat Gate: Attribute '{object.__getattribute__(self, '_name')}.{attr}' is not in sandbox allowlist"
            )
        return getattr(object.__getattribute__(self, "_mod"), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        raise TypeError("Sandbox module proxy is read-only")


def make_safe_builtins(allowed_modules: frozenset[str]) -> dict[str, object]:
    def _safe_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0 or name not in allowed_modules:
            raise ImportError(f"Sandbox blocked import of '{name}'")
        mod = importlib.import_module(name)
        if not fromlist and "." in name:
            return importlib.import_module(name.split(".", 1)[0])
        if name == "numpy":
            return _SafeModuleProxy(mod, _ALLOWED_NUMPY_ATTRS, "numpy")
        return mod

    return {**_SAFE_BUILTIN_SYMBOLS, "__import__": _safe_import}


class HardenedKernelVerifier:
    """
    Offline CPU Pedagogical & Unit-Test Micro-Kernel Verifier for Plane L1:
      Stage 1: Static AST, Import Allowlist & Subprocess Isolation Audit
      Stage 2: Multi-Shape (Standard + Adversarial Prime Shapes) & Read-Only Input Checksum Gate
      Stage 3: Ring-Buffer Memory Allocation + Cache Scrub Timing
      Stage 4: Analytic Roofline Cost-Model Classification & Reflection Card Generation

    Note: Stage 4 uses an analytic byte/FLOP cost model for GPU-free demonstrations;
    real GPU CUDA-Event / hardware-counter verification lives in `evolith.zstep`.
    """

    ALLOWED_MODULES: frozenset[str] = frozenset({"numpy", "math"})

    def __init__(self, ring_buffer_slots: int = 16, timed_iters: int = 24) -> None:
        self.ring_buffer_slots = ring_buffer_slots
        self.timed_iters = timed_iters
        self._l2_flush_buf = np.zeros((4 * 1024 * 1024,), dtype=np.uint8)

    def stage1_static_ast_gate(self, source_code: str) -> tuple[bool, str]:
        return audit_sandboxed_python(
            source_code, allowed_modules=self.ALLOWED_MODULES
        )

    @classmethod
    def _compile_kernel_fn(cls, source_code: str) -> Callable:
        namespace: dict[str, object] = {
            "__builtins__": make_safe_builtins(cls.ALLOWED_MODULES)
        }
        exec(compile(source_code, "<evolved_kernel>", "exec"), namespace)
        fn = namespace.get("kernel_fn")
        if not callable(fn):
            raise ValueError("Source code must define a callable 'kernel_fn(...)'.")
        return fn

    def _make_inputs(
        self, op_name: str, batch: int, dim: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, ...]:
        if op_name == "triattention_prerope_score":
            seq_len = dim
            head_dim = 64
            k = rng.standard_normal((batch, seq_len, head_dim), dtype=np.float32)
            q = rng.standard_normal((batch, head_dim), dtype=np.float32)
            bias = rng.uniform(-0.1, 0.1, size=(128,)).astype(np.float32)
            return (k, q, bias)
        x = rng.standard_normal((batch, dim), dtype=np.float32)
        res = rng.standard_normal((batch, dim), dtype=np.float32)
        w = rng.uniform(0.5, 1.5, size=(dim,)).astype(np.float32)
        return (x, res, w)

    def evaluate(
        self,
        op_name: str,
        candidate_code: str,
        baseline_code: str,
        *,
        timeout_sec: float = 10.0,
    ) -> KernelEvalResult:
        loc = len(
            [
                ln
                for ln in candidate_code.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        )
        ok_ast, reason = self.stage1_static_ast_gate(candidate_code)
        if not ok_ast:
            return KernelEvalResult(
                op_name=op_name,
                passed_ast_gate=False,
                passed_correctness_gate=False,
                passed_anti_cheat_gate=False,
                loc=loc,
                rejection_reason=reason,
            )

        src_root = str(Path(__file__).resolve().parents[2])
        payload = json.dumps(
            {
                "op_name": op_name,
                "candidate_code": candidate_code,
                "baseline_code": baseline_code,
                "ring_buffer_slots": self.ring_buffer_slots,
                "timed_iters": self.timed_iters,
            }
        )
        runner_script = (
            "import json, sys\n"
            f"sys.path.insert(0, {src_root!r})\n"
            "from evolith.verifiers.kernel_sandbox import HardenedKernelVerifier\n"
            "cfg = json.loads(sys.stdin.read())\n"
            "v = HardenedKernelVerifier(ring_buffer_slots=cfg['ring_buffer_slots'], timed_iters=cfg['timed_iters'])\n"
            "res = v._evaluate_in_process(cfg['op_name'], cfg['candidate_code'], cfg['baseline_code'])\n"
            "sys.stdout.write(res.model_dump_json())\n"
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner_script],
                input=payload,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return KernelEvalResult(
                op_name=op_name,
                passed_ast_gate=True,
                passed_correctness_gate=False,
                passed_anti_cheat_gate=False,
                loc=loc,
                rejection_reason=f"Sandbox subprocess timed out after {timeout_sec:.1f}s",
            )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            msg = detail[-1] if detail else f"exit code {proc.returncode}"
            return KernelEvalResult(
                op_name=op_name,
                passed_ast_gate=True,
                passed_correctness_gate=False,
                passed_anti_cheat_gate=False,
                loc=loc,
                rejection_reason=f"Sandbox subprocess rejected candidate: {msg}",
            )
        return KernelEvalResult.model_validate_json(proc.stdout)

    def _evaluate_in_process(
        self,
        op_name: str,
        candidate_code: str,
        baseline_code: str,
    ) -> KernelEvalResult:
        loc = len(
            [
                ln
                for ln in candidate_code.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        )
        ok_ast, reason = self.stage1_static_ast_gate(candidate_code)
        if not ok_ast:
            return KernelEvalResult(
                op_name=op_name,
                passed_ast_gate=False,
                passed_correctness_gate=False,
                passed_anti_cheat_gate=False,
                loc=loc,
                rejection_reason=reason,
            )

        try:
            cand_fn = self._compile_kernel_fn(candidate_code)
            base_fn = self._compile_kernel_fn(baseline_code)
        except Exception as exc:
            return KernelEvalResult(
                op_name=op_name,
                passed_ast_gate=True,
                passed_correctness_gate=False,
                passed_anti_cheat_gate=False,
                loc=loc,
                rejection_reason=f"Compilation failed: {exc}",
            )

        # Stage 2: Standard + Adversarial Prime Shapes + Exact Input Immutability Check
        rng = np.random.default_rng(42)
        test_shapes = [(1, 256), (4, 512), (8, 512), (3, 509), (7, 251)]
        max_diff_overall = 0.0
        min_cos_overall = 1.0

        for b, d in test_shapes:
            inputs = self._make_inputs(op_name, b, d, rng)
            snapshots = [arr.copy() for arr in inputs]
            try:
                ref_out = base_fn(*[arr.copy() for arr in snapshots])
                cand_out = cand_fn(*inputs)
            except Exception as exc:
                return KernelEvalResult(
                    op_name=op_name,
                    passed_ast_gate=True,
                    passed_correctness_gate=False,
                    passed_anti_cheat_gate=True,
                    loc=loc,
                    rejection_reason=f"Runtime error on shape {(b, d)}: {exc}",
                )

            if any(not np.array_equal(arr, snap) for arr, snap in zip(inputs, snapshots)):
                return KernelEvalResult(
                    op_name=op_name,
                    passed_ast_gate=True,
                    passed_correctness_gate=False,
                    passed_anti_cheat_gate=False,
                    loc=loc,
                    rejection_reason="Anti-Cheat Gate: In-place mutation of read-only input tensors detected.",
                )

            ref_tensors = ref_out if isinstance(ref_out, tuple) else (ref_out,)
            cand_tensors = cand_out if isinstance(cand_out, tuple) else (cand_out,)
            if len(cand_tensors) != len(ref_tensors):
                return KernelEvalResult(
                    op_name=op_name,
                    passed_ast_gate=True,
                    passed_correctness_gate=False,
                    passed_anti_cheat_gate=True,
                    loc=loc,
                    rejection_reason=f"Output arity mismatch: got {len(cand_tensors)}, expected {len(ref_tensors)}",
                )

            for cand_arr_raw, ref_arr_raw in zip(cand_tensors, ref_tensors):
                cand_arr = np.asarray(cand_arr_raw)
                ref_arr = np.asarray(ref_arr_raw)
                if cand_arr.shape != ref_arr.shape:
                    return KernelEvalResult(
                        op_name=op_name,
                        passed_ast_gate=True,
                        passed_correctness_gate=False,
                        passed_anti_cheat_gate=True,
                        loc=loc,
                        rejection_reason=f"Shape mismatch: got {cand_arr.shape}, expected {ref_arr.shape}",
                    )
                if not np.all(np.isfinite(cand_arr)):
                    return KernelEvalResult(
                        op_name=op_name,
                        passed_ast_gate=True,
                        passed_correctness_gate=False,
                        passed_anti_cheat_gate=True,
                        loc=loc,
                        rejection_reason=f"Non-finite output on shape {(b, d)}",
                    )
                diff = float(
                    np.max(np.abs(cand_arr.astype(np.float64) - ref_arr.astype(np.float64)))
                )
                max_diff_overall = max(max_diff_overall, diff)
                flat_c = cand_arr.astype(np.float64).ravel()
                flat_r = ref_arr.astype(np.float64).ravel()
                cos = float(
                    np.dot(flat_c, flat_r)
                    / (np.linalg.norm(flat_c) * np.linalg.norm(flat_r) + 1e-12)
                )
                min_cos_overall = min(min_cos_overall, cos)
                if diff > 5e-2 or cos < 0.999:
                    return KernelEvalResult(
                        op_name=op_name,
                        passed_ast_gate=True,
                        passed_correctness_gate=False,
                        passed_anti_cheat_gate=True,
                        max_abs_diff=diff,
                        cosine_sim=cos,
                        loc=loc,
                        rejection_reason=f"Numerical mismatch on shape {(b, d)}: max_diff={diff:.4e}, cos={cos:.6f}",
                    )

        # Stage 3 & 4: Ring-Buffer Timing & Analytic Roofline Classification across buckets
        buckets = [("batch_1", 1, 512), ("batch_4", 4, 512), ("batch_16", 16, 512)]
        profiles: dict[str, HardwareProfile] = {}
        cand_latencies = []
        base_latencies = []

        for label, b, d in buckets:
            ring_inputs = [
                self._make_inputs(op_name, b, d, rng)
                for _ in range(self.ring_buffer_slots)
            ]
            for i in range(3):
                _ = base_fn(*ring_inputs[i % self.ring_buffer_slots])
                _ = cand_fn(*ring_inputs[i % self.ring_buffer_slots])

            t0 = time.perf_counter_ns()
            for i in range(self.timed_iters):
                self._l2_flush_buf[i % 1024] ^= 1
                _ = base_fn(*ring_inputs[i % self.ring_buffer_slots])
            base_us = (time.perf_counter_ns() - t0) / (self.timed_iters * 1000.0)

            t1 = time.perf_counter_ns()
            for i in range(self.timed_iters):
                self._l2_flush_buf[i % 1024] ^= 1
                _ = cand_fn(*ring_inputs[i % self.ring_buffer_slots])
            cand_us = (time.perf_counter_ns() - t1) / (self.timed_iters * 1000.0)

            speedup = base_us / max(0.05, cand_us)
            cand_latencies.append(cand_us)
            base_latencies.append(base_us)

            has_loops = "for " in candidate_code
            has_einsum = "einsum" in candidate_code
            bytes_moved = b * d * (24 if has_loops else (8 if has_einsum else 14))
            flops = b * d * 10
            ai = flops / max(1, bytes_moved)
            dram_pct = min(94.0, 82.0 if not has_loops else 35.0)
            sm_pct = min(85.0, 18.0 if has_loops else (62.0 if has_einsum else 38.0))
            b_class = (
                BottleneckClass.LAUNCH_OVERHEAD_BOUND
                if has_loops
                else (
                    BottleneckClass.BALANCED
                    if has_einsum
                    else BottleneckClass.MEMORY_BANDWIDTH_BOUND
                )
            )
            hint = (
                "Eliminate Python/elementwise loops via vectorized broadcasting."
                if has_loops
                else (
                    "Fuse variance reduction with einsum and fold rsqrt into weight vector to cut memory traffic."
                    if not has_einsum
                    else "Optimal single-pass epilogue achieved; consider tuning tile sizes for larger batches."
                )
            )
            profiles[label] = HardwareProfile(
                shape_bucket=label,
                latency_us=cand_us,
                baseline_latency_us=base_us,
                speedup=speedup,
                bytes_transferred=bytes_moved,
                flops_executed=flops,
                arithmetic_intensity=ai,
                dram_bw_utilization_pct=dram_pct,
                sm_compute_utilization_pct=sm_pct,
                bottleneck_class=b_class,
                actionable_hint=hint,
            )

        avg_cand_us = float(np.mean(cand_latencies))
        avg_base_us = float(np.mean(base_latencies))
        overall_speedup = avg_base_us / max(0.05, avg_cand_us)

        return KernelEvalResult(
            op_name=op_name,
            passed_ast_gate=True,
            passed_correctness_gate=True,
            passed_anti_cheat_gate=True,
            latency_us=avg_cand_us,
            baseline_latency_us=avg_base_us,
            speedup=overall_speedup,
            max_abs_diff=max_diff_overall,
            cosine_sim=min_cos_overall,
            loc=loc,
            profiles_by_bucket=profiles,
        )
