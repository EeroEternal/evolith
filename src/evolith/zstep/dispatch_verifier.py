from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field


class DispatchVerificationError(RuntimeError):
    pass


class DispatchVerificationReport(BaseModel):
    passed: bool
    checked_pairs: int = 0
    verified_selectors: list[str] = Field(default_factory=list)
    verified_thresholds: dict[str, object] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


_SAFE_BUILTINS: dict[str, object] = {
    "ValueError": ValueError,
    "min": min,
    "max": max,
    "len": len,
    "range": range,
    "int": int,
    "tuple": tuple,
    "list": list,
    "sorted": sorted,
    "abs": abs,
}

_BUCKET_CONST_NAMES = frozenset(
    {
        "DECODE_GRAPH_BUCKETS",
        "DSV41_DECODE_BUCKETS",
        "_DSV41_DECODE_BUCKETS",
    }
)
_SELECTOR_FN_NAMES = frozenset(
    {
        "select_decode_bucket",
        "_select_decode_bucket",
        "select_dsv41_decode_bucket",
    }
)


def _extract_literal_assigns(tree: ast.Module, target_names: set[str] | frozenset[str]) -> dict[str, object]:
    found: dict[str, object] = {}
    for node in tree.body:
        name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
            value_node = node.value
        if name in target_names and value_node is not None:
            try:
                found[name] = ast.literal_eval(value_node)
            except (ValueError, SyntaxError):
                pass
    return found


def _compile_isolated_selector(
    tree: ast.Module, fn_name: str, constants: dict[str, object]
) -> Callable[[int, int], int] | None:
    fn_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == fn_name
        ),
        None,
    )
    if fn_node is None:
        return None
    stripped = ast.FunctionDef(
        name=fn_node.name,
        args=fn_node.args,
        body=fn_node.body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    for child in ast.walk(stripped):
        if isinstance(child, ast.arg):
            child.annotation = None
    mod = ast.Module(body=[stripped], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, object] = {"__builtins__": dict(_SAFE_BUILTINS), **constants}
    exec(compile(mod, filename=f"<dispatch:{fn_name}>", mode="exec"), ns)
    fn = ns.get(fn_name)
    return fn if callable(fn) else None  # type: ignore[return-value]


def verify_bucket_selector_exhaustively(
    selector: Callable[[int, int], int],
    ladder: tuple[int, ...],
    label: str,
    max_cap: int = 128,
) -> tuple[int, list[str]]:
    """
    Exhaustively proves Total Coverage, Bounds Soundness, Monotonicity, Idempotence,
    and Captured-Graph Membership over the complete integer domain `max_bucket in 1..=128`
    x `num_seqs in 0..=max_bucket+1` (modeled on vstack VS-008 dispatch table verification).
    """
    violations: list[str] = []
    checked = 0
    if not ladder or any(not isinstance(b, int) or b <= 0 for b in ladder):
        return 0, [f"{label}: bucket ladder `{ladder}` must be a non-empty tuple of positive ints"]
    if list(ladder) != sorted(set(ladder)):
        violations.append(f"{label}: bucket ladder `{ladder}` must be strictly increasing without duplicates")

    for max_b in range(1, max_cap + 1):
        captured = {b for b in ladder if b <= max_b} | {max_b}
        for out_of_range in (0, -1, max_b + 1):
            checked += 1
            try:
                got = selector(out_of_range, max_b)
                violations.append(
                    f"{label}({out_of_range}, {max_b}) returned {got} instead of raising ValueError"
                )
                if len(violations) >= 5:
                    return checked, violations
            except ValueError:
                pass
            except Exception as exc:
                violations.append(
                    f"{label}({out_of_range}, {max_b}) raised unexpected {type(exc).__name__}: {exc}"
                )
                return checked, violations

        prev_b = 0
        for n in range(1, max_b + 1):
            checked += 1
            try:
                b = selector(n, max_b)
            except Exception as exc:
                violations.append(f"{label}({n}, {max_b}) failed: {type(exc).__name__}: {exc}")
                return checked, violations
            if not isinstance(b, int) or isinstance(b, bool):
                violations.append(f"{label}({n}, {max_b}) returned non-int `{b!r}`")
                return checked, violations
            if not (n <= b <= max_b):
                violations.append(
                    f"{label}({n}, {max_b}) = {b} violates bounds `{n} <= bucket <= {max_b}`"
                )
                return checked, violations
            if b not in captured:
                violations.append(
                    f"{label}({n}, {max_b}) = {b} is not in captured graph set {sorted(captured)}"
                )
                return checked, violations
            if b < prev_b:
                violations.append(
                    f"{label} non-monotonic at max_b={max_b}: f({n - 1})={prev_b} > f({n})={b}"
                )
                return checked, violations
            if selector(b, max_b) != b:
                violations.append(
                    f"{label} non-idempotent at max_b={max_b}: f(f({n})) = f({b}) != {b}"
                )
                return checked, violations
            prev_b = b

    return checked, violations


def run_dispatch_verification(
    candidate_dir: Path,
    evidence_dir: Path | None = None,
) -> DispatchVerificationReport:
    """
    Statically & exhaustively verifies SM120 decode bucket selectors and hybrid
    AllReduce/MegaGate/Cutlass dispatch thresholds inside `candidate_dir`.
    No-ops cleanly when synthetic test workspaces do not contain zstep Python files.
    """
    violations: list[str] = []
    verified_selectors: list[str] = []
    verified_thresholds: dict[str, object] = {}
    checked_pairs = 0

    base_ladder: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    for rel_path in (
        "worker/zstep_worker/shm_bridge.py",
        "worker/zstep_worker/models/deepseek_v41.py",
    ):
        file_path = candidate_dir / rel_path
        if not file_path.is_file():
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=rel_path)
        consts = _extract_literal_assigns(tree, _BUCKET_CONST_NAMES | {"MAX_DECODE_GRAPH_SEQS"})
        ladder_obj = (
            consts.get("DSV41_DECODE_BUCKETS")
            or consts.get("_DSV41_DECODE_BUCKETS")
            or consts.get("DECODE_GRAPH_BUCKETS")
        )
        if isinstance(ladder_obj, (tuple, list)):
            ladder = tuple(int(x) for x in ladder_obj)
            if rel_path.endswith("shm_bridge.py"):
                base_ladder = ladder
            verified_thresholds[f"{rel_path}:ladder"] = list(ladder)
            if not ladder or any(b <= 0 for b in ladder) or list(ladder) != sorted(set(ladder)):
                violations.append(
                    f"{rel_path}: bucket ladder `{ladder}` must be strictly increasing positive ints"
                )
        else:
            ladder = base_ladder

        consts_with_default = {"DECODE_GRAPH_BUCKETS": base_ladder, **consts}
        for fn_name in sorted(_SELECTOR_FN_NAMES):
            fn = _compile_isolated_selector(tree, fn_name, consts_with_default)
            if fn is None:
                continue
            pairs, errs = verify_bucket_selector_exhaustively(
                fn, ladder, f"{rel_path}:{fn_name}"
            )
            checked_pairs += pairs
            verified_selectors.append(f"{rel_path}:{fn_name}")
            violations.extend(errs)

    b12x_path = candidate_dir / "worker/zstep_worker/kernels_sm120/b12x_dsv41.py"
    if b12x_path.is_file():
        b12x_tree = ast.parse(b12x_path.read_text(encoding="utf-8"), filename=str(b12x_path))
        b12x_consts = _extract_literal_assigns(
            b12x_tree,
            {"DEFAULT_MEGA_GATE_MIN_TOKENS", "_CUTLASS_LARGE_M_THRESHOLDS"},
        )
        if "DEFAULT_MEGA_GATE_MIN_TOKENS" in b12x_consts:
            mg = b12x_consts["DEFAULT_MEGA_GATE_MIN_TOKENS"]
            verified_thresholds["DEFAULT_MEGA_GATE_MIN_TOKENS"] = mg
            if not isinstance(mg, int) or isinstance(mg, bool) or not (1 <= mg <= 4096):
                violations.append(
                    f"b12x_dsv41.py: DEFAULT_MEGA_GATE_MIN_TOKENS={mg!r} outside valid [1, 4096]"
                )
        if "_CUTLASS_LARGE_M_THRESHOLDS" in b12x_consts:
            ct = b12x_consts["_CUTLASS_LARGE_M_THRESHOLDS"]
            if isinstance(ct, dict):
                verified_thresholds["_CUTLASS_LARGE_M_THRESHOLDS"] = {
                    f"{k}": v for k, v in ct.items()
                }
                for shape_key, thresh in ct.items():
                    if not isinstance(thresh, int) or not (16 <= thresh <= 65536):
                        violations.append(
                            f"b12x_dsv41.py: _CUTLASS_LARGE_M_THRESHOLDS[{shape_key!r}]={thresh!r} outside [16, 65536]"
                        )

    shm_ar_path = candidate_dir / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    if shm_ar_path.is_file():
        ar_tree = ast.parse(shm_ar_path.read_text(encoding="utf-8"), filename=str(shm_ar_path))
        ar_consts = _extract_literal_assigns(ar_tree, {"DEFAULT_ONE_SHOT_MAX_VECS"})
        if "DEFAULT_ONE_SHOT_MAX_VECS" in ar_consts:
            vecs = ar_consts["DEFAULT_ONE_SHOT_MAX_VECS"]
            verified_thresholds["DEFAULT_ONE_SHOT_MAX_VECS"] = vecs
            if (
                not isinstance(vecs, int)
                or isinstance(vecs, bool)
                or not (64 <= vecs <= 65536)
                or vecs % 8 != 0
            ):
                violations.append(
                    f"shm_allreduce.py: DEFAULT_ONE_SHOT_MAX_VECS={vecs!r} must be in [64, 65536] and divisible by 8"
                )

    report = DispatchVerificationReport(
        passed=not violations,
        checked_pairs=checked_pairs,
        verified_selectors=verified_selectors,
        verified_thresholds=verified_thresholds,
        violations=violations,
    )
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "dispatch-verification.json").write_text(
            json.dumps(report.model_dump(), indent=2),
            encoding="utf-8",
        )
    if violations:
        raise DispatchVerificationError(
            "Dispatch verification failed: " + "; ".join(violations)
        )
    return report
