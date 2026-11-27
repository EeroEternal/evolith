from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

from evolith.controller.engine import EvolithRSIEngine
from evolith.zstep.task import (
    ScreenDecision,
    TierMetric,
    VerificationDecision,
    ZStepRSITask,
)


def _has_linux_x86_rust_target() -> bool:
    if shutil.which("rustup") is None:
        return False
    try:
        res = subprocess.run(
            ["rustup", "target", "list", "--installed"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return res.returncode == 0 and "x86_64-unknown-linux-gnu" in res.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return False


def _file_imports_torch(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "torch" or node.module.startswith("torch.")):
                return True
    return False


def _discover_local_python_tests(candidate_dir: Path) -> list[str]:
    tests_root = candidate_dir / "worker/tests"
    if not tests_root.is_dir():
        return []
    has_torch = importlib.util.find_spec("torch") is not None
    return sorted(
        f"worker/tests/{p.name}"
        for p in tests_root.glob("test_*.py")
        if p.is_file() and (has_torch or not _file_imports_torch(p))
    )


def run_local_gates(candidate_dir: Path, task: ZStepRSITask, evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    is_darwin = platform.system() == "Darwin"
    cargo_test = (
        ["cargo", "test", "-p", "zs-kv", "-p", "zs-wire"]
        if is_darwin
        else ["cargo", "test", "--workspace"]
    )
    commands: list[tuple[list[str], dict[str, str] | None]] = [
        (["cargo", "fmt", "--all", "--check"], None),
        (cargo_test, None),
    ]
    # On macOS, typecheck Linux crates (full workspace when a Linux C cross-compiler
    # is installed for C build scripts like `onig_sys`, or pure-Rust crates otherwise)
    if is_darwin and _has_linux_x86_rust_target():
        if shutil.which("x86_64-linux-gnu-gcc") is not None:
            commands.append(
                (
                    [
                        "cargo",
                        "check",
                        "--workspace",
                        "--tests",
                        "--target",
                        "x86_64-unknown-linux-gnu",
                    ],
                    None,
                )
            )
        else:
            commands.append(
                (
                    [
                        "cargo",
                        "check",
                        "--tests",
                        "-p",
                        "zs-kv",
                        "-p",
                        "zs-wire",
                        "-p",
                        "zs-shm",
                        "-p",
                        "zs-scheduler",
                        "--target",
                        "x86_64-unknown-linux-gnu",
                    ],
                    None,
                )
            )
    if shutil.which("ruff") is not None:
        commands.append((["ruff", "check", "worker/"], None))

    py_tests = _discover_local_python_tests(candidate_dir)
    if py_tests:
        py_env = dict(os.environ)
        worker_path = str(candidate_dir / "worker")
        existing_pp = py_env.get("PYTHONPATH", "")
        py_env["PYTHONPATH"] = (
            f"{worker_path}{os.pathsep}{existing_pp}" if existing_pp else worker_path
        )
        commands.append(([sys.executable, "-m", "pytest", "-q", *py_tests], py_env))

    for index, (command, custom_env) in enumerate(commands):
        tool_tag = Path(command[0]).name
        run_kwargs: dict[str, object] = {
            "cwd": candidate_dir,
            "text": True,
            "capture_output": True,
            "timeout": task.local_timeout_sec,
        }
        if custom_env is not None:
            run_kwargs["env"] = custom_env
        result = subprocess.run(command, **run_kwargs)  # type: ignore[arg-type]
        (evidence_dir / f"local-{index:02d}-{tool_tag}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(
                f"Local gate {' '.join(command[:2])} exited {result.returncode}"
            )


def load_benchmark(path: Path) -> dict[str, list[TierMetric]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[TierMetric]] = {}
    for row in payload["tiers"]:
        metric = TierMetric.model_validate(row)
        grouped.setdefault(metric.label, []).append(metric)
    return grouped


def load_parity(path: Path) -> tuple[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prime = next(
        case for case in payload["cases"] if case["name"] == "prime_squares_table"
    )
    rates = {
        mode: float(prime[f"{mode}_match"]["exact_token_match_rate"])
        for mode in ("single", "batched", "chunked")
    }
    return int(payload["triton_recompilations"]), rates


def _median_metric(metrics: list[TierMetric], field: str) -> float:
    return statistics.median(float(getattr(metric, field)) for metric in metrics)


def _validate_positive_finite(
    value: float, label: str, field: str, reasons: list[str]
) -> bool:
    if not math.isfinite(value) or value <= 0.0:
        reasons.append(f"{label} {field} is non-positive or non-finite ({value})")
        return False
    return True


def _safe_geomean_gain(gains: list[float], *, empty_default: float = 0.0) -> float:
    if not gains:
        return empty_default
    if any(not math.isfinite(g) or g <= -1.0 for g in gains):
        return -1.0
    return math.exp(sum(math.log1p(g) for g in gains) / len(gains)) - 1.0


def _check_parity_guards(
    parity_path: Path,
    reasons: list[str],
    *,
    max_cross_mode_logit_drift: float = 8.0,
) -> None:
    if not parity_path.is_file():
        reasons.append(f"parity report missing at {parity_path.name}")
        return
    recompilations, rates = load_parity(parity_path)
    if recompilations:
        reasons.append(f"unexpected Triton recompilations: {recompilations}")
    for mode in ("single", "batched", "chunked"):
        if rates.get(mode) != 1.0:
            reasons.append(f"prime_squares_table {mode} exact match is not 100%")

    payload = json.loads(parity_path.read_text(encoding="utf-8"))
    if "chunked_schedule" in payload or "unchunked_schedule" in payload:
        chunked_sched = payload.get("chunked_schedule")
        if not isinstance(chunked_sched, dict):
            reasons.append("chunked_schedule missing or null in parity report")
        else:
            cp_steps = int(chunked_sched.get("chunked_prefill_steps", 0))
            mixed_steps = int(chunked_sched.get("decode_steps_with_partial_prefill", 0))
            if cp_steps <= 0 or mixed_steps <= 0:
                reasons.append(
                    "chunked_schedule failed exposure gate: "
                    f"chunked_prefill_steps={cp_steps}, decode_steps_with_partial_prefill={mixed_steps}"
                )

    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_name = str(case.get("name", "unknown"))
        svb = case.get("single_vs_batched_first_mismatch")
        bvc = case.get("batched_vs_chunked_first_mismatch")
        if case_name == "prime_squares_table" and (svb is not None or bvc is not None):
            reasons.append(
                "prime_squares_table violated cross-mode invariance (single/batched/chunked diverged)"
            )
        for pair_label, pair in (("single_vs_batched", svb), ("batched_vs_chunked", bvc)):
            if pair is None:
                continue
            if not isinstance(pair, dict):
                reasons.append(f"case {case_name} {pair_label} has malformed mismatch entry")
                continue
            left = pair.get("left_top2")
            right = pair.get("right_top2")
            if (
                not isinstance(left, dict)
                or not isinstance(right, dict)
                or "top1_logit" not in left
                or "top1_logit" not in right
            ):
                reasons.append(
                    f"case {case_name} {pair_label} mismatch missing top1_logit metadata"
                )
                continue
            left_logit = float(left["top1_logit"])
            right_logit = float(right["top1_logit"])
            if not (math.isfinite(left_logit) and math.isfinite(right_logit)):
                reasons.append(
                    f"case {case_name} {pair_label} produced non-finite top1_logit"
                )
            else:
                drift = abs(left_logit - right_logit)
                if drift > max_cross_mode_logit_drift:
                    reasons.append(
                        f"case {case_name} {pair_label} logit drift {drift:.2f} "
                        f"exceeds limit {max_cross_mode_logit_drift:.2f}"
                    )


def evaluate_screen(
    candidate_metrics_path: Path,
    parity_path: Path | None,
    task: ZStepRSITask,
    baseline_metrics_path: Path | None = None,
    *,
    require_parity: bool = False,
) -> ScreenDecision:
    """
    Validates screen-stage correctness, Triton zero-recompilation, and non-regression
    BEFORE committing ~60 minutes of 8xGPU interleaved verification time.
    """
    reasons: list[str] = []
    if parity_path is not None and parity_path.is_file():
        _check_parity_guards(
            parity_path,
            reasons,
            max_cross_mode_logit_drift=task.max_cross_mode_logit_drift,
        )
    elif require_parity:
        reasons.append("screen parity report missing")

    if not candidate_metrics_path.is_file():
        reasons.append("screen benchmark metrics missing")
        return ScreenDecision(passed=False, reasons=reasons)

    candidate = load_benchmark(candidate_metrics_path)
    if not candidate:
        reasons.append("screen benchmark produced no tier metrics")
        return ScreenDecision(passed=False, reasons=reasons)

    for label, rows in sorted(candidate.items()):
        for row in rows:
            for field in (
                "output_throughput_tok_s",
                "median_ttft_ms",
                "mean_tpot_ms",
                "p99_itl_ms",
            ):
                _validate_positive_finite(
                    float(getattr(row, field)), f"screen {label}", field, reasons
                )
            if row.idle_kv_cache_usage is not None and (
                not math.isfinite(row.idle_kv_cache_usage) or row.idle_kv_cache_usage > 0.0
            ):
                reasons.append(
                    f"screen {label} leaked idle KV cache ({row.idle_kv_cache_usage:.4f} > 0.0)"
                )
                break

    gains: list[float] = []
    delta_parts: list[str] = []
    if baseline_metrics_path is not None and baseline_metrics_path.is_file():
        baseline = load_benchmark(baseline_metrics_path)
        for missing_label in sorted(set(baseline) - set(candidate)):
            reasons.append(f"screen candidate missing tier {missing_label}")
        for label in sorted(set(baseline) & set(candidate)):
            base_tp = _median_metric(baseline[label], "output_throughput_tok_s")
            cand_tp = _median_metric(candidate[label], "output_throughput_tok_s")
            if not (
                _validate_positive_finite(base_tp, f"screen baseline {label}", "output_throughput_tok_s", reasons)
                and _validate_positive_finite(cand_tp, f"screen {label}", "output_throughput_tok_s", reasons)
            ):
                continue
            gain = cand_tp / base_tp - 1.0
            gains.append(gain)
            delta_parts.append(f"{label}:{gain:+.2%} ({cand_tp:.1f} tok/s)")
            if gain < -task.max_screen_throughput_regression:
                reasons.append(
                    f"screen {label} throughput regressed {gain:.2%} "
                    f"(limit {-task.max_screen_throughput_regression:.2%})"
                )
            for field in ("median_ttft_ms", "mean_tpot_ms", "p99_itl_ms"):
                base_lat = _median_metric(baseline[label], field)
                cand_lat = _median_metric(candidate[label], field)
                if not (
                    _validate_positive_finite(base_lat, f"screen baseline {label}", field, reasons)
                    and _validate_positive_finite(cand_lat, f"screen {label}", field, reasons)
                ):
                    continue
                reg = cand_lat / base_lat - 1.0
                if reg > task.max_screen_latency_regression:
                    reasons.append(
                        f"screen {label} {field} regressed {reg:.2%} "
                        f"(limit {task.max_screen_latency_regression:.2%})"
                    )
    else:
        for label, rows in sorted(candidate.items()):
            cand_tp = _median_metric(rows, "output_throughput_tok_s")
            if cand_tp <= 0 or not math.isfinite(cand_tp):
                reasons.append(f"screen {label} non-positive throughput {cand_tp}")
            delta_parts.append(f"{label}:{cand_tp:.1f} tok/s")

    geomean_gain = _safe_geomean_gain(gains, empty_default=0.0)
    if (
        baseline_metrics_path is not None
        and baseline_metrics_path.is_file()
        and gains
        and task.min_screen_gain is not None
        and geomean_gain < task.min_screen_gain
    ):
        reasons.append(
            f"screen geomean gain {geomean_gain:+.2%} below required min_screen_gain "
            f"{task.min_screen_gain:+.2%}"
        )
    return ScreenDecision(
        passed=not reasons,
        gain=geomean_gain,
        metric_delta="screen[" + ", ".join(delta_parts) + "]",
        reasons=reasons,
    )


def evaluate_hot_prescreen(
    probe_json_path: Path,
    *,
    min_speedup: float = 1.01,
) -> ScreenDecision:
    """Validate fast isolated GPU operator/collective hot pre-screen before 475 GiB cold loads."""
    if not probe_json_path.is_file():
        return ScreenDecision(
            passed=False,
            gain=-1.0,
            reasons=[f"hot pre-screen output missing at {probe_json_path.name}"],
        )
    try:
        payload = json.loads(probe_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ScreenDecision(
            passed=False,
            gain=-1.0,
            reasons=[f"hot pre-screen output malformed JSON: {exc}"],
        )
    if not isinstance(payload, dict):
        return ScreenDecision(
            passed=False,
            gain=-1.0,
            reasons=["hot pre-screen JSON root must be an object"],
        )
    reasons: list[str] = []
    if payload.get("passed") is False:
        err = str(payload.get("error") or payload.get("reason") or "correctness/smoke failed")
        reasons.append(f"hot pre-screen failed check: {err}")
    raw_speedup = payload.get("geomean_speedup")
    if not isinstance(raw_speedup, (int, float)) or not math.isfinite(float(raw_speedup)) or float(raw_speedup) <= 0.0:
        reasons.append(f"hot pre-screen missing positive geomean_speedup (got {raw_speedup!r})")
        speedup = 0.0
    else:
        speedup = float(raw_speedup)
        if speedup < min_speedup:
            reasons.append(
                f"hot pre-screen geomean_speedup {speedup:.4f}x below min_hot_prescreen_speedup {min_speedup:.4f}x"
            )
    gain = speedup - 1.0 if speedup > 0.0 else -1.0
    summary = str(payload.get("summary") or f"speedup={speedup:.4f}x")
    return ScreenDecision(
        passed=not reasons,
        gain=gain,
        metric_delta=f"hot_prescreen[{summary}]",
        reasons=reasons,
    )


def evaluate_secondary_guard(
    baseline: dict[str, list[TierMetric]],
    candidate: dict[str, list[TierMetric]],
    parity_path: Path | None,
    task: ZStepRSITask,
    *,
    profile_id: str = "secondary",
) -> VerificationDecision:
    """Enforce zero-regression and greedy parity on secondary guard regimes without requiring primary gain."""
    guard_task = task.model_copy(
        update={
            "min_geomean_gain": -task.max_tier_throughput_regression,
            "complexity_penalty_weight": 0.0,
            "min_paired_se_z": 0.0,
        }
    )
    reasons: list[str] = []
    if not baseline:
        reasons.append(f"{profile_id}: baseline benchmark produced no tier metrics")
    if not candidate:
        reasons.append(f"{profile_id}: candidate benchmark produced no tier metrics")
    labels = sorted(set(baseline) & set(candidate))
    for label in sorted(set(baseline) - set(candidate)):
        reasons.append(f"{profile_id}: candidate missing tier {label}")
    gains: list[float] = []
    deltas: list[str] = []
    latency_limits = {
        "median_ttft_ms": guard_task.max_ttft_regression,
        "mean_tpot_ms": guard_task.max_tpot_regression,
        "p99_itl_ms": guard_task.max_p99_itl_regression,
    }
    for label in labels:
        b_tp = _median_metric(baseline[label], "output_throughput_tok_s")
        c_tp = _median_metric(candidate[label], "output_throughput_tok_s")
        if not (
            _validate_positive_finite(b_tp, f"{profile_id} baseline {label}", "output_throughput_tok_s", reasons)
            and _validate_positive_finite(c_tp, f"{profile_id} {label}", "output_throughput_tok_s", reasons)
        ):
            continue
        gain = c_tp / b_tp - 1.0
        gains.append(gain)
        deltas.append(f"{label}:{gain:+.2%}")
        if gain < -guard_task.max_tier_throughput_regression:
            reasons.append(
                f"{profile_id} {label} throughput regressed {gain:.2%} "
                f"(limit {-guard_task.max_tier_throughput_regression:.2%})"
            )
        for field, limit in latency_limits.items():
            b_lat = _median_metric(baseline[label], field)
            c_lat = _median_metric(candidate[label], field)
            if not (math.isfinite(b_lat) and b_lat > 0 and math.isfinite(c_lat) and c_lat > 0):
                continue
            reg = c_lat / b_lat - 1.0
            if reg > limit:
                reasons.append(f"{profile_id} {label} {field} regressed {reg:.2%} (limit {limit:.2%})")
    if parity_path is not None:
        _check_parity_guards(
            parity_path,
            reasons,
            max_cross_mode_logit_drift=guard_task.max_cross_mode_logit_drift,
        )
    geomean_gain = _safe_geomean_gain(gains, empty_default=0.0)
    return VerificationDecision(
        accepted=not reasons,
        geomean_gain=geomean_gain,
        net_gate_ratio=1.0 + geomean_gain,
        metric_delta=f"{profile_id}_guard={geomean_gain:+.2%} [{', '.join(deltas)}]",
        reasons=reasons,
    )


def compute_paired_round_stats(
    baseline: dict[str, list[TierMetric]],
    candidate: dict[str, list[TierMetric]],
) -> tuple[float, float, float] | None:
    """Compute paired per-round geomean gain ``(mean_gain, se_gain, lcb_2se)`` (Evolving Infra Eq. 6)."""
    labels = sorted(set(baseline) & set(candidate))
    if not labels:
        return None
    common_repeats: set[int] | None = None
    by_role_label_rep: dict[str, dict[str, dict[int, float]]] = {"b": {}, "c": {}}
    for label in labels:
        b_map = {
            r.repeat: float(r.output_throughput_tok_s)
            for r in baseline[label]
            if math.isfinite(r.output_throughput_tok_s) and r.output_throughput_tok_s > 0.0
        }
        c_map = {
            r.repeat: float(r.output_throughput_tok_s)
            for r in candidate[label]
            if math.isfinite(r.output_throughput_tok_s) and r.output_throughput_tok_s > 0.0
        }
        reps = set(b_map) & set(c_map)
        common_repeats = reps if common_repeats is None else (common_repeats & reps)
        by_role_label_rep["b"][label] = b_map
        by_role_label_rep["c"][label] = c_map
    if not common_repeats or len(common_repeats) < 2:
        return None
    round_gains: list[float] = []
    for rep in sorted(common_repeats):
        tier_gains = [
            by_role_label_rep["c"][lbl][rep] / by_role_label_rep["b"][lbl][rep] - 1.0
            for lbl in labels
        ]
        round_gains.append(_safe_geomean_gain(tier_gains, empty_default=0.0))
    m = len(round_gains)
    mean_g = statistics.mean(round_gains)
    std_g = statistics.stdev(round_gains)
    se_g = std_g / math.sqrt(m)
    return mean_g, se_g, mean_g - 2.0 * se_g


def evaluate_candidate(
    baseline: dict[str, list[TierMetric]],
    candidate: dict[str, list[TierMetric]],
    parity_path: Path,
    task: ZStepRSITask,
    *,
    changed_lines: int = 0,
) -> VerificationDecision:
    reasons: list[str] = []
    if not baseline:
        reasons.append("baseline benchmark produced no tier metrics")
    if not candidate:
        reasons.append("candidate benchmark produced no tier metrics")
    labels = sorted(set(baseline) & set(candidate))
    for label in sorted(set(baseline) - set(candidate)):
        reasons.append(f"candidate missing tier {label}")
    for label in sorted(set(candidate) - set(baseline)):
        reasons.append(f"baseline missing tier {label}")
    for role_name, grouped in (("baseline", baseline), ("candidate", candidate)):
        for label, rows in sorted(grouped.items()):
            for row in rows:
                for field in (
                    "output_throughput_tok_s",
                    "median_ttft_ms",
                    "mean_tpot_ms",
                    "p99_itl_ms",
                ):
                    _validate_positive_finite(
                        float(getattr(row, field)), f"{role_name} {label}", field, reasons
                    )
    for label, rows in sorted(candidate.items()):
        for row in rows:
            if row.idle_kv_cache_usage is not None and (
                not math.isfinite(row.idle_kv_cache_usage) or row.idle_kv_cache_usage > 0.0
            ):
                reasons.append(
                    f"{label} leaked idle KV cache ({row.idle_kv_cache_usage:.4f} > 0.0)"
                )
                break

    gains: list[float] = []
    tier_deltas: list[str] = []
    latency_limits = {
        "median_ttft_ms": task.max_ttft_regression,
        "mean_tpot_ms": task.max_tpot_regression,
        "p99_itl_ms": task.max_p99_itl_regression,
    }
    for label in labels:
        base_throughput = _median_metric(baseline[label], "output_throughput_tok_s")
        candidate_throughput = _median_metric(candidate[label], "output_throughput_tok_s")
        if not (
            math.isfinite(base_throughput)
            and base_throughput > 0.0
            and math.isfinite(candidate_throughput)
            and candidate_throughput > 0.0
        ):
            continue
        if len(candidate[label]) >= 2:
            tp_samples = [float(m.output_throughput_tok_s) for m in candidate[label]]
            mean_tp = statistics.mean(tp_samples)
            if mean_tp > 0:
                cv = statistics.pstdev(tp_samples) / mean_tp
                if cv > task.max_tier_cv:
                    reasons.append(
                        f"{label} throughput dispersion CV={cv:.2%} exceeds limit {task.max_tier_cv:.2%}"
                    )
        gain = candidate_throughput / base_throughput - 1.0
        gains.append(gain)
        base_tpot = _median_metric(baseline[label], "mean_tpot_ms")
        cand_tpot = _median_metric(candidate[label], "mean_tpot_ms")
        tier_deltas.append(
            f"{label}:{gain:+.2%}(TPOT {cand_tpot:.2f} vs {base_tpot:.2f}ms)"
        )
        if gain < -task.max_tier_throughput_regression:
            reasons.append(f"{label} throughput regressed {gain:.2%}")
        for field, limit in latency_limits.items():
            base_value = _median_metric(baseline[label], field)
            candidate_value = _median_metric(candidate[label], field)
            if not (
                math.isfinite(base_value)
                and base_value > 0.0
                and math.isfinite(candidate_value)
                and candidate_value > 0.0
            ):
                continue
            regression = candidate_value / base_value - 1.0
            if regression > limit:
                reasons.append(f"{label} {field} regressed {regression:.2%}")

    geomean_gain = _safe_geomean_gain(gains, empty_default=-1.0)
    if geomean_gain < task.min_geomean_gain:
        reasons.append(f"geomean throughput gain {geomean_gain:.2%} below required minimum")

    passed_occam, net_gate_ratio = EvolithRSIEngine.occams_razor_gate(
        candidate_score=max(0.0, 1.0 + geomean_gain),
        baseline_score=1.0,
        candidate_loc=changed_lines,
        target_loc=task.complexity_free_lines,
        loc_penalty_weight=task.complexity_penalty_weight,
        min_margin=task.min_geomean_gain,
    )
    if geomean_gain >= task.min_geomean_gain and not passed_occam:
        reasons.append(
            f"Occam's Razor complexity gate failed: net_ratio={net_gate_ratio:.4f} "
            f"for {changed_lines} changed lines (target<={task.complexity_free_lines})"
        )

    _check_parity_guards(
        parity_path,
        reasons,
        max_cross_mode_logit_drift=task.max_cross_mode_logit_drift,
    )

    paired = compute_paired_round_stats(baseline, candidate)
    paired_lcb_2se = paired[2] if paired is not None else None
    if paired is not None and task.min_paired_se_z > 0.0:
        mean_g, se_g, _ = paired
        threshold = task.min_paired_se_z * se_g
        if mean_g <= threshold:
            reasons.append(
                f"paired round mean gain {mean_g:+.2%} <= {task.min_paired_se_z:.1f}*SE ({threshold:.2%})"
            )

    lcb_suffix = f", lcb_2se={paired_lcb_2se:+.2%}" if paired_lcb_2se is not None else ""
    metric_delta = (
        f"geomean={geomean_gain:+.2%} (net_gate={net_gate_ratio:.4f}{lcb_suffix}) "
        f"[{', '.join(tier_deltas)}]"
    )
    return VerificationDecision(
        accepted=not reasons,
        geomean_gain=geomean_gain,
        net_gate_ratio=net_gate_ratio,
        paired_lcb_2se=paired_lcb_2se,
        metric_delta=metric_delta,
        reasons=reasons,
    )
