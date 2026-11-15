from __future__ import annotations

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


def _discover_local_python_tests(candidate_dir: Path) -> list[str]:
    tests_root = candidate_dir / "worker/tests"
    if not tests_root.is_dir():
        return []
    candidates = [
        "worker/tests/test_b12x_dsv41_lifecycle.py",
        "worker/tests/test_shm_layout_parity.py",
    ]
    return [rel for rel in candidates if (candidate_dir / rel).is_file()]


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
    # On macOS, typecheck the full workspace (including Linux-only zs-shm/zs-scheduler/zs-server)
    if is_darwin and _has_linux_x86_rust_target():
        commands.append(
            (["cargo", "check", "--workspace", "--target", "x86_64-unknown-linux-gnu"], None)
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


def _check_parity_guards(
    parity_path: Path,
    reasons: list[str],
    *,
    max_cross_mode_logit_drift: float = 8.0,
) -> None:
    recompilations, rates = load_parity(parity_path)
    if recompilations:
        reasons.append(f"unexpected Triton recompilations: {recompilations}")
    for mode in ("single", "batched", "chunked"):
        if rates.get(mode) != 1.0:
            reasons.append(f"prime_squares_table {mode} exact match is not 100%")

    payload = json.loads(parity_path.read_text(encoding="utf-8"))
    chunked_sched = payload.get("chunked_schedule")
    if isinstance(chunked_sched, dict):
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
            if not isinstance(pair, dict):
                continue
            left = pair.get("left_top2")
            right = pair.get("right_top2")
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            if "top1_logit" not in left or "top1_logit" not in right:
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

    candidate = load_benchmark(candidate_metrics_path)
    if not candidate:
        reasons.append("screen benchmark produced no tier metrics")
        return ScreenDecision(passed=False, reasons=reasons)

    gains: list[float] = []
    delta_parts: list[str] = []
    if baseline_metrics_path is not None and baseline_metrics_path.is_file():
        baseline = load_benchmark(baseline_metrics_path)
        for missing_label in sorted(set(baseline) - set(candidate)):
            reasons.append(f"screen candidate missing tier {missing_label}")
        for label in sorted(set(baseline) & set(candidate)):
            base_tp = _median_metric(baseline[label], "output_throughput_tok_s")
            cand_tp = _median_metric(candidate[label], "output_throughput_tok_s")
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
                reg = cand_lat / base_lat - 1.0
                if reg > task.max_screen_latency_regression:
                    reasons.append(
                        f"screen {label} {field} regressed {reg:.2%} "
                        f"(limit {task.max_screen_latency_regression:.2%})"
                    )
    else:
        for label, rows in sorted(candidate.items()):
            cand_tp = _median_metric(rows, "output_throughput_tok_s")
            if cand_tp <= 0:
                reasons.append(f"screen {label} non-positive throughput {cand_tp}")
            delta_parts.append(f"{label}:{cand_tp:.1f} tok/s")

    geomean_gain = (
        math.exp(sum(math.log1p(g) for g in gains) / len(gains)) - 1.0
        if gains
        else 0.0
    )
    return ScreenDecision(
        passed=not reasons,
        gain=geomean_gain,
        metric_delta="screen[" + ", ".join(delta_parts) + "]",
        reasons=reasons,
    )


def evaluate_candidate(
    baseline: dict[str, list[TierMetric]],
    candidate: dict[str, list[TierMetric]],
    parity_path: Path,
    task: ZStepRSITask,
    *,
    changed_lines: int = 0,
) -> VerificationDecision:
    reasons: list[str] = []
    labels = sorted(set(baseline) & set(candidate))
    for label in sorted(set(baseline) - set(candidate)):
        reasons.append(f"candidate missing tier {label}")
    for label in sorted(set(candidate) - set(baseline)):
        reasons.append(f"baseline missing tier {label}")

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
            regression = candidate_value / base_value - 1.0
            if regression > limit:
                reasons.append(f"{label} {field} regressed {regression:.2%}")

    geomean_gain = (
        math.exp(sum(math.log1p(gain) for gain in gains) / len(gains)) - 1.0
        if gains
        else -1.0
    )
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

    metric_delta = (
        f"geomean={geomean_gain:+.2%} (net_gate={net_gate_ratio:.4f}) "
        f"[{', '.join(tier_deltas)}]"
    )
    return VerificationDecision(
        accepted=not reasons,
        geomean_gain=geomean_gain,
        net_gate_ratio=net_gate_ratio,
        metric_delta=metric_delta,
        reasons=reasons,
    )
