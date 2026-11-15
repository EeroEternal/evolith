from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess

import pytest

from evolith.zstep.gates import (
    evaluate_candidate,
    evaluate_screen,
    load_benchmark,
    run_local_gates,
)
from evolith.zstep.task import ZStepRSITask


def make_task(tmp_path: Path, **overrides: object) -> ZStepRSITask:
    values: dict[str, object] = {
        "project_root": tmp_path,
        "source_repo": tmp_path / "source",
    }
    values.update(overrides)
    return ZStepRSITask(**values)  # type: ignore[arg-type]


def test_local_gates_run_and_preserve_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "ok\n", "")

    monkeypatch.setattr("evolith.zstep.gates.subprocess.run", fake_run)
    monkeypatch.setattr("evolith.zstep.gates.shutil.which", lambda _: None)

    run_local_gates(tmp_path, make_task(tmp_path), tmp_path / "evidence")

    cargo_test = (
        ["cargo", "test", "-p", "zs-kv", "-p", "zs-wire"]
        if platform.system() == "Darwin"
        else ["cargo", "test", "--workspace"]
    )
    assert commands == [["cargo", "fmt", "--all", "--check"], cargo_test]
    assert (tmp_path / "evidence/local-00-cargo.log").read_text() == "ok\n"
    assert (tmp_path / "evidence/local-01-cargo.log").read_text() == "ok\n"


def test_local_gates_include_linux_target_check_ruff_and_python_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["rustup", "target", "list"]:
            return subprocess.CompletedProcess(
                args, 0, "aarch64-apple-darwin\nx86_64-unknown-linux-gnu\n", ""
            )
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "ok\n", "")

    (tmp_path / "worker/tests").mkdir(parents=True)
    (tmp_path / "worker/tests/test_shm_layout_parity.py").write_text("pass\n")

    monkeypatch.setattr("evolith.zstep.gates.platform.system", lambda: "Darwin")
    monkeypatch.setattr("evolith.zstep.gates.subprocess.run", fake_run)
    monkeypatch.setattr("evolith.zstep.gates.shutil.which", lambda name: f"/usr/bin/{name}")

    run_local_gates(tmp_path, make_task(tmp_path), tmp_path / "evidence")

    assert ["cargo", "check", "--workspace", "--target", "x86_64-unknown-linux-gnu"] in commands
    assert ["ruff", "check", "worker/"] in commands
    assert any("pytest" in cmd and "worker/tests/test_shm_layout_parity.py" in cmd for cmd in commands)


def test_local_gate_stops_at_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args, 1, "", "fmt failed\n")

    monkeypatch.setattr("evolith.zstep.gates.subprocess.run", fake_run)
    monkeypatch.setattr("evolith.zstep.gates.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="cargo fmt"):
        run_local_gates(tmp_path, make_task(tmp_path), tmp_path / "evidence")

    assert len(commands) == 1
    assert (tmp_path / "evidence/local-00-cargo.log").read_text() == "fmt failed\n"


def benchmark_file(
    tmp_path: Path,
    name: str,
    throughput: float,
    ttft: float = 100.0,
    tpot: float = 10.0,
    itl: float = 20.0,
    labels: tuple[str, ...] = ("public", "hidden"),
) -> Path:
    rows = []
    for label in labels:
        for repeat in range(3):
            rows.append(
                {
                    "label": label,
                    "repeat": repeat,
                    "output_throughput_tok_s": throughput + repeat,
                    "median_ttft_ms": ttft,
                    "mean_tpot_ms": tpot,
                    "p99_itl_ms": itl,
                }
            )
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"tiers": rows}), encoding="utf-8")
    return path


def benchmark(
    tmp_path: Path,
    name: str,
    throughput: float,
    ttft: float = 100.0,
    tpot: float = 10.0,
    itl: float = 20.0,
) -> dict[str, list[object]]:
    return load_benchmark(benchmark_file(tmp_path, name, throughput, ttft, tpot, itl))  # type: ignore[return-value]


def parity_report(tmp_path: Path, recompilations: int, rate: float, name: str = "parity.json") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "triton_recompilations": recompilations,
                "cases": [
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": rate},
                        "batched_match": {"exact_token_match_rate": rate},
                        "chunked_match": {"exact_token_match_rate": rate},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_screen_gate_blocks_throughput_regression_and_recompilation(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    base_screen = benchmark_file(tmp_path, "base-screen", 90.0, labels=("in512_c1",))
    regressed_screen = benchmark_file(tmp_path, "cand-reg", 80.0, labels=("in512_c1",))
    good_screen = benchmark_file(tmp_path, "cand-good", 92.0, labels=("in512_c1",))

    reg_decision = evaluate_screen(
        regressed_screen, parity_report(tmp_path, 0, 1.0, "p1.json"), task, base_screen
    )
    assert not reg_decision.passed
    assert any("screen in512_c1 throughput regressed" in r for r in reg_decision.reasons)

    recomp_decision = evaluate_screen(
        good_screen, parity_report(tmp_path, 2, 1.0, "p2.json"), task, base_screen
    )
    assert not recomp_decision.passed
    assert any("Triton recompilations" in r for r in recomp_decision.reasons)

    ok_decision = evaluate_screen(
        good_screen, parity_report(tmp_path, 0, 1.0, "p3.json"), task, base_screen
    )
    assert ok_decision.passed
    assert ok_decision.gain > 0


def test_benchmark_parser_groups_all_repeats(tmp_path: Path) -> None:
    parsed = benchmark(tmp_path, "baseline", 100.0)

    assert set(parsed) == {"public", "hidden"}
    assert len(parsed["public"]) == 3
    assert parsed["public"][2].output_throughput_tok_s == 102.0  # type: ignore[attr-defined]


def test_acceptance_requires_gain_and_all_guards(tmp_path: Path) -> None:
    decision = evaluate_candidate(
        benchmark(tmp_path, "base-accept", 100.0),  # type: ignore[arg-type]
        benchmark(tmp_path, "candidate-accept", 105.0, 102.0, 10.2, 20.5),  # type: ignore[arg-type]
        parity_report(tmp_path, 0, 1.0),
        make_task(tmp_path),
        changed_lines=30,
    )

    assert decision.accepted
    assert decision.geomean_gain == pytest.approx((106.0 / 101.0) - 1.0)
    assert "geomean=" in decision.metric_delta


def test_occams_razor_gate_rejects_bloated_marginal_patch(tmp_path: Path) -> None:
    # +3.1% raw gain with 290 changed lines (>120 free lines, penalty=170*0.0002=0.034 -> net_ratio=0.997 < 1.03)
    decision = evaluate_candidate(
        benchmark(tmp_path, "base-bloat", 100.0),  # type: ignore[arg-type]
        benchmark(tmp_path, "cand-bloat", 103.13),  # type: ignore[arg-type]
        parity_report(tmp_path, 0, 1.0),
        make_task(tmp_path),
        changed_lines=290,
    )

    assert not decision.accepted
    assert any("Occam's Razor" in r for r in decision.reasons)


def test_hidden_tier_regression_rejects_candidate(tmp_path: Path) -> None:
    decision = evaluate_candidate(
        benchmark(tmp_path, "base-regression", 100.0),  # type: ignore[arg-type]
        benchmark(tmp_path, "candidate-regression", 95.0),  # type: ignore[arg-type]
        parity_report(tmp_path, 0, 1.0),
        make_task(tmp_path),
    )

    assert not decision.accepted
    assert any("throughput" in reason for reason in decision.reasons)


def test_recompilation_or_prime_square_mismatch_rejects_candidate(tmp_path: Path) -> None:
    base = benchmark(tmp_path, "base-parity", 100.0)
    candidate = benchmark(tmp_path, "candidate-parity", 105.0)

    recompiled = evaluate_candidate(
        base, candidate, parity_report(tmp_path, 1, 1.0), make_task(tmp_path)  # type: ignore[arg-type]
    )
    mismatched = evaluate_candidate(
        base, candidate, parity_report(tmp_path, 0, 0.99), make_task(tmp_path)  # type: ignore[arg-type]
    )

    assert not recompiled.accepted
    assert not mismatched.accepted


def test_missing_candidate_tier_is_rejected(tmp_path: Path) -> None:
    base = benchmark(tmp_path, "base-missing", 100.0)
    candidate = benchmark(tmp_path, "candidate-missing", 105.0)
    del candidate["hidden"]

    decision = evaluate_candidate(
        base, candidate, parity_report(tmp_path, 0, 1.0), make_task(tmp_path)  # type: ignore[arg-type]
    )

    assert not decision.accepted
    assert any("missing" in reason for reason in decision.reasons)


def test_parity_exposure_invariance_and_cross_mode_logit_drift_guards(
    tmp_path: Path,
) -> None:
    base = benchmark(tmp_path, "base-exp", 100.0)
    cand = benchmark(tmp_path, "cand-exp", 106.0)
    task = make_task(tmp_path)

    # 1. Bypassed chunked/mixed schedule fails exposure gate
    bypassed = tmp_path / "parity-bypassed.json"
    bypassed.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "chunked_schedule": {
                    "chunked_prefill_steps": 0,
                    "decode_steps_with_partial_prefill": 0,
                },
                "cases": [
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": 1.0},
                        "batched_match": {"exact_token_match_rate": 1.0},
                        "chunked_match": {"exact_token_match_rate": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    d_bypassed = evaluate_candidate(base, cand, bypassed, task)  # type: ignore[arg-type]
    assert not d_bypassed.accepted
    assert any("exposure gate" in r for r in d_bypassed.reasons)

    # 2. Catastrophic cross-batch logit drift (e.g. slot bleed / stream WAR corruption) fails
    corrupted = tmp_path / "parity-corrupted.json"
    corrupted.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "chunked_schedule": {
                    "chunked_prefill_steps": 4,
                    "decode_steps_with_partial_prefill": 3,
                },
                "cases": [
                    {
                        "name": "france_capital",
                        "single_vs_batched_first_mismatch": {
                            "token_index": 2,
                            "left_top2": {"top1_logit": 21.0},
                            "right_top2": {"top1_logit": 5.5},
                        },
                    },
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": 1.0},
                        "batched_match": {"exact_token_match_rate": 1.0},
                        "chunked_match": {"exact_token_match_rate": 1.0},
                        "single_vs_batched_first_mismatch": None,
                        "batched_vs_chunked_first_mismatch": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    d_corrupted = evaluate_candidate(base, cand, corrupted, task)  # type: ignore[arg-type]
    assert not d_corrupted.accepted
    assert any("logit drift" in r for r in d_corrupted.reasons)

    # 3. Healthy exposure + bounded drift passes
    healthy = tmp_path / "parity-healthy.json"
    healthy.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "chunked_schedule": {
                    "chunked_prefill_steps": 4,
                    "decode_steps_with_partial_prefill": 3,
                },
                "cases": [
                    {
                        "name": "france_capital",
                        "single_vs_batched_first_mismatch": {
                            "token_index": 4,
                            "left_top2": {"top1_logit": 18.0},
                            "right_top2": {"top1_logit": 13.25},
                        },
                    },
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": 1.0},
                        "batched_match": {"exact_token_match_rate": 1.0},
                        "chunked_match": {"exact_token_match_rate": 1.0},
                        "single_vs_batched_first_mismatch": None,
                        "batched_vs_chunked_first_mismatch": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    d_healthy = evaluate_candidate(base, cand, healthy, task)  # type: ignore[arg-type]
    assert d_healthy.accepted

    # 4. Post-run idle KV cache leak fails both screen and verify gates
    leaky_screen = tmp_path / "screen-leak.json"
    leaky_screen.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c1",
                        "repeat": 0,
                        "output_throughput_tok_s": 120.0,
                        "median_ttft_ms": 140.0,
                        "mean_tpot_ms": 8.2,
                        "p99_itl_ms": 8.8,
                        "idle_kv_cache_usage": 0.0125,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    s_leak = evaluate_screen(leaky_screen, None, task)
    assert not s_leak.passed
    assert any("leaked idle KV cache" in r for r in s_leak.reasons)

    cand_leaky = {
        label: [row.model_copy(update={"idle_kv_cache_usage": 0.02}) for row in rows]
        for label, rows in cand.items()
    }
    v_leak = evaluate_candidate(base, cand_leaky, healthy, task)  # type: ignore[arg-type]
    assert not v_leak.accepted
    assert any("leaked idle KV cache" in r for r in v_leak.reasons)
