from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Callable

import pytest

from evolith.zstep.remote import RemotePreflightError, ZStepRemoteRunner
from evolith.zstep.task import ZStepRSITask

Executor = Callable[..., subprocess.CompletedProcess[str]]


def make_task(tmp_path: Path, **overrides: object) -> ZStepRSITask:
    values: dict[str, object] = {
        "project_root": tmp_path,
        "source_repo": tmp_path / "source",
    }
    values.update(overrides)
    return ZStepRSITask(**values)  # type: ignore[arg-type]


def successful_executor(calls: list[list[str]], idle_mib: int = 4) -> Executor:
    def execute(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        joined = " ".join(args)
        if "nvidia-smi" in joined:
            output = "\n".join(f"{index}, {idle_mib} MiB" for index in range(8)) + "\n"
        elif "config.json" in joined:
            output = "deepseek_v41\n"
        else:
            output = ""
        return subprocess.CompletedProcess(args, 0, output, "")

    return execute


def test_preflight_accepts_eight_idle_gpus_with_driver_context_and_matching_model(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    runner = ZStepRemoteRunner(make_task(tmp_path), executor=successful_executor(calls, idle_mib=8))

    runner.preflight()

    assert any("nvidia-smi" in " ".join(call) for call in calls)
    assert any("docker info" in " ".join(call) for call in calls)
    assert any("config.json" in " ".join(call) for call in calls)
    assert all(call[0] in ("ssh", "sshpass") for call in calls)
    assert all("sudo" not in call for call in calls)


def test_preflight_rejects_busy_gpu_before_docker(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def busy(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "0, 120 MiB\n", "")

    runner = ZStepRemoteRunner(make_task(tmp_path), executor=busy)
    with pytest.raises(RemotePreflightError, match="GPU 0"):
        runner.preflight()

    assert len(calls) == 1


def test_preflight_preserves_gpu_query_error(tmp_path: Path) -> None:
    def unavailable(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 255, "", "kex_exchange_identification: Connection closed by remote host"
        )

    runner = ZStepRemoteRunner(make_task(tmp_path), executor=unavailable)
    with pytest.raises(RemotePreflightError, match="kex_exchange_identification"):
        runner.preflight()


def test_preflight_reports_docker_permission_without_sudo(tmp_path: Path) -> None:
    def denied(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(args)
        if "nvidia-smi" in joined:
            output = "\n".join(f"{index}, 0 MiB" for index in range(8)) + "\n"
            return subprocess.CompletedProcess(args, 0, output, "")
        return subprocess.CompletedProcess(args, 1, "", "permission denied docker.sock")

    runner = ZStepRemoteRunner(make_task(tmp_path), executor=denied)
    with pytest.raises(RemotePreflightError, match="Docker access"):
        runner.preflight()


def test_benchmark_script_uses_unique_container_names_compile_guard_and_trap(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    runner = ZStepRemoteRunner(task)

    script_a = runner.benchmark_script(
        "evolith-rsi/runs/g0001/screen/src",
        "evolith-rsi/runs/g0001/screen/metrics.json",
        task.screen_tiers,
        1,
        None,
    )
    script_b = runner.benchmark_script(
        "evolith-rsi/runs/g0002/screen/src",
        "evolith-rsi/runs/g0002/screen/metrics.json",
        task.screen_tiers,
        1,
        None,
    )
    verify_script = runner.benchmark_script(
        "evolith-rsi/runs/g0001/verify/candidate",
        "evolith-rsi/runs/g0001/verify/candidate.json",
        task.verify_tiers,
        1,
        "evolith-rsi/runs/g0001/verify/parity.json",
    )

    assert "evolith-g0001-screen" in script_a
    assert "evolith-g0002-screen" in script_b
    assert script_a != script_b
    assert "if [ ! -x ./target/release/zstep ]; then" in verify_script
    assert "cargo test --release -p zs-shm -p zs-scheduler --lib" in verify_script
    assert "trap 'rm -rf " in verify_script
    assert f'--bench-tiers "{task.verify_tiers}"' in verify_script
    assert "--repeats 1" in verify_script
    assert "--model-dir /data02/DeepSeek-V4.1-Flash-1017" in verify_script
    assert "--max-prefill-chunk 24 --verify-greedy-json docs/benchmarks/phase3_vllm_greedy_reference.json" in verify_script
    assert "sudo" not in verify_script


def test_remote_benchmark_error_cleans_up_orphan_container_and_preserves_output(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def failed(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(args)
        calls.append(joined)
        if "nvidia-smi" in joined:
            output = "\n".join(f"{index}, 0 MiB" for index in range(8)) + "\n"
            return subprocess.CompletedProcess(args, 0, output, "")
        if "docker rm -f" in joined:
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 2, "build output", "vllm: unrecognized arguments")

    runner = ZStepRemoteRunner(make_task(tmp_path), executor=failed)
    with pytest.raises(RuntimeError, match="vllm: unrecognized arguments"):
        runner._run_benchmark(
            "evolith-rsi/runs/g0009/screen/src",
            "metrics.json",
            "public:1:1:512:64",
            1,
            None,
        )

    assert any("docker rm -f evolith-g0009-screen-src-metrics" in c for c in calls)


def test_verify_reuses_completed_round_checkpoints(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(parents=True)
    # Pre-populate round 0 and round 1 checkpoints + parity (simulating interrupted g0003)
    (evidence / "verify-parity.json").write_text("{}", encoding="utf-8")
    for repeat in (0, 1):
        for role in ("candidate", "baseline"):
            (evidence / f"{role}-r{repeat}.json").write_text(
                json.dumps(
                    {
                        "tiers": [
                            {
                                "label": "in512_c16",
                                "repeat": 0,
                                "output_throughput_tok_s": 800.0 + repeat,
                                "median_ttft_ms": 50.0,
                                "mean_tpot_ms": 18.0,
                                "p99_itl_ms": 22.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

    base_tar = tmp_path / "base.tar.gz"
    cand_tar = tmp_path / "cand.tar.gz"
    base_tar.write_bytes(b"base")
    cand_tar.write_bytes(b"cand")

    executed_benchmarks: list[str] = []

    def resume_executor(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(args)
        if "nvidia-smi" in joined:
            return subprocess.CompletedProcess(
                args, 0, "\n".join(f"{i}, 0 MiB" for i in range(8)) + "\n", ""
            )
        if args[0] == "scp" and args[-1].endswith(".json"):
            local_dest = Path(args[-1])
            local_dest.write_text(
                json.dumps(
                    {
                        "tiers": [
                            {
                                "label": "in512_c16",
                                "repeat": 0,
                                "output_throughput_tok_s": 810.0,
                                "median_ttft_ms": 50.0,
                                "mean_tpot_ms": 18.0,
                                "p99_itl_ms": 22.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args, 0, "", "")
        if "docker run" in joined:
            executed_benchmarks.append(joined)
        return subprocess.CompletedProcess(args, 0, "", "")

    runner = ZStepRemoteRunner(make_task(tmp_path, repeats=3), executor=resume_executor)
    cand_metrics, base_metrics, _ = runner.verify("g0003", base_tar, cand_tar, evidence)

    # Only round 2 (candidate-r2 + baseline-r2 = 2 docker runs) should execute; r0 and r1 were reused!
    assert len(executed_benchmarks) == 2
    assert len(json.loads(cand_metrics.read_text())["tiers"]) == 3
    assert len(json.loads(base_metrics.read_text())["tiers"]) == 3


def test_sshpass_never_embeds_password_in_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSHPASS", "secret-value")
    calls: list[list[str]] = []
    runner = ZStepRemoteRunner(make_task(tmp_path), executor=successful_executor(calls))

    runner.preflight()

    assert calls[0][:2] == ["sshpass", "-e"]
    assert all("secret-value" not in arg for call in calls for arg in call)


def test_screen_baseline_recomputes_when_cached_file_lacks_new_c32_tier(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    stale_cache = task.baseline_cache_dir / f"{'a' * 16}-screen-metrics.json"
    stale_cache.parent.mkdir(parents=True)
    # Stale cache only has in512_c1, missing in512_c32
    stale_cache.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c1",
                        "repeat": 0,
                        "output_throughput_tok_s": 90.0,
                        "median_ttft_ms": 150.0,
                        "mean_tpot_ms": 9.0,
                        "p99_itl_ms": 10.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    base_tar = tmp_path / "base.tar.gz"
    base_tar.write_bytes(b"tar")

    docker_runs = 0

    def executor(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal docker_runs
        joined = " ".join(args)
        if "nvidia-smi" in joined:
            return subprocess.CompletedProcess(
                args, 0, "\n".join(f"{i}, 0 MiB" for i in range(8)) + "\n", ""
            )
        if "docker run" in joined:
            docker_runs += 1
        if args[0] == "scp" and args[-1].endswith("-screen-metrics.json"):
            Path(args[-1]).write_text(
                json.dumps(
                    {
                        "tiers": [
                            {
                                "label": "in512_c1",
                                "repeat": 0,
                                "output_throughput_tok_s": 91.0,
                                "median_ttft_ms": 150.0,
                                "mean_tpot_ms": 9.0,
                                "p99_itl_ms": 10.0,
                            },
                            {
                                "label": "in512_c32",
                                "repeat": 0,
                                "output_throughput_tok_s": 1150.0,
                                "median_ttft_ms": 400.0,
                                "mean_tpot_ms": 22.0,
                                "p99_itl_ms": 30.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    runner = ZStepRemoteRunner(task, executor=executor)
    out = runner.screen_baseline("a" * 64, base_tar)
    assert docker_runs == 1
    # Second call reuses the complete 2-tier cache without hitting remote Docker
    runner.screen_baseline("a" * 64, base_tar)
    assert docker_runs == 1
    assert {r["label"] for r in json.loads(out.read_text())["tiers"]} == {"in512_c1", "in512_c32"}


def test_gpu_smoke_runs_fast_pytest_and_cleans_shm_on_failure(tmp_path: Path) -> None:
    task = make_task(tmp_path, enable_gpu_smoke=True)
    cand_tar = tmp_path / "cand.tar.gz"
    cand_tar.write_bytes(b"tar")

    calls: list[str] = []

    def failing_smoke(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(args)
        calls.append(joined)
        if "nvidia-smi" in joined:
            return subprocess.CompletedProcess(
                args, 0, "\n".join(f"{i}, 0 MiB" for i in range(8)) + "\n", ""
            )
        if "pytest -q" in joined:
            return subprocess.CompletedProcess(
                args, 1, "", "TypeError: where(): scalar out= unsupported"
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    runner = ZStepRemoteRunner(task, executor=failing_smoke)
    with pytest.raises(RuntimeError, match="Remote GPU smoke exited 1"):
        runner.gpu_smoke("g0007-smoke", cand_tar)

    assert any("/dev/shm/zstep*" in c for c in calls)
