from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evolith.lab.restart import comparison, plan, policy_environment


def record(tokens, probabilities):
    return {"token_ids": tokens, "token_logprobs": probabilities}


def test_restart_comparison_keeps_existing_exact_id_and_logprob_gate():
    original = [record([1, 2, 3], [-1.0, -0.1, -2.0])]
    assert comparison(original, original)["correct"]
    assert not comparison(original, [record([1, 2, 3], [-1.1, -0.1, -2.0])])["correct"]
    changed = comparison(original, [record([1, 8, 9], [-1.01, -5, -20])])
    assert not changed["correct"]
    details = changed["requests"][0]
    assert details["first_different_token"] == 1
    assert details["max_common_prefix_logprob_difference"] == pytest.approx(0.01)


def test_diagnostic_policies_are_bounded_and_do_not_accept_arbitrary_environment(monkeypatch):
    assert len(plan("both")) == 5
    assert [case["prefix_request"] for case in plan("native")] == [False, False, True]
    assert len(plan("batch_invariant")) == 2
    assert all(case["attention_backend"] == "FLASH_ATTN" for case in plan("batch_invariant"))
    assert len(plan("collective")) == 4
    assert len(plan("no_aot")) == 3
    with pytest.raises(ValueError):
        plan("arbitrary-code")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "original")
    with policy_environment("batch_invariant"):
        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
    assert os.environ["VLLM_BATCH_INVARIANT"] == "original"
    with pytest.raises(ValueError), policy_environment("arbitrary-code"):
        pass


def test_policy_environment_restores_unset_state_on_exception(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with pytest.raises(RuntimeError), policy_environment("native"):
        assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
        raise RuntimeError("probe failed")
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def test_isolated_cache_environment_is_explicit_and_restored(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/previous/inductor")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    with pytest.raises(RuntimeError), policy_environment("native", tmp_path):
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")
        assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "triton")
        assert os.environ["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm")
        raise RuntimeError("probe failed")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/previous/inductor"
    assert "TRITON_CACHE_DIR" not in os.environ
    assert "VLLM_CACHE_ROOT" not in os.environ
    with pytest.raises(ValueError), policy_environment("native", Path("relative")):
        pass


def test_collective_diagnosis_only_pins_the_requested_algorithm(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "original")
    monkeypatch.setenv("NCCL_PROTO", "original-protocol")
    for policy, expected in (("native_ring", "allreduce:ring"), ("native_tree", "allreduce:tree")):
        with policy_environment(policy):
            assert os.environ["NCCL_ALGO"] == expected
            assert os.environ["NCCL_PROTO"] == "original-protocol"
            assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
        assert os.environ["NCCL_ALGO"] == "original"
    monkeypatch.delenv("VLLM_USE_AOT_COMPILE", raising=False)
    with policy_environment("native_no_aot"):
        assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
        assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
        assert os.environ["NCCL_ALGO"] == "original"
    assert "VLLM_USE_AOT_COMPILE" not in os.environ


def test_explicit_diagnostic_attention_backend_preserves_the_native_command(monkeypatch, tmp_path):
    from evolith.lab import engine_worker
    from evolith.lab.engine import EngineConfig, EngineJob, server_command

    job = EngineJob(model_path=tmp_path, output=tmp_path, config=EngineConfig())
    commands = []

    class CommandCaptured(Exception):
        pass

    def capture(command, **_):
        commands.append(command)
        raise CommandCaptured

    monkeypatch.setattr(engine_worker.subprocess, "Popen", capture)
    for backend in (None, "FLASH_ATTN"):
        with pytest.raises(CommandCaptured), engine_worker.server(
            job, tmp_path, attention_backend=backend,
        ):
            pass
    original = server_command(job.config, job.model_path, engine_worker.PORT)
    assert commands == [original, original + ["--attention-backend", "FLASH_ATTN"]]
    with pytest.raises(ValueError), engine_worker.server(
        job, tmp_path, attention_backend="arbitrary-flags",
    ):
        pass
    assert len(commands) == 2


def test_invariant_probe_passes_explicit_backend_and_never_certifies_a_failed_start(
    monkeypatch, tmp_path,
):
    from contextlib import contextmanager

    from evolith.lab import restart
    from evolith.lab.engine_worker import StartupFailure

    @contextmanager
    def failed_server(job, directory, **options):
        assert options == {"attention_backend": "FLASH_ATTN"}
        raise StartupFailure("Unsupported policy.")
        yield

    monkeypatch.setattr(restart, "server", failed_server)
    report = restart.trial(tmp_path, tmp_path, plan("batch_invariant")[0], None)
    assert report["status"] == "startup_failed"
    assert report["self_consistent"] is None
    assert not report["records"]


def test_diagnostic_run_preserves_regressions_without_authorizing_deployment(monkeypatch, tmp_path):
    from evolith.lab import restart

    monkeypatch.setattr(restart, "inspect_runtime", lambda _: (
        {"fingerprint": "test-model"}, {"cuda_available": True},
    ))
    calls = []

    def fake_trial(model_path, output, case, reference):
        calls.append(case)
        tokens = [8] if case["policy"] == "native" and case["prefix_request"] else [1]
        assert os.environ["VLLM_BATCH_INVARIANT"] == (
            "1" if case["policy"] == "batch_invariant" else "0"
        )
        return {"run_id": f"test-{len(calls)}", "status": "observed",
                "records": [record(tokens, [-1.0])]}

    monkeypatch.setattr(restart, "trial", fake_trial)
    result = restart.run(tmp_path, tmp_path, seconds=600)
    report = json.loads(Path(result["report"]).read_text())
    assert len(calls) == 5
    assert all("session_run_id" in call for call in calls)
    assert [entry["correct"] for entry in report["comparisons"]] == [True, False, True]
    assert report["scope"] == "diagnostic only; no promotion or deployment"
    assert report["cache_scope"] == "existing_shared_cache"
    assert not report["cache_directories"]
    calls.clear()
    result = restart.run(tmp_path, tmp_path, policy="native", seconds=600,
                         isolated_cache_root=tmp_path / "cache")
    isolated = json.loads(Path(result["report"]).read_text())
    assert isolated["cache_scope"] == "isolated_per_policy"
    assert set(isolated["cache_directories"]) == {"native"}
    assert Path(isolated["cache_directories"]["native"]).is_dir()
    with pytest.raises(ValueError):
        restart.run(tmp_path, tmp_path, seconds=599)
