from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from evolith.cli import app
from evolith.lab import adapter
from evolith.lab.contracts import KernelConfig, Manifest, Proposal, Shape, content_hash
from evolith.lab.measurement import decide
from evolith.lab.search import execute_job
from evolith.lab.serving import compare_outputs, measure_model_trials, verify_model
from evolith.lab.store import RunStore, new_run_id, write_json
from evolith.lab.worker import Job


def test_no_gpu_import_required_for_cli():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "verify-model" in result.stdout
    assert "demo-evolve" in result.stdout


def test_missing_cuda_exits_once_without_synthetic_results(monkeypatch):
    fake = ModuleType("evolith.lab.gpu")
    fake.environment = lambda: {"cuda_available": False}
    fake.model_info = lambda path: {}
    monkeypatch.setitem(sys.modules, "evolith.lab.gpu", fake)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {"cuda_available": False}


@pytest.mark.parametrize("data", [
    {"num_warps": 32}, {"rows_per_program": 10000}, {"source_code": "print('not allowed')"},
    {"num_warps": "4"}, {"num_warps": True},
])
def test_model_cannot_propose_arbitrary_source_or_launch_parameters(data):
    with pytest.raises(ValidationError):
        KernelConfig.model_validate(data)


def test_proposal_is_bounded_and_strict():
    assert Proposal.model_validate_json(
        '{"candidates":[{"num_warps":8,"rows_per_program":4}]}'
    ).candidates[0].key == "w8-r4"
    for data in ({"candidates": []}, {"candidates": [{}] * 13}, {"candidates": [{}], "exec": "x"}):
        with pytest.raises(ValidationError):
            Proposal.model_validate(data)


@pytest.mark.parametrize("override", [
    {"rounds": 4}, {"iterations": 1001}, {"seed": -1}, {"eps": float("nan")},
    {"shapes": []}, {"min_gain": 1}, {"source": "arbitrary code"},
])
def test_worker_request_resource_and_input_bounds(override):
    data = {
        "config": {}, "shapes": [{"rows": 1, "hidden": 5120, "residual": True}],
        "eps": 1e-6, "seed": 0, "rounds": 11, "iterations": 32, "min_gain": 0.03,
    }
    with pytest.raises(ValidationError):
        Job.model_validate(data | override)


def test_reliable_gain_passes_but_small_or_noisy_gain_does_not():
    assert decide([10.] * 11, [8.] * 11).accepted
    assert not decide([10.] * 11, [9.9] * 11).accepted
    assert not decide([10.] * 11, [12.] * 11).accepted
    assert not decide([10.] * 11, [1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2.]).accepted


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_invalid_measurements_fail_closed(bad):
    with pytest.raises(ValueError):
        decide([10.] * 5, [bad] * 5)


def test_paired_measurements_require_enough_samples():
    with pytest.raises(ValueError):
        decide([1.] * 4, [1.] * 4)
    with pytest.raises(ValueError):
        decide([1.] * 6, [1.] * 5)


def test_shape_contract_does_not_alias_attention_sequence_with_hidden():
    assert Shape(rows=1, hidden=64, residual=False).key != Shape(rows=1, hidden=512, residual=False).key
    with pytest.raises(ValidationError):
        Shape(rows=0, hidden=5120, residual=False)


def test_history_is_append_only_and_unique(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite3")
    first, second = new_run_id(), new_run_id()
    assert first != second
    store.record(first, "search", {"score": 1})
    store.record(second, "search", {"score": 2})
    with pytest.raises(sqlite3.IntegrityError):
        store.record(first, "search", {"score": 3})
    assert [r["report"]["score"] for r in store.list()] == [1, 2]


def test_reports_are_valid_json_and_invalid_write_preserves_existing_report(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"ok": 1}
    assert list(tmp_path.iterdir()) == [path]


def record(tokens=(1, 2), probs=(-0.1, -0.2)):
    return {"records": [{"token_ids": list(tokens), "token_logprobs": list(probs)}]}


def test_model_regression_rejects_wrong_tokens_missing_and_nonfinite_logprobs():
    assert compare_outputs(record(), record())["passed"]
    for other in (record((1, 3)), record(probs=()), record(probs=(float("nan"), -0.2)),
                  record(probs=(-0.2, -0.2)), record(tokens=(), probs=())):
        assert not compare_outputs(record(), other)["passed"]


def test_end_to_end_seconds_are_converted_to_the_reported_microseconds():
    timing = measure_model_trials({"baseline_sec": [1.] * 5, "candidate_sec": [0.8] * 5})
    assert timing.baseline_us == [1_000_000.] * 5
    assert timing.candidate_us == [800_000.] * 5
    assert timing.speedup == 1.25
    assert timing.accepted


def test_hash_is_stable_and_disallows_nan():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        content_hash({"a": float("nan")})


def test_worker_extension_uses_local_model_and_json_arguments(monkeypatch):
    worker = adapter.EvolithWorkerExtension()
    model = object()
    worker.model_runner = SimpleNamespace(model=model)
    monkeypatch.setattr(adapter, "activate", lambda data, m: {"same_model": m is model, **data})
    monkeypatch.setattr(adapter, "stats", lambda m: {"same_model": m is model})
    monkeypatch.setattr(adapter, "restore", lambda m: {"same_model": m is model})
    assert worker.evolith_activate({"run_id": "test"}) == {"same_model": True, "run_id": "test"}
    assert worker.evolith_stats() == {"same_model": True}
    assert worker.evolith_restore() == {"same_model": True}


def test_worker_timeout_terminates_process_group_and_preserves_log(monkeypatch, tmp_path):
    import signal

    from evolith.lab import search

    killed = []

    class Process:
        pid = 987654
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("trusted-worker", timeout)
            return ("worker stopped", None)

    monkeypatch.setattr(search.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(search.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(RuntimeError, match="process group terminated"):
        execute_job({}, tmp_path / "job", timeout=0.01)
    assert killed == [(987654, signal.SIGTERM)]
    assert (tmp_path / "job" / "worker.log").read_text() == "worker stopped"


def test_empty_manifest_cannot_start_an_optimized_model_run(monkeypatch, tmp_path):
    gpu = ModuleType("evolith.lab.gpu")
    gpu.environment = dict
    gpu.model_info = lambda path: {"fingerprint": "weights"}
    vllm = ModuleType("vllm")
    vllm.LLM = lambda **kwargs: pytest.fail("Unverified candidate must not start an engine")
    vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "evolith.lab.gpu", gpu)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    manifest = Manifest(
        run_id="test", model_fingerprint="weights", kernel_hash="source",
        environment={}, eps=1e-6, evaluations=[], winners={}, measurement={},
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    with pytest.raises(ValueError, match="No independently confirmed kernels"):
        verify_model(tmp_path, path, tmp_path)
