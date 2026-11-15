from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evolith.lab import engine_evolution
from evolith.lab.engine import (
    EngineConfig,
    EngineProposal,
    capacity_neighbors,
    server_command,
    source_hash,
)
from evolith.lab.latency import compare_trials, outputs_match, stream_request
from evolith.lab.store import new_run_id, write_json


def trial(config=None, latency=10., workload="same"):
    return {
        "run_id": new_run_id(),
        "config": (config or EngineConfig()).model_dump(), "source_hash": source_hash(),
        "model_fingerprint": "weights", "environment": {"gpu": "5090"},
        "workload_fingerprint": workload, "status": "measured", "correct": True,
        "ttft_us": [latency] * 5, "itl_us": [latency] * 5, "rounds": [{}] * 5,
        "records": [{"token_ids": [1, 2], "token_logprobs": [-0.1, -0.2]}],
    }


@pytest.mark.parametrize("data", [
    {"tensor_parallel": 8}, {"memory_fraction": 1}, {"graph": "eager"},
    {"max_model_len": 128}, {"dtype": "fp8"}, {"source": "print(1)"},
])
def test_model_cannot_reduce_quality_context_or_expand_hardware_budget(data):
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(data)


def test_server_uses_native_graphs_fixed_precision_context_and_no_prefix_cache(tmp_path):
    command = server_command(EngineConfig(), tmp_path, 18080)
    assert "--enforce-eager" not in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert "--no-enable-prefix-caching" in command
    assert "--compilation-config" not in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert len(EngineProposal(candidates=[EngineConfig()]).candidates) == 1


def test_fixed_pareto_gate_rejects_regressions_wrong_tokens_and_unstable_timings():
    baseline = trial()
    assert compare_trials(baseline, trial(latency=8))["accepted"]
    worse_first_token = trial(latency=8) | {"ttft_us": [12.] * 5}
    assert not compare_trials(baseline, worse_first_token)["accepted"]
    incorrect = trial(latency=8)
    incorrect["records"][0]["token_ids"] = [1, 3]
    assert not compare_trials(baseline, incorrect)["accepted"]
    assert not outputs_match(baseline["records"], incorrect["records"])
    with pytest.raises(ValueError):
        compare_trials(baseline, trial(workload="different"))


class Response:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self.lines


class Client:
    def __init__(self, lines):
        self.lines = lines

    def stream(self, *args, **kwargs):
        self.payload = kwargs["json"]
        return Response(self.lines)


def lines():
    return [
        'data: {"choices":[{"delta":{"role":"assistant"},"logprobs":null}]}',
        'data: {"choices":[{"token_ids":[1],"logprobs":{"content":[{"token":"token_id:1","logprob":-0.1}]}}]}',
        'data: {"choices":[{"token_ids":[2],"logprobs":{"content":[{"token":"token_id:2","logprob":-0.2}]}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":2}}',
        "data: [DONE]",
    ]


def test_streaming_latency_skips_role_chunks_and_requires_exact_output_length():
    client = Client(lines())
    result = stream_request(client, "fixture", 2, clock=iter([0., 0.1, 0.2, 0.3, 0.4]).__next__)
    assert result["token_ids"] == [1, 2]
    assert result["ttft_sec"] > 0 and result["itl_sec"] > 0
    assert client.payload["return_token_ids"] and not client.payload["temperature"]
    with pytest.raises(ValueError, match="Incomplete"):
        stream_request(Client(lines()[:-1]), "fixture", 2)


def test_independent_restarts_must_match_outputs_and_provenance():
    first, second = trial(), trial()
    assert len(engine_evolution.combined(first, second)["ttft_us"]) == 10
    with pytest.raises(ValueError):
        engine_evolution.combined(first, second | {"source_hash": "changed"})
    with pytest.raises(ValueError, match="distinct"):
        engine_evolution.combined(first, first)
    with pytest.raises(ValueError, match="distinct"):
        engine_evolution.combined(first, second | {"run_id": True})


def test_controller_keeps_native_when_no_candidate_passes(monkeypatch, tmp_path):
    from evolith.lab.latency import workload_fingerprint

    calls = []

    def execute(module, request, directory, timeout):
        config = EngineConfig.model_validate(request["config"])
        calls.append(config)
        data = trial(config=config, latency=10 if len(calls) == 1 else 12,
                     workload=workload_fingerprint(64))
        data["proposals"] = {"candidates": [EngineConfig(tensor_parallel=2, max_num_seqs=1).model_dump()]}
        path = directory / "trial.json"
        write_json(path, data)
        return {"result": {"status": "measured", "report": str(path)}}

    monkeypatch.setattr(engine_evolution, "execute_request", execute)
    result = engine_evolution.run(tmp_path, tmp_path, candidates=1, seconds=600)
    profile = json.loads(Path(result["profile"]).read_text())
    assert profile["selection"] == "native_baseline"
    assert result["promotions"] == 0
    assert len(calls) == 3
    assert profile["fresh_baseline_proof"]["run_id"] != profile["baseline_proof"]["run_id"]


@pytest.mark.parametrize("failure", ["different_outputs", "startup_failed"])
def test_native_restart_failure_stops_before_emitting_a_profile_or_testing_candidates(
    monkeypatch, tmp_path, failure,
):
    from evolith.lab.latency import workload_fingerprint
    from evolith.lab.store import RunStore

    calls = []

    def execute(module, request, directory, timeout):
        calls.append(request)
        data = trial(workload=workload_fingerprint(64))
        data["proposals"] = {"candidates": [EngineConfig(tensor_parallel=2).model_dump()]}
        if len(calls) == 2:
            if failure == "different_outputs":
                data["records"][0]["token_ids"][0] = 999
            else:
                data.update(status="startup_failed", correct=False)
        path = directory / "trial.json"
        write_json(path, data)
        return {"result": {"status": data["status"], "report": str(path)}}

    monkeypatch.setattr(engine_evolution, "execute_request", execute)
    with pytest.raises((ValueError, RuntimeError)):
        engine_evolution.run(tmp_path, tmp_path, candidates=1, seconds=600)
    assert len(calls) == 2
    assert not calls[1]["propose"]
    assert not list(tmp_path.glob("*/engine.json"))
    report = RunStore(tmp_path / "runs.sqlite3").recent("engine_session", 1)[0]
    assert report["status"] == "failed" and report["profile"] is None


@pytest.mark.parametrize("failure", ["missing", "same_trial", "different_outputs", "different_runtime"])
def test_native_artifact_requires_distinct_compatible_restart_evidence(failure):
    from evolith.lab.engine import EngineArtifact, validate_artifact

    original, fresh = trial(), trial()
    if failure == "missing":
        fresh = None
    elif failure == "same_trial":
        fresh["run_id"] = original["run_id"]
    elif failure == "different_outputs":
        fresh["records"][0]["token_ids"][0] = 999
    else:
        fresh["environment"] = {"gpu": "other"}
    artifact = EngineArtifact(
        config=EngineConfig(), source_hash=source_hash(), model_fingerprint="weights",
        environment={"gpu": "5090"}, workload_fingerprint="same",
        selection="native_baseline", baseline_proof=original, fresh_baseline_proof=fresh,
    )
    with pytest.raises(ValueError):
        validate_artifact(artifact)


def test_native_restart_budget_exhaustion_never_returns_a_deployable_profile(monkeypatch, tmp_path):
    from evolith.lab.latency import workload_fingerprint
    from evolith.lab.search import WorkerTimeout

    calls = []

    def execute(module, request, directory, timeout):
        calls.append(request)
        if len(calls) == 2:
            raise WorkerTimeout("Certification budget exhausted.")
        data = trial(workload=workload_fingerprint(64))
        data["proposals"] = {"candidates": [EngineConfig(tensor_parallel=2).model_dump()]}
        path = directory / "trial.json"
        write_json(path, data)
        return {"result": {"status": "measured", "report": str(path)}}

    monkeypatch.setattr(engine_evolution, "execute_request", execute)
    result = engine_evolution.run(tmp_path, tmp_path, candidates=1, seconds=600)
    assert result["status"] == "stopped" and result["profile"] is None
    assert not list(tmp_path.glob("*/engine.json"))
    assert len(calls) == 2


def test_confirmation_is_new_processes_in_ab_ba_order(monkeypatch, tmp_path):
    from evolith.lab.latency import workload_fingerprint

    calls = []
    chosen = EngineConfig(tensor_parallel=2, max_num_seqs=1)

    def execute(module, request, directory, timeout):
        config = EngineConfig.model_validate(request["config"])
        calls.append(config)
        data = trial(config=config, latency=8 if config == chosen else 10,
                     workload=workload_fingerprint(64))
        data["proposals"] = {"candidates": [chosen.model_dump()]}
        path = directory / "trial.json"
        write_json(path, data)
        return {"result": {"status": "measured", "report": str(path)}}

    monkeypatch.setattr(engine_evolution, "execute_request", execute)
    result = engine_evolution.run(tmp_path, tmp_path, candidates=1, seconds=600)
    assert calls == [EngineConfig(), EngineConfig(), chosen, chosen, EngineConfig()]
    assert result["promotions"] == 1
    profile = json.loads(Path(result["profile"]).read_text())
    assert profile["fresh_baseline_proof"]["config"] == EngineConfig().model_dump()
    assert profile["fresh_candidate_proof"]["config"] == chosen.model_dump()


def test_capacity_repairs_are_automatic_resource_monotone_and_preserve_precision_context(tmp_path):
    failed = EngineConfig(tensor_parallel=2, memory_fraction=0.95)
    repairs = capacity_neighbors(failed)
    assert repairs
    for config in repairs:
        assert config.tensor_parallel == failed.tensor_parallel
        assert config.graph == failed.graph
        assert config.max_num_seqs <= failed.max_num_seqs
        assert config.prefill_tokens <= failed.prefill_tokens
        assert config.memory_fraction >= failed.memory_fraction
        command = server_command(config, tmp_path, 18080)
        assert command[command.index("--max-model-len") + 1] == "2048"
        assert command[command.index("--dtype") + 1] == "bfloat16"


def test_capacity_failure_adapts_the_next_proposal_and_preserves_failed_lineage(monkeypatch, tmp_path):
    from evolith.lab.latency import workload_fingerprint
    from evolith.lab.store import RunStore

    failed = EngineConfig(tensor_parallel=2, memory_fraction=0.95)
    calls = []

    def execute(module, request, directory, timeout):
        config = EngineConfig.model_validate(request["config"])
        calls.append(config)
        data = trial(config=config, latency=10 if len(calls) == 1 else 12,
                     workload=workload_fingerprint(64))
        data["proposals"] = {"candidates": [failed.model_dump()]}
        if len(calls) == 3:
            data.update(status="startup_failed", failure_kind="capacity", correct=False,
                        ttft_us=[], itl_us=[], error="No available memory for the cache blocks")
        path = directory / "trial.json"
        write_json(path, data)
        return {"result": {"status": data["status"], "report": str(path)}}

    monkeypatch.setattr(engine_evolution, "execute_request", execute)
    result = engine_evolution.run(tmp_path, tmp_path, candidates=2, seconds=600)
    assert calls == [EngineConfig(), EngineConfig(), failed, capacity_neighbors(failed)[0]]
    report = json.loads(Path(result["report"]).read_text())
    assert report["cycles"][0]["outcome"] == "startup_failed"
    assert report["cycles"][1]["proposal_source"] == "observed_capacity_constraint_solver"
    assert len(RunStore(tmp_path / "runs.sqlite3").recent("engine_cycle")) == 2
    assert result["promotions"] == 0


def test_matching_archived_capacity_failures_are_hints_not_proofs(tmp_path):
    from evolith.lab.engine_worker import remembered_failures
    from evolith.lab.store import RunStore

    current = {"model_fingerprint": "same-model", "environment": {"gpu": "same-gpu"},
               "workload_fingerprint": "same-workload"}
    path = tmp_path / "old-trial" / "report.json"
    write_json(path, current | {"config": EngineConfig(tensor_parallel=2).model_dump(),
                               "source_hash": "older-controller", "status": "startup_failed"})
    (path.parent / "server.log").write_text("ValueError: No available memory for the cache blocks")
    RunStore(tmp_path / "runs.sqlite3").record("old-session", "engine_session", {
        "cycles": [{"outcome": "startup_failed", "candidate_report": str(path)}],
    })
    hints = remembered_failures(tmp_path, current)
    assert len(hints) == 1
    assert hints[0]["reason"] == "No available memory for the cache blocks"
    assert "not performance proof" in hints[0]["scope"]
    assert remembered_failures(tmp_path, current | {"model_fingerprint": "different-model"}) == []
