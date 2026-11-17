from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from evolith.cli import app
from evolith.lab import autonomy
from evolith.lab.autonomy import (
    Budget,
    StageJob,
    compose,
    configuration_space,
    novel_configs,
    select_target,
)
from evolith.lab.contracts import (
    Evaluation,
    KernelConfig,
    Manifest,
    Proposal,
    Shape,
    Workload,
)
from evolith.lab.measurement import decide
from evolith.lab.observation import Observation, ObservedTarget, aggregate
from evolith.lab.search import WorkerTimeout, execute_request
from evolith.lab.store import RunStore, new_run_id, write_json


def target(rows=7, residual=False, cost=100):
    shape = Shape(rows=rows, hidden=5120, residual=residual)
    return ObservedTarget(
        label=shape.key, shape=shape, calls=100, samples=16, estimated_rank_total_us=cost,
    )


def observation(*targets):
    return Observation(
        model_fingerprint="weights", workload_fingerprint="workload", rank_count=4,
        dropped_contract_calls=0, targets=list(targets),
    )


def test_target_choice_uses_measured_cost_not_a_fixed_decode_shape():
    decode, prefill = target(1, True, 20), target(7, False, 100)
    assert select_target(observation(decode, prefill), {}, {}) == prefill
    assert select_target(observation(decode, prefill), {}, {prefill.key: 10}) == decode
    assert select_target(observation(decode, prefill),
                         {prefill.key: {c.key for c in configuration_space()}}, {}) == decode


def test_unsupported_bottleneck_is_a_gap_not_a_fake_success():
    unsupported = target().model_copy(update={"shape": None, "family": "attention"})
    assert select_target(observation(unsupported), {}, {}) is None


def test_model_suggestions_are_deduplicated_and_exploration_is_bounded():
    proposal = Proposal(candidates=[KernelConfig()] * 4)
    configs = novel_configs(proposal, {"w4-r1"}, 3)
    assert len(configs) == len({c.key for c in configs}) == 3
    assert "w4-r1" not in {c.key for c in configs}
    assert novel_configs(proposal, {c.key for c in configuration_space()}, 4) == []


def test_tp_observations_average_rank_cost_instead_of_summing_parallel_work():
    row = {"label": "norm", "shape": target().shape.model_dump(), "calls": 10,
           "sample_us": [2., 4.], "unsupported_reasons": []}
    result = aggregate([{"targets": [row], "dropped_contract_calls": 0}] * 4, "weights", "workload")
    assert result.targets[0].estimated_rank_total_us == 30
    assert result.targets[0].calls == 40
    with pytest.raises(ValueError):
        aggregate([], "weights", "workload")
    with pytest.raises(ValueError):
        aggregate([{"targets": [row | {"sample_us": []}]}], "weights", "workload")
    with pytest.raises(ValidationError):
        aggregate([{"targets": [row | {"sample_us": [float("nan")]}],
                    "dropped_contract_calls": 0}], "weights", "workload")


@pytest.mark.parametrize("values", [
    {"cycles": 0}, {"cycles": 9}, {"evaluations": 1}, {"seconds": 0},
    {"candidates_per_cycle": 13}, {"min_gain": 0}, {"source": "arbitrary code"},
])
def test_agent_cannot_change_admission_rules_or_escape_budgets(values):
    with pytest.raises(ValidationError):
        Budget.model_validate(values)


def test_workloads_are_bounded_and_cli_does_not_require_gpu():
    for prompts in ([], [""] , ["ok"] * 33):
        with pytest.raises(ValidationError):
            Workload(prompts=prompts)
    result = CliRunner().invoke(app, ["autoevolve", "--help"])
    assert result.exit_code == 0
    assert "--time-budget" in result.stdout
    assert "--rows" not in result.stdout


def test_unknown_worker_cannot_run(tmp_path):
    with pytest.raises(ValueError, match="Unknown worker"):
        execute_request("arbitrary.module", {}, tmp_path / "job", 1)
    assert not (tmp_path / "job").exists()


ENV = {"gpu": "test-gpu", "capability": [12, 0], "packages": {}, "cuda": "12.9"}


def artifact(output, shape, configs, winners=True):
    records = [Evaluation(shape=shape, config=c, correct=True, timing=decide([10.] * 5, [8.] * 5))
               for c in configs]
    if winners:
        records.append(records[0].model_copy(update={"stage": "confirmation"}))
    manifest = Manifest(
        run_id=new_run_id(), model_fingerprint="weights", kernel_hash="source", environment=ENV,
        eps=1e-6, evaluations=records,
        winners={shape.key: configs[0]} if winners else {},
        measurement={"model": {"hidden": 5120}},
    )
    path = output / manifest.run_id / "manifest.json"
    write_json(path, manifest.model_dump())
    return manifest, path


def test_composition_preserves_existing_confirmations_and_checks_compatibility(tmp_path):
    first, _ = artifact(tmp_path, target(7).shape, [KernelConfig()])
    second, _ = artifact(tmp_path, target(3, True).shape, [KernelConfig(num_warps=8)])
    merged = Manifest.model_validate_json(compose(second, first, tmp_path).read_text())
    assert merged.winners == first.winners | second.winners
    assert merged.measurement["lineage"] == [first.run_id, second.run_id]
    assert len(merged.evaluations) == len(first.evaluations) + len(second.evaluations)
    second.environment = ENV | {"gpu": "other"}
    with pytest.raises(ValueError, match="incompatible"):
        compose(second, first, tmp_path)


def simulated_workers(monkeypatch, outcomes=(True, False), winners=True, corrupt=None):
    calls = []
    observations = []

    def execute(module, request, directory, timeout):
        assert module == "evolith.lab.autonomy_worker" and timeout > 0
        job = StageJob.model_validate(request)
        calls.append(job)
        if job.kind == "observe":
            number = len(observations)
            chosen = target(7) if number == 0 else target(3, True, 500)
            obs = observation(chosen, target(1, True, 1))
            observations.append(obs)
            path = directory / "model-report.json"
            write_json(path, {
                "observation": obs.model_dump(), "environment": ENV, "kernel_hash": "source",
                "correctness": {"passed": True},
            })
            write_json(job.proposals, Proposal(candidates=[KernelConfig(), KernelConfig(num_warps=8)]).model_dump())
            return {"result": {"status": "baseline_measured", "report": str(path)}}
        if job.kind == "search":
            configs = Proposal.model_validate_json(job.proposals.read_text()).candidates
            shape = target(99).shape if corrupt == "target" else job.target
            manifest, path = artifact(job.output, shape, configs, winners)
            return {"result": {
                "status": "completed", "manifest": str(path), "evaluations": len(manifest.evaluations),
                "worker_errors": [],
            }}
        manifest = Manifest.model_validate_json(job.manifest.read_text())
        baseline = Manifest.model_validate_json(job.baseline_manifest.read_text()) if job.baseline_manifest else None
        index = sum(c.kind == "verify" for c in calls) - 1
        approved = outcomes[index % len(outcomes)]
        timing = decide([10.] * 5, [8. if approved else 11.] * 5).model_dump()
        if corrupt == "gate":
            timing["accepted"] = True
        proof = {
            "status": "verified", "model": {"fingerprint": "weights"}, "kernel_hash": "source",
            "kernel_run_id": manifest.run_id, "baseline_kernel_run_id": baseline.run_id if baseline else None,
            "required_candidate_shapes": [job.target.key], "workload_fingerprint": "workload",
            "deployment_approved": approved or corrupt == "gate",
            "correctness": {"passed": True}, "adapter_exercised": True,
            "e2e_measurement": timing, "observed_paired_e2e_speedup": timing["speedup"],
        }
        if corrupt == "provenance":
            proof["baseline_kernel_run_id"] = "different-comparison"
        path = directory / "verification.json"
        write_json(path, proof)
        return {"result": {"report": str(path), "deployment_approved": proof["deployment_approved"]}}

    monkeypatch.setattr(autonomy, "execute_request", execute)
    return calls


@pytest.mark.parametrize("outcomes", [(True, False), (True, True), (False, False)])
def test_controller_reobserves_selects_iterates_and_only_keeps_approved_successors(
    monkeypatch, tmp_path, outcomes,
):
    calls = simulated_workers(monkeypatch, outcomes)
    result = autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=2, seconds=60))
    report = json.loads(Path(result["report"]).read_text())
    assert result["stop_reason"] == "cycle_budget"
    assert result["deployment_approved"] == any(outcomes)
    assert [c.target.rows for c in calls if c.kind == "search"] == [7, 3]
    assert len({c.seed for c in calls if c.kind == "search"}) == 2
    assert len(report["cycles"]) == 2
    first, second = report["cycles"]
    assert first["approved"] == outcomes[0] and second["approved"] == outcomes[1]
    second_verify = [c for c in calls if c.kind == "verify"][1]
    assert bool(second_verify.baseline_manifest) == outcomes[0]
    if any(outcomes):
        incumbent = Manifest.model_validate_json(Path(result["approved_manifest"]).read_text())
        assert len(incumbent.winners) == sum(outcomes)
    assert [c for c in calls if c.kind == "observe"][1].history
    assert len(RunStore(tmp_path / "runs.sqlite3").recent("autonomous_cycle")) == 2


def test_failed_gate_cannot_be_overridden_by_a_worker_boolean(monkeypatch, tmp_path):
    simulated_workers(monkeypatch, outcomes=(False,), corrupt="gate")
    result = autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=1, seconds=60))
    assert not result["deployment_approved"]
    assert result["approved_manifest"] is None


@pytest.mark.parametrize("corrupt", ["target", "provenance"])
def test_mismatched_evidence_fails_closed_and_is_archived(monkeypatch, tmp_path, corrupt):
    simulated_workers(monkeypatch, corrupt=corrupt)
    with pytest.raises(ValueError):
        autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=1, seconds=60))
    sessions = RunStore(tmp_path / "runs.sqlite3").recent("autonomous_session")
    assert sessions[0]["status"] == "failed"
    assert sessions[0]["approved_manifest"] is None


def test_no_micro_winner_skips_model_verification_and_respects_evaluation_budget(monkeypatch, tmp_path):
    calls = simulated_workers(monkeypatch, winners=False)
    result = autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=8, evaluations=4, seconds=60))
    assert result["stop_reason"] == "evaluation_budget"
    assert result["evaluations"] <= 4
    assert not any(c.kind == "verify" for c in calls)
    assert not result["deployment_approved"]


def test_wall_time_budget_stops_cleanly_without_inventing_results(monkeypatch, tmp_path):
    def timeout(*args):
        raise WorkerTimeout("test deadline")
    monkeypatch.setattr(autonomy, "execute_request", timeout)
    result = autonomy.run(tmp_path, tmp_path, budget=Budget(seconds=30))
    assert result["stop_reason"] == "wall_time_budget"
    assert result["evaluations"] == 0
    assert not result["deployment_approved"]


def test_real_verifier_switches_back_to_incumbent_not_native_during_ab_trials(monkeypatch, tmp_path):
    from evolith.lab import serving

    _, incumbent_path = artifact(tmp_path, target(7).shape, [KernelConfig()])
    candidate, candidate_path = artifact(tmp_path, target(3, True).shape, [KernelConfig(num_warps=8)])
    incumbent_id = Manifest.model_validate_json(incumbent_path.read_text()).run_id
    modes = []

    class Engine:
        mode = None

        def __init__(self, **kwargs):
            pass

        def collective_rpc(self, name, args=()):
            if name == "evolith_restore":
                self.mode = "native"
            elif name == "evolith_activate":
                self.mode = args[0]["run_id"]
            elif name == "evolith_stats":
                return [{f"accelerated:{target(3, True).shape.key}": 1}] * 4
            return [{}] * 4

        def generate(self, *args, **kwargs):
            modes.append(self.mode)
            completion = SimpleNamespace(
                token_ids=[1], logprobs=[{1: SimpleNamespace(logprob=-0.1)}],
                text="test", finish_reason="length",
            )
            return [SimpleNamespace(outputs=[completion])]

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = Engine, lambda **kwargs: kwargs
    gpu = ModuleType("evolith.lab.gpu")
    gpu.environment = lambda: ENV
    gpu.model_info = lambda path: {"fingerprint": "weights"}
    kernels = ModuleType("evolith.lab.kernels")
    kernels.kernel_hash = lambda: "source"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "evolith.lab.gpu", gpu)
    monkeypatch.setitem(sys.modules, "evolith.lab.kernels", kernels)
    result = serving.verify_model(
        tmp_path, candidate_path, tmp_path, baseline_manifest=incumbent_path,
        required_shapes=[target(3, True).shape.key],
    )
    assert result["status"] == "verified"
    assert "native" not in modes
    assert modes[:12] == [incumbent_id] * 12
    assert set(modes) == {incumbent_id, candidate.run_id}
    proof = json.loads(Path(result["report"]).read_text())
    assert proof["baseline_kernel_run_id"] == incumbent_id


def test_matching_archived_failures_change_next_session_target_priority(monkeypatch, tmp_path):
    simulated_workers(monkeypatch, outcomes=(False,))
    autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=1, seconds=60))
    calls = simulated_workers(monkeypatch, outcomes=(False,))
    original = autonomy.execute_request

    def with_second_hot_target(module, request, directory, timeout):
        result = original(module, request, directory, timeout)
        if request["kind"] == "observe":
            path = Path(result["result"]["report"])
            report = json.loads(path.read_text())
            report["observation"]["targets"][1]["estimated_rank_total_us"] = 75
            write_json(path, report)
        return result

    monkeypatch.setattr(autonomy, "execute_request", with_second_hot_target)
    autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=1, seconds=60))
    assert [c.target.rows for c in calls if c.kind == "search"] == [1]


def test_incomplete_observation_stops_before_search_instead_of_guessing(monkeypatch, tmp_path):
    calls = simulated_workers(monkeypatch)
    original = autonomy.execute_request

    def truncated(module, request, directory, timeout):
        result = original(module, request, directory, timeout)
        if request["kind"] == "observe":
            path = Path(result["result"]["report"])
            report = json.loads(path.read_text())
            report["observation"]["dropped_contract_calls"] = 1
            write_json(path, report)
        return result

    monkeypatch.setattr(autonomy, "execute_request", truncated)
    result = autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=1, seconds=60))
    assert result["stop_reason"] == "observation_capacity_gap"
    assert not any(c.kind == "search" for c in calls)
    assert not result["deployment_approved"]


def test_mixed_observation_preserves_unsupported_contracts_while_searching_supported_targets(
    monkeypatch, tmp_path,
):
    simulated_workers(monkeypatch, outcomes=(False,))
    original = autonomy.execute_request

    def with_capability_gap(module, request, directory, timeout):
        result = original(module, request, directory, timeout)
        if request["kind"] == "observe":
            path = Path(result["result"]["report"])
            report = json.loads(path.read_text())
            unsupported = target().model_copy(update={
                "label": "unsupported", "shape": None,
                "unsupported_reasons": ["3D normalization is outside the trusted capability"],
            })
            report["observation"]["targets"].append(unsupported.model_dump())
            write_json(path, report)
        return result

    monkeypatch.setattr(autonomy, "execute_request", with_capability_gap)
    result = autonomy.run(tmp_path, tmp_path, budget=Budget(cycles=1, seconds=60))
    report = json.loads(Path(result["report"]).read_text())
    assert report["capability_gaps"][0]["label"] == "unsupported"
    assert report["cycles"][0]["capability_gaps"] == report["capability_gaps"]
    assert report["cycles"][0]["target"]["shape"]["rows"] == 7
