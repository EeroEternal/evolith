from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolith.lab import kernel_diagnosis
from evolith.lab import kernel_trace as trace_module
from evolith.lab.kernel_diagnosis import compare_rank, trace_environment, values_equal
from evolith.lab.kernel_trace import (
    KernelTrace,
    descriptor,
    safe_metadata,
    tensor_value,
)


def scalar(value):
    return {"complete": True, "scalar": value}


def tensor(digest="same"):
    return {"complete": True, "finite": True, "dtype": "torch.bfloat16",
            "shape": [2], "stride": [1], "value_sha256": digest}


def event():
    return {"kind": "values", "index": 0, "occurrence": 0, "name": "reduction",
            "source_sha256": "code", "configuration": {"num_warps": 4},
            "inputs": {"in_ptr0": tensor(), "xnumel": scalar(2)},
            "outputs": {"out_ptr0": tensor()}}


def test_first_observed_difference_requires_aligned_complete_floating_outputs():
    first = event()
    second = copy.deepcopy(first)
    second["outputs"]["out_ptr0"]["value_sha256"] = "different"
    second["configuration"] = {"num_warps": 8}
    result = compare_rank([first], [second])
    difference = result["first_observed_output_difference"]
    assert difference["all_observed_inputs_equal"]
    assert difference["inputs_fully_observed"] and difference["configuration_changed"]
    assert len(result["aligned_prefix_configuration_changes"]) == 1
    assert not values_equal(first["outputs"], second["outputs"])
    assert values_equal(first["inputs"], second["inputs"])


@pytest.mark.parametrize("change", ["source", "incomplete", "nonfinite"])
def test_unknown_or_incomplete_evidence_is_not_called_a_numerical_difference(change):
    first, second = event(), event()
    if change == "source":
        second["source_sha256"] = "other-code"
    elif change == "incomplete":
        second["outputs"]["out_ptr0"]["complete"] = False
    else:
        second["outputs"]["out_ptr0"]["finite"] = False
    result = compare_rank([first], [second])
    assert result["first_observed_output_difference"] is None
    assert result["first_alignment_gap"] or result["first_coverage_gap"]


def test_metadata_does_not_render_opaque_values_or_nonfinite_json():
    class Opaque:
        def __repr__(self):
            pytest.fail("Opaque argument contents must not be rendered.")

    assert safe_metadata(Opaque()) == {"unsupported_type": "Opaque"}
    assert safe_metadata(float("nan")) is None
    assert safe_metadata({"x": [1, True, "value"]}) == {"x": [1, True, "value"]}


def test_observer_keeps_original_arguments_and_return_value(tmp_path):
    class FakeTensor:
        pass

    torch = SimpleNamespace(Tensor=FakeTensor, cuda=SimpleNamespace(
        is_current_stream_capturing=lambda: False,
    ))
    tuner = SimpleNamespace(
        fn=SimpleNamespace(src="def kernel(xnumel): pass", __name__="kernel"),
        inductor_meta={"kernel_name": "kernel"}, triton_meta={"signature": {"xnumel": "i32"}},
        launchers=[SimpleNamespace(config=SimpleNamespace(
            kwargs={"XBLOCK": 32}, num_warps=4, num_stages=1,
        ), cache_hash="binary")], mutated_arg_names=[],
    )
    trace = KernelTrace(tmp_path, 0, torch)
    expected = object()
    calls = []

    def original(instance, *args, **kwargs):
        calls.append((instance, args, kwargs))
        return expected

    assert trace.call(original, tuner, (31,), {"stream": 7}) is expected
    assert calls == [(tuner, (31,), {"stream": 7})]
    assert descriptor(tuner)["configuration"]["num_warps"] == 4
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["kind"] == "configuration"


def test_new_output_pointers_are_observed_without_being_mutated_inputs(monkeypatch, tmp_path):
    class FakeTensor:
        def __init__(self, name, value):
            self.name, self.value = name, value

    reads = []

    def readback(value, torch, remaining):
        reads.append(value.name)
        return tensor(str(value.value)), 1

    monkeypatch.setattr(trace_module, "tensor_value", readback)
    torch = SimpleNamespace(Tensor=FakeTensor, cuda=SimpleNamespace(
        is_current_stream_capturing=lambda: False,
    ))
    source, target = FakeTensor("input", 2), FakeTensor("output", 0)
    tuner = SimpleNamespace(
        fn=SimpleNamespace(src="def kernel(): pass", __name__="kernel"),
        inductor_meta={}, triton_meta={"signature": {"in_ptr0": "*fp32", "out_ptr0": "*fp32"}},
        launchers=[], mutated_arg_names=[],
    )
    trace = KernelTrace(tmp_path, 0, torch)
    trace.active = True

    def original(instance, *args, **kwargs):
        assert instance is tuner and args == (source, target) and kwargs == {"stream": 7}
        target.value = source.value * 3
        return "unchanged"

    assert trace.call(original, tuner, (source, target), {"stream": 7}) == "unchanged"
    row = json.loads(trace.path.read_text().splitlines()[-1])
    assert reads == ["input", "output"]
    assert row["inputs"]["in_ptr0"]["value_sha256"] == "2"
    assert row["outputs"]["out_ptr0"]["value_sha256"] == "6"
    assert source.value == 2 and trace.copied_bytes == 2


def test_missing_outputs_do_not_hide_earlier_incoming_differences():
    empty = event() | {"outputs": {}}
    before = event() | {"index": 1, "occurrence": 1}
    after = copy.deepcopy(before)
    after["inputs"]["in_ptr0"] = tensor("incoming-difference")
    after["outputs"]["out_ptr0"] = tensor("output-difference")
    report = compare_rank([empty, before], [empty, after])
    assert report["first_coverage_gap"] == {"index": 0, "reason": "missing_output_snapshots"}
    assert report["empty_output_events"] == [1, 1]
    assert report["first_observed_input_difference"]["index"] == 1
    assert not report["first_observed_output_difference"]["all_observed_inputs_equal"]


def test_reproduction_controls_alone_do_not_establish_attribution(monkeypatch, tmp_path):
    cold, warm, control = [
        {"run_id": name, "prompt_token_ids": [1, 2], "records": []}
        for name in ("cold", "warm", "control")
    ]
    for trial in (cold, warm):
        folder = tmp_path / trial["run_id"] / "trace"
        folder.mkdir(parents=True)
        row = event()
        if trial is warm:
            row["inputs"]["in_ptr0"] = tensor("incoming")
            row["outputs"]["out_ptr0"] = tensor("outgoing")
        for rank in range(4):
            (folder / f"rank-{rank}.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(kernel_diagnosis, "outputs_match", lambda *_: True)
    monkeypatch.setattr(kernel_diagnosis, "comparison", lambda *_: {"correct": False})
    report = kernel_diagnosis.analyze(tmp_path, cold, warm, control)
    assert report["reproduction_controls_passed"]
    assert not report["matched_input_output_divergence"]
    assert not report["attribution_eligible"]


def test_capture_arms_only_for_the_exact_declared_prefill(tmp_path):
    trace = KernelTrace(tmp_path, 0, SimpleNamespace())
    (tmp_path / "arm.json").write_text(json.dumps({"prompt_token_ids": [11, 12]}))
    schedule = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(prompt_token_ids=[11, 12])],
                               total_num_scheduled_tokens=2)
    with trace.request(schedule):
        assert trace.active
    assert not trace.active and trace.finished
    assert (tmp_path / "rank-0-complete.json").is_file()
    other = KernelTrace(tmp_path / "other", 1, SimpleNamespace())
    (other.directory / "arm.json").write_text(json.dumps({"prompt_token_ids": [11, 13]}))
    with pytest.raises(ValueError, match="fixed prefill"), other.request(schedule):
        pass


def test_trace_environment_restores_prior_values(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLITH_KERNEL_TRACE_MODE", "old")
    monkeypatch.delenv("EVOLITH_KERNEL_TRACE_DIR", raising=False)
    with pytest.raises(RuntimeError), trace_environment(tmp_path, True):
        assert os.environ["EVOLITH_KERNEL_TRACE_MODE"] == "observe"
        assert os.environ["EVOLITH_KERNEL_TRACE_DIR"] == str(tmp_path)
        raise RuntimeError("probe interrupted")
    assert os.environ["EVOLITH_KERNEL_TRACE_MODE"] == "old"
    assert "EVOLITH_KERNEL_TRACE_DIR" not in os.environ


def test_diagnostic_worker_command_is_fixed_and_not_model_supplied(monkeypatch, tmp_path):
    from evolith.lab import engine_worker
    from evolith.lab.engine import EngineConfig, EngineJob

    job = EngineJob(model_path=tmp_path, output=tmp_path, config=EngineConfig())
    calls = []

    class Captured(Exception):
        pass

    def popen(command, **kwargs):
        calls.append(command)
        raise Captured

    monkeypatch.setattr(engine_worker.subprocess, "Popen", popen)
    with pytest.raises(Captured), engine_worker.server(job, tmp_path, diagnostic_worker=True):
        pass
    command = calls[0]
    assert command[-2:] == ["--worker-cls", "evolith.lab.trace_worker.KernelTraceWorker"]
    with pytest.raises(ValueError), engine_worker.server(job, tmp_path, diagnostic_worker="source"):
        pass
    assert len(calls) == 1


def test_config_inventory_compares_actual_observations_across_ranks_separately():
    first = event() | {"kind": "configuration"}
    second = first | {"configuration": {"num_warps": 8}}
    inventory = compare_rank([first], [second])["configuration_inventory"]
    assert inventory["shared_keys"] == 1 and inventory["changed_shared_keys"] == 1


def test_tensor_readback_is_copy_only_and_ignores_signed_zero():
    torch = pytest.importorskip("torch")
    data = torch.tensor([0., 1., -2.], dtype=torch.bfloat16)
    before = data.clone()
    observed, copied = tensor_value(data, torch, 1024)
    signed_zero = torch.tensor([-0., 1., -2.], dtype=torch.bfloat16)
    other, _ = tensor_value(signed_zero, torch, 1024)
    assert torch.equal(data, before)
    assert copied == 6 and observed["complete"]
    assert observed["value_sha256"] == other["value_sha256"]
    partial, copied = tensor_value(data, torch, 0)
    assert not partial["complete"] and copied == 0


def test_independent_replay_preserves_inputs_and_selected_launcher(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from evolith.lab.kernel_replay import paired_replay, prepare

    x = torch.tensor([[1., 2.], [3., 4.]])
    weight = torch.ones(2)
    actual = 2 * x
    selected = SimpleNamespace(config=SimpleNamespace(
        kwargs={"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16, num_stages=1,
    ))

    def compile_config(cfg):
        def launch(input_, weight_, output, xnumel, r0_numel, **kwargs):
            assert kwargs == {"stream": 7}
            output.copy_(2 * input_ + (cfg.kwargs["R0_BLOCK"] == 8192))
        return SimpleNamespace(make_launcher=lambda: launch)

    tuner = SimpleNamespace(
        triton_meta={"signature": {key: "value" for key in
                                  ("in_ptr0", "in_ptr1", "out_ptr1", "xnumel", "r0_numel")}},
        mutated_arg_names=[], launchers=[selected], _precompile_config=compile_config,
    )
    args = (x, weight, actual, 2, 2)
    report = paired_replay(tuner, args, prepare(tuner, args, torch),
                           {"stream": 7}, torch, tmp_path, 0)
    assert report["selected_replay_matches_actual"] and not report["paired_outputs_equal"]
    assert report["different_elements"] == 4
    assert tuner.launchers[0] is selected
    assert torch.equal(actual, 2 * x) and torch.equal(weight, torch.ones(2))


def test_replay_rebuilds_stripped_aot_jit_without_mutating_model(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    import hashlib

    from evolith.lab import kernel_replay

    source = "def replay_test(x):\n    return x\n"
    monkeypatch.setattr(kernel_replay, "TARGET", hashlib.sha256(source.encode()).hexdigest())
    original = SimpleNamespace(fn=SimpleNamespace(src=source, _hash_lock=None))
    copied = kernel_replay.replay_compiler(original, tmp_path, 0)
    assert copied is not original and copied.fn is not original.fn
    assert original.fn._hash_lock is None
    assert isinstance(copied.fn.cache_key, str)


def test_driver_has_exactly_three_bounded_nonpublishing_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_diagnosis, "inspect_runtime", lambda _: (
        {"fingerprint": "model"}, {"cuda_available": True},
    ))
    calls = []

    def fake_case(model, output, **options):
        calls.append(options)
        return {"run_id": options["run_id"], "status": "observed"}

    monkeypatch.setattr(kernel_diagnosis, "case", fake_case)
    monkeypatch.setattr(kernel_diagnosis, "analyze", lambda output, *trials: {
        "count": len(trials), "attribution_eligible": False,
    })
    result = kernel_diagnosis.run(tmp_path, tmp_path, tmp_path / "cache", seconds=600)
    report = json.loads(Path(result["report"]).read_text())
    assert [c["enabled"] for c in calls] == [True, True, False]
    assert report["analysis"]["count"] == 3
    assert not list(tmp_path.glob("*/engine.json"))
    with pytest.raises(ValueError):
        kernel_diagnosis.run(tmp_path, tmp_path, tmp_path / "cache", seconds=1681)
