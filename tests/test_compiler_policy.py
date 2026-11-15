from __future__ import annotations

import os
from pathlib import Path

import pytest

from evolith.lab.compiler_policy import compiler_policy_report


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_confirms_explicit_compiler_policy(enabled):
    report = compiler_policy_report(2, str(int(enabled)), enabled)
    assert report["rank"] == 2 and report["requested"] is enabled
    assert report["inductor_deterministic"] is enabled


@pytest.mark.parametrize("rank,requested,actual", [
    (True, "1", True), (4, "1", True), (0, None, True), (0, "yes", True),
    (0, "1", 1), (0, "1", False), (0, "0", True),
])
def test_missing_unsupported_or_ignored_policy_fails_closed(rank, requested, actual):
    with pytest.raises(ValueError):
        compiler_policy_report(rank, requested, actual)


@pytest.mark.parametrize("flag", ["True", "False", "1", "None"])
def test_generated_reduction_metadata_must_confirm_policy(tmp_path, flag):
    from evolith.lab.compiler_policy import reduction_policy_inventory

    (tmp_path / "generated.py").write_text(
        "@triton_heuristics.reduction(inductor_meta={'deterministic': " + flag + "})\n"
        "def generated_norm(): pass\n"
    )
    if flag != "True":
        with pytest.raises(ValueError):
            reduction_policy_inventory(tmp_path, True)
    else:
        assert reduction_policy_inventory(tmp_path, True)[0]["deterministic"] is True


def test_missing_generated_reduction_policy_is_not_assumed_successful(tmp_path):
    from evolith.lab.compiler_policy import reduction_policy_inventory

    with pytest.raises(ValueError):
        reduction_policy_inventory(tmp_path, True)


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_pins_policy_before_import_without_changing_parent(monkeypatch, tmp_path, enabled):
    from evolith.lab import engine_worker
    from evolith.lab.engine import EngineConfig, EngineJob

    monkeypatch.setenv("TORCHINDUCTOR_DETERMINISTIC", str(int(not enabled)))
    captured = []

    class Captured(Exception):
        pass

    def popen(command, **kwargs):
        captured.append(kwargs["env"])
        raise Captured

    monkeypatch.setattr(engine_worker.subprocess, "Popen", popen)
    job = EngineJob(model_path=tmp_path, output=tmp_path, config=EngineConfig())
    with pytest.raises(Captured), engine_worker.server(
        job, tmp_path, diagnostic_worker=enabled, deterministic_compiler=enabled,
    ):
        pass
    assert captured[0]["TORCHINDUCTOR_DETERMINISTIC"] == str(int(enabled))
    assert os.environ["TORCHINDUCTOR_DETERMINISTIC"] == str(int(not enabled))
    if enabled:
        with pytest.raises(ValueError), engine_worker.server(
            job, tmp_path, deterministic_compiler=True,
        ):
            pass
        assert len(captured) == 1


@pytest.mark.parametrize("graph", ["native", "piecewise"])
def test_compiler_policy_reaches_explicit_vllm_options_without_overwriting_graphs(
    monkeypatch, tmp_path, graph,
):
    import json

    from evolith.lab import engine_worker
    from evolith.lab.engine import EngineConfig, EngineJob

    commands = []

    class Captured(Exception):
        pass

    def popen(command, **_):
        commands.append(command)
        raise Captured

    monkeypatch.setattr(engine_worker.subprocess, "Popen", popen)
    job = EngineJob(model_path=tmp_path, output=tmp_path, config=EngineConfig(graph=graph))
    with pytest.raises(Captured), engine_worker.server(
        job, tmp_path, diagnostic_worker=True, deterministic_compiler=True,
    ):
        pass
    command = commands[0]
    assert command.count("--compilation-config") == 1
    config = json.loads(command[command.index("--compilation-config") + 1])
    assert config["inductor_compile_config"] == {"deterministic": True, "combo_kernels": False}
    assert ("cudagraph_mode" not in config if graph == "native"
            else config["cudagraph_mode"] == "PIECEWISE")


def test_pinned_vllm_compilation_schema_preserves_explicit_policy():
    pytest.importorskip("vllm")
    from vllm.config import CompilationConfig

    config = CompilationConfig(inductor_compile_config={"deterministic": True,
                                                         "combo_kernels": False})
    assert config.inductor_compile_config["deterministic"] is True
    assert config.inductor_compile_config["combo_kernels"] is False
    assert config.inductor_compile_config.get("benchmark_combo_kernel") is not True


def test_disabled_combo_fusion_does_not_bypass_deterministic_benchmark_guard():
    pytest.importorskip("torch")
    config = pytest.importorskip("torch._inductor.config")
    benchmarking = pytest.importorskip("torch._inductor.runtime.benchmarking")
    with config.patch({"deterministic": True, "combo_kernels": False}):
        assert config.deterministic is True and config.combo_kernels is False
        with pytest.raises(RuntimeError, match="deterministic mode"):
            benchmarking.may_ban_benchmarking()


def test_pinned_torch_deterministic_policy_is_generic_and_prevents_dynamic_reexpansion():
    pytest.importorskip("torch")
    th = pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    if not hasattr(th, "filter_reduction_configs_for_determinism"):
        pytest.skip("Pinned Torch compiler policy is unavailable.")
    for hints in ({"x": 2048, "r0_": 8192}, {"x": 32, "r0_": 64}, {"x": 128, "r0_": 1024}):
        meta = {"reduction_hint": th.ReductionHint.INNER, "num_load": 3, "num_reduction": 1,
                "add_persistent_rblock": True, "deterministic": True}
        configs = th._reduction_configs(size_hints=hints, inductor_meta=meta,
                                       triton_meta={"native_matmul": False})
        selected = th.filter_reduction_configs_for_determinism(meta, configs)
        assert len(selected) == 1 and selected[0] in configs
    tuner = th.CachingAutotuner.__new__(th.CachingAutotuner)
    tuner.device_props = object()
    tuner.deterministic_mode = True
    # A deterministic tuner must not even query device properties/launchers here.
    tuner._dynamic_scale_rblock()


def test_pinned_torch_reads_policy_in_fresh_interpreter(tmp_path):
    import subprocess
    import sys

    pytest.importorskip("torch")
    env = os.environ | {"TORCHINDUCTOR_DETERMINISTIC": "1",
                        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor")}
    result = subprocess.run(
        [sys.executable, "-c", "from torch._inductor import config; assert config.deterministic is True"],
        env=env, capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("worker_agrees", [False, True])
def test_repair_pair_keeps_policy_separate_and_cannot_admit_on_worker_mismatch(
    monkeypatch, tmp_path, worker_agrees,
):
    import json

    from evolith.lab import kernel_repair

    output = tmp_path / "trials"
    output.mkdir()
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"started_unix": 100, "deadline_unix": 1840,
                                  "gpus": [0, 1, 2, 3], "approved_seconds": 1800}))
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({
        "status": "verified",
        "paired_replays": [{"selected_replay_matches_actual": True, "paired_outputs_equal": False}] * 4,
    }))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"run_id": "frozen", "records": []}))
    original_frozen = frozen.read_bytes()
    monkeypatch.setattr(kernel_repair.time, "time", lambda: 400)
    monkeypatch.setattr(kernel_repair.signal, "alarm", lambda _: 0)
    monkeypatch.setattr(kernel_repair.signal, "signal", lambda *_: None)
    monkeypatch.setattr(kernel_repair, "inspect_runtime", lambda _: ({"fingerprint": "model"}, {}))
    monkeypatch.setattr(kernel_repair, "comparison", lambda *_: {"correct": True})
    calls = []

    def fake_case(_model, target, **options):
        calls.append(options)
        directory = target / options["run_id"] / "trace"
        directory.mkdir(parents=True)
        for rank in range(4):
            (directory / f"rank-{rank}-compiler.json").write_text(json.dumps({
                "rank": rank, "requested": True, "inductor_deterministic": worker_agrees,
            }))
        cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        Path(cache).mkdir(parents=True, exist_ok=True)
        (Path(cache) / "generated.py").write_text(
            "@triton_heuristics.reduction(inductor_meta={'deterministic': True})\n"
            "def generated_norm(): pass\n"
        )
        return {"run_id": options["run_id"], "status": "observed", "records": []}

    monkeypatch.setattr(kernel_repair, "case", fake_case)
    args = (output, tmp_path / "cache", budget, proof, frozen, "deterministic_compiler")
    if worker_agrees:
        result = kernel_repair.run(*args)
        assert result["status"] == "restart_pair_passed" and len(calls) == 2
        report = json.loads(Path(result["report"]).read_text())
        assert report["historical_values"]["admission_compatible"] is False
    else:
        with pytest.raises(ValueError, match="confirm requested compiler policy"):
            kernel_repair.run(*args)
        assert len(calls) == 1
    assert all(not c["rank_local_cache"] and c["deterministic_compiler"] and not c["enabled"]
               for c in calls)
    assert frozen.read_bytes() == original_frozen
    assert not list(output.rglob("engine.json"))


_INTEGER_COUNT = '''
@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 8192},
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0)},
    inductor_meta={'kernel_name': 'triton_red_fused_ge_sum_0', 'deterministic': False},
)
@triton.jit
def triton_red_fused_ge_sum_0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((30 + ks0) // 31)
        tmp2 = tmp0 < ks0
        tmp3 = tl.load(in_ptr0 + (r0_1 + x0*((30 + ks0) // 31)), r0_mask & tmp2 & xmask, other=0.0)
        tmp5 = tl.broadcast_to(tl.load(in_ptr1 + (0)), [1, 1])
        tmp6 = tl.where(tmp2, tmp5, 0.0)
        tmp8 = (tmp3 >= tmp6).to(tl.int64)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp11 = tl.broadcast_to(tl.where(tmp2, tmp8, tmp9), [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 32},
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}},
    inductor_meta={'kernel_name': 'triton_per_fused_ge_sum_1', 'deterministic': False},
)
@triton.jit
def triton_per_fused_ge_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 31
    R0_BLOCK: tl.constexpr = 32
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0_index), r0_mask, other=0.0)
    tmp3 = tl.where(r0_mask, tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK]), 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.int64)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
'''

_FLOAT_NORM = '''
@triton_heuristics.reduction(
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}},
    inductor_meta={'deterministic': FLAG},
)
@triton.jit
def triton_red_fused_mean_pow_0(in_ptr0, out_ptr0, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (r0_index)).to(tl.float32)
    tmp1 = tl.sum(tmp0 * tmp0, 1)[:, None]
    tl.store(out_ptr0 + (0), tmp1, None)
'''


def _integer_kernel(body: str, signature: str = "{'in_ptr0': '*fp32', 'out_ptr0': '*i64', "
                    "'R0_BLOCK': 'constexpr'}") -> str:
    return ("@triton_heuristics.reduction(triton_meta={'signature': " + signature + "},\n"
            "    inductor_meta={'deterministic': False})\n"
            "def generated(in_ptr0, out_ptr0, R0_BLOCK : tl.constexpr):\n"
            "    r0_index = tl.arange(0, R0_BLOCK)[None, :]\n"
            "    tmp0 = tl.load(in_ptr0 + (r0_index))\n" + body)


def test_integer_count_reductions_are_recorded_but_floating_policy_still_required(tmp_path):
    from evolith.lab.compiler_policy import reduction_policy_inventory

    (tmp_path / "count.py").write_text(_INTEGER_COUNT)
    with pytest.raises(ValueError, match="order-sensitive"):
        reduction_policy_inventory(tmp_path, True)
    (tmp_path / "norm.py").write_text(_FLOAT_NORM.replace("FLAG", "True"))
    rows = {row["kernel"]: row for row in reduction_policy_inventory(tmp_path, True)}
    assert rows["triton_red_fused_ge_sum_0"]["integer_only"] is True
    assert rows["triton_per_fused_ge_sum_1"]["integer_only"] is True
    assert rows["triton_red_fused_mean_pow_0"] == rows["triton_red_fused_mean_pow_0"] | {
        "integer_only": False, "deterministic": True}
    (tmp_path / "norm.py").write_text(_FLOAT_NORM.replace("FLAG", "False"))
    with pytest.raises(ValueError, match="triton_red_fused_mean_pow_0"):
        reduction_policy_inventory(tmp_path, True)


@pytest.mark.parametrize("body,signature", [
    # Floating accumulation cast to an integer afterwards is still order-sensitive.
    ("    tmp1 = tl.sum(tmp0, 1).to(tl.int64)\n", None),
    ("    tmp1 = tl.sum((tmp0 >= 0.0).to(tl.float32), 1)\n", None),
    ("    tmp1 = tl.sum(tl.where(tmp0 >= 0.0, 1, 0.5), 1)\n", None),
    ("    tmp1 = triton_helpers.max2((tmp0 >= 0.0).to(tl.int64), 1)\n", None),
    ("    tmp1 = tl.max((tmp0 >= 0.0).to(tl.int64), 1)\n", None),
    ("    tmp1 = tl.sum(tl.atomic_add(out_ptr0, (tmp0 >= 0.0).to(tl.int64)), 1)\n", None),
    ("    tmp1 = tl.sum(unknown_helper(tmp0), 1)\n", None),
    ("    while True:\n        pass\n    tmp1 = tl.sum((tmp0 >= 0.0).to(tl.int64), 1)\n", None),
    ("    tmp1 = tl.sum(tmp0, 1)\n", None),
    ("    tmp1 = tl.sum(tmp0, 1)\n", "{}"),
    ("    tmp1 = tl.sum(tmp0, 1)\n", "{'in_ptr0': OPAQUE}"),
    ("    tmp1 = tl.sum(tmp0, 1)\n", "{'in_ptr0': '*fp32', 'R0_BLOCK': 'i32'}"),
])
def test_unproven_or_floating_reductions_never_receive_integer_exemption(tmp_path, body, signature):
    from evolith.lab.compiler_policy import reduction_policy_inventory

    (tmp_path / "norm.py").write_text(_FLOAT_NORM.replace("FLAG", "True"))
    (tmp_path / "candidate.py").write_text(_integer_kernel(body, *([signature] if signature else [])))
    with pytest.raises(ValueError, match="generated"):
        reduction_policy_inventory(tmp_path, True)


def test_boolean_count_is_integer_only(tmp_path):
    from evolith.lab.compiler_policy import reduction_policy_inventory

    (tmp_path / "norm.py").write_text(_FLOAT_NORM.replace("FLAG", "True"))
    (tmp_path / "count.py").write_text(_integer_kernel("    tmp1 = tl.sum(tmp0 >= 0.0, 1)\n"))
    rows = {row["kernel"]: row for row in reduction_policy_inventory(tmp_path, True)}
    assert rows["generated"]["integer_only"] is True and rows["generated"]["deterministic"] is False


def _repair_setup(monkeypatch, tmp_path, fake_case):
    import json

    from evolith.lab import kernel_repair

    output = tmp_path / "trials"
    output.mkdir()
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"started_unix": 100, "deadline_unix": 1840,
                                  "gpus": [0, 1, 2, 3], "approved_seconds": 1800}))
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({
        "status": "verified",
        "paired_replays": [{"selected_replay_matches_actual": True, "paired_outputs_equal": False}] * 4,
    }))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"run_id": "frozen", "records": []}))
    monkeypatch.setattr(kernel_repair.time, "time", lambda: 400)
    monkeypatch.setattr(kernel_repair.signal, "alarm", lambda _: 0)
    monkeypatch.setattr(kernel_repair.signal, "signal", lambda *_: None)
    monkeypatch.setattr(kernel_repair, "inspect_runtime", lambda _: ({"fingerprint": "model"}, {}))
    monkeypatch.setattr(kernel_repair, "comparison", lambda *_: {"correct": True})
    monkeypatch.setattr(kernel_repair, "case", fake_case)
    return kernel_repair, (output, tmp_path / "cache", budget, proof, frozen, "deterministic_compiler")


def _fake_trial(target, options, mutate=None):
    import json

    directory = target / options["run_id"] / "trace"
    directory.mkdir(parents=True)
    for rank in range(4):
        (directory / f"rank-{rank}-compiler.json").write_text(json.dumps({
            "rank": rank, "requested": True, "inductor_deterministic": True}))
    inductor = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    inductor.mkdir(parents=True, exist_ok=True)
    (inductor / "generated.py").write_text(
        "@triton_heuristics.reduction(inductor_meta={'deterministic': True})\n"
        "def generated_norm(): pass\n")
    if mutate:
        mutate(Path(os.environ["TRITON_CACHE_DIR"]))
    return {"run_id": options["run_id"], "status": "observed", "records": []}


def _seed(tmp_path, name="ABC/chunk_fwd_kernel_o.autotune.json"):
    import json

    seed = tmp_path / "seed"
    (seed / name).parent.mkdir(parents=True)
    (seed / name).write_text(json.dumps({"key": [1], "configs_timings": [[{"kwargs": {}}, [1.0]]]}))
    return seed


def test_autotune_seed_is_copied_before_first_process_and_recorded(monkeypatch, tmp_path):
    import json

    seen = []

    def fake(_model, target, **options):
        seen.append(sorted(p.name for p in Path(os.environ["TRITON_CACHE_DIR"]).rglob("*")))
        return _fake_trial(target, options, mutate=lambda t: (t / "NEW").mkdir(exist_ok=True) or (
            t / "NEW" / "other.autotune.json").write_text("{}"))

    kernel_repair, args = _repair_setup(monkeypatch, tmp_path, fake)
    result = kernel_repair.run(*args, autotune_seed=_seed(tmp_path))
    report = json.loads(Path(result["report"]).read_text())
    assert "chunk_fwd_kernel_o.autotune.json" in seen[0]
    assert list(report["autotune_seed"]["files"]) == ["ABC/chunk_fwd_kernel_o.autotune.json"]
    assert all(t["autotune"]["seeded_unchanged"] for t in report["trials"])
    assert report["trials"][0]["autotune"]["unseeded_decisions"] == ["NEW/other.autotune.json"]


def test_changed_seeded_decision_fails_closed(monkeypatch, tmp_path):
    def fake(_model, target, **options):
        return _fake_trial(target, options, mutate=lambda t: (
            t / "ABC" / "chunk_fwd_kernel_o.autotune.json").write_text("{}"))

    kernel_repair, args = _repair_setup(monkeypatch, tmp_path, fake)
    with pytest.raises(ValueError, match="changed during the run"):
        kernel_repair.run(*args, autotune_seed=_seed(tmp_path))


@pytest.mark.parametrize("name", ["ABC/kernel.cubin", "kernel.autotune.json", "A/B/k.autotune.json"])
def test_autotune_seed_rejects_anything_but_decisions(monkeypatch, tmp_path, name):
    calls = []
    kernel_repair, args = _repair_setup(monkeypatch, tmp_path, lambda *a, **k: calls.append(k))
    with pytest.raises(ValueError, match="only contain decisions"):
        kernel_repair.run(*args, autotune_seed=_seed(tmp_path, name))
    assert not calls


def test_autotune_seed_is_not_available_for_rank_cache_policy(monkeypatch, tmp_path):
    kernel_repair, args = _repair_setup(monkeypatch, tmp_path, lambda *a, **k: None)
    with pytest.raises(ValueError, match="only defined for the compiler policy"):
        kernel_repair.run(*args[:5], "rank_local_cache", autotune_seed=_seed(tmp_path))
