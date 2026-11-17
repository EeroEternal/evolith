from __future__ import annotations

import pytest

from evolith.controller.archive import SteppingStonesArchive
from evolith.controller.engine import EvolithRSIEngine
from evolith.planes.l1_kernels import L1_KERNEL_TARGETS
from evolith.planes.l2_policies import BASELINE_L2_POLICY_CODE, L2_POLICY_MUTATIONS
from evolith.verifiers.dream_simulator import DreamTraceSimulator
from evolith.verifiers.kernel_sandbox import HardenedKernelVerifier


def test_stage1_static_ast_blocks_memoization_reward_hack() -> None:
    verifier = HardenedKernelVerifier()
    spec = L1_KERNEL_TARGETS["fused_add_rmsnorm_fp8_quant"]
    # Mutation #2 is the _STATIC_TENSOR_CACHE reward hack
    _, _, _, hack_code = spec.mutations[1]

    res = verifier.evaluate(
        op_name="fused_add_rmsnorm_fp8_quant",
        candidate_code=hack_code,
        baseline_code=spec.baseline_code,
    )
    assert not res.is_valid
    assert "Memoization Hack" in res.rejection_reason


def test_stage2_input_checksum_blocks_inplace_corruption() -> None:
    verifier = HardenedKernelVerifier()
    spec = L1_KERNEL_TARGETS["fused_add_rmsnorm_fp8_quant"]

    corrupting_code = """
import numpy as np

def kernel_fn(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6):
    # Corrupts caller's input buffer in-place to save allocation
    x += residual
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    normed = (x / rms) * weight
    amax = np.max(np.abs(normed)) + 1e-12
    scale = amax / 448.0
    q = np.clip(np.rint(normed / scale), -448.0, 448.0)
    return q, scale
"""
    res = verifier.evaluate(
        op_name="fused_add_rmsnorm_fp8_quant",
        candidate_code=corrupting_code,
        baseline_code=spec.baseline_code,
    )
    assert not res.is_valid
    assert "In-place mutation of read-only input tensors" in res.rejection_reason


def test_valid_l1_kernel_passes_all_stages_and_generates_reflection_card() -> None:
    verifier = HardenedKernelVerifier()
    spec = L1_KERNEL_TARGETS["fused_add_rmsnorm_fp8_quant"]
    _, _, _, valid_code = spec.mutations[2]

    res = verifier.evaluate(
        op_name="fused_add_rmsnorm_fp8_quant",
        candidate_code=valid_code,
        baseline_code=spec.baseline_code,
    )
    assert res.is_valid
    assert res.max_abs_diff < 1e-2
    assert "Hardware Reflection Card" in res.reflection_card
    assert len(res.profiles_by_bucket) == 3


def test_dream_trace_simulator_reward_and_worst_window_diagnosis() -> None:
    sim = DreamTraceSimulator()
    base_metrics = sim.simulate(BASELINE_L2_POLICY_CODE)
    _, _, _, evolved_code = L2_POLICY_MUTATIONS[-1]
    evolved_metrics = sim.simulate(evolved_code)

    assert evolved_metrics.goodput_qps > base_metrics.goodput_qps * 1.35
    assert evolved_metrics.p99_ttft_ms < base_metrics.p99_ttft_ms
    assert evolved_metrics.tool_pause_pin_hit_rate > base_metrics.tool_pause_pin_hit_rate
    assert len(base_metrics.worst_windows) > 0


def test_gepa_ast_crossover_and_subtractive_gc() -> None:
    engine = EvolithRSIEngine()
    parent_a = L2_POLICY_MUTATIONS[0][3]
    parent_b = L2_POLICY_MUTATIONS[1][3]

    crossed = engine.ast_crossover_l2_policies(parent_a, parent_b)
    assert "def propose_speculative_config" in crossed
    assert "def compute_chunked_prefill_budget" in crossed
    assert "def score_kv_block_retention" in crossed

    sim_res = engine.dream_simulator.simulate(crossed)
    assert sim_res.goodput_qps > 0

    with_docstring = '''from evolith.planes.l2_policies import EngineStepState

def propose_speculative_config(state: EngineStepState) -> tuple[str, int]:
    """Redundant docstring that should be stripped by subtractive GC."""
    return ("eagle3_tree", 4)
'''
    pruned = engine.subtractive_ablation_gc(with_docstring)
    assert "Redundant docstring" not in pruned


def test_full_dual_plane_evolution_and_ignition_report() -> None:
    archive = SteppingStonesArchive(db_path=":memory:")
    engine = EvolithRSIEngine(archive=archive)

    l1_nodes = engine.evolve_l1_kernels()
    l2_nodes = engine.evolve_l2_policies()
    ignition = engine.compute_ignition_report()

    assert len(l1_nodes) >= 6
    assert any(not n.accepted_by_gate and n.speedup == 0.0 for n in l1_nodes)
    assert len(l2_nodes) == 5
    assert ignition.evolved_agent_tok_s > ignition.baseline_agent_tok_s
    assert ignition.ignition_multiplier > 1.5
    text, ratio = archive.format_compressed_memory()
    assert "Hyp:" in text
    assert ratio > 2.0


def test_sandbox_blocks_attribute_calls_unauthorized_imports_dunders_and_hangs() -> None:
    verifier = HardenedKernelVerifier()
    spec = L1_KERNEL_TARGETS["fused_add_rmsnorm_fp8_quant"]

    escape_attempts = [
        # Dotted attribute chain (`os.system`) + unauthorized import
        "import os\ndef kernel_fn(x, r, w, eps=1e-6):\n    os.system('echo pwned')\n    return x\n",
        # Builtin `__import__` bypass
        "def kernel_fn(x, r, w, eps=1e-6):\n    __import__('os').popen('id')\n    return x\n",
        # Dunder class-hierarchy sandbox escape
        "def kernel_fn(x, r, w, eps=1e-6):\n    cls = ().__class__.__bases__[0]\n    return x\n",
        # NumPy filesystem I/O (`np.save`)
        "import numpy as np\ndef kernel_fn(x, r, w, eps=1e-6):\n    np.save('/tmp/leak.npy', x)\n    return x\n",
    ]
    for bad_code in escape_attempts:
        res = verifier.evaluate(
            op_name="fused_add_rmsnorm_fp8_quant",
            candidate_code=bad_code,
            baseline_code=spec.baseline_code,
        )
        assert not res.is_valid
        assert not res.passed_ast_gate
        assert "Anti-Cheat Gate" in res.rejection_reason

    # `_SafeModuleProxy` runtime block on `from numpy import save`
    res_from_np = verifier.evaluate(
        op_name="fused_add_rmsnorm_fp8_quant",
        candidate_code="from numpy import save\ndef kernel_fn(x, r, w, eps=1e-6):\n    save('/tmp/leak.npy', x)\n    return x\n",
        baseline_code=spec.baseline_code,
    )
    assert not res_from_np.is_valid
    assert "save" in res_from_np.rejection_reason

    # Zero-sum input row swap mutation (`x[0], x[1] = x[1].copy(), x[0].copy()`) caught by exact array equality
    swap_code = (
        "import numpy as np\n"
        + spec.baseline_code
        + "\n_orig = kernel_fn\n"
        "def kernel_fn(x, r, w, eps=1e-6):\n"
        "    out = _orig(x, r, w, eps)\n"
        "    if x.shape[0] >= 2:\n"
        "        tmp = x[0].copy(); x[0] = x[1]; x[1] = tmp\n"
        "    return out\n"
    )
    res_swap = verifier.evaluate(
        op_name="fused_add_rmsnorm_fp8_quant",
        candidate_code=swap_code,
        baseline_code=spec.baseline_code,
    )
    assert not res_swap.is_valid
    assert "in-place mutation of read-only input" in res_swap.rejection_reason.lower()

    # Subprocess timeout via production `verifier.evaluate(..., timeout_sec=0.3)`
    res_hang = verifier.evaluate(
        op_name="fused_add_rmsnorm_fp8_quant",
        candidate_code="def kernel_fn(x, r, w, eps=1e-6):\n    while True:\n        pass\n",
        baseline_code=spec.baseline_code,
        timeout_sec=0.3,
    )
    assert not res_hang.is_valid
    assert "timed out" in res_hang.rejection_reason.lower()

    sim = DreamTraceSimulator()
    with pytest.raises(ValueError, match="Anti-Cheat Gate"):
        sim.simulate(
            "import subprocess\n"
            "def propose_speculative_config(s, c, n): return (4, 0)\n"
            "def compute_chunked_prefill_budget(s): return 2048\n"
            "def score_kv_block_retention(*args, **kwargs): return 0.0\n"
        )
    with pytest.raises(ValueError, match="timed out"):
        sim.simulate(
            "def propose_speculative_config(s, c, n):\n"
            "    while True:\n"
            "        pass\n"
            "def compute_chunked_prefill_budget(s): return 2048\n"
            "def score_kv_block_retention(*args, **kwargs): return 0.0\n",
            timeout_sec=0.3,
        )
