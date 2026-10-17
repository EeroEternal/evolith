from __future__ import annotations

import numpy as np

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
