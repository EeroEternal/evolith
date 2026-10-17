from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineStepState:
    """Snapshot of the single-node inference engine state at a scheduling boundary."""

    running_decode_reqs: int
    waiting_prefill_tokens: int
    free_gpu_kv_blocks: int
    total_gpu_kv_blocks: int
    ema_draft_accept_len: float
    ema_tpot_ms: float
    slo_tpot_target_ms: float = 25.0
    slo_ttft_target_ms: float = 800.0


# ============================================================================
# Baseline (Workload-Blind) L2 Scheduler & Speculative Policy Code
# ============================================================================

BASELINE_L2_POLICY_CODE = '''from evolith.planes.l2_policies import EngineStepState

def propose_speculative_config(
    state: EngineStepState,
    req_context_len: int,
    prefix_ngram_match_len: int,
) -> tuple[int, int]:
    """Static baseline: always proposes k=7 draft tokens, ignores batch load and context lookup."""
    return (7, 0)

def compute_chunked_prefill_budget(state: EngineStepState) -> int:
    """Static baseline: fixed 2048 prefill token chunk regardless of decode TPOT pressure."""
    return 2048

def score_kv_block_retention(
    idle_seconds: float,
    block_token_depth: int,
    pre_rope_importance: float,
    is_waiting_for_tool: bool,
    predicted_tool_return_sec: float,
    recompute_cost_ms: float,
) -> float:
    """Naive LRU baseline: evicts purely by idle_seconds, ignoring agent tool calls and geometric score."""
    return -idle_seconds
'''


# ============================================================================
# Stepping-Stone & ParetoMutations for L2 Policy Evolution
# ============================================================================

L2_POLICY_MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "(spec=adaptive,kv=lru)",
        "Make speculative draft length batch-aware and enable 16-token Context Lookup when prefix N-gram matches",
        "Add batch-size phase transition gating + 12-token Context Lookup fill",
        '''from evolith.planes.l2_policies import EngineStepState

def propose_speculative_config(
    state: EngineStepState,
    req_context_len: int,
    prefix_ngram_match_len: int,
) -> tuple[int, int]:
    """Workload-aware speculative drafting + Context Lookup augmentation."""
    if state.running_decode_reqs >= 16 or state.ema_tpot_ms > state.slo_tpot_target_ms * 0.92:
        return (0, 0)
    if prefix_ngram_match_len >= 3 and state.running_decode_reqs <= 8:
        return (4, 12)
    if state.running_decode_reqs <= 4:
        return (10 if state.ema_draft_accept_len >= 4.0 else 6, 2)
    return (4, 0)

def compute_chunked_prefill_budget(state: EngineStepState) -> int:
    return 2048

def score_kv_block_retention(
    idle_seconds: float,
    block_token_depth: int,
    pre_rope_importance: float,
    is_waiting_for_tool: bool,
    predicted_tool_return_sec: float,
    recompute_cost_ms: float,
) -> float:
    return -idle_seconds
''',
    ),
    (
        "(spec=static,kv=continuum_ttl)",
        "Introduce Continuum-style Tool-Pause TTL KV Pinning and TriAttention Pre-RoPE geometric retention",
        "Pin KV blocks during short agent tool calls when recompute_cost_ms is high",
        '''from evolith.planes.l2_policies import EngineStepState

def propose_speculative_config(
    state: EngineStepState,
    req_context_len: int,
    prefix_ngram_match_len: int,
) -> tuple[int, int]:
    return (7, 0)

def compute_chunked_prefill_budget(state: EngineStepState) -> int:
    tpot_headroom = state.slo_tpot_target_ms - state.ema_tpot_ms
    if tpot_headroom <= 2.5:
        return 384
    if state.running_decode_reqs == 0:
        return 8192
    return int(max(512, min(4096, 1536 * (tpot_headroom / 8.0))))

def score_kv_block_retention(
    idle_seconds: float,
    block_token_depth: int,
    pre_rope_importance: float,
    is_waiting_for_tool: bool,
    predicted_tool_return_sec: float,
    recompute_cost_ms: float,
) -> float:
    """Continuum dynamic TTL pinning + TriAttention Pre-RoPE geometric importance."""
    if is_waiting_for_tool:
        remaining = max(0.1, predicted_tool_return_sec - idle_seconds)
        if remaining <= 12.0 and recompute_cost_ms >= 35.0:
            return 2000.0 + (recompute_cost_ms / remaining)
    return (pre_rope_importance * 15.0) / (1.0 + 0.4 * idle_seconds)
''',
    ),
    (
        "(spec=adaptive_lookup,kv=continuum_tri_coopt)",
        "GEPA Pareto Crossover: Combine batch-aware DFlash2+ContextLookup with SLO-aware Chunked Prefill and Continuum TTL + TriAttention KV scoring",
        "Full co-optimization of speculative lookup, dynamic chunked prefill, and tool-pause KV pinning",
        '''from evolith.planes.l2_policies import EngineStepState

def propose_speculative_config(
    state: EngineStepState,
    req_context_len: int,
    prefix_ngram_match_len: int,
) -> tuple[int, int]:
    """Co-optimized DFlash2 + 16-token Context Lookup with TPOT SLO protection."""
    kv_ratio = state.free_gpu_kv_blocks / max(1, state.total_gpu_kv_blocks)
    if state.running_decode_reqs >= 16 or state.ema_tpot_ms > state.slo_tpot_target_ms * 0.90:
        return (0, 0)
    if prefix_ngram_match_len >= 3 and state.running_decode_reqs <= 10:
        return (4, 12)
    if state.running_decode_reqs <= 4 and kv_ratio > 0.15:
        return (10 if state.ema_draft_accept_len >= 4.2 else 6, 4)
    return (4, 2)

def compute_chunked_prefill_budget(state: EngineStepState) -> int:
    """SLO-aware elastic chunked prefill budget."""
    tpot_headroom = state.slo_tpot_target_ms - state.ema_tpot_ms
    if tpot_headroom <= 2.5:
        return 384
    if state.running_decode_reqs == 0:
        return 8192
    scale = min(3.5, max(0.4, tpot_headroom / 6.0))
    return int(max(512, min(4096, (1600 * scale) / (1.0 + 0.12 * state.running_decode_reqs))))

def score_kv_block_retention(
    idle_seconds: float,
    block_token_depth: int,
    pre_rope_importance: float,
    is_waiting_for_tool: bool,
    predicted_tool_return_sec: float,
    recompute_cost_ms: float,
) -> float:
    """Continuum TTL Pinning + TriAttention Pre-RoPE Geometric Compaction score."""
    if is_waiting_for_tool:
        remaining = max(0.1, predicted_tool_return_sec - idle_seconds)
        if remaining <= 12.0 and recompute_cost_ms >= 30.0:
            return 2500.0 + (recompute_cost_ms / remaining)
    return (pre_rope_importance * 18.0 + 0.002 * block_token_depth) / (1.0 + 0.35 * idle_seconds)
''',
    ),
]
