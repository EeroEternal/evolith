from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from evolith.models import SimMetrics, WorstWindowEvent
from evolith.planes.l2_policies import EngineStepState


@dataclass
class AgentRequestTrace:
    req_id: int
    arrival_sec: float
    context_tokens: int
    output_tokens: int
    shared_prefix_blocks: int
    prefix_ngram_match_len: int
    is_tool_call_turn: bool
    tool_pause_sec: float
    pre_rope_importance: float


class DreamTraceSimulator:
    """
    L2/L3 Trace-Driven Discrete-Event Dream Simulator (inspired by Google Dream-RSI).
    Replays multi-turn Coding Agent & RAG traces thousands of times faster than real-time
    to evaluate candidate scheduler, speculative decoding, and KV retention policies.
    """

    def __init__(self, num_requests: int = 320, seed: int = 2026) -> None:
        self.num_requests = num_requests
        self.seed = seed
        self.traces = self._generate_realistic_agent_traces(num_requests, seed)

    @staticmethod
    def _generate_realistic_agent_traces(
        num_requests: int, seed: int
    ) -> list[AgentRequestTrace]:
        rng = np.random.default_rng(seed)
        traces: list[AgentRequestTrace] = []
        t = 0.0
        for i in range(num_requests):
            # Bursty Poisson arrival modeling multi-agent coding swarms
            is_burst = (i % 40) < 12
            inter_arrival = rng.exponential(0.035 if is_burst else 0.14)
            t += float(inter_arrival)
            ctx = int(rng.choice([2048, 4096, 8192, 16384, 24576], p=[0.2, 0.3, 0.25, 0.15, 0.1]))
            out = int(rng.integers(96, 384))
            shared_blocks = int((ctx // 16) * rng.uniform(0.55, 0.92))
            ngram_match = int(rng.choice([0, 2, 4, 6], p=[0.25, 0.20, 0.35, 0.20]))
            is_tool = bool(rng.random() < 0.65)
            pause_sec = float(rng.uniform(2.0, 9.5)) if is_tool else 0.0
            importance = float(rng.uniform(0.3, 1.0))
            traces.append(
                AgentRequestTrace(
                    req_id=i,
                    arrival_sec=t,
                    context_tokens=ctx,
                    output_tokens=out,
                    shared_prefix_blocks=shared_blocks,
                    prefix_ngram_match_len=ngram_match,
                    is_tool_call_turn=is_tool,
                    tool_pause_sec=pause_sec,
                    pre_rope_importance=importance,
                )
            )
        return traces

    @staticmethod
    def _compile_policy_module(source_code: str) -> dict[str, Callable]:
        namespace: dict[str, object] = {}
        exec(compile(source_code, "<evolved_l2_policy>", "exec"), namespace)
        required = (
            "propose_speculative_config",
            "compute_chunked_prefill_budget",
            "score_kv_block_retention",
        )
        funcs: dict[str, Callable] = {}
        for name in required:
            fn = namespace.get(name)
            if not callable(fn):
                raise ValueError(f"Policy code missing callable '{name}'.")
            funcs[name] = fn
        return funcs

    def simulate(self, policy_source_code: str) -> SimMetrics:
        funcs = self._compile_policy_module(policy_source_code)
        propose_spec = funcs["propose_speculative_config"]
        chunk_budget_fn = funcs["compute_chunked_prefill_budget"]
        kv_score_fn = funcs["score_kv_block_retention"]

        total_kv_blocks = 8192
        ttft_list: list[float] = []
        tpot_list: list[float] = []
        accepted_drafts: list[float] = []
        kv_hits = 0
        tool_pin_hits = 0
        tool_turns = 0
        preemptions = 0
        slo_violations = 0
        worst_windows: list[WorstWindowEvent] = []

        ema_accept = 4.6
        ema_tpot = 14.5

        for idx, tr in enumerate(self.traces):
            # Simulate concurrent wave state
            is_wave = (idx % 40) < 12
            running_decodes = 18 if is_wave else 4
            waiting_prefill = 6144 if is_wave else 1024
            free_blocks = 950 if is_wave else 3800

            state = EngineStepState(
                running_decode_reqs=running_decodes,
                waiting_prefill_tokens=waiting_prefill,
                free_gpu_kv_blocks=free_blocks,
                total_gpu_kv_blocks=total_kv_blocks,
                ema_draft_accept_len=ema_accept,
                ema_tpot_ms=ema_tpot,
            )

            # 1. Evaluate KV Cache / Continuum Tool-Pause Retention Policy
            recompute_cost_ms = (tr.context_tokens / 1024.0) * 18.5
            score_tool = float(
                kv_score_fn(
                    idle_seconds=tr.tool_pause_sec * 0.6,
                    block_token_depth=tr.context_tokens,
                    pre_rope_importance=tr.pre_rope_importance,
                    is_waiting_for_tool=tr.is_tool_call_turn,
                    predicted_tool_return_sec=tr.tool_pause_sec,
                    recompute_cost_ms=recompute_cost_ms,
                )
            )
            score_stale_chat = float(
                kv_score_fn(
                    idle_seconds=4.5,
                    block_token_depth=1024,
                    pre_rope_importance=0.25,
                    is_waiting_for_tool=False,
                    predicted_tool_return_sec=0.0,
                    recompute_cost_ms=18.0,
                )
            )

            pinned_survived = score_tool > score_stale_chat and score_tool > 100.0
            if tr.is_tool_call_turn:
                tool_turns += 1
                if pinned_survived:
                    tool_pin_hits += 1
                    kv_hits += 1
                else:
                    preemptions += 1
            else:
                if tr.pre_rope_importance > 0.45 and score_tool > score_stale_chat:
                    kv_hits += 1

            # 2. Evaluate Chunked Prefill Budget Policy
            prefill_budget = int(chunk_budget_fn(state))
            prefill_budget = max(256, min(8192, prefill_budget))
            uncached_tokens = (
                int(tr.context_tokens * 0.12)
                if (pinned_survived or tr.shared_prefix_blocks > 80)
                else tr.context_tokens
            )
            if tr.is_tool_call_turn and not pinned_survived:
                uncached_tokens = tr.context_tokens  # Evicted during tool pause!

            num_chunks = max(1.0, uncached_tokens / float(prefill_budget))
            ttft_ms = num_chunks * (16.0 + 0.014 * prefill_budget) + (
                0.0 if pinned_survived else 55.0
            )

            # 3. Evaluate Speculative Decoding & Context Lookup Policy
            draft_k, lookup_fill = propose_spec(
                state, tr.context_tokens, tr.prefix_ngram_match_len
            )
            draft_k = max(0, min(16, int(draft_k)))
            lookup_fill = max(0, min(16, int(lookup_fill)))
            total_verify_block = draft_k + lookup_fill

            if running_decodes >= 16:
                # Compute-bound phase transition! Speculative verification hurts TPOT at high batch
                verify_penalty_ms = 1.35 * draft_k + 0.35 * lookup_fill
                effective_accept = 1.0 + 0.25 * min(draft_k, 4)
            else:
                # Memory-bandwidth bound: cheap draft + zero-cost context lookup shine!
                verify_penalty_ms = 0.22 * draft_k + 0.04 * lookup_fill
                draft_accepted = min(draft_k, 5.2) * 0.78
                lookup_accepted = (
                    lookup_fill * 0.88 if tr.prefix_ngram_match_len >= 3 else lookup_fill * 0.15
                )
                effective_accept = 1.0 + draft_accepted + lookup_accepted

            # Chunked prefill interference on decode TPOT
            prefill_interference_ms = max(0.0, (prefill_budget - 768) * 0.0045) if running_decodes >= 12 else 0.0
            base_step_ms = 14.5 + (4.2 if running_decodes >= 16 else 0.0)
            step_tpot_ms = (base_step_ms + verify_penalty_ms + prefill_interference_ms) / max(
                1.0, effective_accept * 0.38
            )

            ema_accept = 0.85 * ema_accept + 0.15 * effective_accept
            ema_tpot = 0.85 * ema_tpot + 0.15 * step_tpot_ms

            ttft_list.append(ttft_ms)
            tpot_list.append(step_tpot_ms)
            accepted_drafts.append(effective_accept)

            violated = ttft_ms > state.slo_ttft_target_ms or step_tpot_ms > state.slo_tpot_target_ms
            if violated:
                slo_violations += 1
                if len(worst_windows) < 3:
                    diag = []
                    if running_decodes >= 16 and total_verify_block > 4:
                        diag.append(
                            f"Batch={running_decodes} is Compute-Bound but draft_k={draft_k} caused +{verify_penalty_ms:.1f}ms verify stall."
                        )
                    if tr.is_tool_call_turn and not pinned_survived:
                        diag.append(
                            f"Agent KV evicted during {tr.tool_pause_sec:.1f}s tool pause, forcing {uncached_tokens}-token re-prefill."
                        )
                    if prefill_interference_ms > 3.0:
                        diag.append(
                            f"Chunked prefill budget={prefill_budget} too large during decode wave, adding +{prefill_interference_ms:.1f}ms ITL."
                        )
                    worst_windows.append(
                        WorstWindowEvent(
                            timestamp_sec=round(tr.arrival_sec, 2),
                            running_decode_reqs=running_decodes,
                            waiting_prefill_tokens=waiting_prefill,
                            free_kv_blocks=free_blocks,
                            active_tool_paused_Pinned_blocks=1 if pinned_survived else 0,
                            chosen_draft_k=draft_k,
                            chosen_lookup_tokens=lookup_fill,
                            chosen_prefill_budget=prefill_budget,
                            step_tpot_ms=round(step_tpot_ms, 2),
                            queued_ttft_ms=round(ttft_ms, 2),
                            diagnosis=" ".join(diag) or "High concurrent load spike.",
                        )
                    )

        duration_sec = max(1.0, self.traces[-1].arrival_sec)
        p50_ttft = float(np.percentile(ttft_list, 50))
        p99_ttft = float(np.percentile(ttft_list, 99))
        p50_tpot = float(np.percentile(tpot_list, 50))
        p99_tpot = float(np.percentile(tpot_list, 99))
        slo_viol_rate = slo_violations / float(self.num_requests)
        mean_accept = float(np.mean(accepted_drafts))

        # Goodput = effective QPS meeting both TTFT and TPOT SLOs scaled by speedup
        speed_factor = (22.0 / max(4.0, p50_tpot)) * (320.0 / max(60.0, p50_ttft)) ** 0.35
        goodput_qps = (self.num_requests / duration_sec) * (1.0 - 0.65 * slo_viol_rate) * speed_factor
        raw_tok_s = goodput_qps * 220.0

        fitness = (
            goodput_qps * 12.0
            - 0.04 * p99_ttft
            - 1.5 * p99_tpot
            - 0.25 * preemptions
        )

        return SimMetrics(
            total_requests=self.num_requests,
            duration_sec=round(duration_sec, 2),
            goodput_qps=round(goodput_qps, 2),
            raw_throughput_tok_s=round(raw_tok_s, 1),
            p50_ttft_ms=round(p50_ttft, 1),
            p99_ttft_ms=round(p99_ttft, 1),
            p50_tpot_ms=round(p50_tpot, 2),
            p99_tpot_ms=round(p99_tpot, 2),
            kv_cache_hit_rate=round(kv_hits / float(self.num_requests), 3),
            tool_pause_pin_hit_rate=round(tool_pin_hits / float(max(1, tool_turns)), 3),
            mean_accepted_draft_tokens=round(mean_accept, 2),
            preemption_recompute_count=preemptions,
            slo_violation_rate=round(slo_viol_rate, 3),
            composite_fitness=round(fitness, 2),
            worst_windows=worst_windows,
        )
