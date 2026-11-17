"""Agent-lifecycle prefix-cache accounting and replay conformance checks.

Implements two core mechanisms from Evolving Infra (arXiv:2611.12304):
1. Branch-aware prefix reference counting and compaction detection (Algorithms E.1
   & E.2): tracks active agent branches, increments branch version on context
   compaction/rewrite when ``pi_old`` is no longer a prefix of ``pi_new``, and
   evicts blocks in lexicographic order ``(1[R_b > 0], last_access_tick)``.
2. Replay run validity gates (``benchmarks/replay/checks.py``): input SHA-256
   attestation, pre-run quiet window, steady-state zero failures, monotonic
   per-worker request counters, and ``actual_output`` context mode under
   speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Sequence

BlockKey = tuple[int, ...]
BlockPath = tuple[BlockKey, ...]


def tokenize_into_block_path(tokens: Sequence[int], block_tokens: int) -> BlockPath:
    """Convert a token sequence into a cumulative prefix-hashed path of full blocks."""
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    full_blocks = len(tokens) // block_tokens
    path: list[BlockKey] = []
    acc: list[int] = []
    for idx in range(full_blocks):
        chunk = tuple(int(t) for t in tokens[idx * block_tokens : (idx + 1) * block_tokens])
        # Key each block by cumulative chain hash + chunk tokens so distinct prefixes never alias
        parent_tag = path[-1][0] if path else 0
        h = hash((parent_tag, chunk))
        acc_key = (h, *chunk)
        path.append(acc_key)
        acc.extend(chunk)
    return tuple(path)


def _is_prefix(prefix: BlockPath, full: BlockPath) -> bool:
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


@dataclass
class _BranchState:
    version: int = 1
    path: BlockPath = ()
    counted_blocks: set[BlockKey] = field(default_factory=set)


@dataclass
class _CachedBlock:
    key: BlockKey
    ref_count: int = 0
    last_access_tick: int = 0


@dataclass(frozen=True)
class CacheComparisonReport:
    total_prompt_blocks: int
    ideal_hit_blocks: int
    lru_hit_blocks: int
    branch_aware_hit_blocks: int
    compactions_detected: int

    @property
    def ideal_hit_rate(self) -> float:
        return self.ideal_hit_blocks / max(self.total_prompt_blocks, 1)

    @property
    def lru_hit_rate(self) -> float:
        return self.lru_hit_blocks / max(self.total_prompt_blocks, 1)

    @property
    def branch_aware_hit_rate(self) -> float:
        return self.branch_aware_hit_blocks / max(self.total_prompt_blocks, 1)

    @property
    def lru_lost_kv_reuse(self) -> float:
        if self.ideal_hit_blocks == 0:
            return 0.0
        return 1.0 - (self.lru_hit_blocks / self.ideal_hit_blocks)

    @property
    def branch_aware_lost_kv_reuse(self) -> float:
        if self.ideal_hit_blocks == 0:
            return 0.0
        return 1.0 - (self.branch_aware_hit_blocks / self.ideal_hit_blocks)


class BranchAwareBlockCache:
    """Host/GPU block cache implementing Evolving Infra Algorithms E.1 & E.2."""

    def __init__(
        self,
        capacity_blocks: int,
        *,
        block_tokens: int = 64,
        branch_aware_eviction: bool = True,
    ) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        self.capacity_blocks = capacity_blocks
        self.block_tokens = block_tokens
        self.branch_aware_eviction = branch_aware_eviction
        self._tick: int = 0
        self._blocks: dict[BlockKey, _CachedBlock] = {}
        self._branches: dict[str, _BranchState] = {}
        self.compactions_detected: int = 0

    def active_branches(self) -> int:
        return len(self._branches)

    def cached_blocks(self) -> int:
        return len(self._blocks)

    def ref_count(self, key: BlockKey) -> int:
        blk = self._blocks.get(key)
        return blk.ref_count if blk is not None else 0

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def _set_branch_counts(self, state: _BranchState, target: set[BlockKey]) -> None:
        to_add = target - state.counted_blocks
        to_remove = state.counted_blocks - target
        for key in to_add:
            blk = self._blocks.get(key)
            if blk is not None:
                blk.ref_count += 1
        for key in to_remove:
            blk = self._blocks.get(key)
            if blk is not None and blk.ref_count > 0:
                blk.ref_count -= 1
        state.counted_blocks = set(target) & set(self._blocks.keys())

    def _evict_if_needed(self, needed_slots: int, protected: set[BlockKey]) -> None:
        while len(self._blocks) + needed_slots > self.capacity_blocks:
            candidates = [b for k, b in self._blocks.items() if k not in protected]
            if not candidates:
                candidates = list(self._blocks.values())
            if not candidates:
                break
            if self.branch_aware_eviction:
                victim = min(
                    candidates,
                    key=lambda b: (1 if b.ref_count > 0 else 0, b.last_access_tick),
                )
            else:
                victim = min(candidates, key=lambda b: b.last_access_tick)
            del self._blocks[victim.key]
            for st in self._branches.values():
                st.counted_blocks.discard(victim.key)

    def on_request(self, branch_id: str, prompt_tokens: Sequence[int]) -> tuple[int, int]:
        """Process one request of ``branch_id``; returns ``(leading_hit_blocks, total_blocks)``."""
        path = tokenize_into_block_path(prompt_tokens, self.block_tokens)
        total = len(path)

        # Algorithm E.1: detect new/forked branch vs append vs compaction rewrite
        if branch_id not in self._branches:
            state = _BranchState(version=1, path=())
            self._branches[branch_id] = state
        else:
            state = self._branches[branch_id]
            if not _is_prefix(state.path, path):
                state.version += 1
                self.compactions_detected += 1
                # Algorithm E.2 lines 6-8: release dropped history immediately
                self._set_branch_counts(state, state.counted_blocks & set(path))
        state.path = path

        # Count leading prefix hits
        hits = 0
        for key in path:
            blk = self._blocks.get(key)
            if blk is None:
                break
            blk.last_access_tick = self._next_tick()
            hits += 1

        missing = [k for k in path if k not in self._blocks]
        path_set = set(path)
        if len(path_set) > self.capacity_blocks:
            keep_tail = list(path[-self.capacity_blocks :])
            keep_set = set(keep_tail)
            self._blocks.clear()
            for st in self._branches.values():
                st.counted_blocks.clear()
            for key in keep_tail:
                self._blocks[key] = _CachedBlock(key=key, ref_count=0, last_access_tick=self._next_tick())
            self._set_branch_counts(state, keep_set)
            return hits, total

        self._evict_if_needed(len(missing), protected=path_set)
        for key in missing:
            self._blocks[key] = _CachedBlock(
                key=key, ref_count=0, last_access_tick=self._next_tick()
            )
        self._set_branch_counts(state, path_set)
        return hits, total

    def end_branch(self, branch_id: str) -> None:
        """Algorithm E.2 line 13: release all counts when an agent/sub-agent branch finishes."""
        state = self._branches.pop(branch_id, None)
        if state is not None:
            self._set_branch_counts(state, set())


@dataclass(frozen=True)
class AgentTraceEvent:
    kind: str  # "request" or "end_branch"
    branch_id: str
    prompt_tokens: tuple[int, ...] = ()


def compare_eviction_policies(
    events: Sequence[AgentTraceEvent],
    *,
    capacity_blocks: int,
    block_tokens: int = 64,
) -> CacheComparisonReport:
    """Compare pure LRU against branch-aware ``(1[R_b > 0], T_b)`` eviction and an unlimited oracle."""
    oracle: set[BlockKey] = set()
    lru = BranchAwareBlockCache(
        capacity_blocks, block_tokens=block_tokens, branch_aware_eviction=False
    )
    aware = BranchAwareBlockCache(
        capacity_blocks, block_tokens=block_tokens, branch_aware_eviction=True
    )

    total_blocks = 0
    ideal_hits = 0
    lru_hits = 0
    aware_hits = 0

    for ev in events:
        if ev.kind == "end_branch":
            lru.end_branch(ev.branch_id)
            aware.end_branch(ev.branch_id)
            continue
        if ev.kind != "request":
            raise ValueError(f"Unknown event kind: {ev.kind}")
        path = tokenize_into_block_path(ev.prompt_tokens, block_tokens)
        lead_oracle = 0
        for key in path:
            if key in oracle:
                lead_oracle += 1
            else:
                break
        oracle.update(path)
        h_lru, n_blk = lru.on_request(ev.branch_id, ev.prompt_tokens)
        h_aware, _ = aware.on_request(ev.branch_id, ev.prompt_tokens)
        total_blocks += n_blk
        ideal_hits += lead_oracle
        lru_hits += h_lru
        aware_hits += h_aware

    return CacheComparisonReport(
        total_prompt_blocks=total_blocks,
        ideal_hit_blocks=ideal_hits,
        lru_hit_blocks=lru_hits,
        branch_aware_hit_blocks=aware_hits,
        compactions_detected=aware.compactions_detected,
    )


@dataclass(frozen=True)
class ReplayConformanceResult:
    passed: bool
    reasons: tuple[str, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_agent_replay_run(
    *,
    workload_sha256: str,
    expected_workload_sha256: str,
    tokenizer_sha256: str,
    expected_tokenizer_sha256: str,
    pre_run_queue_depths: Sequence[int],
    pre_run_counter_deltas: Sequence[int],
    completed_requests: int,
    failed_requests: int,
    worker_request_counters: dict[str, Sequence[int]],
    speculative_enabled: bool = False,
    replay_context_mode: str = "actual_output",
) -> ReplayConformanceResult:
    """Enforce Evolving Infra's 4 run-validity gates + speculative context-mode check."""
    reasons: list[str] = []
    if workload_sha256 != expected_workload_sha256:
        reasons.append(
            f"Workload SHA-256 mismatch ({workload_sha256[:12]} != {expected_workload_sha256[:12]})"
        )
    if tokenizer_sha256 != expected_tokenizer_sha256:
        reasons.append(
            f"Tokenizer SHA-256 mismatch ({tokenizer_sha256[:12]} != {expected_tokenizer_sha256[:12]})"
        )
    if any(q != 0 for q in pre_run_queue_depths) or any(d != 0 for d in pre_run_counter_deltas):
        reasons.append("Pre-run quiet window violated: non-empty queue or moving request counter")
    if completed_requests <= 0:
        reasons.append("Steady-state window recorded zero completed requests")
    if failed_requests != 0:
        reasons.append(f"Steady-state window had {failed_requests} failed requests (must be 0)")
    for worker_id, series in sorted(worker_request_counters.items()):
        if len(series) < 2:
            reasons.append(f"Worker `{worker_id}` lacks >= 2 counter scrapes")
            continue
        for prev, curr in zip(series, series[1:]):
            if curr < prev:
                reasons.append(
                    f"Worker `{worker_id}` counter decreased ({prev} -> {curr}), indicating hidden worker restart"
                )
                break
    if speculative_enabled and replay_context_mode != "actual_output":
        reasons.append(
            f"Speculative decoding requires replay_context_mode='actual_output', got {replay_context_mode!r}"
        )
    return ReplayConformanceResult(passed=not reasons, reasons=tuple(reasons))
