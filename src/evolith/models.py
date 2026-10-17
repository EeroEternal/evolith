from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EvolutionPlane(str, Enum):
    L1_KERNEL = "l1_kernel"
    L2_SCHEDULER = "l2_scheduler"
    L3_ROUTER = "l3_router"


class BottleneckClass(str, Enum):
    MEMORY_BANDWIDTH_BOUND = "MEMORY_BANDWIDTH_BOUND"
    COMPUTE_BOUND = "COMPUTE_BOUND"
    LAUNCH_OVERHEAD_BOUND = "LAUNCH_OVERHEAD_BOUND"
    REGISTER_SPILL_BOUND = "REGISTER_SPILL_BOUND"
    BALANCED = "BALANCED"


class HardwareProfile(BaseModel):
    """Hardware counter telemetry and Roofline classification for a kernel execution."""

    shape_bucket: str
    latency_us: float
    baseline_latency_us: float
    speedup: float
    bytes_transferred: int
    flops_executed: int
    arithmetic_intensity: float
    dram_bw_utilization_pct: float
    sm_compute_utilization_pct: float
    bank_conflict_way: float = 1.0
    registers_per_thread: int = 32
    spilling_bytes: int = 0
    primary_stall_reason: str = "stall_long_sb (Global Memory Load Wait)"
    bottleneck_class: BottleneckClass = BottleneckClass.MEMORY_BANDWIDTH_BOUND
    actionable_hint: str = ""

    def to_reflection_card(self) -> str:
        return (
            f"[Hardware Profiler Card | Shape={self.shape_bucket}]\n"
            f"  - Latency: {self.latency_us:.2f} us (Baseline: {self.baseline_latency_us:.2f} us -> Speedup: {self.speedup:.2f}x)\n"
            f"  - Arithmetic Intensity: {self.arithmetic_intensity:.2f} FLOPs/Byte\n"
            f"  - DRAM BW Util: {self.dram_bw_utilization_pct:.1f}% | Compute Util: {self.sm_compute_utilization_pct:.1f}%\n"
            f"  - Registers/Thread: {self.registers_per_thread} (Spilling: {self.spilling_bytes} B)\n"
            f"  - Bottleneck: [{self.bottleneck_class.value}] ({self.primary_stall_reason})\n"
            f"  - Evolution Hint: {self.actionable_hint}"
        )


class KernelEvalResult(BaseModel):
    """Output of the 4-stage Hardened Micro-Kernel Verifier."""

    op_name: str
    passed_ast_gate: bool
    passed_correctness_gate: bool
    passed_anti_cheat_gate: bool
    latency_us: float = 0.0
    baseline_latency_us: float = 0.0
    speedup: float = 0.0
    max_abs_diff: float = 0.0
    cosine_sim: float = 0.0
    loc: int = 0
    rejection_reason: str = ""
    profiles_by_bucket: dict[str, HardwareProfile] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return (
            self.passed_ast_gate
            and self.passed_correctness_gate
            and self.passed_anti_cheat_gate
        )

    @property
    def reflection_card(self) -> str:
        if not self.is_valid:
            return (
                f"[Hardware Reflection Card | Op={self.op_name} | STATUS=REJECTED]\n"
                f"  - Reason: {self.rejection_reason}\n"
                f"  - Gates: AST={self.passed_ast_gate}, Correctness={self.passed_correctness_gate}, "
                f"AntiCheat={self.passed_anti_cheat_gate}"
            )
        cards = [p.to_reflection_card() for p in self.profiles_by_bucket.values()]
        return (
            f"[Hardware Reflection Card | Op={self.op_name} | STATUS=VALID | "
            f"GeoMean Speedup={self.speedup:.2f}x | MaxAbsDiff={self.max_abs_diff:.2e}]\n"
            + "\n".join(cards)
        )


class WorstWindowEvent(BaseModel):
    """Snapshot of the highest SLO-violation window during Dream Trace Simulation."""

    timestamp_sec: float
    running_decode_reqs: int
    waiting_prefill_tokens: int
    free_kv_blocks: int
    active_tool_paused_Pinned_blocks: int
    chosen_draft_k: int
    chosen_lookup_tokens: int
    chosen_prefill_budget: int
    step_tpot_ms: float
    queued_ttft_ms: float
    diagnosis: str


class SimMetrics(BaseModel):
    """Output of the L2/L3 Trace-Driven Discrete-Event Dream Simulator."""

    total_requests: int
    duration_sec: float
    goodput_qps: float
    raw_throughput_tok_s: float
    p50_ttft_ms: float
    p99_ttft_ms: float
    p50_tpot_ms: float
    p99_tpot_ms: float
    kv_cache_hit_rate: float
    tool_pause_pin_hit_rate: float
    mean_accepted_draft_tokens: float
    preemption_recompute_count: int
    slo_violation_rate: float
    composite_fitness: float
    worst_windows: list[WorstWindowEvent] = Field(default_factory=list)

    def pareto_vector(self) -> tuple[float, float, float]:
        """Returns (goodput_qps [maximize], -p99_ttft_ms [maximize], -p99_tpot_ms [maximize])."""
        return (self.goodput_qps, -self.p99_ttft_ms, -self.p99_tpot_ms)

    def to_reflection_card(self) -> str:
        ww = self.worst_windows[0].diagnosis if self.worst_windows else "None"
        return (
            f"• Goodput (SLO-compliant): {self.goodput_qps:.2f} QPS (Raw: {self.raw_throughput_tok_s:.0f} tok/s)\n"
            f"• Latency: P50/P99 TTFT = {self.p50_ttft_ms:.0f}ms / {self.p99_ttft_ms:.0f}ms | "
            f"P50/P99 TPOT = {self.p50_tpot_ms:.1f}ms / {self.p99_tpot_ms:.1f}ms\n"
            f"• KV Cache Hit Rate: {self.kv_cache_hit_rate*100:.1f}% | "
            f"Tool-Pause Pin Hit Rate: {self.tool_pause_pin_hit_rate*100:.1f}%\n"
            f"• Mean Accepted Draft Tokens: {self.mean_accepted_draft_tokens:.2f} | "
            f"Preemptions: {self.preemption_recompute_count}\n"
            f"• Worst-Window Counterfactual Diagnosis: {ww}"
        )


class LineageNode(BaseModel):
    """
    A node in the Darwin Gödel Machine (DGM) / MAP-Elites Stepping-Stones Archive.
    Includes the AIDE^2 16x compressed 4-tuple memory fields.
    """

    node_id: str
    parent_id: str | None = None
    generation: int
    plane: EvolutionPlane
    target_name: str
    niche_key: str  # e.g. "(fusion_level=3, tile_strategy=vectorized)"
    source_code: str
    loc: int
    score: float
    speedup: float = 1.0
    pareto_goodput: float = 0.0
    pareto_p99_ttft_ms: float = 0.0
    pareto_p99_tpot_ms: float = 0.0
    accepted_by_gate: bool = True
    is_stepping_stone: bool = False
    # AIDE^2 16x Prompt-Compression 4-tuple fields:
    hypothesis: str
    patch_summary: str
    metric_delta: str
    failure_root_cause: str = "None"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_compressed_4tuple(self) -> str:
        """Formats the node as a single-line AIDE^2 16x compressed memory entry."""
        status = "MERGED" if self.accepted_by_gate else ("STEPPING_STONE" if self.is_stepping_stone else "REJECTED")
        return (
            f"[{self.node_id} gen={self.generation} {status} niche={self.niche_key}] "
            f"Hyp: {self.hypothesis} | Patch: {self.patch_summary} | "
            f"Delta: {self.metric_delta} | RootCause: {self.failure_root_cause}"
        )
