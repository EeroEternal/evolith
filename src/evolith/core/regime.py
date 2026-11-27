"""Multi-Regime Bottleneck Ledger, Dispatch Route Audit & Piecewise Policy Synthesizer.

Implements three GPU-agnostic RSI laws derived from real hardware deployments:
1. `DispatchAudit`: detects silent fallback to untuned default kernels when runtime
   compute-unit counts (e.g. 156 SMs vs {170, 188}) or shape buckets miss dispatch tables.
2. `MultiRegimeLedger`: aggregates step profiles across heterogeneous regimes (low-batch
   decode, high-batch decode, long-context prefill) so target selection optimizes
   whole-workload value rather than overfitting a single shape (`AGENTS.md`).
3. `PiecewisePolicySynthesizer`: synthesizes zero-regression `(phase, seq_len, batch_tokens)`
   piecewise dispatch tables across competing backends with guardband protection.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from evolith.core.topology import HardwareTopologySnapshot, PrecisionTrack
from evolith.models import EvolutionPlane


class DispatchAuditIssue(BaseModel):
    op_name: str
    selected_tactic: str
    severity: str  # "fallback_route" | "compute_unit_mismatch" | "interconnect_mismatch"
    detail: str


class DispatchAuditReport(BaseModel):
    arch_tag: str
    compute_units: int
    total_audited_ops: int
    optimal_hits: int
    issues: tuple[DispatchAuditIssue, ...] = ()

    @property
    def hit_rate(self) -> float:
        if self.total_audited_ops <= 0:
            return 1.0
        return self.optimal_hits / self.total_audited_ops


def audit_operator_dispatch(
    topology: HardwareTopologySnapshot,
    observations: Sequence[dict[str, object]],
) -> DispatchAuditReport:
    """Audit runtime operator dispatch routes against probed `HardwareTopologySnapshot`."""
    issues: list[DispatchAuditIssue] = []
    optimal = 0
    cu = topology.compute_units_per_device
    all_p2p = topology.all_to_all_p2p_supported()

    for obs in observations:
        op_name = str(obs.get("op_name", "unknown"))
        tactic = str(obs.get("selected_tactic", "") or "unspecified")
        is_fallback = bool(obs.get("is_fallback_route", False))
        raw_tuned_cus = obs.get("tuned_compute_units", ())
        tuned_cus = (
            tuple(int(x) for x in raw_tuned_cus)
            if isinstance(raw_tuned_cus, (list, tuple, set, frozenset))
            else ()
        )
        requires_all_p2p = bool(obs.get("requires_all_to_all_p2p", False))

        op_has_issue = False
        if requires_all_p2p and not all_p2p:
            op_has_issue = True
            issues.append(
                DispatchAuditIssue(
                    op_name=op_name,
                    selected_tactic=tactic,
                    severity="interconnect_mismatch",
                    detail=(
                        f"`{op_name}` tactic `{tactic}` requires full-mesh P2P, "
                        f"but cluster has P2P components {topology.p2p_connected_components()}"
                    ),
                )
            )
        if is_fallback or tactic in ("default", "fallback", "orig_forward_cuda", "stock"):
            op_has_issue = True
            issues.append(
                DispatchAuditIssue(
                    op_name=op_name,
                    selected_tactic=tactic,
                    severity="fallback_route",
                    detail=f"`{op_name}` executed untuned/fallback route `{tactic}`",
                )
            )
        if tuned_cus and cu not in tuned_cus:
            op_has_issue = True
            issues.append(
                DispatchAuditIssue(
                    op_name=op_name,
                    selected_tactic=tactic,
                    severity="compute_unit_mismatch",
                    detail=(
                        f"`{op_name}` dispatch table keys on compute_units={tuned_cus}, "
                        f"missing actual device compute_units={cu} ({topology.device_name})"
                    ),
                )
            )
        if not op_has_issue:
            optimal += 1

    return DispatchAuditReport(
        arch_tag=topology.arch_tag,
        compute_units=cu,
        total_audited_ops=len(observations),
        optimal_hits=optimal,
        issues=tuple(issues),
    )


class WorkloadRegime(BaseModel):
    """A single operational regime in a multi-regime serving profile."""

    regime_id: str
    phase: str  # "decode" | "prefill" | "mixed"
    batch_tokens: int = Field(ge=1)
    seq_len: int = Field(default=1024, ge=1)
    weight: float = Field(default=1.0, gt=0.0)


class RegimeOperatorShare(BaseModel):
    regime_id: str
    op_name: str
    plane: EvolutionPlane = EvolutionPlane.L1_KERNEL
    step_us: float = Field(ge=0.0)
    total_regime_step_us: float = Field(gt=0.0)
    supported: bool = True
    niche_key: str = ""


class MultiRegimeTargetScore(BaseModel):
    op_name: str
    plane: EvolutionPlane
    weighted_share: float
    per_regime_shares: dict[str, float]
    per_regime_us: dict[str, float]
    dominant_regime_id: str
    supported: bool = True
    niche_key: str = ""


class MultiRegimeLedger:
    """Ranks optimization targets across multiple serving regimes without single-shape bias."""

    def __init__(
        self,
        regimes: Sequence[WorkloadRegime],
        entries: Sequence[RegimeOperatorShare],
    ) -> None:
        if not regimes:
            raise ValueError("MultiRegimeLedger requires at least one WorkloadRegime")
        self.regimes = {r.regime_id: r for r in regimes}
        self.entries = tuple(entries)

    def rank_targets(
        self,
        *,
        blocked_niches: set[str] | None = None,
    ) -> tuple[list[MultiRegimeTargetScore], list[MultiRegimeTargetScore]]:
        blocked = blocked_niches or set()
        total_weight = sum(r.weight for r in self.regimes.values())
        grouped: dict[str, list[RegimeOperatorShare]] = {}
        for e in self.entries:
            if e.regime_id not in self.regimes:
                continue
            grouped.setdefault(e.op_name, []).append(e)

        supported: list[MultiRegimeTargetScore] = []
        gaps: list[MultiRegimeTargetScore] = []
        for op_name, items in grouped.items():
            plane = items[0].plane
            is_sup = all(i.supported for i in items)
            niche = items[0].niche_key or f"(plane={plane.value},op={op_name})"
            if niche in blocked:
                continue
            per_share: dict[str, float] = {}
            per_us: dict[str, float] = {}
            weighted_sum = 0.0
            for i in items:
                reg = self.regimes[i.regime_id]
                sh = min(1.0, max(0.0, i.step_us / i.total_regime_step_us))
                per_share[i.regime_id] = round(sh, 6)
                per_us[i.regime_id] = round(i.step_us, 2)
                weighted_sum += reg.weight * sh
            dom_regime = max(per_share, key=lambda k: per_share[k])
            score = MultiRegimeTargetScore(
                op_name=op_name,
                plane=plane,
                weighted_share=round(weighted_sum / total_weight, 6),
                per_regime_shares=per_share,
                per_regime_us=per_us,
                dominant_regime_id=dom_regime,
                supported=is_sup,
                niche_key=niche,
            )
            if is_sup:
                supported.append(score)
            else:
                gaps.append(score)

        supported.sort(key=lambda s: s.weighted_share, reverse=True)
        gaps.sort(key=lambda s: s.weighted_share, reverse=True)
        return supported, gaps


class BackendCellMeasurement(BaseModel):
    """Measured latency and correctness of one backend/tactic at one `(phase, batch_tokens, seq_len)` cell."""

    backend_id: str
    phase: str  # "decode" | "prefill" | "mixed"
    batch_tokens: int = Field(ge=1)
    seq_len: int = Field(default=1024, ge=1)
    latency_us: float = Field(gt=0.0)
    passed_correctness: bool = True
    precision_track: PrecisionTrack = PrecisionTrack.LOSSLESS


class PiecewiseRule(BaseModel):
    """Contiguous `batch_tokens` dispatch interval within `(phase, seq_len)`."""

    phase: str
    seq_len: int
    min_batch_tokens: int
    max_batch_tokens: int
    selected_backend: str
    cell_count: int
    geomean_speedup_vs_incumbent: float
    worst_cell_speedup_vs_incumbent: float


class PiecewiseDispatchPlan(BaseModel):
    incumbent_backend: str
    allowed_track: PrecisionTrack
    min_speedup_guardband: float
    rules: tuple[PiecewiseRule, ...]
    upgraded_cells: int
    total_cells: int
    overall_geomean_speedup: float
    rejected_candidates: tuple[str, ...] = ()

    def resolve(self, *, phase: str, batch_tokens: int, seq_len: int = 1024) -> str:
        """Select backend for runtime `(phase, batch_tokens, seq_len)`, defaulting safely to incumbent."""
        same_phase = [r for r in self.rules if r.phase == phase]
        if not same_phase:
            return self.incumbent_backend
        # Match nearest measured seq_len bucket <= or closest to runtime seq_len
        seq_buckets = sorted({r.seq_len for r in same_phase})
        chosen_seq = min(seq_buckets, key=lambda s: abs(s - seq_len))
        for r in same_phase:
            if r.seq_len == chosen_seq and r.min_batch_tokens <= batch_tokens <= r.max_batch_tokens:
                return r.selected_backend
        return self.incumbent_backend


class PiecewisePolicySynthesizer:
    """Synthesizes zero-regression piecewise dispatch tables from multi-backend cell timings."""

    def __init__(
        self,
        *,
        incumbent_backend: str,
        allowed_track: PrecisionTrack = PrecisionTrack.LOSSLESS,
        min_speedup_guardband: float = 1.015,
    ) -> None:
        if min_speedup_guardband < 1.0:
            raise ValueError("min_speedup_guardband must be >= 1.0 to forbid regressions")
        self.incumbent_backend = incumbent_backend
        self.allowed_track = allowed_track
        self.min_speedup_guardband = min_speedup_guardband

    def synthesize(
        self,
        measurements: Sequence[BackendCellMeasurement],
    ) -> PiecewiseDispatchPlan:
        by_cell: dict[tuple[str, int, int], list[BackendCellMeasurement]] = {}
        rejected: set[str] = set()
        for m in measurements:
            by_cell.setdefault((m.phase, m.seq_len, m.batch_tokens), []).append(m)

        winners: list[tuple[str, int, int, str, float]] = []
        upgraded = 0
        for (phase, seq_len, batch_tokens), cell_items in sorted(by_cell.items()):
            inc = next(
                (
                    x
                    for x in cell_items
                    if x.backend_id == self.incumbent_backend and x.passed_correctness
                ),
                None,
            )
            if inc is None:
                raise ValueError(
                    f"Missing valid incumbent `{self.incumbent_backend}` measurement at "
                    f"({phase}, seq_len={seq_len}, batch_tokens={batch_tokens})"
                )
            best_backend = self.incumbent_backend
            best_speedup = 1.0
            for cand in cell_items:
                if cand.backend_id == self.incumbent_backend:
                    continue
                if not cand.passed_correctness:
                    rejected.add(f"{cand.backend_id}@({phase},B={batch_tokens},L={seq_len}):correctness_failed")
                    continue
                if (
                    self.allowed_track == PrecisionTrack.LOSSLESS
                    and cand.precision_track != PrecisionTrack.LOSSLESS
                ):
                    rejected.add(
                        f"{cand.backend_id}@({phase},B={batch_tokens},L={seq_len}):lossy_track_blocked"
                    )
                    continue
                sp = inc.latency_us / cand.latency_us
                if sp >= self.min_speedup_guardband and sp > best_speedup:
                    best_backend = cand.backend_id
                    best_speedup = sp
            if best_backend != self.incumbent_backend:
                upgraded += 1
            winners.append((phase, seq_len, batch_tokens, best_backend, best_speedup))

        # Compress adjacent batch_tokens cells sharing `(phase, seq_len, selected_backend)`
        rules: list[PiecewiseRule] = []
        idx = 0
        while idx < len(winners):
            p0, s0, b_min, backend0, sp0 = winners[idx]
            b_max = b_min
            speedups = [sp0]
            j = idx + 1
            while j < len(winners):
                pj, sj, bj, backendj, spj = winners[j]
                if pj == p0 and sj == s0 and backendj == backend0:
                    b_max = bj
                    speedups.append(spj)
                    j += 1
                else:
                    break
            prod = 1.0
            for s in speedups:
                prod *= s
            geomean = prod ** (1.0 / len(speedups))
            rules.append(
                PiecewiseRule(
                    phase=p0,
                    seq_len=s0,
                    min_batch_tokens=b_min,
                    max_batch_tokens=b_max,
                    selected_backend=backend0,
                    cell_count=len(speedups),
                    geomean_speedup_vs_incumbent=round(geomean, 6),
                    worst_cell_speedup_vs_incumbent=round(min(speedups), 6),
                )
            )
            idx = j

        overall_prod = 1.0
        for *_, sp in winners:
            overall_prod *= sp
        overall_geo = overall_prod ** (1.0 / len(winners)) if winners else 1.0

        return PiecewiseDispatchPlan(
            incumbent_backend=self.incumbent_backend,
            allowed_track=self.allowed_track,
            min_speedup_guardband=self.min_speedup_guardband,
            rules=tuple(rules),
            upgraded_cells=upgraded,
            total_cells=len(winners),
            overall_geomean_speedup=round(overall_geo, 6),
            rejected_candidates=tuple(sorted(rejected)),
        )
