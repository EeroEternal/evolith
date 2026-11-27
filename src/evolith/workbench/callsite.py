"""Workload-Driven Step Bottleneck Ledger, Call-Site Contracts & `--shadow` Diffing.

Connects measured step-level operator shares to AKA-style frozen call-site
contracts and per-operator `--shadow incumbent` numerical divergence pinpointing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from evolith.core.topology import PrecisionTrack
from evolith.models import EvolutionPlane


class StepOperatorSample(BaseModel):
    op_name: str
    plane: EvolutionPlane = EvolutionPlane.L1_KERNEL
    calls_per_step: int = 1
    total_step_us: float
    supported: bool = True
    niche_key: str = ""
    boundary_note: str = ""
    unit_id: str = ""
    action_rung: str = ""
    hotswap_extern_c: bool = False
    shapes: tuple[tuple[int, ...], ...] = ()
    strides: tuple[tuple[int, ...], ...] = ()
    dtype: str = "bfloat16"
    selected_tactic: str = ""
    is_fallback_route: bool = False
    tuned_compute_units: tuple[int, ...] = ()
    precision_track: PrecisionTrack = PrecisionTrack.LOSSLESS


class RankedBottleneck(BaseModel):
    op_name: str
    plane: EvolutionPlane
    share_of_step: float
    total_step_us: float
    calls_per_step: int = 1
    supported: bool
    niche_key: str
    boundary_note: str = ""
    unit_id: str = ""
    action_rung: str = ""
    hotswap_extern_c: bool = False


class CallSiteContract(BaseModel):
    """Frozen operator call-site specification extracted from a live step."""

    contract_id: str
    op_name: str
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...] = ()
    dtype: str = "bfloat16"
    world_size: int = 1
    rtol: float = 1e-2
    atol: float = 1e-2
    baseline_step_us: float = 0.0
    calls_per_step: int = 1
    share_of_step: float = 0.0
    unit_id: str = ""
    action_rung: str = ""
    hotswap_extern_c: bool = False
    niche_key: str = ""
    selected_tactic: str = ""
    is_fallback_route: bool = False
    precision_track: PrecisionTrack = PrecisionTrack.LOSSLESS


class StepProfileReport(BaseModel):
    """End-to-end resident `profile_step()` bottleneck ledger & extracted contracts."""

    ok: bool = True
    bucket: int
    context_len: int
    warmup_steps: int
    timed_steps: int
    world_size: int
    captured_graph_step_us: float
    instrumented_gpu_step_us: float
    max_rank_step_us: float = 0.0
    attributed_stage_us: float
    unattributed_step_us: float
    coverage_fraction: float
    samples: list[StepOperatorSample] = Field(default_factory=list)
    supported_ranked: list[RankedBottleneck] = Field(default_factory=list)
    capability_gaps: list[RankedBottleneck] = Field(default_factory=list)
    selected_target: RankedBottleneck | None = None
    callsite_contracts: list[CallSiteContract] = Field(default_factory=list)
    blocked_niches_excluded: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class ShadowDiffEntry(BaseModel):
    stage_index: int
    op_name: str
    max_abs_diff: float
    cosine_sim: float
    within_tolerance: bool


class ShadowDiffReport(BaseModel):
    passed: bool
    first_divergent_op: str | None = None
    first_divergent_stage: int | None = None
    entries: list[ShadowDiffEntry] = Field(default_factory=list)


class StepBottleneckLedger:
    """Ranks observed step operators by measured latency share (`AGENTS.md`)."""

    def __init__(self, samples: Sequence[StepOperatorSample]) -> None:
        if not samples:
            raise ValueError("StepBottleneckLedger requires at least one measured sample")
        self.samples = tuple(samples)

    def rank_targets(
        self,
        *,
        blocked_niches: set[str] | None = None,
    ) -> tuple[list[RankedBottleneck], list[RankedBottleneck]]:
        """Return `(supported_ranked, capability_gaps)` ordered by measured step share."""
        blocked = blocked_niches or set()
        total_us = sum(max(0.0, s.total_step_us) for s in self.samples)
        if total_us <= 0.0:
            return [], []

        supported: list[RankedBottleneck] = []
        gaps: list[RankedBottleneck] = []
        for s in sorted(self.samples, key=lambda x: x.total_step_us, reverse=True):
            eff_niche = s.niche_key or f"(plane={s.plane.value},op={s.op_name})"
            if eff_niche in blocked or (s.niche_key and s.niche_key in blocked):
                continue
            item = RankedBottleneck(
                op_name=s.op_name,
                plane=s.plane,
                share_of_step=s.total_step_us / total_us,
                total_step_us=s.total_step_us,
                calls_per_step=s.calls_per_step,
                supported=s.supported,
                niche_key=eff_niche,
                boundary_note=s.boundary_note,
                unit_id=s.unit_id,
                action_rung=s.action_rung,
                hotswap_extern_c=s.hotswap_extern_c,
            )
            if s.supported:
                supported.append(item)
            else:
                gaps.append(item)
        return supported, gaps

    def build_callsite_contracts(
        self,
        *,
        bucket: int = 1,
        world_size: int = 1,
        top_k: int = 4,
        blocked_niches: set[str] | None = None,
    ) -> list[CallSiteContract]:
        """Extract frozen AKA-style `CallSiteContract`s for top supported bottlenecks."""
        supported, _gaps = self.rank_targets(blocked_niches=blocked_niches)
        sample_by_op = {s.op_name: s for s in self.samples}
        contracts: list[CallSiteContract] = []
        for rank_idx, item in enumerate(supported[: max(0, top_k)]):
            src = sample_by_op.get(item.op_name)
            shapes = src.shapes if (src and src.shapes) else ((bucket,),)
            strides = src.strides if src else ()
            dtype = src.dtype if src else "bfloat16"
            contracts.append(
                CallSiteContract(
                    contract_id=f"callsite-b{bucket}-r{rank_idx}-{item.op_name.replace('.', '_')}",
                    op_name=item.op_name,
                    shapes=shapes,
                    strides=strides,
                    dtype=dtype,
                    world_size=world_size,
                    baseline_step_us=round(item.total_step_us, 2),
                    calls_per_step=item.calls_per_step,
                    share_of_step=round(item.share_of_step, 6),
                    unit_id=item.unit_id,
                    action_rung=item.action_rung,
                    hotswap_extern_c=item.hotswap_extern_c,
                    niche_key=item.niche_key,
                    selected_tactic=src.selected_tactic if src else "",
                    is_fallback_route=src.is_fallback_route if src else False,
                    precision_track=src.precision_track if src else PrecisionTrack.LOSSLESS,
                )
            )
        return contracts


def _to_float64_flat(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    return arr


def run_shadow_differential(
    stages: Sequence[tuple[str, Callable[[Any], Any], Callable[[Any], Any]]],
    initial_activation: Any,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> ShadowDiffReport:
    """Run `--shadow incumbent` per-stage comparison on identical step activations.

    Feeds the incumbent activation stream into both `incumbent_fn` and
    `candidate_fn` at every stage so numerical divergence is isolated to the
    exact first operator that drifted rather than compounding downstream.
    """
    entries: list[ShadowDiffEntry] = []
    first_op: str | None = None
    first_idx: int | None = None
    cur = initial_activation

    for idx, (op_name, incumbent_fn, candidate_fn) in enumerate(stages):
        ref_out = incumbent_fn(cur)
        cand_out = candidate_fn(cur)
        ref_arr = _to_float64_flat(ref_out)
        cand_arr = _to_float64_flat(cand_out)
        if ref_arr.shape != cand_arr.shape or not np.all(np.isfinite(cand_arr)):
            max_diff = float("inf")
            cos_sim = 0.0
            ok = False
        else:
            diff = np.abs(cand_arr - ref_arr)
            max_diff = float(np.max(diff)) if diff.size else 0.0
            denom = float(np.linalg.norm(ref_arr) * np.linalg.norm(cand_arr))
            cos_sim = float(np.dot(ref_arr, cand_arr) / denom) if denom > 1e-12 else 1.0
            ok = bool(np.allclose(cand_arr, ref_arr, rtol=rtol, atol=atol) and math.isfinite(max_diff))

        entries.append(
            ShadowDiffEntry(
                stage_index=idx,
                op_name=op_name,
                max_abs_diff=max_diff,
                cosine_sim=cos_sim,
                within_tolerance=ok,
            )
        )
        if not ok and first_op is None:
            first_op = op_name
            first_idx = idx
        cur = ref_out

    return ShadowDiffReport(
        passed=(first_op is None),
        first_divergent_op=first_op,
        first_divergent_stage=first_idx,
        entries=entries,
    )
