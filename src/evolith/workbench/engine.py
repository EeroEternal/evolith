"""Morphable Workbench Engine: 2PC Transactional Hot-Apply, Watchdog & Graduation Bridge."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from pydantic import BaseModel, Field

from evolith.core.envelope import EnvelopeFailureLog, EnvelopeGap, EnvelopeGapCategory
from evolith.core.formal_replay import WorkbenchTxParticipant
from evolith.core.surface import (
    ActionRung,
    BlastRadiusPlan,
    TargetSurface,
    analyze_changeset_blast_radius,
)
from evolith.models import EvolutionPlane
from evolith.workbench.callsite import ShadowDiffReport, run_shadow_differential
from evolith.workbench.state_keeper import (
    GuardedGraphRunner,
    KernelCatalog,
    ProvisionHandle,
    StateKeeper,
)


class WorkbenchTimeoutError(RuntimeError):
    """Hard watchdog tripped during `hot_apply` (prevents multi-rank collective hangs)."""


class HotApplyResult(BaseModel):
    committed: bool
    rolled_back: bool
    rung: ActionRung
    active_version: int
    recaptured_graphs: bool = False
    failure_kind: str = ""  # "" | "envelope_gap" | "timeout_watchdog" | "shadow_drift" | "apply_error"
    failure_detail: str = ""
    blast_radius: BlastRadiusPlan


class GraduationBundle(BaseModel):
    """Verified inner-loop winner ready for cold-start promotion on formal `zstep`."""

    ready_for_cold_verify: bool
    candidate_id: str
    inner_screen_gain: float
    changed_lines: int
    complexity_free_lines: int
    estimated_net_gate_ratio: float
    zero_eager_fallbacks: bool
    shadow_passed: bool
    reasons: list[str] = Field(default_factory=list)


class MorphableWorkbenchEngine:
    """Evolith's built-in resident optimization engine paired with formal `zstep` graduation."""

    def __init__(
        self,
        surface: TargetSurface,
        *,
        state_keeper: StateKeeper | None = None,
        graph_mode: str = "required",
        graph_buckets: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        envelope_log: EnvelopeFailureLog | None = None,
        complexity_free_lines: int = 120,
        complexity_penalty_weight: float = 0.0002,
    ) -> None:
        self.surface = surface
        self.state_keeper = state_keeper or StateKeeper()
        self.catalog = KernelCatalog()
        self.graph_runner = GuardedGraphRunner(mode=graph_mode, buckets=graph_buckets)
        self.tx = WorkbenchTxParticipant()
        self.envelope_log = envelope_log
        self.complexity_free_lines = complexity_free_lines
        self.complexity_penalty_weight = complexity_penalty_weight
        self._provisions: dict[str, ProvisionHandle] = {}
        self._step_fn: Callable[[int, Any], Any] = lambda _rows, x: x

    def bind_step_runner(
        self,
        unit_id: str,
        instance: Any,
        step_fn: Callable[[int, Any], Any],
    ) -> ProvisionHandle:
        old = self._provisions.get(unit_id)
        if old is not None:
            old.revoke()
        prov = ProvisionHandle(unit_id=unit_id, instance=instance)
        self._provisions[unit_id] = prov
        self._step_fn = step_fn
        self.graph_runner.capture_all(
            self._step_fn,
            provisions=tuple(self._provisions.values()),
            buffers=self.state_keeper.all_buffers(),
            catalog=self.catalog,
        )
        return prov

    def dry_run_change(
        self,
        *,
        changed_files: tuple[str, ...] = (),
        changed_config_keys: tuple[str, ...] = (),
        patch_text: str = "",
    ) -> BlastRadiusPlan:
        return analyze_changeset_blast_radius(
            self.surface,
            changed_files=changed_files,
            changed_config_keys=changed_config_keys,
            patch_text=patch_text,
        )

    def hot_apply(
        self,
        *,
        candidate_id: str = "",
        changed_files: tuple[str, ...] = (),
        changed_config_keys: tuple[str, ...] = (),
        patch_text: str = "",
        mutation_fn: Callable[[MorphableWorkbenchEngine], None] | None = None,
        refit_buffers: dict[str, Callable[[Any], Any]] | None = None,
        shadow_stages: Sequence[tuple[str, Callable[[Any], Any], Callable[[Any], Any]]] = (),
        shadow_input: Any = None,
        watchdog_timeout_sec: float = 15.0,
    ) -> HotApplyResult:
        """Atomically apply a candidate change on the resident engine with 2PC + watchdog."""
        plan = self.dry_run_change(
            changed_files=changed_files,
            changed_config_keys=changed_config_keys,
            patch_text=patch_text,
        )

        if not plan.can_hot_apply_in_workbench:
            if self.envelope_log is not None:
                cat = (
                    EnvelopeGapCategory.PINNED_CONTRACT_BOUNDARY
                    if plan.pinned_violations
                    else EnvelopeGapCategory.TOPOLOGY_OR_COLLECTIVE_CHANGE
                )
                self.envelope_log.record(
                    EnvelopeGap(
                        candidate_id=candidate_id,
                        target_engine=self.surface.target_name,
                        plane=EvolutionPlane.L1_KERNEL,
                        category=cat,
                        blocked_by_boundary="; ".join(
                            plan.pinned_violations
                            or plan.forbidden_root_violations
                            or plan.notes
                            or (plan.rung_name,)
                        ),
                        attempted_optimization=f"hot_apply on {changed_files}",
                        required_rung=plan.rung_name,
                        files_involved=changed_files,
                    )
                )
            return HotApplyResult(
                committed=False,
                rolled_back=False,
                rung=plan.rung,
                active_version=self.tx.active_version,
                failure_kind="envelope_gap",
                failure_detail=f"Requires {plan.rung_name}; recorded in EnvelopeFailureLog",
                blast_radius=plan,
            )

        saved_ops = self.catalog.snapshot_ops()
        saved_step_fn = self._step_fn
        saved_provisions = dict(self._provisions)
        saved_buf_snapshots = {
            b.name: (b.data, b.generation, b.fills) for b in self.state_keeper.all_buffers()
        }

        self.tx.apply_action("prepare_v2")
        self.tx.apply_action("pause_epoch")

        def _execute_body() -> ShadowDiffReport | None:
            self.tx.apply_action("execute_swap")
            if refit_buffers:
                for bname, tf_fn in refit_buffers.items():
                    self.state_keeper.refit_buffer(bname, tf_fn)
                self.tx.apply_action("refit_weights_inplace")
            if mutation_fn is not None:
                mutation_fn(self)
            if shadow_stages and shadow_input is not None:
                return run_shadow_differential(shadow_stages, shadow_input)
            return None

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_execute_body)
                shadow_rep = fut.result(timeout=watchdog_timeout_sec)
        except FuturesTimeoutError:
            self._restore_runtime_state(saved_ops, saved_step_fn, saved_provisions, saved_buf_snapshots)
            self.tx.apply_action("abort_and_rollback")
            return HotApplyResult(
                committed=False,
                rolled_back=True,
                rung=plan.rung,
                active_version=self.tx.active_version,
                failure_kind="timeout_watchdog",
                failure_detail=(
                    f"Watchdog tripped after {watchdog_timeout_sec}s; rolled back without "
                    "charging premise strike"
                ),
                blast_radius=plan,
            )
        except Exception as exc:
            self._restore_runtime_state(saved_ops, saved_step_fn, saved_provisions, saved_buf_snapshots)
            self.tx.apply_action("abort_and_rollback")
            return HotApplyResult(
                committed=False,
                rolled_back=True,
                rung=plan.rung,
                active_version=self.tx.active_version,
                failure_kind="apply_error",
                failure_detail=str(exc),
                blast_radius=plan,
            )

        if shadow_rep is not None and not shadow_rep.passed:
            self._restore_runtime_state(saved_ops, saved_step_fn, saved_provisions, saved_buf_snapshots)
            self.tx.apply_action("abort_and_rollback")
            return HotApplyResult(
                committed=False,
                rolled_back=True,
                rung=plan.rung,
                active_version=self.tx.active_version,
                failure_kind="shadow_drift",
                failure_detail=(
                    f"Shadow differential diverged at stage {shadow_rep.first_divergent_stage} "
                    f"(`{shadow_rep.first_divergent_op}`)"
                ),
                blast_radius=plan,
            )

        recaptured = False
        if (
            plan.recapture_cuda_graphs
            or bool(refit_buffers)
            or self.tx.graph_captured_version != self.tx.active_version
            or self.tx.graph_captured_fills != self.tx.state_fills
        ):
            self.graph_runner.capture_all(
                self._step_fn,
                provisions=tuple(self._provisions.values()),
                buffers=self.state_keeper.all_buffers(),
                catalog=self.catalog,
            )
            self.tx.apply_action("recapture_graphs")
            recaptured = True

        self.tx.apply_action("persist_commit")
        self.tx.apply_action("commit_and_resume")
        return HotApplyResult(
            committed=True,
            rolled_back=False,
            rung=plan.rung,
            active_version=self.tx.active_version,
            recaptured_graphs=recaptured,
            blast_radius=plan,
        )

    def _restore_runtime_state(
        self,
        saved_ops: dict[str, Callable[..., Any]],
        saved_step_fn: Callable[[int, Any], Any],
        saved_provisions: dict[str, ProvisionHandle],
        saved_buf_snapshots: dict[str, tuple[Any, int, int]],
    ) -> None:
        for cur_prov in self._provisions.values():
            cur_prov.revoke()
        self.catalog.restore_ops(saved_ops)
        self._step_fn = saved_step_fn
        self._provisions = {
            uid: ProvisionHandle(unit_id=uid, instance=p.instance)
            for uid, p in saved_provisions.items()
        }
        for name, (data, gen, fills) in saved_buf_snapshots.items():
            buf = self.state_keeper.get_buffer(name)
            buf.data = data
            buf.generation = gen
            buf.fills = fills
        self.graph_runner.capture_all(
            self._step_fn,
            provisions=tuple(self._provisions.values()),
            buffers=self.state_keeper.all_buffers(),
            catalog=self.catalog,
        )

    def run_step(self, batch_rows: int, inputs: Any) -> Any:
        return self.graph_runner.run_step(batch_rows, inputs, eager_fn=self._step_fn)

    def prepare_graduation_bundle(
        self,
        *,
        candidate_id: str,
        inner_screen_gain: float,
        changed_lines: int,
        shadow_passed: bool = True,
        min_net_gate_ratio: float = 1.03,
    ) -> GraduationBundle:
        """Check whether an inner-screened candidate is ready for cold-start `zstep` Verify.

        Catches Occam complexity penalties (such as `g0010`'s 138-LOC `-0.36%` tax)
        and eager fallbacks BEFORE spending 60+ minutes of 8xGPU cold-start Verify!
        """
        telem = self.graph_runner.telemetry()
        zero_fallbacks = telem["eager_fallback_steps"] == 0 and telem["graph_guard_failures"] == 0
        excess_loc = max(0, changed_lines - self.complexity_free_lines)
        penalty = excess_loc * self.complexity_penalty_weight
        est_net_gate = (1.0 + inner_screen_gain) - penalty
        reasons: list[str] = []
        if not shadow_passed:
            reasons.append("Shadow per-operator differential check did not pass")
        if not zero_fallbacks:
            reasons.append(
                f"Non-zero eager fallbacks ({telem['eager_fallback_steps']}) or "
                f"graph guard failures ({telem['graph_guard_failures']})"
            )
        if est_net_gate < min_net_gate_ratio:
            reasons.append(
                f"Estimated net_gate_ratio={est_net_gate:.4f} < {min_net_gate_ratio:.4f} "
                f"at {changed_lines} LOC (free={self.complexity_free_lines}, penalty={penalty:.4f})"
            )
        return GraduationBundle(
            ready_for_cold_verify=not reasons,
            candidate_id=candidate_id,
            inner_screen_gain=inner_screen_gain,
            changed_lines=changed_lines,
            complexity_free_lines=self.complexity_free_lines,
            estimated_net_gate_ratio=round(est_net_gate, 4),
            zero_eager_fallbacks=zero_fallbacks,
            shadow_passed=shadow_passed,
            reasons=reasons,
        )
