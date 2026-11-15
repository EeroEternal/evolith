from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from evolith.zstep.archive import ZStepArchive
from evolith.zstep.campaigns import (
    CampaignSpec,
    apply_campaign_to_task,
    resolve_campaign,
)
from evolith.zstep.dispatch_verifier import run_dispatch_verification
from evolith.zstep.gates import (
    evaluate_candidate,
    evaluate_screen,
    load_benchmark,
    run_local_gates,
)
from evolith.zstep.profiling import build_sm120_reflection_card
from evolith.zstep.proposer import build_proposer, parse_proposal_metadata
from evolith.zstep.redteam import (
    PremiseStrikeError,
    compute_spec_pins,
    run_redteam_gate,
)
from evolith.zstep.remote import ZStepRemoteRunner
from evolith.zstep.task import CandidateStatus, PatchStats, ZStepRSITask
from evolith.zstep.workspace import ZStepWorkspace


class ScreenRejectedError(RuntimeError):
    pass


class ZStepRSILoop:
    def __init__(
        self,
        task: ZStepRSITask,
        *,
        workspace: Any | None = None,
        archive: ZStepArchive | None = None,
        proposer: Any | None = None,
        remote: Any | None = None,
    ) -> None:
        self.task = task
        self.workspace = workspace or ZStepWorkspace(task)
        self.archive = archive or ZStepArchive(task.state_dir / "archive.sqlite3")
        self.proposer = proposer or build_proposer(task)
        self.remote = remote or ZStepRemoteRunner(task)

    def _find_cached_baseline_screen(self, baseline_hash: str | None = None) -> Path | None:
        if baseline_hash:
            exact = self.task.baseline_cache_dir / f"{baseline_hash[:16]}-screen-metrics.json"
            if exact.is_file():
                return exact
        if self.task.baseline_cache_dir.is_dir():
            candidates = sorted(self.task.baseline_cache_dir.glob("*-screen-metrics.json"))
            if candidates:
                return candidates[-1]
        return None

    def _find_latest_candidate_screen(self, parent_id: str | None) -> Path | None:
        for row in reversed(self.archive.list_candidates()):
            if (row.get("parent_id") or None) != parent_id:
                continue
            evidence_raw = str(row.get("evidence_dir") or "").strip()
            cid = str(row.get("candidate_id") or "").strip()
            evidence_dir = (
                Path(evidence_raw)
                if evidence_raw
                else (self.task.state_dir / "evidence" / cid if cid else None)
            )
            if evidence_dir is None:
                continue
            screen_file = evidence_dir / "screen-metrics.json"
            if screen_file.is_file():
                return screen_file
        return None

    def _prompt(
        self,
        parent_id: str | None,
        candidate_dir: Path,
        baseline_hash: str | None = None,
        blocked_niches: dict[str, list[str]] | None = None,
        active_task: ZStepRSITask | None = None,
        active_campaign: CampaignSpec | None = None,
    ) -> str:
        task = active_task or self.task
        compressed_history = self.archive.compressed_lineage_memory(limit=8)
        allowed_list = "\n".join(f"  - {root}" for root in task.allowed_roots)
        reflection_card = build_sm120_reflection_card(
            candidate_dir,
            self._find_cached_baseline_screen(baseline_hash),
            blocked_niches=blocked_niches,
            candidate_screen_metrics=self._find_latest_candidate_screen(parent_id),
        )
        campaign_section = (
            f"\n{active_campaign.blueprint}\n" if active_campaign is not None else ""
        )
        niche_example = (
            active_campaign.primary_niche_key
            if active_campaign is not None
            else "(plane=<l1_kernel|l2_scheduler>,focus=<short_tag>)"
        )
        return f"""Parent: {parent_id or 'frozen-zstep-baseline'}
Improve DeepSeek-V4.1-Flash (TP=8, SM120) serving throughput without weakening correctness or latency.

Strict Modification Contract:
- Edit ONLY these allowlisted zstep source paths:
{allowed_list}
- Maximum changed lines (added + deleted): {task.max_changed_lines} LOC.
- Occam's Razor Complexity Gate: patches exceeding {task.complexity_free_lines} LOC incur a net-gain penalty of {task.complexity_penalty_weight:.4f} per extra line; keep mutations surgical and high-leverage.
- Attack the Premise (2-Strike Circuit Breaker): any explicit `niche_key` rejected {task.premise_strike_limit} consecutive times under the same parent is mechanically blocked; choose an unblocked bottleneck/niche.
- Red-Team Boundary Safety: never introduce auxiliary CUDA streams without Event/wait_stream sync, transient pageable tensors with `non_blocking=True`, unguarded `@triton.autotune`, or unchecked Rust indexing.
- Target acceptance bar: >= {task.min_geomean_gain:.1%} geomean output throughput gain, <= {task.max_tier_throughput_regression:.1%} per-tier throughput regression, <= {task.max_tpot_regression:.1%} latency regression, 0 Triton recompilations, 100% prime_squares_table exact greedy parity across single/batched/chunked modes.
- Do not modify benchmark semantics, dependency manifests, model files, or verifier rules.
- Do not access remote hosts, credentials, other workspaces, or hidden verification cases.
- Run relevant local checks. The public screen tiers are {task.screen_tiers}.
{campaign_section}
{reflection_card}

AIDE^2 Compressed Lineage Memory (Previous Candidates — Do NOT Repeat Rejected Hypotheses):
{compressed_history}

At the end of your final response, include one single-line structured summary:
EVOLITH_PROPOSAL_JSON: {{"hypothesis": "<why this bottleneck & mechanism yields >=3% gain>", "niche_key": "{niche_example}", "patch_summary": "<concise summary of code change>"}}
"""

    def _write_failure(self, evidence_dir: Path, error: Exception) -> None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "error": str(error)}, indent=2),
            encoding="utf-8",
        )

    def _propose_and_capture(
        self,
        candidate_dir: Path,
        prompt: str,
        evidence_dir: Path,
        active_task: ZStepRSITask | None = None,
    ) -> tuple[str, PatchStats]:
        task_override = active_task or self.task
        has_prop_task = hasattr(self.proposer, "task")
        prev_prop_task = getattr(self.proposer, "task", None) if has_prop_task else None
        if has_prop_task:
            self.proposer.task = task_override
        try:
            if (
                hasattr(self.workspace, "isolated_proposer_sandbox")
                and hasattr(self.workspace, "capture_patch_stats")
                and hasattr(self.workspace, "apply_patch")
            ):
                prev_ws_task = getattr(self.workspace, "task", None)
                if prev_ws_task is not None:
                    self.workspace.task = task_override
                try:
                    with self.workspace.isolated_proposer_sandbox(candidate_dir) as sandbox_dir:
                        proposer_out = self.proposer.propose(sandbox_dir, prompt, evidence_dir)
                        patch_stats = self.workspace.capture_patch_stats(sandbox_dir)
                    self.workspace.apply_patch(candidate_dir, patch_stats.patch)
                    return proposer_out, patch_stats
                finally:
                    if prev_ws_task is not None:
                        self.workspace.task = prev_ws_task

            proposer_out = self.proposer.propose(candidate_dir, prompt, evidence_dir)
            raw_patch = self.workspace.capture_patch(candidate_dir)
            return proposer_out, PatchStats(
                patch=raw_patch,
                changed_lines=len(raw_patch.splitlines()),
            )
        finally:
            if has_prop_task:
                self.proposer.task = prev_prop_task

    def _run_screen_and_gate(
        self,
        candidate_id: str,
        baseline_screen: Path | None,
        candidate_archive: Path,
        evidence_dir: Path,
        active_task: ZStepRSITask | None = None,
    ) -> None:
        screen_metrics = self.remote.screen(candidate_id, candidate_archive, evidence_dir)
        screen_parity = evidence_dir / "screen-parity.json"
        has_parity = screen_parity.is_file()
        has_tiers = False
        if screen_metrics.is_file():
            try:
                payload = json.loads(screen_metrics.read_text(encoding="utf-8"))
                has_tiers = isinstance(payload, dict) and "tiers" in payload
            except ValueError:
                has_tiers = False

        if has_tiers or has_parity:
            screen_decision = evaluate_screen(
                screen_metrics,
                screen_parity if has_parity else None,
                active_task or self.task,
                baseline_screen,
            )
            (evidence_dir / "screen-decision.json").write_text(
                screen_decision.model_dump_json(indent=2),
                encoding="utf-8",
            )
            self.archive.update_stage(
                candidate_id,
                "verify" if screen_decision.passed else "screen",
                screen_gain=screen_decision.gain,
                metric_delta=screen_decision.metric_delta,
            )
            if not screen_decision.passed:
                raise ScreenRejectedError("; ".join(screen_decision.reasons))

    def _finalize_verification(
        self,
        candidate_id: str,
        baseline_archive: Path,
        candidate_archive: Path,
        source_hash: str,
        changed_lines: int,
        evidence_dir: Path,
        active_task: ZStepRSITask | None = None,
    ) -> CandidateStatus:
        candidate_json, baseline_json, parity_json = self.remote.verify(
            candidate_id, baseline_archive, candidate_archive, evidence_dir
        )
        if hashlib.sha256(candidate_archive.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("candidate source hash changed before verification")
        decision = evaluate_candidate(
            load_benchmark(baseline_json),
            load_benchmark(candidate_json),
            parity_json,
            active_task or self.task,
            changed_lines=changed_lines,
        )
        (evidence_dir / "decision.json").write_text(
            decision.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if decision.accepted:
            self.archive.accept_candidate(
                candidate_id,
                str(evidence_dir),
                decision.geomean_gain,
                net_gate_ratio=decision.net_gate_ratio,
                metric_delta=decision.metric_delta,
            )
            return CandidateStatus.ACCEPTED

        reason = "; ".join(decision.reasons)
        self.archive.finish_candidate(
            candidate_id,
            CandidateStatus.VERIFY_REJECTED,
            str(evidence_dir),
            reason,
            geomean_gain=decision.geomean_gain,
            net_gate_ratio=decision.net_gate_ratio,
            metric_delta=decision.metric_delta,
        )
        return CandidateStatus.VERIFY_REJECTED

    def run(self) -> list[CandidateStatus]:
        statuses: list[CandidateStatus] = []
        for _ in range(self.task.max_generations):
            generation = self.archive.next_generation()
            candidate_id = f"g{generation:04d}-{uuid.uuid4().hex[:8]}"
            parent_id = self.archive.current_parent()
            candidate_dir = self.workspace.create_candidate(
                candidate_id, self.archive.accepted_patches()
            )
            evidence_dir = self.task.state_dir / "evidence" / candidate_id
            evidence_dir.mkdir(parents=True, exist_ok=False)
            self.archive.register_attempt(
                candidate_id, generation, parent_id, str(evidence_dir)
            )
            stage = "propose"
            try:
                blocked = self.archive.blocked_niches(
                    strike_limit=self.task.premise_strike_limit,
                    parent_id=parent_id,
                    exclude_candidate_id=candidate_id,
                )
                active_campaign = resolve_campaign(self.task.campaign, blocked)
                active_task = apply_campaign_to_task(self.task, active_campaign)
                if active_campaign is not None:
                    (evidence_dir / "campaign.json").write_text(
                        json.dumps(
                            {
                                "campaign_id": active_campaign.campaign_id,
                                "priority_rank": active_campaign.priority_rank,
                                "title": active_campaign.title,
                                "primary_niche_key": active_campaign.primary_niche_key,
                                "allowed_roots": list(active_campaign.allowed_roots),
                                "complexity_free_lines": active_campaign.complexity_free_lines,
                                "max_changed_lines": active_campaign.max_changed_lines,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                spec_pins = compute_spec_pins(candidate_dir, active_task.pinned_spec_files)
                baseline_archive = self.workspace.freeze_source(
                    candidate_dir, evidence_dir / "baseline.tar.gz"
                )
                baseline_hash = hashlib.sha256(baseline_archive.read_bytes()).hexdigest()
                proposer_out, patch_stats = self._propose_and_capture(
                    candidate_dir,
                    self._prompt(
                        parent_id,
                        candidate_dir,
                        baseline_hash,
                        blocked,
                        active_task=active_task,
                        active_campaign=active_campaign,
                    ),
                    evidence_dir,
                    active_task=active_task,
                )
                proposal_meta = parse_proposal_metadata(
                    proposer_out,
                    changed_files=patch_stats.changed_files,
                    changed_lines=patch_stats.changed_lines,
                )
                patch_path = self.task.state_dir / "patches" / f"{candidate_id}.patch"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch_path.write_bytes(patch_stats.patch)
                candidate_archive = self.workspace.freeze_source(
                    candidate_dir, evidence_dir / "candidate.tar.gz"
                )
                source_hash = hashlib.sha256(candidate_archive.read_bytes()).hexdigest()
                self.archive.start_candidate(
                    candidate_id,
                    generation,
                    parent_id,
                    patch_stats.patch,
                    source_hash,
                    plane=patch_stats.plane,
                    niche_key=proposal_meta.niche_key,
                    changed_lines=patch_stats.changed_lines,
                    changed_files=patch_stats.changed_files,
                    hypothesis=proposal_meta.hypothesis,
                    patch_summary=proposal_meta.patch_summary,
                    evidence_dir=str(evidence_dir),
                )

                stage = "local"
                if proposal_meta.niche_key in blocked:
                    prior = blocked[proposal_meta.niche_key]
                    raise PremiseStrikeError(
                        f"Premise 2-strike circuit breaker tripped on `{proposal_meta.niche_key}` "
                        f"after {len(prior)} consecutive rejections (last: {prior[-1]})"
                    )
                run_redteam_gate(
                    candidate_dir,
                    patch_stats,
                    active_task,
                    evidence_dir,
                    expected_pins=spec_pins,
                )
                run_dispatch_verification(candidate_dir, evidence_dir)
                run_local_gates(candidate_dir, active_task, evidence_dir)
                if self.task.propose_only:
                    self.archive.update_stage(candidate_id, "preflight")
                    statuses.append(CandidateStatus.PROPOSED)
                    continue
                stage = "preflight"
                self.remote.preflight()
                if active_task.enable_gpu_smoke and hasattr(self.remote, "gpu_smoke"):
                    stage = "screen"
                    self.remote.gpu_smoke(candidate_id, candidate_archive)
                baseline_screen: Path | None = None
                if hasattr(self.remote, "screen_baseline"):
                    stage = "baseline_screen"
                    baseline_screen = self.remote.screen_baseline(
                        baseline_hash, baseline_archive
                    )
                stage = "screen"
                self.archive.update_stage(candidate_id, "screen")
                self._run_screen_and_gate(
                    candidate_id,
                    baseline_screen,
                    candidate_archive,
                    evidence_dir,
                    active_task=active_task,
                )
                stage = "verify"
                self.archive.update_stage(candidate_id, "verify")
                final_status = self._finalize_verification(
                    candidate_id,
                    baseline_archive,
                    candidate_archive,
                    source_hash,
                    patch_stats.changed_lines,
                    evidence_dir,
                    active_task=active_task,
                )
                statuses.append(final_status)
            except Exception as error:
                status = {
                    "propose": CandidateStatus.LOCAL_REJECTED,
                    "local": CandidateStatus.LOCAL_REJECTED,
                    "screen": CandidateStatus.SCREEN_REJECTED,
                }.get(stage, CandidateStatus.FAILED)
                self._write_failure(evidence_dir, error)
                self.archive.finish_candidate(
                    candidate_id, status, str(evidence_dir), str(error)
                )
                statuses.append(status)
                if stage in {"preflight", "baseline_screen"}:
                    break
        return statuses

    def resume(self, candidate_id: str) -> CandidateStatus:
        """
        Resumes an interrupted remote screen/verify candidate from its frozen
        `baseline.tar.gz` and `candidate.tar.gz` archives and round-level checkpoints.
        """
        row = self.archive.get_candidate(candidate_id)
        if str(row["status"]) != CandidateStatus.PROPOSED.value:
            return CandidateStatus(str(row["status"]))

        evidence_raw = str(row["evidence_dir"] or "")
        evidence_dir = (
            Path(evidence_raw)
            if evidence_raw
            else self.task.state_dir / "evidence" / candidate_id
        )
        baseline_archive = evidence_dir / "baseline.tar.gz"
        candidate_archive = evidence_dir / "candidate.tar.gz"
        if not baseline_archive.is_file() or not candidate_archive.is_file():
            raise FileNotFoundError(
                f"Frozen archives missing for {candidate_id} in {evidence_dir}"
            )

        expected_hash = str(row["source_hash"] or "")
        actual_hash = hashlib.sha256(candidate_archive.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError("candidate source hash mismatch on resume")

        campaign_selector = self.task.campaign
        campaign_file = evidence_dir / "campaign.json"
        if (not campaign_selector or campaign_selector == "none") and campaign_file.is_file():
            try:
                saved = json.loads(campaign_file.read_text(encoding="utf-8"))
                if isinstance(saved, dict) and isinstance(saved.get("campaign_id"), str):
                    campaign_selector = saved["campaign_id"]
            except ValueError:
                pass
        active_campaign = resolve_campaign(campaign_selector)
        active_task = apply_campaign_to_task(self.task, active_campaign)

        baseline_hash = hashlib.sha256(baseline_archive.read_bytes()).hexdigest()
        changed_lines = int(row.get("changed_lines") or 0)
        stage = "preflight"
        try:
            self.remote.preflight()
            screen_metrics = evidence_dir / "screen-metrics.json"
            if not screen_metrics.is_file():
                baseline_screen: Path | None = None
                if hasattr(self.remote, "screen_baseline"):
                    stage = "baseline_screen"
                    baseline_screen = self.remote.screen_baseline(
                        baseline_hash, baseline_archive
                    )
                stage = "screen"
                self._run_screen_and_gate(
                    candidate_id,
                    baseline_screen,
                    candidate_archive,
                    evidence_dir,
                    active_task=active_task,
                )
            else:
                parity_path = evidence_dir / "screen-parity.json"
                screen_decision = evaluate_screen(
                    screen_metrics,
                    parity_path if parity_path.is_file() else None,
                    active_task,
                    self._find_cached_baseline_screen(baseline_hash),
                )
                if not screen_decision.passed:
                    raise ScreenRejectedError("; ".join(screen_decision.reasons))

            stage = "verify"
            self.archive.update_stage(candidate_id, "verify")
            return self._finalize_verification(
                candidate_id,
                baseline_archive,
                candidate_archive,
                expected_hash,
                changed_lines,
                evidence_dir,
                active_task=active_task,
            )
        except Exception as error:
            status = (
                CandidateStatus.SCREEN_REJECTED
                if stage == "screen"
                else CandidateStatus.FAILED
            )
            self._write_failure(evidence_dir, error)
            self.archive.finish_candidate(
                candidate_id, status, str(evidence_dir), str(error)
            )
            return status
