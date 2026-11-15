from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from evolith.zstep.archive import ZStepArchive
from evolith.zstep.gates import (
    evaluate_candidate,
    evaluate_screen,
    load_benchmark,
    run_local_gates,
)
from evolith.zstep.profiling import build_sm120_reflection_card
from evolith.zstep.proposer import KimiCodeProposer, parse_proposal_metadata
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
        self.proposer = proposer or KimiCodeProposer(task)
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

    def _prompt(
        self,
        parent_id: str | None,
        candidate_dir: Path,
        baseline_hash: str | None = None,
        blocked_niches: dict[str, list[str]] | None = None,
    ) -> str:
        compressed_history = self.archive.compressed_lineage_memory(limit=8)
        allowed_list = "\n".join(f"  - {root}" for root in self.task.allowed_roots)
        reflection_card = build_sm120_reflection_card(
            candidate_dir,
            self._find_cached_baseline_screen(baseline_hash),
            blocked_niches=blocked_niches,
        )
        return f"""Parent: {parent_id or 'frozen-zstep-baseline'}
Improve DeepSeek-V4.1-Flash (TP=8, SM120) serving throughput without weakening correctness or latency.

Strict Modification Contract:
- Edit ONLY these allowlisted zstep source paths:
{allowed_list}
- Maximum changed lines (added + deleted): {self.task.max_changed_lines} LOC.
- Occam's Razor Complexity Gate: patches exceeding {self.task.complexity_free_lines} LOC incur a net-gain penalty of {self.task.complexity_penalty_weight:.4f} per extra line; keep mutations surgical and high-leverage.
- Attack the Premise (2-Strike Circuit Breaker): any explicit `niche_key` rejected {self.task.premise_strike_limit} consecutive times under the same parent is mechanically blocked; choose an unblocked bottleneck/niche.
- Red-Team Boundary Safety: never introduce auxiliary CUDA streams without Event/wait_stream sync, transient pageable tensors with `non_blocking=True`, unguarded `@triton.autotune`, or unchecked Rust indexing.
- Target acceptance bar: >= {self.task.min_geomean_gain:.1%} geomean output throughput gain, <= {self.task.max_tier_throughput_regression:.1%} per-tier throughput regression, <= {self.task.max_tpot_regression:.1%} latency regression, 0 Triton recompilations, 100% prime_squares_table exact greedy parity across single/batched/chunked modes.
- Do not modify benchmark semantics, dependency manifests, model files, or verifier rules.
- Do not access remote hosts, credentials, other workspaces, or hidden verification cases.
- Run relevant local checks. The public screen tiers are {self.task.screen_tiers}.

{reflection_card}

AIDE^2 Compressed Lineage Memory (Previous Candidates — Do NOT Repeat Rejected Hypotheses):
{compressed_history}

At the end of your final response, include one single-line structured summary:
EVOLITH_PROPOSAL_JSON: {{"hypothesis": "<why this bottleneck & mechanism yields >=3% gain>", "niche_key": "(plane=<l1_kernel|l2_scheduler>,focus=<short_tag>)", "patch_summary": "<concise summary of code change>"}}
"""

    def _write_failure(self, evidence_dir: Path, error: Exception) -> None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "error": str(error)}, indent=2),
            encoding="utf-8",
        )

    def _propose_and_capture(
        self, candidate_dir: Path, prompt: str, evidence_dir: Path
    ) -> tuple[str, PatchStats]:
        if (
            hasattr(self.workspace, "isolated_proposer_sandbox")
            and hasattr(self.workspace, "capture_patch_stats")
            and hasattr(self.workspace, "apply_patch")
        ):
            with self.workspace.isolated_proposer_sandbox(candidate_dir) as sandbox_dir:
                proposer_out = self.proposer.propose(sandbox_dir, prompt, evidence_dir)
                patch_stats = self.workspace.capture_patch_stats(sandbox_dir)
            self.workspace.apply_patch(candidate_dir, patch_stats.patch)
            return proposer_out, patch_stats

        proposer_out = self.proposer.propose(candidate_dir, prompt, evidence_dir)
        raw_patch = self.workspace.capture_patch(candidate_dir)
        return proposer_out, PatchStats(
            patch=raw_patch,
            changed_lines=len(raw_patch.splitlines()),
        )

    def _run_screen_and_gate(
        self,
        candidate_id: str,
        baseline_hash: str,
        baseline_archive: Path,
        candidate_archive: Path,
        evidence_dir: Path,
    ) -> None:
        baseline_screen: Path | None = None
        if hasattr(self.remote, "screen_baseline"):
            baseline_screen = self.remote.screen_baseline(baseline_hash, baseline_archive)
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
                self.task,
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
            self.task,
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
                spec_pins = compute_spec_pins(candidate_dir, self.task.pinned_spec_files)
                baseline_archive = self.workspace.freeze_source(
                    candidate_dir, evidence_dir / "baseline.tar.gz"
                )
                baseline_hash = hashlib.sha256(baseline_archive.read_bytes()).hexdigest()
                proposer_out, patch_stats = self._propose_and_capture(
                    candidate_dir,
                    self._prompt(parent_id, candidate_dir, baseline_hash, blocked),
                    evidence_dir,
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
                    self.task,
                    evidence_dir,
                    expected_pins=spec_pins,
                )
                run_local_gates(candidate_dir, self.task, evidence_dir)
                stage = "preflight"
                self.remote.preflight()
                stage = "screen"
                self.archive.update_stage(candidate_id, "screen")
                self._run_screen_and_gate(
                    candidate_id,
                    baseline_hash,
                    baseline_archive,
                    candidate_archive,
                    evidence_dir,
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
                if stage == "preflight":
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

        baseline_hash = hashlib.sha256(baseline_archive.read_bytes()).hexdigest()
        changed_lines = int(row.get("changed_lines") or 0)
        stage = "preflight"
        try:
            self.remote.preflight()
            screen_metrics = evidence_dir / "screen-metrics.json"
            stage = "screen"
            if not screen_metrics.is_file():
                self._run_screen_and_gate(
                    candidate_id,
                    baseline_hash,
                    baseline_archive,
                    candidate_archive,
                    evidence_dir,
                )
            else:
                parity_path = evidence_dir / "screen-parity.json"
                screen_decision = evaluate_screen(
                    screen_metrics,
                    parity_path if parity_path.is_file() else None,
                    self.task,
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
