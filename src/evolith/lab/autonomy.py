"""Measured target selection and bounded autonomous orchestration, with no GPU imports."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from evolith.lab.contracts import (
    KernelConfig,
    Manifest,
    Proposal,
    Shape,
    StrictModel,
    Timing,
)
from evolith.lab.measurement import decide
from evolith.lab.observation import Observation, ObservedTarget, session_identity
from evolith.lab.search import WorkerTimeout, execute_request
from evolith.lab.store import RunStore, new_run_id, write_json


class Budget(StrictModel):
    cycles: int = Field(default=3, ge=1, le=8)
    evaluations: int = Field(default=32, ge=2, le=128)
    seconds: int = Field(default=1800, ge=30, le=7200)
    candidates_per_cycle: int = Field(default=4, ge=1, le=12)


class StageJob(StrictModel):
    kind: Literal["observe", "search", "verify"]
    model_path: Path
    output: Path
    tensor_parallel: int = Field(default=4, ge=1, le=8)
    max_tokens: int = Field(default=16, ge=1, le=256)
    workload: Path | None = None
    baseline_manifest: Path | None = None
    manifest: Path | None = None
    target: Shape | None = None
    proposals: Path | None = None
    history: list[dict] = Field(default_factory=list, max_length=4)
    rounds: int = Field(default=19, ge=5, le=101)
    iterations: int = Field(default=128, ge=1, le=1000)
    seed: int = Field(default=2026, ge=0)
    worker_timeout: float = Field(default=300, gt=0, le=7200)


def configuration_space() -> list[KernelConfig]:
    return [KernelConfig(num_warps=w, rows_per_program=r)
            for w, r in itertools.product((4, 8, 16), (1, 2, 4, 8))]


def select_target(observation: Observation, attempted: dict[str, set[str]],
                  priors: dict[str, int]) -> ObservedTarget | None:
    space = {c.key for c in configuration_space()}
    eligible = [
        t for t in observation.targets
        if t.eligible and t.estimated_rank_total_us > 0 and space - attempted.get(t.key, set())
    ]
    return max(eligible, key=lambda t: (
        t.estimated_rank_total_us / (1 + len(attempted.get(t.key, set())) + priors.get(t.key, 0)),
        t.key,
    ), default=None)


def novel_configs(proposal: Proposal, attempted: set[str], limit: int) -> list[KernelConfig]:
    choices = {c.key: c for c in [*proposal.candidates, *configuration_space()]
               if c.key not in attempted}
    return list(choices.values())[:limit]


def compose(candidate: Manifest, incumbent: Manifest | None, output: Path) -> Path:
    if incumbent is None:
        return output / candidate.run_id / "manifest.json"
    # Dynamic telemetry is not a compatibility requirement.
    if (candidate.model_fingerprint, candidate.kernel_hash, candidate.eps) != (
        incumbent.model_fingerprint, incumbent.kernel_hash, incumbent.eps,
    ) or any(candidate.environment.get(k) != incumbent.environment.get(k)
             for k in ("gpu", "capability", "packages", "cuda")):
        raise ValueError("Cannot compose incompatible artifacts.")
    merged = candidate.model_copy(deep=True)
    merged.run_id = new_run_id()
    merged.evaluations = incumbent.evaluations + candidate.evaluations
    merged.winners = incumbent.winners | candidate.winners
    merged.measurement["lineage"] = [incumbent.run_id, candidate.run_id]
    merged.measurement["composition"] = "Existing confirmations retained; not a new measurement."
    path = output / merged.run_id / "manifest.json"
    write_json(path, merged.model_dump())
    RunStore(output / "runs.sqlite3").record(merged.run_id, "kernel_composition", merged.model_dump())
    return path


def run(
    model_path: Path, output: Path, *, budget: Budget | None = None,
    tensor_parallel: int = 4, max_tokens: int = 16, workload: Path | None = None,
    rounds: int = 19, iterations: int = 128, seed: int = 2026,
) -> dict:
    budget = budget or Budget()
    started = time.monotonic()
    deadline = started + budget.seconds
    output = output.resolve()
    run_id = new_run_id()
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    store = RunStore(output / "runs.sqlite3")
    report = {
        "run_id": run_id, "status": "running", "budget": budget.model_dump(),
        "cycles": [], "evaluations": 0, "approved_manifest": None,
        "deployment_approved": False, "stop_reason": None,
        "scope": "Autonomous target/configuration selection within the observed trusted capability.",
    }
    attempted: dict[str, set[str]] = {}
    priors: dict[str, int] = {}
    history: list[dict] = []
    identity = None
    incumbent = None

    def stage(job: StageJob, path: Path) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeout("Controller wall-time budget exhausted.")
        result = execute_request(
            "evolith.lab.autonomy_worker", job.model_dump(mode="json"), path, remaining,
        )
        if not isinstance(result.get("result"), dict):
            raise TypeError("Invalid autonomous worker response.")
        return result["result"]

    base = {
        "model_path": model_path.resolve(), "output": output, "tensor_parallel": tensor_parallel,
        "max_tokens": max_tokens, "workload": workload.resolve() if workload else None,
        "rounds": rounds, "iterations": iterations,
    }

    try:
        for index in range(budget.cycles):
            if report["evaluations"] + 2 > budget.evaluations:
                report["stop_reason"] = "evaluation_budget"
                break
            cycle_dir = directory / f"cycle-{index:03d}"
            proposal_path = cycle_dir / "model-proposals.json"
            current = report["approved_manifest"]
            observed = stage(StageJob(
                **base, kind="observe", proposals=proposal_path,
                baseline_manifest=current, history=history[-4:],
            ), cycle_dir / "observe")
            observation_report = json.loads(Path(observed["report"]).read_text())
            if observed["status"] != "baseline_measured" or not observation_report["correctness"]["passed"]:
                raise RuntimeError("Observation baseline failed output regression.")
            observation = Observation.model_validate(observation_report["observation"])
            report["capability_gaps"] = [
                t.model_dump() for t in observation.targets if not t.eligible
            ]
            env = observation_report["environment"]
            current_identity = session_identity(
                observation.model_fingerprint, observation.workload_fingerprint,
                observation_report["kernel_hash"], env, tensor_parallel,
            )
            if identity is None:
                identity = current_identity
                report["identity"] = identity
                prior = [row for row in store.recent("autonomous_cycle")
                         if row.get("identity") == identity]
                history = [row["feedback"] for row in reversed(prior)]
                for row in prior:
                    key = row["target_key"]
                    priors[key] = priors.get(key, 0) + 1
            elif identity != current_identity:
                raise ValueError("Model, workload or runtime changed during the session.")
            if observation.dropped_contract_calls:
                report["stop_reason"] = "observation_capacity_gap"
                break
            target = select_target(observation, attempted, priors)
            if target is None:
                report["stop_reason"] = (
                    "search_space_exhausted" if any(t.eligible for t in observation.targets)
                    else "capability_gap"
                )
                report["capability_gaps"] = [t.model_dump() for t in observation.targets if not t.eligible]
                break
            proposal = Proposal.model_validate_json(proposal_path.read_text())
            configs = novel_configs(
                proposal, attempted.get(target.key, set()),
                min(budget.candidates_per_cycle, (budget.evaluations - report["evaluations"]) // 2),
            )
            selected_path = cycle_dir / "selected-proposals.json"
            write_json(selected_path, Proposal(candidates=configs).model_dump())
            cycle = {
                "cycle": index, "identity": identity, "target_key": target.key,
                "target": target.model_dump(), "configs": [c.key for c in configs],
                "observation_report": observed["report"],
                "capability_gaps": report["capability_gaps"],
                "selection": "Estimated per-rank cumulative cost / (1 + session attempts + historical cycles).",
                "proposal_policy": "Model suggestions first; trusted finite-space exploration fills unseen slots.",
                "baseline_manifest": current, "approved": False,
            }
            report["cycles"].append(cycle)
            write_json(directory / "report.json", report)
            searched = stage(StageJob(
                **base, kind="search", target=target.shape, proposals=selected_path,
                seed=seed + index * 1009,
                worker_timeout=min(300, max(1, deadline - time.monotonic())),
            ), cycle_dir / "search")
            manifest = Manifest.model_validate_json(Path(searched["manifest"]).read_text())
            if (manifest.model_fingerprint != observation.model_fingerprint
                    or manifest.kernel_hash != identity["kernel_hash"]
                    or len(manifest.evaluations) != searched["evaluations"]
                    or len(manifest.evaluations) > 2 * len(configs)
                    or any(e.shape != target.shape or e.config not in configs for e in manifest.evaluations)
                    or any(key != target.shape.key for key in manifest.winners)):
                raise ValueError("Search artifact does not match the selected measured target.")
            report["evaluations"] += len(manifest.evaluations)
            attempted.setdefault(target.key, set()).update(c.key for c in configs)
            cycle["search"] = searched
            feedback = {
                "target": target.shape.key, "configs": cycle["configs"],
                "micro_winners": len(manifest.winners), "outcome": "no_confirmed_gain",
            }
            if searched["status"] == "failed" or searched["worker_errors"]:
                raise RuntimeError("Trusted search worker failed; refusing to learn from incomplete execution.")
            if manifest.winners:
                candidate_path = compose(manifest, incumbent, output)
                verified = stage(StageJob(
                    **base, kind="verify", manifest=candidate_path,
                    baseline_manifest=current, target=target.shape,
                ), cycle_dir / "verify")
                proof = json.loads(Path(verified["report"]).read_text())
                cycle["verification"] = verified
                feedback["outcome"] = "end_to_end_rejected"
                feedback["paired_speedup"] = proof.get("observed_paired_e2e_speedup")
                feedback["timing_reasons"] = proof.get("e2e_measurement", {}).get("reasons", [])
                candidate = Manifest.model_validate_json(candidate_path.read_text())
                if (proof.get("workload_fingerprint") != observation.workload_fingerprint
                        or proof.get("kernel_run_id") != candidate.run_id
                        or proof.get("baseline_kernel_run_id") != (incumbent.run_id if incumbent else None)
                        or proof.get("kernel_hash") != identity["kernel_hash"]
                        or proof.get("model", {}).get("fingerprint") != observation.model_fingerprint
                        or proof.get("required_candidate_shapes") != [target.shape.key]):
                    raise ValueError("Verification provenance differs from the requested comparison.")
                timing = Timing.model_validate(proof["e2e_measurement"]) if proof.get("e2e_measurement") else None
                independently_accepted = timing is not None and decide(
                    timing.baseline_us, timing.candidate_us, seed=2026,
                ).accepted
                if (proof.get("status") == "verified" and proof.get("deployment_approved")
                        and proof.get("correctness", {}).get("passed")
                        and proof.get("adapter_exercised")
                        and proof.get("e2e_measurement", {}).get("accepted")
                        and independently_accepted
                        and verified.get("deployment_approved")):
                    incumbent = Manifest.model_validate_json(candidate_path.read_text())
                    report["approved_manifest"] = str(candidate_path)
                    report["deployment_approved"] = True
                    cycle["approved"] = True
                    feedback["outcome"] = "approved_against_incumbent"
            cycle["feedback"] = feedback
            history.append(feedback)
            cycle_id = f"{run_id}-cycle-{index:03d}"
            write_json(cycle_dir / "report.json", cycle)
            store.record(cycle_id, "autonomous_cycle", cycle)
            write_json(directory / "report.json", report)
        else:
            report["stop_reason"] = "cycle_budget"
        report["status"] = "stopped"
    except WorkerTimeout as exc:
        report["status"], report["stop_reason"] = "stopped", "wall_time_budget"
        report["error"] = str(exc)
    except Exception as exc:
        report["status"], report["stop_reason"] = "failed", "execution_failure"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
        raise
    finally:
        report["elapsed_sec"] = time.monotonic() - started
        write_json(directory / "report.json", report)
        store.record(run_id, "autonomous_session", report)
    return {"status": report["status"], "run_id": run_id, "report": str(directory / "report.json"),
            "stop_reason": report["stop_reason"], "evaluations": report["evaluations"],
            "deployment_approved": report["deployment_approved"],
            "approved_manifest": report["approved_manifest"]}
