"""Workload-driven engine configuration evolution with a strong native reference."""

from __future__ import annotations

import json
import time
from pathlib import Path

from evolith.lab.engine import (
    EngineArtifact,
    EngineConfig,
    EngineJob,
    EngineProposal,
    capacity_neighbors,
    source_hash,
    validate_artifact,
)
from evolith.lab.latency import compare_trials, outputs_match, workload_fingerprint
from evolith.lab.search import WorkerTimeout, execute_request
from evolith.lab.store import RunStore, new_run_id, write_json


def combined(first: dict, second: dict) -> dict:
    if (any(not isinstance(proof.get("run_id"), str) or not proof["run_id"]
            for proof in (first, second))
            or first["run_id"] == second["run_id"]):
        raise ValueError("Independent engine proof requires distinct trial IDs.")
    if not outputs_match(first["records"], second["records"]):
        raise ValueError("Independent engine restarts produced inconsistent outputs.")
    for key in ("config", "source_hash", "model_fingerprint", "environment", "workload_fingerprint"):
        if first[key] != second[key]:
            raise ValueError("Independent confirmation has incompatible provenance.")
    return first | {"ttft_us": first["ttft_us"] + second["ttft_us"],
                    "itl_us": first["itl_us"] + second["itl_us"],
                    "rounds": first["rounds"] + second["rounds"]}


def run(model_path: Path, output: Path, *, candidates: int = 2,
        seconds: int = 7200, rounds: int = 5, max_tokens: int = 64) -> dict:
    if not 1 <= candidates <= 4 or not 600 <= seconds <= 14400:
        raise ValueError("Engine budget out of bounds.")
    output = output.resolve()
    run_id = new_run_id()
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + seconds
    store = RunStore(output / "runs.sqlite3")
    report = {"run_id": run_id, "status": "running", "cycles": [],
              "objective": "Single-chat TTFT/ITL Pareto improvement with fixed BF16 weights/context",
              "profile": None, "promotions": 0}
    feedback = []
    index = 0

    def stage(config, phase, *, reference=None, propose=True, fresh=False):
        nonlocal index
        index += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeout("Engine evolution wall-time budget exhausted.")
        job = EngineJob(
            model_path=model_path.resolve(), output=output, config=config, rounds=rounds,
            max_tokens=max_tokens, seed=2031 + (100003 if fresh else 0), propose=propose,
            history=feedback[-8:], reference_report=reference,
        )
        print(json.dumps({"event": phase, "config": config.key}), flush=True)
        response = execute_request(
            "evolith.lab.engine_worker", job.model_dump(mode="json"),
            directory / f"{index:02d}-{phase}", remaining,
        )["result"]
        path = Path(response["report"])
        trial = json.loads(path.read_text())
        if (trial["config"] != config.model_dump() or trial["source_hash"] != source_hash()
                or trial["workload_fingerprint"] != workload_fingerprint(max_tokens)):
            raise ValueError("Engine worker returned mismatched provenance.")
        if trial["status"] == "measured" and (
            not trial["correct"] or len(trial["ttft_us"]) != rounds or len(trial["itl_us"]) != rounds
        ):
            raise ValueError("Incomplete engine measurement.")
        return trial, path

    def save_selection(trial, selection, baseline_proof=None, candidate_proof=None,
                       fresh_baseline=None, fresh_candidate=None):
        artifact = EngineArtifact(
            config=trial["config"], source_hash=trial["source_hash"],
            model_fingerprint=trial["model_fingerprint"], environment=trial["environment"],
            workload_fingerprint=trial["workload_fingerprint"], selection=selection,
            baseline_proof=baseline_proof, candidate_proof=candidate_proof,
            fresh_baseline_proof=fresh_baseline, fresh_candidate_proof=fresh_candidate,
        )
        validate_artifact(artifact)
        write_json(directory / "engine.json", artifact.model_dump())
        report["profile"] = str(directory / "engine.json")

    try:
        incumbent, incumbent_path = stage(EngineConfig(), "native-baseline")
        if incumbent["status"] != "measured":
            raise RuntimeError(f"Strong native baseline failed: {incumbent_path}")
        report["native_baseline"] = str(incumbent_path)
        restarted, restarted_path = stage(
            EngineConfig(), "certify-native-restart", propose=False, fresh=True,
        )
        report["native_restart_confirmation"] = str(restarted_path)
        if restarted["status"] != "measured":
            raise RuntimeError(f"Native restart certification failed: {restarted_path}")
        combined(incumbent, restarted)
        save_selection(incumbent, "native_baseline", baseline_proof=incumbent,
                       fresh_baseline=restarted)
        queue = EngineProposal.model_validate(incumbent["proposals"]).candidates
        origins = {config.key: "local_model" for config in queue}
        attempted = {EngineConfig().key}
        for cycle_index in range(candidates):
            queue = list({c.key: c for c in queue if c.key not in attempted}.values())
            if not queue:
                report["stop_reason"] = "no_novel_model_proposal"
                break
            config = queue.pop(0)
            attempted.add(config.key)
            trial, trial_path = stage(config, "candidate", reference=incumbent_path)
            cycle = {"cycle": cycle_index, "config": config.model_dump(),
                     "candidate_report": str(trial_path), "incumbent_report": str(incumbent_path),
                     "accepted": False, "proposal_source": origins.get(config.key, "local_model")}
            report["cycles"].append(cycle)
            if trial["status"] != "measured":
                cycle["outcome"] = "startup_failed"
                feedback.append({"config": config.key, "outcome": "startup_failed",
                                 "failure_kind": trial.get("failure_kind"),
                                 "reason": trial.get("error")})
                if trial.get("failure_kind") == "capacity":
                    repairs = capacity_neighbors(config)
                    for repair in repairs:
                        origins[repair.key] = "observed_capacity_constraint_solver"
                    queue = repairs + queue
                write_json(directory / f"cycle-{cycle_index:02d}.json", cycle)
                store.record(f"{run_id}-cycle-{cycle_index:02d}", "engine_cycle", cycle)
                write_json(directory / "report.json", report)
                continue
            if any(trial[key] != incumbent[key] for key in
                   ("model_fingerprint", "environment", "source_hash", "workload_fingerprint")):
                raise ValueError("Candidate changes the fixed model/runtime/workload.")
            pilot = compare_trials(incumbent, trial)
            cycle["pilot"] = pilot
            cycle["outcome"] = "pilot_rejected"
            feedback.append({
                "config": config.key, "outcome": cycle["outcome"],
                "ttft_speedup": pilot["ttft"]["speedup"], "itl_speedup": pilot["itl"]["speedup"],
            })
            if pilot["accepted"]:
                # AB/BA blocks: A0 above, B0 above, then fresh B1 and A1.
                second_trial, second_path = stage(config, "confirm-candidate", propose=False, fresh=True)
                second_base, second_base_path = stage(
                    EngineConfig.model_validate(incumbent["config"]), "confirm-incumbent", fresh=True,
                )
                if second_trial["status"] != "measured" or second_base["status"] != "measured":
                    raise RuntimeError("Independent engine confirmation failed.")
                a, b = combined(incumbent, second_base), combined(trial, second_trial)
                proof = compare_trials(a, b)
                fresh_proof = compare_trials(second_base, second_trial)
                cycle["confirmation"] = proof
                cycle["fresh_confirmation"] = fresh_proof
                cycle["confirmation_reports"] = [str(second_path), str(second_base_path)]
                cycle["outcome"] = "confirmation_rejected"
                if proof["accepted"] and fresh_proof["accepted"]:
                    save_selection(second_trial, "approved_successor", a, b, second_base, second_trial)
                    incumbent, incumbent_path = second_trial, second_path
                    report["promotions"] += 1
                    cycle["accepted"], cycle["outcome"] = True, "approved_successor"
                feedback[-1]["outcome"] = cycle["outcome"]
                if second_base.get("proposals"):
                    queue = EngineProposal.model_validate(second_base["proposals"]).candidates + queue
            elif trial.get("proposals"):
                queue = EngineProposal.model_validate(trial["proposals"]).candidates + queue
            write_json(directory / f"cycle-{cycle_index:02d}.json", cycle)
            store.record(f"{run_id}-cycle-{cycle_index:02d}", "engine_cycle", cycle)
            write_json(directory / "report.json", report)
        else:
            report["stop_reason"] = "candidate_budget"
        report["selected_config"] = incumbent["config"]
        report["status"] = "completed"
    except WorkerTimeout as exc:
        report["status"], report["stop_reason"] = "stopped", "wall_time_budget"
        report["error"] = str(exc)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        raise
    finally:
        write_json(directory / "report.json", report)
        store.record(run_id, "engine_session", report)
    return {"status": report["status"], "report": str(directory / "report.json"),
            "profile": report["profile"], "promotions": report["promotions"],
            "selected_config": report.get("selected_config"), "stop_reason": report.get("stop_reason")}
