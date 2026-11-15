"""Bounded, opt-in rank-cache repair validation. Never creates an engine profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

from evolith.lab.compiler_policy import reduction_policy_inventory
from evolith.lab.engine import source_hash
from evolith.lab.kernel_diagnosis import case, trace_identity
from evolith.lab.restart import comparison, inspect_runtime, policy_environment
from evolith.lab.store import RunStore, new_run_id, write_json


def remaining_seconds(budget: dict, now: float) -> int:
    if (budget.get("gpus") != [0, 1, 2, 3] or budget.get("approved_seconds") != 1800
            or budget["deadline_unix"] - budget["started_unix"] != 1740
            or now < budget["started_unix"]):
        raise ValueError("Invalid inherited repair budget.")
    remaining = int(budget["deadline_unix"] - now - 40)
    if remaining < 600:
        raise TimeoutError("Insufficient time for an independently restarted repair pair.")
    return remaining


def _seed_autotuning(source: Path, target: Path) -> dict:
    """Copy recorded Triton autotune decisions, and nothing else, into a fresh cache."""
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files or target.exists():
        raise ValueError("Autotune seeding requires recorded decisions and a fresh cache.")
    seeded = {}
    for path in files:
        relative = path.relative_to(source)
        if len(relative.parts) != 2 or not relative.name.endswith(".autotune.json"):
            raise ValueError(f"Autotune seed may only contain decisions: {relative}.")
        decision = json.loads(path.read_text())
        if not isinstance(decision.get("key"), list) or not decision.get("configs_timings"):
            raise ValueError(f"Malformed autotune decision: {relative}.")
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / relative)
        seeded[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return seeded


def _autotune_state(target: Path, seeded: dict) -> dict:
    for relative, digest in seeded.items():
        if hashlib.sha256((target / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Seeded autotune decision changed during the run: {relative}.")
    unseeded = sorted(str(path.relative_to(target)) for path in target.rglob("*.autotune.json")
                      if str(path.relative_to(target)) not in seeded)
    return {"seeded_unchanged": True, "unseeded_decisions": unseeded}


def run(output: Path, cache: Path, budget_path: Path, probe_path: Path, frozen: Path,
        policy: str = "rank_local_cache", autotune_seed: Path | None = None) -> dict:
    if policy not in ("rank_local_cache", "deterministic_compiler"):
        raise ValueError("Unsupported diagnostic repair policy.")
    rank_local = policy == "rank_local_cache"
    if autotune_seed is not None and rank_local:
        raise ValueError("Autotune seeding is only defined for the compiler policy.")
    budget = json.loads(budget_path.read_text())
    seconds = remaining_seconds(budget, time.time())
    probe = json.loads(probe_path.read_text())
    replays = probe.get("paired_replays", [])
    if (probe["status"] != "verified" or len(replays) != 4
            or not all(row["selected_replay_matches_actual"] for row in replays)
            or not any(not row["paired_outputs_equal"] for row in replays)):
        raise ValueError("No valid real-input proof of configuration-sensitive RMSNorm output.")
    run_id = new_run_id()
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    namespace = cache / run_id
    namespace.mkdir(parents=True, exist_ok=False)
    seeded = _seed_autotuning(autotune_seed, namespace / "triton") if autotune_seed else {}
    report = {"run_id": run_id, "status": "running", "trials": [], "policy": policy,
              "source_hash": source_hash(), "trace_source_hash": trace_identity(),
              "budget": budget, "remaining_seconds_at_start": seconds,
              "probe_report": str(probe_path), "paired_replays": replays,
              "cache_namespace": str(namespace), "scope": "diagnostic repair only; no admission",
              "autotune_seed": {"source": str(autotune_seed), "files": seeded} if seeded else None}
    old_root = os.environ.get("EVOLITH_RANK_CACHE_ROOT")
    old_alarm = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("Original repair-session deadline exhausted.")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    write_json(directory / "report.json", report)
    try:
        info, env = inspect_runtime(Path("/models/qwen"))
        report.update(model_fingerprint=info["fingerprint"], environment=env)
        records = []
        with policy_environment("native", namespace):
            if rank_local:
                os.environ["EVOLITH_RANK_CACHE_ROOT"] = str(namespace / "rank-local")
            for restart in ("cold", "warm"):
                phase = f"{policy}_{restart}_no_hook"
                if source_hash() != report["source_hash"] or trace_identity() != report["trace_source_hash"]:
                    raise ValueError("Repair source changed during the run.")
                entry = {"run_id": new_run_id(), "phase": phase, "status": "running"}
                report["trials"].append(entry)
                write_json(directory / "report.json", report)
                print(json.dumps({"event": "repair_trial", **entry}), flush=True)
                trial = case(Path("/models/qwen"), output, enabled=False, phase=phase,
                             session=run_id, run_id=entry["run_id"], rank_local_cache=rank_local,
                             deterministic_compiler=not rank_local)
                records.append(trial["records"])
                entry["status"] = trial["status"]
                for rank in range(4):
                    trace = output / trial["run_id"] / "trace"
                    compiler = json.loads((trace / f"rank-{rank}-compiler.json").read_text())
                    if (compiler["rank"] != rank or compiler["requested"] is not (not rank_local)
                            or compiler["inductor_deterministic"] is not (not rank_local)):
                        raise ValueError("Worker did not confirm requested compiler policy.")
                    if not rank_local:
                        continue
                    path = trace / f"rank-{rank}-cache.json"
                    ownership = json.loads(path.read_text())
                    if ownership["rank"] != rank or any(
                        str(namespace / "rank-local" / f"rank-{rank}") not in value
                        for value in ownership["environment"].values()
                    ):
                        raise ValueError("Worker did not confirm isolated cache ownership.")
                entry["compiled_reductions"] = reduction_policy_inventory(namespace, not rank_local)
                if seeded:
                    entry["autotune"] = _autotune_state(namespace / "triton", seeded)
                write_json(directory / "report.json", report)
        report["cold_warm"] = comparison(*records)
        historical = json.loads(frozen.read_text())
        report["historical_values"] = {
            "reference_run_id": historical["run_id"], "admission_compatible": False,
            "scope": "descriptive cross-source value comparison; frozen artifact is not changed",
            "comparisons": [comparison(historical["records"], values) for values in records],
        }
        report["status"] = ("restart_pair_passed" if report["cold_warm"]["correct"]
                            else "repair_rejected")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        if old_root is None:
            os.environ.pop("EVOLITH_RANK_CACHE_ROOT", None)
        else:
            os.environ["EVOLITH_RANK_CACHE_ROOT"] = old_root
        for entry in report["trials"]:
            if entry["status"] == "running":
                entry["status"] = "interrupted" if report["status"] == "running" else "failed"
        if report["status"] == "running":
            report["status"] = "interrupted"
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "report.json", report)
        RunStore(output / "runs.sqlite3").record(run_id, "numerical_repair_session", report)
    return {"report": str(directory / "report.json"), "status": report["status"]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("output", "cache", "budget", "probe", "frozen"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--policy", choices=("rank_local_cache", "deterministic_compiler"),
                        default="rank_local_cache")
    args = parser.parse_args()

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(run(args.output, args.cache, args.budget, args.probe, args.frozen,
                         args.policy), indent=2))


if __name__ == "__main__":
    main()
