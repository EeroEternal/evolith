"""Bounded cold/warm Inductor diagnosis. Cannot create an engine profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from evolith.lab.engine import MODEL_NAME, EngineConfig, EngineJob, source_hash
from evolith.lab.engine_worker import server
from evolith.lab.latency import CHAT_PROMPTS, outputs_match, stream_request
from evolith.lab.restart import comparison, inspect_runtime, policy_environment
from evolith.lab.store import RunStore, new_run_id, write_json


def trace_identity() -> str:
    digest = hashlib.sha256()
    for name in ("kernel_trace.py", "trace_worker.py", "kernel_diagnosis.py", "kernel_replay.py",
                 "rank_cache.py", "rank_worker.py", "kernel_repair.py", "kernel_replay_standalone.py",
                 "compiler_policy.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


@contextmanager
def trace_environment(directory: Path, enabled: bool):
    settings = {"EVOLITH_KERNEL_TRACE_DIR": str(directory),
                "EVOLITH_KERNEL_TRACE_MODE": "observe" if enabled else "off"}
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def case(model: Path, output: Path, *, enabled: bool, phase: str, session: str,
         run_id: str, replay_source: str | None = None, rank_local_cache: bool = False,
         deterministic_compiler: bool = False) -> dict:
    directory = output / run_id
    directory.mkdir()
    traces = directory / "trace"
    traces.mkdir()
    report = {"run_id": run_id, "session_run_id": session, "phase": phase, "status": "running",
              "trace_enabled": enabled, "rank_local_cache": rank_local_cache,
              "deterministic_compiler": deterministic_compiler,
              "source_hash": source_hash(),
              "trace_source_hash": trace_identity(), "warmup_passes": [], "records": [],
              "scope": "diagnostic only; no profile, performance admission or publication"}
    job = EngineJob(model_path=model, output=output, config=EngineConfig(), max_tokens=64)
    write_json(directory / "report.json", report)
    try:
        with trace_environment(traces, enabled), server(
            job, directory, diagnostic_worker=True, rank_local_cache=rank_local_cache,
            deterministic_compiler=deterministic_compiler,
        ) as client:
            for _ in range(5):
                records = [stream_request(client, prompt, 64) for prompt in CHAT_PROMPTS]
                if report["records"] and not outputs_match(report["records"], records):
                    raise ValueError("Within-process output regression during equilibration.")
                report["warmup_passes"].append(records)
                report["records"] = records
            if enabled:
                response = client.post("/tokenize", json={
                    "model": MODEL_NAME, "messages": [{"role": "user", "content": CHAT_PROMPTS[0]}],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                response.raise_for_status()
                prompt = response.json()
                tokens = prompt.get("tokens")
                if (not isinstance(tokens, list) or not tokens or len(tokens) > 2048
                        or any(type(token) is not int for token in tokens)
                        or prompt.get("count") != len(tokens)):
                    raise ValueError("No exact tokenized public prefill evidence.")
                report["prompt_token_ids"] = tokens
                write_json(traces / "arm.json", {"prompt_token_ids": tokens,
                                                "replay_source": replay_source})
                observed = stream_request(client, CHAT_PROMPTS[0], 64)
                report["sampled_request"] = observed
                report["sampled_request_matches"] = outputs_match([records[0]], [observed])
                if not report["sampled_request_matches"]:
                    raise ValueError("Value sampling changed the process's canonical output.")
                post = [stream_request(client, prompt, 64) for prompt in CHAT_PROMPTS]
                report["post_sampling_records"] = post
                if not outputs_match(records, post):
                    raise ValueError("Post-sampling output regression.")
                completions = [json.loads((traces / f"rank-{rank}-complete.json").read_text())
                               for rank in range(4)]
                report["capture_completions"] = completions
                if any(row["launches"] == 0 for row in completions):
                    raise ValueError("Inductor prefill trace is empty on at least one rank.")
                if replay_source:
                    report["paired_replays"] = [
                        json.loads((traces / f"replay-rank-{rank}.json").read_text())
                        for rank in range(4)
                    ]
        report["status"] = "observed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        raise
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        write_json(directory / "report.json", report)
        RunStore(output / "runs.sqlite3").record(run_id, "kernel_restart_trial", report)
    return report


def values_equal(first: dict, second: dict) -> bool:
    if first.keys() != second.keys() or not first:
        return False
    for key, a in first.items():
        b = second[key]
        if not a.get("complete") or not b.get("complete"):
            return False
        if "value_sha256" in a:
            if (not a.get("finite") or not b.get("finite")
                    or any(a.get(k) != b.get(k) for k in
                           ("shape", "stride", "dtype", "value_sha256"))):
                return False
        elif a != b:
            return False
    return True


def values_complete(values: dict) -> bool:
    return bool(values) and all(row.get("complete") and row.get("finite", True)
                                for row in values.values())


def compare_rank(first: list[dict], second: list[dict]) -> dict:
    a = [row for row in first if row["kind"] == "values"]
    b = [row for row in second if row["kind"] == "values"]
    result = {"observed_launches": [len(a), len(b)], "first_observed_output_difference": None,
              "first_observed_input_difference": None,
              "first_alignment_gap": None, "first_coverage_gap": None,
              "empty_output_events": [sum(not row["outputs"] for row in peer) for peer in (a, b)],
              "aligned_prefix_configuration_changes": []}
    inventories = []
    for rows in (first, second):
        inventory = {}
        for row in rows:
            if row["kind"] == "configuration" and row.get("source_sha256"):
                key = (row["name"], row["source_sha256"])
                inventory.setdefault(key, set()).add(json.dumps(row["configuration"], sort_keys=True))
        inventories.append(inventory)
    left_configs, right_configs = inventories
    shared = sorted(left_configs.keys() & right_configs.keys())
    changes = [key for key in shared if left_configs[key] != right_configs[key]]
    result["configuration_inventory"] = {
        "kernel_keys": [len(left_configs), len(right_configs)], "shared_keys": len(shared),
        "changed_shared_keys": len(changes),
        "changes": [{"name": key[0], "source_sha256": key[1],
                     "first": [json.loads(value) for value in sorted(left_configs[key])],
                     "second": [json.loads(value) for value in sorted(right_configs[key])]}
                    for key in changes[:50]],
    }
    for left, right in zip(a, b):
        identity = ("name", "source_sha256", "occurrence")
        if not left.get("source_sha256") or any(left.get(k) != right.get(k) for k in identity):
            result["first_alignment_gap"] = {"index": left["index"],
                                             "first": left["name"], "second": right["name"]}
            break
        changed = left["configuration"] != right["configuration"]
        if changed and len(result["aligned_prefix_configuration_changes"]) < 20:
            result["aligned_prefix_configuration_changes"].append({
                "index": left["index"], "name": left["name"],
                "source_sha256": left["source_sha256"],
                "first": left["configuration"], "second": right["configuration"],
            })
        different_inputs = [
            key for key, value in left["inputs"].items()
            if (peer := right["inputs"].get(key)) is not None
            and "float" in value.get("dtype", "")
            and values_complete({key: value}) and values_complete({key: peer})
            and all(value.get(k) == peer.get(k) for k in ("shape", "stride", "dtype"))
            and value.get("value_sha256") != peer.get("value_sha256")
        ]
        if different_inputs and result["first_observed_input_difference"] is None:
            result["first_observed_input_difference"] = {
                "index": left["index"], "name": left["name"],
                "source_sha256": left["source_sha256"], "arguments": different_inputs,
            }
        if (not left["outputs"] or not right["outputs"]) and result["first_coverage_gap"] is None:
            result["first_coverage_gap"] = {"index": left["index"], "reason": "missing_output_snapshots"}
        floating = {key: value for key, value in left["outputs"].items()
                    if "float" in value.get("dtype", "")}
        other = {key: value for key, value in right["outputs"].items()
                 if "float" in value.get("dtype", "")}
        if (floating or other) and (
            floating.keys() != other.keys() or not values_complete(floating)
            or not values_complete(other)
            or any(any(value.get(k) != other[key].get(k) for k in ("shape", "stride", "dtype"))
                   for key, value in floating.items())
        ):
            if result["first_coverage_gap"] is None:
                result["first_coverage_gap"] = {"index": left["index"],
                                                "reason": "incomplete_nonfinite_or_unaligned_output"}
            break
        if floating and not values_equal(floating, other):
            result["first_observed_output_difference"] = {
                "index": left["index"], "name": left["name"],
                "source_sha256": left["source_sha256"], "configuration_changed": changed,
                "inputs_fully_observed": (values_complete(left["inputs"])
                                          and values_complete(right["inputs"])),
                "all_observed_inputs_equal": values_equal(left["inputs"], right["inputs"]),
                "first_inputs": left["inputs"], "second_inputs": right["inputs"],
                "first_outputs": floating, "second_outputs": other,
            }
            break
    if result["first_alignment_gap"] is None and len(a) != len(b):
        result["first_alignment_gap"] = {"index": min(len(a), len(b)), "reason": "different_counts"}
    return result


def analyze(output: Path, cold: dict, warm: dict, control: dict) -> dict:
    ranks = {}
    for rank in range(4):
        records = []
        for trial in (cold, warm):
            path = output / trial["run_id"] / "trace" / f"rank-{rank}.jsonl"
            records.append([json.loads(line) for line in path.read_text().splitlines()])
        ranks[str(rank)] = compare_rank(*records)
    same_prompt = cold["prompt_token_ids"] == warm["prompt_token_ids"]
    control_matches = outputs_match(warm["records"], control["records"])
    output_comparison = comparison(cold["records"], warm["records"])
    controls = same_prompt and control_matches and not output_comparison["correct"]
    matched_divergence = any(
        (difference := rank["first_observed_output_difference"]) is not None
        and difference["inputs_fully_observed"] and difference["all_observed_inputs_equal"]
        and rank["first_coverage_gap"] is None and rank["first_alignment_gap"] is None
        for rank in ranks.values()
    )
    return {
        "cold_warm": output_comparison,
        "warm_no_hook_control": comparison(warm["records"], control["records"]),
        "same_prefill_token_ids": same_prompt, "ranks": ranks,
        "reproduction_controls_passed": controls,
        "matched_input_output_divergence": matched_divergence,
        "attribution_eligible": controls and matched_divergence,
        "limitations": [
            "Only Inductor launch arguments during one fixed prefill are sampled.",
            "cuBLAS, attention, NCCL and other custom operators are not fully observed.",
            "First observed difference is not necessarily the first model-layer difference.",
            "Sampling synchronizes readbacks; equality checks bound but do not eliminate perturbation.",
            "Incomplete snapshots, alignment gaps or unobserved inputs prevent causal attribution.",
        ],
    }


def run(model: Path, output: Path, cache: Path, seconds: int = 1680) -> dict:
    if not 600 <= seconds <= 1680 or not output.is_absolute() or not cache.is_absolute():
        raise ValueError("Absolute paths and a 600–1680 second budget are required.")
    run_id = new_run_id()
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    cache = cache / run_id
    cache.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "status": "running", "trials": [], "seconds": seconds,
              "source_hash": source_hash(), "trace_source_hash": trace_identity(),
              "cache_directory": str(cache), "scope": "diagnostic only; no profile or publication"}
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("Kernel diagnosis wall-time budget exhausted.")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    write_json(directory / "report.json", report)
    try:
        info, env = inspect_runtime(model)
        report.update(model_fingerprint=info["fingerprint"], environment=env)
        observations = []
        with policy_environment("native", cache):
            for phase, enabled in (("cold_observed", True), ("warm_observed", True),
                                   ("warm_no_hook_control", False)):
                if source_hash() != report["source_hash"] or trace_identity() != report["trace_source_hash"]:
                    raise ValueError("Source changed during the diagnostic.")
                print(json.dumps({"event": "kernel_diagnosis", "phase": phase}), flush=True)
                entry = {"run_id": new_run_id(), "phase": phase, "status": "running"}
                report["trials"].append(entry)
                write_json(directory / "report.json", report)
                observed = case(model, output, enabled=enabled, phase=phase, session=run_id,
                                run_id=entry["run_id"])
                observations.append(observed)
                entry["status"] = observed["status"]
                write_json(directory / "report.json", report)
        report["analysis"] = analyze(output, *observations)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        for entry in report["trials"]:
            if entry["status"] == "running":
                entry["status"] = "failed"
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if report["status"] == "running":
            report["status"] = "interrupted"
        for entry in report["trials"]:
            if entry["status"] == "running":
                entry["status"] = "interrupted"
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "report.json", report)
        RunStore(output / "runs.sqlite3").record(run_id, "kernel_restart_session", report)
    return {"report": str(directory / "report.json"), "status": report["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(run(args.model_path, args.output, args.cache_root), indent=2))


if __name__ == "__main__":
    main()
