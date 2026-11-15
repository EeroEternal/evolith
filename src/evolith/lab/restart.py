"""Bounded restart diagnostics. Results cannot authorize a deployment or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from evolith.lab.engine import EngineConfig, EngineJob, runtime_identity, source_hash
from evolith.lab.engine_worker import StartupFailure, server
from evolith.lab.latency import (
    CHAT_PROMPTS,
    median_latencies,
    outputs_match,
    stream_request,
    workload_fingerprint,
)
from evolith.lab.store import RunStore, new_run_id, write_json


def comparison(first: list[dict], second: list[dict]) -> dict:
    details = []
    for index, (a, b) in enumerate(zip(first, second)):
        left, right = a["token_ids"], b["token_ids"]
        prefix = 0
        while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
            prefix += 1
        differences = [abs(x - y) for x, y in
                       zip(a["token_logprobs"][:prefix], b["token_logprobs"][:prefix])]
        finite = all(math.isfinite(value) for value in differences)
        details.append({
            "request": index, "same_ids": left == right,
            "first_different_token": None if left == right else prefix,
            "common_prefix_tokens": prefix,
            "max_common_prefix_logprob_difference": max(differences, default=0) if finite else None,
        })
    return {"correct": outputs_match(first, second),
            "request_counts": [len(first), len(second)], "requests": details}


def plan(policy: str) -> list[dict]:
    if policy not in ("native", "batch_invariant", "both", "collective", "no_aot"):
        raise ValueError("Unsupported restart diagnostic policy.")
    cases = []
    if policy in ("native", "both"):
        cases += [{"policy": "native", "prefix_request": prefix} for prefix in (False, False, True)]
    if policy in ("batch_invariant", "both"):
        cases += [{"policy": "batch_invariant", "prefix_request": prefix,
                   "attention_backend": "FLASH_ATTN"} for prefix in (False, True)]
    if policy == "collective":
        cases += [{"policy": mode, "prefix_request": prefix}
                  for mode in ("native_ring", "native_tree") for prefix in (False, True)]
    if policy == "no_aot":
        cases += [{"policy": "native_no_aot", "prefix_request": prefix}
                  for prefix in (False, False, True)]
    return cases


@contextmanager
def policy_environment(policy: str, cache_directory: Path | None = None):
    if policy not in ("native", "batch_invariant", "native_ring", "native_tree", "native_no_aot"):
        raise ValueError("Unsupported restart policy.")
    settings = {"VLLM_BATCH_INVARIANT": "1" if policy == "batch_invariant" else "0"}
    if policy in ("native_ring", "native_tree"):
        settings["NCCL_ALGO"] = "allreduce:" + policy.removeprefix("native_")
    if policy == "native_no_aot":
        settings["VLLM_USE_AOT_COMPILE"] = "0"
    if cache_directory is not None:
        if not cache_directory.is_absolute():
            raise ValueError("Diagnostic cache directory must be absolute.")
        settings.update({
            "TORCHINDUCTOR_CACHE_DIR": str(cache_directory / "inductor"),
            "TRITON_CACHE_DIR": str(cache_directory / "triton"),
            "VLLM_CACHE_ROOT": str(cache_directory / "vllm"),
        })
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def inspect_runtime(model_path: Path) -> tuple[dict, dict]:
    from evolith.lab.gpu import environment, model_info

    info, env = model_info(model_path), environment()
    if not env["cuda_available"] or info["model_type"] != "qwen3_5":
        raise ValueError("Restart diagnosis requires actual CUDA and Qwen3.5.")
    return info, env


def trial(model_path: Path, output: Path, case: dict, reference: dict | None) -> dict:
    run_id = new_run_id()
    directory = output / run_id
    directory.mkdir()
    job = EngineJob(model_path=model_path, output=output, config=EngineConfig(), max_tokens=64)
    report = {"run_id": run_id, "status": "running", **case,
              "config": job.config.model_dump(), "warmup_passes": [], "records": [],
              "self_consistent": None, "source_hash": source_hash(),
              "effective_environment": {name: os.environ.get(name) for name in (
                  "VLLM_BATCH_INVARIANT", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
                  "VLLM_ENABLE_V1_MULTIPROCESSING", "NCCL_ALGO", "NCCL_PROTO",
                  "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
                  "VLLM_USE_AOT_COMPILE", "VLLM_DISABLE_COMPILE_CACHE",
              )}}
    write_json(directory / "report.json", report)
    try:
        options = ({"attention_backend": case["attention_backend"]}
                   if "attention_backend" in case else {})
        with server(job, directory, **options) as client:
            if case["prefix_request"]:
                response = client.post("/v1/chat/completions", json={
                    "model": "evolith-qwen", "messages": [
                        {"role": "user", "content": "Explain KV Cache in one sentence."}],
                    "max_tokens": 32, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                response.raise_for_status()
            previous = None
            for _ in range(5):
                records = [stream_request(client, prompt, 64) for prompt in CHAT_PROMPTS]
                report["warmup_passes"].append(records)
                report["records"] = records
                if previous is not None and not outputs_match(previous, records):
                    report["self_consistent"] = False
                    report["status"] = "within_process_regression"
                    break
                if previous is not None:
                    report["self_consistent"] = True
                previous = records
            else:
                report["status"] = "observed"
            report["latency_medians_us"] = dict(zip(("ttft", "itl"), median_latencies(records)))
            if reference:
                report["frozen_native_comparison"] = comparison(reference["records"], records)
    except StartupFailure as exc:
        report.update(status="startup_failed", failure_kind=exc.kind, error=str(exc))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        raise
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        write_json(directory / "report.json", report)
        RunStore(output / "runs.sqlite3").record(run_id, "engine_restart_trial", report)
    return report


def run(model_path: Path, output: Path, *, policy: str = "both", seconds: int = 3600,
        reference_path: Path | None = None, isolated_cache_root: Path | None = None) -> dict:
    cases = plan(policy)
    if not 600 <= seconds <= 7200:
        raise ValueError("Restart diagnostic budget must be 600–7200 seconds.")
    if isolated_cache_root is not None and not isolated_cache_root.is_absolute():
        raise ValueError("Diagnostic cache root must be absolute.")
    output = output.resolve()
    run_id = new_run_id()
    directory = output / run_id
    directory.mkdir(parents=True)
    info, env = inspect_runtime(model_path)
    reference = json.loads(reference_path.read_text()) if reference_path else None
    if reference and any(reference.get(key) != value for key, value in {
        "source_hash": source_hash(), "model_fingerprint": info["fingerprint"],
        "environment": runtime_identity(env), "workload_fingerprint": workload_fingerprint(64),
        "config": EngineConfig().model_dump(),
    }.items()):
        raise ValueError("Frozen reference differs from the diagnostic contract.")
    cache_directories = {}
    if isolated_cache_root is not None:
        for case_policy in {case["policy"] for case in cases}:
            cache = isolated_cache_root / run_id / case_policy
            cache.mkdir(parents=True, exist_ok=False)
            cache_directories[case_policy] = cache
    report = {"run_id": run_id, "status": "running", "scope": "diagnostic only; no promotion or deployment",
              "plan": cases, "seconds": seconds, "trials": [], "comparisons": [], "skipped": [],
              "model_fingerprint": info["fingerprint"], "environment": env,
              "source_hash": source_hash(),
              "cache_scope": "isolated_per_policy" if cache_directories else "existing_shared_cache",
              "cache_directories": {key: str(value) for key, value in cache_directories.items()},
              "reference_run_id": reference.get("run_id") if reference else None,
              "probe_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("Restart diagnostic wall-time budget exhausted.")

    old_alarm = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    first_by_policy = {}
    unsupported = set()
    try:
        for case in cases:
            if case["policy"] in unsupported:
                report["skipped"].append(case | {"reason": "same policy already failed startup"})
                continue
            print(json.dumps({"event": "restart_probe", **case}), flush=True)
            with policy_environment(case["policy"], cache_directories.get(case["policy"])):
                observed = trial(model_path, output, case | {"session_run_id": run_id}, reference)
            report["trials"].append({"run_id": observed["run_id"], "status": observed["status"], **case})
            if observed["status"] == "startup_failed":
                unsupported.add(case["policy"])
                if case["policy"] == "native":
                    raise RuntimeError("Native diagnostic baseline cannot start.")
            elif observed["status"] == "observed":
                first = first_by_policy.setdefault(case["policy"], observed)
                if first is not observed:
                    report["comparisons"].append({
                        "policy": case["policy"], "first": first["run_id"], "second": observed["run_id"],
                        **comparison(first["records"], observed["records"]),
                    })
            write_json(directory / "report.json", report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        if report["status"] == "running":
            report["status"] = "interrupted"
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "report.json", report)
        RunStore(output / "runs.sqlite3").record(run_id, "engine_restart_session", report)
    return {"report": str(directory / "report.json"), "status": report["status"],
            "comparisons": report["comparisons"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("native", "batch_invariant", "both", "collective", "no_aot"),
                        default="both")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--isolated-cache-root", type=Path)
    args = parser.parse_args()

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(run(args.model_path, args.output, policy=args.policy,
                         seconds=args.seconds, reference_path=args.reference,
                         isolated_cache_root=args.isolated_cache_root), indent=2))


if __name__ == "__main__":
    main()
