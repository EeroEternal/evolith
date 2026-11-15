from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from evolith.lab.engine import (
    EngineConfig,
    EngineJob,
    EngineProposal,
    runtime_identity,
    server_command,
    source_hash,
)
from evolith.lab.latency import (
    CHAT_PROMPTS,
    compare_trials,
    local_client,
    median_latencies,
    outputs_match,
    stream_request,
    workload_fingerprint,
)
from evolith.lab.store import RunStore, new_run_id, write_json

PORT = 18111


class StartupFailure(RuntimeError):
    def __init__(self, message: str, kind: str = "startup"):
        super().__init__(message)
        self.kind = kind


def capacity_failure(log_path: Path) -> str | None:
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - 16384))
        tail = stream.read().decode(errors="replace")
    for marker in ("No available memory for the cache blocks", "CUDA out of memory",
                   "not enough KV cache memory"):
        if marker in tail:
            return marker
    return None


@contextmanager
def server(job: EngineJob, directory: Path, *, attention_backend: str | None = None,
           diagnostic_worker: bool = False, rank_local_cache: bool = False,
           deterministic_compiler: bool = False):
    if attention_backend not in (None, "FLASH_ATTN"):
        raise ValueError("Unsupported diagnostic attention backend.")
    if type(diagnostic_worker) is not bool:
        raise ValueError("Diagnostic worker selection must be boolean.")
    if type(rank_local_cache) is not bool or (rank_local_cache and not diagnostic_worker):
        raise ValueError("Rank-local repair requires the explicit diagnostic worker.")
    if (type(deterministic_compiler) is not bool
            or (deterministic_compiler and not diagnostic_worker)):
        raise ValueError("Compiler repair requires the explicit diagnostic worker.")
    env = os.environ.copy()
    # Set before the fresh server interpreter imports Torch, also covering its workers.
    env["TORCHINDUCTOR_DETERMINISTIC"] = "1" if deterministic_compiler else "0"
    command = server_command(job.config, job.model_path, PORT)
    if deterministic_compiler:
        # vLLM forwards this dictionary to standalone_compile(options.config_patches).
        # The import-time environment alone did not survive the real compile path.
        if "--compilation-config" in command:
            index = command.index("--compilation-config") + 1
            compilation = json.loads(command[index])
        else:
            command += ["--compilation-config", "{}"]
            index = len(command) - 1
            compilation = {}
        compilation.setdefault("inductor_compile_config", {}).update(
            deterministic=True, combo_kernels=False,
        )
        command[index] = json.dumps(compilation)
    if attention_backend is not None:
        command += ["--attention-backend", attention_backend]
    if diagnostic_worker:
        worker = ("evolith.lab.rank_worker.RankLocalWorker" if rank_local_cache
                  else "evolith.lab.trace_worker.KernelTraceWorker")
        command += ["--worker-cls", worker]
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env,
        )
        try:
            with local_client(PORT) as client:
                deadline = time.monotonic() + 600
                while True:
                    capacity = capacity_failure(directory / "server.log")
                    if capacity:
                        raise StartupFailure(capacity, "capacity")
                    if process.poll() is not None:
                        raise StartupFailure(f"API startup exited {process.returncode}; see server.log")
                    try:
                        if client.get("/health", timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise StartupFailure("API startup deadline exceeded.")
                    time.sleep(0.5)
                yield client
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def proposal(client, report: dict, history: list[dict]) -> EngineProposal:
    context = {
        "current_config": report["config"],
        "ttft_us": report["ttft_us"], "itl_us": report["itl_us"],
        "recent_outcomes": history[-6:],
    }
    response = client.post("/v1/chat/completions", json={
        "model": "evolith-qwen", "temperature": 0, "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
            "Propose at most two runtime configurations to reduce chat TTFT and inter-token latency "
            "on RTX 5090 without NVLink. Keep the same Qwen3.5-27B BF16 weights and 2048 context. "
            "Use only the provided schema. Avoid repeating failed configurations. "
            "Consider memory headroom, graph execution and communication costs. "
            + json.dumps(context, separators=(",", ":"))}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "engine_proposals", "strict": True, "schema": EngineProposal.model_json_schema(),
        }},
    })
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    return EngineProposal.model_validate_json(answer)


def remembered_failures(output: Path, current: dict) -> list[dict]:
    """Previous capacity failures are hints, never reusable performance proof."""
    hints = []
    for session in reversed(RunStore(output / "runs.sqlite3").recent("engine_session", 8)):
        for cycle in session.get("cycles", []):
            if cycle.get("outcome") != "startup_failed":
                continue
            path = Path(cycle["candidate_report"]).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file():
                continue
            old = json.loads(path.read_text())
            if any(old.get(key) != current.get(key) for key in
                   ("model_fingerprint", "environment", "workload_fingerprint")):
                continue
            log = path.parent / "server.log"
            reason = capacity_failure(log) if log.exists() else None
            hints.append({"config": EngineConfig.model_validate(old["config"]).key,
                          "outcome": "startup_failed", "reason": reason or old.get("error", "")[:100],
                          "scope": "prior-session hint, not performance proof"})
    return hints[-6:]


def evaluate(job: EngineJob) -> dict:
    from evolith.lab.gpu import environment, model_info

    run_id = new_run_id()
    directory = job.output.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    info, env = model_info(job.model_path), environment()
    if not env["cuda_available"] or info["model_type"] != "qwen3_5":
        raise ValueError("This capability requires real CUDA and Qwen3.5.")
    report = {
        "run_id": run_id, "status": "running", "config": job.config.model_dump(),
        "model_fingerprint": info["fingerprint"], "environment": runtime_identity(env),
        "source_hash": source_hash(), "workload_fingerprint": workload_fingerprint(job.max_tokens),
        "max_tokens": job.max_tokens, "seed": job.seed, "records": [],
        "ttft_us": [], "itl_us": [], "rounds": [], "correct": False,
        "scope": "localhost streaming chat; concurrency=1; six fixed public requests; EOS ignored",
    }
    write_json(directory / "report.json", report)
    try:
        with server(job, directory) as client:
            reference = None
            # Fixed equilibration for every configuration, not selective sample deletion.
            for _ in range(5):
                records = [stream_request(client, prompt, job.max_tokens) for prompt in CHAT_PROMPTS]
                if reference is not None and not outputs_match(reference, records):
                    raise ValueError("Warmup output regression.")
                reference = records
            report["records"] = reference
            rng = random.Random(job.seed)
            for _ in range(job.rounds):
                order = list(range(len(CHAT_PROMPTS)))
                rng.shuffle(order)
                measured = {}
                for index in order:
                    measured[index] = stream_request(client, CHAT_PROMPTS[index], job.max_tokens)
                records = [measured[index] for index in range(len(CHAT_PROMPTS))]
                if not outputs_match(reference, records):
                    raise ValueError("Output regression between measurement rounds.")
                ttft, itl = median_latencies(records)
                report["ttft_us"].append(ttft)
                report["itl_us"].append(itl)
                report["rounds"].append({
                    "request_order": order,
                    "ttft_us": [r["ttft_sec"] * 1e6 for r in records],
                    "itl_us": [r["itl_sec"] * 1e6 for r in records],
                })
            report["correct"] = True
            report["status"] = "measured"
            feedback = []
            if job.reference_report:
                reference = json.loads(job.reference_report.read_text())
                report["pilot_comparison"] = compare_trials(reference, report)
                feedback = [{"config": job.config.key, "comparison": {
                    key: report["pilot_comparison"][key] for key in ("correct", "accepted", "stable")
                }, "ttft_speedup": report["pilot_comparison"]["ttft"]["speedup"],
                    "itl_speedup": report["pilot_comparison"]["itl"]["speedup"]}]
            if job.propose and (not feedback or report["pilot_comparison"]["correct"]):
                remembered = remembered_failures(job.output, report)
                report["proposals"] = proposal(
                    client, report, [*remembered, *job.history, *feedback],
                ).model_dump()
    except StartupFailure as exc:
        report["status"] = "startup_failed"
        report["error"] = str(exc)
        report["failure_kind"] = exc.kind
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        raise
    finally:
        write_json(directory / "report.json", report)
        RunStore(job.output / "runs.sqlite3").record(run_id, "engine_trial", report)
    return {"report": str(directory / "report.json"), "status": report["status"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    job = EngineJob.model_validate_json(args.request.read_text())
    def terminate(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    write_json(args.response, {"result": evaluate(job)})


if __name__ == "__main__":
    main()
