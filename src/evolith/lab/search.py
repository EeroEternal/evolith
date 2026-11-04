from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from evolith.lab.contracts import Evaluation, KernelConfig, Manifest, Proposal, Shape
from evolith.lab.store import RunStore, new_run_id, write_json


def execute_job(request: dict, directory: Path, timeout: float) -> list[Evaluation]:
    """Only trusted workers and validated parameters run here, never model source."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", request)
    command = [
        sys.executable, "-m", "evolith.lab.worker",
        "--request", str((directory / "request.json").resolve()),
        "--response", str((directory / "response.json").resolve()),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        (directory / "worker.log").write_text(output)
        raise RuntimeError(f"Worker exceeded {timeout}s; process group terminated.") from None
    (directory / "worker.log").write_text(output)
    if process.returncode:
        raise RuntimeError(f"Worker exited {process.returncode}; see {directory / 'worker.log'}")
    data = json.loads((directory / "response.json").read_text())
    results = [Evaluation.model_validate(r) for r in data["evaluations"]]
    expected = [Shape.model_validate(s).key for s in request["shapes"]]
    if [r.shape.key for r in results] != expected or any(
        r.config != KernelConfig.model_validate(request["config"]) for r in results
    ):
        raise RuntimeError("Worker response does not match the requested candidate/shapes.")
    return results


def search(
    model_path: Path, output: Path, rows: list[int], *,
    proposals: Path | None = None, rounds: int = 11, iterations: int = 32,
    seed: int = 2026, min_gain: float = 0.03, worker_timeout: float = 300,
) -> dict:
    from evolith.lab.gpu import environment, model_info
    from evolith.lab.kernels import kernel_hash
    from evolith.lab.worker import Job

    env = environment()
    if not env["cuda_available"]:
        raise RuntimeError("CUDA is required; refusing synthetic performance results.")
    info = model_info(model_path)
    run_id = new_run_id()
    directory = output.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    configs = (
        Proposal.model_validate_json(proposals.read_text()).candidates if proposals
        else [KernelConfig(num_warps=w, rows_per_program=r) for w in (4, 8) for r in (1, 4)]
    )
    configs = list({c.key: c for c in configs}.values())
    shapes = [
        Shape(rows=n, hidden=info["hidden"], residual=r)
        for n in sorted(set(rows)) for r in (False, True)
    ]
    if not shapes:
        raise ValueError("At least one row count is required.")
    evaluations: list[Evaluation] = []
    errors = []

    def run(config, requested_shapes, job_name, job_seed):
        job = Job(
            config=config, shapes=requested_shapes, eps=info["eps"], seed=job_seed,
            rounds=rounds, iterations=iterations, min_gain=min_gain,
        )
        return execute_job(job.model_dump(), directory / job_name, worker_timeout)

    for config in configs:
        print(f"Evaluating {config.key} on {len(shapes)} shape contracts", flush=True)
        try:
            evaluations.extend(run(config, shapes, config.key, seed))
        except RuntimeError as exc:
            errors.append({"config": config.key, "reason": str(exc)})
    # Search observations cannot directly authorize runtime dispatch.
    # Confirm each finalist with fresh inputs and an independent timing run.
    finalists: dict[str, Evaluation] = {}
    for evaluation in evaluations:
        timing = evaluation.timing
        if evaluation.correct and timing and timing.accepted:
            old = finalists.get(evaluation.shape.key)
            if old is None or timing.lower_bound > old.timing.lower_bound:
                finalists[evaluation.shape.key] = evaluation
    winners = {}
    for config in configs:
        selected = [r.shape for r in finalists.values() if r.config == config]
        if not selected:
            continue
        print(f"Confirming {config.key} on {len(selected)} shapes", flush=True)
        try:
            confirmation = run(config, selected, f"confirm-{config.key}", seed + 100003)
            for evaluation in confirmation:
                evaluation.stage = "confirmation"
                if evaluation.correct and evaluation.timing and evaluation.timing.accepted:
                    winners[evaluation.shape.key] = config
            evaluations.extend(confirmation)
        except RuntimeError as exc:
            errors.append({"config": config.key, "reason": str(exc)})
    manifest = Manifest(
        run_id=run_id, model_fingerprint=info["fingerprint"], kernel_hash=kernel_hash(),
        environment=env, eps=info["eps"], evaluations=evaluations, winners=winners,
        measurement={
            "baseline": "vLLM GemmaRMSNorm static implementation, torch.compile(fullgraph=True)",
            "oracle": "independent float64 with explicit BF16 residual rounding",
            "timer": "CUDA events; warm ring of 8 inputs; randomized paired order",
            "cache": "warm working set; no artificial L2 scrub",
            "rounds": rounds, "iterations": iterations, "seed": seed, "min_gain": min_gain,
            "confirmation_seed": seed + 100003,
            "confidence": "one-sided 99% paired bootstrap; exploratory, not a multiple-testing guarantee",
            "model": info, "worker_errors": errors,
        },
    )
    report = manifest.model_dump()
    write_json(directory / "manifest.json", report)
    RunStore(output / "runs.sqlite3").record(run_id, "kernel_search", report)
    return {"status": "completed" if evaluations else "failed", "run_id": run_id,
            "manifest": str(directory / "manifest.json"), "verified_winners": len(winners),
            "evaluations": len(evaluations), "worker_errors": errors}
