from __future__ import annotations

import argparse
from pathlib import Path

from evolith.lab.autonomy import StageJob
from evolith.lab.store import write_json


def execute(job: StageJob) -> dict:
    if job.kind == "search":
        if job.target is None or job.proposals is None:
            raise ValueError("Search requires a measured target and bounded proposals.")
        from evolith.lab.search import search
        return search(
            job.model_path, job.output, [], targets=[job.target], proposals=job.proposals,
            rounds=job.rounds, iterations=job.iterations, seed=job.seed,
            worker_timeout=job.worker_timeout,
        )
    from evolith.lab.serving import verify_model
    return verify_model(
        job.model_path, job.manifest, job.output, tensor_parallel=job.tensor_parallel,
        max_tokens=job.max_tokens, baseline_only=job.kind == "observe",
        observe=job.kind == "observe", workload=job.workload,
        baseline_manifest=job.baseline_manifest,
        required_shapes=[job.target.key] if job.kind == "verify" and job.target else None,
        proposals=job.proposals if job.kind == "observe" else None, history=job.history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    job = StageJob.model_validate_json(args.request.read_text())
    write_json(args.response, {"result": execute(job)})


if __name__ == "__main__":
    main()
