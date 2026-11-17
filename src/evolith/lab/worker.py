from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import Field

from evolith.lab.contracts import KernelConfig, Shape, StrictModel
from evolith.lab.store import write_json


class Job(StrictModel):
    config: KernelConfig
    shapes: list[Shape] = Field(min_length=1, max_length=32)
    eps: float = Field(gt=0)
    seed: int = Field(ge=0)
    rounds: int = Field(ge=5, le=101)
    iterations: int = Field(ge=1, le=1000)
    min_gain: float = Field(ge=0, lt=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    job = Job.model_validate_json(args.request.read_text())
    import torch

    from evolith.lab.gpu import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; there is no synthetic/CPU fallback.")
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.set_grad_enabled(False)
    results = []
    for shape in job.shapes:
        result = evaluate(
            job.config, shape, job.eps, job.seed, job.rounds, job.iterations, job.min_gain
        )
        results.append(result.model_dump())
        write_json(args.response, {"evaluations": results})
    print(json.dumps({"status": "completed", "evaluations": len(results)}))


if __name__ == "__main__":
    main()
