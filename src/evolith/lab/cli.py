from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from evolith.lab.contracts import refusal
from evolith.lab.store import RunStore


def emit(value: dict) -> None:
    typer.echo(json.dumps(value, indent=2, allow_nan=False))


def register(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor(
        model_path: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    ):
        """Inspect actual CUDA hardware/runtime and optional local model compatibility."""
        try:
            from evolith.lab.gpu import environment, model_info
            report = environment()
            if model_path is not None:
                report["model"] = model_info(model_path)
            emit(report)
            if not report["cuda_available"]:
                raise typer.Exit(2)
        except typer.Exit:
            raise
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            emit(refusal(str(exc)))
            raise typer.Exit(2) from exc

    @app.command("evolve")
    def evolve(
        model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
        output: Annotated[Path, typer.Option()] = Path("results"),
        rows: Annotated[str, typer.Option(help="Exact row counts; no implicit bucket widening.")] = "1,4,16,64",
        proposals: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        rounds: Annotated[int, typer.Option(min=5, max=101)] = 11,
        iterations: Annotated[int, typer.Option(min=1, max=1000)] = 32,
        seed: Annotated[int, typer.Option(min=0)] = 2026,
        min_gain: Annotated[float, typer.Option(min=0, max=0.99)] = 0.03,
        worker_timeout: Annotated[int, typer.Option(min=5)] = 300,
    ):
        """Search trusted Triton parameters, independently verify and measure each finalist."""
        try:
            from evolith.lab.search import search
            result = search(
                model_path, output, [int(n) for n in rows.split(",")],
                proposals=proposals, rounds=rounds, iterations=iterations,
                seed=seed, min_gain=min_gain, worker_timeout=worker_timeout,
            )
            emit(result)
            if result["status"] == "failed":
                raise typer.Exit(2)
        except typer.Exit:
            raise
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            emit(refusal(str(exc)))
            raise typer.Exit(2) from exc

    @app.command("verify-model")
    def verify_model(
        model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
        manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        output: Annotated[Path, typer.Option()] = Path("results"),
        tensor_parallel: Annotated[int, typer.Option(min=1, max=8)] = 4,
        max_tokens: Annotated[int, typer.Option(min=1, max=256)] = 32,
        proposals: Annotated[Path | None, typer.Option(help="Write local-model parameter proposals.")] = None,
        baseline_only: Annotated[bool, typer.Option(help="Measure only the baseline; never install a candidate.")] = False,
    ):
        """Compare real Qwen3.5 generation before/after an opt-in measured kernel adapter."""
        try:
            from evolith.lab.serving import verify_model as verify
            result = verify(
                model_path, manifest, output, tensor_parallel=tensor_parallel,
                max_tokens=max_tokens, proposals=proposals,
                baseline_only=baseline_only,
            )
            emit(result)
            if result["status"] not in ("verified", "baseline_measured"):
                raise typer.Exit(2)
        except typer.Exit:
            raise
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            emit(refusal(str(exc)))
            raise typer.Exit(2) from exc

    @app.command("archive")
    def archive(
        db: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("results/runs.sqlite3"),
    ):
        """Read append-only measured experiment records without triggering work."""
        emit({"runs": RunStore(db).list()})
