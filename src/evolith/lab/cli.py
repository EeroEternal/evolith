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
    @app.command("autoevolve")
    def autoevolve(
        model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
        output: Annotated[Path, typer.Option()] = Path("results"),
        workload: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        cycles: Annotated[int, typer.Option(min=1, max=8)] = 3,
        evaluation_budget: Annotated[int, typer.Option(min=2, max=128)] = 32,
        time_budget: Annotated[int, typer.Option(min=30, max=7200)] = 1800,
        candidates_per_cycle: Annotated[int, typer.Option(min=1, max=12)] = 4,
        tensor_parallel: Annotated[int, typer.Option(min=1, max=8)] = 4,
        max_tokens: Annotated[int, typer.Option(min=1, max=256)] = 16,
        rounds: Annotated[int, typer.Option(min=5, max=101)] = 19,
        iterations: Annotated[int, typer.Option(min=1, max=1000)] = 128,
        seed: Annotated[int, typer.Option(min=0)] = 2026,
    ):
        """Observe, choose targets, propose, verify and iterate within immutable budgets."""
        try:
            from evolith.lab.autonomy import Budget, run
            result = run(
                model_path, output, budget=Budget(
                    cycles=cycles, evaluations=evaluation_budget, seconds=time_budget,
                    candidates_per_cycle=candidates_per_cycle,
                ), tensor_parallel=tensor_parallel, max_tokens=max_tokens, workload=workload,
                rounds=rounds, iterations=iterations, seed=seed,
            )
            emit(result)
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            emit(refusal(str(exc)))
            raise typer.Exit(2) from exc

    @app.command("qwen-ultrafast")
    def qwen_ultrafast(
        remote: Annotated[str, typer.Option(help="SSH destination, for example user@host.")],
        local_repo: Annotated[Path, typer.Option(exists=True, file_okay=False)],
        remote_repo: Annotated[Path, typer.Option()],
        nccl_library: Annotated[Path, typer.Option()],
        output: Annotated[Path, typer.Option()] = Path("results"),
        port: Annotated[int, typer.Option(min=1, max=65535)] = 22,
        model: Annotated[str, typer.Option()] = "models/Qwen3.8-Flash-Next-NVFP4",
        control_path: Annotated[Path | None, typer.Option()] = None,
        candidates: Annotated[str, typer.Option(help="Comma-separated GEMV block sizes.")] = "64,256,512",
        incumbent: Annotated[int, typer.Option()] = 128,
        decode: Annotated[int, typer.Option(min=8, max=256)] = 64,
        rounds: Annotated[int, typer.Option(min=5, max=101)] = 19,
        iterations: Annotated[int, typer.Option(min=1, max=1000)] = 128,
        model_pairs: Annotated[int, typer.Option(min=5, max=19)] = 5,
        time_budget: Annotated[int, typer.Option(min=60, max=7200)] = 7200,
    ):
        """Search bounded Qwen Ultrafast GEMV configs and verify them on eight GPUs."""
        try:
            from evolith.lab.qwen_ultrafast import QwenAdapterConfig, QwenUltrafastAdapter
            adapter = QwenUltrafastAdapter(QwenAdapterConfig(
                remote=remote, port=port, local_repo=local_repo, remote_repo=remote_repo,
                nccl_library=nccl_library, output=output, control_path=control_path,
                model=model, decode=decode, rounds=rounds, iterations=iterations,
                model_pairs=model_pairs, seconds=time_budget,
            ))
            result = adapter.run([int(value) for value in candidates.split(",")], incumbent)
            emit(result)
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            emit(refusal(str(exc)))
            raise typer.Exit(2) from exc

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

    @app.command("zstep-sail-evolve")
    def zstep_sail_evolve(
        remote: Annotated[str, typer.Option(help="SSH destination, e.g. root@10.8.141.36.")],
        local_zstep_repo: Annotated[Path, typer.Option(exists=True, file_okay=False)],
        remote_root: Annotated[Path, typer.Option()],
        model_dir: Annotated[Path, typer.Option()],
        output: Annotated[Path, typer.Option()] = Path("results"),
        port: Annotated[int, typer.Option(min=1, max=65535)] = 22,
        container_image: Annotated[str | None, typer.Option()] = None,
        control_path: Annotated[Path | None, typer.Option()] = None,
        initial_policy_path: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        parent_run: Annotated[str | None, typer.Option()] = None,
        prompt_ids: Annotated[str, typer.Option()] = "671,6102,294,8760,344",
        decode_tokens: Annotated[int, typer.Option()] = 32,
        bench_tier: Annotated[str, typer.Option()] = "evolve:8:16:256:64",
        observe_bench_tier: Annotated[str, typer.Option()] = "observe:4:4:256:16",
        cycles: Annotated[int, typer.Option(min=1, max=8)] = 2,
        evaluations: Annotated[int, typer.Option(min=1, max=64)] = 12,
        model_evaluations: Annotated[int, typer.Option(min=1, max=12)] = 3,
        families_per_cycle: Annotated[int, typer.Option(min=1, max=6)] = 2,
        rounds: Annotated[int, typer.Option(min=5, max=101)] = 11,
        iterations: Annotated[int, typer.Option(min=1, max=1000)] = 25,
        model_pairs: Annotated[int, typer.Option(min=5, max=19)] = 5,
        min_share: Annotated[float, typer.Option(min=0.01, max=0.99)] = 0.03,
        time_budget: Annotated[int, typer.Option(min=60, max=7200)] = 3600,
    ):
        """Observe 8x810E SAIL operator costs, confirm micro-winners, and verify against incumbent."""
        try:
            from evolith.lab.zstep_sail import (
                DEFAULT_SAIL_CONTAINER_IMAGE,
                ZstepSailEvolution,
                ZstepSailEvolutionConfig,
            )

            result = ZstepSailEvolution(
                ZstepSailEvolutionConfig(
                    remote=remote,
                    port=port,
                    local_zstep_repo=local_zstep_repo,
                    remote_root=remote_root,
                    model_dir=model_dir,
                    output=output,
                    container_image=container_image or DEFAULT_SAIL_CONTAINER_IMAGE,
                    control_path=control_path,
                    initial_policy_path=initial_policy_path,
                    parent_run=parent_run,
                    prompt_ids=prompt_ids,
                    decode_tokens=decode_tokens,
                    bench_tier=bench_tier,
                    observe_bench_tier=observe_bench_tier,
                    cycles=cycles,
                    evaluations=evaluations,
                    model_evaluations=model_evaluations,
                    families_per_cycle=families_per_cycle,
                    rounds=rounds,
                    iterations=iterations,
                    model_pairs=model_pairs,
                    min_share=min_share,
                    seconds=time_budget,
                )
            ).run()
            emit(result)
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            emit(refusal(str(exc)))
            raise typer.Exit(2) from exc

    @app.command("archive")
    def archive(
        db: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("results/runs.sqlite3"),
    ):
        """Read append-only measured experiment records without triggering work."""
        emit({"runs": RunStore(db).list()})
