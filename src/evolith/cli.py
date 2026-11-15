from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolith.controller.archive import SteppingStonesArchive
from evolith.controller.engine import EvolithRSIEngine
from evolith.models import EvolutionPlane
from evolith.planes.l1_kernels import L1_KERNEL_TARGETS
from evolith.planes.l2_policies import BASELINE_L2_POLICY_CODE, L2_POLICY_MUTATIONS
from evolith.verifiers.dream_simulator import DreamTraceSimulator
from evolith.verifiers.kernel_sandbox import HardenedKernelVerifier

app = typer.Typer(
    name="evolith",
    help="Evolith: Inference Engine Recursive Self-Improvement (RSI) Framework",
    add_completion=False,
)
console = Console()


@app.command("evolve")
def evolve_cmd(
    plane: str = typer.Option(
        "all",
        "--plane",
        "-p",
        help="Target evolution plane: 'l1' (kernels), 'l2' (scheduler/speculative), or 'all'.",
    ),
    db_path: str = typer.Option(
        ":memory:",
        "--db",
        help="Path to SQLite Stepping-Stones Archive DB (defaults to in-memory).",
    ),
) -> None:
    """Run the dual-plane Inference Engine RSI loop with anti-cheat verification and self-referential ignition."""
    archive = SteppingStonesArchive(db_path=db_path)
    engine = EvolithRSIEngine(archive=archive)

    console.print(
        Panel.fit(
            "[bold cyan]EVOLITH — Inference Engine Recursive Self-Improvement (RSI)[/bold cyan]\n"
            "[dim]Dual-Plane Evolution: Plane L1 (Epilogue/Attention Kernels) + Plane L2 (Scheduler/Speculative/KV)[/dim]",
            border_style="cyan",
        )
    )

    if plane.lower() in ("all", "l1"):
        l1_nodes = engine.evolve_l1_kernels()
        table = Table(
            title="Plane L1: Hardware Kernel Evolution & 4-Stage Anti-Cheat Sandbox",
            show_lines=True,
        )
        table.add_column("Node ID", style="cyan", no_wrap=True)
        table.add_column("Target Kernel", style="magenta")
        table.add_column("MAP-Elites Niche", style="yellow")
        table.add_column("LOC", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Gate Status", justify="center")
        table.add_column("Verification / Reflection Summary")

        for n in l1_nodes:
            if not n.accepted_by_gate and n.speedup == 0.0:
                status = "[bold red]BLOCKED (Reward Hack)[/bold red]"
                summary = f"[red]{n.failure_root_cause}[/red]"
            elif n.accepted_by_gate:
                status = "[bold green]ACCEPTED[/bold green]"
                summary = f"{n.patch_summary} -> [green]{n.metric_delta}[/green]"
            else:
                status = "[yellow]STEPPING STONE[/yellow]"
                summary = f"{n.patch_summary} ({n.metric_delta})"

            table.add_row(
                n.node_id,
                n.target_name,
                n.niche_key,
                str(n.loc),
                f"{n.speedup:.2f}x",
                status,
                summary,
            )
        console.print(table)
        console.print(
            Panel(
                engine.dispatch_table.summary_markdown(),
                title="Hot-Reloaded ShapeBucketDispatchTable",
                border_style="green",
            )
        )

    if plane.lower() in ("all", "l2"):
        l2_nodes = engine.evolve_l2_policies()
        table2 = Table(
            title="Plane L2: Scheduler, Speculative & KV Policy Evolution (Dream Trace Simulator)",
            show_lines=True,
        )
        table2.add_column("Node ID", style="cyan", no_wrap=True)
        table2.add_column("Parent(s)", style="dim")
        table2.add_column("Niche / Strategy", style="yellow")
        table2.add_column("LOC", justify="right")
        table2.add_column("Goodput (QPS)", justify="right", style="green")
        table2.add_column("P99 TTFT / TPOT", justify="right")
        table2.add_column("Role", justify="center")
        table2.add_column("Hypothesis & Patch Summary")

        for n in l2_nodes:
            role = (
                "[yellow]Stepping Stone[/yellow]"
                if n.is_stepping_stone
                else "[bold green]Pareto Winner[/bold green]"
            )
            if n.generation == 0:
                role = "[dim]Baseline[/dim]"
            table2.add_row(
                n.node_id,
                str(n.parent_id or "-"),
                n.niche_key,
                str(n.loc),
                f"{n.pareto_goodput:.1f} ({n.speedup:.2f}x)",
                f"{n.pareto_p99_ttft_ms:.0f}ms / {n.pareto_p99_tpot_ms:.1f}ms",
                role,
                f"{n.patch_summary}",
            )
        console.print(table2)

    if plane.lower() == "all":
        ign = engine.compute_ignition_report()
        console.print(
            Panel(
                f"• [bold]Baseline RSI Coding Agent Speed[/bold]: {ign.baseline_agent_tok_s:.1f} tok/s "
                f"({ign.daily_generations_baseline} evolution rollouts/day)\n"
                f"• [bold]Plane L1 Peak Kernel Speedup[/bold]: {ign.l1_kernel_mean_speedup:.2f}x\n"
                f"• [bold]Plane L2 Goodput Multiplier[/bold]: {ign.l2_goodput_speedup:.2f}x "
                f"(Mean Accepted Draft Tokens: {ign.l2_mean_accepted_draft_tokens:.1f})\n"
                f"• [bold green]Evolved RSI Coding Agent Speed[/bold green]: [bold green]{ign.evolved_agent_tok_s:.1f} tok/s "
                f"({ign.ignition_multiplier:.2f}x Self-Referential Ignition)[/bold green]\n"
                f"• [bold cyan]New Daily Evolution Throughput[/bold cyan]: [bold]{ign.daily_generations_evolved} rollouts/day[/bold]",
                title="Self-Referential Ignition Flywheel (Engine Optimizing Its Own Optimizer)",
                border_style="magenta",
            )
        )


@app.command("verify-kernel")
def verify_kernel_cmd(
    op_name: str = typer.Option(
        "fused_add_rmsnorm_fp8_quant",
        "--op",
        help="Target L1 kernel operator name.",
    ),
) -> None:
    """Run the 4-stage Hardened Kernel Verifier against all mutations (including reward-hacking tests)."""
    if op_name not in L1_KERNEL_TARGETS:
        console.print(f"[red]Unknown op '{op_name}'. Available: {list(L1_KERNEL_TARGETS)}[/red]")
        raise typer.Exit(code=1)

    spec = L1_KERNEL_TARGETS[op_name]
    verifier = HardenedKernelVerifier()

    for idx, (niche, hyp, desc, code) in enumerate(spec.mutations, start=1):
        res = verifier.evaluate(
            op_name=op_name,
            candidate_code=code,
            baseline_code=spec.baseline_code,
        )
        style = "green" if res.is_valid else "red"
        console.print(
            Panel(
                res.reflection_card,
                title=f"Mutation #{idx} [{niche}] — {desc}",
                border_style=style,
            )
        )


@app.command("dream-sim")
def dream_sim_cmd() -> None:
    """Replay 320 multi-turn agent + tool-pause traces in the millisecond Dream Simulator."""
    sim = DreamTraceSimulator()
    base_metrics = sim.simulate(BASELINE_L2_POLICY_CODE)
    _, _, _, best_code = L2_POLICY_MUTATIONS[-1]
    evolved_metrics = sim.simulate(best_code)

    console.print(
        Panel(
            base_metrics.to_reflection_card(),
            title="Generation 0: Workload-Blind Baseline Policy",
            border_style="yellow",
        )
    )
    console.print(
        Panel(
            evolved_metrics.to_reflection_card(),
            title="Generation 4: Co-Optimized Continuum TTL + Context-Lookup Policy",
            border_style="green",
        )
    )


@app.command("archive")
def archive_cmd(
    db_path: str = typer.Option(
        ":memory:",
        "--db",
        help="SQLite DB path (if ':memory:', runs a fresh evolution cycle to populate).",
    ),
) -> None:
    """Inspect the 16x compressed 4-Tuple Stepping-Stones Archive and Pareto Frontier."""
    archive = SteppingStonesArchive(db_path=db_path)
    if not archive.list_nodes():
        engine = EvolithRSIEngine(archive=archive)
        engine.evolve_l1_kernels()
        engine.evolve_l2_policies()

    compressed_text, ratio = archive.format_compressed_memory(limit=12)
    console.print(
        Panel(
            compressed_text,
            title=f"AIDE² {ratio:.1f}x Compressed 4-Tuple Lineage Memory ([Hypothesis, Patch, MetricDelta, RootCause])",
            border_style="cyan",
        )
    )


@app.command("zstep-evolve")
def zstep_evolve_cmd(
    source: str = typer.Option("~/github/zstep", "--source"),
    host: str = typer.Option("10.23.51.33", "--host"),
    user: str = typer.Option("node-user", "--user"),
    generations: int = typer.Option(1, "--generations", min=1, max=100),
    local_timeout: int = typer.Option(1800, "--local-timeout", min=1),
    campaign: str = typer.Option(
        "auto",
        "--campaign",
        "-c",
        help="Target campaign ('auto', 'none', 't1-1', 't1-2', 't1-3', 't2-1', 't2-2', 't2-3').",
    ),
    propose_only: bool = typer.Option(
        False,
        "--propose-only",
        help="Run proposer + local/RedTeam/dispatch gates and freeze archives without remote SSH.",
    ),
    proposer: str = typer.Option(
        "kimi",
        "--proposer",
        help="Proposer backend: 'kimi' (CLI agent) or 'openai' (self-hosted zstep/Nebula OpenAI endpoint).",
    ),
    proposer_api_base: str = typer.Option(
        "http://127.0.0.1:8080/v1",
        "--proposer-api-base",
        help="OpenAI-compatible `/v1` base URL when `--proposer openai` is selected.",
    ),
    proposer_model: str = typer.Option(
        "DeepSeek-V4.1-Flash",
        "--proposer-model",
        help="Served model ID for `--proposer openai`.",
    ),
    gpu_smoke: bool = typer.Option(
        False,
        "--gpu-smoke",
        help="Run fast container GPU smoke tests before full 475 GiB benchmark sweeps.",
    ),
) -> None:
    """Run isolated zstep source evolution; no deployment or upstream commit occurs."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.loop import ZStepRSILoop
    from evolith.zstep.task import ZStepRSITask

    task = ZStepRSITask(
        project_root=Path.cwd(),
        source_repo=Path(source).expanduser().resolve(),
        remote_host=host,
        remote_user=user,
        max_generations=generations,
        local_timeout_sec=local_timeout,
        campaign=campaign,
        propose_only=propose_only,
        proposer_backend=proposer,
        proposer_api_base=proposer_api_base,
        proposer_model=proposer_model,
        enable_gpu_smoke=gpu_smoke,
    )
    with ZStepArchive(task.state_dir / "archive.sqlite3") as archive:
        statuses = ZStepRSILoop(task, archive=archive).run()
    for status in statuses:
        console.print(f"zstep candidate: {status.value}")


@app.command("zstep-submit")
def zstep_submit_cmd(
    patch: str = typer.Option(..., "--patch", "-p", help="Path to unified git diff patch file."),
    hypothesis: str = typer.Option(
        ...,
        "--hypothesis",
        help="Mechanistic bottleneck & speedup hypothesis for lineage memory.",
    ),
    niche_key: str = typer.Option(
        "(plane=auto,target=deepseek_v41)",
        "--niche-key",
        help="MAP-Elites / 2-Strike circuit breaker niche key.",
    ),
    summary: str = typer.Option(
        "",
        "--summary",
        help="Concise summary of the code mutation.",
    ),
    source: str = typer.Option("~/github/zstep", "--source"),
    host: str = typer.Option("10.23.51.33", "--host"),
    user: str = typer.Option("node-user", "--user"),
    local_timeout: int = typer.Option(1800, "--local-timeout", min=1),
    campaign: str = typer.Option(
        "none",
        "--campaign",
        "-c",
        help="Target campaign ('none', 'auto', 't1-1', 't1-2', 't1-3', 't2-1', 't2-2', 't2-3').",
    ),
    propose_only: bool = typer.Option(
        False,
        "--propose-only",
        help="Run Coordinator local/RedTeam/dispatch gates and freeze archives without remote SSH.",
    ),
) -> None:
    """Submit a unified diff directly through Evolith's full Coordinator gates, screen, and 3-round verify."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.loop import ZStepRSILoop
    from evolith.zstep.proposer import PatchProposer
    from evolith.zstep.task import ZStepRSITask

    task = ZStepRSITask(
        project_root=Path.cwd(),
        source_repo=Path(source).expanduser().resolve(),
        remote_host=host,
        remote_user=user,
        max_generations=1,
        local_timeout_sec=local_timeout,
        campaign=campaign,
        propose_only=propose_only,
    )
    proposer = PatchProposer(
        Path(patch),
        hypothesis=hypothesis,
        niche_key=niche_key,
        patch_summary=summary,
    )
    with ZStepArchive(task.state_dir / "archive.sqlite3") as archive:
        statuses = ZStepRSILoop(task, archive=archive, proposer=proposer).run()
    for status in statuses:
        console.print(f"zstep candidate: {status.value}")


@app.command("zstep-campaigns")
def zstep_campaigns_cmd() -> None:
    """List ROI-ranked zstep optimization campaigns and their 2-Strike circuit breaker status."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.campaigns import list_campaigns, resolve_campaign

    db_path = Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3"
    blocked: dict[str, list[str]] = {}
    parent: str | None = None
    if db_path.is_file():
        with ZStepArchive(db_path) as archive:
            parent = archive.current_parent()
            blocked = archive.blocked_niches(strike_limit=2, parent_id=parent)

    next_auto = resolve_campaign("auto", blocked)
    table = Table(
        title=(
            f"zstep ROI-Ranked Optimization Campaigns "
            f"(parent: {parent or 'frozen-zstep-baseline'}, "
            f"auto -> {next_auto.priority_rank if next_auto else 'none'})"
        ),
        show_lines=True,
    )
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Campaign ID", no_wrap=True)
    table.add_column("Plane", no_wrap=True)
    table.add_column("Expected Gain")
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Scoped Allowed Roots")

    for spec in list_campaigns():
        is_frozen = spec.primary_niche_key in blocked
        if is_frozen:
            status_cell = "[bold red]FROZEN (2-Strike)[/bold red]"
        elif next_auto and spec.campaign_id == next_auto.campaign_id:
            status_cell = "[bold green]NEXT (auto)[/bold green]"
        else:
            status_cell = "[green]READY[/green]"
        plane_label = "L1 Kernel" if spec.plane == EvolutionPlane.L1_KERNEL else "L2 Scheduler"
        table.add_row(
            spec.priority_rank,
            spec.campaign_id,
            plane_label,
            spec.expected_gain,
            status_cell,
            "\n".join(spec.allowed_roots),
        )
    console.print(table)


@app.command("zstep-resume")
def zstep_resume_cmd(
    candidate_id: str,
    source: str = typer.Option("~/github/zstep", "--source"),
    host: str = typer.Option("10.23.51.33", "--host"),
    user: str = typer.Option("node-user", "--user"),
) -> None:
    """Resume an interrupted zstep candidate from frozen source archives and round checkpoints."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.loop import ZStepRSILoop
    from evolith.zstep.task import ZStepRSITask

    task = ZStepRSITask(
        project_root=Path.cwd(),
        source_repo=Path(source).expanduser().resolve(),
        remote_host=host,
        remote_user=user,
    )
    with ZStepArchive(task.state_dir / "archive.sqlite3") as archive:
        try:
            status = ZStepRSILoop(task, archive=archive).resume(candidate_id)
        except KeyError:
            console.print(f"Unknown zstep candidate: {candidate_id}")
            raise typer.Exit(code=1) from None
    console.print(f"zstep candidate {candidate_id}: {status.value}")


@app.command("zstep-status")
def zstep_status_cmd() -> None:
    """Show isolated zstep candidate lineage and the best verified baseline."""
    from evolith.zstep.archive import ZStepArchive

    with ZStepArchive(Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3") as archive:
        entries = archive.list_candidates()
        parent = archive.current_parent()
    if not entries:
        console.print("No zstep candidates recorded.")
        return
    table = Table(title=f"zstep RSI (best verified parent: {parent or 'frozen baseline'})")
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Parent", no_wrap=True)
    table.add_column("LOC", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Gain", justify="right", no_wrap=True)
    table.add_column("Summary")
    for entry in entries:
        hyp = str(entry.get("hypothesis") or entry.get("patch_summary") or "")
        detail = str(entry.get("failure_reason") or entry.get("metric_delta") or "")
        plane_short = str(entry.get("plane") or "").replace("l1_kernel", "L1").replace("l2_scheduler", "L2")
        prefix = f"[{plane_short}] " if plane_short else ""
        summary = prefix + (f"{hyp} | {detail}".strip(" |") or "-")
        table.add_row(
            str(entry["candidate_id"]),
            str(entry["parent_id"] or "-"),
            str(entry.get("changed_lines") or 0),
            str(entry["status"]),
            f"{float(entry['geomean_gain']):.2%}",
            summary[:120],
        )
    console.print(table)


@app.command("zstep-evidence")
def zstep_evidence_cmd(candidate_id: str) -> None:
    """Print the evidence directory for a zstep candidate."""
    from evolith.zstep.archive import ZStepArchive

    with ZStepArchive(Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3") as archive:
        try:
            evidence = archive.evidence_dir(candidate_id)
        except KeyError:
            console.print(f"Unknown zstep candidate: {candidate_id}")
            raise typer.Exit(code=1) from None
    console.print(evidence)


if __name__ == "__main__":
    app()
