from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.table import Table

from evolith.controller.archive import SteppingStonesArchive
from evolith.controller.engine import EvolithRSIEngine
from evolith.lab.cli import register
from evolith.models import EvolutionPlane
from evolith.planes.l1_kernels import L1_KERNEL_TARGETS
from evolith.planes.l2_policies import BASELINE_L2_POLICY_CODE, L2_POLICY_MUTATIONS
from evolith.verifiers.dream_simulator import DreamTraceSimulator
from evolith.verifiers.kernel_sandbox import HardenedKernelVerifier

app = typer.Typer(
    name="evolith",
    help="Evolith: independently verified and measured GPU inference optimization.",
    add_completion=False,
)
console = Console()


@app.command("demo-evolve")
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


@app.command("demo-verify-kernel")
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


@app.command("demo-dream-sim")
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


@app.command("demo-archive")
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


def _build_zstep_task(
    *,
    source: str,
    host: str | None,
    user: str | None,
    model_dir: str | None,
    rustup_root: str | None,
    baseline_sha: str | None,
    propose_only: bool,
    extra_fields: dict[str, object],
):
    from evolith.zstep.task import ZStepRSITask

    values: dict[str, object] = {
        "project_root": Path.cwd(),
        "source_repo": Path(source).expanduser().resolve(),
        "propose_only": propose_only,
    }
    if host is not None:
        values["remote_host"] = host
    if user is not None:
        values["remote_user"] = user
    if model_dir is not None:
        values["model_dir"] = model_dir
    if rustup_root is not None:
        values["remote_rustup_root"] = rustup_root
    if baseline_sha is not None:
        values["baseline_sha"] = baseline_sha
    for key, val in extra_fields.items():
        if val is not None:
            values[key] = val
    task = ZStepRSITask(**values)  # type: ignore[arg-type]
    if not propose_only and not task.remote_host.strip():
        raise typer.BadParameter(
            "Remote GPU host is required unless --propose-only is set; "
            "pass --host or set EVOLITH_ZSTEP_HOST."
        )
    return task


@app.command("zstep-evolve")
def zstep_evolve_cmd(
    source: str = typer.Option("~/github/zstep", "--source"),
    host: str | None = typer.Option(None, "--host", envvar="EVOLITH_ZSTEP_HOST"),
    user: str | None = typer.Option(None, "--user", envvar="EVOLITH_ZSTEP_USER"),
    model_dir: str | None = typer.Option(
        None,
        "--model-dir",
        envvar="EVOLITH_ZSTEP_MODEL_DIR",
        help="Remote model weights directory mounted read-only into benchmark containers.",
    ),
    rustup_root: str | None = typer.Option(
        None,
        "--rustup-root",
        envvar="EVOLITH_ZSTEP_RUSTUP_ROOT",
        help="Optional remote rustup installation prefix containing cargo/ and rustup/.",
    ),
    baseline_sha: str | None = typer.Option(None, "--baseline-sha"),
    generations: int = typer.Option(1, "--generations", min=1, max=100),
    local_timeout: int | None = typer.Option(None, "--local-timeout", min=1),
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
    proposer_api_base: str | None = typer.Option(
        None,
        "--proposer-api-base",
        help="OpenAI-compatible `/v1` base URL when `--proposer openai` is selected.",
    ),
    proposer_model: str | None = typer.Option(
        None,
        "--proposer-model",
        help="Served model ID for `--proposer openai`.",
    ),
    gpu_smoke: bool = typer.Option(
        False,
        "--gpu-smoke",
        help="Run fast container GPU smoke tests before full 475 GiB benchmark sweeps.",
    ),
    hw_profile: str | None = typer.Option(
        None,
        "--hw-profile",
        envvar="EVOLITH_ZSTEP_HW_PROFILE",
        help="Hardware profile ('auto', 'sm120-8x6000d-pp2tp4', 'sm120-8x6000d-tp8', 'sm120-8x5090-tp8', 'sm90-8xh100-tp8').",
    ),
    bench_profile: str | None = typer.Option(
        None,
        "--bench-profile",
        envvar="EVOLITH_ZSTEP_BENCH_PROFILE",
        help="Benchmark tier profile ('custom', 'short-throughput', 'longctx-agent').",
    ),
    sudo: bool = typer.Option(
        False,
        "--sudo",
        envvar="EVOLITH_ZSTEP_SUDO",
        help="Allow remote sudo fallback for Docker when SSH user is not in the docker group.",
    ),
    scout_probe: bool = typer.Option(
        False,
        "--scout-probe",
        help="Enable endogenous Stage-1 Scout GPU microbenchmark probe before final patch synthesis.",
    ),
) -> None:
    """Run isolated zstep source evolution; no deployment or upstream commit occurs."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.loop import ZStepRSILoop

    task = _build_zstep_task(
        source=source,
        host=host,
        user=user,
        model_dir=model_dir,
        rustup_root=rustup_root,
        baseline_sha=baseline_sha,
        propose_only=propose_only,
        extra_fields={
            "max_generations": generations,
            "local_timeout_sec": local_timeout,
            "campaign": campaign,
            "proposer_backend": proposer,
            "proposer_api_base": proposer_api_base,
            "proposer_model": proposer_model,
            "enable_gpu_smoke": gpu_smoke,
            "hardware_profile": hw_profile,
            "benchmark_profile": bench_profile,
            "remote_sudo": sudo,
            "enable_scout_probe": scout_probe,
        },
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
    host: str | None = typer.Option(None, "--host", envvar="EVOLITH_ZSTEP_HOST"),
    user: str | None = typer.Option(None, "--user", envvar="EVOLITH_ZSTEP_USER"),
    model_dir: str | None = typer.Option(
        None,
        "--model-dir",
        envvar="EVOLITH_ZSTEP_MODEL_DIR",
        help="Remote model weights directory mounted read-only into benchmark containers.",
    ),
    rustup_root: str | None = typer.Option(
        None,
        "--rustup-root",
        envvar="EVOLITH_ZSTEP_RUSTUP_ROOT",
        help="Optional remote rustup installation prefix containing cargo/ and rustup/.",
    ),
    baseline_sha: str | None = typer.Option(None, "--baseline-sha"),
    local_timeout: int | None = typer.Option(None, "--local-timeout", min=1),
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
    gpu_smoke: bool = typer.Option(
        False,
        "--gpu-smoke",
        help="Run fast container GPU smoke tests before full 475 GiB benchmark sweeps.",
    ),
    hw_profile: str | None = typer.Option(
        None,
        "--hw-profile",
        envvar="EVOLITH_ZSTEP_HW_PROFILE",
        help="Hardware profile ('auto', 'sm120-8x6000d-pp2tp4', 'sm120-8x6000d-tp8', 'sm120-8x5090-tp8', 'sm90-8xh100-tp8').",
    ),
    bench_profile: str | None = typer.Option(
        None,
        "--bench-profile",
        envvar="EVOLITH_ZSTEP_BENCH_PROFILE",
        help="Benchmark tier profile ('custom', 'short-throughput', 'longctx-agent').",
    ),
    sudo: bool = typer.Option(
        False,
        "--sudo",
        envvar="EVOLITH_ZSTEP_SUDO",
        help="Allow remote sudo fallback for Docker when SSH user is not in the docker group.",
    ),
) -> None:
    """Submit a unified diff directly through Evolith's full Coordinator gates, screen, and 3-round verify."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.loop import ZStepRSILoop
    from evolith.zstep.proposer import PatchProposer

    task = _build_zstep_task(
        source=source,
        host=host,
        user=user,
        model_dir=model_dir,
        rustup_root=rustup_root,
        baseline_sha=baseline_sha,
        propose_only=propose_only,
        extra_fields={
            "max_generations": 1,
            "local_timeout_sec": local_timeout,
            "campaign": campaign,
            "enable_gpu_smoke": gpu_smoke,
            "hardware_profile": hw_profile,
            "benchmark_profile": bench_profile,
            "remote_sudo": sudo,
        },
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


def _detect_local_upstream_sha() -> str:
    import subprocess

    upstream_dir = Path.cwd() / "targets/zstep/upstream"
    if not (upstream_dir / ".git").exists():
        return ""
    res = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=upstream_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    return res.stdout.strip() if res.returncode == 0 else ""


@app.command("zstep-campaigns")
def zstep_campaigns_cmd() -> None:
    """List ROI-ranked zstep optimization campaigns and their 2-Strike circuit breaker status."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.campaigns import list_campaigns, resolve_campaign

    db_path = Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3"
    upstream_sha = _detect_local_upstream_sha()
    blocked: dict[str, list[str]] = {}
    parent: str | None = None
    if db_path.is_file():
        with ZStepArchive(db_path) as archive:
            parent = archive.current_parent(upstream_sha=upstream_sha)
            blocked = archive.blocked_niches(strike_limit=2, parent_id=parent)

    parent_label = parent or (f"frozen@{upstream_sha}" if upstream_sha else "frozen-zstep-baseline")
    next_auto = resolve_campaign("auto", blocked)
    table = Table(
        title=(
            f"zstep ROI-Ranked Optimization Campaigns "
            f"(parent: {markup_escape(str(parent_label))}, "
            f"auto -> {next_auto.priority_rank if next_auto else 'exhausted'})"
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
    host: str | None = typer.Option(None, "--host", envvar="EVOLITH_ZSTEP_HOST"),
    user: str | None = typer.Option(None, "--user", envvar="EVOLITH_ZSTEP_USER"),
    model_dir: str | None = typer.Option(
        None,
        "--model-dir",
        envvar="EVOLITH_ZSTEP_MODEL_DIR",
        help="Remote model weights directory mounted read-only into benchmark containers.",
    ),
    rustup_root: str | None = typer.Option(
        None,
        "--rustup-root",
        envvar="EVOLITH_ZSTEP_RUSTUP_ROOT",
        help="Optional remote rustup installation prefix containing cargo/ and rustup/.",
    ),
    baseline_sha: str | None = typer.Option(None, "--baseline-sha"),
    gpu_smoke: bool = typer.Option(
        False,
        "--gpu-smoke",
        help="Run fast container GPU smoke tests when resuming pre-screen candidates.",
    ),
) -> None:
    """Resume an interrupted zstep candidate from frozen source archives and round checkpoints."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.loop import ZStepRSILoop

    task = _build_zstep_task(
        source=source,
        host=host,
        user=user,
        model_dir=model_dir,
        rustup_root=rustup_root,
        baseline_sha=baseline_sha,
        propose_only=False,
        extra_fields={"enable_gpu_smoke": gpu_smoke},
    )
    with ZStepArchive(task.state_dir / "archive.sqlite3") as archive:
        try:
            status = ZStepRSILoop(task, archive=archive).resume(candidate_id)
        except KeyError:
            console.print(f"Unknown zstep candidate: {markup_escape(candidate_id)}")
            raise typer.Exit(code=1) from None
    console.print(f"zstep candidate {markup_escape(candidate_id)}: {status.value}")


@app.command("zstep-status")
def zstep_status_cmd() -> None:
    """Show isolated zstep candidate lineage and the best verified baseline."""
    from evolith.zstep.archive import ZStepArchive

    upstream_sha = _detect_local_upstream_sha()
    with ZStepArchive(Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3") as archive:
        entries = archive.list_candidates()
        parent = archive.current_parent(upstream_sha=upstream_sha)
    if not entries:
        console.print("No zstep candidates recorded.")
        return
    parent_label = parent or (f"frozen baseline ({upstream_sha})" if upstream_sha else "frozen baseline")
    table = Table(
        title=f"zstep RSI (best verified parent: {markup_escape(str(parent_label))})"
    )
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Parent", no_wrap=True)
    table.add_column("LOC", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Gain", justify="right", no_wrap=True)
    table.add_column("Summary")
    for entry in entries:
        hyp = str(entry.get("hypothesis") or entry.get("patch_summary") or "")
        detail = str(entry.get("failure_reason") or entry.get("metric_delta") or "")
        plane_short = (
            str(entry.get("plane") or "")
            .replace("l1_kernel", "L1")
            .replace("l2_scheduler", "L2")
        )
        prefix = f"[{plane_short}] " if plane_short else ""
        summary = prefix + (f"{hyp} | {detail}".strip(" |") or "-")
        table.add_row(
            markup_escape(str(entry["candidate_id"])),
            markup_escape(str(entry["parent_id"] or "-")),
            str(entry.get("changed_lines") or 0),
            markup_escape(str(entry["status"])),
            f"{float(entry['geomean_gain']):.2%}",
            markup_escape(summary[:120]),
        )
    console.print(table)


@app.command("zstep-evidence")
def zstep_evidence_cmd(
    candidate_id: str,
    events: bool = typer.Option(
        False, "--events", help="Also print recorded lifecycle events for this candidate."
    ),
) -> None:
    """Print the evidence directory (and optional event trail) for a zstep candidate."""
    from evolith.zstep.archive import ZStepArchive

    with ZStepArchive(Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3") as archive:
        try:
            evidence = archive.evidence_dir(candidate_id)
        except KeyError:
            console.print(f"Unknown zstep candidate: {markup_escape(candidate_id)}")
            raise typer.Exit(code=1) from None
        event_rows = archive.list_events(candidate_id) if events else []
    console.print(markup_escape(str(evidence)))
    for ev in event_rows:
        line = f"[{ev['event_type']}] stage={ev['stage']} status={ev['status']} {ev['detail']}".rstrip()
        console.print(markup_escape(line))


@app.command("zstep-probe")
def zstep_probe_cmd(
    script: str = typer.Option(..., "--script", "-s", help="Python operator/collective probe script."),
    patch: str | None = typer.Option(None, "--patch", "-p", help="Optional unified git diff to apply before probing."),
    output: str = typer.Option(
        ".evolith/zstep-rsi/probes/latest-probe.json",
        "--output",
        "-o",
        help="Local path to save downloaded JSON probe telemetry.",
    ),
    source: str = typer.Option("~/github/zstep", "--source"),
    host: str | None = typer.Option(None, "--host", envvar="EVOLITH_ZSTEP_HOST"),
    user: str | None = typer.Option(None, "--user", envvar="EVOLITH_ZSTEP_USER"),
    baseline_sha: str | None = typer.Option(None, "--baseline-sha"),
) -> None:
    """Run a fast isolated 8-GPU microbenchmark probe without loading full 475 GiB model weights."""
    import shutil
    import subprocess
    import tempfile
    import uuid

    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.remote import ZStepRemoteRunner
    from evolith.zstep.workspace import ZStepWorkspace

    script_path = Path(script).expanduser().resolve()
    if not script_path.is_file():
        raise typer.BadParameter(f"Probe script not found: {script_path}")

    task = _build_zstep_task(
        source=source,
        host=host,
        user=user,
        model_dir=None,
        rustup_root=None,
        baseline_sha=baseline_sha,
        propose_only=False,
        extra_fields={},
    )
    ws = ZStepWorkspace(task)
    upstream_sha = ws.upstream_head_sha()
    accepted_patches: list[bytes] = []
    db_file = task.state_dir / "archive.sqlite3"
    if db_file.is_file():
        with ZStepArchive(db_file) as archive:
            accepted_patches = archive.accepted_patches(upstream_sha=upstream_sha)
    probe_id = f"probe-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        cand_dir = ws.create_candidate(probe_id, accepted_patches)
        try:
            if patch:
                patch_bytes = Path(patch).expanduser().resolve().read_bytes()
                subprocess.run(["git", "apply", "-"], input=patch_bytes, cwd=cand_dir, check=True)
            tar_path = ws.freeze_source(cand_dir, tmp_dir / "probe-src.tar.gz")
        finally:
            shutil.rmtree(cand_dir, ignore_errors=True)
        out_path = ZStepRemoteRunner(task).run_probe(
            probe_id, tar_path, script_path, Path(output).expanduser().resolve()
        )
    console.print(markup_escape(str(out_path)))


@app.command("zstep-memory")
def zstep_memory_cmd(
    campaign: str = typer.Option(
        "none",
        "--campaign",
        "-c",
        help="Optional campaign ID/alias (e.g. 't1-2', 't1-3', 't2-4') to preview scoped fact filtering.",
    ),
) -> None:
    """Preview the local Memory Ring (facts, stepping stones, directives) injected into zstep prompts."""
    from evolith.zstep.campaigns import resolve_campaign
    from evolith.zstep.memory_ring import load_memory_ring

    spec = resolve_campaign(campaign)
    roots = spec.allowed_roots if spec is not None else ()
    rendered = load_memory_ring(Path.cwd(), allowed_roots=roots)
    if not rendered:
        console.print("No local memory/ entries found.")
        return
    title_scope = spec.campaign_id if spec is not None else "all allowed_roots"
    console.print(
        Panel(
            markup_escape(rendered),
            title=f"Evolith Local Memory Ring (scope: {markup_escape(title_scope)})",
            border_style="cyan",
        )
    )


@app.command("zstep-formal-verify")
def zstep_formal_verify_cmd(
    target: str = typer.Option(
        "targets/zstep/upstream",
        "--target",
        "-t",
        help="Path to zstep source directory to formally verify.",
    ),
    calibrate: bool = typer.Option(
        False,
        "--calibrate",
        help="Run real source-file mutation & historical candidate calibration (requires Z3).",
    ),
    allow_missing_z3: bool = typer.Option(
        False,
        "--allow-missing-z3",
        help="Allow running AST-only checks when Z3 is not installed (default: fail-stop).",
    ),
) -> None:
    """Run fail-stop AST + 3-way Z3 SMT verification (and source/historical calibration) on zstep."""
    from evolith.zstep.formal_verifier import (
        FormalVerificationError,
        calibrate_formal_mutations,
        run_formal_verification,
    )

    target_dir = Path(target).expanduser().resolve()
    if not target_dir.is_dir():
        console.print(f"[red]Target directory not found: {markup_escape(str(target_dir))}[/red]")
        raise typer.Exit(code=1)

    try:
        report = run_formal_verification(
            target_dir,
            require_z3=not allow_missing_z3,
            expect_shm_allreduce=True,
        )
    except FormalVerificationError as exc:
        console.print(f"[bold red]FORMAL VERIFICATION FAILED[/bold red]: {markup_escape(str(exc))}")
        raise typer.Exit(code=1) from None

    table = Table(
        title=(
            f"veriserve/vstack Formal Gate [{report.gate_verdict}] "
            f"| z3={'ON' if report.z3_available else 'OFF'} "
            f"| target={markup_escape(target_dir.name)}"
        ),
        show_lines=True,
    )
    table.add_column("Clause ID", style="cyan", no_wrap=True)
    table.add_column("Pillar", style="magenta", no_wrap=True)
    table.add_column("Method", no_wrap=True)
    table.add_column("Scope", justify="right", no_wrap=True)
    table.add_column("Verdict", justify="center", no_wrap=True)
    table.add_column("Anchored Sites / Notes")

    verdict_styles = {
        "PROVED": "[bold green]PROVED (Z3)[/bold green]",
        "CLOSED_DIFF": "[bold green]CLOSED (3000 diff)[/bold green]",
        "PASS": "[bold green]PASS (AST)[/bold green]",
        "NOT_APPLICABLE": "[dim]N/A (non-PXB)[/dim]",
        "MODEL_ONLY": "[yellow]MODEL_ONLY (non-gating)[/yellow]",
        "INCONCLUSIVE_NO_Z3": "[yellow]INCONCLUSIVE (no Z3)[/yellow]",
        "INCONCLUSIVE_MODEL_MISMATCH": "[bold yellow]INCONCLUSIVE (MODEL)[/bold yellow]",
        "INCONCLUSIVE_TIMEOUT": "[bold red]INCONCLUSIVE (TIMEOUT)[/bold red]",
        "INCONCLUSIVE_PARSE_ERROR": "[bold red]INCONCLUSIVE (PARSE)[/bold red]",
        "INCONCLUSIVE_MISSING_FILE": "[bold red]INCONCLUSIVE (MISSING)[/bold red]",
        "VIOLATION": "[bold red]VIOLATION[/bold red]",
    }
    for cl in report.clauses:
        verdict = verdict_styles.get(
            cl.verdict,
            "[bold green]PASS[/bold green]" if cl.passed else "[bold red]VIOLATION[/bold red]",
        )
        if cl.method == "z3_smt2":
            scope_str = "symbolic"
        elif cl.method == "ast_exact_eval":
            scope_str = "exact AST"
        elif cl.method in ("seeded_diff_3000", "compiled_cpp_diff_3000"):
            scope_str = f"{cl.explored_states} trials"
        else:
            scope_str = f"{cl.explored_states} states"
        notes = "\n".join(cl.anchored_sites) if cl.anchored_sites else (cl.detail or "-")
        table.add_row(
            cl.clause_id,
            cl.pillar,
            cl.method,
            scope_str,
            verdict,
            markup_escape(notes),
        )
    console.print(table)
    if report.contract_sha256:
        console.print(
            f"[dim]Contract SHA256: {report.contract_sha256[:16]} "
            f"| Tree SHA256: {report.target_tree_sha256[:16]} "
            f"| {markup_escape(report.z3_version or 'z3=off')}[/dim]"
        )
    if report.weakest_link:
        console.print(f"[dim]Weakest link: {markup_escape(report.weakest_link)}[/dim]")

    if calibrate:
        cal = calibrate_formal_mutations(target_dir)
        mtable = Table(
            title=(
                f"vstack Source-Mutation Anti-Vacuity Gate "
                f"(killed {cal.killed_mutants}/{cal.total_mutants}, kill_rate={cal.kill_rate:.0%})"
            ),
            show_lines=True,
        )
        mtable.add_column("Mutant ID", style="yellow", no_wrap=True)
        mtable.add_column("Expected Clause", style="cyan", no_wrap=True)
        mtable.add_column("Verdict", justify="center", no_wrap=True)
        mtable.add_column("Killed?", justify="center", no_wrap=True)
        mtable.add_column("Counterexample / Witness")
        for row in cal.rows:
            status = "[bold green]KILLED[/bold green]" if row.killed else "[bold red]SURVIVED[/bold red]"
            mtable.add_row(
                row.mutant_id,
                row.expected_clause,
                row.model_verdict,
                status,
                markup_escape((row.witness or "-")[:110]),
            )
        console.print(mtable)

        if cal.historical_rows:
            htable = Table(
                title=(
                    f"Historical Candidate Agreement "
                    f"(matched {cal.historical_matched}/{cal.historical_checked})"
                ),
                show_lines=True,
            )
            htable.add_column("Candidate", style="cyan", no_wrap=True)
            htable.add_column("Expected", justify="center", no_wrap=True)
            htable.add_column("Actual", justify="center", no_wrap=True)
            htable.add_column("Match", justify="center", no_wrap=True)
            htable.add_column("Historical Ground Truth / Witness")
            for hrow in cal.historical_rows:
                exp_s = "PASS" if hrow.expected_pass else "REJECT"
                act_s = "PASS" if hrow.actual_pass else "REJECT"
                m_s = "[bold green]OK[/bold green]" if hrow.matched_expectation else "[bold red]MISMATCH[/bold red]"
                summary_note = hrow.witness or hrow.historical_note
                htable.add_row(
                    hrow.candidate_id,
                    exp_s,
                    act_s,
                    m_s,
                    markup_escape(summary_note[:110]),
                )
            console.print(htable)

        if not cal.passed:
            raise typer.Exit(code=1)


@app.command("zstep-init-hw")
def zstep_init_hw_cmd(
    host: str | None = typer.Option(None, "--host", envvar="EVOLITH_ZSTEP_HOST"),
    user: str | None = typer.Option(None, "--user", envvar="EVOLITH_ZSTEP_USER"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Optional built-in hardware profile preset instead of live SSH probe.",
    ),
    prefer_pp2: bool = typer.Option(
        True,
        "--prefer-pp2/--prefer-tp8",
        help="Prefer intra-socket PP=2 x TP=4 on dual-NUMA 8-GPU PCIe hosts.",
    ),
) -> None:
    """Probe remote GPU topology (or select a preset) and write hardware-profile.json."""
    from evolith.zstep.hardware import resolve_hardware_profile
    from evolith.zstep.remote import ZStepRemoteRunner

    out_path = Path.cwd() / ".evolith/zstep-rsi/hardware-profile.json"
    if preset:
        profile = resolve_hardware_profile(preset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    else:
        task = _build_zstep_task(
            source="~/github/zstep",
            host=host,
            user=user,
            model_dir=None,
            rustup_root=None,
            baseline_sha=None,
            propose_only=False,
            extra_fields={},
        )
        profile = ZStepRemoteRunner(task).probe_hardware(
            prefer_pp2_on_dual_socket_pcie=prefer_pp2
        )
    console.print(
        Panel(
            markup_escape(profile.model_dump_json(indent=2)),
            title=f"HardwareProfile Ready: {markup_escape(profile.profile_id)}",
            border_style="green",
        )
    )


@app.command("zstep-trace-check")
def zstep_trace_check_cmd(
    parity: str = typer.Option(..., "--parity", "-p", help="Path to runtime parity JSON."),
    metrics: str | None = typer.Option(None, "--metrics", "-m", help="Optional benchmark metrics JSON."),
    allow_unexposed_schedule: bool = typer.Option(
        False, "--allow-unexposed-schedule", help="Skip chunked_schedule exposure requirement."
    ),
) -> None:
    """Run vstack Layer-3 runtime trace conformance check on GPU parity & benchmark JSONs."""
    from evolith.zstep.trace_conformance import (
        TraceConformanceError,
        verify_trace_conformance,
    )

    p_path = Path(parity).expanduser().resolve()
    m_path = Path(metrics).expanduser().resolve() if metrics else None
    try:
        rep = verify_trace_conformance(
            p_path,
            m_path,
            require_schedule_exposure=not allow_unexposed_schedule,
            raise_on_failure=True,
        )
    except TraceConformanceError as exc:
        console.print(f"[bold red]TRACE CONFORMANCE FAILED[/bold red]: {markup_escape(str(exc))}")
        raise typer.Exit(code=1) from None

    table = Table(title=f"vstack Runtime Trace Conformance [{rep.gate_verdict}]", show_lines=True)
    table.add_column("Clause ID", style="cyan", no_wrap=True)
    table.add_column("Pillar", style="magenta", no_wrap=True)
    table.add_column("Verdict", justify="center", no_wrap=True)
    table.add_column("Detail")
    for cl in rep.clauses:
        table.add_row(
            cl.clause_id,
            cl.pillar,
            f"[bold green]{cl.verdict}[/bold green]",
            markup_escape(cl.detail or "-"),
        )
    console.print(table)


@app.command("zstep-compose")
def zstep_compose_cmd(
    min_gain: float = typer.Option(0.015, "--min-gain", help="Minimum near-miss Verify geomean gain."),
    max_gain: float = typer.Option(0.0299, "--max-gain", help="Maximum near-miss Verify geomean gain."),
    output_patch: str | None = typer.Option(
        None, "--output-patch", "-o", help="Optional output path to write the top composed diff."
    ),
) -> None:
    """Discover orthogonal near-miss stepping-stone pairs & failure clusters in archive.sqlite3."""
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.composer import (
        cluster_remote_failures,
        compose_patch_pair,
        find_orthogonal_stepping_stone_pairs,
    )

    db_path = Path.cwd() / ".evolith/zstep-rsi/archive.sqlite3"
    if not db_path.is_file():
        console.print("No archive.sqlite3 found.")
        return

    upstream_sha = _detect_local_upstream_sha()
    with ZStepArchive(db_path) as archive:
        parent = archive.current_parent(upstream_sha=upstream_sha)
        pairs = find_orthogonal_stepping_stone_pairs(
            archive,
            upstream_sha=upstream_sha,
            parent_id=parent,
            min_gain=min_gain,
            max_gain=max_gain,
        )
        clusters = cluster_remote_failures(archive)
        if output_patch and pairs:
            top = pairs[0]
            left_p = bytes(archive.get_candidate(top.left_id)["patch"] or b"")
            right_p = bytes(archive.get_candidate(top.right_id)["patch"] or b"")
            base_tree = Path.cwd() / "targets/zstep/upstream"
            combined = compose_patch_pair(base_tree, left_p, right_p)
            out_file = Path(output_patch).expanduser().resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(combined)
            console.print(f"[green]Wrote composed patch ({len(combined)} bytes): {markup_escape(str(out_file))}[/green]")

    table = Table(title="Orthogonal Stepping-Stone Composition Queue", show_lines=True)
    table.add_column("Left Candidate", style="cyan", no_wrap=True)
    table.add_column("Right Candidate", style="cyan", no_wrap=True)
    table.add_column("Est. Combined Gain", justify="right", style="green")
    table.add_column("LOC", justify="right")
    table.add_column("Disjoint Files")
    for p in pairs:
        table.add_row(
            f"{p.left_id} ({p.left_gain:+.2%})",
            f"{p.right_id} ({p.right_gain:+.2%})",
            f"{p.additive_gain_estimate:+.2%}",
            str(p.combined_loc),
            markup_escape(", ".join((*p.left_files, *p.right_files))),
        )
    console.print(table)

    if clusters:
        ctable = Table(title="Evening Gate Failure Clusters", show_lines=True)
        ctable.add_column("Category", style="yellow")
        ctable.add_column("Count", justify="right")
        ctable.add_column("Sample Root Cause")
        for c in clusters:
            ctable.add_row(c.category, str(c.count), markup_escape(c.sample_reason[:110]))
        console.print(ctable)


@app.command("workbench-formal-check")
def workbench_formal_check_cmd() -> None:
    """Run explicit-state protocol models, directed bug mutants, witnesses, and trace replay."""
    from evolith.core.formal_replay import run_protocol_verification_suite

    report = run_protocol_verification_suite()
    console.print(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(code=1)


@app.command("workbench-dry-run")
def workbench_dry_run_cmd(
    patch_file: Path = typer.Option(..., "--patch", exists=True, readable=True),
) -> None:
    """Compute declarative TargetSurface ActionRung & blast radius for a unified diff."""
    from evolith.core.surface import analyze_changeset_blast_radius, default_zstep_surface

    patch_text = patch_file.read_text(encoding="utf-8", errors="replace")
    files = tuple(
        sorted(
            {
                line[6:].strip()
                for line in patch_text.splitlines()
                if line.startswith("+++ b/") and line[6:].strip()
            }
        )
    )
    plan = analyze_changeset_blast_radius(
        default_zstep_surface(),
        changed_files=files,
        patch_text=patch_text,
    )
    console.print(plan.model_dump_json(indent=2))


@app.command("workbench-daemon")
def workbench_daemon_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(18765, "--port"),
    runner_kind: str = typer.Option("deepseek_v41", "--runner"),
    world_size: int = typer.Option(8, "--world-size"),
    worker_root: str = typer.Option("", "--worker-root"),
    model_dir: str = typer.Option("", "--model-dir"),
    num_kv_pages: int = typer.Option(2560, "--num-kv-pages"),
    buckets: str = typer.Option("1,16,32", "--buckets"),
    envelope_db: str = typer.Option("", "--envelope-db"),
) -> None:
    """Start the resident multi-GPU `StateKeeper` workbench daemon (`evolith.workbench`)."""
    from evolith.workbench.gpu_daemon import WorkbenchDaemonConfig, WorkbenchGpuDaemon

    parsed_buckets = tuple(int(x.strip()) for x in buckets.split(",") if x.strip())
    cfg = WorkbenchDaemonConfig(
        host=host,
        port=port,
        runner_kind=runner_kind,
        world_size=world_size,
        worker_root=worker_root,
        model_dir=model_dir,
        num_kv_pages=num_kv_pages,
        active_decode_buckets=parsed_buckets,
        envelope_db_path=envelope_db,
    )
    daemon = WorkbenchGpuDaemon(cfg)
    daemon.serve_forever()


@app.command("workbench-rpc")
def workbench_rpc_cmd(
    method: str = typer.Argument(
        ...,
        help="RPC method: ping|status|inner_screen|profile_step|hot_apply|sleep|wake|reattach_shm|shutdown",
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(18765, "--port"),
    params_json: str = typer.Option("{}", "--params"),
) -> None:
    """Send a JSON-RPC command to a running `evolith workbench-daemon`."""
    import json

    from evolith.workbench.gpu_daemon import WorkbenchRpcClient

    params = json.loads(params_json)
    client = WorkbenchRpcClient(host=host, port=port)
    resp = client.call(method, **params)
    console.print(json.dumps(resp, indent=2))
    if not resp.get("ok", False):
        raise typer.Exit(code=1)


register(app)

if __name__ == "__main__":
    app()
