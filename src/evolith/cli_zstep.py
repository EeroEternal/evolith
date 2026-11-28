from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.table import Table

from evolith.models import EvolutionPlane


def register_zstep(app: typer.Typer, console: Console) -> None:
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
        gdrive_folder_id: str | None = typer.Option(
            None,
            "--gdrive-folder-id",
            envvar="EVOLITH_GDRIVE_FOLDER_ID",
            help="Optional Google Drive folder ID to automatically archive statistical benchmark artifacts.",
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
                "gdrive_folder_id": gdrive_folder_id,
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
        gdrive_folder_id: str | None = typer.Option(
            None,
            "--gdrive-folder-id",
            envvar="EVOLITH_GDRIVE_FOLDER_ID",
            help="Optional Google Drive folder ID to automatically archive statistical benchmark artifacts.",
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
                "gdrive_folder_id": gdrive_folder_id,
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


    @app.command("zstep-sync-gdrive")
    def zstep_sync_gdrive_cmd(
        folder_id: str = typer.Option(
            ...,
            "--folder-id",
            envvar="EVOLITH_GDRIVE_FOLDER_ID",
            help="Destination Google Drive folder ID for statistical benchmark archiving.",
        ),
        state_dir: str = typer.Option(
            ".evolith/zstep-rsi",
            "--state-dir",
            help="Local Evolith RSI state directory containing archive.sqlite3 and evidence/.",
        ),
        benchmark_dir: list[str] = typer.Option(
            [],
            "--benchmark-dir",
            "-b",
            help="Additional benchmark artifact directory (e.g., docs/benchmarks/...) to sync.",
        ),
        campaign_folder: str = typer.Option(
            "evolith_rsi_runs",
            "--campaign-folder",
            help="Subfolder name under the Google Drive root folder for state_dir artifacts.",
        ),
    ) -> None:
        """Sync local Evolith statistical benchmark telemetry, lineage CSV, and reports to Google Drive."""
        from evolith.zstep.archive import ZStepArchive
        from evolith.zstep.gdrive_sync import GDriveBenchmarkSync

        resolved_state = Path(state_dir).expanduser().resolve()
        rows = None
        db_file = resolved_state / "archive.sqlite3"
        if db_file.is_file():
            with ZStepArchive(db_file) as archive:
                rows = archive.list_candidates()
        syncer = GDriveBenchmarkSync(folder_id)
        report = syncer.sync_state_and_benchmarks(
            resolved_state,
            archive_rows=rows,
            benchmark_dirs=tuple(Path(p).expanduser().resolve() for p in benchmark_dir),
            campaign_folder_name=campaign_folder,
        )
        console.print(
            f"Synced to {report.folder_url}: "
            f"{len(report.uploaded_files)} uploaded, {len(report.updated_files)} updated"
        )
        if report.errors:
            for err in report.errors:
                console.print(f"[yellow]warn:[/yellow] {markup_escape(err)}")


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


