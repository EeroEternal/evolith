from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console


def register_workbench(app: typer.Typer, console: Console) -> None:
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

