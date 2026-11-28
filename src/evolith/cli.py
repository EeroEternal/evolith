from __future__ import annotations

import typer
from rich.console import Console

from evolith.cli_demo import register_demo
from evolith.cli_workbench import register_workbench
from evolith.cli_zstep import register_zstep
from evolith.lab.cli import register

app = typer.Typer(
    name="evolith",
    help="Evolith: independently verified and measured GPU inference optimization.",
    add_completion=False,
)
console = Console()

register_demo(app, console)
register_zstep(app, console)
register_workbench(app, console)
register(app)

if __name__ == "__main__":
    app()
