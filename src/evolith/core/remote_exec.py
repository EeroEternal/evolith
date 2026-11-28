"""Shared SSH command building, ControlMaster multiplexing, and transient-drop detection.

Used by both SM120 (`evolith.zstep.remote`) and PPU-ZW810E SAIL (`evolith.lab.zstep_sail`)
remote runners to keep SSH transport behavior consistent across hardware backends.
"""

from __future__ import annotations

from pathlib import Path

TRANSIENT_SSH_DROP_TOKENS: tuple[str, ...] = (
    "banner exchange",
    "kex_exchange_identification",
    "connection timed out",
    "connection reset by peer",
)


def build_ssh_multiplex_opts(
    control_path: str | Path,
    *,
    persist_seconds: int = 300,
) -> list[str]:
    """Return OpenSSH ControlMaster multiplexing flags."""
    return [
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={control_path}",
        "-o",
        f"ControlPersist={persist_seconds}",
    ]


def build_ssh_base_cmd(
    *,
    port: int | None = None,
    strict_host_key: str = "no",
    known_hosts_devnull: bool = False,
    connect_timeout: int = 20,
    server_alive_interval: int = 15,
    server_alive_count_max: int = 3,
    batch_mode: bool = True,
    use_sshpass: bool = False,
    control_path: str | Path | None = None,
    control_persist: int = 600,
) -> list[str]:
    """Build a consistent non-interactive SSH base command."""
    cmd: list[str] = ["sshpass", "-e", "ssh"] if use_sshpass else ["ssh"]
    if port is not None:
        cmd.extend(["-p", str(port)])
    cmd.extend(["-o", f"StrictHostKeyChecking={strict_host_key}"])
    if known_hosts_devnull:
        cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])
    cmd.extend(
        [
            "-o",
            f"ConnectTimeout={connect_timeout}",
            "-o",
            f"ServerAliveInterval={server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={server_alive_count_max}",
        ]
    )
    if batch_mode:
        cmd.extend(["-o", "BatchMode=yes"])
    if control_path:
        cmd.extend(build_ssh_multiplex_opts(control_path, persist_seconds=control_persist))
    return cmd


def is_transient_ssh_drop(returncode: int, combined_output: str) -> bool:
    """Return True when SSH exited 255 due to a pre-auth or transient transport drop."""
    if returncode != 255:
        return False
    detail = combined_output.lower()
    return any(token in detail for token in TRANSIENT_SSH_DROP_TOKENS)
