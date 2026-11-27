from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
from typing import Callable

from evolith.zstep.task import ZStepRSITask

Executor = Callable[..., subprocess.CompletedProcess[str]]


class RemotePreflightError(RuntimeError):
    pass


class RemoteInfraError(RuntimeError):
    """Transient SSH/SCP transport, timeout, or remote stage lock error (0 premise strikes)."""


def _sanitize_container_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return cleaned[:48] or "run"


class ZStepRemoteRunner:
    def __init__(self, task: ZStepRSITask, executor: Executor | None = None) -> None:
        self.task = task
        self.executor = executor or self._execute
        self.destination = f"{task.remote_user}@{task.remote_host}"
        self._use_sudo_docker: bool = False

    @staticmethod
    def _execute(
        args: list[str], *, cwd: Path | None = None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)

    def _require_host(self) -> None:
        if not self.task.remote_host.strip():
            raise RemotePreflightError(
                "remote_host is required for remote execution; pass --host or set EVOLITH_ZSTEP_HOST"
            )

    def _ssh_multiplex_opts(self) -> list[str]:
        if self.executor != self._execute:
            return []
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPath=/tmp/evolith-ssh-%C",
            "-o",
            "ControlPersist=300",
        ]

    @staticmethod
    def _is_pre_auth_ssh_drop(res: subprocess.CompletedProcess[str]) -> bool:
        if res.returncode != 255:
            return False
        detail = (res.stderr + res.stdout).lower()
        return any(
            token in detail
            for token in (
                "banner exchange",
                "kex_exchange_identification",
                "connection timed out",
                "connection reset by peer",
            )
        )

    def _ssh(self, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        self._require_host()
        command = ["sshpass", "-e", "ssh"] if os.environ.get("SSHPASS") else ["ssh"]
        command.extend(
            [
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=6",
                *self._ssh_multiplex_opts(),
            ]
        )
        if not os.environ.get("SSHPASS"):
            command.extend(["-o", "BatchMode=yes"])
        command.extend([self.destination, shlex.join(args)])
        max_tries = 3 if self.executor == self._execute else 1
        last_res: subprocess.CompletedProcess[str] | None = None
        for attempt in range(max_tries):
            try:
                last_res = self.executor(
                    command, timeout=timeout or self.task.remote_timeout_sec
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise RemoteInfraError(f"SSH transport failed: {exc}") from exc
            if not self._is_pre_auth_ssh_drop(last_res) or attempt + 1 >= max_tries:
                return last_res
            time.sleep(3.0 * (attempt + 1))
        assert last_res is not None
        return last_res

    def _scp(self, source: str, destination: str) -> subprocess.CompletedProcess[str]:
        self._require_host()
        command = ["sshpass", "-e", "scp"] if os.environ.get("SSHPASS") else ["scp"]
        command.extend(
            [
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=6",
                *self._ssh_multiplex_opts(),
            ]
        )
        if not os.environ.get("SSHPASS"):
            command.extend(["-o", "BatchMode=yes"])
        command.extend([source, destination])
        max_tries = 3 if self.executor == self._execute else 1
        last_res: subprocess.CompletedProcess[str] | None = None
        for attempt in range(max_tries):
            try:
                last_res = self.executor(command, timeout=self.task.remote_timeout_sec)
            except (subprocess.TimeoutExpired, OSError) as exc:
                if attempt + 1 >= max_tries:
                    raise RemoteInfraError(f"SCP transfer failed: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if last_res.returncode == 0 or attempt + 1 >= max_tries:
                return last_res
            time.sleep(1.5 * (attempt + 1))
        assert last_res is not None
        return last_res

    @staticmethod
    def _require_success(result: subprocess.CompletedProcess[str], action: str) -> str:
        if result.returncode:
            detail = (result.stderr + result.stdout).strip()
            raise RemoteInfraError(f"{action} exited {result.returncode}: {detail}")
        return result.stdout

    def _expected_gpu_ids(self) -> list[str]:
        ids = [tok.strip() for tok in self.task.gpu_devices.split(",") if tok.strip()]
        return ids or ["0", "1", "2", "3", "4", "5", "6", "7"]

    def _sudo_shell_prefix(self) -> str:
        if not self._use_sudo_docker:
            return ""
        pw = os.environ.get("SSHPASS", "")
        if pw:
            return f"printf '%s\\n' {shlex.quote(pw)} | sudo -S -p '' "
        return "sudo -n "

    def _rustup_env_snippet(self) -> str:
        rustup_prefix = shlex.quote(self.task.remote_rustup_root.strip())
        return f"""CUSTOM_RUSTUP_ROOT={rustup_prefix}
if [ -z "$CUSTOM_RUSTUP_ROOT" ] && [ -n "${{EVOLITH_ZSTEP_RUSTUP_ROOT:-}}" ]; then
  CUSTOM_RUSTUP_ROOT="$EVOLITH_ZSTEP_RUSTUP_ROOT"
elif [ -z "$CUSTOM_RUSTUP_ROOT" ] && [ -d "$HOME/evolith-rsi/rustup/cargo/bin" ]; then
  CUSTOM_RUSTUP_ROOT="$HOME/evolith-rsi/rustup"
elif [ -z "$CUSTOM_RUSTUP_ROOT" ] && [ -d "/data02/zstep-work/rustup/cargo/bin" ]; then
  CUSTOM_RUSTUP_ROOT="/data02/zstep-work/rustup"
fi
if [ -n "$CUSTOM_RUSTUP_ROOT" ] && [ -d "$CUSTOM_RUSTUP_ROOT/cargo/bin" ]; then
  export CARGO_HOME="${{CARGO_HOME:-$CUSTOM_RUSTUP_ROOT/cargo}}"
  export RUSTUP_HOME="${{RUSTUP_HOME:-$CUSTOM_RUSTUP_ROOT/rustup}}"
  export PATH="$CUSTOM_RUSTUP_ROOT/cargo/bin:$HOME/.cargo/bin:$PATH"
else
  export PATH="$HOME/.cargo/bin:$PATH"
fi"""

    def _overlay_env_snippet(self, overlay_dir: str) -> str:
        hw = self.task.resolved_hardware_profile()
        if not hw.overlay_script:
            return "DOCKER_OVERLAY_ARGS=()"
        ov_script = shlex.quote(f"./{hw.overlay_script.lstrip('./')}")
        q_ov_dir = shlex.quote(overlay_dir)
        return (
            f"if [ -x {ov_script} ]; then\n"
            f"  {ov_script} {q_ov_dir}\n"
            f"  DOCKER_OVERLAY_ARGS=($({ov_script} --print-docker-args {q_ov_dir}))\n"
            f"else\n"
            f"  DOCKER_OVERLAY_ARGS=()\n"
            f"fi"
        )

    def _check_gpus(self) -> None:
        result = self._ssh(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader",
            ]
        )
        if result.returncode:
            detail = (result.stderr + result.stdout).strip()
            raise RemotePreflightError(f"Unable to inspect remote GPUs: {detail}")
        rows = [line for line in result.stdout.splitlines() if line.strip()]
        wanted = self._expected_gpu_ids()
        wanted_set = set(wanted)
        seen_wanted = 0
        for row in rows:
            index, memory = [part.strip() for part in row.split(",", 1)]
            if index not in wanted_set:
                continue
            seen_wanted += 1
            used_mib = int(memory.split()[0])
            if used_mib > self.task.max_gpu_idle_mib:
                raise RemotePreflightError(f"GPU {index} is occupied: {memory}")
        if len(rows) < len(wanted) or seen_wanted != len(wanted):
            raise RemotePreflightError(f"Expected {len(wanted)} GPUs, found {len(rows)}")

    def probe_hardware(self, *, prefer_pp2_on_dual_socket_pcie: bool = True):
        """Auto-probe remote GPU architecture, VRAM, and NUMA/PXB/NVLink topology."""
        from evolith.zstep.hardware import infer_hardware_profile_from_probe

        smi = self._ssh(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            timeout=30,
        )
        if smi.returncode:
            raise RemotePreflightError(
                f"Failed to query GPU capabilities on {self.destination}: {(smi.stderr + smi.stdout).strip()}"
            )
        topo = self._ssh(["nvidia-smi", "topo", "-m"], timeout=30)
        topo_txt = topo.stdout if topo.returncode == 0 else ""
        profile = infer_hardware_profile_from_probe(
            smi.stdout,
            topo_txt,
            prefer_pp2_on_dual_socket_pcie=prefer_pp2_on_dual_socket_pcie,
        )
        out_path = self.task.project_root / ".evolith/zstep-rsi/hardware-profile.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        return profile

    def _check_docker(self) -> None:
        docker = self._ssh(["docker", "info", "--format", "{{.ServerVersion}}"])
        if docker.returncode:
            if self.task.remote_sudo:
                pw = os.environ.get("SSHPASS", "")
                sudo_cmd = (
                    f"printf '%s\\n' {shlex.quote(pw)} | sudo -S -p '' docker info --format '{{{{.ServerVersion}}}}'"
                    if pw
                    else "sudo -n docker info --format '{{.ServerVersion}}'"
                )
                sudo_res = self._ssh(["bash", "-lc", sudo_cmd])
                if sudo_res.returncode == 0:
                    self._use_sudo_docker = True
                else:
                    detail = (sudo_res.stderr + sudo_res.stdout).strip()
                    raise RemotePreflightError(
                        f"Docker sudo access failed on {self.destination}: {detail}"
                    )
            else:
                detail = (docker.stderr + docker.stdout).strip()
                raise RemotePreflightError(
                    f"Docker access is required for {self.destination}; {detail}. "
                    "Grant the user Docker access; Evolith will not use sudo."
                )

    def preflight(self) -> None:
        self._check_gpus()
        self._check_docker()
        docker_bin = ["sudo", "-n", "docker"] if (self._use_sudo_docker and not os.environ.get("SSHPASS")) else ["docker"]
        py_expr = (
            "import json, pathlib, sys; d = pathlib.Path(sys.argv[1]); "
            "idx = d / 'model.safetensors.index.json'; "
            "missing = ([f for f in set(json.loads(idx.read_text()).get('weight_map', {}).values()) if not (d / f).is_file()] if idx.is_file() else []); "
            "sys.exit('missing shards: ' + str(sorted(missing)[:3])) if missing else print(json.loads((d / 'config.json').read_text())['model_type'])"
        )
        if self._use_sudo_docker and os.environ.get("SSHPASS"):
            pw = os.environ["SSHPASS"]
            model_check = self._ssh(
                [
                    "bash",
                    "-lc",
                    f"printf '%s\\n' {shlex.quote(pw)} | sudo -S -p '' docker run --rm --entrypoint python3 "
                    f"-v {shlex.quote(f'{self.task.model_dir}:{self.task.model_dir}:ro')} "
                    f"{shlex.quote(self.task.image)} -c {shlex.quote(py_expr)} {shlex.quote(self.task.model_dir)}",
                ]
            )
        else:
            model_check = self._ssh(
                [
                    *docker_bin,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "python3",
                    "-v",
                    f"{self.task.model_dir}:{self.task.model_dir}:ro",
                    self.task.image,
                    "-c",
                    py_expr,
                    self.task.model_dir,
                ]
            )
        if model_check.returncode or model_check.stdout.strip().splitlines()[-1:] != [self.task.model_type]:
            raise RemotePreflightError(
                f"Model must be {self.task.model_type} at {self.task.model_dir}"
            )

    @staticmethod
    def _container_name(source_dir: str, output_path: str) -> str:
        parts = PurePosixPath(source_dir).parts
        scope = "-".join(parts[-3:]) if len(parts) >= 3 else "-".join(parts)
        stem = PurePosixPath(output_path).stem
        return f"evolith-{_sanitize_container_token(f'{scope}-{stem}')}"

    def _docker_gpus_spec(self) -> str:
        # Docker's `--gpus` uses CSV key=val parsing; comma-separated device IDs
        # must carry literal inner double quotes (`"device=0,1,..."`).
        return shlex.quote(f'"device={self.task.gpu_devices}"')

    def benchmark_script(
        self,
        source_dir: str,
        output_path: str,
        tiers: str,
        repeats: int,
        parity_path: str | None,
    ) -> str:
        hw = self.task.resolved_hardware_profile()
        metrics_name = PurePosixPath(output_path).name
        parity_name = PurePosixPath(parity_path).name if parity_path else None
        container_name = self._container_name(source_dir, output_path)
        overlay_dir = f"/tmp/{container_name}-overlay"
        quoted_model_dir = shlex.quote(self.task.model_dir)
        quoted_model_mount = shlex.quote(f"{self.task.model_dir}:{self.task.model_dir}:ro")
        common = (
            f"./target/release/zstep --world-size {hw.world_size} "
            "--python /usr/bin/python3 --worker-root ./worker "
            f"--model-dir {quoted_model_dir} --allreduce-blocks 8 "
            f"--num-kv-pages {self.task.num_kv_pages} --max-step-tokens 4096 --max-decoder-rows 4096 "
            f"--max-pages-per-seq {self.task.max_pages_per_seq} --ready-timeout-secs 900 --step-timeout-secs 180"
        )
        greedy = ""
        if parity_name:
            greedy = (
                f"{common} --max-seqs 16 --max-prefill-chunk {self.task.parity_prefill_chunk} "
                "--verify-greedy-json docs/benchmarks/phase3_vllm_greedy_reference.json "
                f"--verify-output-json /workspace/results/{parity_name} "
                "--require-exact-cases prime_squares_table\n"
            )
        benchmark = (
            f"{common} --max-seqs 128 --max-prefill-chunk 2048 "
            "--max-mixed-step-tokens 3072 --warmup-prompts 16 "
            f"--repeats {repeats} --bench-tiers \"{tiers}\" "
            f"--metrics-output /workspace/results/{metrics_name}\n"
        )
        inner_cmd = (
            'set -euo pipefail\n'
            'trap \'rm -rf /dev/shm/zstep* >/dev/null 2>&1 || true; chown -R "${HOST_UID}:${HOST_GID}" /workspace /fi-cache >/dev/null 2>&1 || true\' EXIT\n'
            + greedy
            + benchmark
        )
        gpus_spec = self._docker_gpus_spec()
        rustup_snip = self._rustup_env_snippet()
        overlay_snip = self._overlay_env_snippet(overlay_dir)
        sudo_pfx = self._sudo_shell_prefix()
        pp_env = (
            f"-e ZSTEP_PIPELINE_PARALLEL_SIZE={hw.pp_size} -e ZSTEP_DECODE_DEVICE_TOKENS=0 "
            f"-e ZSTEP_PP_PREFILL_MICROBATCHES=1 "
            f"-e ZSTEP_PP_DECODE_PIPELINE={'1' if hw.decode_pipeline_1f1b else '0'} "
            if hw.pp_size > 1
            else ""
        )
        worker_env = "".join(
            f"-e {shlex.quote(f'{k}={v}')} "
            for k, v in sorted(self.task.worker_env_overrides.items())
        )
        return f"""set -euo pipefail
trap 'rm -rf {shlex.quote(overlay_dir)} >/dev/null 2>&1 || true' EXIT
{rustup_snip}
cd {shlex.quote(source_dir)}
mkdir -p results
if [ ! -x ./target/release/zstep ]; then
  if [ -d "$HOME/evolith-rsi/cargo-vendor" ]; then
    CARGO_VENDOR_ARGS=(--offline --config "source.crates-io.replace-with='vendored-sources'" --config "source.vendored-sources.directory='$HOME/evolith-rsi/cargo-vendor'")
    cargo "${{CARGO_VENDOR_ARGS[@]}}" test --release -p zs-shm -p zs-scheduler --lib
    cargo "${{CARGO_VENDOR_ARGS[@]}}" build --release -p zs-server
  else
    cargo test --release -p zs-shm -p zs-scheduler --lib
    cargo build --release -p zs-server
  fi
fi
{overlay_snip}
{sudo_pfx}docker run --rm --name {container_name} --gpus {gpus_spec} --shm-size=32g --network=none --entrypoint bash \\
  -v "$PWD:/workspace" -v {quoted_model_mount} -w /workspace \\
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \\
  -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \\
  -e VLLM_HOST_IP=127.0.0.1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \\
  {pp_env}{worker_env}"${{DOCKER_OVERLAY_ARGS[@]}}" {self.task.image} -lc {shlex.quote(inner_cmd)}
"""

    def _remote_run(self, candidate_id: str, stage: str) -> str:
        return f"{self.task.remote_root}/runs/{candidate_id}/{stage}"

    def _upload_and_extract(self, archive: Path, remote_stage: str, source_name: str = "src") -> str:
        self._require_success(self._ssh(["mkdir", "-p", remote_stage]), "create remote stage")
        remote_archive = f"{remote_stage}/source.tar.gz"
        self._require_success(
            self._scp(str(archive), f"{self.destination}:{remote_archive}"), "upload source"
        )
        source_dir = f"{remote_stage}/{source_name}"
        self._require_success(self._ssh(["mkdir", "-p", source_dir]), "create source directory")
        self._require_success(
            self._ssh(["tar", "-xzf", remote_archive, "-C", source_dir]), "extract source"
        )
        return source_dir

    def _cleanup_orphan_container(self, container_name: str) -> None:
        try:
            self._ssh(
                [
                    "bash",
                    "-lc",
                    f"docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true; "
                    f"rm -rf {shlex.quote(f'/tmp/{container_name}-overlay')} >/dev/null 2>&1 || true",
                ],
                timeout=30,
            )
        except Exception:
            pass

    def probe_script(self, source_dir: str, probe_rel_path: str, output_rel_path: str) -> str:
        container_name = self._container_name(source_dir, "gpu-probe")
        overlay_dir = f"/tmp/{container_name}-overlay"
        gpus_spec = self._docker_gpus_spec()
        overlay_snip = self._overlay_env_snippet(overlay_dir)
        sudo_pfx = self._sudo_shell_prefix()
        inner_cmd = (
            "set -euo pipefail\n"
            'trap \'rm -rf /dev/shm/zstep* >/dev/null 2>&1 || true; chown -R "${HOST_UID}:${HOST_GID}" /workspace >/dev/null 2>&1 || true\' EXIT\n'
            "mkdir -p /workspace/results\n"
            "PYTHONPATH=/workspace/worker /usr/bin/python3 "
            f"{shlex.quote(probe_rel_path)} --output {shlex.quote(output_rel_path)}\n"
        )
        return f"""set -euo pipefail
trap 'rm -rf {shlex.quote(overlay_dir)} >/dev/null 2>&1 || true' EXIT
cd {shlex.quote(source_dir)}
{overlay_snip}
{sudo_pfx}docker run --rm --name {container_name} --gpus {gpus_spec} --shm-size=32g --network=none --entrypoint bash \\
  -v "$PWD:/workspace" -w /workspace \\
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 \\
  -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \\
  -e VLLM_HOST_IP=127.0.0.1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \\
  "${{DOCKER_OVERLAY_ARGS[@]}}" {self.task.image} -lc {shlex.quote(inner_cmd)}
"""

    def run_probe(
        self,
        probe_id: str,
        source_archive: Path,
        probe_script_file: Path,
        local_output: Path,
    ) -> Path:
        """Execute a fast isolated 8-GPU operator/collective microbenchmark probe without 475 GiB weights."""
        self._check_gpus()
        self._check_docker()
        stage = self._remote_run(probe_id, "probe")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create probe stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire probe lock")
        container_name = self._container_name(f"{stage}/src", "gpu-probe")
        try:
            source_dir = self._upload_and_extract(source_archive, stage)
            remote_probe = f"{source_dir}/_evolith_probe.py"
            self._require_success(
                self._scp(str(probe_script_file), f"{self.destination}:{remote_probe}"),
                "upload probe script",
            )
            script = self.probe_script(source_dir, "_evolith_probe.py", "results/probe.json")
            try:
                result = self._ssh(["bash", "-lc", script])
            except BaseException:
                self._cleanup_orphan_container(container_name)
                raise
            if result.returncode:
                self._cleanup_orphan_container(container_name)
                detail = (result.stderr + result.stdout).strip()[-4000:]
                err_cls = RemoteInfraError if result.returncode == 255 else RuntimeError
                raise err_cls(f"Remote GPU probe exited {result.returncode}: {detail}")
            return self._download(f"{source_dir}/results/probe.json", local_output)
        finally:
            self._cleanup_remote_stage(stage, lock)

    def gpu_smoke_script(self, source_dir: str, tests: tuple[str, ...]) -> str:
        container_name = self._container_name(source_dir, "gpu-smoke")
        overlay_dir = f"/tmp/{container_name}-overlay"
        test_args = " ".join(shlex.quote(t) for t in tests)
        gpus_spec = self._docker_gpus_spec()
        rustup_snip = self._rustup_env_snippet()
        overlay_snip = self._overlay_env_snippet(overlay_dir)
        sudo_pfx = self._sudo_shell_prefix()
        inner_cmd = (
            "set -euo pipefail\n"
            'trap \'rm -rf /dev/shm/zstep* >/dev/null 2>&1 || true; chown -R "${HOST_UID}:${HOST_GID}" /workspace >/dev/null 2>&1 || true\' EXIT\n'
            "PYTHONPATH=/workspace/worker /usr/bin/python3 -m pytest -q -p no:cacheprovider "
            f"{test_args}\n"
        )
        return f"""set -euo pipefail
trap 'rm -rf {shlex.quote(overlay_dir)} >/dev/null 2>&1 || true' EXIT
{rustup_snip}
cd {shlex.quote(source_dir)}
if [ -f Cargo.toml ]; then
  if [ -d "$HOME/evolith-rsi/cargo-vendor" ]; then
    CARGO_VENDOR_ARGS=(--offline --config "source.crates-io.replace-with='vendored-sources'" --config "source.vendored-sources.directory='$HOME/evolith-rsi/cargo-vendor'")
    cargo "${{CARGO_VENDOR_ARGS[@]}}" test --release -p zs-wire -p zs-kv -p zs-shm -p zs-scheduler --lib
  else
    cargo test --release -p zs-wire -p zs-kv -p zs-shm -p zs-scheduler --lib
  fi
fi
{overlay_snip}
{sudo_pfx}docker run --rm --name {container_name} --gpus {gpus_spec} --shm-size=32g --network=none --entrypoint bash \\
  -v "$PWD:/workspace" -w /workspace \\
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 \\
  -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \\
  -e VLLM_HOST_IP=127.0.0.1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \\
  "${{DOCKER_OVERLAY_ARGS[@]}}" {self.task.image} -lc {shlex.quote(inner_cmd)}
"""

    def gpu_smoke(self, candidate_id: str, source: Path) -> None:
        """
        Runs fast container-level SM120 Python/CUDA smoke tests in seconds before
        committing to full 475 GiB model loading and multi-tier benchmarks.
        """
        if not self.task.gpu_smoke_tests:
            return
        self._check_gpus()
        stage = self._remote_run(candidate_id, "smoke")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create smoke stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire smoke lock")
        container_name = self._container_name(f"{stage}/src", "gpu-smoke")
        try:
            source_dir = self._upload_and_extract(source, stage)
            script = self.gpu_smoke_script(source_dir, self.task.gpu_smoke_tests)
            try:
                result = self._ssh(["bash", "-lc", script])
            except BaseException:
                self._cleanup_orphan_container(container_name)
                raise
            if result.returncode:
                self._cleanup_orphan_container(container_name)
                detail = (result.stderr + result.stdout).strip()[-4000:]
                err_cls = RemoteInfraError if result.returncode == 255 else RuntimeError
                raise err_cls(f"Remote GPU smoke exited {result.returncode}: {detail}")
        finally:
            self._cleanup_remote_stage(stage, lock)

    def _run_benchmark(
        self,
        source_dir: str,
        output_path: str,
        tiers: str,
        repeats: int,
        parity_path: str | None,
    ) -> None:
        self._check_gpus()
        container_name = self._container_name(source_dir, output_path)
        script = self.benchmark_script(source_dir, output_path, tiers, repeats, parity_path)
        try:
            result = self._ssh(["bash", "-lc", script])
        except BaseException:
            self._cleanup_orphan_container(container_name)
            raise
        if result.returncode:
            self._cleanup_orphan_container(container_name)
            detail = (result.stderr + result.stdout).strip()[-4000:]
            err_cls = RemoteInfraError if result.returncode == 255 else RuntimeError
            raise err_cls(f"Remote benchmark exited {result.returncode}: {detail}")

    def _download(self, remote_path: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._require_success(
            self._scp(f"{self.destination}:{remote_path}", str(local_path)),
            "download evidence",
        )
        return local_path

    def _cleanup_remote_stage(self, remote_stage: str, lock: str) -> None:
        try:
            if self.task.keep_remote_stage:
                self._ssh(["rmdir", lock], timeout=30)
            else:
                self._ssh(["rm", "-rf", remote_stage], timeout=60)
        except Exception:
            pass

    def _cache_has_all_screen_tiers(self, path: Path) -> bool:
        expected = {
            item.split(":", 1)[0] for item in self.task.screen_tiers.split(";") if item
        }
        rows = self._load_valid_checkpoint(path)
        if not rows:
            return False
        present = {str(row.get("label", "")) for row in rows if isinstance(row, dict)}
        return expected.issubset(present)

    def screen_baseline(self, baseline_hash: str, baseline_archive: Path) -> Path:
        """
        Measures and caches the frozen parent baseline on `screen_tiers` once per
        parent hash so every candidate's screen pass has an exact throughput/latency
        reference without redundant baseline runs.
        """
        cached = self.task.baseline_cache_dir / f"{baseline_hash[:16]}-screen-metrics.json"
        if cached.is_file() and self._cache_has_all_screen_tiers(cached):
            return cached
        stage_id = f"baseline-{baseline_hash[:12]}"
        stage = self._remote_run(stage_id, "screen")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create baseline screen stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire baseline screen lock")
        try:
            source_dir = self._upload_and_extract(baseline_archive, stage)
            metrics_remote = f"{stage}/metrics.json"
            self._run_benchmark(
                source_dir, metrics_remote, self.task.screen_tiers, 1, None
            )
            self._download(f"{source_dir}/results/metrics.json", cached)
            return cached
        finally:
            self._cleanup_remote_stage(stage, lock)

    def screen(self, candidate_id: str, source: Path, evidence: Path) -> Path:
        stage = self._remote_run(candidate_id, "screen")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create screen stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire screen lock")
        try:
            source_dir = self._upload_and_extract(source, stage)
            metrics_remote = f"{stage}/metrics.json"
            parity_remote = f"{stage}/parity.json"
            self._run_benchmark(
                source_dir, metrics_remote, self.task.screen_tiers, 1, parity_remote
            )
            self._download(f"{source_dir}/results/parity.json", evidence / "screen-parity.json")
            out = self._download(
                f"{source_dir}/results/metrics.json", evidence / "screen-metrics.json"
            )
            return out
        finally:
            self._cleanup_remote_stage(stage, lock)

    @staticmethod
    def _load_valid_checkpoint(path: Path) -> list[dict[str, object]] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tiers = payload.get("tiers")
            if isinstance(tiers, list) and tiers:
                return tiers
        except (OSError, ValueError):
            pass
        return None

    def verify(
        self,
        candidate_id: str,
        baseline_archive: Path,
        candidate_archive: Path,
        evidence: Path,
    ) -> tuple[Path, Path, Path]:
        candidate_out = evidence / "candidate-metrics.json"
        baseline_out = evidence / "baseline-metrics.json"
        parity_local = evidence / "verify-parity.json"
        if (
            self._load_valid_checkpoint(candidate_out) is not None
            and self._load_valid_checkpoint(baseline_out) is not None
            and parity_local.is_file()
        ):
            return candidate_out, baseline_out, parity_local
        stage = self._remote_run(candidate_id, "verify")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create verify stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire verify lock")
        try:
            baseline_dir: str | None = None
            candidate_dir: str | None = None
            combined: dict[str, list[dict[str, object]]] = {"candidate": [], "baseline": []}
            for repeat in range(self.task.repeats):
                for name in ("candidate", "baseline"):
                    output_name = f"{name}-r{repeat}.json"
                    local_round = evidence / output_name
                    need_parity = name == "candidate" and repeat == 0
                    cached_rows = self._load_valid_checkpoint(local_round)
                    if cached_rows is not None and (not need_parity or parity_local.is_file()):
                        for row in cached_rows:
                            row_copy = dict(row)
                            row_copy["repeat"] = repeat
                            combined[name].append(row_copy)
                        continue

                    if name == "candidate":
                        if candidate_dir is None:
                            candidate_dir = self._upload_and_extract(
                                candidate_archive, stage, "candidate"
                            )
                        source_dir = candidate_dir
                    else:
                        if baseline_dir is None:
                            baseline_dir = self._upload_and_extract(
                                baseline_archive, stage, "baseline"
                            )
                        source_dir = baseline_dir

                    parity_remote = f"{stage}/parity.json" if need_parity else None
                    self._run_benchmark(
                        source_dir,
                        f"{stage}/{output_name}",
                        self.task.verify_tiers,
                        1,
                        parity_remote,
                    )
                    local = self._download(
                        f"{source_dir}/results/{output_name}", local_round
                    )
                    if need_parity:
                        self._download(f"{source_dir}/results/parity.json", parity_local)
                    payload = json.loads(local.read_text(encoding="utf-8"))
                    for row in payload["tiers"]:
                        row["repeat"] = repeat
                        combined[name].append(row)

            candidate_metrics = evidence / "candidate-metrics.json"
            baseline_metrics = evidence / "baseline-metrics.json"
            candidate_metrics.write_text(
                json.dumps({"tiers": combined["candidate"]}, indent=2), encoding="utf-8"
            )
            baseline_metrics.write_text(
                json.dumps({"tiers": combined["baseline"]}, indent=2), encoding="utf-8"
            )
            (evidence / "verification-method.json").write_text(
                json.dumps(
                    {
                        "round_major_interleaved": True,
                        "repeats": self.task.repeats,
                        "order": ["candidate", "baseline"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return candidate_metrics, baseline_metrics, parity_local
        finally:
            self._cleanup_remote_stage(stage, lock)
