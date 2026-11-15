from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
from typing import Callable

from evolith.zstep.task import ZStepRSITask

Executor = Callable[..., subprocess.CompletedProcess[str]]


class RemotePreflightError(RuntimeError):
    pass


def _sanitize_container_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return cleaned[:48] or "run"


class ZStepRemoteRunner:
    def __init__(self, task: ZStepRSITask, executor: Executor | None = None) -> None:
        self.task = task
        self.executor = executor or self._execute
        self.destination = f"{task.remote_user}@{task.remote_host}"

    @staticmethod
    def _execute(
        args: list[str], *, cwd: Path | None = None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)

    def _ssh(self, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        command = ["sshpass", "-e", "ssh"] if os.environ.get("SSHPASS") else ["ssh"]
        command.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if not os.environ.get("SSHPASS"):
            command.extend(["-o", "BatchMode=yes"])
        command.extend([self.destination, shlex.join(args)])
        return self.executor(
            command, timeout=timeout or self.task.remote_timeout_sec
        )

    def _scp(self, source: str, destination: str) -> subprocess.CompletedProcess[str]:
        command = ["sshpass", "-e", "scp"] if os.environ.get("SSHPASS") else ["scp"]
        command.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if not os.environ.get("SSHPASS"):
            command.extend(["-o", "BatchMode=yes"])
        command.extend([source, destination])
        return self.executor(command, timeout=self.task.remote_timeout_sec)

    @staticmethod
    def _require_success(result: subprocess.CompletedProcess[str], action: str) -> str:
        if result.returncode:
            detail = (result.stderr + result.stdout).strip()
            raise RuntimeError(f"{action} exited {result.returncode}: {detail}")
        return result.stdout

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
        for row in rows:
            index, memory = [part.strip() for part in row.split(",", 1)]
            used_mib = int(memory.split()[0])
            if used_mib > self.task.max_gpu_idle_mib:
                raise RemotePreflightError(f"GPU {index} is occupied: {memory}")
        if len(rows) != 8:
            raise RemotePreflightError(f"Expected 8 GPUs, found {len(rows)}")

    def preflight(self) -> None:
        self._check_gpus()
        docker = self._ssh(["docker", "info", "--format", "{{.ServerVersion}}"])
        if docker.returncode:
            detail = (docker.stderr + docker.stdout).strip()
            raise RemotePreflightError(
                f"Docker access is required for {self.destination}; {detail}. "
                "Grant the user Docker access; Evolith will not use sudo."
            )
        model_check = self._ssh(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python3",
                "-v",
                f"{self.task.model_dir}:{self.task.model_dir}:ro",
                self.task.image,
                "-c",
                (
                    "import json; "
                    f"print(json.load(open('{self.task.model_dir}/config.json'))['model_type'])"
                ),
            ]
        )
        if model_check.returncode or model_check.stdout.strip() != self.task.model_type:
            raise RemotePreflightError(
                f"Model must be {self.task.model_type} at {self.task.model_dir}"
            )

    @staticmethod
    def _container_name(source_dir: str, output_path: str) -> str:
        parts = PurePosixPath(source_dir).parts
        scope = "-".join(parts[-3:]) if len(parts) >= 3 else "-".join(parts)
        stem = PurePosixPath(output_path).stem
        return f"evolith-{_sanitize_container_token(f'{scope}-{stem}')}"

    def benchmark_script(
        self,
        source_dir: str,
        output_path: str,
        tiers: str,
        repeats: int,
        parity_path: str | None,
    ) -> str:
        metrics_name = PurePosixPath(output_path).name
        parity_name = PurePosixPath(parity_path).name if parity_path else None
        container_name = self._container_name(source_dir, output_path)
        overlay_dir = f"/tmp/{container_name}-overlay"
        common = (
            "./target/release/zstep --world-size 8 "
            "--python /usr/bin/python3 --worker-root ./worker "
            f"--model-dir {self.task.model_dir} --allreduce-blocks 8 "
            "--num-kv-pages 2560 --max-step-tokens 4096 --max-decoder-rows 4096 "
            "--max-pages-per-seq 256 --ready-timeout-secs 900 --step-timeout-secs 180"
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
            'trap \'chown -R "${HOST_UID}:${HOST_GID}" /workspace /fi-cache >/dev/null 2>&1 || true\' EXIT\n'
            + greedy
            + benchmark
        )
        return f"""set -euo pipefail
trap 'rm -rf {shlex.quote(overlay_dir)} >/dev/null 2>&1 || true' EXIT
cd {shlex.quote(source_dir)}
mkdir -p results
if [ ! -x ./target/release/zstep ]; then
  cargo test --release -p zs-shm -p zs-scheduler --lib
  cargo build --release -p zs-server
fi
./scripts/prepare_sm120_overlay.sh {shlex.quote(overlay_dir)}
DOCKER_OVERLAY_ARGS=($(./scripts/prepare_sm120_overlay.sh --print-docker-args {shlex.quote(overlay_dir)}))
docker run --rm --name {container_name} --gpus all --ipc=host --network=host --entrypoint bash \\
  -v "$PWD:/workspace" -v "{self.task.model_dir}:{self.task.model_dir}:ro" -w /workspace \\
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \\
  -e OMP_NUM_THREADS=1 -e NCCL_P2P_DISABLE=1 -e NCCL_P2P_LEVEL=LOC \\
  "${{DOCKER_OVERLAY_ARGS[@]}}" {self.task.image} -lc {shlex.quote(inner_cmd)}
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
                    f"rm -rf {shlex.quote(f'/tmp/{container_name}-overlay')} /dev/shm/zstep* >/dev/null 2>&1 || true",
                ],
                timeout=30,
            )
        except Exception:
            pass

    def gpu_smoke_script(self, source_dir: str, tests: tuple[str, ...]) -> str:
        container_name = self._container_name(source_dir, "gpu-smoke")
        overlay_dir = f"/tmp/{container_name}-overlay"
        test_args = " ".join(shlex.quote(t) for t in tests)
        inner_cmd = (
            "set -euo pipefail\n"
            "PYTHONPATH=/workspace/worker /usr/bin/python3 -m pytest -q "
            f"{test_args}\n"
        )
        return f"""set -euo pipefail
trap 'rm -rf {shlex.quote(overlay_dir)} /dev/shm/zstep* >/dev/null 2>&1 || true' EXIT
cd {shlex.quote(source_dir)}
./scripts/prepare_sm120_overlay.sh {shlex.quote(overlay_dir)}
DOCKER_OVERLAY_ARGS=($(./scripts/prepare_sm120_overlay.sh --print-docker-args {shlex.quote(overlay_dir)}))
docker run --rm --name {container_name} --gpus all --ipc=host --network=host --entrypoint bash \\
  -v "$PWD:/workspace" -w /workspace \\
  -e OMP_NUM_THREADS=1 -e NCCL_P2P_DISABLE=1 -e NCCL_P2P_LEVEL=LOC \\
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
                detail = (result.stderr + result.stdout).strip()
                raise RuntimeError(f"Remote GPU smoke exited {result.returncode}: {detail}")
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
            detail = (result.stderr + result.stdout).strip()
            raise RuntimeError(f"Remote benchmark exited {result.returncode}: {detail}")

    def _download(self, remote_path: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._require_success(
            self._scp(f"{self.destination}:{remote_path}", str(local_path)),
            "download evidence",
        )
        return local_path

    def _cleanup_remote_stage(self, remote_stage: str, lock: str) -> None:
        if self.task.keep_remote_stage:
            self._ssh(["rmdir", lock], timeout=30)
        else:
            self._ssh(["rm", "-rf", remote_stage], timeout=60)

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
        succeeded = False
        try:
            source_dir = self._upload_and_extract(baseline_archive, stage)
            metrics_remote = f"{stage}/metrics.json"
            self._run_benchmark(
                source_dir, metrics_remote, self.task.screen_tiers, 1, None
            )
            self._download(f"{source_dir}/results/metrics.json", cached)
            succeeded = True
            return cached
        finally:
            if succeeded:
                self._cleanup_remote_stage(stage, lock)
            else:
                self._ssh(["rmdir", lock], timeout=30)

    def screen(self, candidate_id: str, source: Path, evidence: Path) -> Path:
        stage = self._remote_run(candidate_id, "screen")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create screen stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire screen lock")
        succeeded = False
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
            succeeded = True
            return out
        finally:
            if succeeded:
                self._cleanup_remote_stage(stage, lock)
            else:
                self._ssh(["rmdir", lock], timeout=30)

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
        stage = self._remote_run(candidate_id, "verify")
        lock = f"{stage}/.lock"
        self._require_success(self._ssh(["mkdir", "-p", stage]), "create verify stage")
        self._require_success(self._ssh(["mkdir", lock]), "acquire verify lock")
        parity_local = evidence / "verify-parity.json"
        succeeded = False
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
            succeeded = True
            return candidate_metrics, baseline_metrics, parity_local
        finally:
            if succeeded:
                self._cleanup_remote_stage(stage, lock)
            else:
                self._ssh(["rmdir", lock], timeout=30)
