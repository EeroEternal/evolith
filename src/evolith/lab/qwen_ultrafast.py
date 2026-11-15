"""Bounded remote optimization for the Rust/CUDA Qwen Ultrafast runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from evolith.lab.contracts import StrictModel, content_hash
from evolith.lab.measurement import decide
from evolith.lab.store import RunStore, new_run_id, write_json


class QwenKernelConfig(StrictModel):
    threads: Literal[64, 128, 256, 512]

    @property
    def key(self) -> str:
        return f"bf16-gemv-t{self.threads}"


class QwenAdapterConfig(StrictModel):
    remote: str = Field(pattern=r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$")
    port: int = Field(default=22, ge=1, le=65535)
    local_repo: Path
    remote_repo: Path
    model: str = Field(min_length=1, max_length=512)
    nccl_library: Path
    output: Path
    control_path: Path | None = None
    gpus: str = Field(default="0,1,2,3,4,5,6,7", pattern=r"^[0-7](,[0-7])*$")
    prompt: str = Field(default="Hello", min_length=1, max_length=256)
    decode: int = Field(default=64, ge=8, le=256)
    max_seq: int = Field(default=80, ge=9, le=2048)
    rounds: int = Field(default=19, ge=5, le=101)
    iterations: int = Field(default=128, ge=1, le=1000)
    model_pairs: int = Field(default=5, ge=5, le=19)
    seconds: int = Field(default=7200, ge=60, le=7200)


class QwenUltrafastAdapter:
    def __init__(self, config: QwenAdapterConfig):
        self.config = config
        self.started = time.monotonic()
        self.run_id = new_run_id()
        self.directory = config.output.resolve() / self.run_id
        self.store = RunStore(config.output / "runs.sqlite3")

    def _remaining(self) -> float:
        remaining = self.config.seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError("Qwen Ultrafast adapter wall-time budget exhausted.")
        return remaining

    def _ssh(self) -> list[str]:
        command = ["ssh", "-p", str(self.config.port), "-o", "BatchMode=yes"]
        if self.config.control_path:
            command += ["-o", f"ControlPath={self.config.control_path}"]
        return command

    def _local(self, command: list[str], directory: Path, timeout: int | None = None) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        timeout = min(timeout or 600, max(1, int(self._remaining())))
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            (directory / "worker.log").write_text(stdout + stderr)
            raise TimeoutError(f"Command exceeded {timeout}s; process group terminated.") from None
        (directory / "worker.log").write_text(stdout + stderr)
        if process.returncode:
            raise RuntimeError(f"Command exited {process.returncode}; see {directory / 'worker.log'}")
        return stdout

    def _remote(self, script: str, directory: Path, timeout: int = 600) -> str:
        bounded = min(timeout, max(1, int(self._remaining())))
        command = (
            f"cd {shlex.quote(str(self.config.remote_repo))} && "
            f"timeout --signal=TERM --kill-after=10 {bounded + 15} bash -lc {shlex.quote(script)}"
        )
        return self._local([*self._ssh(), self.config.remote, command], directory, bounded)

    def _source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        root = self.config.local_repo.resolve()
        for path in sorted(root.glob("crates/**/*.rs")) + sorted(root.glob("crates/**/*.cu")) + sorted(root.glob("crates/**/*.cuh")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
            path = root / name
            if path.exists():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def prepare(self) -> dict:
        root = self.config.local_repo.resolve()
        if not (root / "Cargo.toml").is_file():
            raise ValueError("Local Qwen Ultrafast repository is missing Cargo.toml.")
        identity = content_hash({
            "source": self._source_fingerprint(),
            "model": self.config.model,
            "gpus": self.config.gpus,
            "prompt": self.config.prompt,
            "decode": self.config.decode,
        })
        ssh = " ".join(self._ssh())
        self._local([
            "rsync", "-az", "--exclude", "target", "--exclude", "models", "--exclude", ".git",
            "-e", ssh, f"{root}/", f"{self.config.remote}:{self.config.remote_repo}/",
        ], self.directory / "prepare" / "sync")
        self._remote(
            "export PATH=\"$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH\" && cargo build --release --workspace",
            self.directory / "prepare" / "build", 300,
        )
        processes = self._remote(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            self.directory / "prepare" / "idle", 30,
        ).strip()
        if processes:
            raise RuntimeError("Remote GPUs have active processes; refusing contended measurements.")
        return {"identity": identity, "source": self._source_fingerprint()}

    def benchmark(self, config: QwenKernelConfig, directory: Path, incumbent: int = 128) -> dict:
        baseline = QwenKernelConfig(threads=incumbent)
        output = self._remote(
            f"target/release/kernel_bench --threads {config.threads} "
            f"--baseline-threads {baseline.threads} "
            f"--rounds {self.config.rounds} --iterations {self.config.iterations}",
            directory, 180,
        )
        result = json.loads(output)
        if (result.get("target") != "bf16-gemv"
                or result.get("config") != {"threads": config.threads}
                or result.get("baseline_config") != {"threads": baseline.threads}):
            raise ValueError("Kernel benchmark response does not match the requested candidate/incumbent.")
        return result

    def model_run(self, threads: int, directory: Path) -> dict:
        config = QwenKernelConfig(threads=threads)
        output = self._remote(
            f"export QWEN_NCCL_LIBRARY={shlex.quote(str(self.config.nccl_library))} "
            f"QWEN_GEMV_THREADS={config.threads} && "
            f"target/release/run --model {shlex.quote(self.config.model)} "
            f"--gpus {self.config.gpus} --prompt {shlex.quote(self.config.prompt)} "
            f"--decode {self.config.decode} --max-seq {self.config.max_seq} --graph",
            directory, 240,
        )
        result = json.loads(output)
        if len(result.get("tokens", [])) != self.config.decode or result.get("decode_seconds", 0) <= 0:
            raise ValueError("Invalid full-model response.")
        return result

    def run(self, candidates: list[int], incumbent: int = 128) -> dict:
        self.directory.mkdir(parents=True, exist_ok=False)
        incumbent = QwenKernelConfig(threads=incumbent).threads
        configs = list({QwenKernelConfig(threads=value).threads: value for value in candidates}.values())
        if not 1 <= len(configs) <= 12:
            raise ValueError("Between 1 and 12 distinct candidates are required.")
        report = {
            "run_id": self.run_id,
            "status": "running",
            "scope": "bf16-gemv block-size search plus exact-token full-model promotion",
            "deployment_approved": False,
            "incumbent": {"threads": incumbent},
            "candidates": [],
            "stop_reason": None,
        }
        write_json(self.directory / "report.json", report)
        try:
            provenance = self.prepare()
            report["identity"] = provenance["identity"]
            report["source_fingerprint"] = provenance["source"]
            for value in configs:
                candidate = QwenKernelConfig(threads=value)
                if candidate.threads == incumbent:
                    continue
                cycle = {"config": candidate.model_dump(), "key": candidate.key, "approved": False}
                report["candidates"].append(cycle)
                directory = self.directory / candidate.key
                benchmark = self.benchmark(candidate, directory / "micro", incumbent)
                cycle["benchmark"] = benchmark
                timing = decide(
                    benchmark["baseline_us"], benchmark["candidate_us"], seed=2026,
                )
                cycle["micro_measurement"] = timing.model_dump()
                cycle["micro_accepted"] = bool(benchmark.get("correct")) and timing.accepted
                if cycle["micro_accepted"]:
                    baseline_us, candidate_us = [], []
                    tokens_match = True
                    trials = []
                    for pair in range(self.config.model_pairs):
                        order = (incumbent, candidate.threads) if pair % 2 == 0 else (candidate.threads, incumbent)
                        results = {}
                        for index, threads in enumerate(order):
                            results[threads] = self.model_run(threads, directory / f"model-{pair:02d}-{index}")
                        baseline, proposed = results[incumbent], results[candidate.threads]
                        same = baseline["tokens"] == proposed["tokens"]
                        tokens_match = tokens_match and same
                        baseline_us.append(baseline["decode_seconds"] * 1e6)
                        candidate_us.append(proposed["decode_seconds"] * 1e6)
                        trials.append({
                            "order": list(order), "tokens_match": same,
                            "baseline_decode_seconds": baseline["decode_seconds"],
                            "candidate_decode_seconds": proposed["decode_seconds"],
                        })
                        if not same:
                            break
                    cycle["model_trials"] = trials
                    cycle["tokens_match"] = tokens_match
                    if len(baseline_us) == self.config.model_pairs:
                        model_timing = decide(baseline_us, candidate_us, seed=2026)
                        cycle["model_measurement"] = model_timing.model_dump()
                        cycle["approved"] = tokens_match and model_timing.accepted
                    else:
                        cycle["approved"] = False
                    if cycle["approved"]:
                        parent = report["incumbent"].copy()
                        incumbent = candidate.threads
                        report["incumbent"] = candidate.model_dump()
                        report["deployment_approved"] = True
                        cycle["lineage"] = parent
                    write_json(directory / "report.json", cycle)
                    self.store.record(f"{self.run_id}-{candidate.key}", "qwen_ultrafast_candidate", cycle)
                else:
                    write_json(directory / "report.json", cycle)
                    self.store.record(f"{self.run_id}-{candidate.key}", "qwen_ultrafast_candidate", cycle)
                write_json(self.directory / "report.json", report)
            report["status"] = "stopped"
            report["stop_reason"] = "candidate_budget"
        except Exception as exc:
            report["status"] = "failed"
            report["stop_reason"] = "execution_failure"
            report["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
            raise
        finally:
            report["elapsed_sec"] = time.monotonic() - self.started
            write_json(self.directory / "report.json", report)
            self.store.record(self.run_id, "qwen_ultrafast_session", report)
        return {
            "status": report["status"], "run_id": self.run_id,
            "report": str(self.directory / "report.json"),
            "deployment_approved": report["deployment_approved"],
            "incumbent": report["incumbent"], "stop_reason": report["stop_reason"],
        }
