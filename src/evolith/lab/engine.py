"""Trusted Qwen runtime configuration, not generated Python or kernel source."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from evolith.lab.contracts import StrictModel

MODEL_NAME = "evolith-qwen"
CONTEXT_LIMIT = 2048


class EngineConfig(StrictModel):
    tensor_parallel: Literal[2, 4] = 4
    graph: Literal["native", "full_decode", "piecewise"] = "native"
    max_num_seqs: Literal[1, 4, 8] = 8
    prefill_tokens: Literal[256, 512, 1024, 2048] = 2048
    memory_fraction: Literal[0.90, 0.95] = 0.90

    @property
    def key(self) -> str:
        return (f"tp{self.tensor_parallel}-{self.graph}-s{self.max_num_seqs}"
                f"-p{self.prefill_tokens}-m{self.memory_fraction}")


class EngineProposal(StrictModel):
    candidates: list[EngineConfig] = Field(min_length=1, max_length=3)


def capacity_neighbors(failed: EngineConfig) -> list[EngineConfig]:
    """After measured capacity failure, enumerate resource-monotone legal neighbors."""
    options = [
        failed.model_copy(update={"max_num_seqs": seqs, "prefill_tokens": tokens,
                                  "memory_fraction": memory})
        for seqs in (1, 4, 8) for tokens in (256, 512, 1024, 2048) for memory in (0.90, 0.95)
        if seqs <= failed.max_num_seqs and tokens <= failed.prefill_tokens
        and memory >= failed.memory_fraction
        and (seqs < failed.max_num_seqs or tokens < failed.prefill_tokens
             or memory > failed.memory_fraction)
    ]
    return sorted(options, key=lambda c: (c.max_num_seqs * c.prefill_tokens,
                                          -c.memory_fraction, c.max_num_seqs))


class EngineJob(StrictModel):
    model_path: Path
    output: Path
    config: EngineConfig
    rounds: int = Field(default=5, ge=5, le=15)
    max_tokens: int = Field(default=64, ge=16, le=128)
    seed: int = Field(default=2031, ge=0)
    propose: bool = False
    history: list[dict] = Field(default_factory=list, max_length=8)
    reference_report: Path | None = None


class EngineArtifact(StrictModel):
    schema_version: Literal[1] = 1
    config: EngineConfig
    source_hash: str
    model_fingerprint: str
    environment: dict
    workload_fingerprint: str
    selection: Literal["native_baseline", "approved_successor"]
    baseline_proof: dict | None = None
    candidate_proof: dict | None = None
    fresh_baseline_proof: dict | None = None
    fresh_candidate_proof: dict | None = None


def validate_artifact(artifact: EngineArtifact) -> None:
    from evolith.lab.engine_evolution import combined
    from evolith.lab.latency import compare_trials

    if artifact.source_hash != source_hash():
        raise ValueError("Engine source changed since validation.")
    if artifact.selection == "native_baseline":
        if artifact.config != EngineConfig():
            raise ValueError("Missing validated strong native baseline.")
        for proof in (artifact.baseline_proof, artifact.fresh_baseline_proof):
            if (not proof or proof.get("status") != "measured"
                    or proof.get("correct") is not True
                    or proof.get("config") != artifact.config.model_dump()
                    or any(proof.get(key) != getattr(artifact, key) for key in (
                        "source_hash", "model_fingerprint", "environment", "workload_fingerprint",
                    ))):
                raise ValueError("Missing compatible native restart proof.")
        combined(artifact.baseline_proof, artifact.fresh_baseline_proof)
    if artifact.selection == "approved_successor":
        if (not artifact.baseline_proof or not artifact.candidate_proof
                or not artifact.fresh_baseline_proof or not artifact.fresh_candidate_proof):
            raise ValueError("Missing promotion proof.")
        if (artifact.candidate_proof["config"] != artifact.config.model_dump()
                or artifact.candidate_proof["workload_fingerprint"] != artifact.workload_fingerprint
                or not compare_trials(artifact.baseline_proof, artifact.candidate_proof)["accepted"]
                or not compare_trials(artifact.fresh_baseline_proof, artifact.fresh_candidate_proof)["accepted"]):
            raise ValueError("Promotion evidence fails independent revalidation.")


def serve(profile: Path, model_path: Path, port: int = 18080, container: bool = False) -> None:
    import os

    import torch

    from evolith.lab.gpu import environment, model_info

    artifact = EngineArtifact.model_validate_json(profile.read_text())
    validate_artifact(artifact)
    if container and (not Path("/.dockerenv").exists()
                      or os.environ.get("EVOLITH_LOOPBACK_PUBLISH") != "1"):
        raise ValueError("Container interface requires the controlled loopback publishing helper.")
    if (model_info(model_path)["fingerprint"] != artifact.model_fingerprint
            or runtime_identity(environment()) != artifact.environment
            or torch.cuda.device_count() < artifact.config.tensor_parallel):
        raise ValueError("Serving model/runtime/GPU allocation differs from the artifact.")
    command = server_command(artifact.config, model_path, port, "0.0.0.0" if container else "127.0.0.1")
    os.execv(command[0], command)


def source_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("engine.py", "latency.py", "engine_worker.py", "engine_evolution.py",
                 "measurement.py", "contracts.py", "prepublication.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def server_command(config: EngineConfig, model: Path, port: int,
                   host: str = "127.0.0.1") -> list[str]:
    if host not in ("127.0.0.1", "0.0.0.0") or not 1024 <= port <= 65535:
        raise ValueError("Invalid controlled server address.")
    # Native means the backend's actual defaults, not a restricted capture set.
    compilation: dict = {}
    if config.graph != "native":
        compilation["cudagraph_mode"] = {
            "full_decode": "FULL_DECODE_ONLY", "piecewise": "PIECEWISE",
        }[config.graph]
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model.resolve()), "--served-model-name", MODEL_NAME,
        "--host", host, "--port", str(port), "--dtype", "bfloat16",
        "--tensor-parallel-size", str(config.tensor_parallel),
        "--max-model-len", str(CONTEXT_LIMIT),
        "--max-num-seqs", str(config.max_num_seqs),
        "--max-num-batched-tokens", str(config.prefill_tokens),
        "--gpu-memory-utilization", str(config.memory_fraction),
        "--no-enable-prefix-caching", "--disable-log-stats",
        "--generation-config", "vllm", "--seed", "2026",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
    ]
    if compilation:
        command += ["--compilation-config", json.dumps(compilation)]
    return command


def runtime_identity(env: dict) -> dict:
    return {key: env.get(key) for key in ("gpu", "capability", "packages", "cuda")}
