from __future__ import annotations

from enum import Enum
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from evolith.models import EvolutionPlane


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    LOCAL_REJECTED = "local_rejected"
    SCREEN_REJECTED = "screen_rejected"
    VERIFY_REJECTED = "verify_rejected"
    ACCEPTED = "accepted"
    FAILED = "failed"


class TierMetric(BaseModel):
    label: str
    repeat: int
    output_throughput_tok_s: float
    median_ttft_ms: float
    mean_tpot_ms: float
    p99_itl_ms: float
    steps: int | None = None
    overlapped_decode_steps: int | None = None
    mean_acceptance_length: float | None = None
    prefix_cache_hit_rate: float | None = None
    idle_kv_cache_usage: float | None = None
    non_spec_throughput_tok_s: float | None = None


class RedTeamFinding(BaseModel):
    category: str
    severity: str
    file_path: str
    message: str


class RedTeamReport(BaseModel):
    passed: bool
    spec_pins_verified: bool = True
    checked_categories: tuple[str, ...] = ()
    findings: list[RedTeamFinding] = Field(default_factory=list)


class ProposalMetadata(BaseModel):
    hypothesis: str = "Focused runtime/kernel throughput optimization"
    niche_key: str = "(plane=auto,target=deepseek_v41)"
    patch_summary: str = ""


class PatchStats(BaseModel):
    patch: bytes
    changed_lines: int
    changed_files: tuple[str, ...] = ()
    plane: EvolutionPlane = EvolutionPlane.L2_SCHEDULER


class ScreenDecision(BaseModel):
    passed: bool
    gain: float = 0.0
    metric_delta: str = ""
    reasons: list[str] = Field(default_factory=list)


class VerificationDecision(BaseModel):
    accepted: bool
    geomean_gain: float
    net_gate_ratio: float = 1.0
    metric_delta: str = ""
    reasons: list[str] = Field(default_factory=list)


class ZStepRSITask(BaseModel):
    project_root: Path
    source_repo: Path
    baseline_sha: str = "2bea5a1"
    remote_user: str = Field(
        default_factory=lambda: os.environ.get("EVOLITH_ZSTEP_USER", "ubuntu")
    )
    remote_host: str = Field(
        default_factory=lambda: os.environ.get("EVOLITH_ZSTEP_HOST", "")
    )
    remote_root: str = "evolith-rsi"
    gpu_devices: str = Field(
        default_factory=lambda: os.environ.get(
            "EVOLITH_ZSTEP_GPUS", "0,1,2,3,4,5,6,7"
        )
    )
    remote_rustup_root: str = Field(
        default_factory=lambda: os.environ.get("EVOLITH_ZSTEP_RUSTUP_ROOT", "")
    )
    model_dir: str = Field(
        default_factory=lambda: os.environ.get(
            "EVOLITH_ZSTEP_MODEL_DIR", "/models/DeepSeek-V4.1-Flash"
        )
    )
    model_id: str = "DeepSeek-V4.1-Flash"
    model_type: str = "deepseek_v41"
    image: str = "vllm/vllm-openai:v0.31.0"
    max_generations: int = Field(default=1, ge=1, le=100)
    max_changed_lines: int = Field(default=300, ge=1)
    complexity_free_lines: int = Field(default=120, ge=0)
    complexity_penalty_weight: float = Field(default=0.0002, ge=0.0)
    local_timeout_sec: int = Field(default=600, ge=1)
    remote_timeout_sec: int = Field(default=7200, ge=60)
    max_gpu_idle_mib: int = Field(default=32, ge=0)
    repeats: int = Field(default=3, ge=1)
    min_geomean_gain: float = 0.03
    max_tier_throughput_regression: float = 0.03
    max_ttft_regression: float = 0.05
    max_tpot_regression: float = 0.05
    max_p99_itl_regression: float = 0.05
    max_tier_cv: float = Field(default=0.10, ge=0.0)
    max_screen_throughput_regression: float = 0.02
    max_screen_latency_regression: float = 0.08
    keep_remote_stage: bool = False
    sandbox_root: Path | None = None
    campaign: str = "none"
    propose_only: bool = False
    proposer_backend: str = "kimi"
    proposer_api_base: str = "http://127.0.0.1:8080/v1"
    proposer_model: str = "DeepSeek-V4.1-Flash"
    enable_gpu_smoke: bool = False
    gpu_smoke_tests: tuple[str, ...] = (
        "worker/tests/test_shm_layout_parity.py",
        "worker/tests/test_p2p_probe.py",
    )
    premise_strike_limit: int = Field(default=2, ge=1)
    parity_prefill_chunk: int = Field(default=24, ge=1)
    max_cross_mode_logit_drift: float = Field(default=8.0, ge=0.0)
    screen_tiers: str = "in512_c1:1:8:512:64;in512_c32:32:64:512:64"
    verify_tiers: str = (
        "in512_c16:16:48:512:256;in512_c128:128:256:512:256;"
        "in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256"
    )
    pinned_spec_files: tuple[str, ...] = (
        "docs/benchmarks/phase3_vllm_greedy_reference.json",
        "scripts/prepare_sm120_overlay.sh",
        "crates/zs-wire/src/lib.rs",
        "crates/zs-server/src/lib.rs",
        "worker/zstep_worker/shm_bridge.py",
        "worker/tests/test_shm_layout_parity.py",
    )
    allowed_roots: tuple[str, ...] = (
        "worker/zstep_worker/models/deepseek_v41.py",
        "worker/zstep_worker/models/deepseek_v41_dspark.py",
        "worker/zstep_worker/runtime_loop.py",
        "worker/zstep_worker/p2p_probe.py",
        "worker/zstep_worker/spec_decode.py",
        "worker/zstep_worker/kernels_sm120",
        "crates/zs-scheduler/src",
    )

    @property
    def upstream_dir(self) -> Path:
        return self.project_root / "targets/zstep/upstream"

    @property
    def candidates_dir(self) -> Path:
        return self.project_root / "targets/zstep/candidates"

    @property
    def state_dir(self) -> Path:
        return self.project_root / ".evolith/zstep-rsi"

    @property
    def baseline_cache_dir(self) -> Path:
        return self.state_dir / "baseline-cache"

    @model_validator(mode="after")
    def validate_tiers(self) -> ZStepRSITask:
        public = {item.split(":", 1)[0] for item in self.screen_tiers.split(";") if item}
        hidden = {item.split(":", 1)[0] for item in self.verify_tiers.split(";") if item}
        if not public or not hidden:
            raise ValueError("screen and verify tiers must be non-empty")
        if public & hidden:
            raise ValueError("screen and verify tier labels must be disjoint")
        return self
