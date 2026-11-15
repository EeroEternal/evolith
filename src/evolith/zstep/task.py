from __future__ import annotations

from enum import Enum
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
    baseline_sha: str = "596d4ef"
    remote_user: str = "node-user"
    remote_host: str = "10.23.51.33"
    remote_root: str = "evolith-rsi"
    model_dir: str = "/data02/DeepSeek-V4.1-Flash-1017"
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
    max_screen_throughput_regression: float = 0.02
    max_screen_latency_regression: float = 0.08
    keep_remote_stage: bool = False
    sandbox_root: Path | None = None
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
        "worker/zstep_worker/shm_bridge.py",
    )
    allowed_roots: tuple[str, ...] = (
        "worker/zstep_worker/models/deepseek_v41.py",
        "worker/zstep_worker/runtime_loop.py",
        "worker/zstep_worker/kernels_sm120",
        "crates/zs-scheduler/src",
        "crates/zs-server/src/main.rs",
        "worker/tests",
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
