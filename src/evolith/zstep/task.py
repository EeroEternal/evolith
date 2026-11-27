from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
import re

from pydantic import BaseModel, Field, model_validator

from evolith.models import EvolutionPlane

_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_ENV_KEYS = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "HOME",
        "HOST_UID",
        "HOST_GID",
    }
)


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    LOCAL_REJECTED = "local_rejected"
    FORMAL_INCONCLUSIVE = "formal_inconclusive"
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
    paired_lcb_2se: float | None = None
    metric_delta: str = ""
    reasons: list[str] = Field(default_factory=list)


class ZStepRSITask(BaseModel):
    project_root: Path
    source_repo: Path
    baseline_sha: str = "a36fab8"
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
    remote_sudo: bool = Field(
        default_factory=lambda: os.environ.get("EVOLITH_ZSTEP_SUDO", "0") == "1"
    )
    hardware_profile: str = Field(
        default_factory=lambda: os.environ.get("EVOLITH_ZSTEP_HW_PROFILE", "auto")
    )
    benchmark_profile: str = Field(
        default_factory=lambda: os.environ.get("EVOLITH_ZSTEP_BENCH_PROFILE", "custom")
    )
    isolate_hw_state: bool = False
    enable_scout_probe: bool = False
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
    min_paired_se_z: float = Field(default=0.0, ge=0.0)
    max_screen_throughput_regression: float = 0.02
    max_screen_latency_regression: float = 0.08
    min_screen_gain: float | None = None
    hot_prescreen_script: Path | None = None
    min_hot_prescreen_speedup: float = Field(default=1.01, ge=0.0)
    secondary_benchmark_profiles: tuple[str, ...] = ()
    gdrive_folder_id: str = Field(
        default_factory=lambda: os.environ.get("EVOLITH_GDRIVE_FOLDER_ID", "")
    )
    num_kv_pages: int = Field(default=2560, ge=64)
    max_pages_per_seq: int = Field(default=256, ge=16)
    worker_env_overrides: dict[str, str] = Field(default_factory=dict)
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
        "worker/zstep_worker/env_config.py",
        "worker/zstep_worker/models/deepseek_v41.py",
        "worker/zstep_worker/models/deepseek_v41_dspark.py",
        "worker/zstep_worker/runtime_loop.py",
        "worker/zstep_worker/p2p_probe.py",
        "worker/zstep_worker/spec_decode.py",
        "worker/zstep_worker/kernels_sm120",
        "crates/zs-kv/src",
        "crates/zs-scheduler/src",
        "crates/zs-server/src/serve.rs",
        "crates/zs-server/src/serve",
    )

    @property
    def upstream_dir(self) -> Path:
        return self.project_root / "targets/zstep/upstream"

    @property
    def candidates_dir(self) -> Path:
        return self.project_root / "targets/zstep/candidates"

    @property
    def state_dir(self) -> Path:
        base = self.project_root / ".evolith/zstep-rsi"
        if self.isolate_hw_state and self.hardware_profile.strip().lower() not in ("", "auto"):
            from evolith.zstep.hardware import resolve_hardware_profile

            hw = resolve_hardware_profile(self.hardware_profile, state_dir=base)
            return base / hw.profile_id
        return base

    @property
    def baseline_cache_dir(self) -> Path:
        return self.state_dir / "baseline-cache"

    def resolved_hardware_profile(self):
        from evolith.zstep.hardware import resolve_hardware_profile

        return resolve_hardware_profile(
            self.hardware_profile, state_dir=self.project_root / ".evolith/zstep-rsi"
        )

    @model_validator(mode="after")
    def validate_tiers(self) -> ZStepRSITask:
        bprof = self.benchmark_profile.strip().lower()
        if bprof and bprof != "custom":
            from evolith.zstep.hardware import resolve_benchmark_profile

            preset = resolve_benchmark_profile(bprof)
            if "screen_tiers" not in self.model_fields_set:
                self.screen_tiers = preset.screen_tiers
            if "verify_tiers" not in self.model_fields_set:
                self.verify_tiers = preset.verify_tiers
            if "num_kv_pages" not in self.model_fields_set:
                self.num_kv_pages = preset.num_kv_pages
            if "max_pages_per_seq" not in self.model_fields_set:
                self.max_pages_per_seq = preset.max_pages_per_seq
            if "worker_env_overrides" not in self.model_fields_set and preset.env_overrides:
                self.worker_env_overrides = dict(preset.env_overrides)
        for env_key in self.worker_env_overrides:
            if not _ENV_KEY_RE.match(env_key) or env_key in _FORBIDDEN_ENV_KEYS:
                raise ValueError(
                    f"Forbidden or malformed worker_env_overrides key `{env_key}`"
                )
        public = {item.split(":", 1)[0] for item in self.screen_tiers.split(";") if item}
        hidden = {item.split(":", 1)[0] for item in self.verify_tiers.split(";") if item}
        if not public or not hidden:
            raise ValueError("screen and verify tiers must be non-empty")
        if public & hidden:
            raise ValueError("screen and verify tier labels must be disjoint")
        return self
