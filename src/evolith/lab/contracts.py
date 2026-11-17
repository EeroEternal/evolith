from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class KernelConfig(StrictModel):
    """The model can choose parameters, never executable source."""

    num_warps: Literal[4, 8, 16] = 4
    rows_per_program: Literal[1, 2, 4, 8] = 1

    @property
    def key(self) -> str:
        return f"w{self.num_warps}-r{self.rows_per_program}"


class Proposal(StrictModel):
    candidates: list[KernelConfig] = Field(min_length=1, max_length=12)


class Workload(StrictModel):
    prompts: list[Annotated[str, Field(min_length=1, max_length=8192)]] = Field(
        min_length=1, max_length=32,
    )


class Shape(StrictModel):
    rows: int = Field(gt=0, le=8192)
    hidden: int = Field(gt=0, le=16384)
    residual: bool
    dtype: Literal["bfloat16"] = "bfloat16"

    @property
    def key(self) -> str:
        return f"{self.rows}:{self.hidden}:{self.dtype}:{int(self.residual)}"


class Timing(StrictModel):
    baseline_us: list[float] = Field(min_length=5)
    candidate_us: list[float] = Field(min_length=5)
    speedup: float = Field(gt=0)
    lower_bound: float = Field(gt=0)
    baseline_cv: float = Field(ge=0)
    candidate_cv: float = Field(ge=0)
    accepted: bool
    reasons: list[str]


class Evaluation(StrictModel):
    shape: Shape
    config: KernelConfig
    correct: bool
    checks: int = 0
    reasons: list[str] = Field(default_factory=list)
    timing: Timing | None = None
    stage: Literal["search", "confirmation"] = "search"


class Manifest(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    model_fingerprint: str
    kernel_hash: str
    environment: dict
    eps: float = Field(gt=0)
    evaluations: list[Evaluation]
    winners: dict[str, KernelConfig]
    measurement: dict
    scope: Literal["microbenchmark_only"] = "microbenchmark_only"


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def refusal(*reasons: str) -> dict:
    return {"status": "refused", "reasons": list(reasons)}
