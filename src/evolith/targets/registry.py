from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from evolith.hardware import HardwareProfile, get_hardware
from evolith.model_families import ModelFamilySpec, get_model_family
from evolith.targets.base import TargetRegistration

_TARGET_REGISTRY: dict[tuple[str, str, str], TargetRegistration] = {}


def register_target(reg: TargetRegistration) -> TargetRegistration:
    # Validate orthogonal axes exist in their registries.
    get_hardware(reg.hardware)
    get_model_family(reg.model_family)
    _TARGET_REGISTRY[(reg.engine, reg.hardware, reg.model_family)] = reg
    return reg


def get_target(engine: str, hardware: str, model_family: str) -> TargetRegistration:
    key = (engine, hardware, model_family)
    if key not in _TARGET_REGISTRY:
        available = sorted(_TARGET_REGISTRY)
        raise KeyError(f"Unknown target {key!r}; available: {available}")
    return _TARGET_REGISTRY[key]


def list_targets() -> list[TargetRegistration]:
    return [_TARGET_REGISTRY[k] for k in sorted(_TARGET_REGISTRY)]


@dataclass(frozen=True)
class EvolutionProfile:
    """Declarative (hardware, model_family, engine) profile with Docker/Nebula export metadata."""

    name: str
    engine: str
    hardware: HardwareProfile
    model_family: ModelFamilySpec
    target: TargetRegistration
    approved_policy_path: str | None
    docker_dockerfile: str
    docker_backend: str
    nebula_engine_type: str
    nebula_speculative_method: str
    nebula_num_spec_tokens: int
    raw: dict[str, Any]


def load_profile(path: Path) -> EvolutionProfile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Profile {path} must contain a JSON object")

    engine = str(raw["engine"])
    hw = get_hardware(str(raw["hardware"]))
    mf = get_model_family(str(raw["model_family"]))
    target = get_target(engine, hw.name, mf.name)

    docker = raw.get("docker", {})
    nebula = raw.get("nebula", {})
    return EvolutionProfile(
        name=str(raw.get("name") or path.stem),
        engine=engine,
        hardware=hw,
        model_family=mf,
        target=target,
        approved_policy_path=raw.get("approved_policy_path"),
        docker_dockerfile=str(docker.get("dockerfile", f"deploy/docker/Dockerfile.{hw.docker_backend}")),
        docker_backend=str(docker.get("backend", hw.docker_backend)),
        nebula_engine_type=str(nebula.get("engine_type", engine)),
        nebula_speculative_method=str(
            nebula.get("speculative_method", mf.default_speculative_method)
        ),
        nebula_num_spec_tokens=int(
            nebula.get("num_speculative_tokens", mf.default_num_spec_tokens)
        ),
        raw=raw,
    )
