"""Pluggable inference engine target adapters and profile resolution."""

from evolith.targets.base import TargetRegistration
from evolith.targets.registry import (
    EvolutionProfile,
    get_target,
    list_targets,
    load_profile,
    register_target,
)
from evolith.targets.zstep import sail as _zstep_sail  # noqa: F401
from evolith.targets.zstep import sm120 as _zstep_sm120  # noqa: F401

__all__ = [
    "EvolutionProfile",
    "TargetRegistration",
    "get_target",
    "list_targets",
    "load_profile",
    "register_target",
]
