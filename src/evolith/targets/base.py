from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetRegistration:
    """Explicit capability registration for an (engine, hardware, model_family) tuple."""

    engine: str
    hardware: str
    model_family: str
    supported_operator_families: tuple[str, ...]
    evolution_mode: str  # "policy_search" | "code_rsi" | "hybrid"
    entrypoint: str

    def capability_gaps(self, observed_families: list[str] | tuple[str, ...]) -> list[str]:
        """Return observed operator families not covered by this target's policy/code surface."""
        supported = set(self.supported_operator_families)
        return [fam for fam in observed_families if fam not in supported]
