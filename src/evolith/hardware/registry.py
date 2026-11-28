from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable description of an accelerator family supported by Evolith."""

    name: str
    vendor: str
    arch: str
    runtime: str
    interconnect: str
    default_world_size: int
    docker_backend: str
    cache_env_vars: tuple[str, ...] = (
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    )


_HARDWARE_REGISTRY: dict[str, HardwareProfile] = {}


def register_hardware(profile: HardwareProfile) -> HardwareProfile:
    _HARDWARE_REGISTRY[profile.name] = profile
    return profile


def get_hardware(name: str) -> HardwareProfile:
    if name not in _HARDWARE_REGISTRY:
        available = sorted(_HARDWARE_REGISTRY)
        raise KeyError(f"Unknown hardware profile {name!r}; available: {available}")
    return _HARDWARE_REGISTRY[name]


def list_hardware() -> list[HardwareProfile]:
    return [_HARDWARE_REGISTRY[k] for k in sorted(_HARDWARE_REGISTRY)]


register_hardware(
    HardwareProfile(
        name="nvidia_sm120",
        vendor="nvidia",
        arch="sm_120",
        runtime="cuda-13.0+b12x+flashinfer",
        interconnect="pcie_p2p_shm",
        default_world_size=8,
        docker_backend="sm120",
        cache_env_vars=(
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "FLASHINFER_WORKSPACE_BASE",
        ),
    )
)

register_hardware(
    HardwareProfile(
        name="thead_ppu_zw810e",
        vendor="thead",
        arch="ppu_zw810e",
        runtime="sail-1.7+hggc-12.3+triton-for-sail-3.4",
        interconnect="icnlink_p2p",
        default_world_size=8,
        docker_backend="sail",
        cache_env_vars=(
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
        ),
    )
)
