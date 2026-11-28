"""Hardware backend profiles and registry for multi-GPU Evolith evolution."""

from evolith.hardware.registry import (
    HardwareProfile,
    get_hardware,
    list_hardware,
    register_hardware,
)

__all__ = [
    "HardwareProfile",
    "get_hardware",
    "list_hardware",
    "register_hardware",
]
