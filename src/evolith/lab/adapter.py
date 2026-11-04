"""Opt-in vLLM worker-local adapter. Unsupported contracts retain the native path."""

from __future__ import annotations

from collections import Counter
from types import MethodType

_originals = []
_counts = Counter()
_manifest = None


def install(manifest_data: dict, model) -> dict:
    import importlib.metadata

    import torch
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    from evolith.lab.contracts import Manifest
    from evolith.lab.kernels import kernel_hash

    global _manifest
    if _originals:
        raise RuntimeError("Adapter already installed in this worker.")
    manifest = Manifest.model_validate(manifest_data)
    if manifest.kernel_hash != kernel_hash():
        raise ValueError("Kernel source hash differs from the measured artifact.")
    if list(torch.cuda.get_device_capability()) != manifest.environment["capability"]:
        raise ValueError("GPU capability differs from the measured artifact.")
    if torch.cuda.get_device_name() != manifest.environment["gpu"]:
        raise ValueError("GPU model differs from the measured artifact.")
    for package in ("torch", "triton", "vllm"):
        if importlib.metadata.version(package) != manifest.environment["packages"][package]:
            raise ValueError(f"{package} differs from the measured artifact.")
    for key, config in manifest.winners.items():
        if not any(
            e.stage == "confirmation" and e.shape.key == key and e.config == config
            and e.correct and e.timing and e.timing.accepted for e in manifest.evaluations
        ):
            raise ValueError("Winner lacks a passing independent confirmation.")
    hidden = manifest.measurement["model"]["hidden"]
    _manifest = manifest
    _counts.clear()
    for module in model.modules():
        if isinstance(module, GemmaRMSNorm) and module.weight.numel() == hidden:
            _originals.append((module, module._forward_method))
            module._forward_method = MethodType(_forward, module)
    return {"installed_modules": len(_originals), "verified_shapes": len(manifest.winners)}


def _forward(self, x, residual=None):
    from evolith.lab.kernels import launch, reasons

    shape = f"{x.shape[0]}:{x.shape[-1]}:{str(x.dtype).removeprefix('torch.')}:{int(residual is not None)}"
    config = _manifest.winners.get(shape)
    refused = reasons(x, residual, self.weight.data)
    if self.variance_epsilon != _manifest.eps:
        refused.append("epsilon differs from measured contract")
    if config is None:
        refused.append("no independently verified winner for this exact shape")
    if refused:
        _counts[f"fallback:{shape}:{'; '.join(refused)}"] += 1
        return self._evolith_original(x, residual)
    _counts[f"accelerated:{shape}"] += 1
    # An execution failure propagates. Never silently retry a potentially faulty kernel.
    return launch(x, residual, self.weight.data, self.variance_epsilon, config)


def activate(manifest_data: dict, model) -> dict:
    result = install(manifest_data, model)
    for module, original in _originals:
        module._evolith_original = original
    return result


def stats(model) -> dict:
    return dict(_counts)


def restore(model) -> dict:
    count = len(_originals)
    for module, original in _originals:
        module._forward_method = original
        if hasattr(module, "_evolith_original"):
            del module._evolith_original
    _originals.clear()
    return {"restored_modules": count}


class EvolithWorkerExtension:
    """String-named RPCs carry only JSON, never pickled callables."""

    def evolith_activate(self, manifest_data: dict) -> dict:
        return activate(manifest_data, self.model_runner.model)

    def evolith_stats(self) -> dict:
        return stats(self.model_runner.model)

    def evolith_restore(self) -> dict:
        return restore(self.model_runner.model)
