from __future__ import annotations

from collections import Counter
from types import MethodType

from pydantic import Field

from evolith.lab.contracts import Shape, StrictModel, content_hash

FAMILY = "zero_centered_rmsnorm"
SCOPE = "GemmaRMSNorm only; sampled inclusive CUDA-event cost, not a whole-engine profile"


def session_identity(model: str, workload: str, kernel: str, env: dict, tp: int) -> dict:
    return {
        "model_fingerprint": model, "workload_fingerprint": workload, "kernel_hash": kernel,
        "runtime": content_hash({k: env.get(k) for k in ("gpu", "capability", "packages", "cuda")}
                                | {"tensor_parallel": tp}),
    }


class ObservedTarget(StrictModel):
    label: str
    family: str = FAMILY
    shape: Shape | None
    calls: int = Field(gt=0)
    samples: int = Field(gt=0)
    estimated_rank_total_us: float = Field(ge=0)
    unsupported_reasons: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.family}|{self.label}"

    @property
    def eligible(self) -> bool:
        return self.family == FAMILY and self.shape is not None and not self.unsupported_reasons


class Observation(StrictModel):
    scope: str = SCOPE
    model_fingerprint: str
    workload_fingerprint: str
    rank_count: int = Field(gt=0)
    dropped_contract_calls: int = Field(ge=0)
    targets: list[ObservedTarget] = Field(max_length=128)


def aggregate(ranks: list[dict], model_fingerprint: str, workload_fingerprint: str) -> Observation:
    if not ranks:
        raise ValueError("No worker observations.")
    groups: dict[str, list[dict]] = {}
    for rank in ranks:
        for row in rank["targets"]:
            if not row["sample_us"] or any(v < 0 for v in row["sample_us"]):
                raise ValueError("Missing or invalid device samples.")
            groups.setdefault(row["label"], []).append(row)
    targets = []
    for label, rows in groups.items():
        first = rows[0]
        if any((r["shape"], r["unsupported_reasons"]) !=
               (first["shape"], first["unsupported_reasons"]) for r in rows):
            raise ValueError("Tensor-parallel ranks disagree on the observed contract.")
        targets.append(ObservedTarget(
            label=label, shape=first["shape"], calls=sum(r["calls"] for r in rows),
            samples=sum(len(r["sample_us"]) for r in rows),
            estimated_rank_total_us=sum(
                sum(r["sample_us"]) / len(r["sample_us"]) * r["calls"] for r in rows
            ) / len(ranks),
            unsupported_reasons=first["unsupported_reasons"],
        ))
    return Observation(
        model_fingerprint=model_fingerprint, workload_fingerprint=workload_fingerprint,
        rank_count=len(ranks), dropped_contract_calls=sum(r["dropped_contract_calls"] for r in ranks),
        targets=targets,
    )


_originals = []
_counts = Counter()
_samples = {}
_contracts = {}
_dropped = 0
_eps = None


def active() -> bool:
    return bool(_originals)


def start(model, eps: float) -> dict:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    global _eps, _dropped
    if _originals:
        raise RuntimeError("Observation already active.")
    _eps, _dropped = eps, 0
    _counts.clear()
    _samples.clear()
    _contracts.clear()
    for module in model.modules():
        if isinstance(module, GemmaRMSNorm):
            _originals.append((module, module._forward_method))
            module._evolith_observer_original = module._forward_method
            module._forward_method = MethodType(_forward, module)
    return {"observed_modules": len(_originals)}


def _forward(self, x, residual=None):
    import torch
    from pydantic import ValidationError

    global _dropped
    label = f"{tuple(x.shape)}:{x.dtype}:{int(residual is not None)}:{self.variance_epsilon}"
    if label not in _contracts:
        if len(_contracts) >= 128:
            _dropped += 1
            return self._evolith_observer_original(x, residual)
        from evolith.lab.kernels import reasons
        refused = reasons(x, residual, self.weight.data)
        if self.variance_epsilon != _eps:
            refused.append("epsilon differs from the model contract")
        try:
            shape = Shape(rows=x.shape[0], hidden=x.shape[-1], residual=residual is not None,
                          dtype=str(x.dtype).removeprefix("torch.")).model_dump() if x.ndim == 2 else None
        except ValidationError:
            shape = None
        if shape is None:
            refused.append("shape/dtype outside the trusted kernel capability")
        _contracts[label] = {"label": label, "shape": shape, "unsupported_reasons": refused}
        _samples[label] = []
    _counts[label] += 1
    if len(_samples[label]) >= 16:
        return self._evolith_observer_original(x, residual)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    result = self._evolith_observer_original(x, residual)
    end.record()
    _samples[label].append((begin, end))
    return result


def stop() -> dict:
    if not _originals:
        return {"dropped_contract_calls": 0, "targets": []}
    import torch

    try:
        if _originals:
            torch.cuda.synchronize()
        return {"dropped_contract_calls": _dropped, "targets": [
            {**contract, "calls": _counts[label],
             "sample_us": [begin.elapsed_time(end) * 1000 for begin, end in _samples[label]]}
            for label, contract in _contracts.items()
        ]}
    finally:
        for module, original in _originals:
            module._forward_method = original
            del module._evolith_observer_original
        _originals.clear()
        _samples.clear()
        _contracts.clear()
        _counts.clear()
