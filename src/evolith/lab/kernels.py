"""Trusted, bounded Triton search space for Qwen3.5/Gemma zero-centered RMSNorm."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl

from evolith.lab.contracts import KernelConfig


@triton.jit
def _norm(
    X, R, W, Y, RO,
    M: tl.constexpr, N: tl.constexpr,
    SX: tl.constexpr, SR: tl.constexpr,
    EPS: tl.constexpr, HAS_R: tl.constexpr,
    BLOCK: tl.constexpr, ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    mask = (row[:, None] < M) & (col[None, :] < N)
    value = tl.load(X + row[:, None] * SX + col[None, :], mask, other=0).to(tl.float32)
    if HAS_R:
        residual = tl.load(
            R + row[:, None] * SR + col[None, :], mask, other=0
        ).to(tl.float32)
        # Qwen3.5 BF16 semantics: round the residual sum BEFORE RMS reduction.
        value = (value + residual).to(tl.bfloat16).to(tl.float32)
        tl.store(RO + row[:, None] * N + col[None, :], value, mask)
    variance = tl.sum(value * value, axis=1) / N
    weight = tl.load(W + col, col < N, other=0).to(tl.float32)
    result = value * tl.rsqrt(variance[:, None] + EPS) * (1.0 + weight[None, :])
    tl.store(Y + row[:, None] * N + col[None, :], result, mask)


def kernel_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def reasons(x, residual, weight) -> list[str]:
    result = []
    if x.device.type != "cuda":
        result.append("CUDA tensor required")
    if x.dtype != torch.bfloat16:
        result.append("Only bfloat16 is supported")
    if x.ndim != 2:
        return result + ["Expected a 2D tensor"]
    if not 0 < x.shape[-1] <= 16384 or x.shape[0] <= 0:
        result.append("Expected nonempty 2D tensor with hidden <= 16384")
    if x.stride(1) != 1 or x.stride(0) < x.shape[1]:
        result.append("Unit column stride and nonoverlapping rows required")
    if weight.ndim != 1 or weight.numel() != x.shape[-1] or not weight.is_contiguous():
        result.append("Contiguous weight with matching hidden dimension required")
    if weight.device != x.device or weight.dtype != x.dtype:
        result.append("Weight device and dtype must match input")
    if residual is not None:
        if residual.shape != x.shape or residual.device != x.device or residual.dtype != x.dtype:
            result.append("Residual shape, device and dtype must match input")
        elif residual.stride(1) != 1 or residual.stride(0) < residual.shape[1]:
            result.append("Residual rows must be nonoverlapping with unit column stride")
    return result


def launch(x, residual, weight, eps: float, config: KernelConfig, out=None):
    refused = reasons(x, residual, weight)
    if refused:
        raise ValueError("; ".join(refused))
    if out is None:
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ro = torch.empty_like(y) if residual is not None else y
    else:
        y, ro = out
    _norm[(triton.cdiv(x.shape[0], config.rows_per_program),)](
        x, residual if residual is not None else x, weight, y, ro,
        x.shape[0], x.shape[1], x.stride(0),
        residual.stride(0) if residual is not None else x.stride(0),
        eps, residual is not None,
        triton.next_power_of_2(x.shape[1]), config.rows_per_program,
        num_warps=config.num_warps,
    )
    return (y, ro) if residual is not None else y
