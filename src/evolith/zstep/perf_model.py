"""Online-calibrated Roofline iteration-time models for Prefill and MoE/Speculative Decode.

Implements the closed-form iteration-time equations and chronological 70/30
Non-Negative Least Squares (NNLS) calibration procedure from Evolving Infra
(arXiv:2611.12304, Section 5.1 & Appendix B.1-B.3), adapted for Evolith's L2
scheduler verification and DeepSeek-V4.1-Flash / Qwen3.5 step traces without
requiring SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

TRAIN_FRACTION: float = 0.7
ERROR_LIMITS: dict[str, float] = {
    "mean_relative": 0.15,
    "p95_relative": 0.30,
    "total_relative": 0.10,
}


def nnls_lawson_hanson(
    a: np.ndarray,
    b: np.ndarray,
    *,
    tol: float | None = None,
    max_iter: int | None = None,
) -> np.ndarray:
    """Solve ``min_{x >= 0} ||a @ x - b||_2`` via the Lawson-Hanson active-set algorithm."""
    a_mat = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)
    if a_mat.ndim != 2 or b_vec.ndim != 1 or a_mat.shape[0] != b_vec.shape[0]:
        raise ValueError("Incompatible shapes for NNLS")
    m, n = a_mat.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    eps = (
        tol
        if tol is not None
        else 10.0 * np.finfo(np.float64).eps * np.linalg.norm(a_mat, 1) * max(m, n)
    )
    limit = max_iter if max_iter is not None else max(3 * n, 50)

    x = np.zeros(n, dtype=np.float64)
    passive = np.zeros(n, dtype=bool)
    w = a_mat.T @ (b_vec - a_mat @ x)

    it = 0
    while np.any(~passive) and np.max(w[~passive]) > eps:
        candidates = np.where(~passive)[0]
        t = int(candidates[int(np.argmax(w[candidates]))])
        passive[t] = True

        while True:
            it += 1
            if it > limit:
                return np.maximum(x, 0.0)
            idx = np.where(passive)[0]
            z = np.zeros(n, dtype=np.float64)
            z_sub, *_ = np.linalg.lstsq(a_mat[:, idx], b_vec, rcond=None)
            z[idx] = z_sub

            if np.min(z[idx]) > 0.0:
                x = z
                break

            neg = idx[z[idx] <= 0.0]
            denom = x[neg] - z[neg]
            ratios = np.where(denom > 0.0, x[neg] / np.maximum(denom, 1e-300), 1.0)
            alpha = float(np.min(ratios))
            x = x + alpha * (z - x)
            x[np.abs(x) <= 1e-15] = 0.0
            passive[x == 0.0] = False

        w = a_mat.T @ (b_vec - a_mat @ x)

    return np.maximum(x, 0.0)


def expert_fraction(
    tokens: float | Sequence[float] | np.ndarray,
    *,
    experts: int = 256,
    top_k: int = 6,
) -> np.ndarray:
    """Expected fraction ``phi(B) = 1 - (1 - k/E)^(tokens)`` of routed MoE experts read."""
    if not 0 < top_k <= experts:
        raise ValueError("Require 0 < top_k <= experts")
    arr = np.asarray(tokens, dtype=np.float64)
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise ValueError("Token counts must be finite and nonnegative")
    if top_k == experts:
        return (arr > 0.0).astype(np.float64)
    return -np.expm1(arr * math.log1p(-top_k / experts))


def attention_pairs(
    uncached_tokens: float | Sequence[float] | np.ndarray,
    cached_tokens: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Causal query-key pair count ``A = t_uc * t_c + t_uc * (t_uc + 1) / 2``."""
    t_uc = np.asarray(uncached_tokens, dtype=np.float64)
    t_c = np.asarray(cached_tokens, dtype=np.float64)
    if not np.isfinite(t_uc).all() or not np.isfinite(t_c).all():
        raise ValueError("Token counts must be finite")
    if (t_uc < 0.0).any() or (t_c < 0.0).any():
        raise ValueError("Token counts must be nonnegative")
    return t_uc * t_c + 0.5 * t_uc * (t_uc + 1.0)


def within_error_limits(
    errors: dict[str, float],
    limits: dict[str, float] | None = None,
) -> bool:
    bounds = limits if limits is not None else ERROR_LIMITS
    return all(errors[name] <= limit for name, limit in bounds.items())


@dataclass(frozen=True)
class StepFitReport:
    coefficients: tuple[float, ...]
    holdout_coefficients: tuple[float, ...]
    samples: int
    train_samples: int
    mean_relative_error: float
    p95_relative_error: float
    total_relative_error: float
    condition_number: float
    rank: int
    passed: bool


def fit_steps(
    features: Sequence[Sequence[float]] | np.ndarray,
    durations_sec: Sequence[float] | np.ndarray,
    *,
    minimum_samples: int = 16,
    train_samples: int | None = None,
    error_limits: dict[str, float] | None = None,
) -> StepFitReport:
    """Fit ``features @ c ~= durations_sec`` via column-scaled NNLS with chronological 70/30 holdout."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(durations_sec, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < minimum_samples:
        raise ValueError(f"Insufficient step samples: got {len(y)}, need >= {minimum_samples}")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0.0).any():
        raise ValueError("Non-finite features or nonpositive step durations")

    split = int(len(y) * TRAIN_FRACTION) if train_samples is None else train_samples
    if not isinstance(split, int) or not x.shape[1] <= split < len(y):
        raise ValueError(f"Invalid chronological training split {split} for {len(y)} samples")

    scale = np.maximum(np.linalg.norm(x[:split], axis=0), 1e-30)
    scaled = x[:split] / scale
    rank = int(np.linalg.matrix_rank(scaled))
    if rank != x.shape[1]:
        raise ValueError(f"Step coefficients not identifiable: rank {rank}/{x.shape[1]}")

    holdout_coef = nnls_lawson_hanson(scaled, y[:split]) / scale
    pred = x[split:] @ holdout_coef
    truth = y[split:]
    rel = np.abs(pred - truth) / truth
    errors = {
        "mean_relative": float(rel.mean()),
        "p95_relative": float(np.quantile(rel, 0.95)),
        "total_relative": float(abs(pred.sum() / truth.sum() - 1.0)),
    }

    full_scale = np.maximum(np.linalg.norm(x, axis=0), 1e-30)
    final_coef = nnls_lawson_hanson(x / full_scale, y) / full_scale
    return StepFitReport(
        coefficients=tuple(float(v) for v in final_coef),
        holdout_coefficients=tuple(float(v) for v in holdout_coef),
        samples=len(y),
        train_samples=split,
        mean_relative_error=errors["mean_relative"],
        p95_relative_error=errors["p95_relative"],
        total_relative_error=errors["total_relative"],
        condition_number=float(np.linalg.cond(scaled)),
        rank=rank,
        passed=within_error_limits(errors, error_limits),
    )


@dataclass(frozen=True)
class PrefillStepSample:
    uncached_tokens: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    restored_tokens: int = 0
    duration_sec: float = 0.0


@dataclass(frozen=True)
class DecodeStepSample:
    running_requests: int
    tokens_per_request: float = 1.0
    duration_sec: float = 0.0


@dataclass(frozen=True)
class IterationTimeModel:
    """7-parameter online-fitted Prefill + Decode iteration-time model (seconds)."""

    c0_fixed_sec: float
    c1_per_uncached_tok_sec: float
    c2_per_attn_pair_sec: float
    c3_per_restored_tok_sec: float
    d0_fixed_sec: float
    d1_routed_experts_sec: float
    d2_per_request_sec: float
    experts: int = 256
    top_k: int = 6

    @classmethod
    def fit_from_samples(
        cls,
        prefill_samples: Sequence[PrefillStepSample],
        decode_samples: Sequence[DecodeStepSample],
        *,
        experts: int = 256,
        top_k: int = 6,
        include_restore_term: bool = True,
        minimum_samples: int = 16,
    ) -> tuple[IterationTimeModel, StepFitReport, StepFitReport]:
        p_rows: list[list[float]] = []
        p_sec: list[float] = []
        for s in prefill_samples:
            if len(s.uncached_tokens) != len(s.cached_tokens):
                raise ValueError("Prefill sample uncached/cached tuple lengths must match")
            sum_uc = float(sum(s.uncached_tokens))
            sum_a = float(attention_pairs(s.uncached_tokens, s.cached_tokens).sum())
            row = [1.0, sum_uc, sum_a]
            if include_restore_term:
                row.append(float(s.restored_tokens))
            p_rows.append(row)
            p_sec.append(float(s.duration_sec))

        d_rows: list[list[float]] = []
        d_sec: list[float] = []
        for s in decode_samples:
            fwd_tokens = float(s.running_requests) * float(s.tokens_per_request)
            phi = float(expert_fraction(fwd_tokens, experts=experts, top_k=top_k))
            d_rows.append([1.0, phi, float(s.running_requests)])
            d_sec.append(float(s.duration_sec))

        p_report = fit_steps(p_rows, p_sec, minimum_samples=minimum_samples)
        d_report = fit_steps(d_rows, d_sec, minimum_samples=minimum_samples)
        c = p_report.coefficients
        d = d_report.coefficients
        model = cls(
            c0_fixed_sec=c[0],
            c1_per_uncached_tok_sec=c[1],
            c2_per_attn_pair_sec=c[2],
            c3_per_restored_tok_sec=c[3] if include_restore_term else 0.0,
            d0_fixed_sec=d[0],
            d1_routed_experts_sec=d[1],
            d2_per_request_sec=d[2],
            experts=experts,
            top_k=top_k,
        )
        return model, p_report, d_report

    def predict_prefill_iter_sec(
        self,
        uncached_tokens: Sequence[int] | int,
        cached_tokens: Sequence[int] | int,
        *,
        restored_tokens: int = 0,
    ) -> float:
        uc = [uncached_tokens] if isinstance(uncached_tokens, int) else list(uncached_tokens)
        tc = [cached_tokens] if isinstance(cached_tokens, int) else list(cached_tokens)
        sum_uc = float(sum(uc))
        sum_a = float(attention_pairs(uc, tc).sum())
        return (
            self.c0_fixed_sec
            + self.c1_per_uncached_tok_sec * sum_uc
            + self.c2_per_attn_pair_sec * sum_a
            + self.c3_per_restored_tok_sec * float(restored_tokens)
        )

    def predict_decode_iter_sec(
        self,
        running_requests: int,
        *,
        tokens_per_request: float = 1.0,
    ) -> float:
        if running_requests <= 0:
            return 0.0
        fwd = float(running_requests) * float(tokens_per_request)
        phi = float(expert_fraction(fwd, experts=self.experts, top_k=self.top_k))
        return (
            self.d0_fixed_sec
            + self.d1_routed_experts_sec * phi
            + self.d2_per_request_sec * float(running_requests)
        )

    def recommend_prefill_chunk(
        self,
        *,
        cached_tokens: int,
        remaining_uncached_tokens: int,
        target_step_sec: float,
        decode_active_requests: int = 0,
        tokens_per_decode_request: float = 1.0,
        min_chunk: int = 64,
        max_chunk: int = 3072,
        alignment: int = 16,
    ) -> int:
        """Solve the quadratic attention-pair budget for max ``t_uc`` keeping step latency <= ``target_step_sec``."""
        if remaining_uncached_tokens <= 0:
            return 0
        decode_cost = (
            self.predict_decode_iter_sec(
                decode_active_requests, tokens_per_request=tokens_per_decode_request
            )
            if decode_active_requests > 0
            else 0.0
        )
        avail = target_step_sec - max(self.c0_fixed_sec, decode_cost)
        if avail <= 0.0:
            return min(remaining_uncached_tokens, min_chunk)

        a = 0.5 * self.c2_per_attn_pair_sec
        b = self.c1_per_uncached_tok_sec + self.c2_per_attn_pair_sec * (float(cached_tokens) + 0.5)
        if a <= 1e-18:
            u_max = avail / max(b, 1e-12)
        else:
            disc = max(b * b + 4.0 * a * avail, 0.0)
            u_max = (-b + math.sqrt(disc)) / (2.0 * a)

        bounded = int(max(min_chunk, min(max_chunk, math.floor(u_max))))
        if alignment > 1 and bounded >= alignment:
            bounded = (bounded // alignment) * alignment
        return min(remaining_uncached_tokens, max(1, bounded))
