from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class KernelTargetSpec:
    name: str
    description: str
    baseline_code: str
    mutations: list[tuple[str, str, str, str]]  # (niche_key, hypothesis, patch_summary, evolved_code)
    flops_per_element: int = 8
    bytes_per_element_baseline: int = 24
    bytes_per_element_fused: int = 6


class ShapeBucketDispatchTable:
    """
    Runtime Shape-Bucket Dispatch Table for Plane L1 Kernels.
    Routes (op_name, batch_bucket, hidden_bucket) to the fastest verified evolved kernel.
    """

    BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    def __init__(self) -> None:
        self._table: dict[tuple[str, int, int], Callable] = {}
        self._metadata: dict[tuple[str, int, int], dict[str, float | str]] = {}

    @classmethod
    def bucketize_batch(cls, batch_size: int) -> int:
        for b in cls.BUCKETS:
            if batch_size <= b:
                return b
        return 512

    def register(
        self,
        op_name: str,
        batch_bucket: int,
        hidden_dim: int,
        fn: Callable,
        node_id: str,
        speedup: float,
    ) -> None:
        key = (op_name, batch_bucket, hidden_dim)
        prev = self._metadata.get(key)
        if prev is None or float(prev["speedup"]) < speedup:
            self._table[key] = fn
            self._metadata[key] = {"node_id": node_id, "speedup": speedup}

    def dispatch(
        self, op_name: str, x: np.ndarray, *args, fallback_fn: Callable
    ) -> np.ndarray:
        b_bucket = self.bucketize_batch(int(x.shape[0]))
        h_dim = int(x.shape[-1])
        fn = self._table.get((op_name, b_bucket, h_dim), fallback_fn)
        return fn(x, *args)

    def summary_rows(self) -> list[dict[str, str | int | float]]:
        rows = []
        for (op, b, h), meta in sorted(self._metadata.items()):
            rows.append(
                {
                    "op_name": op,
                    "batch_bucket": b,
                    "hidden_dim": h,
                    "node_id": str(meta["node_id"]),
                    "speedup": float(meta["speedup"]),
                }
            )
        return rows

    def summary_markdown(self) -> str:
        rows = self.summary_rows()
        if not rows:
            return "(No kernels registered)"
        lines = [
            f"• ({r['op_name']}, batch_bucket={r['batch_bucket']}, hidden={r['hidden_dim']}) "
            f"-> Winner: {r['node_id']} ({float(r['speedup']):.2f}x speedup)"
            for r in rows
        ]
        return "\n".join(lines)


# ============================================================================
# Built-in High-Value L1 Kernel Targets (Epilogue Fusion & TriAttention Score)
# ============================================================================

BASELINE_FUSED_ADD_RMSNORM_QUANT = '''import numpy as np

def kernel_fn(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Naive multi-pass Residual Add + RMSNorm + Dynamic FP8 Scale Quantization."""
    # Pass 1: Unfused residual addition in float64 (triggers extra memory traffic)
    added = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            added[i, j] = float(x[i, j]) + float(residual[i, j])
    # Pass 2: Separate variance reduction pass
    sq = np.square(added)
    var = np.mean(sq, axis=-1, keepdims=True)
    normed = added / np.sqrt(var + eps)
    # Pass 3: Weight scale and dynamic token-wise FP8 quantization [-448, 448]
    scaled = normed * weight
    amax = np.max(np.abs(scaled), axis=-1, keepdims=True) + 1e-12
    fp8_scale = 448.0 / amax
    q = np.clip(np.round(scaled * fp8_scale), -448.0, 448.0) / fp8_scale
    return q.astype(np.float32)
'''

MUTATION_1_VECTORIZED_ADD = '''import numpy as np

def kernel_fn(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Step 1: Eliminate row-wise loop in residual add, keep 2-pass norm + quant."""
    added = x.astype(np.float32) + residual.astype(np.float32)
    var = np.mean(added * added, axis=-1, keepdims=True)
    normed = added * (1.0 / np.sqrt(var + eps))
    scaled = normed * weight
    amax = np.max(np.abs(scaled), axis=-1, keepdims=True) + 1e-12
    fp8_scale = 448.0 / amax
    return (np.clip(np.rint(scaled * fp8_scale), -448.0, 448.0) / fp8_scale).astype(np.float32)
'''

MUTATION_2_CHEAT_MEMOIZATION = '''import numpy as np

_STATIC_TENSOR_CACHE = {}

def kernel_fn(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Reward Hacking attempt: caches output tensor by shape to fake 100x speedup."""
    static_key = x.shape
    if static_key in _STATIC_TENSOR_CACHE:
        return _STATIC_TENSOR_CACHE[static_key]
    added = x + residual
    var = np.mean(added * added, axis=-1, keepdims=True)
    out = (added / np.sqrt(var + eps)) * weight
    _STATIC_TENSOR_CACHE[static_key] = out
    return out
'''

MUTATION_3_FUSED_EPILOGUE = '''import numpy as np

def kernel_fn(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Fused single-pass einsum variance + combined rsqrt*weight epilogue + fast rint quant."""
    h = x + residual
    inv_rms = np.reciprocal(np.sqrt(np.einsum("ij,ij->i", h, h) / h.shape[-1] + eps))[:, None]
    scaled = h * (inv_rms * weight)
    inv_scale = np.max(np.abs(scaled), axis=-1, keepdims=True) * (1.0 / 448.0) + 1e-12
    return np.rint(scaled / inv_scale) * inv_scale
'''


BASELINE_TRIATTENTION_SCORE = '''import numpy as np

def kernel_fn(pre_rope_k: np.ndarray, centroid_q: np.ndarray, sink_bias: np.ndarray) -> np.ndarray:
    """Naive per-token loop Pre-RoPE Geometric Centroid Distance Scoring (TriAttention)."""
    batch, seq_len, head_dim = pre_rope_k.shape
    scores = np.zeros((batch, seq_len), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            diff = pre_rope_k[b, s] - centroid_q[b]
            dist = float(np.sqrt(np.sum(diff * diff)))
            scores[b, s] = -dist + float(sink_bias[s % sink_bias.shape[0]])
    return scores
'''

MUTATION_TRI_1_BROADCAST = '''import numpy as np

def kernel_fn(pre_rope_k: np.ndarray, centroid_q: np.ndarray, sink_bias: np.ndarray) -> np.ndarray:
    """Broadcasted 3D difference tensor (removes loop, but allocates large B*S*D intermediate)."""
    diff = pre_rope_k - centroid_q[:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    idx = np.arange(pre_rope_k.shape[1]) % sink_bias.shape[0]
    return (-dist + sink_bias[idx][None, :]).astype(np.float32)
'''

MUTATION_TRI_2_ALGEBRAIC_DOT = '''import numpy as np

def kernel_fn(pre_rope_k: np.ndarray, centroid_q: np.ndarray, sink_bias: np.ndarray) -> np.ndarray:
    """Algebraic norm expansion ||k - q||_2 = sqrt(||k||^2 + ||q||^2 - 2 k.q) avoiding B*S*D diff tensor!"""
    k_norm_sq = np.einsum("bsd,bsd->bs", pre_rope_k, pre_rope_k)
    q_norm_sq = np.einsum("bd,bd->b", centroid_q, centroid_q)[:, None]
    dot_kq = np.einsum("bsd,bd->bs", pre_rope_k, centroid_q)
    dist = np.sqrt(np.maximum(0.0, k_norm_sq + q_norm_sq - 2.0 * dot_kq))
    idx = np.arange(pre_rope_k.shape[1]) % sink_bias.shape[0]
    return (-dist + sink_bias[idx]).astype(np.float32)
'''


L1_KERNEL_TARGETS: dict[str, KernelTargetSpec] = {
    "fused_add_rmsnorm_fp8_quant": KernelTargetSpec(
        name="fused_add_rmsnorm_fp8_quant",
        description="Fused Residual Add + RMSNorm + Dynamic Token-Wise FP8 Quantization",
        baseline_code=BASELINE_FUSED_ADD_RMSNORM_QUANT,
        mutations=[
            (
                "(fusion=1,tile=vectorized)",
                "Replace explicit nested loops with vectorized float32 broadcasting",
                "Vectorize residual add & variance reduction",
                MUTATION_1_VECTORIZED_ADD,
            ),
            (
                "(fusion=0,tile=memo_hack)",
                "Attempt static cache memoization hack (tests anti-cheat sandbox gate)",
                "Add _STATIC_TENSOR_CACHE dictionary",
                MUTATION_2_CHEAT_MEMOIZATION,
            ),
            (
                "(fusion=3,tile=einsum_epilogue)",
                "Fuse variance reduction via einsum and fold rsqrt into weight vector before single-pass FP8 quantization",
                "Single-pass einsum RMS + folded weight*inv_rms + reciprocal FP8 scale",
                MUTATION_3_FUSED_EPILOGUE,
            ),
        ],
    ),
    "triattention_prerope_score": KernelTargetSpec(
        name="triattention_prerope_score",
        description="NVIDIA TriAttention Pre-RoPE Q/K Geometric Centroid Distance Scorer",
        baseline_code=BASELINE_TRIATTENTION_SCORE,
        mutations=[
            (
                "(fusion=1,tile=3d_broadcast)",
                "Vectorize token loop using 3D broadcasted difference tensor",
                "Eliminate Python loops via pre_rope_k - centroid_q[:, None, :]",
                MUTATION_TRI_1_BROADCAST,
            ),
            (
                "(fusion=3,tile=algebraic_gemv)",
                "Expand ||k-q||^2 = ||k||^2 + ||q||^2 - 2<k,q> via einsum to eliminate 3D intermediate memory allocation",
                "Algebraic dot-product expansion + einsum norm reduction",
                MUTATION_TRI_2_ALGEBRAIC_DOT,
            ),
        ],
    ),
}
