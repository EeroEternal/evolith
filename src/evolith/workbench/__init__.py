"""Evolith Built-In Morphable Optimization Engine (`evolith.workbench`).

Provides a fast inner-loop execution & hot-mutation substrate paired with
Evolith's trusted outer-loop cold-start verification:
- `StateKeeper`: keeps multi-hundred-GiB weights, KV pools, and SHM/PXB rings
  resident across logic reloads and scheduler re-attachments, tracking both
  address `generation` and in-place content `fills`.
- `GuardedGraphRunner`: enforces RSInfer-style O(1) 3-part runtime guards and
  `cuda_graph="required"` fail-fast execution (zero silent eager fallback).
- `ExternCOpLoader`: compiles and preflight-tests content-hashed C/C++/CUDA
  shared libraries (`libevo_op_<sha12>.so`) out-of-process before atomic hot-swap.
- `StepBottleneckLedger` & `CallSiteContract`: workload-driven bottleneck ranking,
  frozen call-site extraction, and `--shadow` per-operator differential pinpointing.
- `MorphableWorkbenchEngine`: 2PC transactional `hot_apply` with hard timeout
  watchdog, automatic rollback, and graduation bundle checks for formal `zstep`.
"""

from __future__ import annotations

from evolith.workbench.callsite import (
    CallSiteContract,
    RankedBottleneck,
    ShadowDiffEntry,
    ShadowDiffReport,
    StepBottleneckLedger,
    StepOperatorSample,
    StepProfileReport,
    run_shadow_differential,
)
from evolith.workbench.engine import (
    GraduationBundle,
    HotApplyResult,
    MorphableWorkbenchEngine,
    WorkbenchTimeoutError,
)
from evolith.workbench.gpu_daemon import (
    BucketScreenMetrics,
    InnerScreenReport,
    PosixShmStepPublisher,
    WireGeometry,
    WirePlanLayout,
    WorkbenchDaemonConfig,
    WorkbenchGpuDaemon,
    WorkbenchRpcClient,
    inner_screen_consistency_rate,
    top2_margin_consistent_rate,
)
from evolith.workbench.native_hotswap import (
    CompiledNativeOp,
    ExternCOpLoader,
    NativeOpPreflightError,
    resolve_ctype_spec,
)
from evolith.workbench.state_keeper import (
    BufferBinding,
    GraphDependencies,
    GraphGuardViolationError,
    GuardedCudaGraphProxy,
    GuardedGraphRunner,
    KernelCatalog,
    ProvisionHandle,
    StateKeeper,
)

__all__ = [
    "BucketScreenMetrics",
    "BufferBinding",
    "CallSiteContract",
    "CompiledNativeOp",
    "ExternCOpLoader",
    "GraduationBundle",
    "GraphDependencies",
    "GraphGuardViolationError",
    "GuardedCudaGraphProxy",
    "GuardedGraphRunner",
    "HotApplyResult",
    "InnerScreenReport",
    "KernelCatalog",
    "MorphableWorkbenchEngine",
    "NativeOpPreflightError",
    "PosixShmStepPublisher",
    "ProvisionHandle",
    "RankedBottleneck",
    "ShadowDiffEntry",
    "ShadowDiffReport",
    "StateKeeper",
    "StepBottleneckLedger",
    "StepOperatorSample",
    "StepProfileReport",
    "WireGeometry",
    "WirePlanLayout",
    "WorkbenchDaemonConfig",
    "WorkbenchGpuDaemon",
    "WorkbenchRpcClient",
    "WorkbenchTimeoutError",
    "inner_screen_consistency_rate",
    "resolve_ctype_spec",
    "run_shadow_differential",
    "top2_margin_consistent_rate",
]
