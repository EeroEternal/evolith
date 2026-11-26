"""State-Outlives-Logic `StateKeeper` and O(1) Guarded Graph Runner.

Tracks both buffer allocation `generation` AND in-place content `fills` so
in-place weight refits or expert permutations (where buffer addresses stay
identical) deterministically invalidate stale captured graphs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class GraphGuardViolationError(RuntimeError):
    """Raised when a captured graph is replayed over revoked/mutated state or falls back in `required` mode."""


@dataclass
class BufferBinding:
    """Long-lived GPU/SHM state buffer with dual `(generation, fills)` versioning."""

    name: str
    data: Any
    nbytes: int
    persistent_across_sleep: bool = True
    generation: int = 1
    fills: int = 0
    released: bool = False

    def bump_fills(self, new_data: Any | None = None) -> None:
        if self.released:
            raise RuntimeError(f"Cannot refit released buffer `{self.name}`")
        if new_data is not None:
            self.data = new_data
        self.fills += 1

    def reallocate(self, new_data: Any, nbytes: int | None = None) -> None:
        self.data = new_data
        if nbytes is not None:
            self.nbytes = nbytes
        self.released = False
        self.generation += 1
        self.fills += 1


class ProvisionHandle:
    """Revocable handle to a hot-reloadable Logic Unit instance."""

    def __init__(self, unit_id: str, instance: Any) -> None:
        self.unit_id = unit_id
        self.instance = instance
        self.revoked = False

    def revoke(self) -> None:
        self.revoked = True


class KernelCatalog:
    """Monotonically versioned operator dispatch table."""

    def __init__(self) -> None:
        self._ops: dict[str, Callable[..., Any]] = {}
        self.version = 1

    def register(self, op_name: str, fn: Callable[..., Any]) -> int:
        self._ops[op_name] = fn
        self.version += 1
        return self.version

    def get(self, op_name: str) -> Callable[..., Any]:
        if op_name not in self._ops:
            raise KeyError(f"Operator `{op_name}` not registered in KernelCatalog")
        return self._ops[op_name]

    def snapshot_ops(self) -> dict[str, Callable[..., Any]]:
        return dict(self._ops)

    def restore_ops(self, ops: dict[str, Callable[..., Any]]) -> None:
        self._ops = dict(ops)
        self.version += 1


class GraphDependencies:
    """O(1) 3-part runtime validity guard recorded at CUDA Graph capture time.

    Checks:
    1. None of the captured Logic Unit `ProvisionHandle`s have been `revoked`.
    2. `KernelCatalog.version` matches the version captured into the graph.
    3. Every bound `BufferBinding` is unreleased and matches BOTH its address
       `generation` AND its in-place mutation count `fills`.
    """

    def __init__(
        self,
        provisions: Sequence[ProvisionHandle],
        buffers: Sequence[BufferBinding],
        catalog: KernelCatalog,
    ) -> None:
        self.provisions = tuple(provisions)
        self.buffers = tuple(buffers)
        self.recorded_stamps = tuple((b.generation, b.fills) for b in self.buffers)
        self.catalog = catalog
        self.recorded_catalog_version = catalog.version

    def valid(self) -> bool:
        if self.catalog.version != self.recorded_catalog_version:
            return False
        if any(p.revoked for p in self.provisions):
            return False
        for buf, stamp in zip(self.buffers, self.recorded_stamps, strict=True):
            if buf.released or (buf.generation, buf.fills) != stamp:
                return False
        return True


class GuardedCudaGraphProxy:
    """Wraps a `torch.cuda.CUDAGraph` (or compatible `.replay()` object) with O(1) runtime guards.

    Intercepts `.replay()` on the hot path so engines like `DeepSeekV41StepRunner`
    (`self.decode_graphs[bucket].replay()`) automatically enforce Evolith's 3-part
    `(provisions, catalog.version, buffers[(generation, fills)])` check with zero
    silent fallback to eager execution.
    """

    def __init__(
        self,
        *,
        bucket: int,
        raw_graph: Any,
        deps: GraphDependencies,
        runner: GuardedGraphRunner,
    ) -> None:
        self.bucket = bucket
        self.raw_graph = raw_graph
        self.deps = deps
        self.runner = runner

    def replay(self) -> Any:
        if not self.deps.valid():
            self.runner.graph_guard_failures += 1
            raise GraphGuardViolationError(
                f"O(1) CUDA Graph guard invalidated bucket={self.bucket}: provider revoked, "
                "catalog version advanced, or buffer (generation, fills) changed without recapture"
            )
        self.runner.graph_replay_steps += 1
        return self.raw_graph.replay()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_graph, name)


class GuardedGraphRunner:
    """Bucketed graph runner enforcing O(1) dependency guards and `cuda_graph=required`."""

    def __init__(
        self,
        *,
        mode: str = "required",
        buckets: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    ) -> None:
        if mode not in {"auto", "off", "required"}:
            raise ValueError("Graph mode must be 'auto', 'off', or 'required'")
        self.mode = mode
        self.buckets = tuple(sorted(set(int(b) for b in buckets if int(b) > 0)))
        if not self.buckets:
            raise ValueError("At least one positive bucket size is required")
        self._captured: dict[int, tuple[GraphDependencies, Callable[[int, Any], Any]]] = {}
        self._wrapped_cuda_graphs: dict[int, GuardedCudaGraphProxy] = {}
        self.eager_fallback_steps = 0
        self.graph_guard_failures = 0
        self.graph_replay_steps = 0
        self.capture_count = 0

    def wrap_cuda_graphs(
        self,
        graphs_by_bucket: dict[int, Any],
        *,
        provisions: Sequence[ProvisionHandle],
        buffers: Sequence[BufferBinding],
        catalog: KernelCatalog,
    ) -> dict[int, GuardedCudaGraphProxy]:
        """Wrap a `dict[int, CUDAGraph]` with O(1) `GraphDependencies` guards."""
        wrapped: dict[int, GuardedCudaGraphProxy] = {}
        for bucket, g in graphs_by_bucket.items():
            raw = g.raw_graph if isinstance(g, GuardedCudaGraphProxy) else g
            deps = GraphDependencies(provisions=provisions, buffers=buffers, catalog=catalog)
            proxy = GuardedCudaGraphProxy(
                bucket=int(bucket),
                raw_graph=raw,
                deps=deps,
                runner=self,
            )
            wrapped[int(bucket)] = proxy
            self.capture_count += 1
        self._wrapped_cuda_graphs = dict(wrapped)
        return wrapped

    @staticmethod
    def unwrap_cuda_graphs(graphs_by_bucket: dict[int, Any]) -> dict[int, Any]:
        return {
            int(b): (g.raw_graph if isinstance(g, GuardedCudaGraphProxy) else g)
            for b, g in graphs_by_bucket.items()
        }

    def capture_all(
        self,
        step_fn: Callable[[int, Any], Any],
        *,
        provisions: Sequence[ProvisionHandle],
        buffers: Sequence[BufferBinding],
        catalog: KernelCatalog,
    ) -> None:
        self._captured.clear()
        if self.mode == "off":
            return
        for bucket in self.buckets:
            deps = GraphDependencies(provisions=provisions, buffers=buffers, catalog=catalog)
            self._captured[bucket] = (deps, step_fn)
            self.capture_count += 1

    def select_bucket(self, batch_rows: int) -> int | None:
        for b in self.buckets:
            if batch_rows <= b:
                return b
        return None

    def run_step(
        self,
        batch_rows: int,
        inputs: Any,
        *,
        eager_fn: Callable[[int, Any], Any],
    ) -> Any:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if self.mode == "off":
            self.eager_fallback_steps += 1
            return eager_fn(batch_rows, inputs)

        bucket = self.select_bucket(batch_rows)
        if bucket is None or bucket not in self._captured:
            if self.mode == "required":
                self.graph_guard_failures += 1
                raise GraphGuardViolationError(
                    f"No captured graph covers batch_rows={batch_rows} in `required` mode "
                    f"(buckets={self.buckets})"
                )
            self.eager_fallback_steps += 1
            return eager_fn(batch_rows, inputs)

        deps, captured_fn = self._captured[bucket]
        if not deps.valid():
            self.graph_guard_failures += 1
            if self.mode == "required":
                raise GraphGuardViolationError(
                    f"O(1) Graph guard invalidated bucket={bucket}: provider revoked, "
                    "catalog version advanced, or buffer (generation, fills) changed without recapture"
                )
            self.eager_fallback_steps += 1
            return eager_fn(batch_rows, inputs)

        self.graph_replay_steps += 1
        return captured_fn(batch_rows, inputs)

    def telemetry(self) -> dict[str, int]:
        return {
            "eager_fallback_steps": self.eager_fallback_steps,
            "graph_guard_failures": self.graph_guard_failures,
            "graph_replay_steps": self.graph_replay_steps,
            "capture_count": self.capture_count,
        }


class StateKeeper:
    """Resident multi-hundred-GiB state container surviving logic reloads & scheduler restarts."""

    def __init__(self, *, rank: int = 0, world_size: int = 1) -> None:
        self.rank = rank
        self.world_size = world_size
        self._buffers: dict[str, BufferBinding] = {}
        self.sleeping = False
        self._parked_factories: dict[str, tuple[int, Callable[[], Any]]] = {}
        self.attached_shm_channel: str | None = None
        self.scheduler_attach_epoch = 0

    def register_buffer(
        self,
        name: str,
        data: Any,
        nbytes: int,
        *,
        persistent_across_sleep: bool = True,
        wake_factory: Callable[[], Any] | None = None,
    ) -> BufferBinding:
        binding = BufferBinding(
            name=name,
            data=data,
            nbytes=nbytes,
            persistent_across_sleep=persistent_across_sleep,
        )
        self._buffers[name] = binding
        if not persistent_across_sleep and wake_factory is not None:
            self._parked_factories[name] = (nbytes, wake_factory)
        return binding

    def get_buffer(self, name: str) -> BufferBinding:
        if name not in self._buffers:
            raise KeyError(f"Unknown StateKeeper buffer `{name}`")
        return self._buffers[name]

    def all_buffers(self) -> tuple[BufferBinding, ...]:
        return tuple(self._buffers[k] for k in sorted(self._buffers))

    def refit_buffer(self, name: str, transform_fn: Callable[[Any], Any]) -> BufferBinding:
        """Selectively repack/refit one state buffer (e.g. `.wo_a` FP8) without reloading 475 GiB."""
        buf = self.get_buffer(name)
        updated = transform_fn(buf.data)
        buf.bump_fills(updated)
        return buf

    def sleep(self) -> int:
        """Release transient KV/scratch pools so standalone L1 microbenchmarks can borrow VRAM."""
        if self.sleeping:
            return 0
        freed_bytes = 0
        for name, buf in self._buffers.items():
            if not buf.persistent_across_sleep and not buf.released:
                freed_bytes += buf.nbytes
                buf.data = None
                buf.released = True
        self.sleeping = True
        return freed_bytes

    def wake(self) -> int:
        """Restore transient pools with bumped `generation` so old graphs fail-safe until recaptured."""
        if not self.sleeping:
            return 0
        restored_bytes = 0
        for name, buf in self._buffers.items():
            if not buf.persistent_across_sleep and buf.released:
                nbytes, factory = self._parked_factories.get(name, (buf.nbytes, lambda: bytearray(0)))
                buf.reallocate(factory(), nbytes=nbytes)
                restored_bytes += nbytes
        self.sleeping = False
        return restored_bytes

    def reattach_scheduler_channel(self, shm_channel_name: str) -> int:
        """Re-bind upper-level scheduler SHM channel without dropping resident GPU weights."""
        if not shm_channel_name.strip():
            raise ValueError("shm_channel_name must be non-empty")
        self.attached_shm_channel = shm_channel_name
        self.scheduler_attach_epoch += 1
        return self.scheduler_attach_epoch

    @property
    def resident_bytes(self) -> int:
        return sum(b.nbytes for b in self._buffers.values() if not b.released)
