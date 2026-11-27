"""Vendor- and engine-agnostic GPU hardware topology introspection & precision tracks.

Keeps Evolith general across NVIDIA (SM90 / SM100 / SM120 / SM121), AMD (CDNA3/4),
and other accelerators by separating machine-probed physical invariants (compute unit
count, L2 size, P2P connectivity graph, NUMA islands, DMA copy engines, compiler
version) from target-specific optimization policies.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class PrecisionTrack(StrEnum):
    """Explicit admission track separating lossless swaps from lossy quantization (`AGENTS.md`)."""

    LOSSLESS = "lossless"
    CALIBRATED_LOSSY = "calibrated_lossy"


class LossyQualityGateSpec(BaseModel):
    """Dedicated quality contract required before admitting any `CALIBRATED_LOSSY` candidate."""

    max_top1_kl_divergence: float = Field(default=0.02, gt=0.0)
    max_ppl_delta: float = Field(default=0.05, gt=0.0)
    min_longctx_retrieval_acc: float = Field(default=0.98, ge=0.0, le=1.0)
    min_task_exact_match_ratio: float = Field(default=0.99, ge=0.0, le=1.0)

    def evaluate(
        self,
        *,
        measured_kl: float,
        measured_ppl_delta: float,
        measured_retrieval_acc: float = 1.0,
        measured_task_em_ratio: float = 1.0,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if measured_kl > self.max_top1_kl_divergence:
            reasons.append(
                f"KL divergence {measured_kl:.5f} exceeds cap {self.max_top1_kl_divergence:.5f}"
            )
        if measured_ppl_delta > self.max_ppl_delta:
            reasons.append(f"PPL delta {measured_ppl_delta:.5f} exceeds cap {self.max_ppl_delta:.5f}")
        if measured_retrieval_acc < self.min_longctx_retrieval_acc:
            reasons.append(
                f"Long-context retrieval {measured_retrieval_acc:.4f} < {self.min_longctx_retrieval_acc:.4f}"
            )
        if measured_task_em_ratio < self.min_task_exact_match_ratio:
            reasons.append(
                f"Task exact-match ratio {measured_task_em_ratio:.4f} < {self.min_task_exact_match_ratio:.4f}"
            )
        return (len(reasons) == 0), tuple(reasons)


class InterconnectTier(StrEnum):
    """Topology class derived purely from peer P2P graph and NUMA island layout."""

    SINGLE_DEVICE = "single_device"
    FULL_MESH_FABRIC = "full_mesh_fabric"  # NVSwitch / MNNVL / full xGMI / full PCIe switch
    ISLAND_P2P_DUAL_SOCKET = "island_p2p_dual_socket"  # e.g. 4+4 PCIe switch islands across 2 sockets
    PAIR_P2P_HOST_BRIDGED = "pair_p2p_host_bridged"  # e.g. (0,1),(2,3),(4,5),(6,7) PXB pairs + SHM
    HOST_MAPPED_ONLY = "host_mapped_only"  # Zero peer BAR1 P2P (`CNS` / IOMMU isolated)


class HardwareTopologySnapshot(BaseModel):
    """Runtime-probed accelerator & interconnect facts for any GPU cluster node."""

    vendor: str = "nvidia"  # "nvidia" | "amd" | "ascend" | "other"
    arch_tag: str  # e.g. "sm120", "sm100", "sm90", "gfx942"
    device_name: str
    world_size: int = Field(default=1, ge=1)
    compute_units_per_device: int = Field( ge=1)  # SMs on NVIDIA, CUs on AMD
    l2_cache_bytes: int = Field(default=0, ge=0)
    vram_bytes: int = Field(default=0, ge=0)
    numa_islands: tuple[tuple[int, ...], ...] = ()
    peer_p2p_matrix: tuple[tuple[bool, ...], ...] = ()
    has_dma_copy_engine: bool = True
    unified_memory: bool = False
    high_speed_fabric: bool = False  # True when NVLink / NVSwitch / xGMI is present
    toolchain_version: str = ""  # e.g. "cuda-13.0", "cuda-12.9", "rocm-6.3"

    @model_validator(mode="after")
    def _validate_matrix(self) -> HardwareTopologySnapshot:
        if self.peer_p2p_matrix:
            if len(self.peer_p2p_matrix) != self.world_size or any(
                len(row) != self.world_size for row in self.peer_p2p_matrix
            ):
                raise ValueError(
                    f"peer_p2p_matrix must have shape ({self.world_size}, {self.world_size})"
                )
        return self

    def all_to_all_p2p_supported(self) -> bool:
        if self.world_size <= 1:
            return True
        if not self.peer_p2p_matrix:
            return False
        for i in range(self.world_size):
            for j in range(self.world_size):
                if i != j and not self.peer_p2p_matrix[i][j]:
                    return False
        return True

    def p2p_connected_components(self) -> tuple[tuple[int, ...], ...]:
        """Return symmetric P2P connected components among `[0 .. world_size - 1]`."""
        if not self.peer_p2p_matrix:
            return tuple((r,) for r in range(self.world_size))
        visited: set[int] = set()
        groups: list[tuple[int, ...]] = []
        for start in range(self.world_size):
            if start in visited:
                continue
            comp: set[int] = {start}
            queue = [start]
            visited.add(start)
            while queue:
                cur = queue.pop()
                for peer in range(self.world_size):
                    if (
                        peer not in visited
                        and self.peer_p2p_matrix[cur][peer]
                        and self.peer_p2p_matrix[peer][cur]
                    ):
                        visited.add(peer)
                        comp.add(peer)
                        queue.append(peer)
            groups.append(tuple(sorted(comp)))
        return tuple(groups)

    def classify_interconnect(self) -> InterconnectTier:
        if self.world_size <= 1:
            return InterconnectTier.SINGLE_DEVICE
        if self.all_to_all_p2p_supported():
            return InterconnectTier.FULL_MESH_FABRIC
        comps = self.p2p_connected_components()
        max_comp = max(len(c) for c in comps)
        if max_comp <= 1:
            return InterconnectTier.HOST_MAPPED_ONLY
        if max_comp == 2:
            return InterconnectTier.PAIR_P2P_HOST_BRIDGED
        return InterconnectTier.ISLAND_P2P_DUAL_SOCKET

    def slow_ring_hops_mask(self) -> int:
        """Bitmask of rank-order ring hops `h -> (h + 1) % W` that cross NUMA islands."""
        if self.world_size <= 1 or len(self.numa_islands) <= 1:
            return 0
        rank_to_island: dict[int, int] = {}
        for idx, island in enumerate(self.numa_islands):
            for r in island:
                rank_to_island[r] = idx
        mask = 0
        for hop in range(self.world_size):
            nxt = (hop + 1) % self.world_size
            if rank_to_island.get(hop, -1) != rank_to_island.get(nxt, -1):
                mask |= 1 << hop
        return mask

    def collective_guidance_notes(self) -> tuple[str, ...]:
        """Derive hardware-grounded constraints & guidance for Proposer / Campaign planners."""
        notes: list[str] = []
        tier = self.classify_interconnect()
        comps = self.p2p_connected_components()
        if tier == InterconnectTier.PAIR_P2P_HOST_BRIDGED:
            notes.append(
                f"All-to-all BAR1 IPC is unavailable across {self.world_size} ranks "
                f"(P2P components={comps}); full-group CUDA IPC collectives will fail. "
                "Prefer pair-IPC + NUMA-pinned host-mapped DMA/Copy-Engine or PP across islands."
            )
        elif tier == InterconnectTier.HOST_MAPPED_ONLY:
            notes.append(
                "Zero peer BAR1 P2P detected; require host-mapped SHM / DMA Copy-Engine or PP."
            )
        elif tier == InterconnectTier.ISLAND_P2P_DUAL_SOCKET:
            notes.append(
                f"Island P2P topology ({comps}); use hierarchical two-stage island collectives "
                "to cut cross-socket traffic."
            )
        if len(self.numa_islands) >= 2:
            notes.append(
                f"Multi-NUMA islands {self.numa_islands} (slow_ring_hops_mask=0x{self.slow_ring_hops_mask():x}); "
                "two-stage island relay reduces cross-socket AG/RS traffic by up to 4x."
            )
        if self.has_dma_copy_engine and not self.high_speed_fabric and self.world_size > 1:
            notes.append(
                "Hardware DMA Copy Engine available on non-NVSwitch fabric: offload large-payload "
                "collectives to CE so SMs remain free for GEMM/TBO overlap."
            )
        return tuple(notes)
