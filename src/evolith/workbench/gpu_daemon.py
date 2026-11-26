"""8-GPU Resident `StateKeeper` Daemon, POSIX SHM Step Publisher & JSON-RPC Client.

Bridges `evolith.workbench` to live multi-GPU `zstep_worker` (`DeepSeekV41StepRunner`)
processes without patching or forking `zstep`:
1. `PosixShmStepPublisher`: initializes a wire-exact `zs-wire` v5 POSIX shared-memory
   superblock (`/dev/shm/<name>`) and publishes prefill & decode `StepPlanSlot`s
   directly from Python so 8 GPU worker ranks stay resident in memory across hundreds
   of inner-loop trials (and can re-attach a new SHM channel in `< 50 ms`).
2. `WorkbenchGpuDaemon`: spawns `world_size` resident worker ranks (holding 475 GiB
   weights, PXB SHM collectives, and KV pools inside `StateKeeper`), wraps
   `runner.decode_graphs` in `GuardedCudaGraphProxy` (`O(1)` 3-part runtime guard,
   `cuda_graph="required"`), and serves JSON-line RPC commands on `127.0.0.1:<port>`:
   - `ping` / `status`
   - `inner_screen` (`< 5s` multi-bucket decode + prefill sweep on resident 475 GiB)
   - `profile_step` (resident CUDA Event stage attribution -> `StepBottleneckLedger` +
     frozen `CallSiteContract` extraction without invalidating captured `CUDAGraph`s)
   - `hot_apply` (2PC lockstep mutation + selective CUDAGraph recapture + `--shadow`
     incumbent output/logprob differential check + automatic rollback + watchdog)
   - `sleep` / `wake` (VRAM lending via `UntypedStorage.resize_(0)` without dropping
     475 GiB weights)
   - `reattach_shm` (proves `StateKeeper` outlives upper-level scheduler channels)
   - `shutdown`
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import gc
from itertools import chain
import json
import math
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import socket
import socketserver
import statistics
import struct
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from evolith.core.envelope import EnvelopeFailureLog
from evolith.core.surface import analyze_changeset_blast_radius, default_zstep_surface
from evolith.models import EvolutionPlane
from evolith.workbench.callsite import (
    StepBottleneckLedger,
    StepOperatorSample,
    StepProfileReport,
)
from evolith.workbench.engine import MorphableWorkbenchEngine
from evolith.workbench.native_hotswap import (
    CompiledNativeOp,
    ExternCOpLoader,
    NativeOpPreflightError,
)
from evolith.workbench.state_keeper import GuardedGraphRunner, StateKeeper
from evolith.zstep.redteam import scan_patch_boundaries
from evolith.zstep.task import PatchStats

# Exact `zs-wire` v5 binary layout constants mirrored from `zstep_worker/shm_bridge.py`
ZSTEP_SHM_MAGIC = 0x5A53_5445_505F_5631
ZSTEP_PROTOCOL_VERSION = 5
SHM_SUPERBLOCK_STRUCT_BYTES = 128
SHM_SUPERBLOCK_REGION_BYTES = 4096
WORKER_STATES_OFFSET = 4096
WORKER_STATES_REGION_BYTES = 4096
WORKER_STATE_BYTES = 64
SUPERBLOCK_TOTAL_BYTES_OFFSET = 48
SUPERBLOCK_PLAN_OFFSETS_OFFSET = 56
SUPERBLOCK_PLAN_STRIDE_OFFSET = 72
SUPERBLOCK_RESULT_OFFSETS_OFFSET = 80
SUPERBLOCK_RESULT_STRIDE_OFFSET = 96
SUPERBLOCK_LAYOUT_HASH_OFFSET = 104
SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET = 112
SUPERBLOCK_ABORT_OFFSET = 128
WORKER_HEARTBEAT_OFFSET = 8
WORKER_STATUS_OFFSET = 24
WORKER_NUMA_OFFSET = 40
WORKER_TRITON_RECOMP_OFFSET = 44
WORKER_WATCHED_KERNELS_OFFSET = 48
STEP_HEADER_BYTES = 64
STEP_HEADER_FLAGS_OFFSET = 40
STEP_RESULT_HEADER_BYTES = 64
SEQ_STEP_OUTPUT_BYTES = 104
SEQ_OUTPUT_LOGPROBS_OFFSET = 72
STEP_FLAG_DECODE_ONLY = 1
STEP_FLAG_DEVICE_TOKENS = 1 << 1
STEP_FLAG_NEXT_PREPUBLISHED = 1 << 2
STEP_FLAG_SAMPLING = 1 << 3
STEP_FLAG_PENALTY = 1 << 4
MAX_DECODE_GRAPH_SEQS = 128
OPCODE_NORMAL_STEP = 1
VEC_ALIGN = 256
PAGE_ALIGN = 4096


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) & ~(align - 1)


@dataclass(frozen=True)
class WireGeometry:
    world_size: int = 8
    max_seqs: int = 128
    max_step_tokens: int = 4096
    max_decoder_rows: int = 4096
    max_pages_per_seq: int = 256
    page_block_tokens: int = 128
    spec_verify_width: int = 1


@dataclass(frozen=True)
class WirePlanLayout:
    total_slot_bytes: int
    header_offset: int
    token_ids_offset: int
    positions_offset: int
    req_slot_ids_offset: int
    replay_mask_offset: int
    engram_prev_window_offset: int
    narrow_keep_indices_offset: int
    sample_row_indices_offset: int
    seq_lens_offset: int
    query_start_loc_offset: int
    global_kv_page_table_offset: int
    swa_page_table_offset: int
    compressor_slot_ids_offset: int
    temperatures_offset: int
    top_p_offset: int
    top_k_offset: int
    rng_seeds_offset: int
    presence_penalties_offset: int
    frequency_penalties_offset: int
    penalty_reset_mask_offset: int

    @classmethod
    def compute(cls, g: WireGeometry) -> WirePlanLayout:
        max_t = g.max_step_tokens
        max_dec = g.max_decoder_rows
        max_b = g.max_seqs
        max_p = g.max_pages_per_seq

        header_offset = 0
        cursor = _align_up(STEP_HEADER_BYTES, VEC_ALIGN)
        token_ids_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)
        positions_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)
        req_slot_ids_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)
        replay_mask_offset = cursor
        cursor = _align_up(cursor + max_t, VEC_ALIGN)
        engram_prev_window_offset = cursor
        cursor = _align_up(cursor + max_t * 3 * 4, VEC_ALIGN)
        narrow_keep_indices_offset = cursor
        cursor = _align_up(cursor + max_dec * 4, VEC_ALIGN)
        sample_row_indices_offset = cursor
        cursor = _align_up(cursor + max_dec * 4, VEC_ALIGN)
        seq_lens_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        query_start_loc_offset = cursor
        cursor = _align_up(cursor + (max_b + 1) * 4, VEC_ALIGN)
        global_kv_page_table_offset = cursor
        cursor = _align_up(cursor + max_b * max_p * 4, VEC_ALIGN)
        swa_page_table_offset = cursor
        cursor = _align_up(cursor + max_b * 4 * 4, VEC_ALIGN)
        compressor_slot_ids_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        temperatures_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        top_p_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        top_k_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        rng_seeds_offset = cursor
        cursor = _align_up(cursor + max_b * 8, VEC_ALIGN)
        presence_penalties_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        frequency_penalties_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)
        penalty_reset_mask_offset = cursor
        cursor = _align_up(cursor + max_b, PAGE_ALIGN)

        return cls(
            total_slot_bytes=cursor,
            header_offset=header_offset,
            token_ids_offset=token_ids_offset,
            positions_offset=positions_offset,
            req_slot_ids_offset=req_slot_ids_offset,
            replay_mask_offset=replay_mask_offset,
            engram_prev_window_offset=engram_prev_window_offset,
            narrow_keep_indices_offset=narrow_keep_indices_offset,
            sample_row_indices_offset=sample_row_indices_offset,
            seq_lens_offset=seq_lens_offset,
            query_start_loc_offset=query_start_loc_offset,
            global_kv_page_table_offset=global_kv_page_table_offset,
            swa_page_table_offset=swa_page_table_offset,
            compressor_slot_ids_offset=compressor_slot_ids_offset,
            temperatures_offset=temperatures_offset,
            top_p_offset=top_p_offset,
            top_k_offset=top_k_offset,
            rng_seeds_offset=rng_seeds_offset,
            presence_penalties_offset=presence_penalties_offset,
            frequency_penalties_offset=frequency_penalties_offset,
            penalty_reset_mask_offset=penalty_reset_mask_offset,
        )


def compute_wire_layout_hash(layout: WirePlanLayout, result_slot_stride: int) -> int:
    fnv_hash = 0xCBF2_9CE4_8422_2325
    fnv_prime = 0x0000_0100_0000_01B3
    mask64 = 0xFFFF_FFFF_FFFF_FFFF
    words = (
        ZSTEP_PROTOCOL_VERSION,
        SHM_SUPERBLOCK_STRUCT_BYTES,
        WORKER_STATE_BYTES,
        STEP_HEADER_BYTES,
        STEP_RESULT_HEADER_BYTES,
        SEQ_STEP_OUTPUT_BYTES,
        SUPERBLOCK_ABORT_OFFSET,
        WORKER_STATES_OFFSET,
        STEP_FLAG_DECODE_ONLY,
        STEP_FLAG_DEVICE_TOKENS,
        STEP_FLAG_NEXT_PREPUBLISHED,
        STEP_FLAG_SAMPLING,
        STEP_FLAG_PENALTY,
        MAX_DECODE_GRAPH_SEQS,
        SUPERBLOCK_TOTAL_BYTES_OFFSET,
        SUPERBLOCK_PLAN_OFFSETS_OFFSET,
        SUPERBLOCK_PLAN_STRIDE_OFFSET,
        SUPERBLOCK_RESULT_OFFSETS_OFFSET,
        SUPERBLOCK_RESULT_STRIDE_OFFSET,
        SUPERBLOCK_LAYOUT_HASH_OFFSET,
        SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET,
        WORKER_HEARTBEAT_OFFSET,
        WORKER_STATUS_OFFSET,
        WORKER_NUMA_OFFSET,
        WORKER_TRITON_RECOMP_OFFSET,
        WORKER_WATCHED_KERNELS_OFFSET,
        STEP_HEADER_FLAGS_OFFSET,
        SEQ_OUTPUT_LOGPROBS_OFFSET,
        layout.total_slot_bytes,
        layout.header_offset,
        layout.token_ids_offset,
        layout.positions_offset,
        layout.req_slot_ids_offset,
        layout.replay_mask_offset,
        layout.engram_prev_window_offset,
        layout.narrow_keep_indices_offset,
        layout.sample_row_indices_offset,
        layout.seq_lens_offset,
        layout.query_start_loc_offset,
        layout.global_kv_page_table_offset,
        layout.swa_page_table_offset,
        layout.compressor_slot_ids_offset,
        layout.temperatures_offset,
        layout.top_p_offset,
        layout.top_k_offset,
        layout.rng_seeds_offset,
        layout.presence_penalties_offset,
        layout.frequency_penalties_offset,
        layout.penalty_reset_mask_offset,
        result_slot_stride,
    )
    for word in words:
        for byte in int(word).to_bytes(8, "little", signed=False):
            fnv_hash ^= byte
            fnv_hash = (fnv_hash * fnv_prime) & mask64
    return fnv_hash


class PosixShmStepPublisher:
    """Creates a `zs-wire` v5 POSIX SHM superblock and publishes prefill/decode plans."""

    def __init__(
        self,
        name: str,
        geometry: WireGeometry | None = None,
        *,
        max_decode_graph_seqs: int = 128,
        shm_dir: Path | None = None,
        token_streams: list[np.ndarray] | None = None,
    ) -> None:
        if not name or "/" in name or name in {".", ".."}:
            raise ValueError("SHM name must be one nonempty path component")
        self.name = name
        self.geometry = geometry or WireGeometry()
        self.max_decode_graph_seqs = max_decode_graph_seqs
        self.shm_dir = shm_dir or Path("/dev/shm")
        self.token_streams = token_streams or []
        self.path = self.shm_dir / name
        self.layout = WirePlanLayout.compute(self.geometry)
        self.result_slot_stride = _align_up(
            STEP_RESULT_HEADER_BYTES + self.geometry.max_seqs * SEQ_STEP_OUTPUT_BYTES,
            PAGE_ALIGN,
        )
        self.layout_hash = compute_wire_layout_hash(self.layout, self.result_slot_stride)
        plans_base = SHM_SUPERBLOCK_REGION_BYTES + WORKER_STATES_REGION_BYTES
        self.plan_slot_offsets = (plans_base, plans_base + self.layout.total_slot_bytes)
        results_base = plans_base + 2 * self.layout.total_slot_bytes
        self.result_slot_offsets = (results_base, results_base + self.result_slot_stride)
        self.total_shm_bytes = results_base + 2 * self.result_slot_stride

        self.shm_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(self.path),
            os.O_CREAT | os.O_TRUNC | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            os.ftruncate(fd, self.total_shm_bytes)
            self._mapping = mmap.mmap(
                fd,
                self.total_shm_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)

        self._write_superblock()
        for slot in (0, 1):
            pt = self._array(
                self.plan_slot_offsets[slot] + self.layout.global_kv_page_table_offset,
                self.geometry.max_seqs * self.geometry.max_pages_per_seq,
                np.int32,
            )
            pt.fill(-1)
        self._closed = False

    def _write_superblock(self) -> None:
        g = self.geometry
        struct.pack_into(
            "<Q9I",
            self._mapping,
            0,
            ZSTEP_SHM_MAGIC,
            ZSTEP_PROTOCOL_VERSION,
            g.world_size,
            g.max_seqs,
            g.max_step_tokens,
            g.max_decoder_rows,
            g.max_pages_per_seq,
            g.page_block_tokens,
            g.spec_verify_width,
            os.getpid(),
        )
        struct.pack_into("<Q", self._mapping, SUPERBLOCK_TOTAL_BYTES_OFFSET, self.total_shm_bytes)
        struct.pack_into(
            "<2Q",
            self._mapping,
            SUPERBLOCK_PLAN_OFFSETS_OFFSET,
            self.plan_slot_offsets[0],
            self.plan_slot_offsets[1],
        )
        struct.pack_into(
            "<Q", self._mapping, SUPERBLOCK_PLAN_STRIDE_OFFSET, self.layout.total_slot_bytes
        )
        struct.pack_into(
            "<2Q",
            self._mapping,
            SUPERBLOCK_RESULT_OFFSETS_OFFSET,
            self.result_slot_offsets[0],
            self.result_slot_offsets[1],
        )
        struct.pack_into(
            "<Q", self._mapping, SUPERBLOCK_RESULT_STRIDE_OFFSET, self.result_slot_stride
        )
        struct.pack_into("<Q", self._mapping, SUPERBLOCK_LAYOUT_HASH_OFFSET, self.layout_hash)
        struct.pack_into(
            "<I",
            self._mapping,
            SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET,
            self.max_decode_graph_seqs,
        )

    def _array(self, offset: int, count: int, dtype: Any) -> np.ndarray:
        return np.frombuffer(self._mapping, dtype=np.dtype(dtype), count=count, offset=offset)

    def assign_linear_page_tables(self, *, total_seqs: int, pages_per_seq: int) -> None:
        """Populate 0-based KV page indices (`-1` = null block) in both parity slots."""
        max_b = self.geometry.max_seqs
        max_p = self.geometry.max_pages_per_seq
        if total_seqs > max_b or pages_per_seq > max_p:
            raise ValueError(
                f"Requested ({total_seqs}, {pages_per_seq}) exceeds geometry ({max_b}, {max_p})"
            )
        for slot in (0, 1):
            pt = self._array(
                self.plan_slot_offsets[slot] + self.layout.global_kv_page_table_offset,
                max_b * max_p,
                np.int32,
            ).reshape(max_b, max_p)
            pt.fill(-1)
            for s in range(total_seqs):
                base_page = s * pages_per_seq
                pt[s, :pages_per_seq] = np.arange(
                    base_page, base_page + pages_per_seq, dtype=np.int32
                )

    def publish_prefill_step(
        self,
        *,
        step_id: int,
        slot_ids: Sequence[int],
        context_len: int,
        prompt_seed: int = 101,
    ) -> None:
        num_seqs = len(slot_ids)
        total_tokens = num_seqs * context_len
        if total_tokens > self.geometry.max_step_tokens:
            raise ValueError(
                f"Prefill tokens {total_tokens} exceed max_step_tokens={self.geometry.max_step_tokens}"
            )
        base = self.plan_slot_offsets[step_id & 1]
        tok = self._array(base + self.layout.token_ids_offset, total_tokens, np.int32)
        pos = self._array(base + self.layout.positions_offset, total_tokens, np.int32)
        req_slots = self._array(base + self.layout.req_slot_ids_offset, total_tokens, np.int32)
        seq_lens = self._array(base + self.layout.seq_lens_offset, num_seqs, np.int32)
        qsl = self._array(base + self.layout.query_start_loc_offset, num_seqs + 1, np.int32)
        prev_win = self._array(
            base + self.layout.engram_prev_window_offset, num_seqs * 3, np.int32
        ).reshape(num_seqs, 3)
        sample_rows = self._array(base + self.layout.sample_row_indices_offset, num_seqs, np.int32)

        qsl[0] = 0
        for i, s_id in enumerate(slot_ids):
            start = i * context_len
            end = start + context_len
            qsl[i + 1] = end
            seq_lens[i] = context_len
            pos[start:end] = np.arange(context_len, dtype=np.int32)
            req_slots[start:end] = int(s_id)
            stream = self._seq_stream(int(s_id), context_len + 64, prompt_seed)
            tok[start:end] = stream[:context_len]
            prev_win[i, :] = -1
            sample_rows[i] = end - 1

        self._publish_header(
            base=base,
            step_id=step_id,
            opcode=OPCODE_NORMAL_STEP,
            num_seqs=num_seqs,
            num_encoder_tokens=total_tokens,
            encoder_graph_bucket=total_tokens,
            num_decoder_rows=total_tokens,
            decoder_graph_bucket=0,
            num_sample_rows=num_seqs,
            flags=0,
        )

    def _seq_stream(self, slot_id: int, min_len: int, seed: int) -> np.ndarray:
        if self.token_streams:
            base = self.token_streams[slot_id % len(self.token_streams)]
            if len(base) >= min_len:
                return base
            reps = (min_len + len(base) - 1) // len(base)
            return np.tile(base, reps)[:min_len]
        return ((np.arange(min_len, dtype=np.int32) * 13 + seed + slot_id * 7) % 4096) + 16

    def publish_decode_step(
        self,
        *,
        step_id: int,
        num_seqs: int,
        decode_pos: int,
        decode_graph_bucket: int,
        token_seed: int = 101,
    ) -> None:
        base = self.plan_slot_offsets[step_id & 1]
        tok = self._array(base + self.layout.token_ids_offset, num_seqs, np.int32)
        pos = self._array(base + self.layout.positions_offset, num_seqs, np.int32)
        req_slots = self._array(base + self.layout.req_slot_ids_offset, num_seqs, np.int32)
        seq_lens = self._array(base + self.layout.seq_lens_offset, num_seqs, np.int32)
        qsl = self._array(base + self.layout.query_start_loc_offset, num_seqs + 1, np.int32)
        prev_win = self._array(
            base + self.layout.engram_prev_window_offset, num_seqs * 3, np.int32
        ).reshape(num_seqs, 3)
        sample_rows = self._array(base + self.layout.sample_row_indices_offset, num_seqs, np.int32)

        idx = np.arange(num_seqs, dtype=np.int32)
        pos[:] = int(decode_pos)
        req_slots[:] = idx
        seq_lens[:] = int(decode_pos) + 1
        qsl[:] = np.arange(num_seqs + 1, dtype=np.int32)
        sample_rows[:] = idx
        for s in range(num_seqs):
            stream = self._seq_stream(s, decode_pos + 4, token_seed)
            tok[s] = int(stream[decode_pos])
            prev_win[s, 0] = int(stream[decode_pos - 3]) if decode_pos >= 3 else -1
            prev_win[s, 1] = int(stream[decode_pos - 2]) if decode_pos >= 2 else -1
            prev_win[s, 2] = int(stream[decode_pos - 1]) if decode_pos >= 1 else -1

        self._publish_header(
            base=base,
            step_id=step_id,
            opcode=OPCODE_NORMAL_STEP,
            num_seqs=num_seqs,
            num_encoder_tokens=num_seqs,
            encoder_graph_bucket=0,
            num_decoder_rows=num_seqs,
            decoder_graph_bucket=decode_graph_bucket,
            num_sample_rows=num_seqs,
            flags=STEP_FLAG_DECODE_ONLY,
        )

    def _publish_header(
        self,
        *,
        base: int,
        step_id: int,
        opcode: int,
        num_seqs: int,
        num_encoder_tokens: int,
        encoder_graph_bucket: int,
        num_decoder_rows: int,
        decoder_graph_bucket: int,
        num_sample_rows: int,
        flags: int,
    ) -> None:
        struct.pack_into(
            "<12I",
            self._mapping,
            base + 8,
            opcode,
            num_seqs,
            num_encoder_tokens,
            encoder_graph_bucket,
            num_decoder_rows,
            decoder_graph_bucket,
            num_sample_rows,
            0,
            flags,
            0,
            0,
            0,
        )
        struct.pack_into("<Q", self._mapping, base, int(step_id))

    def close(self, *, unlink: bool = True) -> None:
        if not self._closed:
            self._mapping.close()
            self._closed = True
        if unlink:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


class WorkbenchDaemonConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18765
    runner_kind: str = "deepseek_v41"  # "deepseek_v41" | "micro_step" | "mock"
    world_size: int = 8
    worker_root: str = ""
    model_dir: str = ""
    shm_name: str = ""
    num_kv_pages: int = 2560
    allreduce_blocks: int = 8
    allreduce_threads: int = 256
    active_decode_buckets: tuple[int, ...] = (1, 16, 32)
    startup_timeout_sec: float = 900.0
    rpc_watchdog_timeout_sec: float = 30.0
    envelope_db_path: str = ""


class BucketScreenMetrics(BaseModel):
    bucket: int
    steps: int
    mean_step_us: float
    median_step_us: float
    p99_step_us: float
    throughput_tok_s: float
    sample_tokens: list[int] = Field(default_factory=list)


class InnerScreenReport(BaseModel):
    ok: bool
    elapsed_ms: float
    context_len: int
    geomean_tok_s: float
    eager_fallback_steps: int
    graph_guard_failures: int
    graph_replay_steps: int
    triton_recompilations: int
    buckets: list[BucketScreenMetrics] = Field(default_factory=list)


def recapture_dsv41_decode_buckets(runner: Any, buckets: Sequence[int]) -> dict[int, Any]:
    """Selectively recapture `torch.cuda.CUDAGraph` decode buckets on resident 475 GiB runner.

    Returns ONLY `{bucket: new_raw_cudagraph}` for the recaptured buckets so un-recaptured
    buckets retain their revoked `GuardedCudaGraphProxy` and fail-closed if accidentally invoked.
    """
    import torch
    import torch.distributed as dist
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

    target_buckets = [int(b) for b in buckets if 0 < int(b) <= runner.max_graph_bucket]
    if not target_buckets:
        return {}

    unlock_workspace()
    pre_pw_hooks = getattr(runner, "_evolith_pre_pw_hooks", ())
    post_pw_hooks = getattr(runner, "_evolith_post_pw_hooks", ())
    for obj, attr, val in pre_pw_hooks:
        setattr(obj, attr, val)
    try:
        if getattr(runner, "_evolith_graph_pool", None) is None and runner.decode_graphs:
            any_g = next(iter(GuardedGraphRunner.unwrap_cuda_graphs(runner.decode_graphs).values()))
            if hasattr(any_g, "pool"):
                runner._evolith_graph_pool = any_g.pool()
        for b in target_buckets:
            runner.decode_graphs.pop(b, None)
        gc.collect()
        torch.cuda.empty_cache()
        runner._zero_all_caches()
        dist.barrier(group=runner.exchange_group)

        runner.mla_bt_gpu.zero_()
        runner.swa_bt_gpu.zero_()
        runner.comp_bt_gpu.fill_(runner.dummy_comp_block)
        runner.lookback_gpu.fill_(-1)
        runner.slot_mapping_gpu.fill_(-1)
        if runner._swa_pool is not None:
            runner.swa_slot_mapping_gpu.fill_(-1)
        runner.seq_lens_gpu.zero_()

        graph_pool = getattr(runner, "_evolith_graph_pool", None)
        if graph_pool is None:
            graph_pool = torch.cuda.graph_pool_handle()
            runner._evolith_graph_pool = graph_pool
        capture_stream = torch.cuda.Stream(device=runner.device)
        fresh_raw: dict[int, Any] = {}

        with torch.inference_mode():
            for b in target_buckets:
                runner.token_ids_gpu[:b].zero_()
                runner.positions_gpu[:b].zero_()
                runner.seq_lens_gpu[:b].fill_(1)
                runner.seq_lens_gpu[b:].zero_()
                runner.slot_mapping_gpu[:b].fill_(-1)
                sl_cpu = torch.ones(b, dtype=torch.int32)

                with torch.cuda.stream(capture_stream):
                    for w_idx in range(2):
                        attn_meta = runner._build_step_attn_metadata(
                            num_reqs=b,
                            num_tokens=b,
                            max_query_len=1,
                            max_seq_len=runner.decode_graph_max_seq_len,
                            qsl_gpu=runner.decode_qsl_gpu[: b + 1],
                            qsl_cpu=runner.decode_qsl_cpu[: b + 1],
                            seq_lens_gpu=runner.seq_lens_gpu[:b],
                            seq_lens_cpu=sl_cpu,
                            positions_gpu=runner.positions_gpu[:b],
                            slot_mapping_gpu=runner.slot_mapping_gpu[:b],
                            swa_slot_mapping_gpu=runner.swa_slot_mapping_gpu[:b],
                            is_prefilling_cpu=runner.decode_is_prefilling_cpu[:b],
                            for_capture=(w_idx == 0),
                        )
                        with set_forward_context(
                            attn_meta,
                            runner.vllm_config,
                            num_tokens=b,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        ):
                            if runner.pp_size > 1 and not runner.is_first_pp:
                                from vllm.sequence import IntermediateTensors

                                d_inter = IntermediateTensors(
                                    {
                                        "hidden_states": runner._pp_inter_hidden[:b],
                                        "pre_mix": runner._pp_inter_pre_mix[:b],
                                    }
                                )
                                hidden = runner.model(
                                    input_ids=runner.token_ids_gpu[:b],
                                    positions=runner.positions_gpu[:b],
                                    intermediate_tensors=d_inter,
                                    lookback_token_ids=runner.lookback_gpu,
                                )
                            else:
                                hidden = runner.model(
                                    input_ids=runner.token_ids_gpu[:b],
                                    positions=runner.positions_gpu[:b],
                                    lookback_token_ids=runner.lookback_gpu,
                                )
                            if runner.is_last_pp:
                                runner._sample_rows_into_argmax(
                                    hidden, None, b, all_ranks=True
                                )
                            elif (
                                runner._pp_inter_hidden is not None
                                and runner._pp_inter_pre_mix is not None
                            ):
                                runner._pp_inter_hidden[:b].copy_(hidden["hidden_states"])
                                runner._pp_inter_pre_mix[:b].copy_(hidden["pre_mix"])
                capture_stream.synchronize()
                dist.barrier(group=runner.exchange_group)

                with torch.cuda.stream(capture_stream):
                    attn_meta = runner._build_step_attn_metadata(
                        num_reqs=b,
                        num_tokens=b,
                        max_query_len=1,
                        max_seq_len=runner.decode_graph_max_seq_len,
                        qsl_gpu=runner.decode_qsl_gpu[: b + 1],
                        qsl_cpu=runner.decode_qsl_cpu[: b + 1],
                        seq_lens_gpu=runner.seq_lens_gpu[:b],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=runner.positions_gpu[:b],
                        slot_mapping_gpu=runner.slot_mapping_gpu[:b],
                        swa_slot_mapping_gpu=runner.swa_slot_mapping_gpu[:b],
                        is_prefilling_cpu=runner.decode_is_prefilling_cpu[:b],
                        for_capture=True,
                    )
                capture_stream.synchronize()
                g = torch.cuda.CUDAGraph()
                with set_forward_context(
                    attn_meta,
                    runner.vllm_config,
                    num_tokens=b,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ):
                    with torch.cuda.graph(g, pool=graph_pool, stream=capture_stream):
                        runner._build_step_attn_metadata(
                            num_reqs=b,
                            num_tokens=b,
                            max_query_len=1,
                            max_seq_len=runner.decode_graph_max_seq_len,
                            qsl_gpu=runner.decode_qsl_gpu[: b + 1],
                            qsl_cpu=runner.decode_qsl_cpu[: b + 1],
                            seq_lens_gpu=runner.seq_lens_gpu[:b],
                            seq_lens_cpu=sl_cpu,
                            positions_gpu=runner.positions_gpu[:b],
                            slot_mapping_gpu=runner.slot_mapping_gpu[:b],
                            swa_slot_mapping_gpu=runner.swa_slot_mapping_gpu[:b],
                            is_prefilling_cpu=runner.decode_is_prefilling_cpu[:b],
                            for_capture=False,
                        )
                        if runner.pp_size > 1 and not runner.is_first_pp:
                            from vllm.sequence import IntermediateTensors

                            d_inter = IntermediateTensors(
                                {
                                    "hidden_states": runner._pp_inter_hidden[:b],
                                    "pre_mix": runner._pp_inter_pre_mix[:b],
                                }
                            )
                            hidden = runner.model(
                                input_ids=runner.token_ids_gpu[:b],
                                positions=runner.positions_gpu[:b],
                                intermediate_tensors=d_inter,
                                lookback_token_ids=runner.lookback_gpu,
                            )
                        else:
                            hidden = runner.model(
                                input_ids=runner.token_ids_gpu[:b],
                                positions=runner.positions_gpu[:b],
                                lookback_token_ids=runner.lookback_gpu,
                            )
                        if runner.is_last_pp:
                            runner._sample_rows_into_argmax(
                                hidden, None, b, all_ranks=True
                            )
                        elif (
                            runner._pp_inter_hidden is not None
                            and runner._pp_inter_pre_mix is not None
                        ):
                            runner._pp_inter_hidden[:b].copy_(hidden["hidden_states"])
                            runner._pp_inter_pre_mix[:b].copy_(hidden["pre_mix"])
                capture_stream.synchronize()
                dist.barrier(group=runner.exchange_group)
                fresh_raw[b] = g

        runner._zero_all_caches()
        dist.barrier(group=runner.exchange_group)
        return fresh_raw
    finally:
        for obj, attr, val in post_pw_hooks:
            setattr(obj, attr, val)
        lock_workspace()


def top2_margin_consistent_rate(
    inc_tokens: Sequence[int],
    inc_logprobs: Sequence[float],
    inc_ru_tokens: Sequence[int],
    inc_ru_logprobs: Sequence[float],
    cand_tokens: Sequence[int],
    cand_logprobs: Sequence[float],
    cand_ru_tokens: Sequence[int] = (),
    cand_ru_logprobs: Sequence[float] = (),
    *,
    tie_margin: float = 1.5,
    logprob_atol: float = 5.0,
) -> float:
    """Compute top-2 tie-aware greedy parity rate between incumbent and candidate.

    - Confident positions (`inc_lp - inc_ru_lp > tie_margin`) MUST match `inc_tok == cand_tok`
      strictly and stay within `logprob_atol`.
    - Genuine BF16 near-ties (`inc_lp - inc_ru_lp <= tie_margin`) allow symmetric top-2 swap
      only when top-2 sets overlap (`cand_tok == inc_ru_tok` or `inc_tok == cand_ru_tok`)
      and `abs(inc_lp - cand_lp) <= logprob_atol`.
    """
    _ = cand_ru_logprobs
    if not inc_tokens:
        return 1.0
    consistent = 0
    for idx, (itok, ctok, ilp, clp) in enumerate(
        zip(inc_tokens, cand_tokens, inc_logprobs, cand_logprobs, strict=True)
    ):
        if abs(ilp - clp) > logprob_atol:
            continue
        if itok == ctok:
            consistent += 1
            continue
        if idx < len(inc_ru_tokens) and idx < len(inc_ru_logprobs):
            inc_margin = ilp - inc_ru_logprobs[idx]
            inc_top2 = {itok, inc_ru_tokens[idx]}
            cand_top2 = {ctok, cand_ru_tokens[idx]} if idx < len(cand_ru_tokens) else {ctok}
            if inc_margin <= tie_margin and bool(inc_top2 & cand_top2):
                consistent += 1
    return consistent / len(inc_tokens)


def inner_screen_consistency_rate(
    ref_screen: dict[str, Any],
    cand_screen: dict[str, Any],
    *,
    tie_margin: float = 1.5,
    logprob_atol: float = 5.0,
) -> float:
    """Compute sample-weighted `top2_margin_consistent_rate` across all screen buckets."""
    total_samples = 0
    total_ok = 0.0
    for rb, cb in zip(ref_screen.get("buckets", ()), cand_screen.get("buckets", ()), strict=True):
        n = len(rb.get("sample_tokens", ()))
        if n <= 0:
            continue
        r = top2_margin_consistent_rate(
            rb["sample_tokens"],
            rb["sample_logprobs"],
            rb.get("runner_up_tokens", ()),
            rb.get("runner_up_logprobs", ()),
            cb["sample_tokens"],
            cb["sample_logprobs"],
            cb.get("runner_up_tokens", ()),
            cb.get("runner_up_logprobs", ()),
            tie_margin=tie_margin,
            logprob_atol=logprob_atol,
        )
        total_samples += n
        total_ok += r * n
    return (total_ok / total_samples) if total_samples > 0 else 1.0


class _MockGraph:
    def __init__(self, scale_ref: list[float], bucket: int) -> None:
        self._scale_ref = scale_ref
        self._bucket = bucket

    def replay(self) -> float:
        return self._bucket * self._scale_ref[0]


class _MockGpuRunner:
    """Fast deterministic runner used for CPU/macOS daemon protocol & guard tests."""

    def __init__(self, rank: int, world_size: int, buckets: Sequence[int]) -> None:
        self.rank = rank
        self.world_size = world_size
        self.is_result_rank = rank == 0
        all_init_buckets = sorted({*buckets, 64})
        self.max_graph_bucket = max(all_init_buckets)
        self.latency_scale: list[float] = [1.0]
        self.token_bias: int = 0
        self.weight_array = np.ones(64, dtype=np.float32)
        self.kv_pool_bytes = 64 * 1024 * 1024
        self._kv_freed = False
        self.decode_graphs: dict[int, Any] = {
            int(b): _MockGraph(self.latency_scale, int(b)) for b in all_init_buckets
        }

    def zero_caches(self) -> None:
        pass

    def recapture(self, buckets: Sequence[int]) -> dict[int, Any]:
        return {int(b): _MockGraph(self.latency_scale, int(b)) for b in buckets}

    def run_prefill(
        self, _plan: Any, _header: Any
    ) -> tuple[list[tuple[int, int, float, int, float]], int]:
        return [], 250

    def run_decode(
        self, bucket: int, num_seqs: int, decode_pos: int
    ) -> tuple[list[tuple[int, int, float, int, float]], int]:
        if self._kv_freed:
            raise RuntimeError("KV pool is sleeping/released")
        self.decode_graphs[bucket].replay()
        exec_us = max(1, int(round((800.0 + 18.0 * bucket) * self.latency_scale[0])))
        outs = [
            (
                s,
                int(((s * 17 + decode_pos * 13 + 41) % 4096) + self.token_bias),
                float(-0.25 - 0.01 * (s % 8)),
                int(((s * 17 + decode_pos * 13 + 42) % 4096) + self.token_bias),
                float(-5.25 - 0.01 * (s % 8)),
            )
            for s in range(num_seqs)
        ]
        return (outs if self.is_result_rank else []), exec_us

    def free_kv_storage(self) -> int:
        if self._kv_freed:
            return 0
        self._kv_freed = True
        return self.kv_pool_bytes

    def restore_kv_storage(self) -> int:
        if not self._kv_freed:
            return 0
        self._kv_freed = False
        return self.kv_pool_bytes

    def vram_info_mib(self) -> tuple[int, int]:
        used = 59400 + (0 if self._kv_freed else 64)
        total = 85651
        return total - used, total

    def profile_stages(self, bucket: int) -> dict[str, Any]:
        if self._kv_freed:
            raise RuntimeError("KV pool is sleeping/released")
        scale = self.latency_scale[0]
        stages = {
            "dsv4.b12x_moe_ffn": {
                "step_us": round((340.0 + 9.0 * bucket) * scale, 2),
                "calls": 40,
                "shapes": [[bucket, 5120]],
                "strides": [[5120, 1]],
                "dtype": "bfloat16",
            },
            "dsv4.mhc_allreduce": {
                "step_us": round((220.0 + 4.5 * bucket) * scale, 2),
                "calls": 80,
                "shapes": [[bucket, 5120], [bucket, 4, 5120]],
                "strides": [[5120, 1], [20480, 5120, 1]],
                "dtype": "bfloat16",
            },
            "dsv4.mla_dsa_attn": {
                "step_us": round((160.0 + 3.0 * bucket) * scale, 2),
                "calls": 40,
                "shapes": [[bucket, 5120]],
                "strides": [[5120, 1]],
                "dtype": "bfloat16",
            },
            "dsv4.lm_head_logits": {
                "step_us": round((38.0 + 0.8 * bucket) * scale, 2),
                "calls": 1,
                "shapes": [[bucket, 5120]],
                "strides": [[5120, 1]],
                "dtype": "bfloat16",
            },
            "dsv4.shm_top2_argmax": {
                "step_us": round((18.0 + 0.3 * bucket) * scale, 2),
                "calls": 1,
                "shapes": [[bucket, 16160]],
                "strides": [[16160, 1]],
                "dtype": "bfloat16",
            },
        }
        attr_sum = sum(float(v["step_us"]) for v in stages.values())
        return {
            "rank": self.rank,
            "instrumented_step_us": round(attr_sum + 24.0 * scale, 2),
            "stages": stages,
        }

    def close(self, *, abort: bool = False) -> None:
        _ = abort


class _RankResidentContext:
    """Per-rank resident state inside a spawned worker process (or in-process mock)."""

    def __init__(
        self,
        cfg: WorkbenchDaemonConfig,
        rank: int,
        dist_init_file: str,
        shm_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world_size = cfg.world_size
        self.shm_dir = shm_dir
        self.state_keeper = StateKeeper(rank=rank, world_size=cfg.world_size)
        self.state_keeper.reattach_scheduler_channel(cfg.shm_name)
        env_log = EnvelopeFailureLog(Path(cfg.envelope_db_path)) if cfg.envelope_db_path else None
        self.engine = MorphableWorkbenchEngine(
            default_zstep_surface(),
            state_keeper=self.state_keeper,
            graph_mode="required",
            graph_buckets=cfg.active_decode_buckets,
            envelope_log=env_log,
        )
        self._undo_fn: Callable[[], None] | None = None
        self._prev_catalog_ops: dict[str, Callable[..., Any]] | None = None
        self._native_loader = ExternCOpLoader(Path(dist_init_file).parent / "native_cache")
        self._bridge: Any = None
        self._kv_storage_records: list[tuple[Any, int]] = []
        self._watched_triton: list[tuple[str, Any]] = []
        self._triton_recompiles: int = 0

        if cfg.runner_kind == "mock":
            self.runner: Any = _MockGpuRunner(rank, cfg.world_size, cfg.active_decode_buckets)
            self.state_keeper.register_buffer(
                "weights",
                self.runner.weight_array,
                nbytes=int(59_400 * 1024 * 1024),
                persistent_across_sleep=True,
            )
            self.state_keeper.register_buffer(
                "shm_collectives",
                {"pxb": True},
                nbytes=256 * 1024 * 1024,
                persistent_across_sleep=True,
            )
            self.state_keeper.register_buffer(
                "kv_pools",
                {"active": True},
                nbytes=self.runner.kv_pool_bytes,
                persistent_across_sleep=False,
                wake_factory=lambda: {"active": True},
            )
            self._bind_and_wrap_graphs(self.runner.decode_graphs)
        elif cfg.runner_kind == "deepseek_v41":
            self._init_dsv41(dist_init_file)
        else:
            raise ValueError(f"Unsupported runner_kind `{cfg.runner_kind}`")

    def _init_dsv41(self, dist_init_file: str) -> None:
        if self.cfg.worker_root and self.cfg.worker_root not in sys.path:
            sys.path.insert(0, self.cfg.worker_root)
        import torch

        from zstep_worker.models.deepseek_v41 import DeepSeekV41StepRunner
        from zstep_worker.numa_pin import pin_process_to_device_numa
        from zstep_worker.shm_bridge import ShmBridge

        orig_install_pw = getattr(DeepSeekV41StepRunner, "_install_piecewise_hooks", None)
        if callable(orig_install_pw) and not getattr(
            DeepSeekV41StepRunner, "_evolith_pw_wrapped", False
        ):

            def _saving_install_pw(runner_self: Any) -> None:
                saved_pre: list[tuple[Any, str, Any]] = []
                lang_m = getattr(runner_self.model, "language_model", runner_self.model)
                inner_m = getattr(lang_m, "model", lang_m)
                eh = getattr(inner_m, "engram_hash", None)
                if eh is not None and hasattr(eh, "forward"):
                    saved_pre.append((eh, "forward", eh.forward))
                for lyr in getattr(inner_m, "layers", ()):
                    attn = getattr(lyr, "attn", None)
                    if attn is not None:
                        for attr in ("_prepare_and_attn", "_prepare_and_attn_fn"):
                            if hasattr(attn, attr):
                                saved_pre.append((attn, attr, getattr(attn, attr)))
                orig_install_pw(runner_self)
                runner_self._evolith_pre_pw_hooks = saved_pre
                runner_self._evolith_post_pw_hooks = [
                    (obj, attr, getattr(obj, attr)) for obj, attr, _ in saved_pre
                ]

            DeepSeekV41StepRunner._install_piecewise_hooks = _saving_install_pw  # type: ignore[method-assign]
            DeepSeekV41StepRunner._evolith_pw_wrapped = True  # type: ignore[attr-defined]

        pin_process_to_device_numa(self.rank)
        self._bridge = ShmBridge(name=self.cfg.shm_name)
        self.runner = DeepSeekV41StepRunner(
            model_dir=Path(self.cfg.model_dir),
            rank=self.rank,
            bridge=self._bridge,
            dist_init_file=dist_init_file,
            allreduce_name=f"{self.cfg.shm_name}_ar",
            allreduce_blocks=self.cfg.allreduce_blocks,
            allreduce_threads=self.cfg.allreduce_threads,
            num_kv_pages=self.cfg.num_kv_pages,
        )

        weight_bytes = sum(
            int(t.numel() * t.element_size())
            for t in chain(self.runner.model.parameters(), self.runner.model.buffers())
        )
        shm_bytes = 0
        for attr in ("shm_ar", "shm_ar_tbo", "shm_ar_decode", "shm_ag"):
            comm = getattr(self.runner, attr, None)
            if comm is not None:
                cap = int(
                    getattr(comm, "capacity_elements", getattr(comm, "capacity_input_elements", 0))
                )
                shm_bytes += cap * 2

        seen_ptrs: set[int] = set()
        kv_bytes = 0
        for raw_t in getattr(self.runner, "_kv_raw_buffers", []):
            st = raw_t.untyped_storage()
            sid = int(st._cdata) if hasattr(st, "_cdata") else id(st)
            if sid not in seen_ptrs:
                seen_ptrs.add(sid)
                nb = int(st.nbytes())
                self._kv_storage_records.append((st, nb))
                kv_bytes += nb

        self.state_keeper.register_buffer(
            "weights",
            self.runner.model,
            nbytes=weight_bytes,
            persistent_across_sleep=True,
        )
        self.state_keeper.register_buffer(
            "shm_collectives",
            {"shm_ar": self.runner.shm_ar},
            nbytes=shm_bytes,
            persistent_across_sleep=True,
        )
        self.state_keeper.register_buffer(
            "kv_pools",
            self.runner._kv_raw_buffers,
            nbytes=kv_bytes,
            persistent_across_sleep=False,
            wake_factory=lambda: self.runner._kv_raw_buffers,
        )

        try:
            from zstep_worker.kernels_sm120.moe_router import dsv41_sqrt_softplus_topk_kernel
            from zstep_worker.kernels_sm120.norm_quant import rmsnorm_quantize_mxfp8_group32_kernel

            self._watched_triton.extend(
                [
                    ("norm_mxfp8", rmsnorm_quantize_mxfp8_group32_kernel),
                    ("router_dsv41", dsv41_sqrt_softplus_topk_kernel),
                ]
            )
        except ImportError:
            pass
        extra = getattr(self.runner, "watched_triton_kernels", None)
        if callable(extra):
            self._watched_triton.extend(extra())
        for _kname, kern in self._watched_triton:
            orig_compile = getattr(kern, "_do_compile", None)
            if callable(orig_compile) and not getattr(kern, "_evolith_spied", False):
                def _make_spy(fn: Callable[..., Any]) -> Callable[..., Any]:
                    def _wrapped(*a: Any, **kw: Any) -> Any:
                        self._triton_recompiles += 1
                        return fn(*a, **kw)
                    return _wrapped

                kern._do_compile = _make_spy(orig_compile)
                kern._evolith_spied = True

        if self.runner.decode_graphs:
            any_raw = next(iter(GuardedGraphRunner.unwrap_cuda_graphs(self.runner.decode_graphs).values()))
            if hasattr(any_raw, "pool"):
                self.runner._evolith_graph_pool = any_raw.pool()
        self._bind_and_wrap_graphs(self.runner.decode_graphs, replace_all=True)
        torch.cuda.synchronize(self.runner.device)

    def _bind_and_wrap_graphs(
        self, fresh_raw_graphs: dict[int, Any], *, replace_all: bool = False
    ) -> None:
        prov = self.engine.bind_step_runner(
            "models.runner",
            self.runner,
            lambda rows, x: (rows, x),
        )
        wrapped_subset = self.engine.graph_runner.wrap_cuda_graphs(
            fresh_raw_graphs,
            provisions=(prov,),
            buffers=self.state_keeper.all_buffers(),
            catalog=self.engine.catalog,
        )
        if replace_all:
            self.runner.decode_graphs = wrapped_subset
        else:
            merged = dict(self.runner.decode_graphs)
            merged.update(wrapped_subset)
            self.runner.decode_graphs = merged

    def vram_stats_mib(self) -> dict[str, int]:
        if self.cfg.runner_kind == "mock":
            free_m, total_m = self.runner.vram_info_mib()
        else:
            import torch

            free_b, total_b = torch.cuda.mem_get_info(self.runner.device)
            free_m = int(free_b // (1024 * 1024))
            total_m = int(total_b // (1024 * 1024))
        return {
            "rank": self.rank,
            "used_mib": total_m - free_m,
            "free_mib": free_m,
            "total_mib": total_m,
        }

    def status_payload(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "vram": self.vram_stats_mib(),
            "sleeping": self.state_keeper.sleeping,
            "active_version": self.engine.tx.active_version,
            "scheduler_attach_epoch": self.state_keeper.scheduler_attach_epoch,
            "attached_shm_channel": self.state_keeper.attached_shm_channel,
            "resident_bytes": self.state_keeper.resident_bytes,
            "buffers": [
                {
                    "name": b.name,
                    "nbytes": b.nbytes,
                    "generation": b.generation,
                    "fills": b.fills,
                    "released": b.released,
                }
                for b in self.state_keeper.all_buffers()
            ],
            "triton_recompilations": self._triton_recompiles,
            "graph_telemetry": self.engine.graph_runner.telemetry(),
        }

    def execute_published_step(
        self,
        *,
        step_id: int,
        is_decode: bool,
        bucket: int,
        num_seqs: int,
        decode_pos: int,
    ) -> tuple[list[tuple[int, int, float, int, float]], int]:
        if self.cfg.runner_kind == "mock":
            if not is_decode:
                return self.runner.run_prefill(None, None)
            return self.runner.run_decode(bucket, num_seqs, decode_pos)

        plan = self._bridge.plan(step_id)
        header = plan.header()
        launched = self.runner.launch(plan, header)
        seq_results, gpu_exec_us = self.runner.finish(launched)
        outs: list[tuple[int, int, float, int, float]] = []
        if self.runner.is_result_rank:
            for r in seq_results:
                lp = struct.unpack("<f", struct.pack("<I", r.accepted_logprob_bits & 0xFFFF_FFFF))[
                    0
                ]
                ru_lp = struct.unpack(
                    "<f", struct.pack("<I", r.runner_up_logprob_bits & 0xFFFF_FFFF)
                )[0]
                outs.append(
                    (
                        int(r.req_slot),
                        int(r.accepted_token),
                        float(lp),
                        int(r.runner_up_token),
                        float(ru_lp),
                    )
                )
        return outs, int(gpu_exec_us)

    def zero_all_caches(self) -> None:
        if self.cfg.runner_kind == "mock":
            self.runner.zero_caches()
        else:
            import torch
            import torch.distributed as dist

            self.runner._zero_all_caches()
            self.runner.mla_bt_gpu.zero_()
            self.runner.swa_bt_gpu.zero_()
            self.runner.comp_bt_gpu.fill_(self.runner.dummy_comp_block)
            self.runner.lookback_gpu.fill_(-1)
            self.runner.slot_mapping_gpu.fill_(-1)
            if self.runner._swa_pool is not None:
                self.runner.swa_slot_mapping_gpu.fill_(-1)
                if hasattr(self.runner._swa_pool, "reset"):
                    self.runner._swa_pool.reset()
            self.runner.seq_lens_gpu.zero_()
            self.runner.query_start_loc_gpu.zero_()
            if hasattr(self.runner, "slot_lb_gpu"):
                self.runner.slot_lb_gpu.fill_(-1)
            if hasattr(self.runner, "slot_last_tok_gpu"):
                self.runner.slot_last_tok_gpu.zero_()
            torch.cuda.synchronize(self.runner.device)
            dist.barrier(group=self.runner.exchange_group)

    def run_inner_screen_rank(self, cmd: dict[str, Any]) -> dict[str, Any]:
        from unittest.mock import patch

        buckets = [int(b) for b in cmd["buckets"]]
        context_len = int(cmd["context_len"])
        warmup_steps = int(cmd["warmup_steps"])
        prefill_step_ids: list[int] = [int(x) for x in cmd["prefill_step_ids"]]
        decode_schedule: dict[str, list[tuple[int, int]]] = cmd["decode_schedule"]

        self.zero_all_caches()
        recomp_count = 0
        with ExitStack() as stack:
            spies = []
            for _kname, kern in self._watched_triton:
                if hasattr(kern, "_do_compile"):
                    spies.append(
                        stack.enter_context(
                            patch.object(kern, "_do_compile", wraps=kern._do_compile)
                        )
                    )

            for sid in prefill_step_ids:
                self.execute_published_step(
                    step_id=sid,
                    is_decode=False,
                    bucket=0,
                    num_seqs=0,
                    decode_pos=0,
                )

            bucket_rows: list[dict[str, Any]] = []
            for b in buckets:
                sched = decode_schedule[str(b)]
                for sid, dpos in sched[:warmup_steps]:
                    self.execute_published_step(
                        step_id=sid,
                        is_decode=True,
                        bucket=b,
                        num_seqs=b,
                        decode_pos=dpos,
                    )

                before_replays = self.engine.graph_runner.graph_replay_steps
                step_times_us: list[int] = []
                first_tokens: list[int] = []
                first_logprobs: list[float] = []
                for idx, (sid, dpos) in enumerate(sched[warmup_steps:]):
                    outs, exec_us = self.execute_published_step(
                        step_id=sid,
                        is_decode=True,
                        bucket=b,
                        num_seqs=b,
                        decode_pos=dpos,
                    )
                    step_times_us.append(exec_us)
                    if idx == 0 and outs:
                        first_tokens = [t for _, t, _, _, _ in outs[: min(8, len(outs))]]
                        first_logprobs = [lp for _, _, lp, _, _ in outs[: min(8, len(outs))]]

                after_replays = self.engine.graph_runner.graph_replay_steps
                expected_replays = len(sched[warmup_steps:])
                if after_replays - before_replays != expected_replays:
                    missed = expected_replays - (after_replays - before_replays)
                    self.engine.graph_runner.eager_fallback_steps += max(1, missed)
                    raise RuntimeError(
                        f"TRACE_CUDA_GRAPH_REQUIRED: bucket={b} expected {expected_replays} "
                        f"guarded CUDAGraph replays, observed {after_replays - before_replays}"
                    )

                mean_us = float(statistics.fmean(step_times_us))
                median_us = float(statistics.median(step_times_us))
                sorted_us = sorted(step_times_us)
                p99_idx = min(len(sorted_us) - 1, max(0, math.ceil(0.99 * len(sorted_us)) - 1))
                p99_us = float(sorted_us[p99_idx])
                bucket_rows.append(
                    {
                        "bucket": b,
                        "steps": expected_replays,
                        "mean_step_us": round(mean_us, 2),
                        "median_step_us": round(median_us, 2),
                        "p99_step_us": round(p99_us, 2),
                        "throughput_tok_s": round((b * 1_000_000.0) / max(mean_us, 1e-6), 2),
                        "sample_tokens": first_tokens,
                        "sample_logprobs": first_logprobs,
                    }
                )
            recomp_count = sum(s.call_count for s in spies)

        return {
            "rank": self.rank,
            "triton_recompilations": recomp_count,
            "graph_telemetry": self.engine.graph_runner.telemetry(),
            "buckets": bucket_rows,
            "context_len": context_len,
        }

    def apply_candidate_rank(self, cmd: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(cmd.get("candidate_id", "wb-cand"))
        changed_files = tuple(str(x) for x in cmd.get("changed_files", ()))
        changed_config_keys = tuple(str(x) for x in cmd.get("changed_config_keys", ()))
        patch_text = str(cmd.get("patch_text", ""))
        hook_code = str(cmd.get("hook_code", ""))
        compiled_native = cmd.get("compiled_native_op")
        recapture_buckets = [int(b) for b in cmd.get("recapture_buckets", self.cfg.active_decode_buckets)]
        refit_weight = bool(cmd.get("refit_weights", False))
        watchdog_sec = float(cmd.get("watchdog_timeout_sec", 15.0))

        if patch_text.strip() and changed_files:
            findings = scan_patch_boundaries(
                PatchStats(
                    patch=patch_text.encode("utf-8"),
                    changed_lines=sum(
                        1
                        for ln in patch_text.splitlines()
                        if (ln.startswith("+") and not ln.startswith("+++"))
                        or (ln.startswith("-") and not ln.startswith("---"))
                    ),
                    changed_files=changed_files,
                )
            )
            blocking = [f.message for f in findings if f.severity == "blocking"]
            if blocking:
                return {
                    "committed": False,
                    "rolled_back": False,
                    "failure_kind": "redteam_blocked",
                    "failure_detail": "; ".join(blocking),
                    "active_version": self.engine.tx.active_version,
                }

        prev_undo = self._undo_fn
        pre_apply_catalog_ops = self.engine.catalog.snapshot_ops()
        new_undo_holder: list[Callable[[], None] | None] = [None]

        def _mut_fn(_eng: MorphableWorkbenchEngine) -> None:
            if isinstance(compiled_native, dict):
                comp = CompiledNativeOp(
                    op_name=str(compiled_native["op_name"]),
                    symbol_name=str(compiled_native["symbol_name"]),
                    content_hash=str(compiled_native["content_hash"]),
                    so_path=Path(str(compiled_native["so_path"])),
                    backend=str(compiled_native.get("backend", "cuda")),
                )
                native_fn = self._native_loader.bind_symbol_in_process(
                    comp,
                    argtypes=tuple(compiled_native.get("argtypes", ())),
                    restype=compiled_native.get("restype", "i32"),
                )
                self.engine.catalog.register(comp.op_name, native_fn)
            if hook_code.strip():
                ns: dict[str, Any] = {}
                exec(compile(hook_code, f"<workbench:{candidate_id}>", "exec"), ns)
                apply_fn = ns.get("apply")
                if not callable(apply_fn):
                    raise ValueError("`hook_code` must define `def apply(runner, engine): ...`")
                undo_cb = apply_fn(self.runner, self.engine)
                if callable(undo_cb):
                    new_undo_holder[0] = undo_cb

        refit_map = {"weights": lambda w: w} if refit_weight else None
        res = self.engine.hot_apply(
            candidate_id=candidate_id,
            changed_files=changed_files,
            changed_config_keys=changed_config_keys,
            patch_text=patch_text,
            mutation_fn=_mut_fn,
            refit_buffers=refit_map,
            watchdog_timeout_sec=watchdog_sec,
        )
        if not res.committed:
            if new_undo_holder[0] is not None:
                try:
                    new_undo_holder[0]()
                except Exception:
                    pass
            return res.model_dump(mode="json")

        try:
            raw_graphs = (
                self.runner.recapture(recapture_buckets)
                if self.cfg.runner_kind == "mock"
                else recapture_dsv41_decode_buckets(self.runner, recapture_buckets)
            )
            self._bind_and_wrap_graphs(raw_graphs)
        except Exception as exc:
            if new_undo_holder[0] is not None:
                try:
                    new_undo_holder[0]()
                except Exception:
                    pass
            raw_graphs = (
                self.runner.recapture(recapture_buckets)
                if self.cfg.runner_kind == "mock"
                else recapture_dsv41_decode_buckets(self.runner, recapture_buckets)
            )
            self._bind_and_wrap_graphs(raw_graphs)
            return {
                "committed": False,
                "rolled_back": True,
                "rung": int(res.rung),
                "active_version": self.engine.tx.active_version,
                "recaptured_graphs": False,
                "failure_kind": "recapture_error",
                "failure_detail": str(exc),
                "blast_radius": res.blast_radius.model_dump(mode="json"),
            }

        self._prev_undo_for_shadow = prev_undo
        self._prev_catalog_ops = pre_apply_catalog_ops
        self._undo_fn = new_undo_holder[0]
        return res.model_dump(mode="json")

    def rollback_last_committed_rank(self, recapture_buckets: Sequence[int]) -> None:
        if self._undo_fn is not None:
            self._undo_fn()
            self._undo_fn = getattr(self, "_prev_undo_for_shadow", None)
        if self._prev_catalog_ops is not None:
            self.engine.catalog.restore_ops(self._prev_catalog_ops)
            self._prev_catalog_ops = None
        raw_graphs = (
            self.runner.recapture(recapture_buckets)
            if self.cfg.runner_kind == "mock"
            else recapture_dsv41_decode_buckets(self.runner, recapture_buckets)
        )
        self._bind_and_wrap_graphs(raw_graphs)

    def sleep_rank(self) -> dict[str, Any]:
        before = self.vram_stats_mib()
        if self.cfg.runner_kind == "mock":
            self.runner.free_kv_storage()
            self.state_keeper.sleep()
        else:
            import torch
            import torch.distributed as dist

            self.runner.decode_graphs = {}
            self.runner.piecewise_graphs.clear()
            self.runner.verify_graphs.clear()
            gc.collect()
            for st, _nb in self._kv_storage_records:
                st.resize_(0)
            self.state_keeper.sleep()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.runner.device)
            dist.barrier(group=self.runner.exchange_group)
        after = self.vram_stats_mib()
        return {
            "rank": self.rank,
            "freed_vram_mib": max(0, before["used_mib"] - after["used_mib"]),
            "vram_after": after,
        }

    def wake_rank(self, recapture_buckets: Sequence[int]) -> dict[str, Any]:
        before = self.vram_stats_mib()
        if self.cfg.runner_kind == "mock":
            self.runner.restore_kv_storage()
            self.state_keeper.wake()
            raw_graphs = self.runner.recapture(recapture_buckets)
            self._bind_and_wrap_graphs(raw_graphs)
        else:
            import torch
            import torch.distributed as dist

            for st, orig_nb in self._kv_storage_records:
                st.resize_(orig_nb)
            for raw_t in getattr(self.runner, "_kv_raw_buffers", []):
                raw_t.zero_()
            self.state_keeper.wake()
            torch.cuda.synchronize(self.runner.device)
            dist.barrier(group=self.runner.exchange_group)
            raw_graphs = recapture_dsv41_decode_buckets(self.runner, recapture_buckets)
            self._bind_and_wrap_graphs(raw_graphs)
        after = self.vram_stats_mib()
        return {
            "rank": self.rank,
            "restored_vram_mib": max(0, after["used_mib"] - before["used_mib"]),
            "kv_generation": self.state_keeper.get_buffer("kv_pools").generation,
            "vram_after": after,
        }

    def reattach_shm_rank(self, new_shm_name: str) -> dict[str, Any]:
        if self.cfg.runner_kind != "mock":
            from zstep_worker.shm_bridge import ShmBridge

            old_bridge = self._bridge
            new_bridge = ShmBridge(name=new_shm_name)
            self._bridge = new_bridge
            self.runner.bridge = new_bridge
            if old_bridge is not None:
                old_bridge.close()
        epoch = self.state_keeper.reattach_scheduler_channel(new_shm_name)
        return {
            "rank": self.rank,
            "scheduler_attach_epoch": epoch,
            "attached_shm_channel": new_shm_name,
        }

    def begin_profile_mode_rank(self, bucket: int) -> None:
        self._end_profile_cleanup_if_needed()
        if self.cfg.runner_kind == "mock":
            self._prof_state = {"bucket": bucket, "mock": True}
            return

        import torch
        import torch.distributed as dist
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

        runner = self.runner
        if bucket not in runner.decode_graphs:
            raise ValueError(f"Bucket {bucket} is not an active decode graph bucket")

        cudart: ctypes.CDLL | None = None
        for lib_name in ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                cand_lib = ctypes.CDLL(lib_name)
                if hasattr(cand_lib, "cudaEventRecordWithFlags"):
                    cand_lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                    cand_lib.cudaEventCreate.restype = ctypes.c_int
                    cand_lib.cudaEventRecordWithFlags.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_uint,
                    ]
                    cand_lib.cudaEventRecordWithFlags.restype = ctypes.c_int
                    cand_lib.cudaEventElapsedTime.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                    ]
                    cand_lib.cudaEventElapsedTime.restype = ctypes.c_int
                    cand_lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
                    cand_lib.cudaEventDestroy.restype = ctypes.c_int
                    cudart = cand_lib
                    break
            except OSError:
                continue
        if cudart is None:
            raise RuntimeError("libcudart.so with cudaEventRecordWithFlags not found")

        created_raw_events: list[ctypes.c_void_p] = []

        def _new_ext_event() -> ctypes.c_void_p:
            ev = ctypes.c_void_p()
            rc = cudart.cudaEventCreate(ctypes.byref(ev))
            if rc != 0:
                raise RuntimeError(f"cudaEventCreate failed with code={rc}")
            created_raw_events.append(ev)
            return ev

        def _rec_ext_event(ev: ctypes.c_void_p, stream_handle: int) -> None:
            # `cudaEventRecordExternal = 0x01` records an external EVENT_RECORD graph node
            # inside `torch.cuda.graph(...)` that can be queried via `cudaEventElapsedTime`
            # after each `prof_g.replay()` with zero Python eager skew!
            rc = cudart.cudaEventRecordWithFlags(
                ev, ctypes.c_void_p(int(stream_handle)), ctypes.c_uint(1)
            )
            if rc != 0:
                raise RuntimeError(f"cudaEventRecordWithFlags(external) failed with code={rc}")

        pre_pw_hooks = getattr(runner, "_evolith_pre_pw_hooks", ())
        post_pw_hooks = getattr(runner, "_evolith_post_pw_hooks", ())

        saved_attrs: list[tuple[Any, str, bool, Any]] = []
        saved_globals: list[tuple[dict[str, Any], str, Any]] = []
        wrapped_cache: dict[int, Callable[..., Any]] = {}
        active_depth = [0]
        record_ext_flag = [False]
        capture_meta_flag = [False]
        stage_meta: dict[str, dict[str, Any]] = {}
        captured_ext_pairs: dict[str, list[tuple[ctypes.c_void_p, ctypes.c_void_p]]] = {}
        whole_ext_pair: list[tuple[ctypes.c_void_p, ctypes.c_void_p]] = []
        active_capture_stream_handle = [0]

        def _capture_sig(stage_key: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if stage_key in stage_meta:
                return
            tensors = [
                a
                for a in (*args, *kwargs.values())
                if isinstance(a, torch.Tensor) and a.ndim >= 1
            ][:2]
            if tensors:
                stage_meta[stage_key] = {
                    "shapes": [[int(d) for d in t.shape] for t in tensors],
                    "strides": [[int(d) for d in t.stride()] for t in tensors],
                    "dtype": str(tensors[0].dtype).replace("torch.", ""),
                }

        def _make_wrapper(orig_fn: Callable[..., Any], stage_key: str) -> Callable[..., Any]:
            cached = wrapped_cache.get(id(orig_fn))
            if cached is not None:
                return cached

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                if active_depth[0] > 0:
                    return orig_fn(*args, **kwargs)
                if capture_meta_flag[0]:
                    _capture_sig(stage_key, args, kwargs)
                if not record_ext_flag[0]:
                    active_depth[0] += 1
                    try:
                        return orig_fn(*args, **kwargs)
                    finally:
                        active_depth[0] -= 1
                sh = active_capture_stream_handle[0]
                ev_s = _new_ext_event()
                ev_e = _new_ext_event()
                _rec_ext_event(ev_s, sh)
                active_depth[0] += 1
                try:
                    return orig_fn(*args, **kwargs)
                finally:
                    active_depth[0] -= 1
                    _rec_ext_event(ev_e, sh)
                    captured_ext_pairs.setdefault(stage_key, []).append((ev_s, ev_e))

            wrapped_cache[id(orig_fn)] = _wrapped
            return _wrapped

        def _patch_attr(obj: Any, attr: str, stage_key: str) -> None:
            if obj is None or not hasattr(obj, attr):
                return
            inst_dict = getattr(obj, "__dict__", {})
            had_inst = isinstance(inst_dict, dict) and (attr in inst_dict)
            orig_val = getattr(obj, attr)
            if not callable(orig_val):
                return
            saved_attrs.append((obj, attr, had_inst, orig_val))
            setattr(obj, attr, _make_wrapper(orig_val, stage_key))

        def _patch_global(gdict: dict[str, Any], key: str, stage_key: str) -> None:
            if key not in gdict or not callable(gdict[key]):
                return
            orig_val = gdict[key]
            saved_globals.append((gdict, key, orig_val))
            gdict[key] = _make_wrapper(orig_val, stage_key)

        unlock_workspace()
        for obj, attr, val in pre_pw_hooks:
            setattr(obj, attr, val)
        prof_g: Any = None
        try:
            if getattr(runner, "_evolith_graph_pool", None) is None and runner.decode_graphs:
                any_g = next(
                    iter(GuardedGraphRunner.unwrap_cuda_graphs(runner.decode_graphs).values())
                )
                if hasattr(any_g, "pool"):
                    runner._evolith_graph_pool = any_g.pool()
            runner.decode_graphs.pop(bucket, None)
            gc.collect()
            torch.cuda.empty_cache()
            runner._zero_all_caches()
            dist.barrier(group=runner.exchange_group)

            runner.mla_bt_gpu.zero_()
            runner.swa_bt_gpu.zero_()
            runner.comp_bt_gpu.fill_(runner.dummy_comp_block)
            runner.lookback_gpu.fill_(-1)
            runner.slot_mapping_gpu.fill_(-1)
            if runner._swa_pool is not None:
                runner.swa_slot_mapping_gpu.fill_(-1)
            runner.seq_lens_gpu.zero_()

            lang_m = getattr(runner.model, "language_model", runner.model)
            inner_m = getattr(lang_m, "model", lang_m)
            layers = list(getattr(inner_m, "layers", ()))

            _patch_attr(runner, "_build_step_attn_metadata", "dsv4.attn_meta_prep")
            eh = getattr(inner_m, "engram_hash", None)
            if eh is not None:
                _patch_attr(eh, "forward", "dsv4.engram_lookup")
            for lyr in layers:
                if getattr(lyr, "engram", None) is not None:
                    _patch_attr(lyr.engram, "forward", "dsv4.engram_lookup")
                if getattr(lyr, "attn", None) is not None:
                    _patch_attr(lyr.attn, "forward", "dsv4.mla_dsa_attn")
                ffn_mod = getattr(lyr, "ffn", None) or getattr(lyr, "mlp", None)
                if ffn_mod is not None:
                    _patch_attr(ffn_mod, "forward", "dsv4.b12x_moe_ffn")

            seen_gdicts: dict[int, dict[str, Any]] = {}
            for mod_obj in (inner_m, *layers):
                fwd = getattr(type(mod_obj), "forward", None)
                gd = getattr(fwd, "__globals__", None)
                if isinstance(gd, dict):
                    seen_gdicts[id(gd)] = gd
            for gd in seen_gdicts.values():
                _patch_global(gd, "mhc_shifted_post_pre", "dsv4.mhc_allreduce")
                _patch_global(gd, "mhc_post_tilelang", "dsv4.mhc_allreduce")

            _patch_attr(runner.model, "compute_logits_local", "dsv4.lm_head_logits")
            if getattr(runner, "argmax_channel", None) is not None:
                _patch_attr(
                    runner.argmax_channel,
                    "gather_top2_from_logits",
                    "dsv4.shm_top2_argmax",
                )

            graph_pool = getattr(runner, "_evolith_graph_pool", None) or torch.cuda.graph_pool_handle()
            capture_stream = torch.cuda.Stream(device=runner.device)
            active_capture_stream_handle[0] = int(capture_stream.cuda_stream)

            runner.token_ids_gpu[:bucket].zero_()
            runner.positions_gpu[:bucket].zero_()
            runner.seq_lens_gpu[:bucket].fill_(1)
            runner.seq_lens_gpu[bucket:].zero_()
            runner.slot_mapping_gpu[:bucket].fill_(-1)
            sl_cpu = torch.ones(bucket, dtype=torch.int32)

            with torch.inference_mode():
                with torch.cuda.stream(capture_stream):
                    for w_idx in range(2):
                        capture_meta_flag[0] = w_idx == 0
                        record_ext_flag[0] = False
                        attn_meta = runner._build_step_attn_metadata(
                            num_reqs=bucket,
                            num_tokens=bucket,
                            max_query_len=1,
                            max_seq_len=runner.decode_graph_max_seq_len,
                            qsl_gpu=runner.decode_qsl_gpu[: bucket + 1],
                            qsl_cpu=runner.decode_qsl_cpu[: bucket + 1],
                            seq_lens_gpu=runner.seq_lens_gpu[:bucket],
                            seq_lens_cpu=sl_cpu,
                            positions_gpu=runner.positions_gpu[:bucket],
                            slot_mapping_gpu=runner.slot_mapping_gpu[:bucket],
                            swa_slot_mapping_gpu=runner.swa_slot_mapping_gpu[:bucket],
                            is_prefilling_cpu=runner.decode_is_prefilling_cpu[:bucket],
                            for_capture=(w_idx == 0),
                        )
                        with set_forward_context(
                            attn_meta,
                            runner.vllm_config,
                            num_tokens=bucket,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        ):
                            if runner.pp_size > 1 and not runner.is_first_pp:
                                from vllm.sequence import IntermediateTensors

                                d_inter = IntermediateTensors(
                                    {
                                        "hidden_states": runner._pp_inter_hidden[:bucket],
                                        "pre_mix": runner._pp_inter_pre_mix[:bucket],
                                    }
                                )
                                hidden = runner.model(
                                    input_ids=runner.token_ids_gpu[:bucket],
                                    positions=runner.positions_gpu[:bucket],
                                    intermediate_tensors=d_inter,
                                    lookback_token_ids=runner.lookback_gpu,
                                )
                            else:
                                hidden = runner.model(
                                    input_ids=runner.token_ids_gpu[:bucket],
                                    positions=runner.positions_gpu[:bucket],
                                    lookback_token_ids=runner.lookback_gpu,
                                )
                            if runner.is_last_pp:
                                runner._sample_rows_into_argmax(hidden, None, bucket, all_ranks=True)
                            elif (
                                runner._pp_inter_hidden is not None
                                and runner._pp_inter_pre_mix is not None
                            ):
                                runner._pp_inter_hidden[:bucket].copy_(hidden["hidden_states"])
                                runner._pp_inter_pre_mix[:bucket].copy_(hidden["pre_mix"])
                capture_meta_flag[0] = False
                capture_stream.synchronize()
                dist.barrier(group=runner.exchange_group)

                with torch.cuda.stream(capture_stream):
                    attn_meta = runner._build_step_attn_metadata(
                        num_reqs=bucket,
                        num_tokens=bucket,
                        max_query_len=1,
                        max_seq_len=runner.decode_graph_max_seq_len,
                        qsl_gpu=runner.decode_qsl_gpu[: bucket + 1],
                        qsl_cpu=runner.decode_qsl_cpu[: bucket + 1],
                        seq_lens_gpu=runner.seq_lens_gpu[:bucket],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=runner.positions_gpu[:bucket],
                        slot_mapping_gpu=runner.slot_mapping_gpu[:bucket],
                        swa_slot_mapping_gpu=runner.swa_slot_mapping_gpu[:bucket],
                        is_prefilling_cpu=runner.decode_is_prefilling_cpu[:bucket],
                        for_capture=True,
                    )
                capture_stream.synchronize()

                prof_g = torch.cuda.CUDAGraph()
                ws_ev = _new_ext_event()
                we_ev = _new_ext_event()
                whole_ext_pair.append((ws_ev, we_ev))
                with set_forward_context(
                    attn_meta,
                    runner.vllm_config,
                    num_tokens=bucket,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ):
                    with torch.cuda.graph(prof_g, pool=graph_pool, stream=capture_stream):
                        _rec_ext_event(ws_ev, active_capture_stream_handle[0])
                        record_ext_flag[0] = True
                        runner._build_step_attn_metadata(
                            num_reqs=bucket,
                            num_tokens=bucket,
                            max_query_len=1,
                            max_seq_len=runner.decode_graph_max_seq_len,
                            qsl_gpu=runner.decode_qsl_gpu[: bucket + 1],
                            qsl_cpu=runner.decode_qsl_cpu[: bucket + 1],
                            seq_lens_gpu=runner.seq_lens_gpu[:bucket],
                            seq_lens_cpu=sl_cpu,
                            positions_gpu=runner.positions_gpu[:bucket],
                            slot_mapping_gpu=runner.slot_mapping_gpu[:bucket],
                            swa_slot_mapping_gpu=runner.swa_slot_mapping_gpu[:bucket],
                            is_prefilling_cpu=runner.decode_is_prefilling_cpu[:bucket],
                            for_capture=False,
                        )
                        if runner.pp_size > 1 and not runner.is_first_pp:
                            from vllm.sequence import IntermediateTensors

                            d_inter = IntermediateTensors(
                                {
                                    "hidden_states": runner._pp_inter_hidden[:bucket],
                                    "pre_mix": runner._pp_inter_pre_mix[:bucket],
                                }
                            )
                            hidden = runner.model(
                                input_ids=runner.token_ids_gpu[:bucket],
                                positions=runner.positions_gpu[:bucket],
                                intermediate_tensors=d_inter,
                                lookback_token_ids=runner.lookback_gpu,
                            )
                        else:
                            hidden = runner.model(
                                input_ids=runner.token_ids_gpu[:bucket],
                                positions=runner.positions_gpu[:bucket],
                                lookback_token_ids=runner.lookback_gpu,
                            )
                        if runner.is_last_pp:
                            runner._sample_rows_into_argmax(hidden, None, bucket, all_ranks=True)
                        elif (
                            runner._pp_inter_hidden is not None
                            and runner._pp_inter_pre_mix is not None
                        ):
                            runner._pp_inter_hidden[:bucket].copy_(hidden["hidden_states"])
                            runner._pp_inter_pre_mix[:bucket].copy_(hidden["pre_mix"])
                        record_ext_flag[0] = False
                        _rec_ext_event(we_ev, active_capture_stream_handle[0])
                capture_stream.synchronize()
                runner._zero_all_caches()
                dist.barrier(group=runner.exchange_group)
        finally:
            record_ext_flag[0] = False
            capture_meta_flag[0] = False
            for gdict, gkey, gorig in reversed(saved_globals):
                gdict[gkey] = gorig
            for obj, attr, had_inst, orig_val in reversed(saved_attrs):
                if had_inst:
                    setattr(obj, attr, orig_val)
                else:
                    try:
                        delattr(obj, attr)
                    except AttributeError:
                        setattr(obj, attr, orig_val)
            for obj, attr, val in post_pw_hooks:
                setattr(obj, attr, val)
            lock_workspace()

        runner.decode_graphs[bucket] = prof_g
        self._prof_state = {
            "mock": False,
            "bucket": bucket,
            "cudart": cudart,
            "prof_g": prof_g,
            "created_raw_events": created_raw_events,
            "captured_ext_pairs": captured_ext_pairs,
            "whole_ext_pair": whole_ext_pair[0],
            "stage_meta": stage_meta,
            "per_step_whole_us": [],
            "per_stage_step_us": {},
        }

    def exec_profiled_step_rank(self, cmd: dict[str, Any]) -> None:
        st = getattr(self, "_prof_state", None)
        if st is None:
            raise RuntimeError("Profile mode has not been initialized on worker rank")
        bucket = int(cmd["bucket"])
        num_seqs = int(cmd.get("num_seqs", bucket))
        sid = int(cmd["step_id"])
        dpos = int(cmd["decode_pos"])
        if st.get("mock"):
            self.execute_published_step(
                step_id=sid,
                is_decode=True,
                bucket=bucket,
                num_seqs=num_seqs,
                decode_pos=dpos,
            )
            return

        import torch

        rec = bool(cmd.get("record_events", False))
        with torch.inference_mode():
            self.execute_published_step(
                step_id=sid,
                is_decode=True,
                bucket=bucket,
                num_seqs=num_seqs,
                decode_pos=dpos,
            )
        if rec:
            torch.cuda.synchronize(self.runner.device)
            cudart: ctypes.CDLL = st["cudart"]
            ms_val = ctypes.c_float(0.0)
            ws_ev, we_ev = st["whole_ext_pair"]
            if cudart.cudaEventElapsedTime(ctypes.byref(ms_val), ws_ev, we_ev) == 0:
                st["per_step_whole_us"].append(float(ms_val.value) * 1000.0)
            for sk, ev_pairs in st["captured_ext_pairs"].items():
                tot_us = 0.0
                for es, ee in ev_pairs:
                    if cudart.cudaEventElapsedTime(ctypes.byref(ms_val), es, ee) == 0:
                        tot_us += float(ms_val.value) * 1000.0
                st["per_stage_step_us"].setdefault(sk, []).append(tot_us)

    def _end_profile_cleanup_if_needed(self) -> dict[str, Any] | None:
        st = getattr(self, "_prof_state", None)
        if st is None:
            return None
        self._prof_state = None
        bucket = int(st["bucket"])
        if st.get("mock"):
            return self.runner.profile_stages(bucket)

        import torch

        runner = self.runner
        cudart: ctypes.CDLL = st["cudart"]
        runner.decode_graphs.pop(bucket, None)
        del st["prof_g"]
        torch.cuda.synchronize(runner.device)
        for ev in st["created_raw_events"]:
            cudart.cudaEventDestroy(ev)

        raw_graphs = recapture_dsv41_decode_buckets(runner, self.cfg.active_decode_buckets)
        self._bind_and_wrap_graphs(raw_graphs)

        stages_out: dict[str, Any] = {}
        stage_meta = st["stage_meta"]
        ext_pairs = st["captured_ext_pairs"]
        for sk, vals in st["per_stage_step_us"].items():
            meta = stage_meta.get(sk, {})
            stages_out[sk] = {
                "step_us": round(float(statistics.median(vals)), 2),
                "calls": len(ext_pairs.get(sk, ())),
                "shapes": meta.get("shapes", [[bucket]]),
                "strides": meta.get("strides", []),
                "dtype": meta.get("dtype", "bfloat16"),
            }
        per_step_whole_us = st["per_step_whole_us"]
        med_whole = (
            round(float(statistics.median(per_step_whole_us)), 2)
            if per_step_whole_us
            else round(sum(float(v["step_us"]) for v in stages_out.values()), 2)
        )
        return {
            "rank": self.rank,
            "instrumented_step_us": med_whole,
            "stages": stages_out,
        }

    def end_profile_mode_rank(self) -> dict[str, Any]:
        out = self._end_profile_cleanup_if_needed()
        if out is None:
            raise RuntimeError("Profile mode was not active")
        return out

    def close(self, *, abort: bool = False) -> None:
        try:
            self.runner.close(abort=abort)
        finally:
            if self._bridge is not None:
                self._bridge.close()


def _worker_process_main(
    rank: int,
    cfg_dict: dict[str, Any],
    dist_init_file: str,
    shm_dir_str: str,
    cmd_q: Any,
    resp_q: Any,
) -> None:
    cfg = WorkbenchDaemonConfig(**cfg_dict)
    ctx: _RankResidentContext | None = None
    try:
        ctx = _RankResidentContext(cfg, rank, dist_init_file, Path(shm_dir_str))
        resp_q.put({"ok": True, "event": "ready", "status": ctx.status_payload()})
        while True:
            cmd = cmd_q.get()
            op = cmd.get("op")
            if op == "shutdown":
                ctx.close(abort=False)
                ctx = None
                resp_q.put({"ok": True, "rank": rank})
                return
            try:
                if op == "status":
                    resp_q.put({"ok": True, "data": ctx.status_payload()})
                elif op == "zero_caches":
                    ctx.zero_all_caches()
                    resp_q.put({"ok": True, "rank": rank})
                elif op == "exec_step":
                    outs, exec_us = ctx.execute_published_step(
                        step_id=int(cmd["step_id"]),
                        is_decode=bool(cmd["is_decode"]),
                        bucket=int(cmd["bucket"]),
                        num_seqs=int(cmd["num_seqs"]),
                        decode_pos=int(cmd["decode_pos"]),
                    )
                    resp_q.put(
                        {
                            "ok": True,
                            "data": {"rank": rank, "outputs": outs, "gpu_exec_us": exec_us},
                        }
                    )
                elif op == "inner_screen":
                    resp_q.put({"ok": True, "data": ctx.run_inner_screen_rank(cmd)})
                elif op == "apply_candidate":
                    resp_q.put({"ok": True, "data": ctx.apply_candidate_rank(cmd)})
                elif op == "rollback_last":
                    ctx.rollback_last_committed_rank(cmd["recapture_buckets"])
                    resp_q.put({"ok": True, "rank": rank})
                elif op == "sleep":
                    resp_q.put({"ok": True, "data": ctx.sleep_rank()})
                elif op == "wake":
                    resp_q.put({"ok": True, "data": ctx.wake_rank(cmd["recapture_buckets"])})
                elif op == "reattach_shm":
                    resp_q.put({"ok": True, "data": ctx.reattach_shm_rank(cmd["shm_name"])})
                elif op == "begin_profile_mode":
                    ctx.begin_profile_mode_rank(int(cmd["bucket"]))
                    resp_q.put({"ok": True, "rank": rank})
                elif op == "exec_profiled_step":
                    ctx.exec_profiled_step_rank(cmd)
                    resp_q.put({"ok": True, "rank": rank})
                elif op == "end_profile_mode":
                    resp_q.put({"ok": True, "data": ctx.end_profile_mode_rank()})
                elif op == "simulate_hang":
                    time.sleep(float(cmd.get("seconds", 5.0)))
                    resp_q.put({"ok": True, "rank": rank})
                else:
                    resp_q.put({"ok": False, "error": f"Unknown worker op `{op}`"})
            except Exception as exc:
                resp_q.put(
                    {
                        "ok": False,
                        "rank": rank,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1500:],
                    }
                )
    except BaseException as exc:
        resp_q.put(
            {
                "ok": False,
                "event": "init_failed",
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
        )
        if ctx is not None:
            try:
                ctx.close(abort=True)
            except Exception:
                pass


def _load_reference_token_streams(worker_root: str) -> list[np.ndarray]:
    """Load real BPE prompt+greedy token streams from zstep's golden reference JSON if present."""
    if not worker_root:
        return []
    ref_path = Path(worker_root).resolve().parent / "docs/benchmarks/phase3_vllm_greedy_reference.json"
    if not ref_path.is_file():
        return []
    try:
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
        cases = payload.get("cases", [])
        # Put high-confidence reasoning/factual cases first (`prime_squares_table`, `dijkstra_shortest_path`, ...)
        ordered = sorted(
            cases,
            key=lambda c: (0 if c.get("name") == "prime_squares_table" else 1, -len(c.get("prompt_token_ids", []))),
        )
        full_corpus: list[int] = []
        raw_streams: list[list[int]] = []
        for c in ordered:
            seq = [int(t) for t in c.get("prompt_token_ids", [])] + [
                int(t) for t in c.get("expected_greedy_token_ids", [])
            ]
            if seq:
                raw_streams.append(seq)
                full_corpus.extend(seq)
        if not raw_streams or not full_corpus:
            return []
        out: list[np.ndarray] = []
        for seq in raw_streams:
            extended = list(seq)
            while len(extended) < 2048:
                extended.extend(full_corpus)
            out.append(np.array(extended[:2048], dtype=np.int32))
        return out
    except Exception:
        return []


class WorkbenchGpuDaemon:
    """Coordinator managing `world_size` resident `StateKeeper` GPU ranks & JSON-RPC."""

    def __init__(self, cfg: WorkbenchDaemonConfig, *, shm_dir: Path | None = None) -> None:
        self.cfg = cfg.model_copy(
            update={
                "shm_name": cfg.shm_name or f"evo_wb_{os.getpid()}",
            }
        )
        self.shm_dir = shm_dir or (
            Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
        )
        self._token_streams = _load_reference_token_streams(self.cfg.worker_root)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="evolith_wb_daemon_")
        self._dist_init_file = str(Path(self._tmp_dir.name) / "dist_init")
        self._native_loader = ExternCOpLoader(Path(self._tmp_dir.name) / "native_cache")
        self._publisher = PosixShmStepPublisher(
            self.cfg.shm_name,
            WireGeometry(world_size=self.cfg.world_size),
            shm_dir=self.shm_dir,
            token_streams=self._token_streams,
        )
        self._step_counter = 0
        self._mp_ctx = mp.get_context("spawn")
        self._cmd_queues: list[Any] = []
        self._resp_queues: list[Any] = []
        self._procs: list[mp.Process] = []
        self._started = False
        self._killed_by_watchdog = False

    def start(self) -> dict[str, Any]:
        if self._started:
            return self.status()
        t0 = time.perf_counter()
        cfg_dict = self.cfg.model_dump(mode="json")
        for r in range(self.cfg.world_size):
            cq = self._mp_ctx.Queue()
            rq = self._mp_ctx.Queue()
            proc = self._mp_ctx.Process(
                target=_worker_process_main,
                args=(r, cfg_dict, self._dist_init_file, str(self.shm_dir), cq, rq),
                name=f"evolith-statekeeper-r{r}",
                daemon=False,
            )
            proc.start()
            self._cmd_queues.append(cq)
            self._resp_queues.append(rq)
            self._procs.append(proc)

        statuses: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.cfg.startup_timeout_sec
        for r, rq in enumerate(self._resp_queues):
            rem = max(0.1, deadline - time.monotonic())
            try:
                msg = rq.get(timeout=rem)
            except Exception as exc:
                self.force_kill_all()
                raise RuntimeError(
                    f"Rank {r} timed out during StateKeeper initialization after "
                    f"{self.cfg.startup_timeout_sec}s"
                ) from exc
            if not msg.get("ok"):
                self.force_kill_all()
                raise RuntimeError(f"Rank {r} failed startup: {msg.get('error')}\n{msg.get('traceback', '')}")
            statuses.append(msg["status"])

        self._started = True
        boot_s = round(time.perf_counter() - t0, 3)
        return {"ok": True, "boot_seconds": boot_s, "ranks": statuses}

    def force_kill_all(self) -> None:
        self._killed_by_watchdog = True
        for p in self._procs:
            if p.is_alive():
                p.kill()
        for p in self._procs:
            p.join(timeout=3.0)
        self._publisher.close(unlink=True)
        self._tmp_dir.cleanup()

    def _broadcast_and_collect(
        self, payload: dict[str, Any], *, timeout_sec: float | None = None
    ) -> list[dict[str, Any]]:
        if self._killed_by_watchdog:
            raise RuntimeError("Daemon worker group was terminated by watchdog")
        timeout = timeout_sec if timeout_sec is not None else self.cfg.rpc_watchdog_timeout_sec
        for cq in self._cmd_queues:
            cq.put(payload)
        return self._collect_only(timeout_sec=timeout)

    def status(self) -> dict[str, Any]:
        ranks = self._broadcast_and_collect({"op": "status"})
        total_used_mib = sum(int(r["vram"]["used_mib"]) for r in ranks)
        return {
            "ok": True,
            "world_size": self.cfg.world_size,
            "runner_kind": self.cfg.runner_kind,
            "shm_name": self.cfg.shm_name,
            "total_used_vram_mib": total_used_mib,
            "sleeping": bool(ranks[0]["sleeping"]),
            "active_version": int(ranks[0]["active_version"]),
            "scheduler_attach_epoch": int(ranks[0]["scheduler_attach_epoch"]),
            "ranks": ranks,
        }

    def _prepare_screen_plan_schedule(
        self,
        *,
        buckets: Sequence[int],
        context_len: int,
        warmup_steps: int,
        timed_steps: int,
    ) -> tuple[list[int], dict[str, list[tuple[int, int]]]]:
        max_b = max(int(b) for b in buckets)
        pages_per_seq = max(1, math.ceil((context_len + warmup_steps + timed_steps + 4) / 128))
        self._publisher.assign_linear_page_tables(
            total_seqs=max_b, pages_per_seq=pages_per_seq
        )
        # Note: PosixShmStepPublisher has 2 double-buffered slots (`step_id & 1`),
        # so in live multi-step execution we publish step-by-step OR when lockstep
        # ranks execute sequentially we coordinate per step!
        return [], {}

    def inner_screen(
        self,
        *,
        buckets: Sequence[int] | None = None,
        context_len: int = 512,
        warmup_steps: int = 4,
        timed_steps: int = 20,
        prompt_seed: int = 101,
    ) -> dict[str, Any]:
        """Run `< 5s` resident inner screen across all 8 GPUs using POSIX SHM plans."""
        t0 = time.perf_counter()
        target_buckets = [int(b) for b in (buckets or self.cfg.active_decode_buckets)]
        max_b = max(target_buckets)
        pages_per_seq = max(1, math.ceil((context_len + warmup_steps + timed_steps + 8) / 128))
        self._publisher.assign_linear_page_tables(
            total_seqs=max_b, pages_per_seq=pages_per_seq
        )
        self._broadcast_and_collect({"op": "zero_caches"})
        status_before = self.status()["ranks"]
        before_replays = int(status_before[0]["graph_telemetry"]["graph_replay_steps"])
        before_recomp = sum(int(r.get("triton_recompilations", 0)) for r in status_before)

        # Publish & execute 1-seq prefill steps matching zstep's `num_reqs=1` piecewise graphs
        seqs_per_chunk = 1
        for start_s in range(0, max_b, seqs_per_chunk):
            chunk_slots = list(range(start_s, min(max_b, start_s + seqs_per_chunk)))
            self._step_counter += 1
            self._publisher.publish_prefill_step(
                step_id=self._step_counter,
                slot_ids=chunk_slots,
                context_len=context_len,
                prompt_seed=prompt_seed,
            )
            self._step_all_ranks(
                step_id=self._step_counter,
                is_decode=False,
                bucket=0,
                num_seqs=len(chunk_slots),
                decode_pos=0,
            )

        bucket_metrics: list[dict[str, Any]] = []
        for b in target_buckets:
            for w in range(warmup_steps):
                self._step_counter += 1
                dpos = context_len + w
                self._publisher.publish_decode_step(
                    step_id=self._step_counter,
                    num_seqs=b,
                    decode_pos=dpos,
                    decode_graph_bucket=b,
                )
                self._step_all_ranks(
                    step_id=self._step_counter,
                    is_decode=True,
                    bucket=b,
                    num_seqs=b,
                    decode_pos=dpos,
                )

            st_pre = self.status()["ranks"][0]["graph_telemetry"]
            rep_start = int(st_pre["graph_replay_steps"])
            step_us_list: list[int] = []
            sample_tokens: list[int] = []
            sample_logprobs: list[float] = []
            runner_up_tokens: list[int] = []
            runner_up_logprobs: list[float] = []
            for idx in range(timed_steps):
                self._step_counter += 1
                dpos = context_len + warmup_steps + idx
                self._publisher.publish_decode_step(
                    step_id=self._step_counter,
                    num_seqs=b,
                    decode_pos=dpos,
                    decode_graph_bucket=b,
                )
                r_outs = self._step_all_ranks(
                    step_id=self._step_counter,
                    is_decode=True,
                    bucket=b,
                    num_seqs=b,
                    decode_pos=dpos,
                )
                # Use max GPU step time across ranks (true synchronized step duration)
                max_us = max(int(r["gpu_exec_us"]) for r in r_outs)
                step_us_list.append(max_us)
                if idx == 0:
                    for r in r_outs:
                        if r["outputs"]:
                            sample_tokens = [int(x[1]) for x in r["outputs"][:8]]
                            sample_logprobs = [float(x[2]) for x in r["outputs"][:8]]
                            runner_up_tokens = [int(x[3]) for x in r["outputs"][:8]]
                            runner_up_logprobs = [float(x[4]) for x in r["outputs"][:8]]
                            break

            st_post = self.status()["ranks"][0]["graph_telemetry"]
            rep_end = int(st_post["graph_replay_steps"])
            if rep_end - rep_start != timed_steps:
                raise RuntimeError(
                    f"TRACE_CUDA_GRAPH_REQUIRED: bucket={b} expected {timed_steps} "
                    f"guarded CUDAGraph replays, observed {rep_end - rep_start}"
                )

            mean_us = float(statistics.fmean(step_us_list))
            median_us = float(statistics.median(step_us_list))
            s_us = sorted(step_us_list)
            p99_us = float(s_us[min(len(s_us) - 1, max(0, math.ceil(0.99 * len(s_us)) - 1))])
            bucket_metrics.append(
                {
                    "bucket": b,
                    "steps": timed_steps,
                    "mean_step_us": round(mean_us, 2),
                    "median_step_us": round(median_us, 2),
                    "p99_step_us": round(p99_us, 2),
                    "throughput_tok_s": round((b * 1_000_000.0) / max(mean_us, 1e-6), 2),
                    "sample_tokens": sample_tokens,
                    "sample_logprobs": sample_logprobs,
                    "runner_up_tokens": runner_up_tokens,
                    "runner_up_logprobs": runner_up_logprobs,
                }
            )

        ranks_final = self.status()["ranks"]
        st_final = ranks_final[0]["graph_telemetry"]
        after_recomp = sum(int(r.get("triton_recompilations", 0)) for r in ranks_final)
        delta_recomp = max(0, after_recomp - before_recomp)
        if delta_recomp != 0:
            raise RuntimeError(
                f"TRACE_ZERO_TRITON_RECOMPILATIONS violated during inner_screen: {delta_recomp}"
            )
        geomean_tok_s = math.exp(
            statistics.fmean(math.log(max(m["throughput_tok_s"], 1e-6)) for m in bucket_metrics)
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "context_len": context_len,
            "geomean_tok_s": round(geomean_tok_s, 2),
            "eager_fallback_steps": int(st_final["eager_fallback_steps"]),
            "graph_guard_failures": int(st_final["graph_guard_failures"]),
            "graph_replay_steps": int(st_final["graph_replay_steps"]) - before_replays,
            "triton_recompilations": delta_recomp,
            "buckets": bucket_metrics,
        }

    def _collect_only(self, timeout_sec: float | None = None) -> list[dict[str, Any]]:
        timeout = timeout_sec if timeout_sec is not None else self.cfg.rpc_watchdog_timeout_sec
        deadline = time.monotonic() + timeout
        out: list[dict[str, Any]] = []
        first_error: str | None = None
        for r, rq in enumerate(self._resp_queues):
            rem = max(0.02, deadline - time.monotonic())
            try:
                msg = rq.get(timeout=rem)
            except Exception as exc:
                self.force_kill_all()
                raise TimeoutError(
                    f"Hard multi-rank watchdog tripped on rank {r} after {timeout}s; "
                    "SIGKILLed worker group cleanly"
                ) from exc
            if not msg.get("ok"):
                if first_error is None:
                    first_error = f"Rank {r} error: {msg.get('error')}"
            else:
                out.append(msg.get("data", msg))
        if first_error is not None:
            raise RuntimeError(first_error)
        return out

    def _step_all_ranks(
        self,
        *,
        step_id: int,
        is_decode: bool,
        bucket: int,
        num_seqs: int,
        decode_pos: int,
    ) -> list[dict[str, Any]]:
        return self._broadcast_and_collect(
            {
                "op": "exec_step",
                "step_id": step_id,
                "is_decode": is_decode,
                "bucket": bucket,
                "num_seqs": num_seqs,
                "decode_pos": decode_pos,
            }
        )

    def hot_apply(
        self,
        *,
        candidate_id: str = "wb-cand",
        changed_files: Sequence[str] = (),
        changed_config_keys: Sequence[str] = (),
        patch_text: str = "",
        hook_code: str = "",
        native_op: dict[str, Any] | None = None,
        recapture_buckets: Sequence[int] | None = None,
        refit_weights: bool = False,
        shadow_check: bool = True,
        shadow_bucket: int = 16,
        shadow_logprob_atol: float = 2.0,
        shadow_min_token_match: float = 0.875,
        watchdog_timeout_sec: float = 20.0,
    ) -> dict[str, Any]:
        """2PC transactional `hot_apply` across all 8 resident GPU ranks + `--shadow` check.

        When `native_op` is provided (`cuda_source` or `c_source`), compiles ONCE into a
        content-addressed `libevo_<op>_<sha12>.so` and runs mandatory out-of-process
        preflight BEFORE touching any of the 8 resident 475 GiB `StateKeeper` ranks.
        """
        t0 = time.perf_counter()
        target_buckets = [int(b) for b in (recapture_buckets or self.cfg.active_decode_buckets)]
        probe_b = shadow_bucket if shadow_bucket in target_buckets else target_buckets[0]

        compiled_native_payload: dict[str, Any] | None = None
        native_compile_ms = 0.0
        native_preflight_ms = 0.0
        if native_op is not None:
            op_name = str(native_op["op_name"])
            symbol_name = str(native_op["symbol_name"])
            smoke_expr = str(native_op.get("smoke_expr", ""))
            preflight_timeout = float(native_op.get("preflight_timeout_sec", 15.0))
            try:
                tc0 = time.perf_counter()
                if "cuda_source" in native_op and native_op["cuda_source"]:
                    extra_nvcc = tuple(str(f) for f in native_op.get("extra_flags", ("-O3",)))
                    comp = self._native_loader.compile_cuda_source(
                        op_name,
                        str(native_op["cuda_source"]),
                        symbol_name=symbol_name,
                        arch=native_op.get("arch"),
                        extra_nvcc_flags=extra_nvcc,
                    )
                    cuda_dev = str(native_op.get("preflight_cuda_device", "0"))
                elif "c_source" in native_op and native_op["c_source"]:
                    extra_c = tuple(str(f) for f in native_op.get("extra_flags", ("-O2",)))
                    comp = self._native_loader.compile_c_source(
                        op_name,
                        str(native_op["c_source"]),
                        symbol_name=symbol_name,
                        extra_cflags=extra_c,
                    )
                    cuda_dev = None
                else:
                    raise NativeOpPreflightError(
                        "`native_op` must supply non-empty `cuda_source` or `c_source`"
                    )
                native_compile_ms = round((time.perf_counter() - tc0) * 1000.0, 2)

                tp0 = time.perf_counter()
                self._native_loader.preflight_in_subprocess(
                    comp,
                    smoke_expr=smoke_expr,
                    timeout_sec=preflight_timeout,
                    cuda_device=cuda_dev,
                )
                native_preflight_ms = round((time.perf_counter() - tp0) * 1000.0, 2)
                compiled_native_payload = {
                    "op_name": comp.op_name,
                    "symbol_name": comp.symbol_name,
                    "content_hash": comp.content_hash,
                    "so_path": str(comp.so_path),
                    "backend": comp.backend,
                    "argtypes": list(native_op.get("argtypes", ())),
                    "restype": native_op.get("restype", "i32"),
                }
            except NativeOpPreflightError as exc:
                return {
                    "ok": False,
                    "committed": False,
                    "rolled_back": False,
                    "failure_kind": "native_preflight_rejected",
                    "failure_detail": str(exc),
                    "native_compile_ms": native_compile_ms,
                    "native_preflight_ms": native_preflight_ms,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        inc_tokens: list[int] = []
        inc_lps: list[float] = []
        inc_ru_tokens: list[int] = []
        inc_ru_lps: list[float] = []
        if shadow_check:
            self._broadcast_and_collect({"op": "zero_caches"})
            inc_probe = self.inner_screen(
                buckets=[probe_b], context_len=512, warmup_steps=4, timed_steps=1, prompt_seed=777
            )
            b0 = inc_probe["buckets"][0]
            inc_tokens = b0["sample_tokens"]
            inc_lps = b0["sample_logprobs"]
            inc_ru_tokens = b0.get("runner_up_tokens", [])
            inc_ru_lps = b0.get("runner_up_logprobs", [])

        try:
            rank_results = self._broadcast_and_collect(
                {
                    "op": "apply_candidate",
                    "candidate_id": candidate_id,
                    "changed_files": list(changed_files),
                    "changed_config_keys": list(changed_config_keys),
                    "patch_text": patch_text,
                    "hook_code": hook_code,
                    "compiled_native_op": compiled_native_payload,
                    "recapture_buckets": target_buckets,
                    "refit_weights": refit_weights,
                    "watchdog_timeout_sec": watchdog_timeout_sec,
                },
                timeout_sec=watchdog_timeout_sec + 10.0,
            )
        except TimeoutError as exc:
            return {
                "ok": False,
                "committed": False,
                "rolled_back": False,
                "killed_worker_group": True,
                "failure_kind": "timeout_watchdog",
                "failure_detail": str(exc),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        r0 = rank_results[0]
        if not r0.get("committed"):
            return {
                "ok": False,
                **r0,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        if shadow_check and inc_tokens:
            self._broadcast_and_collect({"op": "zero_caches"})
            cand_probe = self.inner_screen(
                buckets=[probe_b], context_len=512, warmup_steps=4, timed_steps=1, prompt_seed=777
            )
            cb0 = cand_probe["buckets"][0]
            cand_tokens = cb0["sample_tokens"]
            cand_lps = cb0["sample_logprobs"]
            cand_ru_tokens = cb0.get("runner_up_tokens", [])
            cand_ru_lps = cb0.get("runner_up_logprobs", [])
            outlier_atol = max(6.0, shadow_logprob_atol * 3.0)
            matches = top2_margin_consistent_rate(
                inc_tokens,
                inc_lps,
                inc_ru_tokens,
                inc_ru_lps,
                cand_tokens,
                cand_lps,
                cand_ru_tokens,
                cand_ru_lps,
                logprob_atol=outlier_atol,
            )
            lp_diffs = [abs(a - b) for a, b in zip(inc_lps, cand_lps, strict=True)]
            median_lp_diff = float(statistics.median(lp_diffs)) if lp_diffs else 0.0
            max_lp_diff = max(lp_diffs, default=0.0)
            if (
                matches < shadow_min_token_match
                or median_lp_diff > shadow_logprob_atol
                or max_lp_diff > outlier_atol
            ):
                self._broadcast_and_collect(
                    {"op": "rollback_last", "recapture_buckets": target_buckets}
                )
                return {
                    "ok": False,
                    "committed": False,
                    "rolled_back": True,
                    "active_version": r0["active_version"],
                    "failure_kind": "shadow_drift",
                    "failure_detail": (
                        f"Shadow differential check failed vs incumbent on bucket={probe_b}: "
                        f"token_match_rate={matches:.2%} (min={shadow_min_token_match:.2%}), "
                        f"median_logprob_diff={median_lp_diff:.4f} (max={shadow_logprob_atol:.4f}), "
                        f"max_logprob_diff={max_lp_diff:.4f} (max={outlier_atol:.4f})"
                    ),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        return {
            "ok": True,
            **r0,
            "native_so_path": compiled_native_payload["so_path"] if compiled_native_payload else "",
            "native_content_hash": compiled_native_payload["content_hash"] if compiled_native_payload else "",
            "native_compile_ms": native_compile_ms,
            "native_preflight_ms": native_preflight_ms,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def sleep(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        ranks = self._broadcast_and_collect({"op": "sleep"})
        total_freed = sum(int(r["freed_vram_mib"]) for r in ranks)
        return {
            "ok": True,
            "sleeping": True,
            "freed_vram_mib_total": total_freed,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "ranks": ranks,
        }

    def wake(self, *, recapture_buckets: Sequence[int] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        target_buckets = [int(b) for b in (recapture_buckets or self.cfg.active_decode_buckets)]
        ranks = self._broadcast_and_collect(
            {"op": "wake", "recapture_buckets": target_buckets}
        )
        total_restored = sum(int(r["restored_vram_mib"]) for r in ranks)
        return {
            "ok": True,
            "sleeping": False,
            "restored_vram_mib_total": total_restored,
            "kv_generation": int(ranks[0]["kv_generation"]),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "ranks": ranks,
        }

    def reattach_shm(self, new_shm_name: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        new_pub = PosixShmStepPublisher(
            new_shm_name,
            WireGeometry(world_size=self.cfg.world_size),
            shm_dir=self.shm_dir,
            token_streams=self._token_streams,
        )
        old_pub = self._publisher
        self._publisher = new_pub
        self.cfg = self.cfg.model_copy(update={"shm_name": new_shm_name})
        ranks = self._broadcast_and_collect(
            {"op": "reattach_shm", "shm_name": new_shm_name}
        )
        old_pub.close(unlink=True)
        return {
            "ok": True,
            "shm_name": new_shm_name,
            "scheduler_attach_epoch": int(ranks[0]["scheduler_attach_epoch"]),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def profile_step(
        self,
        *,
        bucket: int = 32,
        context_len: int = 512,
        warmup_steps: int = 3,
        timed_steps: int = 8,
        top_k_contracts: int = 4,
        blocked_niches: Sequence[str] = (),
        prompt_seed: int = 101,
    ) -> dict[str, Any]:
        """Measure resident step bottlenecks via CUDA Events & export `CallSiteContract`s.

        1. Measures ground-truth captured `CUDAGraph` step latency on `bucket`.
        2. Runs lockstep CUDA Event stage attribution across all `world_size` ranks
           using 1-step-at-a-time SHM parity publication (`step_id & 1` safe), then
           cleanly restores the resident `GuardedCudaGraphProxy` without recapture.
        3. Feeds measured non-overlapping stage shares + explicit `unattributed_step_us`
           into `StepBottleneckLedger` and extracts frozen `CallSiteContract`s.
        """
        t0 = time.perf_counter()
        target_b = int(bucket)
        if target_b not in self.cfg.active_decode_buckets:
            target_b = int(self.cfg.active_decode_buckets[-1])

        graph_scr = self.inner_screen(
            buckets=[target_b],
            context_len=context_len,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            prompt_seed=prompt_seed,
        )
        captured_graph_step_us = float(graph_scr["buckets"][0]["median_step_us"])

        pages_per_seq = max(1, math.ceil((context_len + warmup_steps + timed_steps + 8) / 128))
        self._publisher.assign_linear_page_tables(
            total_seqs=target_b, pages_per_seq=pages_per_seq
        )
        self._broadcast_and_collect({"op": "zero_caches"})
        self._broadcast_and_collect({"op": "begin_profile_mode", "bucket": target_b})
        rank_profiles: list[dict[str, Any]] = []
        try:
            for s_idx in range(target_b):
                self._step_counter += 1
                self._publisher.publish_prefill_step(
                    step_id=self._step_counter,
                    slot_ids=[s_idx],
                    context_len=context_len,
                    prompt_seed=prompt_seed,
                )
                self._step_all_ranks(
                    step_id=self._step_counter,
                    is_decode=False,
                    bucket=0,
                    num_seqs=1,
                    decode_pos=0,
                )

            total_iters = warmup_steps + timed_steps
            for i in range(total_iters):
                self._step_counter += 1
                dpos = context_len + i
                self._publisher.publish_decode_step(
                    step_id=self._step_counter,
                    num_seqs=target_b,
                    decode_pos=dpos,
                    decode_graph_bucket=target_b,
                )
                self._broadcast_and_collect(
                    {
                        "op": "exec_profiled_step",
                        "step_id": self._step_counter,
                        "bucket": target_b,
                        "num_seqs": target_b,
                        "decode_pos": dpos,
                        "record_events": i >= warmup_steps,
                    }
                )
        finally:
            rank_profiles = self._broadcast_and_collect({"op": "end_profile_mode"})

        surface = default_zstep_surface()
        unit_by_id = {u.unit_id: u for u in surface.units}
        stage_unit_map: dict[str, tuple[str, str]] = {
            "dsv4.b12x_moe_ffn": (
                "kernels.b12x_sm120",
                "worker/zstep_worker/kernels_sm120/b12x_dsv41.py",
            ),
            "dsv4.mla_dsa_attn": (
                "kernels.b12x_sm120",
                "worker/zstep_worker/kernels_sm120/b12x_dsv41.py",
            ),
            "dsv4.mhc_allreduce": (
                "kernels.shm_allreduce_csrc",
                "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu",
            ),
            "dsv4.shm_top2_argmax": (
                "kernels.shm_allreduce_csrc",
                "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu",
            ),
            "dsv4.lm_head_logits": (
                "models.runner",
                "worker/zstep_worker/models/deepseek_v41.py",
            ),
            "dsv4.engram_lookup": (
                "models.runner",
                "worker/zstep_worker/models/deepseek_v41.py",
            ),
            "dsv4.attn_meta_prep": (
                "models.runner",
                "worker/zstep_worker/models/deepseek_v41.py",
            ),
        }

        all_stage_keys: list[str] = sorted(
            {sk for rp in rank_profiles for sk in rp.get("stages", {})}
        )
        rank_whole_list = [float(rp["instrumented_step_us"]) for rp in rank_profiles]
        mean_whole_us = round(float(statistics.fmean(rank_whole_list)), 2)
        max_rank_us = round(max(rank_whole_list, default=0.0), 2)

        samples: list[StepOperatorSample] = []
        attributed_us = 0.0
        for sk in all_stage_keys:
            vals = [float(rp["stages"].get(sk, {}).get("step_us", 0.0)) for rp in rank_profiles]
            mean_stage_us = round(float(statistics.fmean(vals)), 2)
            attributed_us += mean_stage_us
            r0_meta = next(
                (rp["stages"][sk] for rp in rank_profiles if sk in rp.get("stages", {})),
                {},
            )
            uid, rep_file = stage_unit_map.get(
                sk, ("models.runner", "worker/zstep_worker/models/deepseek_v41.py")
            )
            blast = analyze_changeset_blast_radius(surface, changed_files=(rep_file,))
            udecl = unit_by_id.get(uid)
            shapes_t = tuple(tuple(int(d) for d in shp) for shp in r0_meta.get("shapes", ()))
            strides_t = tuple(tuple(int(d) for d in st) for st in r0_meta.get("strides", ()))
            samples.append(
                StepOperatorSample(
                    op_name=sk,
                    plane=EvolutionPlane.L1_KERNEL,
                    calls_per_step=int(r0_meta.get("calls", 1)),
                    total_step_us=mean_stage_us,
                    supported=blast.can_hot_apply_in_workbench,
                    niche_key=f"(plane={EvolutionPlane.L1_KERNEL.value},op={sk})",
                    unit_id=uid,
                    action_rung=blast.rung_name,
                    hotswap_extern_c=bool(udecl.hotswap_extern_c) if udecl else False,
                    shapes=shapes_t,
                    strides=strides_t,
                    dtype=str(r0_meta.get("dtype", "bfloat16")),
                )
            )

        effective_whole_us = max(mean_whole_us, round(attributed_us, 2))
        unattributed_us = round(max(0.0, effective_whole_us - attributed_us), 2)
        if unattributed_us > 0.0:
            samples.append(
                StepOperatorSample(
                    op_name="step.unattributed_glue",
                    plane=EvolutionPlane.L1_KERNEL,
                    calls_per_step=1,
                    total_step_us=unattributed_us,
                    supported=False,
                    niche_key="(plane=l1_kernel,op=step.unattributed_glue)",
                    boundary_note=(
                        "Inter-stage stream glue, embedding, and final RMSNorm outside "
                        "instrumented module hooks (tracked explicitly as observation gap)"
                    ),
                )
            )

        blocked_set = set(str(x) for x in blocked_niches)
        ledger = StepBottleneckLedger(samples)
        supported_ranked, capability_gaps = ledger.rank_targets(blocked_niches=blocked_set)
        contracts = ledger.build_callsite_contracts(
            bucket=target_b,
            world_size=self.cfg.world_size,
            top_k=top_k_contracts,
            blocked_niches=blocked_set,
        )
        cov = (attributed_us / effective_whole_us) if effective_whole_us > 0.0 else 0.0
        report = StepProfileReport(
            ok=True,
            bucket=target_b,
            context_len=context_len,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            world_size=self.cfg.world_size,
            captured_graph_step_us=round(captured_graph_step_us, 2),
            instrumented_gpu_step_us=round(effective_whole_us, 2),
            max_rank_step_us=max_rank_us,
            attributed_stage_us=round(attributed_us, 2),
            unattributed_step_us=unattributed_us,
            coverage_fraction=round(min(1.0, max(0.0, cov)), 4),
            samples=samples,
            supported_ranked=supported_ranked,
            capability_gaps=capability_gaps,
            selected_target=supported_ranked[0] if supported_ranked else None,
            callsite_contracts=contracts,
            blocked_niches_excluded=sorted(blocked_set),
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
        return report.model_dump(mode="json")

    def shutdown(self) -> dict[str, Any]:
        if not self._started or self._killed_by_watchdog:
            self._publisher.close(unlink=True)
            self._tmp_dir.cleanup()
            return {"ok": True}
        try:
            for cq in self._cmd_queues:
                cq.put({"op": "shutdown"})
            for rq in self._resp_queues:
                rq.get(timeout=15.0)
            for p in self._procs:
                p.join(timeout=10.0)
        finally:
            self.force_kill_all()
        return {"ok": True}

    def dispatch_rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        method = str(request.get("method", ""))
        params = dict(request.get("params") or {})
        if method == "ping":
            return {"ok": True, "pong": True, "shm_name": self.cfg.shm_name}
        if method == "status":
            return self.status()
        if method == "inner_screen":
            return self.inner_screen(**params)
        if method == "profile_step":
            return self.profile_step(**params)
        if method == "hot_apply":
            return self.hot_apply(**params)
        if method == "sleep":
            return self.sleep()
        if method == "wake":
            return self.wake(**params)
        if method == "reattach_shm":
            return self.reattach_shm(**params)
        if method == "simulate_hang":
            self._broadcast_and_collect(
                {"op": "simulate_hang", "seconds": float(params.get("seconds", 5.0))},
                timeout_sec=float(params.get("watchdog_timeout_sec", 0.2)),
            )
            return {"ok": True}
        if method == "shutdown":
            return self.shutdown()
        return {"ok": False, "error": f"Unknown RPC method `{method}`"}

    def serve_forever(self) -> None:
        self.start()
        daemon_ref = self

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline()
                if not raw:
                    return
                try:
                    req = json.loads(raw.decode("utf-8"))
                    method = req.get("method")
                    resp = daemon_ref.dispatch_rpc(req)
                except Exception as exc:
                    method = None
                    resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()
                if method == "shutdown":
                    self.server.shutdown()

        class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        with _ThreadedServer((self.cfg.host, self.cfg.port), _Handler) as srv:
            self.cfg = self.cfg.model_copy(update={"port": int(srv.server_address[1])})
            srv.serve_forever()


class WorkbenchRpcClient:
    """Lightweight JSON-line TCP client for `WorkbenchGpuDaemon`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18765, timeout_sec: float = 120.0) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        payload = json.dumps({"method": method, "params": params}) + "\n"
        with socket.create_connection((self.host, self.port), timeout=self.timeout_sec) as sock:
            sock.sendall(payload.encode("utf-8"))
            buf = bytearray()
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in chunk:
                    break
        if not buf:
            raise RuntimeError(f"Empty RPC response for `{method}`")
        return json.loads(buf.decode("utf-8").splitlines()[0])
