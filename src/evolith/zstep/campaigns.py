from __future__ import annotations

from dataclasses import dataclass

from evolith.models import EvolutionPlane
from evolith.zstep.task import ZStepRSITask


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    aliases: tuple[str, ...]
    priority_rank: str
    title: str
    plane: EvolutionPlane
    expected_gain: str
    primary_niche_key: str
    allowed_roots: tuple[str, ...]
    complexity_free_lines: int
    max_changed_lines: int
    blueprint: str


CAMPAIGNS: tuple[CampaignSpec, ...] = (
    CampaignSpec(
        campaign_id="t1-1-device-decode",
        aliases=("t1-1", "device-decode"),
        priority_rank="T1-1",
        title="Device-Resident Decode + GPU Engram Lookback Window",
        plane=EvolutionPlane.L2_SCHEDULER,
        expected_gain="+5% ~ +12% TPOT & c1-c32 throughput",
        primary_niche_key="(plane=l2_scheduler,focus=dsv41_device_token_overlap)",
        allowed_roots=(
            "worker/zstep_worker/models/deepseek_v41.py",
            "worker/zstep_worker/kernels_sm120/shm_allreduce.py",
            "crates/zs-server/src/main.rs",
            "worker/tests",
        ),
        complexity_free_lines=180,
        max_changed_lines=320,
        blueprint=(
            "[Active Campaign T1-1 Blueprint — Device-Resident Decode + GPU Engram Window]\n"
            "  - Reference Blueprint: `worker/zstep_worker/models/qwen35_moe.py` (`launch`/`finish` split, "
            "`parity = header.step_id & 1` double-buffered host staging & CUDA events, `STEP_FLAG_DEVICE_TOKENS`).\n"
            "  - Why `deepseek_v41` is currently non-overlapped: `crates/zs-server/src/main.rs` disables "
            "`engine.set_device_token_overlap` on `is_dsv41` ONLY because `DeepSeekV41StepRunner` lacks "
            "`launch`/`finish` and reads `engram_prev_window` from host every decode step.\n"
            "  - Exact Invariants Required (vstack Pillars: Agreement + Isolation + Integrity):\n"
            "    1. CRITICAL TP8 Agreement (`shm_allreduce.py`): `tp8_argmax_gather_shm_kernel` currently "
            "writes `out_top2_tokens` ONLY under `if (rank == 0)`! Before reading `out_tokens_gpu` on "
            "`STEP_FLAG_DEVICE_TOKENS` steps, ALL 8 ranks must merge `slot_pairs` into `out_top2_tokens` "
            "so ranks 0..7 feed identical sampled token IDs.\n"
            "    2. Per-Slot GPU State (`deepseek_v41.py`): maintain `last_token_gpu[max_seqs]` and "
            "`slot_lookback_gpu[max_seqs, 3]` (`[t-1, t-2, t-3]`). Note PyTorch dtype invariant: "
            "`comp_bt_gpu` is `torch.int32`, whereas `index_select`/`index_copy_` strictly require "
            "`torch.int64` (`long`) index tensors (cast `.to(torch.int64)` or allocate slot index buffers as `torch.int64`). "
            "On any non-`STEP_FLAG_DEVICE_TOKENS` step "
            "(prefill/mixed or initial decode step), seed from host `tok_np` & `prev_win_np`; at step end "
            "(inside captured decode graph & after eager sample), shift `slot_lookback_gpu[slot] = "
            "[cur_input_tok, prev_lb_0, prev_lb_1]` and store `last_token_gpu[slot] = sampled_top1_tok`.\n"
            "    3. Parity Double-Buffering: use `parity = header.step_id & 1` for host pinned staging "
            "buffers and `start_events[parity]`/`end_events[parity]` so `launch(step N+1)` never overwrites "
            "buffers still read by `step N`.\n"
            "    4. Enable `engine.set_device_token_overlap(std::env::var(\"ZSTEP_DECODE_DEVICE_TOKENS\")"
            ".map_or(true, |v| v != \"0\"))` in `crates/zs-server/src/main.rs`."
        ),
    ),
    CampaignSpec(
        campaign_id="t1-2-dispatch-buckets",
        aliases=("t1-2", "dispatch-buckets"),
        priority_rank="T1-2",
        title="SM120 Shape-Bucket Ladder & Hybrid SHM/NCCL/MegaGate Dispatch Table",
        plane=EvolutionPlane.L1_KERNEL,
        expected_gain="+4% ~ +10% multi-tier geomean throughput",
        primary_niche_key="(plane=l1_kernel,focus=sm120_dispatch_buckets)",
        allowed_roots=(
            "worker/zstep_worker/models/deepseek_v41.py",
            "worker/zstep_worker/kernels_sm120/b12x_dsv41.py",
            "worker/zstep_worker/kernels_sm120/shm_allreduce.py",
            "worker/tests",
        ),
        complexity_free_lines=90,
        max_changed_lines=180,
        blueprint=(
            "[Active Campaign T1-2 Blueprint — SM120 Shape-Bucket & Dispatch Threshold Optimization]\n"
            "  - High-Leverage Knobs Backed by Audited Telemetry:\n"
            "    1. Finer Decode CUDAGraph Bucket Ladder in `deepseek_v41.py`: stock `(1,2,4,8,16,32,64,128)` "
            "pads `num_seqs=17..24` to `32` and `num_seqs=33..48` to `64` (~2x wasted MoE/MLA rows under "
            "`c32`/`c128` mixed retirement). Adding intermediate runner-local buckets (e.g. `24, 48, 96`) "
            "eliminates padding waste without altering SHM wire layout.\n"
            "    2. Hybrid AllReduce Crossover (`ZSTEP_SHM_MAX_AR_NUMEL`, `DEFAULT_ONE_SHOT_MAX_VECS`): "
            "telemetry shows `one_shot_max_vecs=4096` only covers `M<=6` at `H=5120`, forcing `M=8..32` "
            "decode into two-shot SHM, and NCCL vs `shm_v5` has a sharp crossover between `T=192` and `T=4096`.\n"
            "    3. Router `DEFAULT_MEGA_GATE_MIN_TOKENS` (`32`) & `_CUTLASS_LARGE_M_THRESHOLDS` in `b12x_dsv41.py`.\n"
            "  - Formal Verification Guard: Evolith's `dispatch_verifier` exhaustively checks bucket "
            "coverage, bounds (`num_seqs <= bucket <= max_bucket`), monotonicity, idempotence, and "
            "128-bit vector alignment (`% 8 == 0`) before GPU execution."
        ),
    ),
    CampaignSpec(
        campaign_id="t1-3-chunked-prefill",
        aliases=("t1-3", "chunked-prefill"),
        priority_rank="T1-3",
        title="Adaptive Rust Chunked-Prefill & Mixed-Step Token Budget",
        plane=EvolutionPlane.L2_SCHEDULER,
        expected_gain="+5% ~ +15% c32-c128 throughput within 5% P99 ITL cap",
        primary_niche_key="(plane=l2_scheduler,focus=adaptive_chunked_prefill)",
        allowed_roots=(
            "crates/zs-scheduler/src",
            "crates/zs-server/src/main.rs",
        ),
        complexity_free_lines=120,
        max_changed_lines=240,
        blueprint=(
            "[Active Campaign T1-3 Blueprint — Adaptive Chunked-Prefill & Mixed-Step Budget]\n"
            "  - Target: `crates/zs-scheduler/src/serving.rs` (`max_prefill_chunk=2048`, `max_mixed_step_tokens=3072`).\n"
            "  - Mechanism: dynamically scale piggybacked prefill chunk budget based on active in-flight "
            "decode sequence count and remaining KV headroom so `c16/c32` saturates SM120 tensor cores "
            "while `c128` protects `p99_itl_ms` and `mean_tpot_ms` from prefill head-of-line stalls.\n"
            "  - Preserve all `zs-scheduler` RAII slot/KV reservation & `chunked_schedule` exposure invariants."
        ),
    ),
    CampaignSpec(
        campaign_id="t2-1-step-overlap",
        aliases=("t2-1", "step-overlap"),
        priority_rank="T2-1",
        title="Ping-Pong Host Step Preparation & SHM Staging Overlap",
        plane=EvolutionPlane.L2_SCHEDULER,
        expected_gain="+3% ~ +7% step-bubble reduction",
        primary_niche_key="(plane=l2_scheduler,focus=pingpong_host_staging)",
        allowed_roots=(
            "worker/zstep_worker/models/deepseek_v41.py",
            "worker/zstep_worker/runtime_loop.py",
            "crates/zs-scheduler/src",
            "worker/tests",
        ),
        complexity_free_lines=140,
        max_changed_lines=260,
        blueprint=(
            "[Active Campaign T2-1 Blueprint — Ping-Pong Host Staging & Consolidated Decode I/O]\n"
            "  - Consolidate fragmented host-to-device metadata copies and overlap CPU numpy/table "
            "preparation for step N+1 with GPU execution of step N using strict parity-indexed "
            "double buffers and CUDA Event synchronization (zero single-buffer WAR hazards)."
        ),
    ),
    CampaignSpec(
        campaign_id="t2-2-sm120-fusion",
        aliases=("t2-2", "sm120-fusion"),
        priority_rank="T2-2",
        title="SM120 Epilogue / Router / Norm+Quant Lightweight Kernel Fusion",
        plane=EvolutionPlane.L1_KERNEL,
        expected_gain="+2% ~ +5% kernel launch & bandwidth reduction",
        primary_niche_key="(plane=l1_kernel,focus=sm120_lightweight_fusion)",
        allowed_roots=(
            "worker/zstep_worker/kernels_sm120/norm_quant.py",
            "worker/zstep_worker/kernels_sm120/moe_router.py",
            "worker/zstep_worker/kernels_sm120/b12x_dsv41.py",
            "worker/tests",
        ),
        complexity_free_lines=120,
        max_changed_lines=240,
        blueprint=(
            "[Active Campaign T2-2 Blueprint — SM120 Lightweight Kernel Fusion]\n"
            "  - Fuse adjacent lightweight elementwise/reduction/router operations in SM120 helper "
            "kernels without introducing runtime `@triton.autotune` or dynamic `tl.constexpr` shapes "
            "(strict `0 Triton recompilations` gate)."
        ),
    ),
    CampaignSpec(
        campaign_id="t2-3-spec-decode-graph",
        aliases=("t2-3", "spec-decode"),
        priority_rank="T2-3",
        title="Segmented Speculative Draft/Verify CUDAGraph & Lease-Safe Rollback",
        plane=EvolutionPlane.L2_SCHEDULER,
        expected_gain="1.8x ~ 3.3x single-stream & low-concurrency serving throughput",
        primary_niche_key="(plane=l2_scheduler,focus=spec_decode_verify_graph)",
        allowed_roots=(
            "worker/zstep_worker/spec_decode.py",
            "worker/zstep_worker/models/deepseek_v41.py",
            "worker/zstep_worker/models/deepseek_v41_dspark.py",
            "worker/zstep_worker/models/qwen35_moe.py",
            "worker/zstep_worker/models/qwen35_mtp.py",
            "crates/zs-scheduler/src",
            "worker/tests",
        ),
        complexity_free_lines=160,
        max_changed_lines=300,
        blueprint=(
            "[Active Campaign T2-3 Blueprint — Segmented Speculative Draft/Verify CUDAGraph]\n"
            "  - Invariants Distilled from Live 8x RTX 6000D DSpark/MTP Verification:\n"
            "    1. Zero Host Sync & Zero NCCL: never call `.item()`/`.cpu()` or `dist.broadcast` "
            "inside draft/verify loops; capture segmented `verify_graphs` across `spec_verify_width` "
            "and keep `--speculative-method none` (`M=1` graph) 100% unregressed.\n"
            "    2. Recurrent State Rollback (`qwen35_moe.py` / `qwen35_mtp.py`): on partial draft "
            "rejection, rewind 30-layer GDN `(conv_state, ssm_state_vk)` to the step-start snapshot "
            "and guard SHM `compressor_slot_ids` behind `num_recurrent_checkpoints > 0`.\n"
            "    3. Prefix Lease Page Accounting (`serving.rs`): after `promote_completed_prefixes_in_batch`, "
            "enforce `seq.page_ids.len() == prefix_lease.pages() + seq.kv_leases.len()` on cross-page "
            "rejection rewind so online `idle_kv_cache_usage == 0.0` always holds."
        ),
    ),
)

_CAMPAIGN_LOOKUP: dict[str, CampaignSpec] = {}
for _spec in CAMPAIGNS:
    _CAMPAIGN_LOOKUP[_spec.campaign_id] = _spec
    _CAMPAIGN_LOOKUP[_spec.priority_rank.lower()] = _spec
    for _alias in _spec.aliases:
        _CAMPAIGN_LOOKUP[_alias.lower()] = _spec


def list_campaigns() -> tuple[CampaignSpec, ...]:
    return CAMPAIGNS


def resolve_campaign(
    campaign_selector: str,
    blocked_niches: dict[str, list[str]] | None = None,
) -> CampaignSpec | None:
    """
    Resolves a campaign selector (`'none'`, `'auto'`, or a campaign ID/alias like `'t1-1'`).
    In `'auto'` mode, selects the highest-ROI campaign whose primary niche has not been
    frozen by the Premise 2-Strike circuit breaker.
    """
    key = campaign_selector.strip().lower()
    if not key or key == "none":
        return None
    if key == "auto":
        blocked = blocked_niches or {}
        for spec in CAMPAIGNS:
            if spec.primary_niche_key not in blocked:
                return spec
        return None
    if key not in _CAMPAIGN_LOOKUP:
        valid = ", ".join(["auto", "none", *[c.campaign_id for c in CAMPAIGNS]])
        raise ValueError(f"Unknown campaign `{campaign_selector}`; valid options: {valid}")
    return _CAMPAIGN_LOOKUP[key]


def apply_campaign_to_task(task: ZStepRSITask, spec: CampaignSpec | None) -> ZStepRSITask:
    if spec is None:
        return task
    return task.model_copy(
        update={
            "allowed_roots": spec.allowed_roots,
            "complexity_free_lines": spec.complexity_free_lines,
            "max_changed_lines": spec.max_changed_lines,
        }
    )
