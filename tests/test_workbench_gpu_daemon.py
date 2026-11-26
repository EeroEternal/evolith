from __future__ import annotations

from pathlib import Path
import struct
import threading
import time

import pytest

from evolith.workbench import (
    GraphGuardViolationError,
    GuardedGraphRunner,
    KernelCatalog,
    PosixShmStepPublisher,
    ProvisionHandle,
    StateKeeper,
    WireGeometry,
    WorkbenchDaemonConfig,
    WorkbenchGpuDaemon,
    WorkbenchRpcClient,
)
from evolith.workbench.gpu_daemon import (
    SUPERBLOCK_LAYOUT_HASH_OFFSET,
    SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET,
    SUPERBLOCK_PLAN_OFFSETS_OFFSET,
    SUPERBLOCK_PLAN_STRIDE_OFFSET,
    SUPERBLOCK_RESULT_OFFSETS_OFFSET,
    SUPERBLOCK_RESULT_STRIDE_OFFSET,
    SUPERBLOCK_TOTAL_BYTES_OFFSET,
    ZSTEP_PROTOCOL_VERSION,
    ZSTEP_SHM_MAGIC,
)


def test_guarded_cuda_graph_proxy_enforces_o1_three_part_guard() -> None:
    keeper = StateKeeper(rank=0, world_size=8)
    keeper.register_buffer("weights", [1, 2, 3], nbytes=475_000_000_000)
    catalog = KernelCatalog()
    prov = ProvisionHandle("models.runner", object())
    runner = GuardedGraphRunner(mode="required", buckets=(1, 16, 32))

    replays: list[int] = []

    class DummyCudaGraph:
        def __init__(self, b: int) -> None:
            self.b = b

        def replay(self) -> int:
            replays.append(self.b)
            return self.b

    wrapped = runner.wrap_cuda_graphs(
        {1: DummyCudaGraph(1), 16: DummyCudaGraph(16)},
        provisions=(prov,),
        buffers=keeper.all_buffers(),
        catalog=catalog,
    )
    assert wrapped[16].replay() == 16
    assert runner.graph_replay_steps == 1
    assert runner.graph_guard_failures == 0

    # In-place weight refit bumps `fills` -> invalidates wrapped CUDAGraph immediately!
    keeper.refit_buffer("weights", lambda arr: arr)
    with pytest.raises(GraphGuardViolationError, match="O\\(1\\) CUDA Graph guard"):
        wrapped[16].replay()
    assert runner.graph_guard_failures == 1


def test_posix_shm_step_publisher_wire_layout_and_plans(tmp_path: Path) -> None:
    geom = WireGeometry(world_size=8, max_seqs=32, max_step_tokens=1024, max_decoder_rows=1024)
    pub = PosixShmStepPublisher("evo_test_shm", geom, shm_dir=tmp_path)
    try:
        raw_sb = bytes(pub._mapping[:256])
        vals = struct.unpack_from("<Q9I", raw_sb, 0)
        assert vals[0] == ZSTEP_SHM_MAGIC
        assert vals[1] == ZSTEP_PROTOCOL_VERSION
        assert vals[2] == 8
        assert vals[3] == 32
        assert struct.unpack_from("<Q", raw_sb, SUPERBLOCK_TOTAL_BYTES_OFFSET)[0] == pub.total_shm_bytes
        assert struct.unpack_from("<2Q", raw_sb, SUPERBLOCK_PLAN_OFFSETS_OFFSET) == pub.plan_slot_offsets
        assert (
            struct.unpack_from("<Q", raw_sb, SUPERBLOCK_PLAN_STRIDE_OFFSET)[0]
            == pub.layout.total_slot_bytes
        )
        assert struct.unpack_from("<2Q", raw_sb, SUPERBLOCK_RESULT_OFFSETS_OFFSET) == pub.result_slot_offsets
        assert (
            struct.unpack_from("<Q", raw_sb, SUPERBLOCK_RESULT_STRIDE_OFFSET)[0]
            == pub.result_slot_stride
        )
        assert struct.unpack_from("<Q", raw_sb, SUPERBLOCK_LAYOUT_HASH_OFFSET)[0] == pub.layout_hash
        assert struct.unpack_from("<I", raw_sb, SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET)[0] == 128

        pub.assign_linear_page_tables(total_seqs=8, pages_per_seq=4)
        pub.publish_prefill_step(step_id=1, slot_ids=[0, 1, 2, 3], context_len=128)
        slot1_base = pub.plan_slot_offsets[1]
        assert struct.unpack_from("<Q", pub._mapping, slot1_base)[0] == 1
        hdr_fields = struct.unpack_from("<12I", pub._mapping, slot1_base + 8)
        assert hdr_fields[1] == 4  # num_seqs
        assert hdr_fields[2] == 512  # num_encoder_tokens

        pub.publish_decode_step(
            step_id=2, num_seqs=4, decode_pos=128, decode_graph_bucket=4
        )
        slot0_base = pub.plan_slot_offsets[0]
        assert struct.unpack_from("<Q", pub._mapping, slot0_base)[0] == 2
        dec_fields = struct.unpack_from("<12I", pub._mapping, slot0_base + 8)
        assert dec_fields[8] == 1  # STEP_FLAG_DECODE_ONLY
    finally:
        pub.close(unlink=True)
    assert not (tmp_path / "evo_test_shm").exists()


def test_multiprocess_workbench_daemon_rpc_hotswap_shadow_sleep_and_reattach(
    tmp_path: Path,
) -> None:
    env_db = tmp_path / "envelope.sqlite3"
    cfg = WorkbenchDaemonConfig(
        host="127.0.0.1",
        port=0,
        runner_kind="mock",
        world_size=2,
        shm_name="evo_wb_unit",
        active_decode_buckets=(1, 16, 32),
        envelope_db_path=str(env_db),
    )
    daemon = WorkbenchGpuDaemon(cfg, shm_dir=tmp_path)
    srv_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    srv_thread.start()

    # Wait for TCP server port assignment
    deadline = time.monotonic() + 15.0
    while daemon.cfg.port == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert daemon.cfg.port > 0

    client = WorkbenchRpcClient(host="127.0.0.1", port=daemon.cfg.port, timeout_sec=30.0)
    try:
        for _ in range(40):
            try:
                if client.call("ping").get("pong"):
                    break
            except OSError:
                time.sleep(0.05)

        st0 = client.call("status")
        assert st0["ok"]
        assert st0["world_size"] == 2
        assert st0["active_version"] == 1

        # 1. Baseline resident inner_screen
        scr_base = client.call(
            "inner_screen",
            buckets=[1, 16, 32],
            context_len=128,
            warmup_steps=2,
            timed_steps=5,
        )
        assert scr_base["ok"]
        assert scr_base["eager_fallback_steps"] == 0
        assert scr_base["triton_recompilations"] == 0
        base_geomean = scr_base["geomean_tok_s"]

        # 2. Valid speedup `hot_apply` (20% lower latency, identical greedy tokens) -> commits!
        fast_hook = (
            "def apply(runner, engine):\n"
            "    old = runner.latency_scale[0]\n"
            "    runner.latency_scale[0] = 0.8\n"
            "    return lambda: runner.latency_scale.__setitem__(0, old)\n"
        )
        res_fast = client.call(
            "hot_apply",
            candidate_id="cand-fast",
            changed_files=["worker/zstep_worker/kernels_sm120/b12x_dsv41.py"],
            hook_code=fast_hook,
            recapture_buckets=[1, 16, 32],
            shadow_check=True,
        )
        assert res_fast["ok"] and res_fast["committed"]
        scr_fast = client.call(
            "inner_screen",
            buckets=[1, 16, 32],
            context_len=128,
            warmup_steps=1,
            timed_steps=4,
        )
        assert scr_fast["geomean_tok_s"] > base_geomean * 1.15

        # 3. Un-recaptured bucket `64` retains revoked O(1) guard -> fails closed immediately!
        with pytest.raises(Exception):
            daemon.inner_screen(buckets=[64], context_len=128, warmup_steps=1, timed_steps=1)

        # 4. Buggy candidate corrupting output tokens fails `--shadow` check & rolls back cleanly!
        buggy_hook = (
            "def apply(runner, engine):\n"
            "    old_b = runner.token_bias\n"
            "    runner.token_bias = 77\n"
            "    return lambda: setattr(runner, 'token_bias', old_b)\n"
        )
        res_drift = client.call(
            "hot_apply",
            candidate_id="cand-drift",
            changed_files=["worker/zstep_worker/kernels_sm120/b12x_dsv41.py"],
            hook_code=buggy_hook,
            recapture_buckets=[1, 16, 32],
            shadow_check=True,
        )
        assert not res_drift["committed"]
        assert res_drift["rolled_back"]
        assert res_drift["failure_kind"] == "shadow_drift"

        # Confirm rollback preserved cand-fast's exact tokens & throughput
        scr_after_rb = client.call(
            "inner_screen",
            buckets=[16],
            context_len=128,
            warmup_steps=1,
            timed_steps=3,
        )
        assert scr_after_rb["buckets"][0]["sample_tokens"] == scr_fast["buckets"][1]["sample_tokens"]

        # 5. Pinned contract boundary (`crates/zs-wire/src/lib.rs`) rejected into EnvelopeFailureLog
        res_gap = client.call(
            "hot_apply",
            candidate_id="cand-pinned",
            changed_files=["crates/zs-wire/src/lib.rs"],
            shadow_check=False,
        )
        assert not res_gap["committed"]
        assert res_gap["failure_kind"] == "envelope_gap"

        # 6. Sleep / Wake releases KV VRAM without dropping 475 GiB weights & bumps generation
        slp = client.call("sleep")
        assert slp["ok"] and slp["sleeping"]
        assert slp["freed_vram_mib_total"] == 128
        wk = client.call("wake", recapture_buckets=[1, 16, 32])
        assert wk["ok"] and not wk["sleeping"]
        assert wk["restored_vram_mib_total"] == 128
        assert wk["kv_generation"] == 2

        # 7. Reattach new POSIX SHM channel without restarting workers
        reatt = client.call("reattach_shm", new_shm_name="evo_wb_epoch2")
        assert reatt["ok"]
        assert reatt["scheduler_attach_epoch"] == 2
        scr_ep2 = client.call(
            "inner_screen",
            buckets=[1, 16],
            context_len=128,
            warmup_steps=1,
            timed_steps=2,
        )
        assert scr_ep2["ok"]
    finally:
        client.call("shutdown")
        srv_thread.join(timeout=10.0)


def test_multiprocess_watchdog_sigkills_hung_worker_group(tmp_path: Path) -> None:
    cfg = WorkbenchDaemonConfig(
        runner_kind="mock",
        world_size=2,
        shm_name="evo_wb_hang",
        rpc_watchdog_timeout_sec=0.3,
    )
    daemon = WorkbenchGpuDaemon(cfg, shm_dir=tmp_path)
    daemon.start()
    with pytest.raises(TimeoutError, match="Hard multi-rank watchdog tripped"):
        daemon.dispatch_rpc(
            {"method": "simulate_hang", "params": {"seconds": 5.0, "watchdog_timeout_sec": 0.15}}
        )
    assert all(not p.is_alive() for p in daemon._procs)


def test_native_op_abi_and_multiprocess_daemon_so_hotswap(tmp_path: Path) -> None:
    import ctypes
    import shutil

    from evolith.workbench import ExternCOpLoader, resolve_ctype_spec

    assert resolve_ctype_spec("u64") is ctypes.c_uint64
    assert resolve_ctype_spec("i32") is ctypes.c_int32
    assert resolve_ctype_spec("void") is None
    assert (
        ExternCOpLoader.detect_cuda_arch_flag("sm_120")
        == "-gencode=arch=compute_120,code=sm_120"
    )

    if not shutil.which("cc") and not shutil.which("clang") and not shutil.which("gcc"):
        pytest.skip("No host C compiler available")

    cfg = WorkbenchDaemonConfig(
        host="127.0.0.1",
        port=0,
        runner_kind="mock",
        world_size=2,
        shm_name="evo_wb_native",
        active_decode_buckets=(1, 16),
    )
    daemon = WorkbenchGpuDaemon(cfg, shm_dir=tmp_path)
    srv_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    srv_thread.start()
    deadline = time.monotonic() + 15.0
    while daemon.cfg.port == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert daemon.cfg.port > 0

    client = WorkbenchRpcClient(host="127.0.0.1", port=daemon.cfg.port, timeout_sec=30.0)
    try:
        for _ in range(40):
            try:
                if client.call("ping").get("pong"):
                    break
            except OSError:
                time.sleep(0.05)

        scr0 = client.call("inner_screen", buckets=[1, 16], context_len=128, warmup_steps=1, timed_steps=3)

        # 1. Crashing native `.so` dies inside disposable preflight subprocess without harming workers!
        res_crash = client.call(
            "hot_apply",
            candidate_id="native-crash",
            changed_files=["worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu"],
            native_op={
                "op_name": "bad_op",
                "symbol_name": "evo_bad_op",
                "c_source": "#include <stdlib.h>\nint evo_bad_op(int x) { abort(); return x; }\n",
                "argtypes": ["i32"],
                "restype": "i32",
                "smoke_expr": "fn.argtypes = [ctypes.c_int32]; fn.restype = ctypes.c_int32; fn(1)",
            },
            recapture_buckets=[1, 16],
        )
        assert not res_crash["ok"] and not res_crash["committed"]
        assert res_crash["failure_kind"] == "native_preflight_rejected"

        # 2. Valid native `.so` compiled once, preflighted, dlopened across all ranks -> commits!
        res_ok = client.call(
            "hot_apply",
            candidate_id="native-fast",
            changed_files=["worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu"],
            native_op={
                "op_name": "lat_pct",
                "symbol_name": "evo_lat_pct",
                "c_source": "int evo_lat_pct(int base_pct) { return base_pct - 25; }\n",
                "argtypes": ["i32"],
                "restype": "i32",
                "smoke_expr": (
                    "fn.argtypes = [ctypes.c_int32]; fn.restype = ctypes.c_int32; "
                    "assert fn(100) == 75"
                ),
            },
            hook_code=(
                "def apply(runner, engine):\n"
                "    fn = engine.catalog.get('lat_pct')\n"
                "    old = runner.latency_scale[0]\n"
                "    runner.latency_scale[0] = fn(100) / 100.0\n"
                "    return lambda: runner.latency_scale.__setitem__(0, old)\n"
            ),
            recapture_buckets=[1, 16],
            shadow_check=True,
        )
        assert res_ok["ok"] and res_ok["committed"]
        assert res_ok["native_content_hash"]

        scr1 = client.call("inner_screen", buckets=[1, 16], context_len=128, warmup_steps=1, timed_steps=3)
        assert scr1["geomean_tok_s"] > scr0["geomean_tok_s"] * 1.25

        # 3. Numerically buggy native `.so` passes smoke check, caught by `--shadow` & rolled back!
        res_drift = client.call(
            "hot_apply",
            candidate_id="native-drift",
            changed_files=["worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu"],
            native_op={
                "op_name": "tok_bias",
                "symbol_name": "evo_tok_bias",
                "c_source": "int evo_tok_bias(int x) { return x + 91; }\n",
                "argtypes": ["i32"],
                "restype": "i32",
                "smoke_expr": "fn.argtypes = [ctypes.c_int32]; fn.restype = ctypes.c_int32; assert fn(0) == 91",
            },
            hook_code=(
                "def apply(runner, engine):\n"
                "    fn = engine.catalog.get('tok_bias')\n"
                "    old_b = runner.token_bias\n"
                "    runner.token_bias = fn(0)\n"
                "    return lambda: setattr(runner, 'token_bias', old_b)\n"
            ),
            recapture_buckets=[1, 16],
            shadow_check=True,
        )
        assert not res_drift["committed"] and res_drift["rolled_back"]
        assert res_drift["failure_kind"] == "shadow_drift"

        scr2 = client.call("inner_screen", buckets=[1, 16], context_len=128, warmup_steps=1, timed_steps=3)
        assert scr2["buckets"][1]["sample_tokens"] == scr1["buckets"][1]["sample_tokens"]
    finally:
        client.call("shutdown")
        srv_thread.join(timeout=10.0)


def test_top2_margin_consistent_rate_handles_symmetric_ties_and_rejects_confident_flips() -> None:
    from evolith.workbench.gpu_daemon import (
        inner_screen_consistency_rate,
        top2_margin_consistent_rate,
    )

    # Near-tie swaps (`inc_margin <= 1.5`) with overlapping top-2 (`{1,201} & {28,201}`) pass!
    inc_toks = [1, 4189, 1]
    inc_lps = [19.25, 31.5, 20.625]
    inc_ru_toks = [201, 477, 305]
    inc_ru_lps = [19.0, 27.75, 19.5]

    cand_valid_toks = [28, 4189, 28]
    cand_valid_lps = [19.625, 31.375, 20.25]
    cand_valid_ru = [201, 4442, 1]
    assert (
        top2_margin_consistent_rate(
            inc_toks,
            inc_lps,
            inc_ru_toks,
            inc_ru_lps,
            cand_valid_toks,
            cand_valid_lps,
            cand_valid_ru,
        )
        == 1.0
    )

    # Sample-weighted across buckets (`1 + 8 + 8 = 17` samples -> `16/17 = 94.12%`)
    ref_scr = {
        "buckets": [
            {
                "sample_tokens": [1],
                "sample_logprobs": [19.125],
                "runner_up_tokens": [201],
                "runner_up_logprobs": [19.125],
            },
            {
                "sample_tokens": inc_toks,
                "sample_logprobs": inc_lps,
                "runner_up_tokens": inc_ru_toks,
                "runner_up_logprobs": inc_ru_lps,
            },
        ]
    }
    cand_scr = {
        "buckets": [
            {
                "sample_tokens": [28],
                "sample_logprobs": [19.5],
                "runner_up_tokens": [305],
                "runner_up_logprobs": [18.75],
            },
            {
                "sample_tokens": cand_valid_toks,
                "sample_logprobs": cand_valid_lps,
                "runner_up_tokens": cand_valid_ru,
                "runner_up_logprobs": [19.0, 28.25, 19.875],
            },
        ]
    }
    assert inner_screen_consistency_rate(ref_scr, cand_scr) == 0.75

    # Flipping confident slot 1 (`31.5 - 27.75 = 3.75 > 1.5`) to runner-up `477` MUST fail!
    cand_bad_confident = [28, 477, 28]
    cand_bad_lps = [19.625, 31.25, 20.25]
    assert (
        top2_margin_consistent_rate(
            inc_toks,
            inc_lps,
            inc_ru_toks,
            inc_ru_lps,
            cand_bad_confident,
            cand_bad_lps,
            cand_valid_ru,
        )
        < 0.7
    )


def test_workbench_daemon_profile_step_exports_ledger_and_contracts(tmp_path: Path) -> None:
    cfg = WorkbenchDaemonConfig(
        host="127.0.0.1",
        port=0,
        runner_kind="mock",
        world_size=2,
        active_decode_buckets=(1, 16, 32),
        rpc_watchdog_timeout_sec=10.0,
    )
    daemon = WorkbenchGpuDaemon(cfg, shm_dir=tmp_path)
    srv_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    srv_thread.start()
    try:
        deadline = time.monotonic() + 10.0
        while daemon.cfg.port == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        client = WorkbenchRpcClient(host="127.0.0.1", port=daemon.cfg.port, timeout_sec=15.0)
        for _ in range(50):
            try:
                if client.call("ping").get("pong"):
                    break
            except OSError:
                time.sleep(0.05)

        prof = client.call(
            "profile_step", bucket=32, context_len=128, warmup_steps=1, timed_steps=3
        )
        assert prof["ok"] is True
        assert prof["bucket"] == 32
        assert prof["world_size"] == 2
        assert prof["captured_graph_step_us"] > 0.0
        assert 0.85 <= prof["coverage_fraction"] <= 1.0
        assert abs(
            (prof["attributed_stage_us"] + prof["unattributed_step_us"])
            - prof["instrumented_gpu_step_us"]
        ) < 1e-3

        # Top target on mock runner is `dsv4.b12x_moe_ffn`, followed by `dsv4.mhc_allreduce`
        assert prof["selected_target"]["op_name"] == "dsv4.b12x_moe_ffn"
        assert any(g["op_name"] == "step.unattributed_glue" for g in prof["capability_gaps"])
        contracts = prof["callsite_contracts"]
        assert len(contracts) == 4
        c_ar = next(c for c in contracts if c["op_name"] == "dsv4.mhc_allreduce")
        assert c_ar["hotswap_extern_c"] is True
        assert c_ar["action_rung"] == "RUNG_3_HOTSWAP_NATIVE_OP"
        assert c_ar["shapes"] == [[32, 5120], [32, 4, 5120]]

        # Blocking the top niche automatically pivots selection to the next bottleneck!
        prof_next = client.call(
            "profile_step",
            bucket=32,
            context_len=128,
            warmup_steps=1,
            timed_steps=2,
            blocked_niches=["(plane=l1_kernel,op=dsv4.b12x_moe_ffn)"],
        )
        assert prof_next["selected_target"]["op_name"] == "dsv4.mhc_allreduce"

        # Resident guarded CUDAGraphs remain completely intact after `profile_step()`!
        scr_after = client.call(
            "inner_screen", buckets=[1, 32], context_len=128, warmup_steps=1, timed_steps=3
        )
        assert scr_after["ok"] is True
        assert scr_after["eager_fallback_steps"] == 0
        assert scr_after["graph_guard_failures"] == 0
    finally:
        client.call("shutdown")
        srv_thread.join(timeout=10.0)
