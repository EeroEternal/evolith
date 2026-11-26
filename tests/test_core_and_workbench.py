from __future__ import annotations

import ctypes
import json
from pathlib import Path
import time

import numpy as np
import pytest

from evolith.core import (
    ActionRung,
    EnvelopeFailureLog,
    EnvelopeGap,
    EnvelopeGapCategory,
    analyze_changeset_blast_radius,
    are_patches_closure_orthogonal,
    compute_system_identity,
    default_zstep_surface,
    diff_identities,
    materialize_readonly_snapshot,
    run_protocol_verification_suite,
)
from evolith.core.formal_replay import replay_kv_trace, replay_workbench_tx_trace
from evolith.models import EvolutionPlane
from evolith.workbench import (
    ExternCOpLoader,
    GraphGuardViolationError,
    GuardedGraphRunner,
    KernelCatalog,
    MorphableWorkbenchEngine,
    NativeOpPreflightError,
    ProvisionHandle,
    StateKeeper,
    StepBottleneckLedger,
    StepOperatorSample,
    run_shadow_differential,
)
from evolith.zstep.archive import ZStepArchive
from evolith.zstep.composer import find_orthogonal_stepping_stone_pairs
from evolith.zstep.redteam import scan_patch_boundaries
from evolith.zstep.task import CandidateStatus, PatchStats
from evolith.zstep.trace_conformance import TraceConformanceError, verify_trace_conformance


def test_system_identity_diff_and_readonly_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "worker").mkdir(parents=True)
    (repo / "worker" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    id1 = compute_system_identity(
        repo,
        config={"max_tokens": 4096},
        allowed_env_keys=("ZSTEP_MODE", "UNUSED_SECRET"),
        environ={"ZSTEP_MODE": "tp8", "UNLISTED_VAR": "ignored"},
        topology={"tp": 8, "pp": 1},
    )
    # Changing an unlisted env var or `.git/` does not alter deterministic identity
    (repo / ".git" / "HEAD").write_text("deadbeef\n", encoding="utf-8")
    id1_repeat = compute_system_identity(
        repo,
        config={"max_tokens": 4096},
        allowed_env_keys=("ZSTEP_MODE", "UNUSED_SECRET"),
        environ={"ZSTEP_MODE": "tp8", "UNLISTED_VAR": "changed"},
        topology={"tp": 8, "pp": 1},
    )
    assert id1.digest == id1_repeat.digest
    assert diff_identities(id1, id1_repeat).identical

    (repo / "worker" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    id2 = compute_system_identity(
        repo,
        config={"max_tokens": 8192},
        allowed_env_keys=("ZSTEP_MODE",),
        environ={"ZSTEP_MODE": "pp2tp4"},
        topology={"tp": 4, "pp": 2},
    )
    diff = diff_identities(id1, id2)
    assert not diff.identical
    assert diff.modified_files == ("worker/mod.py",)
    assert diff.changed_config_keys == ("max_tokens",)
    assert diff.changed_env_keys == ("ZSTEP_MODE",)
    assert diff.topology_changed

    snap_dir, snap_id = materialize_readonly_snapshot(repo, tmp_path / "snapshots", id2)
    assert snap_id.digest == id2.digest
    staged_file = snap_dir / "worker" / "mod.py"
    assert staged_file.is_file()
    assert (staged_file.stat().st_mode & 0o222) == 0  # read-only


def test_envelope_failure_log_and_zstep_archive_integration(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite3"
    with ZStepArchive(db_path) as archive:
        archive.register_attempt("g0001-test", 1, None, str(tmp_path / "ev"))
        gap = archive.record_envelope_gap(
            EnvelopeGap(
                candidate_id="g0001-test",
                target_engine="zstep",
                plane=EvolutionPlane.L1_KERNEL,
                category=EnvelopeGapCategory.RESOURCE_BUDGET_CEILING,
                blocked_by_boundary="STATIC_RUNNER_PXB_TOTAL_BUDGET (256 MiB PP=1 ceiling)",
                attempted_optimization="4-slot 320 MiB PXB ring",
                observed_bottleneck_share=0.38,
                required_rung="RUNG_6_ENVELOPE_GAP",
                files_involved=("worker/zstep_worker/models/deepseek_v41.py",),
            )
        )
        gaps = archive.list_envelope_gaps()
        assert len(gaps) == 1
        assert gaps[0].gap_id == gap.gap_id
        assert gaps[0].observed_bottleneck_share == pytest.approx(0.38)
        events = archive.list_events("g0001-test")
        assert any(ev["event_type"] == "envelope_gap" for ev in events)


def test_declarative_surface_action_ladder_and_composer_closure(tmp_path: Path) -> None:
    surface = default_zstep_surface()

    # 1. Live scheduler config -> RUNG_1_LIVE_CONFIG, no graph recapture
    p_live = analyze_changeset_blast_radius(
        surface,
        changed_config_keys=("sched.max_mixed_step_tokens",),
    )
    assert p_live.rung == ActionRung.RUNG_1_LIVE_CONFIG
    assert not p_live.recapture_cuda_graphs

    # 2. C++/CUDA kernel edit -> RUNG_3_HOTSWAP_NATIVE_OP + transitive reload closure
    p_cu = analyze_changeset_blast_radius(
        surface,
        changed_files=("worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu",),
    )
    assert p_cu.rung == ActionRung.RUNG_3_HOTSWAP_NATIVE_OP
    assert p_cu.recapture_cuda_graphs
    assert "kernels.shm_allreduce_csrc" in p_cu.directly_touched_units
    assert "models.runner" in p_cu.reload_closure_units

    # 3. Weight packing hook in b12x -> RUNG_4_REFIT_STATE
    p_refit = analyze_changeset_blast_radius(
        surface,
        changed_files=("worker/zstep_worker/kernels_sm120/b12x_dsv41.py",),
        patch_text="+ lyr._b12x_wo_a_packed = block_fp8_linear.pack_weight(w)",
    )
    assert p_refit.rung == ActionRung.RUNG_4_REFIT_STATE

    # 4. Pinned wire struct -> RUNG_6_ENVELOPE_GAP
    p_pin = analyze_changeset_blast_radius(
        surface,
        changed_files=("crates/zs-wire/src/lib.rs",),
    )
    assert p_pin.rung == ActionRung.RUNG_6_ENVELOPE_GAP
    assert not p_pin.can_hot_apply_in_workbench

    # 5. Disjoint files contending on `pxb_ring_budget` are rejected by closure composer
    ok_conflict, reason_conflict = are_patches_closure_orthogonal(
        surface,
        left_files=("worker/zstep_worker/models/deepseek_v41.py",),
        right_files=("worker/zstep_worker/kernels_sm120/shm_allreduce.py",),
        left_patch_text="+ self.shm_ar_decode = HostMappedShmAllReduce(pxb_max_elements=655360)",
        right_patch_text="+ enable_pxb_p2p = True",
    )
    assert not ok_conflict
    assert "pxb_ring_budget" in reason_conflict

    # Verify ZStepArchive stepping-stone composer filters implicit resource collisions
    db_path = tmp_path / "comp.sqlite3"
    with ZStepArchive(db_path) as archive:
        p1 = (
            b"--- a/worker/zstep_worker/models/deepseek_v41.py\n"
            b"+++ b/worker/zstep_worker/models/deepseek_v41.py\n"
            b"@@ -1 +1 @@\n-x\n+HostMappedShmAllReduce(enable_pxb_p2p=True)\n"
        )
        p2 = (
            b"--- a/worker/zstep_worker/kernels_sm120/shm_allreduce.py\n"
            b"+++ b/worker/zstep_worker/kernels_sm120/shm_allreduce.py\n"
            b"@@ -1 +1 @@\n-y\n+pxb_max_elements = 1024\n"
        )
        p3 = (
            b"--- a/crates/zs-scheduler/src/lib.rs\n"
            b"+++ b/crates/zs-scheduler/src/lib.rs\n"
            b"@@ -1 +1 @@\n-z\n+let chunk = 1024;\n"
        )
        for idx, (cid, pb, files) in enumerate(
            (
                ("g0001-a", p1, ("worker/zstep_worker/models/deepseek_v41.py",)),
                ("g0002-b", p2, ("worker/zstep_worker/kernels_sm120/shm_allreduce.py",)),
                ("g0003-c", p3, ("crates/zs-scheduler/src/lib.rs",)),
            ),
            start=1,
        ):
            archive.start_candidate(
                cid,
                idx,
                None,
                pb,
                f"sha-{idx}",
                changed_lines=10,
                changed_files=files,
                upstream_sha="5cc06d8",
            )
            archive.finish_candidate(
                cid,
                CandidateStatus.VERIFY_REJECTED,
                "",
                "geomean=+2.10% < +3.00%",
                geomean_gain=0.021,
            )
        pairs = find_orthogonal_stepping_stone_pairs(archive, upstream_sha="5cc06d8", parent_id=None)
        pair_ids = {(p.left_id, p.right_id) for p in pairs}
        assert ("g0001-a", "g0002-b") not in pair_ids
        assert ("g0001-a", "g0003-c") in pair_ids


def test_redteam_blocks_probe_residue_and_undeclared_env_reads() -> None:
    bad_patch = (
        b"diff --git a/worker/zstep_worker/models/deepseek_v41.py b/worker/zstep_worker/models/deepseek_v41.py\n"
        b"--- a/worker/zstep_worker/models/deepseek_v41.py\n"
        b"+++ b/worker/zstep_worker/models/deepseek_v41.py\n"
        b"@@ -10,2 +10,4 @@\n"
        b"+    # PROBE temporary timing hook\n"
        b"+    fast = os.getenv('SECRET_BENCH_FASTPATH', '0')\n"
    )
    findings = scan_patch_boundaries(
        PatchStats(
            patch=bad_patch,
            changed_lines=2,
            changed_files=("worker/zstep_worker/models/deepseek_v41.py",),
            plane=EvolutionPlane.L2_SCHEDULER,
        )
    )
    msgs = [f.message for f in findings if f.severity == "blocking"]
    assert any("PROBE" in m for m in msgs)
    assert any("os.getenv" in m for m in msgs)

    # Declared env_config.py is permitted to read os.getenv
    ok_patch = (
        b"diff --git a/worker/zstep_worker/env_config.py b/worker/zstep_worker/env_config.py\n"
        b"--- a/worker/zstep_worker/env_config.py\n"
        b"+++ b/worker/zstep_worker/env_config.py\n"
        b"@@ -10,1 +10,2 @@\n"
        b"+    flag = os.getenv('ZSTEP_DECLARED_FLAG', '1')\n"
    )
    ok_findings = scan_patch_boundaries(
        PatchStats(
            patch=ok_patch,
            changed_lines=1,
            changed_files=("worker/zstep_worker/env_config.py",),
            plane=EvolutionPlane.L2_SCHEDULER,
        )
    )
    assert not [f for f in ok_findings if f.severity == "blocking"]


def test_trace_conformance_seis_and_cuda_graph_required_clauses(tmp_path: Path) -> None:
    parity_file = tmp_path / "parity.json"
    metrics_file = tmp_path / "metrics.json"
    base_parity = {
        "triton_recompilations": 0,
        "chunked_schedule": {"chunked_prefill_steps": 3, "decode_steps_with_partial_prefill": 2},
        "cases": [
            {
                "name": "prime_squares_table",
                "single_match": {"exact_token_match_rate": 1.0},
                "batched_match": {"exact_token_match_rate": 1.0},
                "chunked_match": {"exact_token_match_rate": 1.0},
            }
        ],
    }
    base_metrics = {
        "tiers": [
            {
                "label": "in512_c16",
                "repeat": 0,
                "output_throughput_tok_s": 1200.0,
                "median_ttft_ms": 180.0,
                "mean_tpot_ms": 12.5,
                "p99_itl_ms": 13.0,
                "idle_kv_cache_usage": 0.0,
                "steps": 100,
                "overlapped_decode_steps": 80,
            }
        ]
    }
    parity_file.write_text(json.dumps(base_parity), encoding="utf-8")
    metrics_file.write_text(json.dumps(base_metrics), encoding="utf-8")

    # Stock legacy artifacts remain CONFORMANT when optional SEIS/graph telemetry is not required
    rep_legacy = verify_trace_conformance(parity_file, metrics_file, raise_on_failure=True)
    assert rep_legacy.passed

    # Silent eager fallback triggers VIOLATION
    bad_graph_metrics = {**base_metrics, "eager_fallback_steps": 4, "graph_guard_failures": 0}
    metrics_file.write_text(json.dumps(bad_graph_metrics), encoding="utf-8")
    with pytest.raises(TraceConformanceError, match="TRACE_CUDA_GRAPH_REQUIRED"):
        verify_trace_conformance(parity_file, metrics_file, require_cuda_graph=True)

    # SEIS vocabulary slicing hack triggers VIOLATION
    clean_graph_metrics = {**base_metrics, "eager_fallback_steps": 0, "graph_guard_failures": 0}
    metrics_file.write_text(json.dumps(clean_graph_metrics), encoding="utf-8")
    bad_seis_parity = {
        **base_parity,
        "seis_invariants": {
            "eos_stop_consistent": True,
            "vocab_slicing_detected": False,
            "vocab_size_checked": 80000,
            "distinct_request_sequences": 4,
            "decode_steps_exercised": 64,
        },
    }
    parity_file.write_text(json.dumps(bad_seis_parity), encoding="utf-8")
    with pytest.raises(TraceConformanceError, match="TRACE_SEIS_ANTI_CHEAT_INVARIANTS"):
        verify_trace_conformance(
            parity_file,
            metrics_file,
            require_cuda_graph=True,
            require_seis_checks=True,
            expected_vocab_size=129280,
        )


def test_tla_style_protocol_verification_and_trace_replay_suite() -> None:
    report = run_protocol_verification_suite()
    assert report.passed, f"Protocol verification failed: {report.failures}"
    assert report.normal_models_checked == 3
    assert report.bug_mutants_killed == report.bug_mutants_total == 5
    assert report.witnesses_generated == 4
    assert report.in_memory_guard_mutations_caught == 2

    # Tampered trace SHA-256 is rejected fail-closed
    from evolith.core.formal_replay import ProtocolTrace

    tampered = ProtocolTrace(
        protocol="workbench_2pc",
        case_name="tampered",
        kind="witness",
        actions=("prepare_v2",),
        states=[{}],
        trace_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        replay_workbench_tx_trace(tampered)
    with pytest.raises(ValueError, match="Malformed KV"):
        replay_kv_trace(tampered)


def test_workbench_state_keeper_graph_guard_and_extern_c_hotswap(tmp_path: Path) -> None:
    keeper = StateKeeper(rank=0, world_size=8)
    w_buf = keeper.register_buffer("weights_475g", [10, 20], nbytes=475_000_000_000)
    kv_buf = keeper.register_buffer(
        "kv_pool",
        bytearray(1024),
        nbytes=1024,
        persistent_across_sleep=False,
        wake_factory=lambda: bytearray(1024),
    )
    catalog = KernelCatalog()
    catalog.register("scale_op", lambda x: x * 2)
    prov = ProvisionHandle("models.runner", object())
    runner = GuardedGraphRunner(mode="required", buckets=(1, 8, 32))
    runner.capture_all(
        lambda rows, x: (rows, catalog.get("scale_op")(x)),
        provisions=(prov,),
        buffers=(w_buf, kv_buf),
        catalog=catalog,
    )
    assert runner.run_step(4, 5, eager_fn=lambda r, x: (r, x)) == (4, 10)

    # In-place weight refit bumps `fills` (same Python object / address generation!) -> blocks stale graph
    keeper.refit_buffer("weights_475g", lambda arr: [v + 1 for v in arr])
    with pytest.raises(GraphGuardViolationError, match="O\\(1\\) Graph guard"):
        runner.run_step(4, 5, eager_fn=lambda r, x: (r, x))

    # Sleep / wake releases transient KV pool without dropping 475 GB weights & bumps generation
    freed = keeper.sleep()
    assert freed == 1024
    assert not w_buf.released
    restored = keeper.wake()
    assert restored == 1024
    assert kv_buf.generation == 2
    assert keeper.reattach_scheduler_channel("zstep_shm_epoch2") == 1

    # Real content-hashed C shared library hot-swap + subprocess crash isolation
    loader = ExternCOpLoader(tmp_path / "native_cache")
    if loader.compiler:
        op_v1 = loader.compile_c_source(
            "affine",
            "int evo_affine(int x) { return 2 * x; }\n",
            symbol_name="evo_affine",
        )
        op_v2 = loader.compile_c_source(
            "affine",
            "int evo_affine(int x) { return 3 * x + 1; }\n",
            symbol_name="evo_affine",
        )
        assert op_v1.so_path != op_v2.so_path
        fn1 = loader.load_verified_symbol(
            op_v1,
            argtypes=(ctypes.c_int,),
            restype=ctypes.c_int,
            smoke_expr="assert fn(4) == 8",
        )
        fn2 = loader.load_verified_symbol(
            op_v2,
            argtypes=(ctypes.c_int,),
            restype=ctypes.c_int,
            smoke_expr="assert fn(4) == 13",
        )
        assert fn1(7) == 14
        assert fn2(7) == 22

        # A crashing `abort()` native mutation dies inside preflight subprocess without harming parent!
        op_crash = loader.compile_c_source(
            "affine_bad",
            "#include <stdlib.h>\nint evo_affine(int x) { abort(); return x; }\n",
            symbol_name="evo_affine",
        )
        with pytest.raises(NativeOpPreflightError, match="Subprocess preflight crashed"):
            loader.load_verified_symbol(
                op_crash,
                argtypes=(ctypes.c_int,),
                restype=ctypes.c_int,
                smoke_expr="fn(1)",
            )
        # Resident v2 kernel remains intact after rejected crash candidate
        assert fn2(7) == 22


def test_workbench_engine_2pc_shadow_watchdog_and_graduation(tmp_path: Path) -> None:
    env_log = EnvelopeFailureLog(tmp_path / "wb.sqlite3")
    wb = MorphableWorkbenchEngine(default_zstep_surface(), envelope_log=env_log)
    wb.state_keeper.register_buffer("wo_a_packed", np.ones(4, dtype=np.float32), nbytes=16)
    wb.bind_step_runner("models.runner", object(), lambda rows, x: x * 2.0)
    assert wb.run_step(8, 3.0) == pytest.approx(6.0)

    # 1. Valid hot_apply with selective buffer refit + passing `--shadow` check commits & recaptures
    res_ok = wb.hot_apply(
        candidate_id="cand-ok",
        changed_files=("worker/zstep_worker/kernels_sm120/b12x_dsv41.py",),
        patch_text="+ block_fp8_linear.pack_weight(w)",
        refit_buffers={"wo_a_packed": lambda arr: arr * 2.0},
        mutation_fn=lambda eng: eng.bind_step_runner(
            "models.runner", object(), lambda rows, x: x * 2.001
        ),
        shadow_stages=(
            ("attn_wo_a", lambda a: a * 2.0, lambda a: a * 2.001),
            ("mhc_norm", lambda a: a + 1.0, lambda a: a + 1.0),
        ),
        shadow_input=np.array([1.0, 2.0], dtype=np.float32),
    )
    assert res_ok.committed
    assert res_ok.recaptured_graphs
    assert wb.tx.active_version == 2
    assert wb.run_step(8, 3.0) == pytest.approx(6.003)

    # 2. Divergent `--shadow` operator pinpoints exact failing stage and rolls back atomically
    shadow_only = run_shadow_differential(
        (
            ("op0_clean", lambda a: a + 1.0, lambda a: a + 1.0),
            ("op1_buggy", lambda a: a * 2.0, lambda a: a * 9.0),
            ("op2_clean", lambda a: a - 0.5, lambda a: a - 0.5),
        ),
        np.array([1.0, 2.0]),
    )
    assert not shadow_only.passed
    assert shadow_only.first_divergent_stage == 1
    assert shadow_only.first_divergent_op == "op1_buggy"

    res_drift = wb.hot_apply(
        candidate_id="cand-drift",
        changed_files=("worker/zstep_worker/kernels_sm120/b12x_dsv41.py",),
        mutation_fn=lambda eng: eng.bind_step_runner(
            "models.runner", object(), lambda rows, x: x * 99.0
        ),
        shadow_stages=(("op1_buggy", lambda a: a * 2.0, lambda a: a * 99.0),),
        shadow_input=np.array([1.0, 2.0]),
    )
    assert not res_drift.committed and res_drift.rolled_back
    assert res_drift.failure_kind == "shadow_drift"
    assert wb.tx.active_version == 2
    assert wb.run_step(8, 3.0) == pytest.approx(6.003)

    # 3. Hung candidate trips watchdog timeout and rolls back cleanly
    res_hang = wb.hot_apply(
        candidate_id="cand-hang",
        changed_files=("worker/zstep_worker/models/deepseek_v41.py",),
        mutation_fn=lambda _eng: time.sleep(0.25),
        watchdog_timeout_sec=0.04,
    )
    assert not res_hang.committed and res_hang.rolled_back
    assert res_hang.failure_kind == "timeout_watchdog"
    assert wb.run_step(8, 3.0) == pytest.approx(6.003)

    # 4. Pinned file edit records an EnvelopeGap without mutating engine
    res_gap = wb.hot_apply(
        candidate_id="cand-wire",
        changed_files=("crates/zs-wire/src/lib.rs",),
    )
    assert res_gap.failure_kind == "envelope_gap"
    assert len(env_log.list_gaps()) == 1

    # 5. Bottleneck ranking + Graduation bundle catches 138-LOC Occam penalty vs 115-LOC pass!
    ledger = StepBottleneckLedger(
        (
            StepOperatorSample(
                op_name="shm_v5_ar_plus_mhc",
                total_step_us=4070.0,
                niche_key="(plane=l1_kernel,op=ar_mhc)",
            ),
            StepOperatorSample(
                op_name="wire_header_copy",
                total_step_us=5000.0,
                supported=False,
                boundary_note="Pinned crates/zs-wire/src/lib.rs",
            ),
        )
    )
    sup, gaps = ledger.rank_targets()
    assert sup[0].op_name == "shm_v5_ar_plus_mhc"
    assert gaps[0].op_name == "wire_header_copy"

    grad_138 = wb.prepare_graduation_bundle(
        candidate_id="g0010",
        inner_screen_gain=0.0322,
        changed_lines=138,
    )
    assert not grad_138.ready_for_cold_verify
    assert grad_138.estimated_net_gate_ratio == pytest.approx(1.0286)

    grad_115 = wb.prepare_graduation_bundle(
        candidate_id="g0011",
        inner_screen_gain=0.0322,
        changed_lines=115,
    )
    assert grad_115.ready_for_cold_verify
    assert grad_115.estimated_net_gate_ratio == pytest.approx(1.0322)
