"""Portability & correctness tests for Evolith's multi-GPU RSI primitives (`evolith.core`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolith.core import (
    BackendCellMeasurement,
    HardwareTopologySnapshot,
    InterconnectTier,
    LossyQualityGateSpec,
    MultiRegimeLedger,
    PiecewisePolicySynthesizer,
    PrecisionTrack,
    RegimeOperatorShare,
    StateResourceDecl,
    TargetSurface,
    UnitDecl,
    WorkloadRegime,
    are_patches_closure_orthogonal,
    audit_operator_dispatch,
)


def _pair_p2p_matrix(world_size: int = 8) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple((i == j) or ((i ^ 1) == j) for j in range(world_size))
        for i in range(world_size)
    )


def _full_mesh_matrix(world_size: int = 8) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(True for _ in range(world_size)) for _ in range(world_size))


def test_hardware_topology_classifies_sm120_sm90_sm100_and_rocm() -> None:
    sm120 = HardwareTopologySnapshot(
        vendor="nvidia",
        arch_tag="sm120",
        device_name="NVIDIA RTX 6000D",
        world_size=8,
        compute_units_per_device=156,
        l2_cache_bytes=112 * 1024 * 1024,
        persisting_l2_cache_max_bytes=70 * 1024 * 1024,
        shared_mem_per_block_optin_bytes=99 * 1024,
        vram_bytes=85 * 1024**3,
        numa_islands=((0, 1, 2, 3), (4, 5, 6, 7)),
        peer_p2p_matrix=_pair_p2p_matrix(8),
        has_dma_copy_engine=True,
        high_speed_fabric=False,
        toolchain_version="cuda-13.0",
    )
    assert not sm120.all_to_all_p2p_supported()
    assert sm120.p2p_connected_components() == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert sm120.classify_interconnect() == InterconnectTier.PAIR_P2P_HOST_BRIDGED
    assert sm120.slow_ring_hops_mask() == ((1 << 3) | (1 << 7))
    notes = " ".join(sm120.collective_guidance_notes())
    assert "All-to-all BAR1 IPC is unavailable" in notes
    assert "Copy Engine" in notes
    assert "Per-block opt-in shared memory = 99 KiB" in notes
    assert "Persisting L2 cache window cap = 70.0 MiB" in notes

    sm100 = HardwareTopologySnapshot(
        vendor="nvidia",
        arch_tag="sm100",
        device_name="NVIDIA B200",
        world_size=8,
        compute_units_per_device=148,
        numa_islands=((0, 1, 2, 3), (4, 5, 6, 7)),
        peer_p2p_matrix=_full_mesh_matrix(8),
        high_speed_fabric=True,
        toolchain_version="cuda-13.4",
    )
    assert sm100.all_to_all_p2p_supported()
    assert sm100.classify_interconnect() == InterconnectTier.FULL_MESH_FABRIC

    rocm = HardwareTopologySnapshot(
        vendor="amd",
        arch_tag="gfx942",
        device_name="AMD Instinct MI300X",
        world_size=8,
        compute_units_per_device=304,
        peer_p2p_matrix=_full_mesh_matrix(8),
        high_speed_fabric=True,
        toolchain_version="rocm-6.3",
    )
    assert rocm.classify_interconnect() == InterconnectTier.FULL_MESH_FABRIC
    assert rocm.compute_units_per_device == 304


def test_dispatch_audit_catches_156sm_table_miss_fallback_and_p2p_violation() -> None:
    topo = HardwareTopologySnapshot(
        vendor="nvidia",
        arch_tag="sm120",
        device_name="NVIDIA RTX 6000D",
        world_size=8,
        compute_units_per_device=156,
        peer_p2p_matrix=_pair_p2p_matrix(8),
    )
    report = audit_operator_dispatch(
        topo,
        [
            {
                "op_name": "cake_sparse_mla_dsv41_mixed",
                "selected_tactic": "default",
                "tuned_compute_units": (48, 170, 188),
            },
            {
                "op_name": "sm12x_mega_gate",
                "selected_tactic": "splitk_cost_110sm",
                "tuned_compute_units": (110,),
            },
            {
                "op_name": "dsa_indexer_prefill",
                "selected_tactic": "orig_forward_cuda",
                "is_fallback_route": True,
            },
            {
                "op_name": "pcie_ipc_ar_tp8",
                "selected_tactic": "COPY_ENGINE_ISLAND",
                "requires_all_to_all_p2p": True,
            },
            {
                "op_name": "b12x_moe_ffn",
                "selected_tactic": "w4a8_156sm_tuned",
                "tuned_compute_units": (156, 170, 188),
            },
        ],
    )
    assert report.total_audited_ops == 5
    assert report.optimal_hits == 1
    severities = {i.severity for i in report.issues}
    assert severities == {"fallback_route", "compute_unit_mismatch", "interconnect_mismatch"}


def test_multi_regime_ledger_balances_decode_and_prefill_bottlenecks() -> None:
    regimes = (
        WorkloadRegime(regime_id="decode_m1", phase="decode", batch_tokens=1, seq_len=4096, weight=1.0),
        WorkloadRegime(regime_id="decode_m32", phase="decode", batch_tokens=32, seq_len=4096, weight=1.0),
        WorkloadRegime(regime_id="prefill_16k", phase="prefill", batch_tokens=2048, seq_len=16384, weight=1.0),
    )
    entries = (
        RegimeOperatorShare(regime_id="decode_m1", op_name="comm_allreduce", step_us=1240, total_regime_step_us=7040),
        RegimeOperatorShare(regime_id="decode_m32", op_name="comm_allreduce", step_us=2550, total_regime_step_us=11510),
        RegimeOperatorShare(regime_id="prefill_16k", op_name="comm_allreduce", step_us=85000, total_regime_step_us=110000),
        RegimeOperatorShare(regime_id="decode_m1", op_name="mla_dsa_attn", step_us=2170, total_regime_step_us=7040),
        RegimeOperatorShare(regime_id="decode_m32", op_name="mla_dsa_attn", step_us=2920, total_regime_step_us=11510),
        RegimeOperatorShare(regime_id="prefill_16k", op_name="mla_dsa_attn", step_us=12000, total_regime_step_us=110000),
    )
    ledger = MultiRegimeLedger(regimes, entries)
    supported, gaps = ledger.rank_targets()
    assert not gaps
    assert supported[0].op_name == "comm_allreduce"
    assert supported[0].dominant_regime_id == "prefill_16k"
    assert supported[1].op_name == "mla_dsa_attn"


def test_piecewise_synthesizer_prevents_crossover_regressions_and_blocks_lossy_on_lossless_track() -> None:
    # Mirrors real RTX 6000D GDN / MoE / AllReduce crossover curves:
    # candidate wins at B=1 and B>=1024, but regresses at B=4..8; lossy candidate is faster but blocked!
    synth = PiecewisePolicySynthesizer(
        incumbent_backend="incumbent",
        allowed_track=PrecisionTrack.LOSSLESS,
        min_speedup_guardband=1.02,
    )
    cells = [
        BackendCellMeasurement(backend_id="incumbent", phase="prefill", batch_tokens=1, seq_len=4096, latency_us=400.0),
        BackendCellMeasurement(backend_id="sm120_cute", phase="prefill", batch_tokens=1, seq_len=4096, latency_us=320.0),
        BackendCellMeasurement(backend_id="incumbent", phase="prefill", batch_tokens=4, seq_len=4096, latency_us=760.0),
        BackendCellMeasurement(backend_id="sm120_cute", phase="prefill", batch_tokens=4, seq_len=4096, latency_us=1160.0),
        BackendCellMeasurement(backend_id="incumbent", phase="prefill", batch_tokens=8, seq_len=4096, latency_us=1910.0),
        BackendCellMeasurement(backend_id="sm120_cute", phase="prefill", batch_tokens=8, seq_len=4096, latency_us=2140.0),
        BackendCellMeasurement(
            backend_id="nvfp4_lossy",
            phase="prefill",
            batch_tokens=8,
            seq_len=4096,
            latency_us=1200.0,
            precision_track=PrecisionTrack.CALIBRATED_LOSSY,
        ),
        BackendCellMeasurement(backend_id="incumbent", phase="prefill", batch_tokens=16, seq_len=4096, latency_us=5490.0),
        BackendCellMeasurement(backend_id="sm120_cute", phase="prefill", batch_tokens=16, seq_len=4096, latency_us=4570.0),
    ]
    plan = synth.synthesize(cells)
    assert plan.upgraded_cells == 2
    assert plan.resolve(phase="prefill", batch_tokens=1, seq_len=4096) == "sm120_cute"
    assert plan.resolve(phase="prefill", batch_tokens=4, seq_len=4096) == "incumbent"
    assert plan.resolve(phase="prefill", batch_tokens=8, seq_len=4096) == "incumbent"
    assert plan.resolve(phase="prefill", batch_tokens=16, seq_len=4096) == "sm120_cute"
    assert all(r.worst_cell_speedup_vs_incumbent >= 1.0 for r in plan.rules)
    assert any("lossy_track_blocked" in msg for msg in plan.rejected_candidates)


def test_lossy_quality_gate_spec_and_declarative_surface_portability() -> None:
    gate = LossyQualityGateSpec(max_top1_kl_divergence=0.015, max_ppl_delta=0.04)
    ok, reasons = gate.evaluate(measured_kl=0.008, measured_ppl_delta=0.02, measured_retrieval_acc=0.99)
    assert ok and not reasons
    ok_bad, reasons_bad = gate.evaluate(measured_kl=0.03, measured_ppl_delta=0.02)
    assert not ok_bad and "KL divergence" in reasons_bad[0]

    # Verify a non-zstep ROCm target surface works with custom declarative resource patterns
    rocm_surface = TargetSurface(
        target_name="rocm_engine",
        allowed_roots=("kernels_hip", "runner"),
        units=(
            UnitDecl(unit_id="hip.comm", kind="native_op", path_prefixes=("kernels_hip",), hotswap_extern_c=True),
            UnitDecl(unit_id="py.runner", kind="logic", path_prefixes=("runner",)),
        ),
        state_resources=(
            StateResourceDecl(
                name="xgmi_ipc_slab",
                budget_bytes=512 * 1024 * 1024,
                patch_patterns=(r"\bhipIpcGetMemHandle\b|\bxgmi_slab_bytes\b",),
            ),
        ),
    )
    orthogonal, reason = are_patches_closure_orthogonal(
        rocm_surface,
        left_files=("kernels_hip/allreduce.cpp",),
        right_files=("runner/step.py",),
        left_patch_text="+ xgmi_slab_bytes = 256 * 1024 * 1024",
        right_patch_text="+ hipIpcGetMemHandle(&h, ptr)",
    )
    assert not orthogonal
    assert "xgmi_ipc_slab" in reason


def test_regime_benchmark_profiles_env_overrides_and_security(tmp_path: Path) -> None:
    from evolith.zstep.remote import ZStepRemoteRunner
    from evolith.zstep.task import ZStepRSITask

    t_long = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        remote_host="gpu-host",
        benchmark_profile="longctx-tp8",
    )
    assert t_long.max_pages_per_seq == 264
    assert t_long.num_kv_pages == 2560
    assert t_long.worker_env_overrides == {"ZSTEP_MAX_MODEL_LEN": "33792"}

    t_spec = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        remote_host="gpu-host",
        benchmark_profile="spec-dspark",
    )
    assert t_spec.worker_env_overrides == {
        "ZSTEP_SPEC_METHOD": "dspark",
        "ZSTEP_NUM_SPEC_TOKENS": "5",
    }
    script = ZStepRemoteRunner(t_spec).benchmark_script(
        "runs/g0001/screen/src",
        "runs/g0001/screen/metrics.json",
        t_spec.screen_tiers,
        1,
        None,
    )
    assert "ZSTEP_SPEC_METHOD=dspark" in script
    assert "ZSTEP_NUM_SPEC_TOKENS=5" in script
    assert "$HOME/evolith-rsi/container-cache:/root/.cache" in script
    assert "peak_vram_used_mib" in script

    with pytest.raises(ValueError, match="Forbidden"):
        ZStepRSITask(
            project_root=tmp_path,
            source_repo=tmp_path / "src",
            worker_env_overrides={"LD_PRELOAD": "/tmp/evil.so"},
        )


def test_min_screen_gain_and_hot_prescreen_fast_reject_before_cold_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.gates import evaluate_screen
    from evolith.zstep.loop import ZStepRSILoop
    from evolith.zstep.task import CandidateStatus, ZStepRSITask

    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    for p, tp in ((base_file, 1000.0), (cand_file, 997.0)):  # -0.30%
        p.write_text(
            json.dumps(
                {
                    "tiers": [
                        {
                            "label": "in512_c1",
                            "repeat": 0,
                            "output_throughput_tok_s": tp,
                            "median_ttft_ms": 50.0,
                            "mean_tpot_ms": 10.0,
                            "p99_itl_ms": 15.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    task_with_floor = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        min_screen_gain=0.005,
    )
    dec = evaluate_screen(cand_file, None, task_with_floor, base_file, require_parity=False)
    assert not dec.passed
    assert any("min_screen_gain" in r for r in dec.reasons)

    # Now test 20s hot_prescreen_script fast-rejecting before screen/verify
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)
    probe_py = tmp_path / "hot_probe.py"
    probe_py.write_text("print('probe')\n", encoding="utf-8")

    class _Ws:
        def __init__(self, t: ZStepRSITask) -> None:
            self.task = t

        def create_candidate(self, cid: str, patches: list[bytes]) -> Path:
            d = self.task.candidates_dir / cid
            d.mkdir(parents=True)
            (d / "x.txt").write_text("v1", encoding="utf-8")
            return d

        def capture_patch(self, d: Path) -> bytes:
            (d / "x.txt").write_text("v2", encoding="utf-8")
            return b"diff"

        def freeze_source(self, d: Path, dst: Path) -> Path:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((d / "x.txt").read_bytes())
            return dst

    class _Prop:
        def propose(self, *args: object) -> str:
            return 'EVOLITH_PROPOSAL_JSON: {"hypothesis":"h","niche_key":"(plane=l1_kernel,focus=x)","patch_summary":"s"}'

    class _Rem:
        def __init__(self, speedup: float, sec_cand_tp: float = 100.2) -> None:
            self.speedup = speedup
            self.sec_cand_tp = sec_cand_tp
            self.screen_calls = 0
            self.verify_calls = 0
            self.task: ZStepRSITask | None = None

        def preflight(self) -> None:
            pass

        def run_probe(self, probe_id: str, archive: Path, script: Path, out: Path) -> Path:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "geomean_speedup": self.speedup,
                        "summary": f"speedup={self.speedup:.3f}x",
                    }
                ),
                encoding="utf-8",
            )
            return out

        def screen_baseline(self, bh: str, archive: Path) -> Path:
            p = tmp_path / f"sb-{bh}.json"
            lbl = "ctx8k_c2" if self.task and self.task.benchmark_profile == "longctx-tp8" else "in512_c1"
            p.write_text(
                json.dumps(
                    {
                        "tiers": [
                            {
                                "label": lbl,
                                "repeat": 0,
                                "output_throughput_tok_s": 100.0,
                                "median_ttft_ms": 50.0,
                                "mean_tpot_ms": 10.0,
                                "p99_itl_ms": 15.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return p

        def screen(self, cid: str, archive: Path, ev: Path) -> Path:
            self.screen_calls += 1
            is_sec = self.task is not None and self.task.benchmark_profile == "short-throughput"
            lbl = "in512_c1" if is_sec else "ctx8k_c2"
            tp = self.sec_cand_tp if is_sec else 112.0
            (ev / "screen-parity.json").write_text(
                json.dumps(
                    {
                        "triton_recompilations": 0,
                        "cases": [
                            {
                                "name": "prime_squares_table",
                                "single_match": {"exact_token_match_rate": 1.0},
                                "batched_match": {"exact_token_match_rate": 1.0},
                                "chunked_match": {"exact_token_match_rate": 1.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = ev / "screen-metrics.json"
            out.write_text(
                json.dumps(
                    {
                        "tiers": [
                            {
                                "label": lbl,
                                "repeat": 0,
                                "output_throughput_tok_s": tp,
                                "median_ttft_ms": 50.0,
                                "mean_tpot_ms": 10.0,
                                "p99_itl_ms": 15.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return out

        def verify(self, cid: str, ba: Path, ca: Path, ev: Path) -> tuple[Path, Path, Path]:
            self.verify_calls += 1
            for name, tp in (("baseline.json", 100.0), ("candidate.json", 110.0)):
                (ev / name).write_text(
                    json.dumps(
                        {
                            "tiers": [
                                {
                                    "label": "ctx8k_c16",
                                    "repeat": 0,
                                    "output_throughput_tok_s": tp,
                                    "median_ttft_ms": 50.0,
                                    "mean_tpot_ms": 10.0,
                                    "p99_itl_ms": 15.0,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            par = ev / "parity.json"
            par.write_text(
                json.dumps(
                    {
                        "triton_recompilations": 0,
                        "cases": [
                            {
                                "name": "prime_squares_table",
                                "single_match": {"exact_token_match_rate": 1.0},
                                "batched_match": {"exact_token_match_rate": 1.0},
                                "chunked_match": {"exact_token_match_rate": 1.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return ev / "candidate.json", ev / "baseline.json", par

    t_slow = ZStepRSITask(
        project_root=tmp_path / "r1",
        source_repo=tmp_path / "src",
        benchmark_profile="longctx-tp8",
        hot_prescreen_script=probe_py,
        min_hot_prescreen_speedup=1.05,
        secondary_benchmark_profiles=("short-throughput",),
    )
    rem_slow = _Rem(speedup=0.99)
    rem_slow.task = t_slow
    loop_slow = ZStepRSILoop(
        t_slow,
        workspace=_Ws(t_slow),
        archive=ZStepArchive(t_slow.state_dir / "archive.sqlite3"),
        proposer=_Prop(),
        remote=rem_slow,
    )
    assert loop_slow.run() == [CandidateStatus.SCREEN_REJECTED]
    assert rem_slow.screen_calls == 0
    assert rem_slow.verify_calls == 0

    # Fast candidate passes hot_prescreen (1.20x), passes longctx-tp8 (+10%), AND passes neutral short-throughput (+0.2%)
    t_good = ZStepRSITask(
        project_root=tmp_path / "r2",
        source_repo=tmp_path / "src",
        benchmark_profile="longctx-tp8",
        hot_prescreen_script=probe_py,
        min_hot_prescreen_speedup=1.05,
        secondary_benchmark_profiles=("short-throughput",),
    )
    rem_good = _Rem(speedup=1.20, sec_cand_tp=100.2)
    rem_good.task = t_good
    loop_good = ZStepRSILoop(
        t_good,
        workspace=_Ws(t_good),
        archive=ZStepArchive(t_good.state_dir / "archive.sqlite3"),
        proposer=_Prop(),
        remote=rem_good,
    )
    assert loop_good.run() == [CandidateStatus.ACCEPTED]
    assert rem_good.screen_calls == 2  # primary + secondary guard
    assert rem_good.verify_calls == 1

    # Candidate that wins on longctx (+10%) but regresses short-throughput (-5%) is rejected by secondary guard!
    t_reg = ZStepRSITask(
        project_root=tmp_path / "r3",
        source_repo=tmp_path / "src",
        benchmark_profile="longctx-tp8",
        hot_prescreen_script=probe_py,
        min_hot_prescreen_speedup=1.05,
        secondary_benchmark_profiles=("short-throughput",),
    )
    rem_reg = _Rem(speedup=1.20, sec_cand_tp=95.0)
    rem_reg.task = t_reg
    loop_reg = ZStepRSILoop(
        t_reg,
        workspace=_Ws(t_reg),
        archive=ZStepArchive(t_reg.state_dir / "archive.sqlite3"),
        proposer=_Prop(),
        remote=rem_reg,
    )
    assert loop_reg.run() == [CandidateStatus.VERIFY_REJECTED]


def test_pareto_vram_tail_latency_guards_and_speculative_regime_rows(tmp_path: Path) -> None:
    import json
    from evolith.zstep.gates import evaluate_candidate, evaluate_screen, load_benchmark
    from evolith.zstep.task import ZStepRSITask

    # 1. Speculative WorkloadRegime physical row modeling (P < B * W under adaptive confidence pruning)
    reg_full = WorkloadRegime(
        regime_id="spec_b8_full",
        phase="speculative",
        batch_tokens=48,
        batch_seqs=8,
        draft_depth=5,
        verify_width=6,
    )
    assert reg_full.effective_physical_rows() == 48
    reg_pruned = WorkloadRegime(
        regime_id="spec_b8_pruned",
        phase="speculative",
        batch_tokens=34,
        batch_seqs=8,
        draft_depth=7,
        verify_width=8,
        physical_rows=34,
    )
    assert reg_pruned.effective_physical_rows() == 34

    # 2. PiecewisePolicySynthesizer Pareto VRAM & p99_itl_ms rejection
    synth = PiecewisePolicySynthesizer(
        incumbent_backend="base_dspark",
        min_speedup_guardband=1.02,
        max_vram_used_mib=80000.0,
        max_p99_itl_ms=25.0,
    )
    plan = synth.synthesize(
        [
            BackendCellMeasurement(
                backend_id="base_dspark",
                phase="speculative",
                batch_tokens=6,
                seq_len=1024,
                latency_us=5000.0,
                vram_used_mib=84950.0,
                p99_itl_ms=12.0,
            ),
            BackendCellMeasurement(
                backend_id="oom_risk_cand",
                phase="speculative",
                batch_tokens=6,
                seq_len=1024,
                latency_us=3800.0,
                vram_used_mib=84900.0,
                p99_itl_ms=10.0,
            ),
            BackendCellMeasurement(
                backend_id="spiky_itl_cand",
                phase="speculative",
                batch_tokens=6,
                seq_len=1024,
                latency_us=3900.0,
                vram_used_mib=73100.0,
                p99_itl_ms=42.0,
            ),
            BackendCellMeasurement(
                backend_id="async_engram_dspark",
                phase="speculative",
                batch_tokens=6,
                seq_len=1024,
                latency_us=4100.0,
                vram_used_mib=73100.0,
                p99_itl_ms=11.0,
            ),
        ]
    )
    assert plan.resolve(phase="speculative", batch_tokens=6, seq_len=1024) == "async_engram_dspark"
    assert any("vram_budget_exceeded" in r for r in plan.rejected_candidates)
    assert any("p99_itl_exceeded" in r for r in plan.rejected_candidates)

    # 3. cold-prefill preset & ZStepRSITask / gates Pareto VRAM + p99_itl_ms enforcement
    t_cold = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        benchmark_profile="cold-prefill",
        max_vram_used_mib=80000.0,
        max_p99_itl_ms=30.0,
    )
    assert "prefill32k_v" in t_cold.verify_tiers
    base_json = tmp_path / "b.json"
    cand_vram_bad = tmp_path / "c_vram.json"
    cand_ok = tmp_path / "c_ok.json"
    parity_json = tmp_path / "p.json"
    parity_json.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "cases": [
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": 1.0},
                        "batched_match": {"exact_token_match_rate": 1.0},
                        "chunked_match": {"exact_token_match_rate": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    base_json.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "spec_c2",
                        "repeat": 0,
                        "output_throughput_tok_s": 300.0,
                        "median_ttft_ms": 40.0,
                        "mean_tpot_ms": 5.0,
                        "p99_itl_ms": 12.0,
                        "peak_vram_used_mib": 84950.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cand_vram_bad.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "spec_c2",
                        "repeat": 0,
                        "output_throughput_tok_s": 420.0,
                        "median_ttft_ms": 38.0,
                        "mean_tpot_ms": 3.5,
                        "p99_itl_ms": 11.0,
                        "peak_vram_used_mib": 84500.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cand_ok.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "spec_c2",
                        "repeat": 0,
                        "output_throughput_tok_s": 420.0,
                        "median_ttft_ms": 38.0,
                        "mean_tpot_ms": 3.5,
                        "p99_itl_ms": 11.0,
                        "peak_vram_used_mib": 73150.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert not evaluate_screen(cand_vram_bad, None, t_cold, base_json).passed
    v_bad = evaluate_candidate(load_benchmark(base_json), load_benchmark(cand_vram_bad), parity_json, t_cold)
    assert not v_bad.accepted and any("max_vram_used_mib" in r for r in v_bad.reasons)
    v_ok = evaluate_candidate(load_benchmark(base_json), load_benchmark(cand_ok), parity_json, t_cold)
    assert v_ok.accepted


def test_green_context_topology_and_cold_prefill_step_tokens_and_ttft_gates(tmp_path: Path) -> None:
    import json
    from evolith.zstep.gates import evaluate_candidate, evaluate_screen, load_benchmark
    from evolith.zstep.remote import ZStepRemoteRunner
    from evolith.zstep.task import ZStepRSITask

    topo = HardwareTopologySnapshot(
        vendor="nvidia",
        arch_tag="sm120",
        device_name="NVIDIA RTX 6000D",
        world_size=8,
        compute_units_per_device=156,
        peer_p2p_matrix=_pair_p2p_matrix(8),
        green_context_supported=True,
        min_sm_partition_size=8,
        sm_coscheduled_alignment=8,
        toolchain_version="cuda-13.0",
    )
    notes = " ".join(topo.collective_guidance_notes())
    assert "min_partition=8" in notes
    assert "coscheduled_align=8" in notes

    reg = WorkloadRegime(
        regime_id="cold_prefill_16k",
        phase="prefill",
        batch_tokens=8192,
        seq_len=16384,
        prefill_chunk_tokens=8192,
        comm_green_ctx_sms=8,
    )
    assert reg.prefill_chunk_tokens == 8192
    assert reg.comm_green_ctx_sms == 8

    synth = PiecewisePolicySynthesizer(
        incumbent_backend="decode_reuse_mla",
        min_speedup_guardband=1.02,
        max_ttft_ms=900.0,
    )
    plan = synth.synthesize(
        [
            BackendCellMeasurement(
                backend_id="decode_reuse_mla",
                phase="prefill",
                batch_tokens=4096,
                seq_len=16384,
                latency_us=18000.0,
                ttft_ms=1450.0,
            ),
            BackendCellMeasurement(
                backend_id="slow_ttft_cand",
                phase="prefill",
                batch_tokens=4096,
                seq_len=16384,
                latency_us=12000.0,
                ttft_ms=1100.0,
            ),
            BackendCellMeasurement(
                backend_id="b12x_mg_prefill",
                phase="prefill",
                batch_tokens=4096,
                seq_len=16384,
                latency_us=11500.0,
                ttft_ms=720.0,
            ),
        ]
    )
    assert plan.resolve(phase="prefill", batch_tokens=4096, seq_len=16384) == "b12x_mg_prefill"
    assert any("ttft_exceeded" in r for r in plan.rejected_candidates)

    t_cold = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        remote_host="gpu-host",
        benchmark_profile="cold-prefill",
        max_ttft_ms=1200.0,
    )
    assert t_cold.max_step_tokens == 8192
    script = ZStepRemoteRunner(t_cold).benchmark_script(
        "runs/g0001/screen/src",
        "runs/g0001/screen/metrics.json",
        t_cold.screen_tiers,
        1,
        None,
    )
    assert "--max-step-tokens 8192 --max-decoder-rows 8192" in script

    base_f = tmp_path / "base_cp.json"
    cand_bad_ttft = tmp_path / "cand_bad_ttft.json"
    cand_good_ttft = tmp_path / "cand_good_ttft.json"
    parity_f = tmp_path / "parity_cp.json"
    parity_f.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "cases": [
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": 1.0},
                        "batched_match": {"exact_token_match_rate": 1.0},
                        "chunked_match": {"exact_token_match_rate": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for p, tp, ttft in ((base_f, 40.0, 1500.0), (cand_bad_ttft, 55.0, 1300.0), (cand_good_ttft, 60.0, 950.0)):
        p.write_text(
            json.dumps(
                {
                    "tiers": [
                        {
                            "label": "prefill16k_v",
                            "repeat": 0,
                            "output_throughput_tok_s": tp,
                            "median_ttft_ms": ttft,
                            "mean_tpot_ms": 9.0,
                            "p99_itl_ms": 11.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    assert not evaluate_screen(cand_bad_ttft, None, t_cold, base_f).passed
    dec_bad = evaluate_candidate(load_benchmark(base_f), load_benchmark(cand_bad_ttft), parity_f, t_cold)
    assert not dec_bad.accepted and any("max_ttft_ms" in r for r in dec_bad.reasons)
    dec_ok = evaluate_candidate(load_benchmark(base_f), load_benchmark(cand_good_ttft), parity_f, t_cold)
    assert dec_ok.accepted
    assert "TTFT 950.0 vs 1500.0ms" in dec_ok.metric_delta


