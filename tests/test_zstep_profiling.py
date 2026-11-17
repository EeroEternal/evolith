from __future__ import annotations

import json
from pathlib import Path

from evolith.zstep.profiling import build_sm120_reflection_card


def test_reflection_card_extracts_sm120_operator_and_screen_telemetry(
    tmp_path: Path,
) -> None:
    bench_dir = tmp_path / "docs/benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "phase37_6000d_operator_audit.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "device_name": "NVIDIA RTX 6000D",
                    "compute_capability": "12.0",
                    "multi_processor_count": 156,
                    "torch_version": "2.13.0+cu130",
                },
                "block32_fp8_and_session": {
                    "shapes": {
                        "fused_wqa_wkv": {
                            "cg_us": {
                                "1": {"b12x_us": 10.27},
                                "32": {"b12x_us": 10.86},
                                "128": {"b12x_us": 16.92},
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    screen_path = tmp_path / "base-screen.json"
    screen_path.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c1",
                        "output_throughput_tok_s": 88.7,
                        "median_ttft_ms": 167.9,
                        "mean_tpot_ms": 8.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    card = build_sm120_reflection_card(tmp_path, screen_path)

    assert "NVIDIA RTX 6000D" in card
    assert "fused_wqa_wkv" in card
    assert "in512_c1: out=88.7 tok/s" in card
    assert "Coordinator Contract Reminders" in card


def test_reflection_card_extracts_full_subsystem_microbenches_overlap_and_frozen_niches(
    tmp_path: Path,
) -> None:
    bench_dir = tmp_path / "docs/benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "step_03_moe_mxfp4_sm120.json").write_text(
        json.dumps(
            {
                "router": {"with_correction_bias": {"6": {"latency_us": 5.74}}},
                "moe": {
                    "tp8_local_rank": {
                        "results": {
                            "6": {"latency_us": 70.24, "effective_gbps": 1172.7}
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (bench_dir / "step_04_dsv41_attention_sm120.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"mode": "decode", "batch": 1, "kv_len": 8192, "latency_us": 13.92},
                    {"mode": "extend", "rows": 4096, "kv_len": 8192, "latency_us": 1429.28},
                ]
            }
        ),
        encoding="utf-8",
    )
    (bench_dir / "step_05a_mhc_engram_sm120.json").write_text(
        json.dumps(
            {
                "mhc": {"results": [{"tokens": 6, "latency_us": 11.41}]},
                "engram": {
                    "results": [
                        {
                            "results": {
                                "48": {
                                    "pinned_latency_us": 20.03,
                                    "vram_latency_us": 3.65,
                                    "pinned_over_vram_latency": 5.48,
                                }
                            }
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (bench_dir / "step_05b_pcie_allreduce_8gpu.json").write_text(
        json.dumps(
            {
                "host_mapped_shm": {"results": {"6": {"80_call_budget_ms": 5.23}}},
                "nccl": {"results": {"6": {"80_call_budget_ms": 4.20}}},
            }
        ),
        encoding="utf-8",
    )
    screen_path = tmp_path / "base-screen-2tier.json"
    screen_path.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c32",
                        "output_throughput_tok_s": 1120.4,
                        "median_ttft_ms": 410.0,
                        "mean_tpot_ms": 24.5,
                        "steps": 500,
                        "overlapped_decode_steps": 450,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    card = build_sm120_reflection_card(
        tmp_path,
        screen_path,
        blocked_niches={"(plane=l2_scheduler,focus=decode_h2d_pack)": ["gain -0.1%", "gain -0.2%"]},
    )

    assert "MXFP4_MoE_TP8(T6=70.24us@1172.7GB/s)" in card
    assert "decode_B1@8k=13.92us" in card
    assert "Engram_L1(T48:pinned=20.03us/vram=3.65us(5.48x))" in card
    assert "T6:shm=5.23ms vs nccl=4.20ms" in card
    assert "overlap=450/500(90%)" in card
    assert "BLOCKED `(plane=l2_scheduler,focus=decode_h2d_pack)`" in card
    assert "measured telemetry above" in card


def test_reflection_card_diagnoses_high_tau_wallclock_divergence(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c1",
                        "output_throughput_tok_s": 118.6,
                        "median_ttft_ms": 168.0,
                        "mean_tpot_ms": 8.43,
                    },
                    {
                        "label": "in512_c32",
                        "output_throughput_tok_s": 840.0,
                        "median_ttft_ms": 380.0,
                        "mean_tpot_ms": 33.3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cand_stalled = tmp_path / "spec-stalled.json"
    cand_stalled.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c1",
                        "output_throughput_tok_s": 119.1,
                        "median_ttft_ms": 170.0,
                        "mean_tpot_ms": 8.40,
                        "mean_acceptance_length": 3.53,
                        "prefix_cache_hit_rate": 0.493,
                    },
                    {
                        "label": "in512_c32",
                        "output_throughput_tok_s": 960.0,
                        "median_ttft_ms": 380.0,
                        "mean_tpot_ms": 29.0,
                        "mean_acceptance_length": 2.40,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    card = build_sm120_reflection_card(
        tmp_path, base_path, candidate_screen_metrics=cand_stalled
    )
    assert "[Algorithmic-vs-Wall-Clock Divergence]" in card
    assert "in512_c1(tau=3.53, speedup=1.00x)" in card
    # c32 achieved 1.14x >= 1.05x and is NOT falsely flagged despite TPOT=29.0ms > 8ms
    assert "in512_c32(tau=2.40" not in card
    assert "segmented draft/verify CUDAGraphs" in card
