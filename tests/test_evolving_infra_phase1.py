from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from evolith.planes.l2_policies import BASELINE_L2_POLICY_CODE, L2_POLICY_MUTATIONS
from evolith.verifiers.dream_simulator import DreamTraceSimulator
from evolith.zstep.agent_replay import (
    AgentTraceEvent,
    compare_eviction_policies,
    validate_agent_replay_run,
)
from evolith.zstep.gates import compute_paired_round_stats, evaluate_candidate
from evolith.zstep.perf_model import (
    DecodeStepSample,
    IterationTimeModel,
    PrefillStepSample,
    attention_pairs,
    expert_fraction,
    fit_steps,
)
from evolith.zstep.remote import ZStepRemoteRunner
from evolith.zstep.task import TierMetric, ZStepRSITask


def _load_scrub_module():
    path = Path(__file__).resolve().parents[1] / "tools/scrub_transcript.py"
    spec = importlib.util.spec_from_file_location("scrub_transcript", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nnls_iteration_time_model_fits_dsv4_and_calibrates_dream_sim() -> None:
    rng = np.random.default_rng(42)
    true_c = (0.0045, 4.2e-5, 1.8e-9, 6.0e-6)
    true_d = (0.0068, 0.0185, 0.00022)

    prefill_samples: list[PrefillStepSample] = []
    for _ in range(36):
        uc = (int(rng.integers(128, 3072)), int(rng.integers(64, 1024)))
        tc = (int(rng.integers(0, 32768)), int(rng.integers(0, 16384)))
        rst = int(rng.integers(0, 8192))
        sum_a = float(attention_pairs(uc, tc).sum())
        dur = (
            true_c[0]
            + true_c[1] * sum(uc)
            + true_c[2] * sum_a
            + true_c[3] * rst
        ) * float(rng.uniform(0.99, 1.01))
        prefill_samples.append(PrefillStepSample(uc, tc, rst, dur))

    decode_samples: list[DecodeStepSample] = []
    for b in list(range(1, 33)) + [48, 64, 96, 128]:
        tfwd = float(rng.choice([1.0, 4.0, 6.0]))
        phi = float(expert_fraction(b * tfwd, experts=256, top_k=6))
        dur = (true_d[0] + true_d[1] * phi + true_d[2] * b) * float(rng.uniform(0.99, 1.01))
        decode_samples.append(DecodeStepSample(b, tfwd, dur))

    model, p_rep, d_rep = IterationTimeModel.fit_from_samples(
        prefill_samples, decode_samples, experts=256, top_k=6
    )
    assert p_rep.passed and p_rep.mean_relative_error < 0.03
    assert d_rep.passed and d_rep.mean_relative_error < 0.03

    # Long cached context shrinks recommended prefill chunk under a tight step latency target
    short_ctx_chunk = model.recommend_prefill_chunk(
        cached_tokens=512,
        remaining_uncached_tokens=4096,
        target_step_sec=0.090,
        decode_active_requests=32,
    )
    long_ctx_chunk = model.recommend_prefill_chunk(
        cached_tokens=131072,
        remaining_uncached_tokens=4096,
        target_step_sec=0.090,
        decode_active_requests=32,
    )
    assert long_ctx_chunk < short_ctx_chunk

    # Rank-deficient feature matrix fails closed
    with pytest.raises(ValueError, match="not identifiable"):
        fit_steps([[1.0, 2.0]] * 20, [0.1] * 20, minimum_samples=16)

    # Calibrated DreamTraceSimulator executes in isolated subprocess and differentiates policies
    sim = DreamTraceSimulator(num_requests=80, seed=7, calibrated_model=model)
    base_m = sim.simulate(BASELINE_L2_POLICY_CODE)
    best_m = sim.simulate(L2_POLICY_MUTATIONS[-1][3])
    assert best_m.goodput_qps > base_m.goodput_qps


def test_branch_aware_eviction_preserves_paused_branch_over_compacted_history() -> None:
    block = 4
    sys_prompt = tuple(range(100, 100 + block))
    # Branch A pauses for a long sub-agent call (3 blocks total)
    a_turn1 = sys_prompt + tuple(range(200, 200 + 2 * block))
    # Branch B builds a 3-block history, then compacts it into a 1-block summary
    b_pre_compact = sys_prompt + tuple(range(300, 300 + 2 * block))
    b_post_compact = sys_prompt + tuple(range(400, 400 + block))
    # Branch C arrives needing 2 fresh blocks, forcing eviction under capacity_blocks=6
    c_turn1 = sys_prompt + tuple(range(500, 500 + 2 * block))
    # Branch A resumes after its tool call and extends its original prefix!
    a_turn2 = a_turn1 + tuple(range(250, 250 + block))

    events = [
        AgentTraceEvent("request", "branch-A", a_turn1),
        AgentTraceEvent("request", "branch-B", b_pre_compact),
        AgentTraceEvent("request", "branch-B", b_post_compact),
        AgentTraceEvent("request", "branch-C", c_turn1),
        AgentTraceEvent("request", "branch-A", a_turn2),
        AgentTraceEvent("end_branch", "branch-C"),
    ]
    report = compare_eviction_policies(events, capacity_blocks=6, block_tokens=block)
    assert report.compactions_detected == 1
    # Pure LRU evicts paused branch-A's blocks while keeping B's dead pre-compaction blocks!
    assert report.branch_aware_hit_blocks > report.lru_hit_blocks
    assert report.branch_aware_lost_kv_reuse < report.lru_lost_kv_reuse


def test_replay_conformance_gates_catch_counter_reset_and_recorded_spec_context() -> None:
    ok = validate_agent_replay_run(
        workload_sha256="abc123",
        expected_workload_sha256="abc123",
        tokenizer_sha256="def456",
        expected_tokenizer_sha256="def456",
        pre_run_queue_depths=[0, 0],
        pre_run_counter_deltas=[0, 0],
        completed_requests=120,
        failed_requests=0,
        worker_request_counters={"w0": [10, 45, 90], "w1": [12, 48, 82]},
        speculative_enabled=True,
        replay_context_mode="actual_output",
    )
    assert ok.passed

    bad = validate_agent_replay_run(
        workload_sha256="abc123",
        expected_workload_sha256="abc123",
        tokenizer_sha256="def456",
        expected_tokenizer_sha256="def456",
        pre_run_queue_depths=[0, 1],
        pre_run_counter_deltas=[0, 0],
        completed_requests=120,
        failed_requests=1,
        worker_request_counters={"w0": [10, 45, 3]},
        speculative_enabled=True,
        replay_context_mode="recorded_output",
    )
    assert not bad.passed
    assert len(bad.reasons) == 4


def test_paired_round_se_gate_and_remote_probe_runner(tmp_path: Path) -> None:
    parity = tmp_path / "parity.json"
    parity.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "chunked_schedule": {"decode_steps_with_partial_prefill": 2},
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
    base = {
        "t1": [
            TierMetric(label="t1", repeat=r, output_throughput_tok_s=100.0, median_ttft_ms=50.0, mean_tpot_ms=10.0, p99_itl_ms=12.0)
            for r in range(3)
        ]
    }
    # Noisy candidate: median +3.5%, but round 1 dips to -1.0% so mean < 2*SE
    cand = {
        "t1": [
            TierMetric(label="t1", repeat=0, output_throughput_tok_s=104.0, median_ttft_ms=50.0, mean_tpot_ms=10.0, p99_itl_ms=12.0),
            TierMetric(label="t1", repeat=1, output_throughput_tok_s=99.0, median_ttft_ms=50.0, mean_tpot_ms=10.0, p99_itl_ms=12.0),
            TierMetric(label="t1", repeat=2, output_throughput_tok_s=103.5, median_ttft_ms=50.0, mean_tpot_ms=10.0, p99_itl_ms=12.0),
        ]
    }
    stats = compute_paired_round_stats(base, cand)
    assert stats is not None
    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        remote_host="gpu-test",
        min_paired_se_z=2.0,
    )
    dec = evaluate_candidate(base, cand, parity, task)
    assert not dec.accepted
    assert any("paired round mean gain" in r for r in dec.reasons)

    # Test remote probe runner
    src_tar = tmp_path / "src.tar.gz"
    src_tar.write_bytes(b"tar")
    probe_py = tmp_path / "probe.py"
    probe_py.write_text("print('ok')\n", encoding="utf-8")
    out_json = tmp_path / "probe-out.json"

    def fake_exec(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(args)
        if "nvidia-smi" in joined:
            return subprocess.CompletedProcess(args, 0, "\n".join(f"{i}, 0 MiB" for i in range(8)) + "\n", "")
        if "config.json" in joined:
            return subprocess.CompletedProcess(args, 0, "deepseek_v41\n", "")
        if args[0] == "scp" and args[-1].endswith("probe-out.json"):
            Path(args[-1]).write_text('{"pxb_us": 26.8}\n', encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    res_path = ZStepRemoteRunner(task, executor=fake_exec).run_probe(
        "p001", src_tar, probe_py, out_json
    )
    assert json.loads(res_path.read_text(encoding="utf-8"))["pxb_us"] == 26.8


def test_scrub_transcript_redacts_secrets_ips_paths_and_fails_closed(tmp_path: Path) -> None:
    scrub_mod = _load_scrub_module()
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "PLANNER_RESPONSE",
                "content": "Connected to 10.23.51.32 from /Users/alice/work with SSHPASS='s3cr3tPass'",
                "tool_calls": [{"name": "run_command", "arguments": {"cmd": "echo sk-proj1234567890abcdefghijklmnop"}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_md = tmp_path / "clean.md"
    rc = scrub_mod.main([str(transcript), "-o", str(out_md), "--redact", "mycluster"])
    assert rc == 0
    cleaned = out_md.read_text(encoding="utf-8")
    assert "10.23.51.32" not in cleaned and "<ip>" in cleaned
    assert "/Users/alice" not in cleaned and "/Users/<user>" in cleaned
    assert "s3cr3tPass" not in cleaned
    assert "sk-proj1234567890" not in cleaned

    # Fail-closed (exit code 1) when an unprefixed high-entropy secret token remains
    untagged = tmp_path / "raw.txt"
    untagged.write_text("leak: aB3dE6gH9jK2mN5pQ8rS1tU4vW7xY0zA\n", encoding="utf-8")
    rc_bad = scrub_mod.main([str(untagged), "-o", str(tmp_path / "bad.md")])
    assert rc_bad == 1


def test_campaign_t24_and_archive_upstream_rebase_isolation(tmp_path: Path) -> None:
    from evolith.zstep.archive import ZStepArchive
    from evolith.zstep.campaigns import resolve_campaign

    spec = resolve_campaign("t2-4")
    assert spec is not None
    assert spec.campaign_id == "t2-4-agent-adaptive-kv-sched"
    assert "crates/zs-kv/src" in spec.allowed_roots
    assert "crates/zs-scheduler/src" in spec.allowed_roots

    with ZStepArchive(tmp_path / "archive.sqlite3") as archive:
        archive.start_candidate(
            "g0003-oldbase",
            3,
            None,
            b"diff --git a/old b/old\n",
            "hash_old",
            hypothesis="2-slot PXB ring",
            upstream_sha="2bea5a1",
        )
        archive.accept_candidate("g0003-oldbase", "/tmp/ev3", 0.0372)

        # Preserve full append-only history, while isolating patch stacking onto rebased `682eb08`
        assert len(archive.list_candidates()) == 1
        assert "g0003-oldbase" in archive.compressed_lineage_memory()
        assert archive.current_parent(upstream_sha="682eb08") is None
        assert archive.accepted_patches(upstream_sha="682eb08") == []
        assert archive.current_parent(upstream_sha="2bea5a1") == "g0003-oldbase"
        assert len(archive.accepted_patches(upstream_sha="2bea5a1")) == 1

