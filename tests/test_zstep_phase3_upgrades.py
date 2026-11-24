from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from evolith.cli import app
from evolith.zstep.archive import ZStepArchive
from evolith.zstep.composer import (
    cluster_remote_failures,
    compose_patch_pair,
    extract_and_validate_scout_probe,
    find_orthogonal_stepping_stone_pairs,
    validate_monotonic_gate,
)
from evolith.zstep.formal_verifier import (
    FormalInconclusiveError,
    FormalViolationError,
    find_z3_binary,
    run_formal_verification,
)
from evolith.zstep.hardware import (
    infer_hardware_profile_from_probe,
    resolve_benchmark_profile,
    resolve_hardware_profile,
)
from evolith.zstep.loop import ZStepRSILoop
from evolith.zstep.remote import ZStepRemoteRunner
from evolith.zstep.task import CandidateStatus, ZStepRSITask
from evolith.zstep.trace_conformance import (
    TraceConformanceError,
    verify_trace_conformance,
)


def _write_modern_zstep_fixture(
    root: Path,
    *,
    rust_swa_mult: int = 4,
    cli_swa_mult: int = 4,
    verify_checks_draft_horizon: bool = True,
    tbo_pxb_max_tokens: int = 1024,
    unmodeled_py_2x: bool = False,
) -> Path:
    shm_dir = root / "worker/zstep_worker/kernels_sm120"
    csrc_dir = shm_dir / "csrc"
    csrc_dir.mkdir(parents=True, exist_ok=True)

    (csrc_dir / "shm_allreduce.cpp").write_text(
        """void launch_shm_v5_pxb_allreduce_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor normed_out,
    torch::Tensor control,
    torch::Tensor rank_shm_ptrs_u64,
    std::uintptr_t my_hbm_ptr,
    std::uintptr_t pt_hbm_ptr,
    int64_t slot_stride_bytes,
    int64_t pxb_slot_stride_bytes,
    double eps,
    double weight_offset,
    int64_t rank,
    int64_t world_size,
    int64_t blocks,
    int64_t threads,
    int64_t one_shot_max_vecs,
    int64_t pxb_max_vecs,
    int64_t use_cluster,
    std::uintptr_t stream_handle) {
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % 8 == 0, "shape");
    const int64_t vecs_per_row = hidden / 8;
    TORCH_CHECK(vecs_per_row <= 1536, "hidden limit");
    const int64_t eff_blocks =
        (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && rows * vecs_per_row <= 40960) ? 8 : blocks;
    if (rows < eff_blocks) {
        TORCH_CHECK(eff_blocks % rows == 0, "divisibility");
    }
    if (rows * vecs_per_row > one_shot_max_vecs) {
        TORCH_CHECK(vecs_per_row % world_size == 0, "world_size mod");
    }
}
""",
        encoding="utf-8",
    )
    (csrc_dir / "shm_allreduce_kernels.cu").write_text(
        """void launch_shm_v5_allreduce_cuda(int64_t num_vecs, int blocks, int my_hbm_ptr, int pxb_bfp) {
    const int64_t small_lim = (pxb_bfp == 1) ? 20480 : 40960;
    const bool small_pxb = (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && num_vecs <= small_lim);
    const int launch_blocks = small_pxb ? 8 : blocks;
}

void launch_shm_v5_allreduce_add_rmsnorm_cuda(int64_t rows, int64_t vecs_per_row, int blocks, int my_hbm_ptr) {
    const int64_t num_vecs = rows * vecs_per_row;
    const bool small_pxb = (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && num_vecs <= 40960);
    const int launch_blocks = small_pxb ? 8 : blocks;
}
""",
        encoding="utf-8",
    )
    epilogue_mult = 2 if unmodeled_py_2x else 3
    (shm_dir / "shm_allreduce.py").write_text(
        f"""NUM_RING_SLOTS = 4
NUM_PXB_RING_SLOTS = 2
SHM_V5_HEADER_BYTES = 131072
DEFAULT_ONE_SHOT_MAX_VECS = 4096
DEFAULT_PXB_MAX_ELEMENTS = 4096 * 5120


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class HostMappedShmAllReduce:
    def _init_pxb_ipc(self, exchange_group, pxb_max_elements: int = DEFAULT_PXB_MAX_ELEMENTS):
        aligned_elems = (pxb_max_elements // 8) * 8
        pxb_rank_stride = _align_up(aligned_elems * 2, 4096)
        pxb_slot_stride = 2 * pxb_rank_stride
        small_pxb_stride = 2 * _align_up(40960 * 16, 4096)
        pxb_total_bytes = _align_up(
            SHM_V5_HEADER_BYTES + NUM_PXB_RING_SLOTS * (pxb_slot_stride + small_pxb_stride),
            4096,
        )
        dist.all_gather(status_list, status_t, group=exchange_group)
        self._pxb_enabled = True

    def can_fuse_add_rmsnorm(self, input):
        rows, hidden = int(input.shape[0]), int(input.shape[1])
        vecs_per_row = hidden // 8
        num_vecs = rows * vecs_per_row
        small_pxb = self._pxb_enabled and 32 <= self.blocks <= 56 and num_vecs <= 40960
        eff_blocks = 8 if small_pxb else self.blocks
        capped_threads = 512 if self.threads > 512 else self.threads
        eff_threads = 256 if small_pxb else capped_threads
        vecs_per_cta = (num_vecs + eff_blocks - 1) // eff_blocks
        launch_threads = 512 if (eff_threads == 256 and (vecs_per_cta > 256 or vecs_per_row > 256)) else eff_threads
        if vecs_per_row > {epilogue_mult} * launch_threads or vecs_per_row > 1536:
            return False
        if rows < eff_blocks and (eff_blocks % rows) != 0:
            return False
        if num_vecs > self.one_shot_max_vecs and (vecs_per_row % self.world_size) != 0:
            return False
        return True
""",
        encoding="utf-8",
    )

    serve_dir = root / "crates/zs-server/src/serve"
    serve_dir.mkdir(parents=True, exist_ok=True)
    (root / "crates/zs-server/src/serve.rs").write_text(
        f"fn init(args: &ServeArgs) {{\n    let min_active_swa = {rust_swa_mult} * args.max_seqs as usize;\n}}\n",
        encoding="utf-8",
    )
    (serve_dir / "cli.rs").write_text(
        f"fn check(args: &ServeArgs) {{\n    let min_active_swa = {cli_swa_mult} * args.max_seqs as usize;\n}}\n",
        encoding="utf-8",
    )

    c_hi_expr = (
        "np.minimum(pt_np.shape[1] - 1, (p_max + self.num_spec_tokens) // 128)"
        if verify_checks_draft_horizon
        else "np.minimum(pt_np.shape[1] - 1, p_max // 128)"
    )
    models_dir = root / "worker/zstep_worker/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "deepseek_v41.py").write_text(
        f"""class _SwaPagePool:
    def map_active_windows(self, active_pt, seq_pos_min, seq_pos_max):
        for i in range(1):
            p_lo = max(0, (int(seq_pos_min[i]) - 127) // 128)
            p_hi = min(int(active_pt.shape[1]) - 1, int(seq_pos_max[i]) // 128)
        return active_pt, False


class DeepSeekV41StepRunner:
    def __init__(self):
        self._init_impl()

    def _init_impl(self):
        ar_max_tokens = 8192 if self.pp_size > 1 else max(self.max_step_tokens, 2048)
        self.shm_ar = HostMappedShmAllReduce(
            capacity_elements=ar_max_tokens * 8192,
            enable_pxb_p2p=self.p2p_report.pxb_shm_enabled,
            pxb_max_elements=min(ar_max_tokens, 8192) * 5120,
        )
        if self.pp_size == 1:
            self.shm_ar_tbo = HostMappedShmAllReduce(
                capacity_elements={tbo_pxb_max_tokens} * 5120,
                enable_pxb_p2p=True,
                pxb_max_elements={tbo_pxb_max_tokens} * 5120,
            )

    def _matches_gpu_bt_cache(self, bt_key, pt_np, slots_np, c_hi):
        return True

    def step_verify(self, pt_np, p_max, v_bucket, seq_slots_np):
        c_hi = {c_hi_expr}
        self._last_verify_bt_sig = (v_bucket, tuple(seq_slots_np))
""",
        encoding="utf-8",
    )
    (models_dir / "deepseek_v41_dspark.py").write_text(
        """def draft(runner, req_slots, eff_b):
    sig_match = getattr(runner, "_last_verify_bt_sig", None) == (eff_b, tuple(req_slots))
    bt_cached = bool(sig_match)
    return bt_cached
""",
        encoding="utf-8",
    )
    return root


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_cross_language_z3_clauses_catch_all_three_bugs_and_prove_fixed_tree(tmp_path: Path) -> None:
    clean = _write_modern_zstep_fixture(tmp_path / "clean")
    rep = run_formal_verification(clean, require_z3=True)
    assert rep.passed
    assert rep.gate_verdict == "CLOSED"
    assert len(rep.contract_sha256) == 64
    assert len(rep.target_tree_sha256) == 64
    by_id = {c.clause_id: c for c in rep.clauses}
    assert by_id["Z3_CROSS_LANG_SWA_POOL_SAFETY"].verdict == "PROVED"
    assert by_id["Z3_DSPARK_BT_CACHE_HORIZON"].verdict == "PROVED"
    assert by_id["STATIC_RUNNER_PXB_TOTAL_BUDGET"].verdict == "PASS"

    # 1. Rust SWA reservation `2 * max_seqs` vs Python 3-page verify straddle -> VIOLATION
    swa_bug = _write_modern_zstep_fixture(tmp_path / "swa_bug", rust_swa_mult=2, cli_swa_mult=2)
    with pytest.raises(FormalViolationError, match="Z3_CROSS_LANG_SWA_POOL_SAFETY.*3 pages/seq"):
        run_formal_verification(swa_bug, require_z3=True)

    # 2. Verify `p_max // 128` vs DSpark `(p_max + k) // 128` -> VIOLATION
    bt_bug = _write_modern_zstep_fixture(
        tmp_path / "bt_bug", verify_checks_draft_horizon=False
    )
    with pytest.raises(FormalViolationError, match="Z3_DSPARK_BT_CACHE_HORIZON.*p_max=127"):
        run_formal_verification(bt_bug, require_z3=True)

    # 3. Dual `shm_ar + shm_ar_tbo` = 325.25 MiB > 256 MiB on PP=1 -> VIOLATION
    pxb_bug = _write_modern_zstep_fixture(tmp_path / "pxb_bug", tbo_pxb_max_tokens=4096)
    with pytest.raises(FormalViolationError, match="STATIC_RUNNER_PXB_TOTAL_BUDGET.*PP=1"):
        run_formal_verification(pxb_bug, require_z3=True)

    # 4. Unmodeled 2x epilogue bound raises FormalInconclusiveError (distinct from FormalViolationError)
    unmod = _write_modern_zstep_fixture(tmp_path / "unmod", unmodeled_py_2x=True)
    with pytest.raises(FormalInconclusiveError, match="INCONCLUSIVE_MODEL_MISMATCH"):
        run_formal_verification(unmod, require_z3=True)


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_formal_inconclusive_is_fail_closed_without_burning_premise_strikes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)

    class _UnmodeledWorkspace:
        def __init__(self, task: ZStepRSITask) -> None:
            self.task = task

        def create_candidate(self, candidate_id: str, patches: list[bytes]) -> Path:
            return _write_modern_zstep_fixture(
                self.task.candidates_dir / candidate_id, unmodeled_py_2x=True
            )

        def capture_patch(self, candidate_dir: Path) -> bytes:
            return b"diff --git a/shm.py b/shm.py\n+x\n"

        def freeze_source(self, source_dir: Path, destination: Path) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"frozen")
            return destination

    class _Proposer:
        def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
            return (
                'EVOLITH_PROPOSAL_JSON: {"hypothesis": "Stricter 2x epilogue bound", '
                '"niche_key": "(plane=l1_kernel,focus=epilogue_2x)", "patch_summary": "2x"}'
            )

    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "src",
        propose_only=True,
        max_generations=2,
    )
    archive = ZStepArchive(task.state_dir / "archive.sqlite3")
    statuses = ZStepRSILoop(
        task,
        workspace=_UnmodeledWorkspace(task),
        archive=archive,
        proposer=_Proposer(),
        remote=object(),
    ).run()

    # Fail-closed (both blocked before GPU!), recorded as FORMAL_INCONCLUSIVE, 0 premise strikes!
    assert statuses == [
        CandidateStatus.FORMAL_INCONCLUSIVE,
        CandidateStatus.FORMAL_INCONCLUSIVE,
    ]
    assert archive.blocked_niches(strike_limit=2) == {}


def test_trace_conformance_three_state_outcomes_and_cli(tmp_path: Path) -> None:
    parity_ok = tmp_path / "parity-ok.json"
    metrics_ok = tmp_path / "metrics-ok.json"
    parity_ok.write_text(
        json.dumps(
            {
                "triton_recompilations": 0,
                "chunked_schedule": {
                    "chunked_prefill_steps": 6,
                    "decode_steps_with_partial_prefill": 4,
                },
                "cases": [
                    {
                        "name": "prime_squares_table",
                        "single_match": {"exact_token_match_rate": 1.0},
                        "batched_match": {"exact_token_match_rate": 1.0},
                        "chunked_match": {"exact_token_match_rate": 1.0},
                        "single_vs_batched_first_mismatch": None,
                        "batched_vs_chunked_first_mismatch": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metrics_ok.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "label": "in512_c16",
                        "repeat": 0,
                        "output_throughput_tok_s": 900.0,
                        "median_ttft_ms": 40.0,
                        "mean_tpot_ms": 11.0,
                        "p99_itl_ms": 15.0,
                        "steps": 120,
                        "overlapped_decode_steps": 110,
                        "idle_kv_cache_usage": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rep = verify_trace_conformance(parity_ok, metrics_ok, evidence_dir=tmp_path / "ev")
    assert rep.passed and rep.gate_verdict == "CONFORMANT"
    assert (tmp_path / "ev/trace-conformance.json").is_file()

    runner = CliRunner()
    cli_res = runner.invoke(
        app, ["zstep-trace-check", "--parity", str(parity_ok), "--metrics", str(metrics_ok)]
    )
    assert cli_res.exit_code == 0, cli_res.stdout
    assert "CONFORMANT" in cli_res.stdout

    # Incomplete trace (missing chunked_schedule) -> INCONCLUSIVE_INCOMPLETE_TRACE (never PASS!)
    parity_inc = tmp_path / "parity-inc.json"
    parity_inc.write_text(
        json.dumps({"triton_recompilations": 0, "cases": []}), encoding="utf-8"
    )
    with pytest.raises(TraceConformanceError, match="INCONCLUSIVE_INCOMPLETE_TRACE"):
        verify_trace_conformance(parity_inc)

    # KV leak -> VIOLATION
    metrics_leak = tmp_path / "metrics-leak.json"
    metrics_leak.write_text(
        json.dumps({"tiers": [{"label": "in512_c16", "idle_kv_cache_usage": 0.025, "steps": 10}]}),
        encoding="utf-8",
    )
    with pytest.raises(TraceConformanceError, match="VIOLATION.*leaked KV blocks"):
        verify_trace_conformance(parity_ok, metrics_leak)


def test_hardware_and_benchmark_profiles_plus_remote_32_topology_inference(tmp_path: Path) -> None:
    smi_32 = "\n".join(
        f"{i}, NVIDIA RTX 6000D, 85651 MiB, 12.0" for i in range(8)
    )
    topo_32 = (
        "GPU0\t X \tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\n"
        "GPU4\tSYS\tSYS\tSYS\tSYS\t X \tPXB\tNODE\tNODE\n"
    )
    prof = infer_hardware_profile_from_probe(smi_32, topo_32, prefer_pp2_on_dual_socket_pcie=True)
    assert prof.profile_id == "sm120-8x6000d-pp2tp4"
    assert prof.pp_size == 2 and prof.tp_size == 4
    assert prof.decode_pipeline_1f1b is True
    assert prof.interconnect == "pcie_pxb_dual_numa"

    h100 = resolve_hardware_profile("sm90-8xh100-tp8")
    assert h100.gpu_arch == "sm90"
    assert h100.overlay_script == ""
    assert h100.kernel_backend_dir == "worker/zstep_worker/kernels_sm90"

    bprof = resolve_benchmark_profile("longctx-agent")
    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "zstep",
        remote_host="10.23.51.32",
        remote_user="node-user",
        remote_sudo=True,
        hardware_profile="sm120-8x6000d-pp2tp4",
        benchmark_profile="longctx-agent",
        isolate_hw_state=True,
    )
    assert task.screen_tiers == bprof.screen_tiers
    assert task.state_dir == tmp_path / ".evolith/zstep-rsi/sm120-8x6000d-pp2tp4"

    remote = ZStepRemoteRunner(task)
    bscript = remote.benchmark_script("/tmp/src", "/tmp/out.json", task.screen_tiers, 1, None)
    assert "/data02/zstep-work/rustup" in bscript
    assert "ZSTEP_PIPELINE_PARALLEL_SIZE=2" in bscript
    assert "ZSTEP_PP_DECODE_PIPELINE=1" in bscript


def test_stepping_stone_composer_scout_probe_and_monotonic_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base_repo"
    (base / "worker").mkdir(parents=True)
    (base / "worker/a.py").write_text("X = 1\n", encoding="utf-8")
    (base / "worker/b.py").write_text("Y = 2\n", encoding="utf-8")

    git_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def _make_patch(tag: str, rel_file: str, new_text: str) -> bytes:
        work = tmp_path / f"mk_{tag}"
        work.mkdir()
        (work / "worker").mkdir()
        (work / "worker/a.py").write_text("X = 1\n", encoding="utf-8")
        (work / "worker/b.py").write_text("Y = 2\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=work, env=git_env, check=True)
        subprocess.run(["git", "add", "-A"], cwd=work, env=git_env, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=work, env=git_env, check=True)
        (work / rel_file).write_text(new_text, encoding="utf-8")
        return subprocess.run(
            ["git", "diff"], cwd=work, env=git_env, capture_output=True, check=True
        ).stdout

    patch_a = _make_patch("a", "worker/a.py", "X = 10\n")
    patch_b = _make_patch("b", "worker/b.py", "Y = 20\n")
    patch_bad = _make_patch("bad", "worker/a.py", "X = -999  # BUG\n")

    archive = ZStepArchive(tmp_path / "archive.sqlite3")
    archive.start_candidate(
        "g0021-left",
        21,
        None,
        patch_a,
        "h1",
        changed_lines=2,
        changed_files=("worker/a.py",),
        hypothesis="Compress AllGather",
        upstream_sha="a36fab8",
    )
    archive.finish_candidate(
        "g0021-left",
        CandidateStatus.VERIFY_REJECTED,
        str(tmp_path / "ev1"),
        "geomean gain 2.88% below 3.00% threshold",
        geomean_gain=0.0288,
    )

    archive.start_candidate(
        "g0025-right",
        25,
        None,
        patch_b,
        "h2",
        changed_lines=2,
        changed_files=("worker/b.py",),
        hypothesis="Decode Sparse MLA Graph",
        upstream_sha="a36fab8",
    )
    archive.finish_candidate(
        "g0025-right",
        CandidateStatus.VERIFY_REJECTED,
        str(tmp_path / "ev2"),
        "geomean gain 2.88% below 3.00% threshold",
        geomean_gain=0.0288,
    )

    archive.start_candidate(
        "g0018-win",
        18,
        None,
        patch_b,
        "h3",
        changed_lines=2,
        changed_files=("worker/b.py",),
        hypothesis="Accepted winner",
        upstream_sha="a36fab8",
    )
    archive.accept_candidate("g0018-win", str(tmp_path / "ev3"), 0.0783)

    archive.start_candidate(
        "g0004-bad",
        4,
        "g0018-win",
        patch_bad,
        "h4",
        changed_lines=2,
        changed_files=("worker/a.py",),
        hypothesis="Bad patch",
        upstream_sha="a36fab8",
    )
    archive.finish_candidate(
        "g0004-bad",
        CandidateStatus.SCREEN_REJECTED,
        str(tmp_path / "ev4"),
        "CUDA error: out of memory",
    )

    pairs = find_orthogonal_stepping_stone_pairs(archive, upstream_sha="a36fab8", parent_id=None)
    assert len(pairs) == 1
    assert pairs[0].left_id == "g0021-left" and pairs[0].right_id == "g0025-right"
    assert pairs[0].additive_gain_estimate > 0.058

    combined = compose_patch_pair(base, patch_a, patch_b)
    assert b"+X = 10" in combined and b"+Y = 20" in combined

    clusters = cluster_remote_failures(archive)
    assert any(c.category == "hbm_oom_or_ring_budget" for c in clusters)

    ok_mono, msg_mono = validate_monotonic_gate(
        archive,
        base,
        lambda tree: "# BUG" not in (tree / "worker/a.py").read_text(encoding="utf-8"),
        must_reject_ids=("g0004-bad",),
    )
    assert ok_mono, msg_mono

    # Scout probe validator: accepts clean probe, blocks network/subprocess imports
    valid_probe = (
        "EVOLITH_SCOUT_PROBE_BEGIN\n"
        "import argparse, json\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--output', required=True)\n"
        "EVOLITH_SCOUT_PROBE_END"
    )
    assert extract_and_validate_scout_probe(valid_probe) is not None
    with pytest.raises(ValueError, match="Forbidden module `socket`"):
        extract_and_validate_scout_probe(
            "EVOLITH_SCOUT_PROBE_BEGIN\nimport socket, sys\nprint(sys.argv)\nEVOLITH_SCOUT_PROBE_END"
        )
