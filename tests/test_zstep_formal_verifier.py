from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evolith.cli import app
from evolith.zstep.archive import ZStepArchive
from evolith.zstep.formal_verifier import (
    FormalVerificationError,
    calibrate_formal_mutations,
    calibrate_historical_candidates,
    find_z3_binary,
    model_check_double_buffer_ring,
    model_check_kv_prefix_spec_lifecycle,
    run_formal_verification,
)
from evolith.zstep.loop import ZStepRSILoop
from evolith.zstep.task import CandidateStatus, ZStepRSITask


def _write_synthetic_zstep_tree(
    root: Path,
    *,
    inline_csrc: bool = False,
    define_pxb_slot_const: bool = True,
    num_pxb_slots: int = 2,
    py_has_div_guard: bool = True,
    cpp_uses_eff_blocks: bool = True,
    cu_ar_vecs_cap: int = 40960,
    keep_all_gather_consensus: bool = True,
) -> Path:
    py_dir = root / "worker/zstep_worker/kernels_sm120"
    py_dir.mkdir(parents=True, exist_ok=True)
    (root / "worker/zstep_worker/models").mkdir(parents=True, exist_ok=True)
    (root / "worker/zstep_worker/models/deepseek_v41.py").write_text(
        "class _LaunchedStep:\n    pass\n", encoding="utf-8"
    )

    cpp_div = "eff_blocks % rows == 0" if cpp_uses_eff_blocks else "blocks % rows == 0"
    cpp_div_guard = "eff_blocks" if cpp_uses_eff_blocks else "blocks"
    cpp_code = f"""void launch_shm_v5_pxb_allreduce_add_rmsnorm(int64_t rows, int64_t hidden, int64_t blocks, int64_t my_hbm_ptr, int64_t one_shot_max_vecs, int64_t world_size) {{
    const int64_t vecs_per_row = hidden / 8;
    TORCH_CHECK(vecs_per_row <= 1536, "hidden limit");
    const int64_t eff_blocks =
        (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && rows * vecs_per_row <= 40960) ? 8 : blocks;
    if (rows < {cpp_div_guard}) {{
        TORCH_CHECK({cpp_div}, "divisibility");
    }}
    if (rows * vecs_per_row > one_shot_max_vecs) {{
        TORCH_CHECK(vecs_per_row % world_size == 0, "world_size mod");
    }}
}}
"""

    cu_code = f"""void launch_shm_v5_allreduce_cuda(int64_t num_vecs, int blocks, int my_hbm_ptr) {{
    const bool small_pxb = (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && num_vecs <= {cu_ar_vecs_cap});
    const int launch_blocks = small_pxb ? 8 : blocks;
}}

void launch_shm_v5_allreduce_add_rmsnorm_cuda(int64_t rows, int64_t vecs_per_row, int blocks, int my_hbm_ptr) {{
    const int64_t num_vecs = rows * vecs_per_row;
    const bool small_pxb = (my_hbm_ptr != 0 && blocks >= 32 && blocks <= 56 && num_vecs <= 40960);
    const int launch_blocks = small_pxb ? 8 : blocks;
}}
"""

    div_line = (
        "        if rows < eff_blocks and (eff_blocks % rows) != 0:\n            return False\n"
        if py_has_div_guard
        else ""
    )
    ag_line = (
        "        dist.all_gather(status_list, status_t, group=exchange_group)\n"
        if keep_all_gather_consensus
        else "        pass\n"
    )
    pxb_slot_const_line = f"NUM_PXB_RING_SLOTS = {num_pxb_slots}\n" if define_pxb_slot_const else ""
    slot_sym = "NUM_PXB_RING_SLOTS" if define_pxb_slot_const else "NUM_RING_SLOTS"
    inline_prefix = ""
    if inline_csrc:
        inline_prefix = (
            f'_CPP_SOURCE = r"""\n{cpp_code}\n"""\n\n'
            f'_CUDA_SOURCE = r"""\n{cu_code}\n"""\n\n'
        )
    else:
        cpp_dir = py_dir / "csrc"
        cpp_dir.mkdir(parents=True, exist_ok=True)
        (cpp_dir / "shm_allreduce.cpp").write_text(cpp_code, encoding="utf-8")
        (cpp_dir / "shm_allreduce_kernels.cu").write_text(cu_code, encoding="utf-8")

    (py_dir / "shm_allreduce.py").write_text(
        f"""{inline_prefix}NUM_RING_SLOTS = 4
{pxb_slot_const_line}SHM_V5_HEADER_BYTES = 131072
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
            SHM_V5_HEADER_BYTES + {slot_sym} * (pxb_slot_stride + small_pxb_stride),
            4096,
        )
{ag_line}        self._pxb_enabled = True

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
        if vecs_per_row > 3 * launch_threads or vecs_per_row > 1536:
            return False
{div_line}        if num_vecs > self.one_shot_max_vecs and (vecs_per_row % self.world_size) != 0:
            return False
        return True

    def close(self):
        pass
""",
        encoding="utf-8",
    )
    return root


def test_explicit_state_models_are_explicitly_non_gating_model_only() -> None:
    ok_ring = model_check_double_buffer_ring()
    assert ok_ring.passed
    assert not ok_ring.enforced
    assert ok_ring.verdict == "MODEL_ONLY"
    assert ok_ring.explored_states > 10

    war_ring = model_check_double_buffer_ring(enforce_slot_wait=False)
    assert not war_ring.passed
    assert war_ring.pillar == "Isolation"
    assert "WAR hazard" in (war_ring.counterexample or "")

    ok_kv = model_check_kv_prefix_spec_lifecycle()
    assert ok_kv.passed
    assert not ok_kv.enforced
    assert ok_kv.verdict == "MODEL_ONLY"
    assert ok_kv.explored_states > 8

    leak_kv = model_check_kv_prefix_spec_lifecycle(trim_kv_leases_on_spec_rewind=False)
    assert not leak_kv.passed
    assert leak_kv.pillar == "Conservation"

    zero_sentinel = model_check_kv_prefix_spec_lifecycle(guard_zero_init_restore_slot=False)
    assert not zero_sentinel.passed
    assert zero_sentinel.pillar == "Isolation"


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_fail_stop_inline_and_split_3way_z3_plus_real_source_mutations(tmp_path: Path) -> None:
    split_tree = _write_synthetic_zstep_tree(tmp_path / "split_clean", inline_csrc=False)
    report = run_formal_verification(split_tree, tmp_path / "ev", require_z3=True)
    assert report.passed
    assert report.gate_verdict == "CLOSED"
    assert "3000-trial" in report.weakest_link
    assert (tmp_path / "ev/formal-verification.json").is_file()
    by_id = {c.clause_id: c for c in report.clauses}
    assert by_id["STATIC_PXB_RING_BYTES_BOUND"].enforced is True
    assert by_id["DIFF_PY_DESCRIPTOR_EQUIVALENCE"].enforced is True
    assert by_id["DIFF_PY_DESCRIPTOR_EQUIVALENCE"].verdict == "CLOSED_DIFF"
    assert by_id["DIFF_PY_DESCRIPTOR_EQUIVALENCE"].explored_states == 3000
    assert by_id["Z3_FUSED_AR_3WAY_SOUNDNESS"].enforced is True
    assert by_id["Z3_FUSED_AR_3WAY_SOUNDNESS"].verdict == "PROVED"
    assert by_id["SM_DOUBLE_BUFFER_RING_MODEL"].enforced is False
    assert by_id["SM_DOUBLE_BUFFER_RING_MODEL"].verdict == "MODEL_ONLY"
    assert by_id["SM_KV_PREFIX_SPEC_MODEL"].enforced is False
    assert by_id["SM_KV_PREFIX_SPEC_MODEL"].verdict == "MODEL_ONLY"

    # Real source-file mutation suite (both split and inline _CPP_SOURCE/_CUDA_SOURCE layouts)
    cal_split = calibrate_formal_mutations(split_tree)
    assert cal_split.passed
    assert cal_split.total_mutants == 11
    assert cal_split.killed_mutants == 11
    assert cal_split.kill_rate == 1.0

    inline_tree = _write_synthetic_zstep_tree(tmp_path / "inline_clean", inline_csrc=True)
    cal_inline = calibrate_formal_mutations(inline_tree)
    assert cal_inline.passed
    assert cal_inline.killed_mutants == 11

    # g0001-class: missing NUM_PXB_RING_SLOTS using 4-slot NUM_RING_SLOTS in inline layout -> REJECTED!
    g0001_like = _write_synthetic_zstep_tree(
        tmp_path / "g0001_inline_oom", inline_csrc=True, define_pxb_slot_const=False
    )
    with pytest.raises(FormalVerificationError, match="Formal verification REJECTED.*STATIC_PXB_RING_BYTES_BOUND"):
        run_formal_verification(g0001_like, require_z3=True)

    # g0004/g0009-class: inline _CPP_SOURCE uses raw `blocks % rows == 0` -> REJECTED!
    g0004_like = _write_synthetic_zstep_tree(
        tmp_path / "g0004_inline_div", inline_csrc=True, cpp_uses_eff_blocks=False
    )
    with pytest.raises(FormalVerificationError, match="Formal verification REJECTED.*Z3_FUSED_AR_3WAY_SOUNDNESS"):
        run_formal_verification(g0004_like, require_z3=True)

    # g0010-class: inline _CUDA_SOURCE has `10240` in AR vs `40960` in Fused -> REJECTED!
    g0010_like = _write_synthetic_zstep_tree(
        tmp_path / "g0010_inline_lockstep", inline_csrc=True, cu_ar_vecs_cap=10240
    )
    with pytest.raises(FormalVerificationError, match="Formal verification REJECTED.*Z3_FUSED_AR_3WAY_SOUNDNESS"):
        run_formal_verification(g0010_like, require_z3=True)

    # Fail-stop INCONCLUSIVE check: missing C++ launcher file MUST fail closed as INCONCLUSIVE!
    broken_missing_cpp = _write_synthetic_zstep_tree(tmp_path / "missing_cpp", inline_csrc=False)
    (broken_missing_cpp / "worker/zstep_worker/kernels_sm120/csrc/shm_allreduce.cpp").unlink()
    with pytest.raises(FormalVerificationError, match="Formal verification INCONCLUSIVE.*INCONCLUSIVE_PARSE_ERROR"):
        run_formal_verification(broken_missing_cpp, require_z3=True)


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_historical_zstep_candidates_ground_truth_if_present() -> None:
    candidates_dir = Path.cwd() / "targets/zstep/candidates"
    upstream_dir = Path.cwd() / "targets/zstep/upstream"
    if not candidates_dir.is_dir() or not upstream_dir.is_dir():
        pytest.skip("Historical zstep candidate trees not present in workspace")

    rows = calibrate_historical_candidates(candidates_dir, upstream_dir)
    assert len(rows) >= 10
    mismatches = [r for r in rows if not r.matched_expectation]
    assert not mismatches, f"Historical calibration mismatches: {mismatches}"

    by_id = {r.candidate_id: r for r in rows}
    for must_fail_oom in ("g0001-03408986", "g0002-a989db09"):
        if must_fail_oom in by_id:
            assert not by_id[must_fail_oom].actual_pass
            assert "STATIC_PXB_RING_BYTES_BOUND" in by_id[must_fail_oom].violated_clauses

    for must_fail_z3 in ("g0004-eac6c3f7", "g0009-9da0950d", "g0010-47b2aec5"):
        if must_fail_z3 in by_id:
            assert not by_id[must_fail_z3].actual_pass
            assert "Z3_FUSED_AR_3WAY_SOUNDNESS" in by_id[must_fail_z3].violated_clauses

    for must_pass in (
        "upstream",
        "g0003-b96b840f",
        "g0008-67b2dd20",
        "g0011-bcaef1a3",
        "g0012-243da173",
        "g0015-7a608d48",
        "g0018-dc4581de",
    ):
        if must_pass in by_id:
            assert by_id[must_pass].actual_pass


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_zstep_loop_blocks_candidate_failing_formal_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evolith.zstep.loop.run_local_gates", lambda *args: None)

    class _BuggyWorkspace:
        def __init__(self, task: ZStepRSITask) -> None:
            self.task = task

        def create_candidate(self, candidate_id: str, patches: list[bytes]) -> Path:
            cand = self.task.candidates_dir / candidate_id
            return _write_synthetic_zstep_tree(cand, keep_all_gather_consensus=False)

        def capture_patch(self, candidate_dir: Path) -> bytes:
            return b"diff --git a/shm_allreduce.py b/shm_allreduce.py\n"

        def freeze_source(self, source_dir: Path, destination: Path) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"frozen")
            return destination

    class _NoopProposer:
        def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
            return (
                'EVOLITH_PROPOSAL_JSON: {"hypothesis": "Drop PXB barrier", '
                '"niche_key": "(plane=l1_kernel,focus=pxb)", "patch_summary": "drop"}'
            )

    task = ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "source",
        propose_only=True,
    )
    archive = ZStepArchive(task.state_dir / "archive.sqlite3")
    statuses = ZStepRSILoop(
        task,
        workspace=_BuggyWorkspace(task),
        archive=archive,
        proposer=_NoopProposer(),
        remote=object(),
    ).run()
    assert statuses == [CandidateStatus.LOCAL_REJECTED]
    cand = archive.list_candidates()[0]
    assert "STATIC_PXB_RING_BYTES_BOUND" in str(cand["failure_reason"])


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_cli_zstep_formal_verify_width_independent(tmp_path: Path) -> None:
    upstream = Path.cwd() / "targets/zstep/upstream"
    target = upstream if upstream.is_dir() else _write_synthetic_zstep_tree(tmp_path / "ref")
    runner = CliRunner()
    for cols in ("80", "160"):
        res = runner.invoke(
            app,
            ["zstep-formal-verify", "--target", str(target), "--calibrate"],
            env={"COLUMNS": cols},
        )
        assert res.exit_code == 0, res.stdout
        assert "z3=ON" in res.stdout
        assert "CLOSED" in res.stdout
        assert "kill_rate=100%" in res.stdout


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_missing_mutation_pattern_errors_and_campaigns_pin_kv_ledger(tmp_path: Path) -> None:
    from evolith.zstep.campaigns import apply_campaign_to_task, resolve_campaign

    # 1. If a source mutation target string is absent (e.g. `NUM_PXB_RING_SLOTS = 2` written as `1 + 1`),
    # calibrate_formal_mutations MUST raise FormalVerificationError rather than counting it as killed!
    alt_tree = _write_synthetic_zstep_tree(tmp_path / "alt_const")
    shm_py = alt_tree / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    shm_py.write_text(
        shm_py.read_text(encoding="utf-8").replace(
            "NUM_PXB_RING_SLOTS = 2", "NUM_PXB_RING_SLOTS = 1 + 1"
        ),
        encoding="utf-8",
    )
    assert run_formal_verification(alt_tree, require_z3=True).gate_verdict == "CLOSED"
    with pytest.raises(FormalVerificationError, match="Calibration source pattern"):
        calibrate_formal_mutations(alt_tree)

    # 2. Campaigns T2-1 and T2-3 SHA-256 pin crates/zs-kv/src/lib.rs (BlockPool/BlockLease choke-point)
    base_task = ZStepRSITask(project_root=tmp_path, source_repo=tmp_path / "source")
    for alias in ("t2-1", "t2-3"):
        applied = apply_campaign_to_task(base_task, resolve_campaign(alias))
        assert "crates/zs-kv/src/lib.rs" in applied.pinned_spec_files

    # 3. expect_shm_allreduce=True fails closed as INCONCLUSIVE_MISSING_FILE if shm_allreduce.py is absent
    empty_worker = tmp_path / "empty_worker"
    (empty_worker / "worker/zstep_worker").mkdir(parents=True)
    with pytest.raises(FormalVerificationError, match="INCONCLUSIVE_MISSING_FILE"):
        run_formal_verification(empty_worker, expect_shm_allreduce=True)

    # Without expect_shm_allreduce, gate_verdict is NOT_APPLICABLE (never contradictory INCONCLUSIVE+passed=True)
    na_report = run_formal_verification(empty_worker, expect_shm_allreduce=False)
    assert na_report.passed is True
    assert na_report.gate_verdict == "NOT_APPLICABLE"


@pytest.mark.skipif(find_z3_binary() is None, reason="z3 binary not installed")
def test_subprocess_timeout_dunder_ban_and_modeling_mismatch_semantics(tmp_path: Path) -> None:
    import time

    # 1. Infinite loop in candidate can_fuse_add_rmsnorm is killed by subprocess timeout -> INCONCLUSIVE_TIMEOUT
    hang_tree = _write_synthetic_zstep_tree(tmp_path / "hang_loop")
    hang_py = hang_tree / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    hang_py.write_text(
        hang_py.read_text(encoding="utf-8").replace(
            "    def can_fuse_add_rmsnorm(self, input):\n",
            "    def can_fuse_add_rmsnorm(self, input):\n        while True:\n            pass\n",
            1,
        ),
        encoding="utf-8",
    )
    t0 = time.monotonic()
    with pytest.raises(
        FormalVerificationError,
        match="Formal verification INCONCLUSIVE.*INCONCLUSIVE_TIMEOUT",
    ):
        run_formal_verification(hang_tree, require_z3=True, diff_timeout_sec=0.4)
    assert (time.monotonic() - t0) < 4.0

    # 2. Dunder attribute sandbox-escape attempt is rejected as INCONCLUSIVE_PARSE_ERROR
    dunder_tree = _write_synthetic_zstep_tree(tmp_path / "dunder_escape")
    dunder_py = dunder_tree / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    dunder_py.write_text(
        dunder_py.read_text(encoding="utf-8").replace(
            "    def can_fuse_add_rmsnorm(self, input):\n",
            "    def can_fuse_add_rmsnorm(self, input):\n        _ = ().__class__\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        FormalVerificationError,
        match="Formal verification INCONCLUSIVE.*dunder attribute",
    ):
        run_formal_verification(dunder_tree, require_z3=True)

    # 3. Unmodeled stricter guard (`3 * launch_threads` -> `2 * launch_threads`) is classified as
    # INCONCLUSIVE (INCONCLUSIVE_MODEL_MISMATCH), NOT REJECTED / VIOLATION!
    unmodeled_tree = _write_synthetic_zstep_tree(tmp_path / "unmodeled_2x")
    unmodeled_py = unmodeled_tree / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    unmodeled_py.write_text(
        unmodeled_py.read_text(encoding="utf-8").replace(
            "vecs_per_row > 3 * launch_threads",
            "vecs_per_row > 2 * launch_threads",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalVerificationError) as exc_info:
        run_formal_verification(unmodeled_tree, require_z3=True)
    msg = str(exc_info.value)
    assert msg.startswith("Formal verification INCONCLUSIVE:")
    assert "INCONCLUSIVE_MODEL_MISMATCH" in msg
    assert "not a proven candidate bug" in msg
