from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolith.zstep.redteam import (
    RedTeamRejectedError,
    compute_spec_pins,
    run_redteam_gate,
    scan_patch_boundaries,
)
from evolith.zstep.task import PatchStats, ZStepRSITask


def make_task(tmp_path: Path) -> ZStepRSITask:
    return ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "source",
    )


def _diff_for(file_path: str, added_lines: list[str]) -> PatchStats:
    body = "\n".join(f"+{line}" for line in added_lines)
    patch = (
        f"diff --git a/{file_path} b/{file_path}\n"
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        f"@@ -1,2 +1,{len(added_lines) + 2} @@\n"
        f"{body}\n"
    ).encode("utf-8")
    return PatchStats(
        patch=patch,
        changed_lines=len(added_lines),
        changed_files=(file_path,),
    )


def test_spec_pins_detect_tampered_reference_file(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    cand_dir = tmp_path / "cand"
    ref_file = cand_dir / "docs/benchmarks/phase3_vllm_greedy_reference.json"
    ref_file.parent.mkdir(parents=True)
    ref_file.write_text('{"cases": []}\n', encoding="utf-8")

    pins = compute_spec_pins(cand_dir, task.pinned_spec_files)
    # Tamper with the reference file after pinning
    ref_file.write_text('{"cases": [{"hacked": true}]}\n', encoding="utf-8")

    stats = _diff_for("worker/zstep_worker/runtime_loop.py", ["x = 1"])
    with pytest.raises(RedTeamRejectedError, match="Coordinator-pinned spec file"):
        run_redteam_gate(
            cand_dir,
            stats,
            task,
            tmp_path / "evidence",
            expected_pins=pins,
        )

    report = json.loads((tmp_path / "evidence/redteam-report.json").read_text())
    assert report["passed"] is False
    assert report["spec_pins_verified"] is False


def test_redteam_blocks_unsynchronized_cuda_stream_war_hazard(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    unsafe_stats = _diff_for(
        "worker/zstep_worker/models/deepseek_v41.py",
        [
            "self.aux_stream = torch.cuda.Stream()",
            "with torch.cuda.stream(self.aux_stream):",
            "    self.decode_ids.copy_(next_ids)",
        ],
    )
    with pytest.raises(RedTeamRejectedError, match="overwrite_war"):
        run_redteam_gate(tmp_path, unsafe_stats, task, tmp_path / "ev-unsafe")

    # Pairing auxiliary stream with explicit Event synchronization passes
    safe_stats = _diff_for(
        "worker/zstep_worker/models/deepseek_v41.py",
        [
            "self.aux_stream = torch.cuda.Stream()",
            "self.ready_evt = torch.cuda.Event()",
            "self.aux_stream.wait_event(self.ready_evt)",
        ],
    )
    report = run_redteam_gate(tmp_path, safe_stats, task, tmp_path / "ev-safe")
    assert report.passed is True


def test_redteam_blocks_transient_async_h2d_autotune_unchecked_rust_and_spoofing() -> None:
    transient_h2d = _diff_for(
        "worker/zstep_worker/runtime_loop.py",
        ["ids_gpu = torch.tensor(step_ids, dtype=torch.int32).to(device, non_blocking=True)"],
    )
    assert any(f.category == "async_lifetime" and f.severity == "blocking" for f in scan_patch_boundaries(transient_h2d))

    autotune_patch = _diff_for(
        "worker/zstep_worker/kernels_sm120/norm_quant.py",
        ["@triton.autotune(configs=[...], key=['M'])", "def fused_norm_kernel(): pass"],
    )
    assert any(f.category == "execution_context" and f.severity == "blocking" for f in scan_patch_boundaries(autotune_patch))

    unchecked_rust = _diff_for(
        "crates/zs-scheduler/src/serving.rs",
        ["let slot = unsafe { self.slots.get_unchecked(idx) };"],
    )
    assert any(f.category == "indices_offsets" and f.severity == "blocking" for f in scan_patch_boundaries(unchecked_rust))

    spoof_recomp = _diff_for(
        "crates/zs-server/src/main.rs",
        ["report.triton_recompilations = Some(0);"],
    )
    assert any(f.category == "anti_cheat" and f.severity == "blocking" for f in scan_patch_boundaries(spoof_recomp))


def test_redteam_records_non_blocking_warnings_for_views_and_slot_reset(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    stats = _diff_for(
        "worker/zstep_worker/models/deepseek_v41.py",
        [
            "window = torch.as_strided(buf, size=(8, 3), stride=(3, 1))",
            "engram_slots = window",
        ],
    )
    report = run_redteam_gate(tmp_path, stats, task, tmp_path / "ev-warn")
    assert report.passed is True
    categories = {f.category for f in report.findings}
    assert "aliases_views" in categories
    assert "initial_values" in categories


def test_redteam_blocks_tp8_device_tokens_when_shm_allreduce_only_writes_rank0(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    cand_dir = tmp_path / "cand-tp8"
    dsv41_file = cand_dir / "worker/zstep_worker/models/deepseek_v41.py"
    shm_file = cand_dir / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    dsv41_file.parent.mkdir(parents=True)
    shm_file.parent.mkdir(parents=True)

    dsv41_file.write_text(
        "if header.flags & STEP_FLAG_DEVICE_TOKENS:\n    ids = self.out_tokens_gpu[:num_seqs, 0]\n",
        encoding="utf-8",
    )
    shm_file.write_text(
        "if (rank == 0) {\n    out_top2_tokens[row * 2 + 0] = best_idx;\n}\n",
        encoding="utf-8",
    )
    stats = _diff_for(
        "worker/zstep_worker/models/deepseek_v41.py",
        ["if header.flags & STEP_FLAG_DEVICE_TOKENS:"],
    )
    with pytest.raises(RedTeamRejectedError, match="agreement"):
        run_redteam_gate(cand_dir, stats, task, tmp_path / "ev-tp8-bad")

    # Once all 8 ranks populate out_top2_tokens (removing rank == 0 guard), Agreement gate passes!
    shm_file.write_text(
        "out_top2_tokens[row * 2 + 0] = best_idx;\nout_top2_tokens[row * 2 + 1] = second_idx;\n",
        encoding="utf-8",
    )
    report = run_redteam_gate(cand_dir, stats, task, tmp_path / "ev-tp8-good")
    assert report.passed is True


def test_redteam_blocks_pytorch_dtype_overload_traps_and_graph_sabotage() -> None:
    # g0004: int32 index buffer (including sliced arena) passed to index_copy_ without int64 cast
    bad_idx = _diff_for(
        "worker/zstep_worker/models/deepseek_v41.py",
        ["self.last_token_gpu.index_copy_(0, self.comp_bt_gpu[:n], sampled)"],
    )
    assert any(
        f.category == "indices_offsets" and f.severity == "blocking"
        for f in scan_patch_boundaries(bad_idx)
    )
    good_idx = _diff_for(
        "worker/zstep_worker/models/deepseek_v41.py",
        [
            "self.last_token_gpu.index_copy_(0, self.comp_bt_gpu[:n].to(torch.int64), sampled)",
            "rows = table.index_select(0, self.slot_ids_gpu[:bs].long())",
        ],
    )
    assert not any(
        f.category == "indices_offsets" and f.severity == "blocking"
        for f in scan_patch_boundaries(good_idx)
    )

    # g0005: torch.where(..., out=...) with Python numeric scalar in either position 2 or 3
    for where_line in (
        "torch.where(mask, next_tok, 0, out=self.staging_gpu)",
        "torch.where(mask, 0, next_tok, out=self.staging_gpu)",
    ):
        bad_where = _diff_for(
            "worker/zstep_worker/models/deepseek_v41.py",
            [where_line],
        )
        assert any(
            f.category == "value_ranges" and f.severity == "blocking"
            for f in scan_patch_boundaries(bad_where)
        )

    # Baseline-sabotage reward hack: disabling CUDA Graphs on model runner
    sabotage = _diff_for(
        "worker/zstep_worker/models/qwen35_moe.py",
        ["self.max_graph_bucket = 0"],
    )
    assert any(
        f.category == "anti_cheat" and f.severity == "blocking"
        for f in scan_patch_boundaries(sabotage)
    )

    # Zero-initialized SHM recurrent slot sentinel warning
    sentinel_warn = _diff_for(
        "worker/zstep_worker/models/qwen35_moe.py",
        ["if restore_slot >= 0: self.restore_gdn(restore_slot)"],
    )
    assert any(
        f.category == "initial_values" and f.severity == "warning"
        for f in scan_patch_boundaries(sentinel_warn)
    )


def test_redteam_ast_blocks_hotpath_host_sync_and_dist_broadcast(
    tmp_path: Path,
) -> None:
    task = make_task(tmp_path)
    cand_dir = tmp_path / "cand-ast"
    model_file = cand_dir / "worker/zstep_worker/models/deepseek_v41.py"
    model_file.parent.mkdir(parents=True)
    model_file.write_text(
        "class Runner:\n"
        "    def __init__(self, t):\n"
        "        self.v = t.item()\n"
        "    def execute(self, logits, tok_gpu):\n"
        "        host_ids = self.out_tokens_cpu[:4].tolist()\n"
        "        n_acc = int(logits.argmax().item())\n"
        "        dist.broadcast(tok_gpu, src=0)\n"
        "        return host_ids, n_acc\n",
        encoding="utf-8",
    )
    # Post-sync CPU buffer `.tolist()` alone passes without false-positive rejection
    safe_cpu_patch = (
        "diff --git a/worker/zstep_worker/models/deepseek_v41.py b/worker/zstep_worker/models/deepseek_v41.py\n"
        "--- a/worker/zstep_worker/models/deepseek_v41.py\n"
        "+++ b/worker/zstep_worker/models/deepseek_v41.py\n"
        "@@ -4,1 +4,2 @@\n"
        "     def execute(self, logits, tok_gpu):\n"
        "+        host_ids = self.out_tokens_cpu[:4].tolist()\n"
    ).encode("utf-8")
    safe_stats = PatchStats(
        patch=safe_cpu_patch,
        changed_lines=1,
        changed_files=("worker/zstep_worker/models/deepseek_v41.py",),
    )
    assert run_redteam_gate(cand_dir, safe_stats, task, tmp_path / "ev-ast-safe").passed

    # GPU `.item()` and `dist.broadcast` inside `execute` are blocked
    patch = (
        "diff --git a/worker/zstep_worker/models/deepseek_v41.py b/worker/zstep_worker/models/deepseek_v41.py\n"
        "--- a/worker/zstep_worker/models/deepseek_v41.py\n"
        "+++ b/worker/zstep_worker/models/deepseek_v41.py\n"
        "@@ -5,1 +5,3 @@\n"
        "         host_ids = self.out_tokens_cpu[:4].tolist()\n"
        "+        n_acc = int(logits.argmax().item())\n"
        "+        dist.broadcast(tok_gpu, src=0)\n"
    ).encode("utf-8")
    stats = PatchStats(
        patch=patch,
        changed_lines=2,
        changed_files=("worker/zstep_worker/models/deepseek_v41.py",),
    )
    with pytest.raises(RedTeamRejectedError, match="execute"):
        run_redteam_gate(cand_dir, stats, task, tmp_path / "ev-ast")
