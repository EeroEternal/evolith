from __future__ import annotations

from pathlib import Path
import pytest

from evolith.zstep.dispatch_verifier import (
    DispatchVerificationError,
    run_dispatch_verification,
)


def test_dispatch_verifier_proves_real_upstream_zstep_selectors_and_thresholds(
    tmp_path: Path,
) -> None:
    upstream_root = Path(__file__).resolve().parents[1] / "targets/zstep/upstream"
    if not upstream_root.is_dir():
        pytest.skip("targets/zstep/upstream not cloned in test environment")

    report = run_dispatch_verification(upstream_root, tmp_path / "evidence")
    assert report.passed is True
    # 128 max_buckets * 3 out-of-range + sum(1..128)=8256 = 8640 pairs
    assert report.checked_pairs >= 8640
    assert "worker/zstep_worker/shm_bridge.py:select_decode_bucket" in report.verified_selectors
    assert report.verified_thresholds["DEFAULT_MEGA_GATE_MIN_TOKENS"] == 32
    assert report.verified_thresholds["DEFAULT_ONE_SHOT_MAX_VECS"] == 4096
    assert (tmp_path / "evidence/dispatch-verification.json").is_file()


def test_dispatch_verifier_accepts_finer_runner_bucket_ladder(tmp_path: Path) -> None:
    dsv41 = tmp_path / "worker/zstep_worker/models/deepseek_v41.py"
    dsv41.parent.mkdir(parents=True)
    dsv41.write_text(
        """
DSV41_DECODE_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)

def _select_decode_bucket(num_seqs: int, max_bucket: int = 128) -> int:
    if num_seqs <= 0 or num_seqs > max_bucket:
        raise ValueError("out of range")
    for b in DSV41_DECODE_BUCKETS:
        if b >= num_seqs and b <= max_bucket:
            return b
    return max_bucket
""",
        encoding="utf-8",
    )

    report = run_dispatch_verification(tmp_path)
    assert report.passed is True
    assert report.checked_pairs == 8640
    assert "worker/zstep_worker/models/deepseek_v41.py:_select_decode_bucket" in report.verified_selectors


def test_dispatch_verifier_rejects_broken_bucket_selector_and_unaligned_shm_vecs(
    tmp_path: Path,
) -> None:
    dsv41 = tmp_path / "worker/zstep_worker/models/deepseek_v41.py"
    dsv41.parent.mkdir(parents=True)
    # Bug: returns bucket > max_bucket when max_bucket=20 and num_seqs=18 -> returns 32!
    dsv41.write_text(
        """
DSV41_DECODE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

def _select_decode_bucket(num_seqs: int, max_bucket: int = 128) -> int:
    if num_seqs <= 0 or num_seqs > max_bucket:
        raise ValueError("out of range")
    for b in DSV41_DECODE_BUCKETS:
        if b >= num_seqs:
            return b
    return max_bucket
""",
        encoding="utf-8",
    )
    shm_ar = tmp_path / "worker/zstep_worker/kernels_sm120/shm_allreduce.py"
    shm_ar.parent.mkdir(parents=True)
    shm_ar.write_text("DEFAULT_ONE_SHOT_MAX_VECS = 4099\n", encoding="utf-8")

    with pytest.raises(DispatchVerificationError) as exc_info:
        run_dispatch_verification(tmp_path, tmp_path / "ev-fail")

    msg = str(exc_info.value)
    assert "violates bounds" in msg
    assert "divisible by 8" in msg
