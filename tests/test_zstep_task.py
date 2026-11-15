from pathlib import Path

import pytest
from pydantic import ValidationError

from evolith.zstep.task import ZStepRSITask


def make_task(tmp_path: Path) -> ZStepRSITask:
    return ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "original-zstep",
        baseline_sha="596d4ef",
    )


def test_task_keeps_all_source_inside_evolith(tmp_path: Path) -> None:
    task = make_task(tmp_path)

    assert task.upstream_dir == tmp_path / "targets/zstep/upstream"
    assert task.candidates_dir == tmp_path / "targets/zstep/candidates"
    assert task.state_dir == tmp_path / ".evolith/zstep-rsi"
    assert task.baseline_cache_dir == tmp_path / ".evolith/zstep-rsi/baseline-cache"
    assert task.max_generations == 1
    # T0-1a: default screen_tiers covers both single-stream latency (c1) and batched concurrency (c32)
    screen_labels = {item.split(":", 1)[0] for item in task.screen_tiers.split(";")}
    assert screen_labels == {"in512_c1", "in512_c32"}
    assert task.premise_strike_limit == 2
    assert task.parity_prefill_chunk == 24


def test_task_rejects_overlapping_public_and_hidden_tiers(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        ZStepRSITask(
            project_root=tmp_path,
            source_repo=tmp_path / "original-zstep",
            baseline_sha="596d4ef",
            screen_tiers="in512_c1:1:8:512:64",
            verify_tiers="in512_c1:1:8:512:256",
        )
