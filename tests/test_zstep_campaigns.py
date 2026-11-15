from __future__ import annotations

from pathlib import Path
import pytest

from evolith.zstep.campaigns import (
    apply_campaign_to_task,
    list_campaigns,
    resolve_campaign,
)
from evolith.zstep.task import ZStepRSITask


def test_campaigns_are_ordered_by_roi_priority_and_resolve_aliases() -> None:
    campaigns = list_campaigns()
    assert [c.priority_rank for c in campaigns] == [
        "T1-1",
        "T1-2",
        "T1-3",
        "T2-1",
        "T2-2",
    ]
    assert resolve_campaign("none") is None
    assert resolve_campaign("t1-1") == campaigns[0]
    assert resolve_campaign("device-decode") == campaigns[0]
    assert resolve_campaign("T1-2") == campaigns[1]
    assert resolve_campaign("chunked-prefill") == campaigns[2]

    with pytest.raises(ValueError, match="Unknown campaign"):
        resolve_campaign("t9-unknown")


def test_auto_campaign_fails_over_when_primary_niche_is_frozen() -> None:
    campaigns = list_campaigns()
    assert resolve_campaign("auto", blocked_niches={}) == campaigns[0]

    blocked_t1_1 = {campaigns[0].primary_niche_key: ["g0001-a", "g0002-b"]}
    assert resolve_campaign("auto", blocked_niches=blocked_t1_1) == campaigns[1]

    blocked_t1_1_and_2 = {
        **blocked_t1_1,
        campaigns[1].primary_niche_key: ["g0003-c", "g0004-d"],
    }
    assert resolve_campaign("auto", blocked_niches=blocked_t1_1_and_2) == campaigns[2]


def test_apply_campaign_to_task_narrows_allowed_roots_and_complexity_budget(
    tmp_path: Path,
) -> None:
    base = ZStepRSITask(project_root=tmp_path, source_repo=tmp_path / "zstep")
    spec = resolve_campaign("t1-1")
    assert spec is not None

    scoped = apply_campaign_to_task(base, spec)
    assert scoped.allowed_roots == spec.allowed_roots
    assert "worker/zstep_worker/kernels_sm120/shm_allreduce.py" in scoped.allowed_roots
    assert "worker/zstep_worker/kernels_sm120/norm_quant.py" not in scoped.allowed_roots
    assert scoped.complexity_free_lines == 180
    assert scoped.max_changed_lines == 320
    # Original task instance remains untouched
    assert base.complexity_free_lines == 120
