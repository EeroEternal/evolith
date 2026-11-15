from __future__ import annotations

import os
from pathlib import Path

import pytest

from evolith.lab.rank_cache import isolate_rank_cache


def test_cache_paths_are_owned_by_rank_without_changing_vllm_or_math(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "previous")
    monkeypatch.setenv("TRITON_CACHE_DIR", "previous")
    first = isolate_rank_cache(tmp_path, 0)
    second = isolate_rank_cache(tmp_path, 1)
    assert set(first["environment"]) == {"TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"}
    assert set(first["environment"].values()).isdisjoint(second["environment"].values())
    assert all(Path(path).is_dir() for path in second["environment"].values())
    assert os.environ["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm")
    assert isolate_rank_cache(tmp_path, 1) == second


@pytest.mark.parametrize("rank", [-1, 4, True, "0"])
def test_invalid_ranks_do_not_change_cache_environment(monkeypatch, tmp_path, rank):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "unchanged")
    with pytest.raises(ValueError):
        isolate_rank_cache(tmp_path, rank)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "unchanged"


def test_relative_namespace_is_rejected():
    with pytest.raises(ValueError):
        isolate_rank_cache(Path("relative"), 0)


def test_repair_budget_cannot_be_reset_or_extended():
    from evolith.lab.kernel_repair import remaining_seconds

    budget = {"started_unix": 100, "deadline_unix": 1840,
              "gpus": [0, 1, 2, 3], "approved_seconds": 1800}
    assert remaining_seconds(budget, 400) == 1400
    with pytest.raises(TimeoutError):
        remaining_seconds(budget, 1300)
    with pytest.raises(ValueError):
        remaining_seconds(budget | {"deadline_unix": 1900}, 400)


def test_cache_repair_worker_requires_explicit_trusted_launcher(monkeypatch, tmp_path):
    from evolith.lab import engine_worker
    from evolith.lab.engine import EngineConfig, EngineJob

    job = EngineJob(model_path=tmp_path, output=tmp_path, config=EngineConfig())
    commands = []

    class Captured(Exception):
        pass

    def launch(command, **_):
        commands.append(command)
        raise Captured

    monkeypatch.setattr(engine_worker.subprocess, "Popen", launch)
    with pytest.raises(ValueError), engine_worker.server(job, tmp_path, rank_local_cache=True):
        pass
    assert not commands
    with pytest.raises(Captured), engine_worker.server(
        job, tmp_path, diagnostic_worker=True, rank_local_cache=True,
    ):
        pass
    assert commands[0][-1] == "evolith.lab.rank_worker.RankLocalWorker"


def test_replay_clone_does_not_use_live_autotuner_pickle_protocol(tmp_path):
    from evolith.lab.kernel_replay import replay_compiler

    class LiveAutotuner:
        def __getstate__(self):
            pytest.fail("Live compiled launchers must not be pickled.")

    original = LiveAutotuner()
    original.fn = None
    original.metadata = {"unchanged": True}
    copied = replay_compiler(original, tmp_path, 0)
    assert copied is not original and copied.metadata == original.metadata
