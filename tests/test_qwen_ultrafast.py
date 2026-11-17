import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from evolith.cli import app
from evolith.lab.qwen_ultrafast import (
    QwenAdapterConfig,
    QwenKernelConfig,
    QwenUltrafastAdapter,
)
from evolith.lab.store import RunStore


def settings(tmp_path):
    return QwenAdapterConfig(
        remote="user@example.com",
        local_repo=tmp_path,
        remote_repo=Path("/remote/qwen"),
        nccl_library=Path("/remote/libnccl.so.2"),
        output=tmp_path / "results",
        model="models/test-qwen",
        model_pairs=5,
        rounds=5,
    )


class FakeAdapter(QwenUltrafastAdapter):
    def __init__(self, config, *, micro=True, tokens=True):
        super().__init__(config)
        self.micro = micro
        self.tokens = tokens
        self.model_calls = []

    def prepare(self):
        return {"identity": "identity", "source": "source"}

    def benchmark(self, config, directory, incumbent=128):
        directory.mkdir(parents=True, exist_ok=True)
        return {
            "target": "bf16-gemv",
            "config": {"threads": config.threads},
            "baseline_config": {"threads": incumbent},
            "correct": True,
            "baseline_us": [10.0] * 5,
            "candidate_us": [8.0 if self.micro else 11.0] * 5,
        }

    def model_run(self, threads, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model_calls.append(threads)
        return {
            "tokens": [11] * self.config.decode if self.tokens or threads == 128 else [12] * self.config.decode,
            "decode_seconds": 1.0 if threads == 128 else 0.8,
        }


def test_config_is_bounded_and_cli_needs_no_gpu(tmp_path):
    with pytest.raises(ValidationError):
        QwenKernelConfig(threads=32)
    with pytest.raises(ValidationError):
        QwenKernelConfig.model_validate({"threads": 128, "source": "code"})
    with pytest.raises(ValidationError):
        QwenAdapterConfig.model_validate(settings(tmp_path).model_dump() | {"model_pairs": 4})
    result = CliRunner().invoke(app, ["qwen-ultrafast", "--help"])
    assert result.exit_code == 0
    assert "--candidates" in result.stdout


def test_micro_and_model_gates_promote_and_archive(tmp_path):
    adapter = FakeAdapter(settings(tmp_path))
    result = adapter.run([256])
    assert result["deployment_approved"]
    assert result["incumbent"] == {"threads": 256}
    assert adapter.model_calls == [128, 256, 256, 128, 128, 256, 256, 128, 128, 256]
    report = json.loads(Path(result["report"]).read_text())
    assert report["candidates"][0]["micro_accepted"]
    assert report["candidates"][0]["model_measurement"]["accepted"]
    assert len(RunStore(tmp_path / "results" / "runs.sqlite3").recent("qwen_ultrafast_candidate")) == 1


def test_second_candidate_compares_against_the_new_incumbent(tmp_path):
    adapter = FakeAdapter(settings(tmp_path))
    result = adapter.run([256, 512])
    assert result["incumbent"] == {"threads": 256}
    assert adapter.model_calls[10:12] == [256, 512]
    report = json.loads(Path(result["report"]).read_text())
    assert report["candidates"][0]["approved"]
    assert not report["candidates"][1]["approved"]


def test_micro_rejection_never_loads_the_full_model(tmp_path):
    adapter = FakeAdapter(settings(tmp_path), micro=False)
    result = adapter.run([256])
    assert not result["deployment_approved"]
    assert adapter.model_calls == []
    assert result["incumbent"] == {"threads": 128}


def test_mismatched_benchmark_provenance_fails_closed(tmp_path):
    adapter = QwenUltrafastAdapter(settings(tmp_path))
    adapter._remote = lambda *args: json.dumps({
        "target": "bf16-gemv",
        "config": {"threads": 512},
        "baseline_config": {"threads": 128},
    })
    with pytest.raises(ValueError, match="candidate/incumbent"):
        adapter.benchmark(QwenKernelConfig(threads=256), tmp_path / "unused")


def test_token_mismatch_rejects_despite_faster_timings(tmp_path):
    adapter = FakeAdapter(settings(tmp_path), tokens=False)
    result = adapter.run([256])
    assert not result["deployment_approved"]
    assert adapter.model_calls == [128, 256]
    report = json.loads(Path(result["report"]).read_text())
    assert not report["candidates"][0]["tokens_match"]
    assert "model_measurement" not in report["candidates"][0]
