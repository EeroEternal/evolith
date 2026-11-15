from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolith.lab import engine, prepublication
from evolith.lab.engine import EngineArtifact, EngineConfig, source_hash
from evolith.lab.latency import CHAT_PROMPTS, workload_fingerprint


@pytest.fixture
def profile(tmp_path):
    proof = {
        "run_id": "approved-reference", "status": "measured", "correct": True,
        "source_hash": source_hash(), "config": EngineConfig().model_dump(),
        "model_fingerprint": "test-weights", "environment": {"gpu": "test"},
        "workload_fingerprint": workload_fingerprint(64), "max_tokens": 64,
        "ttft_us": [100.] * 5, "itl_us": [10.] * 5, "rounds": [{} for _ in range(5)],
        "records": [{"token_ids": list(range(64)), "token_logprobs": [-0.1] * 64}
                    for _ in CHAT_PROMPTS],
    }
    artifact = EngineArtifact(
        config=EngineConfig(), source_hash=source_hash(), model_fingerprint="test-weights",
        environment={"gpu": "test"}, workload_fingerprint=workload_fingerprint(64),
        selection="native_baseline", baseline_proof=proof,
        fresh_baseline_proof=copy.deepcopy(proof) | {"run_id": "approved-independent-restart"},
    )
    path = tmp_path / "profile.json"
    path.write_text(artifact.model_dump_json())
    return path, artifact


@pytest.mark.parametrize("change", [None, "ids", "logprobs", "stream_error"])
def test_fresh_process_must_match_approved_reference_on_every_pass(
    monkeypatch, tmp_path, profile, change,
):
    path, artifact = profile
    original = path.read_bytes()
    calls = []

    @contextmanager
    def client(_):
        yield SimpleNamespace(get=lambda *args, **kwargs: SimpleNamespace(status_code=200))

    def stream(_, prompt, tokens):
        calls.append((prompt, tokens))
        if change == "stream_error":
            raise ValueError("Incomplete generation.")
        record = copy.deepcopy(artifact.baseline_proof["records"][0])
        if change == "ids":
            record["token_ids"][0] = 999
        if change == "logprobs":
            record["token_logprobs"][0] -= 0.051
        return record

    monkeypatch.setattr(prepublication, "local_client", client)
    monkeypatch.setattr(prepublication, "stream_request", stream)
    output = tmp_path / "report.json"
    report = prepublication.verify(path, output)
    assert report["correct"] == (change is None)
    assert report["status"] == ("verified" if change is None else "failed")
    assert len(calls) == (30 if change is None else 1 if change == "stream_error" else 6)
    assert path.read_bytes() == original
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("key,value", [
    ("source_hash", "changed"), ("model_fingerprint", "other"),
    ("environment", {"gpu": "other"}), ("workload_fingerprint", "other"),
    ("config", EngineConfig(tensor_parallel=2).model_dump()),
    ("status", "failed"), ("correct", False), ("max_tokens", 1), ("records", []),
])
def test_missing_or_incompatible_reference_fails_before_connecting(
    monkeypatch, tmp_path, profile, key, value,
):
    path, artifact = profile
    artifact.baseline_proof[key] = value
    path.write_text(artifact.model_dump_json())
    monkeypatch.setattr(prepublication, "local_client", lambda _: pytest.fail("Must not connect."))
    report = prepublication.verify(path, tmp_path / "report.json")
    assert report["status"] == "failed" and not report["correct"]


def test_reference_must_include_every_token_for_all_six_requests(profile):
    _, artifact = profile
    for proof in (artifact.baseline_proof, artifact.fresh_baseline_proof):
        proof["records"][0]["token_ids"].pop()
        proof["records"][0]["token_logprobs"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        prepublication.expected_proof(artifact)


def test_health_check_has_a_deadline(monkeypatch, tmp_path, profile):
    path, _ = profile
    clock = iter(range(0, 10000, 1000))
    monkeypatch.setattr(prepublication, "time", SimpleNamespace(
        monotonic=lambda: next(clock), sleep=lambda _: None,
    ))

    @contextmanager
    def client(_):
        yield SimpleNamespace(get=lambda *args, **kwargs: pytest.fail("Deadline already expired."))

    monkeypatch.setattr(prepublication, "local_client", client)
    report = prepublication.verify(path, tmp_path / "report.json")
    assert report["status"] == "failed"
    assert "readiness deadline" in report["error"]


def test_serving_verifier_is_included_in_the_engine_source_identity(monkeypatch, tmp_path):
    for path in Path(engine.__file__).parent.glob("*.py"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    before = source_hash()
    verifier = tmp_path / "prepublication.py"
    verifier.write_text(verifier.read_text() + "\n# A verifier change must invalidate old profiles.\n")
    assert source_hash() != before
