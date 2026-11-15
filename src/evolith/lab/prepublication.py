"""Check a fresh process against approved outputs before opening its relay."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import httpx

from evolith.lab.engine import EngineArtifact, validate_artifact
from evolith.lab.latency import (
    CHAT_PROMPTS,
    local_client,
    outputs_match,
    stream_request,
    workload_fingerprint,
)
from evolith.lab.store import write_json


def expected_proof(artifact: EngineArtifact) -> dict:
    validate_artifact(artifact)
    proof = (artifact.baseline_proof if artifact.selection == "native_baseline"
             else artifact.fresh_candidate_proof)
    if not proof or proof.get("status") != "measured" or proof.get("correct") is not True:
        raise ValueError("Missing measured serving-reference proof.")
    for key in ("source_hash", "model_fingerprint", "environment", "workload_fingerprint"):
        if proof.get(key) != getattr(artifact, key):
            raise ValueError(f"Serving-reference {key} differs from the artifact.")
    if proof.get("config") != artifact.config.model_dump():
        raise ValueError("Serving-reference configuration differs from the artifact.")
    tokens = proof.get("max_tokens")
    if (type(tokens) is not int or not 16 <= tokens <= 128
            or workload_fingerprint(tokens) != artifact.workload_fingerprint):
        raise ValueError("Invalid serving-reference workload.")
    records = proof.get("records", [])
    if (len(records) != len(CHAT_PROMPTS) or not outputs_match(records, records)
            or any(len(r["token_ids"]) != tokens
                   or any(type(t) is not int for t in r["token_ids"]) for r in records)):
        raise ValueError("Incomplete serving-reference token evidence.")
    return proof


def verify(profile: Path, output: Path, port: int = 8000) -> dict:
    report = {"status": "running", "correct": False, "warmup_passes": [],
              "scope": "prepublication correctness only; no performance admission"}
    started = time.monotonic()
    write_json(output, report)
    try:
        if not 1024 <= port <= 65535:
            raise ValueError("Invalid local verification port.")
        artifact = EngineArtifact.model_validate_json(profile.read_text())
        proof = expected_proof(artifact)
        report.update(source_hash=artifact.source_hash, reference_run_id=proof.get("run_id"),
                      selection=artifact.selection, config=artifact.config.model_dump())
        with local_client(port) as client:
            deadline = time.monotonic() + 600
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Prepublication server readiness deadline exceeded.")
                try:
                    if client.get("/health", timeout=2).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.5)
            for _ in range(5):
                records = [stream_request(client, prompt, proof["max_tokens"])
                           for prompt in CHAT_PROMPTS]
                report["warmup_passes"].append(records)
                write_json(output, report)
                if not outputs_match(proof["records"], records):
                    raise ValueError("Fresh-process outputs differ from the approved reference.")
        report.update(status="verified", correct=True)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    def expired(*_):
        raise TimeoutError("Prepublication wall-time budget exhausted.")

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(660)
    try:
        report = verify(args.profile, args.output, args.port)
    finally:
        signal.alarm(0)
    print(json.dumps({"report": str(args.output), "status": report["status"],
                      "correct": report["correct"]}))
    raise SystemExit(0 if report["correct"] else 1)


if __name__ == "__main__":
    main()
