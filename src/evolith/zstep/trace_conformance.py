"""vstack Layer-3 Runtime Trace Conformance Checker for Evolith × zstep.

Bridges static formal models and real GPU execution artifacts (`parity.json` and
`metrics.json`), enforcing three distinct outcomes:
- ``CONFORMANT``: all required runtime execution paths were exercised and obeyed
  safety/invariance/conservation properties;
- ``VIOLATION``: a runtime trace explicitly broke KV page conservation, batch-mode
  invariance, step-overlap bounds, or zero-JIT-recompilation invariants;
- ``INCONCLUSIVE_INCOMPLETE_TRACE``: trace artifact is missing, truncated, or failed
  to exercise required schedule states (never silently counted as ``CONFORMANT``).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TraceConformanceError(RuntimeError):
    pass


class TraceCheckClause(BaseModel):
    clause_id: str
    pillar: str
    verdict: Literal["CONFORMANT", "VIOLATION", "INCONCLUSIVE_INCOMPLETE_TRACE", "NOT_APPLICABLE"]
    passed: bool
    detail: str = ""
    witness: str | None = None


class TraceConformanceReport(BaseModel):
    passed: bool
    gate_verdict: Literal["CONFORMANT", "VIOLATION", "INCONCLUSIVE_INCOMPLETE_TRACE"]
    parity_sha256: str = ""
    metrics_sha256: str = ""
    clauses: list[TraceCheckClause] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


def _sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_trace_conformance(
    parity_path: Path | None,
    metrics_path: Path | None = None,
    *,
    evidence_dir: Path | None = None,
    max_cross_mode_logit_drift: float = 8.0,
    require_schedule_exposure: bool = True,
    raise_on_failure: bool = True,
) -> TraceConformanceReport:
    """Validate real GPU parity and benchmark execution traces against formal invariants."""
    clauses: list[TraceCheckClause] = []

    if parity_path is None or not parity_path.is_file():
        clauses.append(
            TraceCheckClause(
                clause_id="TRACE_PARITY_ARTIFACT_PRESENT",
                pillar="Completeness",
                verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                passed=False,
                witness="Missing runtime parity trace JSON artifact",
            )
        )
    else:
        try:
            payload = json.loads(parity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            clauses.append(
                TraceCheckClause(
                    clause_id="TRACE_PARITY_ARTIFACT_PRESENT",
                    pillar="Completeness",
                    verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                    passed=False,
                    witness=f"Malformed parity trace JSON: {exc}",
                )
            )
            payload = None

        if isinstance(payload, dict):
            # 1. Zero Triton JIT recompilation
            if "triton_recompilations" not in payload:
                clauses.append(
                    TraceCheckClause(
                        clause_id="TRACE_JIT_ZERO_RECOMPILATIONS",
                        pillar="Determinism",
                        verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                        passed=False,
                        witness="Missing `triton_recompilations` counter in parity trace",
                    )
                )
            else:
                recomp = int(payload["triton_recompilations"])
                clauses.append(
                    TraceCheckClause(
                        clause_id="TRACE_JIT_ZERO_RECOMPILATIONS",
                        pillar="Determinism",
                        verdict="CONFORMANT" if recomp == 0 else "VIOLATION",
                        passed=(recomp == 0),
                        detail=f"triton_recompilations={recomp}",
                        witness=(
                            None
                            if recomp == 0
                            else f"Runtime triggered {recomp} unexpected Triton JIT recompilations"
                        ),
                    )
                )

            # 2. Schedule exposure (single -> batched -> chunked + mixed decode)
            chunked_sched = payload.get("chunked_schedule")
            if not isinstance(chunked_sched, dict):
                if require_schedule_exposure:
                    clauses.append(
                        TraceCheckClause(
                            clause_id="TRACE_SCHEDULE_EXPOSURE",
                            pillar="Exposure",
                            verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                            passed=False,
                            witness="Missing `chunked_schedule` telemetry in parity trace",
                        )
                    )
                else:
                    clauses.append(
                        TraceCheckClause(
                            clause_id="TRACE_SCHEDULE_EXPOSURE",
                            pillar="Exposure",
                            verdict="NOT_APPLICABLE",
                            passed=True,
                            detail="Schedule exposure check disabled",
                        )
                    )
            else:
                cp_steps = int(chunked_sched.get("chunked_prefill_steps", 0))
                mixed_steps = int(chunked_sched.get("decode_steps_with_partial_prefill", 0))
                exposed = cp_steps > 0 and mixed_steps > 0
                clauses.append(
                    TraceCheckClause(
                        clause_id="TRACE_SCHEDULE_EXPOSURE",
                        pillar="Exposure",
                        verdict="CONFORMANT" if exposed else "INCONCLUSIVE_INCOMPLETE_TRACE",
                        passed=exposed,
                        detail=(
                            f"chunked_prefill_steps={cp_steps}, "
                            f"decode_steps_with_partial_prefill={mixed_steps}"
                        ),
                        witness=(
                            None
                            if exposed
                            else (
                                f"Trace did not exercise mixed chunked schedule "
                                f"(chunked_prefill_steps={cp_steps}, mixed_decode_steps={mixed_steps})"
                            )
                        ),
                    )
                )

            # 3. Cross-mode batch-composition invariance
            cases = payload.get("cases")
            if not isinstance(cases, list) or not cases:
                clauses.append(
                    TraceCheckClause(
                        clause_id="TRACE_BATCH_COMPOSITION_INVARIANCE",
                        pillar="Invariance",
                        verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                        passed=False,
                        witness="Parity trace contains zero verification cases",
                    )
                )
            else:
                prime_case = next(
                    (c for c in cases if isinstance(c, dict) and c.get("name") == "prime_squares_table"),
                    None,
                )
                if prime_case is None:
                    clauses.append(
                        TraceCheckClause(
                            clause_id="TRACE_BATCH_COMPOSITION_INVARIANCE",
                            pillar="Invariance",
                            verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                            passed=False,
                            witness="Missing `prime_squares_table` canary case in parity trace",
                        )
                    )
                else:
                    inv_err: str | None = None
                    incomplete_err: str | None = None
                    max_seen_drift = 0.0
                    for mode in ("single", "batched", "chunked"):
                        m_obj = prime_case.get(f"{mode}_match")
                        if not isinstance(m_obj, dict) or "exact_token_match_rate" not in m_obj:
                            incomplete_err = f"Missing `{mode}_match.exact_token_match_rate`"
                            break
                        rate = float(m_obj["exact_token_match_rate"])
                        if rate != 1.0:
                            inv_err = f"prime_squares_table {mode} exact_token_match_rate={rate:.4f} != 1.0"
                            break

                    if inv_err is None and incomplete_err is None:
                        if (
                            prime_case.get("single_vs_batched_first_mismatch") is not None
                            or prime_case.get("batched_vs_chunked_first_mismatch") is not None
                        ):
                            inv_err = "prime_squares_table diverged across single/batched/chunked modes"

                    if inv_err is None and incomplete_err is None:
                        for case in cases:
                            if not isinstance(case, dict):
                                continue
                            cname = str(case.get("name", "unknown"))
                            for plabel in ("single_vs_batched", "batched_vs_chunked"):
                                pair = case.get(f"{plabel}_first_mismatch")
                                if pair is None:
                                    continue
                                if not isinstance(pair, dict):
                                    incomplete_err = f"{cname} {plabel} malformed mismatch object"
                                    break
                                left = pair.get("left_top2")
                                right = pair.get("right_top2")
                                if (
                                    not isinstance(left, dict)
                                    or not isinstance(right, dict)
                                    or "top1_logit" not in left
                                    or "top1_logit" not in right
                                ):
                                    incomplete_err = f"{cname} {plabel} missing top1_logit"
                                    break
                                lval = float(left["top1_logit"])
                                rval = float(right["top1_logit"])
                                if not (math.isfinite(lval) and math.isfinite(rval)):
                                    inv_err = f"{cname} {plabel} non-finite logit ({lval}, {rval})"
                                    break
                                drift = abs(lval - rval)
                                max_seen_drift = max(max_seen_drift, drift)
                                if drift > max_cross_mode_logit_drift:
                                    inv_err = (
                                        f"{cname} {plabel} logit drift {drift:.2f} > "
                                        f"{max_cross_mode_logit_drift:.2f}"
                                    )
                                    break
                            if inv_err or incomplete_err:
                                break

                    if incomplete_err:
                        clauses.append(
                            TraceCheckClause(
                                clause_id="TRACE_BATCH_COMPOSITION_INVARIANCE",
                                pillar="Invariance",
                                verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                                passed=False,
                                witness=incomplete_err,
                            )
                        )
                    elif inv_err:
                        clauses.append(
                            TraceCheckClause(
                                clause_id="TRACE_BATCH_COMPOSITION_INVARIANCE",
                                pillar="Invariance",
                                verdict="VIOLATION",
                                passed=False,
                                witness=inv_err,
                            )
                        )
                    else:
                        clauses.append(
                            TraceCheckClause(
                                clause_id="TRACE_BATCH_COMPOSITION_INVARIANCE",
                                pillar="Invariance",
                                verdict="CONFORMANT",
                                passed=True,
                                detail=(
                                    f"prime_squares_table 100% invariant across 3 modes; "
                                    f"max_logit_drift={max_seen_drift:.3f} <= {max_cross_mode_logit_drift:.2f}"
                                ),
                            )
                        )

    # 4. Benchmark metrics trace: KV conservation & step overlap bounds
    if metrics_path is not None:
        if not metrics_path.is_file():
            clauses.append(
                TraceCheckClause(
                    clause_id="TRACE_KV_AND_STEP_CONSERVATION",
                    pillar="Conservation",
                    verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                    passed=False,
                    witness=f"Missing benchmark metrics file {metrics_path.name}",
                )
            )
        else:
            try:
                m_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                clauses.append(
                    TraceCheckClause(
                        clause_id="TRACE_KV_AND_STEP_CONSERVATION",
                        pillar="Conservation",
                        verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                        passed=False,
                        witness=f"Malformed benchmark metrics JSON: {exc}",
                    )
                )
                m_payload = None

            if isinstance(m_payload, dict):
                tiers = m_payload.get("tiers")
                if not isinstance(tiers, list) or not tiers:
                    clauses.append(
                        TraceCheckClause(
                            clause_id="TRACE_KV_AND_STEP_CONSERVATION",
                            pillar="Conservation",
                            verdict="INCONCLUSIVE_INCOMPLETE_TRACE",
                            passed=False,
                            witness="Benchmark metrics trace contains zero tier rows",
                        )
                    )
                else:
                    kv_violation: str | None = None
                    for row in tiers:
                        if not isinstance(row, dict):
                            continue
                        lbl = str(row.get("label", "tier"))
                        idle_kv = row.get("idle_kv_cache_usage")
                        if idle_kv is not None and float(idle_kv) > 1e-6:
                            kv_violation = (
                                f"Tier {lbl} leaked KV blocks after retirement: "
                                f"idle_kv_cache_usage={float(idle_kv):.6f} > 0"
                            )
                            break
                        steps = row.get("steps")
                        ov_steps = row.get("overlapped_decode_steps")
                        if steps is not None and int(steps) <= 0:
                            kv_violation = f"Tier {lbl} recorded non-positive steps={steps}"
                            break
                        if steps is not None and ov_steps is not None:
                            if int(ov_steps) < 0 or int(ov_steps) > int(steps):
                                kv_violation = (
                                    f"Tier {lbl} invalid overlap accounting: "
                                    f"overlapped_decode_steps={ov_steps} out of [0, {steps}]"
                                )
                                break
                    clauses.append(
                        TraceCheckClause(
                            clause_id="TRACE_KV_AND_STEP_CONSERVATION",
                            pillar="Conservation",
                            verdict="VIOLATION" if kv_violation else "CONFORMANT",
                            passed=(kv_violation is None),
                            detail=f"Checked {len(tiers)} tier trace rows for KV zero-leakage & step bounds",
                            witness=kv_violation,
                        )
                    )

    has_violation = any(c.verdict == "VIOLATION" for c in clauses)
    has_incomplete = any(c.verdict == "INCONCLUSIVE_INCOMPLETE_TRACE" for c in clauses)
    if has_violation:
        gate_verdict: Literal["CONFORMANT", "VIOLATION", "INCONCLUSIVE_INCOMPLETE_TRACE"] = "VIOLATION"
    elif has_incomplete:
        gate_verdict = "INCONCLUSIVE_INCOMPLETE_TRACE"
    else:
        gate_verdict = "CONFORMANT"

    passed = gate_verdict == "CONFORMANT"
    violations = [
        f"{c.clause_id} [{c.verdict}]: {c.witness}"
        for c in clauses
        if not c.passed
    ]
    report = TraceConformanceReport(
        passed=passed,
        gate_verdict=gate_verdict,
        parity_sha256=_sha256_file(parity_path),
        metrics_sha256=_sha256_file(metrics_path),
        clauses=clauses,
        violations=violations,
    )
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "trace-conformance.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if raise_on_failure and not passed:
        raise TraceConformanceError(
            f"Trace conformance {gate_verdict}: " + "; ".join(violations)
        )
    return report
