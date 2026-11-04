from __future__ import annotations

import math
import random
import statistics

from evolith.lab.contracts import Timing


def decide(
    baseline: list[float],
    candidate: list[float],
    *,
    min_gain: float = 0.03,
    max_cv: float = 0.10,
    seed: int = 0,
) -> Timing:
    """Paired bootstrap of median log speedup, lower one-sided 99% bound."""
    if len(baseline) != len(candidate) or len(baseline) < 5:
        raise ValueError("At least five paired timing samples are required.")
    if any(not math.isfinite(v) or v <= 0 for v in baseline + candidate):
        raise ValueError("Timings must be finite and positive.")
    if not 0 <= min_gain < 1 or not 0 < max_cv < 1:
        raise ValueError("Invalid measurement thresholds.")
    ratios = [math.log(a / b) for a, b in zip(baseline, candidate)]
    rng = random.Random(seed)
    boot = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)
    )
    lower = math.exp(boot[19])
    speedup = math.exp(statistics.median(ratios))
    base_cv = statistics.stdev(baseline) / statistics.mean(baseline)
    cand_cv = statistics.stdev(candidate) / statistics.mean(candidate)
    reasons = []
    if lower <= 1 + min_gain:
        reasons.append(f"99% bootstrap lower bound {lower:.4f} <= {1 + min_gain:.4f}")
    if max(base_cv, cand_cv) > max_cv:
        reasons.append(f"Timing CV exceeds {max_cv:.1%}; rerun on an idle GPU.")
    return Timing(
        baseline_us=baseline,
        candidate_us=candidate,
        speedup=speedup,
        lower_bound=lower,
        baseline_cv=base_cv,
        candidate_cv=cand_cv,
        accepted=not reasons,
        reasons=reasons,
    )
