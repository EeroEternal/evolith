# Evolith's governing principle

Evolith is intended to be a self-evolving inference optimizer, not a collection
of manually chosen kernel optimizations. Let measured workload observations
choose targets and let archived outcomes guide subsequent experiments. Do not
hardcode a currently frequent shape as the permanent optimization objective.

Correctness, independent measurement, admission rules, resource budgets and
fail-stop behavior are trusted infrastructure, not agent-editable rewards.
Never weaken them to obtain a winner. Microbenchmark wins are not deployment
approval. Compare a proposed successor against the current approved incumbent,
not just the original native implementation.

Keep capability boundaries explicit. An unsupported bottleneck is a capability
gap, not a successful optimization. Parameter search is not arbitrary source
evolution; partial operator observation is not a whole-engine profile.

Keep synthetic demos separate from real measurements. Preserve failed trials
and append-only lineage. Never cache outputs or mutate inputs to pass a gate.
Do not modify existing services or publish changes without user authorization.

Run CPU tests with `uv run pytest -q`; GPU validation requires the documented
offline container and compatible hardware. Follow README measurement limits.
