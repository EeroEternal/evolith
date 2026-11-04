# Self-directed evolution controller

## Principle

Evolith chooses what to optimize from measured workload evidence. A currently
frequent shape must not become a permanent hardcoded objective. Human work
extends the capability/tooling boundary, not a list of supposedly winning
hotspots.

Trusted infrastructure stays outside the search: independent correctness
checks, input immutability, execution failure handling, promotion thresholds,
measurement procedure and resource limits.

## Current automatic loop

1. Start a separate, bounded worker with frozen model weights.
2. Warm up the current incumbent, then observe actual normalization calls.
3. Count contracts and sample up to 16 inclusive CUDA-event durations per
   contract per rank, with at most 128 contracts. Average estimated cumulative
   cost across TP ranks instead of summing concurrent rank work.
4. Rank supported contracts by estimated cumulative cost, discounted by session
   attempts and recent matching archived cycles. Select the target from data,
   not a fixed batch size or residual flag.
5. Ask the local model for schema-constrained parameter proposals, including
   observed costs and recent outcomes. Try model suggestions first, then fill
   unused slots from the trusted finite parameter space. Do not repeat a
   target/configuration pair within the session.
6. Search only the selected exact contract. Rejected trials remain in history.
   A fresh-seed confirmation is required before any full-model replacement.
7. Compose confirmed changes with the incumbent's confirmed contracts. Preserve
   original confirmation records and identify the composition as lineage, not
   a new measurement.
8. Compare the proposed successor against the **current incumbent** using the
   same workload, weights and engine settings. Require the changed contract to
   execute on every TP rank. Baseline AB/BA trials reinstall the incumbent,
   not the original native implementation.
9. Independently recompute the fixed timing gate from raw paired samples in
   the controller. Adoption requires passing output regression, actual dispatch
   and the end-to-end gate. A worker's approval boolean alone is insufficient.
10. Keep the incumbent on rejection; re-observe and continue. Stop on a resource
    budget, exhausted supported search space, capability gap or execution error.

Sessions start from native execution. Approved artifacts are returned for
inspection/use, not silently deployed into an external service. An approved
incumbent within the session is used for subsequent observations and proposal
generation. Session restarts can reuse matching archived outcomes as guidance,
but do not automatically install an earlier session's artifact.

## Fixed limits and records

- Defaults: 3 cycles, 32 evaluations including confirmations, 1,800 seconds,
  4 candidate configurations per cycle.
- Maximums: 8 cycles, 128 evaluations, 7,200 seconds, 12 configurations.
- Before starting search, reserve enough evaluation capacity for independent
  confirmation of every requested configuration. Unused reserve is not a win.
- Workers use separate process groups with deadline termination. Allow a
  termination grace period; use the offline container's outer timeout to bound
  the entire process tree, including engine workers.
- Keep `EVOLITH_TIMEOUT` longer than `--time-budget` so cleanup can publish its
  final report. The helper's PID, memory and CPU limits still apply.
- Exact-target search and proposal validation do not allow generated Python,
  new imports, arbitrary launch dimensions or gate changes.
- Atomic session reports are progress snapshots. Per-cycle JSON and SQLite
  records preserve append-only outcomes and lineage.
- Historical guidance is bounded to recent cycles and matched by model
  fingerprint, workload fingerprint, kernel hash and runtime identity. GPU
  temperatures/clocks are not identity fields. Timing remains exploratory.

## Boundaries

This is an outer autonomous controller over a **bounded capability**, not
unrestricted self-modifying code:

- Observation currently covers GemmaRMSNorm. It cannot identify uninstrumented
  attention, communication, GEMM or scheduler bottlenecks. It must not claim
  whole-engine bottleneck discovery.
- Event samples include host launch gaps, add instrumentation overhead and
  initially sample each contract's first 16 calls after warmup. Their estimated
  cumulative costs rank experiments; they are not deployment evidence or
  exclusive device-kernel attribution.
- Non-supported observed contracts produce a capability gap rather than fake
  progress. Unknown, unobserved families need a future observation capability.
- Generation is serialized and eager; CUDA-graph-safe swapping, production
  load/SLOs and full quality benchmarks are not implemented.
- Candidate source still comes from the trusted template. Arbitrary source
  evolution and L2/L3 action capabilities remain future extensions.
- Fingerprints do not hash full model weights. Records are local trusted
  artifacts, not signed attestations.

Extending this system should add independently verifiable observation/action
capabilities. It should not hardcode the hotspot from the last experiment or
relax the admission gate when a search finds no improvement.
