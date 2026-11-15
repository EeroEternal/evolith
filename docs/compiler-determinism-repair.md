# Compiler determinism repair candidate

Status: **drift localized; no repair accepted**. The deterministic Inductor
policy alone did not stop restart drift. A diagnostic round then showed that,
with that policy plus identical recorded Triton autotune decisions for vLLM's
linear-attention kernels, six processes across three independently compiled
caches were bit-identical. No engine fix, new reference, optimization or
deployment is accepted; see the final section for scope and open decisions.

## Evidence and scope

The actual archived initial RMSNorm module from that failed run has
`deterministic=False`. With its preserved size hints and metadata, pinned
Torch 2.10 generates two legitimate configurations. Prior same-input GPU
replay already proved these configurations differ by one BF16 element.
Private cache directories cannot make their floating-point arithmetic agree.
This does not prove that all whole-model drift comes from this operator.

Pinned Torch supplies `TORCHINDUCTOR_DETERMINISTIC=1`. In a fresh interpreter
the flag enables the compiler's generic determinism policy. Its reduction
filter selects one existing configuration, and `CachingAutotuner` sets its
`deterministic_mode` from the generated metadata. That also disables dynamic
reduction-block expansion. The policy does not hardcode a shape, source hash,
rank's previously selected winner or particular block size in engine code.

## Implementation boundaries

- The trusted diagnostic launcher sets the flag **before** starting the
  server interpreter. It also supplies
  `inductor_compile_config={"deterministic": true, "combo_kernels": false}` through vLLM's
  `--compilation-config`; the real GPU run showed that the environment alone
  was insufficient, despite worker import-time checks succeeding.
- The explicit dictionary reaches vLLM's `compiler_config` and then
  `standalone_compile(options.config_patches)`. Existing graph options are
  preserved; the compilation schema is checked in the pinned CPU image.
  The exact intervening context that lost the environment-only setting has
  not been demonstrated. The startup exception described below confirms the
  explicit route reached Inductor's determinism guard, not model correctness.
- Normal engine launches explicitly keep the previous policy (`0`).
  The new candidate requires the diagnostic worker and an explicit Boolean.
- Each rank checks the actual Torch setting and fails if it is unsupported
  or ignored.
- After each restart, the controller checks generated reduction metadata as
  well. Missing or mismatched metadata is a failure, not assumed support.
- This candidate does not enable the rejected rank-cache policy, disable
  graphs/AOT, change BF16 weights or loosen output comparisons.
- No engine profile, admission or publication is produced. Source hashes
  include the new diagnostic policy verifier.

Tests cover policy routing without modifying parent environment, false or
missing worker reports, generated metadata mismatch, independent restart
orchestration and preservation of frozen outputs. Pinned-Torch CPU tests
check generic reduction selection at several sizes, disabled dynamic
expansion and flag loading in a fresh interpreter.

CPU preparation is archived at
`results/compiler-determinism-cpu-20261106.3vmlh4f6/`.
The proof script reads the preserved generated module; its output includes
the pinned filter, constructor and dynamic-scaling implementation for audit.
It has no GPU access and is not a numerical/performance gate.

## Hardware attempt and budget-blocked retry

A separately authorized four-GPU, 30-minute round used the fixed original
deadline in
`results/determinism-validation-20261106.swj8kgep/validation-budget.json`.

The environment-only trial `20261106T025412Z-0dd38a4b5203` completed all five
canonical warmup workloads. All four worker checks reported `True`, but
**all 60 generated reduction modules reported `deterministic=False`**.
The controller correctly failed session `20261106T025408Z-5ff8a4718dbb` before
starting its warm restart. No cold/warm, frozen-output, performance or
whole-engine numerical success is claimed.

The raw driver labelled its unfinished attempt `interrupted` after the
exception, while its overall result and session correctly say `failed`.
Those raw statuses remain unchanged; derived annotations explain the
bookkeeping limitation.

Evidence/source snapshots are on both hosts in the round archive. The first
evidence SHA-256 is
`30a71b2a4b8d27730dcc9146da836a3d24278da816253517fb5d37f6aa7ea0f1`.
Generated binaries remain in the remote cache; generated source and tuning
metadata are copied into the evidence archive.

The explicit-parameter correction subsequently passed **161 tests, seven
skipped locally** and **167 tests, one skipped in the pinned image**.
Scoped Ruff and whitespace checks passed. Besides the existing AnyIO warning,
the vLLM schema import emitted SWIG deprecation warnings. These are not
numerical validation failures.

The user explicitly permitted one corrected cold/warm retry within the
**same** original deadline. Its launcher found only **634 seconds** remaining,
below its **650-second** launch safety reserve, and refused to start.
`explicit-routing-retry/` preserves that source and blocked invocation.
No new budget was created, no threshold lowered, no warm/repeat trial or
further GPU experiment launched.

The API remained stopped and the frozen reference unchanged throughout this
round. The CPU-only wiring correction was not treated as an accepted repair.

## Explicit parameter startup conflict and compatibility retry

The next separately authorized round is archived at
`results/explicit-determinism-20261106.7ng78o_g/`.
Session `20261106T032044Z-567c2387592d`, trial
`20261106T032049Z-aa248bf73449`, failed startup after **133.16 seconds**,
before any public request or warmup completed.

The direct stack is:
`Scheduler.create_combo_kernel_nodes` → `speedup_by_combo_kernel` →
`benchmark_fused_nodes` → `benchmark_gpu` → `may_ban_benchmarking`.
Inductor correctly refused unvetted GPU benchmarking in deterministic mode.
No reduction-policy inventory could be completed before this startup failure.

Pinned vLLM automatically enables both `combo_kernels` and
`benchmark_combo_kernel` when neither is explicitly provided. Pinned Torch
only creates these timed combo nodes when `config.combo_kernels` is true.
The compatibility correction explicitly sets **only `combo_kernels=False`**
in addition to the existing `deterministic=True`. It does not whitelist
benchmarks, disable the determinism guard, select a shape-specific reduction
layout, or alter BF16, CUDA graphs/AOT, prompts or output gates.

CPU checks: **161 passed/eight skipped locally; 168 passed/one skipped in the
pinned image**. Tests verify that vLLM preserves explicit combo disabling
and that the benchmark guard still raises in deterministic mode.
The initial startup failure and source are immutable; its evidence SHA-256:
`7726466b25002a4c8d9b8b57a95b525311e8e9aac5b934f46fc153e0fdac94ba`.

The user explicitly permitted a compatibility retry **within this round's
original deadline**. Its budget is copied byte-for-byte, not restarted.
The retry is recorded separately under `compatibility-retry/`. Generated
metadata, independent cold/warm outputs and frozen values must all pass
before a fresh-cache repeat. The retry failed before any warm process or
repeat; no further GPU experiment was launched.

### Compatibility retry outcome

Session `20261106T033155Z-f365187f9153`, cold trial
`20261106T033200Z-1b9f9bbcea96`, finished after **480.28 seconds** including
launcher overhead. All **five canonical six-request passes** completed with
identical token IDs and logprobs across those passes. This is only
within-process evidence, not an independent restart result.

The generated-policy inventory contains **40 `True` and eight `False`**
reduction modules. The eight exceptions are two integer-sum helpers, each
present on four devices: one compares floating-point elements and converts
the predicates to `int64` before accumulating, the other sums the resulting
`int64` partial counts. Their floating-point inputs must not be confused
with floating-point reduction accumulators. Complete generated sources are
preserved in `policy-audit.json` and `cache-metadata/`.

The all-reductions metadata check is broader than a floating-reduction
check. It remained unchanged and fail-closed: no exception based on kernel
name, output dtype or observed result was introduced. Its raw exception and
both failed parent statuses are preserved. This limitation must be addressed
before treating the checker as a typed floating-reduction verifier; it is
not grounds to accept the model.

A separate **CPU-only, descriptive** comparison of the preserved cold
records against the unchanged old frozen reference also fails:

| Request (one-based) | Same token IDs | Max common-prefix logprob difference |
| --- | --- | --- |
| 1 | yes | 0.06202179 |
| 2 | yes | 0.02957985 |
| 3 | yes | 0.08797252 |
| 4 | no, first difference at token 12 | 0.07385504 |
| 5 | yes | 0.01004969 |
| 6 | yes | 0.04905996 |

Thus correcting the integer-helper check alone would **not** make this
candidate satisfy the frozen-output contract. Differences exceed the
unchanged `0.05` limit, and exact token identity fails. No failed report was
rewritten as passed. `derived-comparison.json` and `derived-summary.json`
record this additional evidence without creating admission artifacts.

The retry used engine source hash
`f53968df8fad1ba2fe28f329c890eab094f6389c1050118296af19abc82d54b5`
and diagnostic hash
`a74df8bc34dfbb0774b8cc6b7f35bc652408b40d2ef00b864ff9f90d334736ce`.
All source-manifest entries still matched after the run. The cache contains
2,135 files (616,223,231 bytes); 877 generated-source/metadata files were
copied into the round archive, with compiled binaries preserved remotely.

Final cleanup at **2026-11-06 03:46:19 UTC** confirmed that both published
containers remained exited, port 18080 had no listener, no lab container or
GPU compute process remained, and all eight GPUs used only 2 MiB each.
Both budget files have identical hashes and the frozen reference retains
SHA-256 `46751a4bcb1f4b0ede4d6880056b975deff960079a42e4cd7eb894e7cedb8387`.
No profile, promotion or publication occurred.

## Restart-stability round

Separately requested by the user: determine whether the candidate is stable
across restarts before any decision about a new reference. Archive:
`results/stability-validation-20261106.51fb4465/` (both hosts; final evidence
1,839 files, SHA-256
`0a53c4e9d5eeb42028ee1c276edf67406bf821838208aab0eb42764627d0cb0e`).
Four GPUs, fixed 1,740-second deadline; the old frozen values were recorded
descriptively only and were not a gate.

### Typed reduction check

The eight `False` modules come from vLLM's sampler helper
`batched_count_greater_than` (logprob rank counting), a separate
`torch.compile` outside the model compilation config. Why that compile sees
`config.deterministic=False` was not established. Integer addition is
associative even with wraparound, so reduction order cannot change those sums.

`reduction_policy_inventory` now runs conservative dtype propagation over each
generated kernel. A kernel is exempt from the flag only if every reduction is a
`tl.sum` over provably integer or Boolean values; any other reduction,
helper, atomic, unknown call, unsupported statement or missing pointer type
leaves it order-sensitive, requiring the requested flag. At least one
order-sensitive reduction must be verified. On archived real kernels it
exempts exactly the eight counting kernels, keeps all 40 floating reductions
of the compatibility retry, and still fails the environment-only round.
CPU: 175 passed/eight skipped locally, 182 passed/one skipped in the pinned
image; scoped Ruff and whitespace checks pass. Engine hash unchanged
(`f53968df…`); diagnostic hash `4f153ed8612a862e93f487254f2ae3aaa2da1d0cf23aad5296e534a83f60300d`.

### Outcome

The first launch invoked the non-executable `scripts/container.sh` directly
and failed with permission denied before any container started; it is
preserved as `launch-attempt-1*`. The relaunch reused the same budget file.

| Session | Processes | Generated policy | Cold vs warm |
| --- | --- | --- | --- |
| `20261106T045550Z-6ff18ae6ecb6` | cold `…045554Z-69ca5618b57e`, warm `…050333Z-ed2cb0bdc330` | 40 deterministic + 8/16 integer | bit-identical |
| `20261106T050610Z-5480cfae112a` | cold `…050612Z-d9083a87a9a5`, warm `…051148Z-06e3d938c7b2` | 40 deterministic + 8/16 integer | **failed**: request 3 diverges at token 15, max difference 0.109 |

Every process was bit-identical across its own five passes. Only one of the
six cross-process pairs (the first cold/warm pair) was bit-identical; the
others diverge at request 3 token 15 and/or request 4 token 12, with common
prefix differences up to 0.135. The driver correctly stopped with
`rejected_restart`; the four-process check did not run. All four processes
also differ from the old frozen values.

Because warm processes reused the same cached deterministic Inductor kernels
and still diverged from their cold process, the Inductor reduction policy
cannot be the remaining cause. Candidates outside it include per-process
Triton autotuning of non-Inductor kernels (for example the linear-attention
`chunk_*` kernels), library GEMM algorithm selection and collective
communication; none is proven. Making a new reference from this candidate
is not justified.

Final cleanup at 2026-11-06 05:24:28 UTC: both published containers exited,
no lab container, no GPU process, all eight GPUs at 2 MiB, no listener on
18080, source manifest unchanged, frozen reference SHA-256 unchanged.

## Autotune-decision diagnosis

Archive: `results/autotune-seed-20261106.eb1a5d5d/` (both hosts; final
evidence 1,074 files, SHA-256
`5319dbc2ead87cc24a5fcbccdf27a0a6ddd9cb7dff3afff27b70d931fc825ec6`).
Four GPUs, new fixed 1,740-second deadline; diagnosis only.

### Offline evidence

Pinned vLLM sets `TRITON_CACHE_AUTOTUNING=1`. Its linear-attention (FLA)
kernels use `triton.autotune`: Triton times every candidate configuration
and keeps the minimum, then persists the timings as `*.autotune.json`.
All four ranks share one file name per tuning key, so a cold process uses
each rank's own measurement while a later process reads whichever rank's
file was persisted.

Comparing the persisted files of the two stability sessions: **six of eight
tuned kernels selected different configurations** (for example
`chunk_fwd_kernel_o` with `BK=32` versus `BK=64`). In most cases the two best
candidates had identical measured times, so the choice is decided by timing
noise. Block sizes and warp/stage counts may change floating-point
accumulation order.

### Intervention

`kernel_repair.run(..., autotune_seed=...)` (opt-in, compiler policy only)
copies validated `*.autotune.json` decision files, and nothing else, into
each fresh Triton cache before the first process. Inductor still compiles
from scratch. After each process it verifies the seeded files are unchanged
and records any unseeded decisions. Seed: the eight files of stability
session `20261106T045550Z-6ff18ae6ecb6`. CPU: 181 passed/eight skipped
locally, 188 passed/one skipped in the pinned image. Diagnostic hash
`d50f6a56d1d63afd74814ce08d4b157c6e5d98a17a0297a74035c22047cfd652`; engine
hash unchanged.

### Result

| Session | Processes | Order-sensitive reductions deterministic | Unseeded decisions |
| --- | --- | --- | --- |
| `20261106T053647Z-7b99f237f988` | cold `…f87f6baeb1c1`, warm `…8785db04edb3` | 40, 40 | none |
| `20261106T054510Z-114b499a6347` | cold `…bfea7838aae0`, warm `…a3b7300b5a27` | 40, 40 | none |

All **6/6** pairs among the new processes and all **8/8** comparisons with
the two seed-session processes were bit-identical (same token IDs, zero
logprob difference). Without seeding, the previous round had only one
bit-identical pair out of six. The old frozen values still differ, as
expected for a different arithmetic path; they were descriptive only.

This identifies per-process Triton autotune selection as the remaining
drift source **for this workload, image, GPU set and TP4 configuration**.
Other shapes create other tuning keys that would again be tuned by timing.
Seeding fixed decisions is a diagnostic intervention, not an accepted engine
policy: its performance against the incumbent and the policy for new tuning
keys are not established, and any new reference requires explicit user
approval.

Cleanup at 2026-11-06 05:52:00 UTC: published containers exited, no lab
container or GPU process, all eight GPUs at 2 MiB, no listener on 18080,
source and seed manifests unchanged, frozen reference unchanged.
