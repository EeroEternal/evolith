# Numerical repair attempt, 2026-11-05

Status: same-input RMSNorm configuration sensitivity verified; rank-local
cache isolation **rejected** in a separately authorized cold/warm regression.
The engine numerical issue remains unresolved. No accepted fix or deployment
is claimed.

This page records two separately authorized four-GPU, 30-minute rounds,
not an extension of one budget. In the first round, the initial GPU probe
created an absolute deadline, shared by all subsequent probes and repair
validation. Failed attempts did not reset it. The published model and relay
stay stopped; all model experiments are offline containers without host ports.

## Preserved failures

Two complete server starts passed their five canonical warmup workloads,
then failed in the added replay machinery:

1. `20261105T152613Z-fbd85de24043`: the AOT-deserialized Triton JIT function
   had no hash lock, preventing independent recompilation.
2. `20261105T153547Z-baecd6fbd6e8`: copying a live `CachingAutotuner` invoked
   its pickle protocol, which explicitly rejects existing launchers.

Both are observer failures, not successful model trials or evidence that the
numerical issue disappeared. The second attempt preserved the exact initial
RMSNorm inputs and actual outputs on all four ranks before failing. A first
standalone replay lacked an initialized CUDA context and also failed.
Both model-probe source snapshots and all failure logs remain archived.
The first failed standalone invocation had no dedicated prelaunch source
snapshot; the corrected successful version is preserved separately.

The replay path now reconstructs stripped JIT functions independently and
copies compiler state without invoking live-launcher pickling. Standalone
replay initializes the CUDA context before loading compiled kernels.

## Direct same-input proof

`standalone-context/report.json` uses the preserved real tensors and the
original generated kernel module/compile metadata. On **one GPU**, it runs
both observed configurations without starting another model server.
It reproduces the recorded actual output exactly for each rank's original
configuration.

- Inputs: identical across all four preserved ranks.
- Output: BF16, shape `[31, 5120]`, 158,720 elements.
- `R0_BLOCK=2048` versus `8192`: exactly **one element differs**,
  maximum absolute difference **0.001953125**.
- `XBLOCK`, warp/stage counts, weights, precision and source remain fixed.
- Source SHA-256:
  `7b10b483e1476993e0513ec366676b0ff071082e759ab13265c39d36cae73a1c`.

This proves local configuration-dependent arithmetic on real inputs. It does
not establish that this is the only source of whole-model restart drift.
The failed parent requests remain failed; standalone replay is not an engine
correctness or performance gate.

## Repair hypothesis

`RankLocalWorker` gives each rank private Inductor and Triton cache directories
before normal device setup and model loading. It does **not** choose a block
size, change arithmetic, disable CUDA graphs/AOT, or weaken output comparison.
The aim is to prevent shared autotuning state from replacing another rank's
selected configurations across restart.

The repair is explicitly opt-in through the trusted diagnostic launcher.
Normal engine launches retain their original worker. This remains an
experimental policy unless the independent cold/warm whole-workload pair
passes. The pair uses no launch-observation wrapper, five full workloads per
process, the unchanged BF16 configuration and the existing exact-token /
0.05-logprob gate.

Any historical frozen-output comparison is labelled descriptive and
cross-source, not artifact admission. The frozen artifact is never rewritten.
No `engine.json`, promotion, publication, commit or push is authorized.

Archive: `results/numerical-repair-20261105.s5vjt_un/` on both hosts.
The repair launcher refused to start when only 480 seconds remained, below
its 650-second launch reserve. It did not extend or reset the original
deadline. At that point the rank-cache hypothesis was untested on hardware;
the subsequent separately authorized validation is recorded below.

Final CPU checks: **141 passed, four skipped locally**; **144 passed, one
skipped in the pinned image**. Scoped Ruff and `git diff --check` passed.
The CPU container needed an explicit writable compiler-cache/user environment;
the initial harness failure and correction are not a model numerical result.
The pre-existing AnyIO `CallSpec2` deprecation warning remains.

Cleanup verified at **2026-11-05T15:50:32Z**: protected model and relay
containers exited, port 18080 had no listener, no lab container or GPU compute
process remained, and all eight GPUs were at 2 MiB. Old frozen outputs and
failed statuses were not changed. No profile, deployment, commit or push
was performed.

`round-evidence.tar.gz` contains raw reports, saved tensors and failure logs;
its SHA-256 is
`fdd97dd75d207962e8902a2e27844a10a4f7f87488b235c98dd0dd99adf396ad`.
The initial, corrected-probe, candidate and final source snapshots remain
separate from those immutable raw measurements.

## Separately authorized rank-cache validation: rejected

The user explicitly authorized another four-GPU, 30-minute round, restricted
to testing this candidate. A new budget file was created; the original
budget and frozen reference were not reset or changed.

- Archive: `results/rank-cache-validation-20261105.bxsqjge4/` on both hosts.
- Session: `20261105T160310Z-cad7d3daf134`, status `repair_rejected`.
- Cold: `20261105T160314Z-0696d91d99b9`.
- Warm: `20261105T161103Z-f583673a0f3d`.
- Elapsed: **607.09 seconds**, including the container lifecycle.
- Both processes passed their own five complete canonical warmup workloads.
- No launch-observation hook was installed. Precision, graphs, AOT, public
  prompts, sampling policy and comparison thresholds remained unchanged.
- Every rank confirmed its own Inductor/Triton paths; each private rank
  directory actually contained 2,227 generated files.

The strict cold/warm comparison **failed**. Request index 2 first diverged
at token index **14** (the third request, fifteenth token). Requests 0–3
also exceeded the 0.05 logprob tolerance on their comparable common histories;
the maximum was **0.08850997686386108**. The other five requests had matching
token IDs, but matching IDs alone do not satisfy the gate.

Both descriptive comparisons against the historical frozen outputs also
failed. They remain cross-source, non-admission comparisons, not replacement
references or approvals. No performance win is claimed after correctness
failure.

This result rejects rank-local Inductor/Triton cache isolation **as a
sufficient repair**. It does not negate the direct RMSNorm configuration
sensitivity proof, or establish that every cache-related cause is excluded.
Prior replay records also had distinct per-rank generated-module keys, so
cross-rank cache-key collision was a hypothesis, not a demonstrated cause.
No additional GPU experiment or new switch was tried after rejection.

Raw reports, logs, ownership records and 2,481 copied source/tuning-metadata
files are in `evidence.tar.gz` (2,502 files overall). Generated-cache inventory
covers 9,010 files / 1,476,201,676 bytes; compiled binaries remain preserved
in the remote cache. Evidence SHA-256:
`51ba9b4ab0017ee8b28e819e5745c926c05a399fe21ca150023c4304ecb81d52`.

Cleanup reverified at **2026-11-05T16:15:48Z**: both protected services exited,
no port-18080 listener, no lab container or GPU compute process, all eight
GPUs at 2 MiB. The frozen-reference SHA-256 remained
`46751a4bcb1f4b0ede4d6880056b975deff960079a42e4cd7eb894e7cedb8387`.
No profile, promotion, publication, commit or push occurred.
