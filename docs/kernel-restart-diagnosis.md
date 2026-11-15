# Bounded kernel-level restart diagnosis

Status: the authorized three-process hardware session completed. Drift was
reproduced and narrowed, but no causal operator or numerical fix was proven.
The published API and relay remain stopped. No baseline, threshold, precision,
weights, graph mode or performance admission is changed.

## Declared experiment

Use the pinned vLLM 0.19.0 image, Qwen3.5-27B BF16 and GPUs 0–3. A fresh
cache namespace is shared by exactly three server processes:

1. Cold cache with per-rank Inductor launch observation.
2. Warm cache with the same observation.
3. Warm cache with the same worker class but launch observation disabled.

Each process performs five complete canonical workloads. Observed processes
then tokenize public request 0 and arm a one-prefill readback. Workers require
the **actual scheduler prompt IDs** to equal the declared tokenized IDs and
require the complete prefill to be scheduled together. Decode activations are
not compared because greedy histories may diverge. The sampled request and a
subsequent full workload must still match the process's preceding outputs.
The no-hook control must match the observed warm process before local
attribution is considered. This control alone is insufficient: complete,
matching inputs and differing outputs are also required. This is not a
latency experiment.

The inner session budget is 1,680 seconds. The outer container deadline is
1,740 seconds, reserving termination/cleanup time below the user-approved
30-minute ceiling. There is no automatic retry or fallback. Startup, capture,
provenance or output-equality failures stop the session and preserve evidence.

## Observation and limits

`KernelTraceWorker` inherits the stock GPU worker. It installs a wrapper only
after normal device/distributed initialization. The wrapper calls the original
`CachingAutotuner.run` with the same arguments and returns its original result.
It records the selected launch configuration and compiled source identity on
each rank; it does not select configurations or disable compilation/graphs.
Function metadata is preserved for serialization/introspection.

After equilibration, the one armed prefill additionally hashes logical tensor
argument values before/after observed Inductor calls. Readbacks are copied to
CPU and never mutate model storage. Floating hashes normalize signed zero;
nonfinite and incomplete observations cannot establish a numerical difference.
Limits per rank are 4,096 events, 16 MiB event logs, 8 MiB per tensor and
1 GiB cumulative readbacks. Saved kernel sources are separately bounded.

This does **not** fully cover cuBLAS, FlashAttention, NCCL, direct Triton or
other custom operators. CUDA-graph-captured calls are not read back. The first
observed floating-output difference is not necessarily the first model-layer
difference. Matching observed inputs plus different complete outputs can
identify a local candidate; changed incoming values instead point somewhere
earlier, possibly outside coverage. Alignment gaps, skipped data, readback
synchronization and instrumentation effects remain explicit limitations.

No dynamic model hooks are inserted into the compiled graph. Nevertheless,
logging/readbacks can perturb scheduling, memory and autotuning. Reproducing
the cold/warm output failure and passing same-process/no-hook controls are
necessary checks, not proof of universal noninterference.

## Provenance

Reports use `kernel_restart_trial` and `kernel_restart_session`, not engine
promotion records. They cannot create `engine.json` or open a relay. The
diagnostic worker is a fixed trusted CLI choice, not a model proposal field.
The three observation modules have a separate combined source hash; the
trusted launcher change also changes the seven-file engine source identity.
Old artifacts are preserved and not rehashed into compatibility.

Local CPU validation before hardware: **129 passed, two skipped** (the CUDA
suite and optional Torch readback unit test). The pinned-image CPU run passed
**130 tests, one skipped**. Scoped Ruff passed.

## Hardware result, 2026-11-05

Session `20261105T124129Z-871a0cda5bab` completed in **774.44 seconds**
(785.5 seconds for the outer launch command), below the 30-minute ceiling.
Exactly three fresh servers ran on GPUs 0–3, with 104 canonical 64-token
requests in total:

| Phase | Trial |
| --- | --- |
| Cold, observed | `20261105T124133Z-74bc6b04acdb` |
| Warm, observed | `20261105T124939Z-70ec9bf468de` |
| Warm, no hook | `20261105T125205Z-61695724b535` |

- Cold/warm strict comparison **failed**: public request 2 first differed at
  token index 26; maximum common-history logprob difference across requests
  was approximately 0.12965, above 0.05.
- The warm/no-hook pair had **identical token IDs and logprobs** for all six
  requests. Both observed processes passed the sampled-request and
  post-sampling full-workload checks against their own pre-sampling outputs.
  These controls retained the failure; they do not prove universal
  noninterference.
- The same 31 actual prefill token IDs were verified on every observed rank.
  Each rank recorded 450 prefill launches per observed process, with no
  dropped events or capture skips. Each copied 206,005,308 bytes.
- Warm and no-hook logs explicitly load AOT artifacts. This is association
  with cache lifecycle, not proof that AOT alone causes the failure.

### Earliest confirmed observations, not a causal verdict

All indices below are zero-based **observed Inductor launch indices**, not
model layer indices:

| Rank | Initial RMSNorm `R0_BLOCK`, cold/warm | First differing floating input | First recorded differing floating output |
| --- | --- | --- | --- |
| 0 | 2048 / 2048 | 6 | 8 |
| 1 | 8192 / 8192 | 6 | 8 |
| 2 | 2048 / 8192 | 4 | 8 |
| 3 | 2048 / 2048 | 6 | 8 |

The initial RMSNorm at launch 1 had identical complete input hashes on both
sides, also equal across ranks. Its source identity is
`7b10b483e1476993e0513ec366676b0ff071082e759ab13265c39d36cae73a1c`.
Only rank 2 changed that launch's reduction block size; `XBLOCK=1`,
`num_warps=16` and `num_stages=1` remained equal. This makes that RMSNorm a
specific candidate, not a proven cause.

Rank 2 first exposes changed BF16 inputs of shape `[31, 12, 128]` at
`triton_per_fused_1` (launch 4). Ranks 0, 1 and 3 first expose a changed BF16
input of shape `[31, 5120]` at the subsequent fused residual/RMSNorm
(launch 6). At launch 8, outputs already receive changed inputs, so that
later output difference cannot establish a local cause there.

### Discovered observer gap and post-run repair

The measured observer used `mutated_arg_names` to select post-call buffers.
That list excludes ordinary, newly allocated `out_ptr*` outputs. Consequently
**386 of 450 events per rank had no output snapshot**, including the initial
RMSNorm. No event-budget drops occurred; this is a real coverage defect.
The original report's `first_coverage_gap: null` and
`attribution_eligible: true` were therefore overbroad, not causal evidence.

The original source, reports and four append-only records remain unchanged.
A separately identified derived analysis reports missing outputs from launch
0, the earlier incoming differences above, and **attribution ineligible**.
The observer now includes ordinary output pointers; analysis explicitly
reports empty outputs and incoming differences, and requires matched-input
output divergence as well as reproduction controls. Regression tests cover
both defects.

Post-repair CPU validation: **132 passed, two skipped locally**;
**133 passed, one skipped in the pinned image**. Scoped Ruff passed. The
existing AnyIO `CallSpec2` deprecation warning remains.
**The repaired observer has not been GPU-tested. No additional GPU run was
started.** A future authorized experiment should directly capture the initial
RMSNorm output and replay its same real inputs under both observed
configurations before attempting a numerical fix.

### Evidence and final state

Local and remote archive:
`results/kernel-localization-20261105.9sjk6204/`.
It contains the original source snapshot, four records, raw per-rank hashes,
kernel sources, logs, derived analysis and cleanup evidence. The cache manifest
covers 2,911 files totaling 1,203,430,012 bytes; 1,039 source/configuration
metadata files were copied. Cache binaries remain on the GPU host.

Measured engine source:
`21271a7650ff10c4ae54b299d437971c0582dd6f15049855444ff405970ee0b0`.
Measured observer source:
`e9c1916d7bb348f168e4fe6c4ea7fc7f3e95c5d0db4d89e483a9b4f9ff4be463`.
Post-repair observer/analysis source:
`246bdb633ae10f792619d4f51dd81e90efc6b3dc4f6bc351ab9687abd0fbbbeb`.
The three identities are not silently substituted.

Evidence archive SHA-256:
`80ccacb3601fd3f1e280e34b8c65b77c1b76b4ada8e0f005228eaaf8fadfc61e`.

After the hardware run, both protected service containers were still exited,
port 18080 had no listener, no lab containers or GPU compute processes remained,
and every GPU was at 2 MiB. No profile was created, old reference was not
replaced, and no deployment, commit or push was performed.

The invocation below now uses the **post-repair** observer, not the frozen
observer that produced the preceding hardware evidence:

```bash
EVOLITH_IMAGE=sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90 \
EVOLITH_GPUS=0,1,2,3 EVOLITH_MEMORY=192g EVOLITH_CPUS=24 \
EVOLITH_TIMEOUT=1740 bash scripts/container.sh python \
  -m evolith.lab.kernel_diagnosis --model-path /models/qwen \
  --output /results --cache-root /cache/kernel-diagnosis
```
