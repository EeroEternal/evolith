# Restart reproducibility investigation

The user authorized a subsequent bounded
[kernel-level diagnosis](kernel-restart-diagnosis.md). The conclusions and
snapshots below describe the preceding four-session round, not a resolved
numerical fix.

Status: the four bounded GPU diagnostic sessions are complete. The numerical
restart problem is **not fixed**; disabling AOT and enabling batch invariance
were not sufficient. Fail-stop profile/publication fixes are implemented.
The published API
`evolith-qwen` and its relay remain stopped. Diagnostic servers run only inside
an offline container; no host port is published.

## Evidence and hypothesis

The previous live native TP=4 process was internally repeatable but differed
from the frozen native reference on two of six greedy token sequences.
Other requests exceeded the fixed 0.05 logprob tolerance. The cause was not
established; neither deployment acceptance nor acceleration was claimed.

The installed vLLM 0.19.0 documentation states that seeds alone do not guarantee
online reproducibility. Its documented online mechanism is batch invariance:

- [v0.19.0 reproducibility](https://docs.vllm.ai/en/v0.19.0/usage/reproducibility/)
- [v0.19.0 batch invariance](https://docs.vllm.ai/en/v0.19.0/features/batch_invariance/)

The image contains `VLLM_BATCH_INVARIANT`. Version 0.19.0 documents compute
capability >=9.0 and tested server GPUs/models; this is **not** a certification
of Qwen3.5-27B on consumer SM120. Batch invariance can change numerical kernels
and performance. It is a hypothesis to test, not an automatic correctness pass.

## Predeclared diagnostic plan

`evolith.lab.restart` runs at most five fresh API processes on the same four
GPUs, model weights, BF16 dtype, 2048 context and native graph configuration:

1. Native mode, canonical complete-workload warmup.
2. Native mode again, identical warmup.
3. Native mode, first handle the same additional short request used in the
   earlier deployment check, then canonical warmup.
4. Batch-invariant mode, canonical warmup.
5. Batch-invariant mode with that initial short request.

Every case retains up to five complete warmup passes with token IDs and
logprobs, compares consecutive passes, and uses the existing `outputs_match`
rule unchanged. The six public requests still generate exactly 64 tokens
with EOS ignored, temperature zero and thinking disabled. The short
perturbation request has a separately declared 32-token limit.

Within-process output regression stops that case. Startup failure is recorded,
and remaining cases of that unsupported policy are explicitly skipped. Other
execution errors stop the diagnosis. A 3600-second wall-time budget and outer
container timeout bound the initial run.

Reports compare both within a policy across fresh starts and against the old
frozen native reference. An invariant-mode pair that matches each other but
fails the old-reference gate must **not** be called an approved successor.
No reference outputs, thresholds, prompts or old artifacts are rewritten.

## Provenance and boundaries

This probe has a separate source hash. The first experiment and the isolated
cache experiment keep all six original model/measurement files unchanged on
the GPU machine, allowing direct investigation against the old source identity.
A locally staged explicit-backend launcher change will be synced only after
these old-source controls finish and are archived. It creates a new source
identity; the frozen-reference compatibility check is not relaxed.
The report's `effective_environment` field captures the parent
environment at launch; vLLM may apply additional internal settings afterward,
so it is not a complete worker-state fingerprint.

This is diagnostic infrastructure, not model-generated code or an expanded
optimization reward. Reports use `engine_restart_trial` and
`engine_restart_session` records and cannot authorize `serve-engine`.
The recorded latency medians are diagnostic observations, not admitted
performance improvements.

## Running the probe

Historical first run, with the archived original six-file source identity:

```bash
EVOLITH_GPUS=0,1,2,3 EVOLITH_MEMORY=192g EVOLITH_CPUS=24 \
EVOLITH_TIMEOUT=3700 bash scripts/container.sh python -m evolith.lab.restart \
  --model-path /models/qwen --output /results --policy both --seconds 3600 \
  --reference /results/20261105T045220Z-89d9b9feeb49/report.json
```

## First hardware findings

Session `20261105T105407Z-b11a986fb378` completed in 640.64 seconds.
All three native processes were internally consistent and passed the original
exact-token/0.05-logprob rule against each other. The largest inter-process
logprob difference was 0.000020862. The initial short request did not change
the observed outputs. All three still failed the frozen reference, with the
same token divergences at request 2/token 28 and request 3/token 14 as the
earlier deployment. This does not establish general restart nondeterminism:
the old-reference discrepancy is stable in these three new starts.

The invariant process failed because vLLM 0.19.0 requires an **explicit**
attention backend when `VLLM_BATCH_INVARIANT=1`, even though native mode
automatically selects `FLASH_ATTN`. The error reports backend `None`; this is
not evidence that SM120 or Qwen3.5 is unsupported. The second invariant case
was skipped. A corrected probe must pass `--attention-backend FLASH_ATTN`.

The five append-only records, raw logs, warmup outputs and exact source bundle
were backed up under `results/restart-20261105.85zjf9eg`.

### Next bounded isolation test

Before changing the six original hashed files, repeat the three native cases
with a newly created cache namespace:

```bash
EVOLITH_GPUS=0,1,2,3 EVOLITH_MEMORY=192g EVOLITH_CPUS=24 \
EVOLITH_TIMEOUT=3700 bash scripts/container.sh python -m evolith.lab.restart \
  --model-path /models/qwen --output /results --policy native --seconds 3600 \
  --reference /results/20261105T045220Z-89d9b9feeb49/report.json \
  --isolated-cache-root /cache/restart-diagnostics
```

The first process sees empty Inductor, Triton and vLLM cache directories. The
second and third reuse this same new namespace. Existing caches are neither
deleted nor modified by this isolation setting. Cache paths and launch
environment are retained. This tests the cache hypothesis without changing
the old source identity, reference outputs or admission rules. It is not an
optimization trial. The process count and time budget remain bounded.

CPU cache-isolation tests: five passed; scoped lint and diff checks passed.
No restart fix or acceleration is claimed yet.

An available fallback control is `--policy collective`: four fresh native processes,
pinning only `NCCL_ALGO=allreduce:ring` or `allreduce:tree`, each with and without
the short initial request. Other NCCL settings remain untouched. These
algorithm choices can change BF16 reduction order; they are diagnostic
variants, not approved native-equivalent configurations. They use the existing
shared caches, retain the old six-file source identity on the GPU machine,
and compare to the unchanged frozen reference under the same strict gate.

### Isolated-cache findings and next control

Session `20261105T111233Z-5045ce0f98f7` completed in 758.06 seconds, with three
internally consistent native processes. The first cold-cache process differed
from both later starts: request 2 diverged at token 28 and logprob differences
exceeded 0.05 on multiple requests. The second and third matched each other.
All three still failed the old frozen reference. Logs show the first process
**saving** an AOT compiled function and the later processes **directly loading**
that function on all four ranks. This narrows the investigation to cache/lifecycle
behavior; it does not yet prove a particular compiler defect.

The four append-only records, source snapshot, raw outputs/logs and a manifest
of the 2,911 cache files (1,217,288,798 bytes) are archived under
`results/restart-cache-20261105.ydbh0dzp`. Compiled cache files remain preserved
on the remote machine; the local manifest is not a copy of the binaries.

The next prioritized control is `--policy no_aot --isolated-cache-root
/cache/restart-diagnostics`, with the same three native cases and a fresh
namespace. It changes only `VLLM_USE_AOT_COMPILE=0` relative to the previous
native isolation test. Inductor/Triton compilation and CUDA graphs remain
enabled. This control keeps the original six hashed files and frozen
reference unchanged on the GPU machine. It has a 1,800-second inner budget
and 1,900-second outer container deadline. NCCL algorithm controls are
deferred until this more direct cache hypothesis is tested.

## Fail-closed publication fix

The controlled deployment helper previously opened its relay after checking
only historical profile validity. A fresh process could therefore become
reachable before its outputs were checked. The updated helper starts only
the private model, runs `evolith.lab.prepublication`, and opens the relay only
after all five full workloads match the selected artifact's approved output.
It retains failed output evidence and cleans up only newly created resource IDs.
The verifier uses the unchanged exact-ID/0.05-logprob rule and also checks proof
provenance, workload completeness and deadlines.

This is a safety fix, not a numerical restart fix or a new performance proof.
It has CPU regression tests but has not restarted the published services.
The verifier is included in the new seven-file engine source hash, so old
profiles cannot authorize this changed source. The old-source GPU controls
continue on the original six-file snapshot; the new launcher and verifier
are not synced over a running diagnostic.

Validation of the publication guard: the complete local CPU suite and an
isolated, GPU-unassigned container using the pinned image both passed
**110 tests**, with the CUDA suite skipped. The first image test invocation
had four harness failures because `/tmp` was mounted `noexec` and the working
directory was read-only; using an executable private `/tmp` as the test working
directory resolved them without code changes. One existing AnyIO/pytest
deprecation warning remains. The corrected invariant-backend routing then
passed the focused 28-test restart/publication/service suite locally.

## No-AOT result and bounded invariant follow-up

Session `20261105T112736Z-914511dbb5c1` completed in 764.09 seconds. Disabling
AOT alone did **not** fix cold/warm restart drift. Request 2 diverged at token
26 and request 3 at token 14 between the first and subsequent processes;
several logprob differences also exceeded 0.05. Each process was internally
consistent. All three failed the original frozen reference. This rules out
disabling AOT alone as a sufficient fix, not every possible AOT contribution.

Raw evidence and the exact old-source snapshot are retained in
`results/restart-no-aot-20261105.1_wm28gd`. The remaining Inductor cache contains
172 `.best_config` and 92 `.kernel_perf` files. Their presence is a lead for
further autotuning investigation, not proof that one particular kernel caused
the drift. No unsupported causal claim or numerical fix is made.

The next bounded run uses the corrected explicit `FLASH_ATTN` backend and
`--policy batch_invariant --isolated-cache-root /cache/restart-diagnostics`,
with two fresh processes (canonical then short-request-perturbed warmup), a
1,800-second inner budget and a 1,900-second outer deadline. The new launcher
and publication guard change the engine source identity. Therefore this run
does **not** pass the old source-incompatible report as `--reference`; the
compatibility check remains strict. Raw outputs may be compared descriptively
with historical outputs, but such a cross-source analysis cannot authorize a
promotion or replace the old reference. No profile is created by this probe.

## Native reference certification fix

The controller now requires a separately measured native restart before it
writes even a native-baseline profile or starts candidate search. The two
trials must have distinct IDs, matching source/model/runtime/workload/config
and unchanged strict token/logprob agreement. If they differ, the session
fails with **no profile**, rather than accepting the later outputs as a new
reference. Startup failure or exhausted certification budget likewise cannot
emit a fallback native profile. Artifact validation rechecks the certificate.

This is another fail-stop infrastructure fix, not a claim that backend
numerical nondeterminism is solved. It uses the existing time budget and keeps
the additional certification timings separate from paired performance samples.
It was staged locally while the invariant probe ran on its frozen
`a61d32a5c1412132aff904a9d3c642f384473a0fce1d4730d4c47ea137b7c29b`
source bundle. The running probe was not edited.

Local validation after adding native restart certification: **118 passed,
one CUDA suite skipped**, with scoped Ruff, shell syntax and diff checks
passing. The new certificate has not yet been exercised by a real controller
run; the currently running invariant probe is diagnostic-only.

The two warmed processes were also compared directly from each archived cache
experiment, not merely inferred from their comparisons to the cold process.
In both the native-isolated and no-AOT runs, all six warmed token sequences
and every generated-token logprob were exactly identical. Installed PyTorch
2.10.0 cache code explicitly restores saved launch configurations; the no-AOT
cache stores configurations with 2, 4, 8 and 16 warps. This is mechanism evidence
for the autotuning hypothesis, not attribution to a particular kernel or proof
of a rank-level race.

## Final invariant result and outcome

Session `20261105T114430Z-501cb1587298` completed in 847.61 seconds.
Both invariant servers started with explicit `FLASH_ATTN`; each was internally
consistent over all five workload passes. Across the two fresh processes,
request 2 diverged at token 28 and request 3 at token 58, with several logprob
differences above 0.05. The corrected invariant setting is therefore **not a
verified fix** on this Qwen3.5-27B/SM120/runtime combination. This two-case
control changes both cache lifecycle and the initial short request, so it does
not isolate which of those factors caused this invariant-mode discrepancy.

The three records, source bundle, logs, full outputs, historical native report
and 348 saved tuning configurations from the invariant/no-AOT namespaces are
retained in `results/restart-invariant-20261105.st_zxtje`. Descriptive comparisons
also show both invariant processes differ from the old native outputs. They
cross source identities and are **not compatible admission evidence**.

In total, the four sessions launched twelve fresh API processes: eleven
completed their observations, one failed because its backend was unspecified,
and its planned second invariant case was skipped. Successful cases retain
330 canonical streamed requests, each with exactly 64 generated tokens.
Cumulative session wall time was 3,010.40 seconds. None of these diagnostic
latencies is an admitted speedup. No reference was replaced, no profile was
created by the probe, and no published API was restarted.

The established finding is cache/lifecycle-associated numerical drift, not a
proven defect in one named operator. A backend/kernel-level fix remains open.
The implemented infrastructure changes prevent that drift from silently
authorizing a native profile or an exposed relay. They do not turn failed
numerical checks into accepted results.

## Final validation and cleanup

Final safety-code source identity:
`0a94218e496ed7f8f412f226bf353e9b2631406f85e3caebf55d053da8e33e2e`.
This is separate from both GPU diagnostic source snapshots.
Local CPU tests and CPU tests in the pinned image each passed **118 tests**,
with the CUDA suite skipped. Scoped lint, shell syntax and diff checks passed.
The image still reports the existing AnyIO/pytest deprecation warning.

At `2026-11-05T12:02:47Z`, exact IDs, names and ownership labels confirmed
that both published service containers remained exited. Port 18080 had no
listener, no lab container or GPU compute process remained, and all eight GPUs
reported 2 MiB baseline memory. Existing unrelated services were not stopped.

The final outcome index is
`results/restart-final-20261105T120247Z/summary.json`; its directory also retains
the final source bundle. The original Triton kernel source is unchanged.
All new work remains uncommitted and unpushed. No additional GPU run is active.
