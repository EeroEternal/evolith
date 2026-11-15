# Qwen engine validation, 2026-11-05

## Outcome

Evolith built a controlled OpenAI-compatible runtime on vLLM, not a new CUDA
execution engine. Two bounded engine-evolution sessions produced **zero
promotions**. The final profile retains the measured native TP=4 configuration.
There is **no demonstrated evolutionary performance gain**.

**Final state:** the user chose to stop the API and end this experiment.
Both dedicated containers were stopped. A later user request authorized a
separate [restart investigation](restart-reproducibility.md), not reopening
these services.

The fixed contract is Qwen3.5-27B BF16, text requests, 2048-token context and
two or four RTX 5090 GPUs. No weights, precision, context limit or admission
thresholds were relaxed to produce a winner.

## Final strong native reference

Session: `20261105T045218Z-78d3416e0890`.
Baseline trial: `20261105T045220Z-89d9b9feeb49`.

- vLLM 0.19.0, PyTorch 2.10.0+cu129, Triton 3.6.0.
- Four RTX 5090 GPUs, BF16, sequence capacity 8, prefill budget 2048,
  GPU memory fraction 0.90, prefix caching disabled.
- Native compilation defaults, not an eager debug reference. The startup log
  confirms `FULL_AND_PIECEWISE`, capture sizes `[1, 2, 4, 8, 16]` and
  `enforce_eager=False`.
- Six fixed public chat requests, concurrency one, temperature zero, thinking
  disabled, exactly 64 generated tokens per request with EOS ignored.
- Five complete workload warmup passes, then five measured rounds with
  randomized request order. All samples are retained.

Each table value summarizes the five **per-round workload medians**, not a
production traffic percentile:

| Metric | Median | Round range | Timing CV |
| --- | ---: | ---: | ---: |
| TTFT | 96.467 ms | 96.289–118.027 ms | 9.537% |
| Average ITL | 12.864 ms | 12.859–12.871 ms | 0.0343% |

These are localhost HTTP delivery measurements, including request processing
and streaming overhead, not device-only kernel times. They do not establish
concurrent throughput, tail latency or long-context quality.

## Autonomous candidates

The local model selected the first candidate using measured TTFT/ITL and archived
startup-failure hints. Its capacity failure activated the resource-constraint
solver. The solver's proposals are labeled separately from model proposals:

| Trial | Proposal source | TP | Sequence capacity | Prefill budget | Memory fraction | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `20261105T045757Z-f21e5a12d613` | Local model | 2 | 4 | 2048 | 0.90 | KV-cache capacity failure |
| `20261105T050014Z-0f3dc93294fe` | Capacity solver | 2 | 1 | 256 | 0.95 | KV-cache capacity failure |
| `20261105T050302Z-4e762ac94422` | Capacity solver | 2 | 1 | 256 | 0.90 | KV-cache capacity failure |

The second trial's log reports approximately **−0.08 GiB available KV-cache
memory** after runtime profiling. This describes the measured configuration
and harness, not a proof that all two-GPU BF16 deployments are impossible.
No candidate silently changed settings or fell back to another backend.

The session stopped at its three-candidate budget. No candidate reached an
independent hardware AB/BA confirmation. That promotion path is covered by CPU
simulation tests, not demonstrated by a successful hardware promotion here.

## Earlier diagnostic probe

Session `20261105T043551Z-afb8447b0a5f` used two workload warmup passes and
explicitly bounded small graph-capture sets. Its source identity differs from
the final native-default session. Do not compare these baselines as an
optimization or transfer its evidence into the final profile.

- TP=2, sequence capacity 8, prefill 2048, memory 0.95: startup capacity failure.
- TP=2, sequence capacity 4, prefill 2048, memory 0.95: measured, but output
  verification failed and ITL regressed to 21.532 ms.
- That measured candidate's paired TTFT speedup estimate was 1.0676×, but its
  99% lower bound was 0.9971×. Its paired ITL speedup was 0.5977×.
  Neither metric supports promotion. Do not substitute a ratio of marginal
  medians for the paired statistic.

Both sessions retained native TP=4. Older capacity failures can inform proposals
as hints, but never authorize a performance claim.

## Evidence and checks

Raw reports, startup logs, worker requests and 13 exported append-only SQLite
evolution records are backed up locally at:

`results/engine-20261105.f4lhyxka/`

Its `index.json` identifies both sessions and the frozen model/measurement source
snapshot used by the final evolution session:

`b4b905b8bb49867acd3d50b877a625c0af130321896b88cff429905ae87273c2`

The initial probe's original source snapshot is not preserved. Its raw results
and source hash remain available; only the final session has a matching frozen
source bundle in this export.

- CPU suite: **86 passed, 1 CUDA suite skipped**.
- GPU suite before eleven additional CPU-only tests: **84 passed** on the same
  model/measurement sources. The added tests cover service isolation, archived
  failure hints, capacity-repair lineage and byte-preserving loopback forwarding.
  All **24 engine/deployment tests** also pass in the actual GPU image.
- Engine/lab and new-test Ruff checks, shell syntax, lockfile and package builds
  passed.
- Whole-repository Ruff still reports **15 existing legacy-demo findings**.
  The affected tracked files are unchanged. This is not a clean full-repository
  lint result.

## Deployment

The selected profile is:

`/home/bodesi/evolith-lab/results/20261105T045218Z-78d3416e0890/engine.json`

The dedicated service is `evolith-qwen`, with model name `evolith-qwen`.
The first readiness check timed out:
`20261105T051120Z-a8b3ed2d44f7`.
The model was healthy internally, but Docker ignored the configured published
port on its internal network. No host listener existed.

A separate CPU-only `evolith-qwen-loopback` relay corrected connectivity without
restarting or changing the model. The model remains isolated. The relay uses
the host network namespace and binds only IPv4 `127.0.0.1:18080`, forwards only
to the private model-container address and has no GPU or weight mount. Actual
`ss` inspection confirms no wildcard listener. This delivery-only module was
added after the evolution snapshot and has its own source hash in deployment
reports; the six hashed model/measurement files remain unchanged.

Model discovery, ordinary chat and streamed token delivery work. However, the
fresh deployed native TP=4 process **does not match** the frozen baseline:

- Report `20261105T052711Z-0d4f4c00b2ae` records the failed streaming check.
- Report `20261105T053409Z-074a5f3c449d` records functional-only status.
- Two of six requests have different greedy token sequences.
- Additional requests exceed the fixed 0.05 generated-token logprob tolerance.
- Two consecutive complete workload passes within the live process match each
  other. Dependency-version comparison found no relevant differences.

The cause of this cross-restart drift is **not established**. We did not relax
the gate, reinterpret it as a successful optimization or mark deployment
accepted. The localhost endpoint was an **experimental native baseline only**.
Live diagnostic timings are not promotion evidence and must not be compared
with the earlier reference as an improvement.

The relay source, helper and all three deployment reports are preserved in a
supplemental export alongside the evolution evidence. Existing unrelated
services and model weights are unchanged.

## User-requested shutdown

Shutdown record: `20261105T060150Z-36390eecc5e0`.

- Stopped only `evolith-qwen-loopback` and `evolith-qwen`, after verifying their
  exact container IDs and Evolith ownership labels.
- Confirmed both containers exited and `127.0.0.1:18080` was free.
- Confirmed no GPU compute applications remained. All eight GPUs reported
  2 MiB of baseline memory use.
- Preserved code, raw results, stopped containers and the internal network.
  Existing unrelated services were not stopped.

The shutdown record and `shutdown-export.json` are saved alongside the 16
evolution/deployment records in the local evidence backup. The API is no longer
running. Cross-restart output drift remains unresolved, and neither a deployment
acceptance nor a speedup is claimed.

See [operation and admission rules](qwen-engine.md) for requests, SSH tunneling
and removal of this dedicated service.
