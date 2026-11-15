# Evolith-managed Qwen engine

This capability builds an inference runtime **on the vLLM execution backend**.
It does not claim a new attention implementation or a rewritten CUDA engine.
Evolith supplies the constrained configuration evolution, independent admission,
reproducible profile and controlled localhost deployment.

**Current status (2026-11-05):** the user ended this experiment. The model and
loopback relay are stopped, GPU allocations are released, and port 18080 is not
listening. There is no admitted speedup or accepted live deployment. Code,
reports, stopped containers and the internal network are retained. The commands
below describe future manual operation, not a currently running API.
The user subsequently authorized a separate
[restart reproducibility investigation](restart-reproducibility.md); it does
not reopen the published API.

## Fixed user objective and contract

- Chat latency: time to first token (TTFT) and average inter-token latency (ITL).
- Existing Qwen3.5-27B BF16 weights, unchanged.
- Two or four RTX 5090 GPUs, 2048-token context limit, text requests.
- OpenAI-compatible API, exposed only on host `127.0.0.1`.

The optimizer cannot change precision, weights, the context limit, measured
output length, workload or admission rules. It can select tensor parallelism,
native/full-decode/piecewise graph mode, sequence capacity, prefill token budget
and a bounded GPU memory fraction. Queueing/concurrent load is not the objective
of this first capability.

## Evolution

On the GPU machine, use a pre-existing compatible image and local weights:

```bash
export EVOLITH_IMAGE=vllm/vllm-openai:v0.19.0
export EVOLITH_GPUS=0,1,2,3
export EVOLITH_MEMORY=192g
export EVOLITH_CPUS=24
export EVOLITH_TIMEOUT=5700
bash scripts/container.sh evolve-engine \
  --model-path /models/qwen --output /results \
  --candidates 2 --time-budget 5400 --rounds 5 --max-tokens 64
```

The reference is native graph-mode vLLM, TP=4, sequence capacity 8 and prefill
budget 2048. Prefix caching is disabled for every configuration. Enabling normal
graph execution relative to an eager debug baseline is not credited as an
evolutionary improvement.

The local model proposes validated configuration JSON using measured TTFT/ITL
and recent outcomes. Novel proposals guide subsequent experiments. The schema
cannot carry source code, arbitrary flags, evaluation edits or paths.

All experiments run isolated from external networking. Each trial loads a fresh
API process and measures six public chat fixtures at concurrency one. Five full
workload passes warm up before sampling. Outputs use temperature zero and a
fixed generation length with EOS ignored so that early termination cannot win.
The benchmark requests token IDs and generated-token logprobs for verification.

TTFT starts before the HTTP request and ends at the first generated-token event,
not the initial assistant-role frame. ITL is the delivery interval between first
and last token divided by the number of token intervals. All token IDs/logprobs,
per-request sampled latencies and per-round medians are retained in ignored
result artifacts. No prompt-completion text is needed for the comparison.

## Admission, not optimistic selection

- Greedy token IDs must match the incumbent; finite generated-token logprob
  differences must be at most 0.05.
- At least one median latency must have a one-sided 99% paired-bootstrap lower
  speedup bound above 1.03. The other latency's bound must be at least 1.0.
- Baseline/candidate timing CV must remain at most 10%.
- A promising pilot triggers independent fresh-process confirmation in AB/BA
  order. Both fresh and combined comparisons must pass.
- A successor replaces only the current approved incumbent, not a weaker
  historical baseline. The deployer recomputes the timing gate from the saved
  proof and checks source/runtime/model compatibility.
- Startup-infeasible configurations are recorded explicitly, never benchmarked
  through an undisclosed fallback. Other execution failures stop the session.

Before any configuration search or native-profile emission, the updated
controller measures a second fresh native process and requires compatible
provenance, distinct trial IDs and strict output agreement. A mismatch,
restart startup failure or exhausted certification budget produces no profile.
It does not replace the first reference with the second process's outputs.
This extra fixed certification stage is retained separately, not pooled into
the pilot's paired timing samples; subsequent performance confirmation still
uses A0/B0/B1/A1. The same wall-time budget includes this additional process.

Capacity failures also activate a trusted resource-constraint solver. It
enumerates legal resource-monotone neighbors (less sequence/workspace demand,
bounded memory fraction), without changing weights, precision, context limit
or admission. It labels these proposals separately from model proposals.
Matching older startup failures can inform model proposals as hints across
controller versions; they cannot be reused as performance evidence.

The returned `engine.json` selects either an approved successor or the measured
native baseline. Zero promotions is a valid result, not an acceleration claim.
Reports are append-only SQLite records plus atomic JSON snapshots.

For measured results, see [2026-11-05 validation](qwen-engine-validation-2026-11-05.md).
Both sessions produced zero promotions; the selected native TP=4 baseline had
96.467 ms TTFT and 12.864 ms average ITL on this fixed workload. These numbers
are not a claim of improvement over native vLLM.

These are exploratory small-workload measurements, not multiple-testing-adjusted
production evidence. Full quality, long-context, concurrent traffic, tail SLOs,
GPU topology portability and energy are not established. Model fingerprints use
config/index/shard sizes, not a full weight-content digest.

## Localhost deployment

Use the profile path returned by the experiment, translated from `/results/...`
to the host project's `results/...` directory:

```bash
export EVOLITH_PORT=18080
bash scripts/serve.sh /absolute/project/results/<run-id>/engine.json
```

The helper uses the required number of GPUs from `EVOLITH_GPUS` and mounts code,
model and profile read-only. The model stays on a dedicated internal Docker
bridge without external egress. A separate CPU-only, read-only byte relay uses
the host network namespace but binds **only** `127.0.0.1:18080` and forwards to
one fixed private model-container address. It has no GPUs or model-weight mount.
This avoids relying on Docker port publishing, which was ignored for the
internal network on the validation machine. Both relay sockets disable Nagle
buffering. Existing model or relay containers are never replaced.
Model loading and verification can take several minutes. Before starting the
relay, the updated helper waits for the private model server and runs five
complete workload passes against the approved artifact's exact token IDs and
0.05-logprob limit. A mismatch, malformed proof, startup failure or deadline
failure prevents publication and removes only this launch's own containers and
network. Evidence is retained in `results/prepublication-*/report.json`.
This helper change is CPU-tested, not a newly accepted hardware deployment.

The new verifier is part of the engine source identity. Historical profiles,
including the failed 2026-11-05 profile, are intentionally invalid for this
changed source; they are not rewritten or automatically replaced. No current
profile is newly approved. Direct `serve-engine` is the private worker entry
point, not a substitute for the helper's prepublication check.
Native profiles also require distinct independent-restart proof during
artifact validation, not just when the controller first writes them.

A successful helper check is not a new performance admission. Also verify
actual host listening addresses and model/chat endpoints. In the historical
2026-11-05 run, basic requests and streaming worked,
but fresh output matching **failed**. That experimental native endpoint was
subsequently stopped at the user's request, without acceptance.

After readiness:

```bash
curl http://127.0.0.1:18080/v1/models
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"evolith-qwen","messages":[{"role":"user","content":"Explain KV Cache briefly."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

For another computer, use an SSH tunnel rather than opening a public port:

```bash
ssh -N -L 18080:127.0.0.1:18080 -p SSH_PORT USER@GPU_HOST
```

The model is named `evolith-qwen` and its relay `evolith-qwen-loopback` by default.
To remove only these dedicated services:

```bash
docker stop evolith-qwen-loopback
docker rm evolith-qwen-loopback
docker stop evolith-qwen
docker rm evolith-qwen
docker network rm evolith-qwen-internal
```

Existing services and model weights are not changed.
