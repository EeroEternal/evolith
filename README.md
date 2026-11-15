# Evolith

**Measurement-first GPU inference optimization.** Evolith searches a bounded,
trusted Triton parameter space, verifies candidates against an independent
reference, and measures them against an optimized implementation before
considering a runtime replacement. It does not claim recursive self-improvement
from synthetic speedups.

The governing principle is **self-directed improvement**, not a permanent,
human-chosen hotspot: observe the workload, choose a supported target, propose
changes, independently verify, retain only an approved successor, and repeat.
Correctness and measurement gates are not evolvable rewards. See
[AGENTS.md](AGENTS.md).

## Implemented scope

- **L0:** separate candidate worker processes with timeouts, strict parameter
  schemas, immutable inputs, NaN/Inf rejection, padded/prime shapes, independent
  float64 reference, randomized paired CUDA-event timings and raw samples.
- **L1:** BF16, zero-centered RMSNorm with optional residual addition, matching
  Qwen3.5/Gemma semantics. Residual addition rounds to BF16 before normalization.
  Search varies warps and rows per program, not arbitrary generated source.
- **Promotion:** a fresh-seed confirmation is mandatory. Exact shape contracts
  are eligible only when the one-sided 99% paired-bootstrap lower bound exceeds
  a 3% gain and timing CV is at most 10%. This is an exploratory gate, not a
  multiple-testing or independent replication guarantee.
- **Actual model regression:** an opt-in vLLM worker adapter, frozen weights,
  deterministic generated-token/logprob comparison, repeated AB/BA trials,
  observed fallback counters, and a separate end-to-end deployment gate.
- **Model-guided iteration:** the local model can propose strict parameter JSON
  for another search round. Model-generated Python is never executed.
- **Autonomous controller:** `autoevolve` re-observes the running workload,
  chooses an exact contract by sampled cumulative cost and archived experience,
  avoids repeated configurations within a session, and continues within fixed
  time/evaluation/cycle budgets. A successor must beat the current approved
  incumbent, not merely the original native implementation.
- **Qwen Ultrafast:** `qwen-ultrafast` adds a remote Rust/CUDA adapter with a
  bounded BF16 GEMV block-size space, independent component checks, exact-token
  full-model AB/BA verification and successor-versus-incumbent promotion.
- **History:** unique run IDs, atomic JSON reports, source/environment metadata,
  and append-only SQLite records. No result is replaced by a later run.

The first supported vLLM target is **Qwen3.5-27B BF16**. Qwen3.8-Flash-Next is
supported only through the separate Qwen Ultrafast adapter, not the tested vLLM
0.19 runtime. L2 runtime policy optimization, a calibrated trace simulator,
arbitrary kernel generation and self-hosted acceleration remain future work.

## CPU tests

```bash
uv sync --extra dev
uv run pytest -q
```

GPU tests skip without CUDA/Triton. The old NumPy prototype remains available
only through `demo-evolve`, `demo-verify-kernel`, `demo-dream-sim` and
`demo-archive`. Its inferred hardware counters, synthetic goodput and ignition
estimates are **not measured results** and cannot authorize the new adapter.

## GPU runtime

Use a CUDA runtime supporting RTX 5090 (SM120), PyTorch 2.10, Triton 3.6 and
vLLM 0.19 on Linux Python 3.10–3.13 (tested: 3.12). Install into an isolated
environment with `uv sync --extra gpu --python 3.12`, or
use the container helper. The helper requires an existing compatible image and
complete local model weights. It intentionally disables networking and downloads.

```bash
export EVOLITH_IMAGE=vllm/vllm-openai:v0.19.0
export EVOLITH_MODEL=/data/models/Qwen3.5-27B
export EVOLITH_GPUS=2

bash scripts/container.sh doctor --model-path /models/qwen
```

### Autonomous evolution

No manually selected row counts or hotspot are required:

```bash
export EVOLITH_GPUS=0,1,2,3
export EVOLITH_MEMORY=192g
export EVOLITH_CPUS=24
export EVOLITH_TIMEOUT=1900
bash scripts/container.sh autoevolve \
  --model-path /models/qwen --output /results \
  --cycles 3 --evaluation-budget 32 --time-budget 1800
```

The result identifies the session report, stop reason and optional approved
manifest. A budget stop with no approved artifact is a valid result; the native
engine remains unchanged. Nothing is installed into a persistent service.
Use `--workload /workspace/workloads/local.json` with a local
`{"prompts":["Your representative request"]}` file to change the workload.
The file is data, never executable code. Keep private workloads in the ignored
`workloads/` directory.

The initial observation capability covers GemmaRMSNorm only. It estimates
per-contract cost from capped CUDA-event samples; **it is not a whole-engine
profiler or arbitrary source evolution**. See
[controller design and limits](docs/autonomous-evolution.md).

### Qwen Ultrafast remote adapter

`qwen-ultrafast` adds a separate bounded capability for the Rust/CUDA
[Qwen Ultrafast](https://github.com/EeroEternal/qwen-ultrafast) runtime. It
syncs a reviewed source tree over SSH, builds it on the remote SM120 machine,
and searches only `bf16-gemv` block sizes `{64, 128, 256, 512}`. The component
benchmark checks a CPU reference and retains alternating CUDA-event samples.
Only candidates passing the paired microbenchmark gate proceed to full-model
AB/BA trials. Every trial requires all 64 generated tokens to match the current
incumbent. A successor must also pass the independent end-to-end timing gate;
microbenchmark wins and faster-but-different outputs are rejected.

```bash
uv run evolith qwen-ultrafast \
  --remote user@gpu-server --port 22 \
  --local-repo /path/to/qwen-ultrafast \
  --remote-repo /remote/path/to/qwen-ultrafast \
  --nccl-library /remote/path/to/libnccl.so.2 \
  --candidates 64,256,512 --model-pairs 5 --output results
```

The adapter refuses active remote GPU processes, bounds local and remote
execution time, and writes candidates, raw timings, rejections and lineage to
atomic JSON and append-only SQLite records. It does not deploy a winner or
change the runtime default. This is not whole-engine observation or arbitrary
Rust/CUDA source evolution; other operators require their own trusted
benchmark and observation capability.

### Individual experimental stages

For manual diagnosis or reproduction of one measured stage:

```bash
export EVOLITH_GPUS=2
bash scripts/container.sh evolve \
  --model-path /models/qwen --output /results --rows 1,4,16,64
```

The final JSON names the run's `/results/<run-id>/manifest.json`. No CPU or
synthetic fallback is used when GPU requirements fail. No winner is also a
valid experimental outcome.

Verify a measured artifact on four GPUs. Two 32 GB cards held the weights but
did not leave enough profiled workspace/KV-cache headroom in the tested runtime:

```bash
export EVOLITH_GPUS=0,1,2,3
export EVOLITH_MEMORY=128g
export EVOLITH_CPUS=24
bash scripts/container.sh verify-model \
  --model-path /models/qwen \
  --manifest /results/<run-id>/manifest.json \
  --output /results \
  --tensor-parallel 4 \
  --proposals /results/next-proposals.json
```

An end-to-end run refuses an empty manifest, a different model/runtime, or a
changed kernel source. Successful correctness checks do **not** automatically
approve deployment: `deployment_approved` additionally requires a passing
end-to-end timing gate. Reports retain the native fallback reasons and counts.
The adapter is removed when verification ends.
Worker integration uses string-named RPCs with JSON arguments; it does not
enable vLLM's insecure callable/pickle serialization.

If no candidate passes confirmation, use `verify-model --baseline-only` with the
same manifest to measure the unmodified model and generate next-round proposals.
This never installs a candidate or reports an optimization speedup.

Feed local-model proposals into the next measured round:

```bash
export EVOLITH_GPUS=2
bash scripts/container.sh evolve \
  --model-path /models/qwen --output /results \
  --proposals /results/next-proposals.json
```

The source and model mounts are read-only; only `results/` and `cache/` are
writable. The helper exposes no network port and does not stop other services.
Existing services can still create resource contention; use idle devices.
For GPU tests, install pytest and dependencies into `cache/testdeps` offline,
then run `bash scripts/container.sh test -q /workspace/tests`.

## Measurement boundaries

- The microbenchmark uses a warm ring of eight inputs, not a universal cache
  scrub. CUDA-event time includes device idle gaps from host launches.
- Padded tensors contain NaN/noise sentinels. This detects some erroneous reads
  but does not prove memory safety; use Compute Sanitizer for broader coverage.
- The reference explicitly checks output finiteness and bytewise input
  preservation. It does not establish model-wide quality equivalence.
- End-to-end tests currently cover six serialized, short deterministic text
  generations in eager mode, not production load, full logits, long contexts,
  SLO goodput or energy.
- Model fingerprints cover config, index and shard sizes, not full weight
  contents. Treat artifacts as local trusted records, not signed attestations.
- Worker processes isolate crashes/timeouts, **not hostile code**. The trusted
  kernel template is part of the reviewed codebase; JSON cannot add imports,
  functions or arbitrary launch dimensions.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the original research blueprint.
That blueprint is aspirational, not a list of implemented capabilities.

Actual first-machine results and reproduction commands:
[2026-11-04 validation](docs/validation-2026-11-04.md).
