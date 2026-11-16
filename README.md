# Evolith (`evolith`)

> **Guardrailed Recursive Self-Improvement (RSI) for LLM Inference Engines**  
> Source-level GPU evolution harness (`Rust` + `CUDA` + `Triton` + `Python`) with coordinator-pinned safety contracts, Red-Team static/formal gates, and interleaved multi-tier hardware verification.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Source-Level GPU RSI](https://img.shields.io/badge/RSI-Source--Level%20GPU%20%2B%20Dual--Plane-00b4d8.svg)](./ARCHITECTURE.md)

---

## Overview

`Evolith` separates **Immutable Coordinator Safety Contracts** from **Evolvable Engine Planes** (`L1` GPU Kernels / Collectives / Graph Capture and `L2` Runtime Scheduler / Speculative / KV Policies). The repository contains two clearly delineated layers:

1. **Production Source-Level GPU RSI Pipeline (`src/evolith/zstep/`)**:
   - Operates on real multi-GPU inference engines (`zstep` running `DeepSeek-V4.1-Flash` & `Qwen3.5-35B-A3B` at `TP=8` on `8× NVIDIA RTX 6000D` SM120).
   - Enforces **Coordinator Spec-Pinning** (SHA-256 verification of greedy reference fixtures, wire protocols, and overlays), **7-Category Red-Team Boundary & TP8 Agreement Audits**, **Exhaustive AST Dispatch Verification** (`17,280` `(num_seqs, max_bucket)` pairs), **Premise 2-Strike Circuit Breakers**, **Frozen Baseline/Candidate Tarballs**, **2-Stage Remote GPU Screening**, and **3-Round Round-Major Interleaved Verification** with **Occam's Razor Complexity Penalties**.
   - Supports external LLM proposers (`--proposer openai` self-hosted endpoint, `--proposer kimi`), direct unified-diff submission (`evolith zstep-submit` / `PatchProposer`), and crash-resilient checkpoint resumption (`evolith zstep-resume`).

2. **Offline CPU Pedagogical & Unit-Test Harness (`planes/`, `verifiers/`, `controller/`)**:
   - A dependency-free, GPU-free reference harness (`evolith evolve`, `verify-kernel`, `dream-sim`) used to demonstrate and unit-test MAP-Elites stepping-stones archives, 16× 4-tuple lineage compression, GEPA AST function crossover, subtractive AST pruning, and isolated subprocess/AST sandboxing.
   - **Important Note on Fidelity**: The offline CPU harness uses **pre-seeded candidate mutations and analytic Roofline/queuing cost formulas** (`BottleneckClass`: `MEMORY_BANDWIDTH_BOUND`, `COMPUTE_BOUND`, `LAUNCH_OVERHEAD_BOUND`, `REGISTER_SPILL_BOUND`, `BALANCED`) rather than hardware performance counters. All real performance claims in `Evolith` come exclusively from `src/evolith/zstep/` hardware verification on physical GPUs.

---

## Verified 8×GPU RSI Results (`zstep` × `DeepSeek-V4.1-Flash`, `TP=8` SM120)

Across 8 recorded generations in `.evolith/zstep-rsi/archive.sqlite3` on `8× NVIDIA RTX 6000D`:

- **`g0001`–`g0006` (Rejected by Local / Screen Gates)**: Caught cargo build failures (`g0001`), zero-gain H2D metadata packing (`g0002: -0.08%`), `CUDA_DEVICE_MAX_CONNECTIONS` contention (`g0003: +0.34%`), PyTorch `int32` index tensor errors (`g0004`), `torch.where(..., out=...)` scalar overload errors (`g0005`), and CUDAGraph teardown allocator crashes (`g0006`) — each distilled into pre-GPU static `RedTeam` gates.
- **`g0007-abf839ad` (`verify_rejected`, raw `+2.43%`, `net_gate = 0.9923`)**: Achieved `+9.89%` on low-concurrency `screen`, but hidden 3-round `verify` caught `+6.97%` `p99_itl_ms` tail-latency regression at `c128` caused by raising mixed-step token budget (`3199 > 3072`) plus a `-3.20%` Occam's Razor penalty on `280` LOC.
- **`g0008-67b2dd20` (`ACCEPTED`, `+5.67%` 3-round geomean, `net_gate = 1.0563`, `182` LOC)**:
  - Stream-ordered TP=8 `device_token_overlap` (`launch`/`finish`) + GPU-resident 3-token Engram window + all-rank Top-2 SHM reduction;
  - Intermediate decode `CUDAGraph` bucket ladder `(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)`;
  - Stage-1 self-slice direct HBM write in `shm_v5` (`M <= 384`) + `NCCL_SHM_USE_CUDA_MEMCPY=1` Mode 3 (`6ch × 4MiB` Xeon DDIO) for eager prefill/mixed steps (`-18 ms` per `M=3072` step).

| Verify Tier (3-Round Interleaved Median) | Baseline (`tok/s`) | `g0008` (`tok/s`) | Throughput | Median TTFT | Mean TPOT | P99 ITL |
|---|---:|---:|---:|---:|---:|---:|
| `in512_c16` | 769.04 | **825.74** | **+7.37%** | -3.41% | **-7.17%** | -6.10% |
| `in512_c128` | 1722.54 | **1809.61** | **+5.05%** | -3.42% | **-5.05%** | -3.38% |
| `in1024_c32` | 813.89 | **862.84** | **+6.01%** | -3.37% | **-5.65%** | -3.40% |
| `in1024_c128` | 1203.00 | **1254.40** | **+4.27%** | -3.39% | **-4.00%** | -3.35% |

---

## Quickstart

### Installation

```bash
uv venv
uv pip install -e ".[dev]"
```

### 1. Source-Level GPU RSI on `zstep` (Production Pipeline)

```bash
# Inspect ROI-ranked optimization campaigns and 2-Strike circuit breaker status
uv run evolith zstep-campaigns

# Submit a unified git patch directly through Coordinator local/RedTeam/Dispatch gates,
# remote 8xGPU screen, and 3-round round-major interleaved verification
uv run evolith zstep-submit --source targets/zstep/upstream --campaign t1-1 \
  --patch path/to/candidate.patch \
  --niche-key "(plane=l2_scheduler,focus=dsv41_device_token_overlap)" \
  --hypothesis "..." --summary "..."

# Autonomous proposer loop via an OpenAI-compatible endpoint (e.g. self-hosted zstep/Nebula)
uv run evolith zstep-evolve --source targets/zstep/upstream --campaign auto \
  --proposer openai --proposer-api-base http://127.0.0.1:8080/v1 --proposer-model DeepSeek-V4.1-Flash

# Status, evidence inspection, and crash-recovery resume
uv run evolith zstep-status
uv run evolith zstep-evidence <candidate-id>
uv run evolith zstep-resume <candidate-id>
```

### 2. Offline CPU Pedagogical Harness (Synthetic Cost Models)

```bash
uv run evolith evolve
uv run evolith verify-kernel --op fused_add_rmsnorm_fp8_quant
uv run evolith dream-sim
uv run evolith archive
```

### 3. Run the Test Suite

```bash
uv run pytest -v
uv run ruff check src tests
```

---

## Project Structure

```text
evolith/
├── ARCHITECTURE.md                         # Architectural blueprint & safety contracts
├── docs/
│   └── zstep-session-lessons.md            # Empirical SM120 / zstep RSI failure & speedup postmortems
├── pyproject.toml                          # Package & CLI configuration
├── src/evolith/
│   ├── cli.py                              # Typer + Rich CLI (`zstep-*`, `evolve`, `verify-kernel`, `dream-sim`)
│   ├── models.py                           # Pydantic models (`BottleneckClass`, `HardwareProfile`, `LineageNode`)
│   ├── zstep/                              # Production Source-Level GPU RSI Pipeline
│   │   ├── archive.py                      # SQLite lineage archive, checkpoints & 2-Strike circuit breaker
│   │   ├── campaigns.py                    # ROI-ranked T1/T2 campaign specs & scoped allowed_roots
│   │   ├── dispatch_verifier.py            # Exhaustive AST bucket/threshold verifier (17,280 pairs)
│   │   ├── gates.py                        # Parity, screen, interleaved verify & Occam's Razor gates
│   │   ├── loop.py                         # End-to-end Coordinator loop (`submit`, `evolve`, `resume`)
│   │   ├── profiling.py                    # SM120 operator/collective telemetry & reflection card builder
│   │   ├── proposer.py                     # OpenAI, PatchProposer & Kimi sandboxed proposer backends
│   │   ├── redteam.py                      # SHA-256 Spec-Pinning + 7-category Red-Team & TP8 Agreement audit
│   │   ├── remote.py                       # SSH/Docker 8xGPU execution, container cleanup & GPU idle checks
│   │   ├── task.py                         # ZStepRSITask & decision schemas
│   │   └── workspace.py                    # Isolated git staging, diff inspection & deterministic tarballs
│   ├── planes/                             # Offline pedagogical L1/L2 synthetic mutation targets
│   │   ├── l1_kernels.py
│   │   └── l2_policies.py
│   ├── verifiers/                          # Offline AST + isolated-subprocess verifiers & analytic simulators
│   │   ├── kernel_sandbox.py
│   │   └── dream_simulator.py
│   └── controller/                         # MAP-Elites archive, GEPA AST crossover & subtractive GC
│       ├── archive.py
│       └── engine.py
└── tests/                                  # Unit & integration tests covering all gates, sandboxes & loops
    ├── test_evolith.py
    └── test_zstep_*.py
```

---

## Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — Architectural blueprint, anti-reward-hacking specification, and roadmap.
- **[docs/zstep-session-lessons.md](./docs/zstep-session-lessons.md)** — Empirical engineering lessons from full-stack `zstep` SM120 optimization (`g0001`–`g0008` postmortems, APC, speculative decoding, and PCIe collective tuning).
