# Evolith (`evolith`)

> **Living Silicon & Self-Evolving Inference Engines**  
> A Dual-Plane Recursive Self-Improvement (RSI) Framework for LLM Inference Engines (`vLLM` / `SGLang` / `TensorRT-LLM`).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Dual-Plane RSI](https://img.shields.io/badge/RSI-Dual--Plane%20L1%2BL2-00b4d8.svg)](./ARCHITECTURE.md)

---

## Why `Evolith`?

Traditional LLM inference engines rely on human kernel engineers to hand-tune CUDA/Triton kernels and static runtime heuristics (`chunked_prefill_size=2048`, fixed speculative draft depth $k=5$, naive LRU KV cache eviction). Whenever hardware changes (`H100` $\rightarrow$ `B200` $\rightarrow$ `MI350X`) or workload distributions shift (short chat $\rightarrow$ 128K multi-turn coding agents with tool pauses), static heuristics degrade by **30%–60%**.

**Evolith** decouples the inference engine into **Immutable Safety Contracts** and **Two Orthogonal Evolvable Planes**, closing a **Self-Referential Ignition Flywheel**:

1. **Plane L1 — Hardware Kernel & Graph Compiler Evolution**:
   - Evolves custom epilogue fusions (`fused_add_rmsnorm_fp8_quant`) and long-context attention scoring (`triattention_prerope_score`) across shape buckets (`Batch = 1, 16, 128`).
   - Protected by a **4-Stage Hardened Anti-Cheat Kernel Sandbox**:
     - **Stage 1 (Static AST Gate)**: Blocks reward hacking such as global tensor memoization (`_STATIC_TENSOR_CACHE`), thread-block skipping, or host-synchronization tricks.
     - **Stage 2 (Prime-Shape & Checksum Gate)**: Verifies numerical accuracy across power-of-two and prime/ragged dimensions (`Batch=13`) while checking SHA-256 input tensor checksums against in-place corruption.
     - **Stage 3 (Ring-Buffer + L2 Cache Scrub Timing)**: Eliminates L2 cache warm-start illusions.
     - **Stage 4 (Roofline Reflection Card)**: Classifies each candidate into `MEMORY_BOUND`, `COMPUTE_BOUND`, `LATENCY_LAUNCH_BOUND`, or `REGISTER_SPILL` and generates structured reflection cards.
2. **Plane L2 — Runtime Scheduler, Speculative & KV Policy Evolution**:
   - Evolves pure-Python `@evolvable_policy` hooks (`propose_speculative_config`, `compute_chunked_prefill_budget`, `score_kv_block_retention`).
   - Verified in **~20 milliseconds** via the **Trace-Driven Dream Simulator** (`DreamTraceSimulator`), replaying 320 bursty multi-turn coding agent + tool-execution traces without waiting 15 minutes per live benchmark.
3. **Evolutionary Controller (DGM + GEPA + HarnessLens)**:
   - **Stepping-Stones Archive (`SteppingStonesArchive`)**: SQLite-backed MAP-Elites grid + Pareto frontier with **16x Compressed 4-Tuple Lineage Memory** (`[Hypothesis, PatchSummary, MetricDelta, FailureRootCause]`).
   - **GEPA AST Function-Level Crossover**: Splices the best speculative drafter from Parent A with the best Continuum TTL KV eviction policy from Parent B via Python AST transformation.
   - **HarnessLens Anti-Overfitting Defense**: Complexity-penalized pre-commit gate (`occams_razor_gate`) + periodic AST **Subtractive Ablation GC** ("Dream Pruning").

---

## Quickstart

### Installation

```bash
uv venv
uv pip install -e ".[dev]"
```

### 1. Run Full Dual-Plane Evolution & Self-Referential Ignition

```bash
uv run evolith evolve
```

### 2. Run the 4-Stage Anti-Cheat Kernel Sandbox

Inspect how `Evolith` catches and blocks the `_STATIC_TENSOR_CACHE` memoization reward hack while profiling valid single-pass and shape-aware kernels across buckets:

```bash
uv run evolith verify-kernel --op fused_add_rmsnorm_fp8_quant
```

### 3. Run the Millisecond Trace-Driven Dream Simulator

Compare the Generation-0 workload-blind baseline against the Generation-4 co-optimized policy across 320 multi-turn coding agent traces:

```bash
uv run evolith dream-sim
```

### 4. Inspect the 16x Compressed 4-Tuple Lineage Archive

```bash
uv run evolith archive
```

### 5. Run the Test Suite

```bash
uv run pytest -v
```

---

## Project Structure

```text
evolith/
├── ARCHITECTURE.md                      # Detailed bilingual engineering blueprint
├── pyproject.toml                       # Package & CLI configuration
├── src/evolith/
│   ├── cli.py                           # Typer + Rich CLI (`evolve`, `verify-kernel`, `dream-sim`, `archive`)
│   ├── models.py                        # Pydantic models for Roofline profiles, SimMetrics, and LineageNode
│   ├── planes/
│   │   ├── l1_kernels.py                # ShapeBucketDispatchTable & L1 kernel evolution targets
│   │   └── l2_policies.py               # EngineStepState & L2 scheduler/speculative/KV policy mutations
│   ├── verifiers/
│   │   ├── kernel_sandbox.py            # 4-Stage Hardened Anti-Cheat Kernel Verifier
│   │   └── dream_simulator.py           # Millisecond Discrete-Event Trace Simulator (320 agent requests)
│   └── controller/
│       ├── archive.py                   # SQLite Stepping-Stones Archive & 16x 4-Tuple Memory Compression
│       └── engine.py                    # EvolithRSIEngine (GEPA AST Crossover, Occam Gate, Subtractive GC)
└── tests/
    └── test_evolith.py                  # End-to-end unit tests for all verifiers, gates, and evolution loops
```

---

## Documentation

See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for the full architectural blueprint, mathematical formulations, anti-reward-hacking specification, and roadmap.