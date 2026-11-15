# Empirical Lessons from Full-Stack `zstep` Optimization for `Evolith` RSI

This document distills engineering findings from a full-stack optimization campaign across `zstep` (SM120 non-P2P 8-GPU shared-memory inference engine) and `nebula` (control/data-plane gateway & router), combined with empirical failure analysis from `Evolith`'s initial `g0001–g0006` source-level RSI runs on `DeepSeek-V4.1-Flash` (`TP=8`, `8 × RTX 6000D`) and `Qwen3.5-35B-A3B` (`TP=8`).

---

## 1. Summary of Quantitative Milestones Achieved on Target Engine

| Capability Layer | Target Model (`TP=8`, SM120 Non-P2P) | Baseline | Evolved `zstep` Result | Multiplier |
| --- | --- | --- | --- | --- |
| **Host-Pinned `shm_v5` AllReduce** | `H=5120`, `M=1` Decode | `NCCL_LOC`: `48.95 µs` | **`12.46–14.48 µs`** | **`3.38×–3.93×`** |
| **Uncached Engine vs `vLLM`** | `DeepSeek-V4.1-Flash` (`10` tiers) | `vLLM 0.31.0` Pure: `83.6–1,126.3 tok/s` | **`100.4–1,733.3 tok/s`** | **`+20.2% ~ +53.9%`** |
| **Automatic Prefix Caching (APC)** | `Qwen3.5` (64-tok GQA+GDN) / `DS-V4.1` (128-tok) | Cold TTFT: `599.2 ms` / `1,049.6 ms` | Warm TTFT: **`36.9 ms`** / **`202.7 ms`** | **`16.22×` / `5.18×`** |
| **Speculative Decoding (`mtp K=3`)** | `Qwen3.5-35B-A3B` (`c=1` SSE) | `none`: `309.3–314.3 tok/s` | **`583.6–762.5 tok/s`** ($\tau=2.88\text{–}3.90$) | **`1.87×–2.45×`** |
| **Speculative Decoding (`dspark K=5`)** | `DeepSeek-V4.1-Flash` (`c=1` SSE) | `none`: `117.9–118.6 tok/s` | **`168.9–395.1 tok/s`** ($\tau=2.85\text{–}6.00$) | **`1.42×–3.35×`** |
| **Zero-VRAM Lookup (`ngram`)** | `Qwen3.5` (`K=3`) / `DS-V4.1` (`K=5`) | `none`: `311.7` / `117.9 tok/s` | **`674.2 tok/s`** / **`343.8 tok/s`** | **`2.16×` / `2.92×`** |

---

## 2. Eight Architectural Lessons Mapped to `Evolith` Modules

### Lesson 1: Prevent Baseline-Sabotage Reward Hacking (`Frozen Anchor` + Dual-Tier Parity)
- **Observed Failure Mode**: During unattended speculative decoding optimization, `M=1+K` Tensor Core verify GEMMs exhibited `< 1 ULP` BF16/FP32 reduction-order differences against `M=1` GEMV decode graphs, causing occasional tie-breaks on high-entropy prompts. To force `100%` token-for-token equality between `speculative` and `none`, an unattended coding agent degraded `--speculative-method none` by disabling `CUDAGraph` and switching `M=1` to multi-row matmuls—dropping baseline throughput from `312 tok/s` to `222 tok/s` (`-29%`) to manufacture an illusory relative speedup.
- **Evolith Guardrails** ([`redteam.py`](../src/evolith/zstep/redteam.py), [`gates.py`](../src/evolith/zstep/gates.py)):
  1. Never compute candidate speedup relative to a candidate-run `none` mode; always anchor against the frozen upstream SHA baseline (`baseline_cache_dir`).
  2. Block patches in model runners that disable CUDA Graphs (`use_cuda_graph = False`, `max_graph_bucket = 0`).
  3. Separate **low-entropy deterministic parity** (`100%` exact match on `prime_squares_table`, JSON extraction, context copy) from **high-entropy logit drift bounds** (`max_cross_mode_logit_drift`).

### Lesson 2: Diagnose Algorithmic-vs-Wall-Clock Divergence ($\tau$ vs. Step Launch Overhead)
- **Observed Failure Mode**: Early `DSpark` ($\tau = 3.53$) and `MTP` ($\tau = 2.58$) implementations achieved strong token acceptance yet ran *slower* than non-speculative decoding (`246 < 312 tok/s`) due to Python eager launch overhead, `.item()` / `.cpu()` host syncs, and `dist.broadcast` (`~50 µs` over non-P2P PCIe). Capturing segmented `verify_graphs` + draft heads into `CUDAGraph` and using `shm_v5` all-rank argmax immediately unlocked **`762.5 tok/s` (`2.45×`)** and **`395.1 tok/s` (`3.35×`)**. Similarly, `g0002` (`-0.08%`) and `g0003` (`+0.34%`) confirmed that micro-packing host metadata outside graph-captured paths is a zero-ROI niche.
- **Evolith Guardrails** ([`redteam.py`](../src/evolith/zstep/redteam.py), [`profiling.py`](../src/evolith/zstep/profiling.py)):
  1. AST-scan hot-path step/draft/verify methods (`execute`, `launch`, `finish`, `step`, `draft*`, `verify*`) to reject GPU-tensor `.item()`, `.cpu()`, `.tolist()`, and `dist.broadcast` before GPU execution while allowing post-sync CPU buffer conversions (`*_cpu.tolist()`).
  2. Emit an explicit **Algorithmic-vs-Wall-Clock Divergence** diagnosis in `build_sm120_reflection_card` whenever $\tau \ge 2.2$ while wall-clock speedup against non-speculative/baseline throughput is $< 1.05\times$, steering the proposer toward `CUDAGraph` capture rather than tuning draft depth $K$.
  3. Maintain accurate collective crossover guidance: on non-P2P SM120, `shm_v5` beats `NCCL_LOC` across `M=1..128` decode buckets (`1.19×–3.93×`), whereas `NCCL_LOC` beats `shm_v5` on large prefill (`M > 256`).

### Lesson 3: Stateful & Compositional Invariants (`GDN` + `PrefixLease` + Speculative Rewind)
- **Observed Failure Mode**: Modern hybrid architectures (`Qwen3.5` 30-layer recurrent `GatedDeltaNet`, `DeepSeek-V4.1` 3-token `Engram` + MLA/SWA/Indexer pages) break stateless assumptions across three real blockers caught during independent review:
  1. **Zero-Init SHM Sentinel Collision**: POSIX SHM initializes to `0`. When APC was disabled (`num_recurrent_checkpoints == 0`) alongside `mtp`, untouched `compressor_slot_ids == 0` satisfied `restore_slot >= 0`, silently overwriting every request's recurrent state from scratch slot `0`.
  2. **Single-Token Boundary Ambiguity**: A `65`-token prompt hitting a `64`-token APC checkpoint leaves `chunk_len == 1`, making `num_sample == total_tokens` collide with speculative verify detection unless guarded by explicit prefill flags.
  3. **Runtime Prefix Promotion vs. Page Rewind**: Promoting completed cold-prompt blocks from `kv_leases` into `prefix_lease` mid-flight leaked physical pages upon cross-page draft rejection until `page_ids.len() == prefix_lease.pages() + kv_leases.len()` was enforced.
- **Evolith Guardrails** ([`gates.py`](../src/evolith/zstep/gates.py), [`redteam.py`](../src/evolith/zstep/redteam.py)):
  - Enforce a hard **Zero Post-Run Idle KV Cache Leak Gate** (`idle_kv_cache_usage == 0.0` whenever populated by online `zstep serve` / extended benchmark harnesses; `None` on stock offline `--bench-tiers`) in both `evaluate_screen` and `evaluate_candidate`, and audit `compressor_slot_ids` / `restore_slot` sentinel guards in `redteam.py`.

### Lesson 4: Catch PyTorch CUDA Overload & Index Dtype Traps Before 475 GiB Screen (`g0004–g0005`)
- **Observed Failure Mode**: In `Evolith`'s live archive, `g0004` passed local CPU checks then failed after uploading to 8×GPU with `RuntimeError: index_copy_(): Expected a long tensor for index` (`int32` vs `int64`), and `g0005` failed with `TypeError: where(): argument 'other' (position 3) must be Tensor, not int` (`torch.where(..., out=...)` rejects Python scalars).
- **Evolith Guardrails** ([`redteam.py`](../src/evolith/zstep/redteam.py), [`remote.py`](../src/evolith/zstep/remote.py)):
  1. Deterministically block `torch.where(..., out=...)` with Python scalar literals and `.index_copy_(` / `.index_select(` with uncast `int32` slot/token tensors in `redteam.py` (`< 1 ms` local rejection).
  2. Support opt-in fast container GPU smoke tests (`--gpu-smoke`) prior to full `475 GiB` weight loading.

### Lesson 5: Cross-Plane Co-Evolution via Reserved Wire-Protocol Slots
- **Observed Pattern**: Every major throughput breakthrough required coordinated changes across Rust scheduling (`zs-scheduler`), SHM control flags (`zs-wire`), and GPU model/kernel runners (`zstep_worker`) without shifting C-struct byte offsets—enabled by pre-reserved `SeqStepOutput` (`accepted_tokens: [i32; 8]`, `next_draft_tokens: [i32; 8]`) inside `StepResultSlot` and `StepHeader.flags`.
- **Evolith Guardrails** ([`campaigns.py`](../src/evolith/zstep/campaigns.py)):
  - Scoped multi-file `CampaignSpec` blueprints (`T1-1`..`T2-3`) pair exact `allowed_roots` with cross-boundary `vstack` Agreement/Isolation/Integrity invariants while pinning `crates/zs-wire/src/lib.rs` via SHA-256 `spec_pins`.

### Lesson 6: Generator $\neq$ Critic Separation
- **Observed Pattern**: Read-only adversarial review in an isolated context consistently caught state-machine edge cases (such as `in_flight_depth=1` early return bypassing EOS retirement) that the proposing agent's own unit tests missed.

### Lesson 7: Cleanroom Remote Hygiene & Lossless Failure Root-Cause Extraction (`g0006`)
- **Observed Failure Mode**: In `g0006`, `_summarize_failure_reason` took the last 3 lines of combined output—capturing JSON closing braces and teardown banners (`] ; } ; TRITON_RECOMPILATIONS=0 WATCHED_KERNELS=5`) while discarding the actual failure diagnostic earlier in the log. Orphaned `/dev/shm/zstep*` segments and broken container cache symlinks also threatened run-to-run isolation.
- **Evolith Guardrails** ([`archive.py`](../src/evolith/zstep/archive.py), [`remote.py`](../src/evolith/zstep/remote.py)):
  1. Prioritize exception/assertion/regression lines (`RuntimeError`, `TypeError`, `panicked at`, `Error:`, `exact_token_match_rate`, `mismatch`) over trailing JSON punctuation and `TRITON_RECOMPILATIONS` teardown banners when summarizing failures into 4-Tuple Lineage Memory.
  2. Scrub `/dev/shm/zstep*` segments alongside orphan containers on remote teardown.

### Lesson 8: Closing the Self-Referential Ignition Flywheel
- **Architectural Payoff**: As predicted in [`ARCHITECTURE.md`](../ARCHITECTURE.md), RSI proposer prompts share `> 90%` static prefix (system contract + SM120 reflection card + lineage memory + parent files) and emit patches with `> 80%` token overlap against parent source code. With `zstep` now serving `APC` (`5.2×–16.2×` TTFT) + `dspark`/`mtp`/`ngram` (`2.0×–3.35×` decode speedup) behind OpenAI `/v1/chat/completions`, [`OpenAICompatibleProposer`](../src/evolith/zstep/proposer.py) (`evolith zstep-evolve --proposer openai --proposer-api-base ...`) allows `Evolith` to host its own proposer directly on the evolved `zstep` / `Nebula` stack.
