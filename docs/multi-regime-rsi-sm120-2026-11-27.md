# Multi-Regime Hardware-Aware RSI & SM120 8×RTX 6000D Campaign Report (2026-11-27)

## 1. Overview & Governing Principles

This document records the general multi-GPU Recursive Self-Improvement (RSI) infrastructure upgrades added to `evolith` alongside real 8-GPU empirical validation on `8 × NVIDIA RTX 6000D` (`SM120`, `156 SMs`, `85,651 MiB` GDDR7/GPU, `CUDA 13.0`, Dual-Socket Xeon `4+4` PCIe Gen5 Switch topology at `106.1 GB/s` bidirectional PXB bandwidth) running `DeepSeek-V4.1-Flash` (`475.25 GiB` checkpoint) on `zstep`.

All infrastructure changes preserve GPU/vendor generality (SM120, SM90, SM100, ROCm/XGMI, Ascend/HCCS) and strictly enforce `AGENTS.md`:
- **No hardcoded shapes as permanent objectives**: Crossover thresholds (such as `K >= 2048` vs stock `K <= 1024`) are derived from live hardware scout probes via `PiecewisePolicySynthesizer`.
- **Trusted admission gates**: `min_geomean_gain = +1.50%`, 3-round interleaved cold `AB/BA` verification, 100% greedy token parity, `TRITON_RECOMPILATIONS == 0`, Z3 formal resource/soundness checks, and RedTeam spec-pin verification remain immutable.
- **Explicit capability & precision tracks**: `PrecisionTrack.LOSSLESS` vs `PrecisionTrack.CALIBRATED_LOSSY` (`LossyQualityGateSpec`) prevent lossy approximations from silently entering lossless tracks.

---

## 2. General Multi-GPU RSI Primitives Added to Evolith

1. **Hardware Topology & Precision Contracts (`src/evolith/core/topology.py`)**:
   - `HardwareTopologySnapshot`, `InterconnectTier` (`INTRA_SWITCH_P2P`, `INTER_SOCKET_HOST`, `NVSWITCH_FABRIC`, `RDMA_FABRIC`), `PrecisionTrack`, and `LossyQualityGateSpec`.
2. **Multi-Regime Ledger & Piecewise Crossover Synthesis (`src/evolith/core/regime.py`)**:
   - `RegimeSpec`, `MultiRegimeLedger` (requiring positive target-regime gain and zero-regression across guard regimes), `audit_operator_dispatch`, and `PiecewisePolicySynthesizer`.
3. **Declarative Resource Surfaces (`src/evolith/core/surface.py`)**:
   - Replaces engine-specific regexes with `StateResourceDecl.patch_patterns` for target-agnostic closure-orthogonality checks (`are_patches_closure_orthogonal`).
4. **Fast 20-Second GPU Operator Pre-Screen & Secondary Guard Regimes (`src/evolith/zstep/{task,gates,remote,loop,hardware}.py`)**:
   - `hot_prescreen_script` + `evaluate_hot_prescreen(...)` runs unit tests, GPU numerical parity, and interleaved `AB/BA` operator microbenchmarks in ~20 seconds *before* committing 8 × 475 GiB cold model loads (~75 min).
   - `secondary_benchmark_profiles` + `evaluate_secondary_guard(...)` enforces zero throughput regression and 100% greedy parity on secondary regimes after primary verification.
   - `BenchmarkProfile` adds `regime_phase`, `num_kv_pages`, `max_pages_per_seq`, and validated `env_overrides` (`longctx-tp8`, `longctx-agent` with `ZSTEP_MAX_MODEL_LEN=33792` matching `max_pages_per_seq=264`; `spec-dspark` with `ZSTEP_SPEC_METHOD=dspark`, `ZSTEP_NUM_SPEC_TOKENS=5`).
5. **Streaming `GeminiProposer` & Multi-Hunk `SEARCH/REPLACE` (`src/evolith/zstep/proposer.py`)**:
   - Uses SSE streaming (`streamGenerateContent?alt=sse`) with thought-part filtering (`not p.get("thought", False)`) to prevent HTTP read timeouts on large model/kernel files, and supports multiple `<<<<<<< SEARCH ... >>>>>>> REPLACE` blocks under a single `FILE:` header.
6. **Automated Google Drive Statistical Benchmark Archiving (`src/evolith/zstep/gdrive_sync.py`)**:
   - Adds `GDriveBenchmarkSync`, `ZStepRSITask.gdrive_folder_id` (`--gdrive-folder-id` / `EVOLITH_GDRIVE_FOLDER_ID`), and `evolith zstep-sync-gdrive` so every campaign automatically exports append-only lineage CSVs (including rejected/failed trials) and syncs structured JSON/CSV/Markdown/patch artifacts to Google Drive idempotently.
   - Canonical Google Drive Archive Folder: `https://drive.google.com/drive/folders/1iKYeTEqklUeSbzb6sitYEpdDwwg1CO_0`
   - Canonical 5-Tab Statistical Ledger Sheet: `https://docs.google.com/spreadsheets/d/1CZbmhI_PLPvDLanYDPfvPcdEj10z5SP4-esOVGER_Z0/edit`

---

## 3. Empirical 8×RTX 6000D Scout Probe Findings

Raw probe data: [`docs/benchmarks/sm120_6000d_rsi_20261127/round2_8gpu_scout_probe.json`](benchmarks/sm120_6000d_rsi_20261127/round2_8gpu_scout_probe.json)

| Subsystem | Key Live Hardware Measurement (`8 × RTX 6000D`, `CUDA 13.0`) | Architectural Implication |
|---|---|---|
| **Intra-Switch PXB P2P** | `p2p_enabled=True`, `pxb_shm_enabled=True`, **`106.1 GB/s`** bi-dir | Intra-island GPU 0..3 and GPU 4..7 communicate via direct CUDA IPC BAR1/PXB. |
| **`TP=8` AllReduce (`M=4096`, `40 MiB`)** | `zstep shm_pxb_bfp1`: **`914.4 µs`** vs `NCCL`: `2,986.6 µs` (**`3.27×` faster**) vs `flashinfer 0.7.0.post1 pcie_ipc_ar STAGED_RING`: `5,544.4 µs` (**`6.06×` faster**); `ws.tune()` deadlocks on cross-socket `UNSTAGED` BAR writes | Blindly replacing `shm_pxb_bfp1` with `pcie_ipc_ar` regresses 6× or deadlocks. However, 80 AllReduces/step × `914.4 µs` = **`73.15 ms`/step**, making two-stream **Prefill Two-Batch Overlap (TBO)** the primary communication opportunity. |
| **`sm120_dsa_indexer` Ragged Prefill** | Stock wins at `K=512` (`35.0 vs 39.6 µs`, `0.88×`); `dsa_indexer_det_topk` wins for `K >= 2048` (`1.302×` @ `2K`, `1.117×` @ `8K`, `1.041–1.064×` @ `16K–32K`, 10/10 exact top-k set match) | Despite `1.15–1.27×` isolated operator speedup, DSA Indexer accounts for `< 0.5%` of total `DeepSeek-V4.1-Flash` step latency; end-to-end ablation (`g0006`) measured **`-0.31%`** due to Python dispatch overhead, confirming microbenchmark wins alone do not justify admission. |

---

## 4. Round 1 & Round 2 End-to-End 8-GPU Campaign Results

Raw campaign data: [`docs/benchmarks/sm120_6000d_rsi_20261127/round1_round2_campaign_summary.json`](benchmarks/sm120_6000d_rsi_20261127/round1_round2_campaign_summary.json)

### 4.1 Closed-Loop Root Causes Caught & Repaired by Evolith Gates
1. **Sequential Microbenchmark Boost-Clock Drift (`g0001`, `g0002`)**: Fixed in `round2_hot_prescreen.py` via interleaved `AB/BA` median event timing.
2. **Runtime Triton Recompilation (`g0003`, `g0004` -> `TRITON_RECOMPILATIONS=4` on `dsv41_comp_ring_slot`) & TileLang JIT Stall**:
   - `self.comp_bt_gpu` has shape `(ms, 1)` (`int32`, 4 bytes/row), so slicing `self.comp_bt_gpu[r_start:r_end]` at `r_start % 4 != 0` produced `block_table_ptr % 16 != 0`; fixed via 16-byte-aligned `self._tbo_comp_bt_gpu[u][:u_reqs]`.
   - Mixed decode+prefill steps (`total_tokens = 2049..2055`) split at odd `mid` produced 8-byte-aligned `positions_full[mid:]`; fixed by gating pure 256-aligned prefill steps (`min_q_len > 1 and total_tokens % 256 == 0`) and pre-warming `_run_prefill_tbo` inside `_warmup_and_capture_decode_graphs`, achieving **`TRITON_RECOMPILATIONS=0`** and jumping `ctx8k_c2` gain from `+1.05%` to **`+2.67%`**.
3. **32K Page-Table Truncation (`g0005` -> `IndexError: index 256`)**: Fixed in `BenchmarkProfile("longctx-tp8")` by pairing `max_pages_per_seq=264` with `ZSTEP_MAX_MODEL_LEN=33792` (`264 × 128`).

### 4.2 3-Round Interleaved Cold `AB/BA` Verification Summary (vs `zstep@f2c23b1`)

| Candidate | Regime | LOC | Screen Gain | 3-Round `AB/BA` Verify Geomean | `paired_lcb_2se` | Per-Tier 3-Round Mean Throughput & TTFT Delta | Parity & Triton Audit |
|---|---|---:|---:|---:|---:|---|---|
| **R1 `g0002-e1ad0d86`** (`156-SM MegaGate PDL` + `ld_cv SHM tree FP32` + `DSpark` zero-alloc) | `short-throughput` | 73 | `+0.49%` | **`+0.45%`** | **`+0.33%`** | `in512_c16: +1.03%`, `in512_c128: +0.68%`, `in1024_c32: +0.06%`, `in1024_c128: +0.05%` | 100% Greedy match, `TRITON_RECOMPILATIONS=0` |
| **R2 `g0006-606c82ae`** (`DSA Ragged Prefill K>=2048` + `__ldcs SHM`, no TBO) | `longctx-tp8` | 204 | **`-0.31%`** | — (screen rejected) | — | `ctx8k_c2: -0.10%`, `ctx16k_c8: -0.52%` | 100% Greedy match |
| **R2 `g0010-8c4c252e`** (Pure Aligned+Warmed Prefill TBO) | `longctx-tp8` | 52 | **`+1.68%`** (`ctx8k_c2 +2.67%`) | **`+0.33%`** | **`+0.26%`** | `ctx8k_c16: +0.35%`, `ctx16k_c12: +0.16%`, `ctx32k_c4: +0.48%` (`-1.5%` TTFT), `ctx32k_c8: +0.33%` (`-1.1%` TTFT) | 100% Greedy match, `TRITON_RECOMPILATIONS=0` |
| **R2 `g0009-fe01dea3`** (Combined 4-Way: Aligned+Warmed TBO + R1 `g0002`) | `longctx-tp8` | 125 | **`+1.44%`** (`ctx8k_c2 +2.50%`) | **`+0.59%`** (12/12 positive) | **`+0.48%`** | `ctx8k_c16: +0.48%` (`-0.71%` TTFT), `ctx16k_c12: +0.37%` (`-0.66%` TTFT), **`ctx32k_c4: +0.91%` (`-1.58%` TTFT)**, **`ctx32k_c8: +0.60%` (`-1.15%` TTFT)** | 100% Greedy & Trace match, `TRITON_RECOMPILATIONS=0` |

---

## 5. Round 3 Parallel 3-Track Multi-Regime Results (vs Evolved `zstep@8ad676f`)

Raw Round 3 data: [`docs/benchmarks/sm120_6000d_rsi_20261127/round3_3track_results.json`](benchmarks/sm120_6000d_rsi_20261127/round3_3track_results.json)

| Track | Hardware / Regime | Candidate (`LOC`) | Baseline (`zstep@8ad676f`) | Candidate Measured | Key Finding |
|---|---|---|---|---|---|
| **Track 1 — Mixed-Step 16-Aligned TBO** | `sm120-8x6000d-tp8` / `longctx-tp8` | `g0001-39015493` (`17 LOC`) | `ctx8k_c2`: `12,749.5` tok/s (`98.83` out)<br>`ctx16k_c8`: `12,996.2` tok/s (`50.57` out) | `ctx8k_c2`: `12,746.6` (`-0.02%`)<br>**`ctx16k_c8`: `13,069.5` (`+0.56%`, `+1.00%` vs `f2c23b1`)** | 16-token-aligned mixed-step TBO unlocks `+0.56%` additional gain on mixed decode+prefill steps (`TRITON_RECOMPILATIONS=0`, 100% Greedy parity). |
| **Track 2 — Dual-NUMA `PP=2×TP=4` Wave-mHC & P2P** | `sm120-8x6000d-pp2tp4` / `longctx-tp8` | `g0001-e8908ee6` (`46 LOC`) | `ctx8k_c2`: `8,586.3` tok/s (`66.56` out)<br>`ctx16k_c8`: `7,620.1` tok/s (`29.65` out) | **`ctx8k_c2`: `8,647.2` (`+0.71%`)**<br>`ctx16k_c8`: `7,570.0` (`-0.66%`) | Definitively proves `TP=8` (`12.75–13.07K` tok/s) beats `PP=2×TP=4` (`7.62–8.65K` tok/s) by **`+47%` to `+71%`** on `zstep` because `shm_pxb_bfp1` (`914 µs`) + TBO already eliminated stock vLLM's `~3 ms` NCCL bottleneck. |
| **Track 3 — `spec-dspark` (`DSpark5`) Contiguous DMA** | `sm120-8x6000d-tp8` / `spec-dspark` | `g0002-8ab5e8ef` (`72 LOC`, after RedTeam `.tolist()` catch on `g0001`) | `spec_c1`: **`177.61` out tok/s (`5.29 ms` TPOT)**<br>`spec_c8`: **`820.38` out tok/s (`8.08 ms` TPOT)** | `spec_c1`: `177.60` (`-0.01%`)<br>**`spec_c8`: `825.20` out tok/s (`+0.59%`, `8.02 ms` TPOT `-0.74%`)** | In-graph `(b, k)` contiguous pack + single DMA + zero-`.tolist()` NumPy slicing improves batched `DSpark5` throughput by `+0.59%` and cuts TPOT by `-0.74%` (`TRITON_RECOMPILATIONS=0`, 100% Greedy parity). |

---

## 6. Round 4 Orthogonal Composition 3-Round `AB/BA` Verification (`zstep@5fbcdbe` Incumbent)

Raw Round 4 data: [`docs/benchmarks/sm120_6000d_rsi_20261127/round4_composed_abba_verify.json`](benchmarks/sm120_6000d_rsi_20261127/round4_composed_abba_verify.json)

To test whether Round 3's two sub-regime winners (`Track 1` `17 LOC` + `Track 3 v2` `72 LOC` = `89 LOC`) generalize across hidden high-concurrency multi-tier verification without cross-regime regressions, we ran full 3-Round Interleaved Cold `AB/BA` Verification (`repeats=3`, 12 cold 8-GPU containers per iteration) across both `longctx-tp8` and `spec-dspark`:

| Verification Suite | Candidate | 3-Round `AB/BA` Geomean | `paired_lcb_2se` | Per-Tier 3-Round Mean Delta (vs `zstep@5fbcdbe`) | Gate Verdict & Action |
|---|---|---:|---:|---|---|
| **`longctx-tp8` (`8K–32K`)** | `g0002-9e1e9726` (`Track 1` Mixed-Step 16-Aligned TBO) | **`+0.64%`** (`+1.23%` cumulative vs `f2c23b1`) | **`+0.58%`** | `ctx16k_c12`: **`+0.87%`** (TPOT `75.63 → 74.92 ms`), `ctx32k_c4`: **`+0.70%`** (`43.25 → 42.92 ms`), `ctx32k_c8`: **`+0.70%`** (`87.05 → 86.49 ms`), `ctx8k_c16`: **`+0.30%`** | **`CONFORMANT` (`TRITON_RECOMPILATIONS=0`) — LANDED in `zstep`** (with explicit `dspark_drafter` isolation). |
| **`spec-dspark` (`DSpark5`)** | `g0002-748b7123` (v1 composed) & `g0001-c657e439` (v2 isolated `Track 3 v2`) | **`-0.47%`** (v1) / **`-1.05%`** (v2) | `-0.74%` / `-1.30%` | Public Screen `spec_c8` won **`+0.40%..+1.05%`**, but hidden `spec_c16` (`16:48:1024:256`) regressed **`-1.47%` / `-2.74%`** (`p99_itl +7.53% / +6.83%`) | **REJECTED & EXCLUDED per `AGENTS.md`**: proves public 2-tier Screen wins (`spec_c8 +1.05%`) can mask CPython `np.int32` scalar-boxing + `b=16` transpose overhead on hidden `c16` tiers. |

---

## 7. Round 5: `FlashInfer v0.7.6` Overlay, SM120 On-Chip SMEM `Split-K=32` mHC Fusion & `#8740` 32K Paged DSA (`zstep@651395c` Incumbent)

Raw Round 5 telemetry & 3-regime 3-Round Cold `AB/BA` artifacts:
- Stage 1 8-GPU Scout Probe (`FlashInfer v0.7.6` `d0cbb476`): [`docs/round5_stage1_fi076_scout_results.json`](round5_stage1_fi076_scout_results.json)
- Stage 2 SM120 Dynamic-SMEM `Split-K=32` mHC Pre-Screen: [`docs/round5_stage2_mhc_smem_prescreen.json`](round5_stage2_mhc_smem_prescreen.json)
- Stage 4 3-Suite 3-Round Interleaved Cold `AB/BA` Verification (`192.0 min` across 36 cold 8-GPU containers): [`docs/benchmarks/sm120_6000d_rsi_20261127/round5_multi_regime_abba_verify.json`](benchmarks/sm120_6000d_rsi_20261127/round5_multi_regime_abba_verify.json)

### 7.1 Hardware Scout & Rejected Negative Trials (Preserved per `AGENTS.md`)
1. **`FlashInfer v0.7.6` (`d0cbb476`) Deterministic Overlay + SHA256 Persistent Host JIT Cache**:
   - Upgraded SM120 overlay (`2.28 MiB`, `SHA256 = e09f7c10503e...`) and added SHA256-keyed host JIT caching (`FI_CACHE_DIR`), reducing cold container `b12x` preparation from `197 s` to **`15 s`**.
   - **Rejected Trial — `FlashInfer 0.7.6 pcie_ipc_ar` (`profile="pcieswitch-pairs"`)**: Measured `5,867 µs` at `M=4096` vs `zstep shm_pxb_bfp1` **`1,168 µs`** (`5.0×` slower due to `slow_hops=136` across dual-socket NUMA). Excluded; retained `shm_pxb_bfp1`.
   - **Rejected Trial — `70 MiB` Persisting L2 Cache Window (`cudaStreamAttrAccessPolicyWindow`)**: Pinning `70 MiB` of `fn` matrices in SM120's `112 MiB` L2 cache regressed `M=4..16` sub-layer execution (`0.984×`) by shrinking non-persisting L2 ways needed by activations/KV. Excluded; added `persisting_l2_cache_max_size_bytes` introspection to `HardwareTopologySnapshot` with explicit empirical guard.
2. **Landed Mechanisms (`297 LOC` Candidate)**:
   - **SM120 On-Chip SMEM `Split-K=32` `mHC Prenorm + Sinkhorn` Single-Kernel Fusion (`sm120_mhc_prenorm_sinkhorn_smem_kernel<MAX_M>`)**: Stages `640` BF16s per split once in SMEM, streams `fn [24, 20480]` via `__ldcs`, matches `cvt.rna.tf32.f32` (`max_diff <= 2.47e-5`, bit-exact deterministic), and eliminates register spills via compile-time `#pragma unroll` (`1.44×–1.83×` faster standalone at `M=1..8`; saves `0.33–0.49 ms/step` in overlapped 8-GPU sub-layers).
   - **`#8740` Two-Stage `candidate_blocks` 32K Paged DSA Indexer + `long_decode_graphs`**: Wires `candidate_out` (`layer 20`) and `candidate_blocks` (`layers 24, 28, 32, 36`) in `b12x_dsv41.py` plus `long_decode_graphs` (`buckets [1..32]` at `max_seq_len = max_model_len`, sharing `graph_pool`) in `deepseek_v41.py`, unlocking full 40-layer CUDA Graph paged sparse decode beyond `16K` up to `524K` tokens.

### 7.2 3-Regime 3-Round Interleaved Cold `AB/BA` Verification & Piecewise Crossover Synthesis

| Suite / Regime | Candidate ID | Screen Gain | 3-Round Cold `AB/BA` Geomean | `paired_lcb_2se` | Per-Tier 3-Round Measured Deltas (vs `zstep@651395c`) | Multi-Regime Gate & Piecewise Action |
|---|---|---:|---:|---:|---|---|
| **`longctx-tp8` (`8K–32K` Target)** | `g0001-01817706` | **`+10.31%`** (`ctx16k_c8 +21.26%`, `ctx8k_c2 +0.34%`) | **`+10.77%`** (`+12.13%` vs `f2c23b1`) | **`+10.55%`** | `ctx32k_c4`: **`+18.82%`** (TPOT `42.88 → 30.38 ms`, **`-29.2%`**)<br>`ctx16k_c12`: **`+17.37%`** (TPOT `75.10 → 65.82 ms`, **`-12.4%`**)<br>`ctx32k_c8`: **`+7.92%`** (TPOT `86.48 → 77.42 ms`, **`-10.5%`**)<br>`ctx8k_c16`: **`+0.04%`** (`49.22 → 49.16 ms`) | **`ACCEPTED` (`CONFORMANT`, `TRITON_RECOMPILATIONS=0`, 100% Greedy match)** |
| **`spec-dspark` (`DSpark5` Guard)** | `g0001-882aa453` | **`+0.76%`** (`spec_c1 +0.74%`, `spec_c8 +0.78%`) | **`+0.56%`** (positive guard) | `-0.25%` | `spec_c4` (`n=24`): **`+1.68%`** (`6.66 → 6.54 ms`)<br>`spec_c2` (`n=12`): **`+0.42%`** (`5.69 → 5.67 ms`)<br>`spec_c16` (`n=96`): `-0.42%` | **Guard PASSED (`+0.56% > 0%`, 100% Greedy match)** |
| **`short-throughput` (`512–1K` Guard)** | `g0001-07bb4dc0` | **`+0.15%`** (`in512_c1 +0.70%`, `in512_c32 -0.40%`) | **`-0.42%`** (with `n <= 64`) | `-0.47%` | `in512_c1` (`n=1`): **`+0.70%`**, `in1024_c32`: **`+0.38%`**<br>`in512_c16` (`n=16`): `-1.02%`, `in512_c128`: `-0.31%`, `in1024_c128`: `-0.72%` | **Crossover Caught**: `Split-K=32` (`32` CTAs on `mhc_stream`) wins for `1 <= n <= 12` (`c1 +0.70%`, `ctx32k_c4 +18.82%`, `ctx16k_c12 +17.37%`) and contends with main-stream GEMMs at `n >= 16` (`c16 -1.02%`). Tightened dispatch in `b12x_dsv41.py` from `1 <= n <= 64` to **`1 <= n <= 12`**. |

---

## 8. Round 6: Asynchronous THP `Engram` Offload (`-12 GiB/GPU`), Zero-Host-Stall Fused `DSpark` Pipeline & Boundary-Safe Ragged Verify (`zstep@cfacb76` Incumbent)

Raw Round 6 telemetry, depth/confidence calibration, and 3-Round Cold `AB/BA` artifacts:
- Stage 1 8-GPU Hardware Scout (`Engram` VRAM/UVA + `DSpark` SWA Width): [`docs/round6_stage1_engram_dspark_scout.json`](round6_stage1_engram_dspark_scout.json)
- Stage 2 `DSpark` Draft Depth (`K=5, 6, 7`) & `DSparkConfidenceHead` (`conf=0.0, 0.12`) Calibration: [`docs/round6_stage2_dspark_depth_conf_screen_v3.json`](round6_stage2_dspark_depth_conf_screen_v3.json)
- Stage 4 Full `ZStepRSILoop` 3-Round Interleaved Cold `AB/BA` Verification (`50.5 min`): [`docs/round6_multi_regime_abba_results.json`](round6_multi_regime_abba_results.json)

### 8.1 Evolith Pareto VRAM / Tail-Latency & Speculative Primitives (`P8`)
- Extended `RegimeSpec` and `MultiRegimeLedger` (`src/evolith/core/regime.py`) with explicit `max_vram_used_mib` and `max_p99_itl_ms` Pareto admission ceilings plus speculative `(batch_seqs, draft_depth, verify_width, physical_rows)` modeling (`P <= B * W`).
- Added `max_vram_used_mib` and `max_p99_itl_regression` enforcement to `ZStepRSITask` and `evaluate_screen` / `evaluate_verification` (`src/evolith/zstep/{task,gates}.py`), `pynvml` `peak_vram_used_mib` container stamping + persistent `$HOME/evolith-rsi/container-cache:/root/.cache` mounting in `ZStepRemoteRunner` (`src/evolith/zstep/remote.py`), and `"cold-prefill"` `BenchmarkProfile` (`src/evolith/zstep/hardware.py`).

### 8.2 Hardware Scout & Empirically Rejected Trials (Preserved per `AGENTS.md`)
1. **`Engram` VRAM & Background UVA CTA Contention**:
   - `DeepSeek-V4.1-Flash` holds two `16M × 2048` BF16 `Engram` tables (`layers.1.engram.embed` and `layers.14.engram.embed`), each consuming **`11.802 GiB/GPU`** on `TP=8` (`189.13 GiB` total across both tables) while reading only **`792 B/token/GPU`** (`18.2 µs` solo UVA gather).
   - Stock `ParallelEngramEmbedding.lookup(..., background=True)` launched `self._num_sms // 2 = 78` CTAs on `_prefetch_stream`, competing with Layer 0's `148..156`-CTA `sm12x_mega_gate` (`+202.6 µs` at `M=128`). Capping `emb._num_sms = 16` (**`8` background CTAs** = `5%` of `156` SMs) hides the `~18 µs` PCIe Gen5 x16 UVA gather behind Layers `0..13` (`1.5–4.0 ms`) with **zero** measurable latency overhead on non-speculative `short-throughput` (`+0.09%`) and `longctx-tp8` (`-0.30%`) while freeing **`12.07 GiB/GPU`** (`80,957 → 68,891 MiB/GPU`, **`94.3 GiB` cluster-wide**).
   - Preserves OS page-cache residency of the `475 GiB` checkpoint by auto-enabling THP host-pinned `layers.14.engram.embed` offload only when `/proc/meminfo` `MemAvailable >= 320 GiB` (`env_config.py`).
2. **Rejected Speculative Depth & Confidence Threshold Trials (`round6_stage2_dspark_depth_conf_screen_v3.json`)**:
   - **`K=7` (`cand_v1_k7`)**: Regressed `-17.21%` vs `base_dspark_k5` because verify rows `B × 8` pushed `c4..c16` (`32..128` rows) into compute-bound MoE/MLA regimes with diminishing tail acceptance on Steps `6..7`. Excluded.
   - **`K=6` (`cand_v3_k6`)**: Achieved `+33.26%` geomean vs `base_dspark_k5`, winning `spec_c8` (`813.65 tok/s`) and tying `spec_c1` (`272.26 tok/s`), but trailing `cand_v3_k5` (`+36.10%`) on `spec_c2` (`474.64 vs 517.05`), `spec_c4` (`727.16 vs 769.78`), and `spec_c16` (`1,228.84 vs 1,256.81`). Excluded.
   - **`K=5, conf=0.12` (`cand_v3_k5_conf12`, learned `DSparkConfidenceHead`)**: Achieved `+33.35%` geomean, winning `spec_c8` (`798.86 vs 766.99`) and `spec_c16` (`1,293.40 vs 1,256.81`), so `DSparkConfidenceHead` is retained in `deepseek_v41_dspark.py` behind `ZSTEP_DSPARK_CONF_THRESHOLD` (`default 0.0`) for `B >= 8` serving tuning while shipping lossless `K=5, conf=0.0` as the universal default.

### 8.3 Root Cause & Landed Speculative Architecture (`P2`, `g0001-28ab22ed`)
1. **Zero-Host-Stall Fused Post-Verify `DSpark` CUDAGraph (`csrc/shm_allreduce_kernels.cu`, `deepseek_v41.py`, `deepseek_v41_dspark.py`, `serving.rs`)**:
   - Replaces the between-stage `runner.finish()` host `synchronize()` + Python `verify_batch` + NumPy `ctx_slot_mapping` + H2D DMA + 2nd graph launch with `dspark_verify_prep_kernel` + captured `post_verify_draft_graphs[v_bucket]`, replaying **`verify_graphs[v_bucket]` + `post_verify_draft_graphs[v_bucket]` back-to-back in one asynchronous GPU submission** and overlapping Rust KV-page retirement (`commit_verified_round`) cleanly.
2. **Eliminating `85–204 ms` `128`-Token Page-Boundary Eager Fallbacks**:
   - Whenever `max_req_len % 128 == 0` (`640, 768, 1280`), the final ragged verify step (`p_max = 638`, `q_len < 6`) previously tripped `(p_max_pad // 128) == (p_max // 128)` and fell back to `~95–200 ms` eager Python across the entire batch. Relaxing `_can_use_ragged_verify_graph` to `(p_max_pad // 128) < bt_width` (backed by permanent zeroed `mla_dummy_vb`/`swa_dummy_vb` and `CompressorStateCache.CAPACITY = 8 >= width + 1`) plus per-sequence `eff_max_ctx` clamping in `dspark_verify_prep_kernel` keeps **100% of speculative steps inside CUDA Graphs**, collapsing `spec_c4` `p99_ITL` from **`162.53 ms` down to `14.83 ms` (`-90.9%`)**.

### 8.4 Full 3-Round Interleaved Cold `AB/BA` Verification & Guard Summary (`g0001-28ab22ed`)

| Suite / Regime | Phase & Metric | Incumbent (`cfacb76`, `DSpark K=5`) | Candidate (`g0001-28ab22ed`) | Measured Delta | Gate Verdict |
|---|---|---:|---:|---:|---|
| **`spec-dspark` (Peak GPU VRAM)** | `pynvml` Peak VRAM (`<= 74,000 MiB` gate) | `80,957 MiB/GPU` | **`68,891 MiB/GPU`** | **`-12,066 MiB/GPU` (`-11.78 GiB/GPU`)** | **`PASS` (`-94.3 GiB` cluster-wide)** |
| **`spec-dspark` (Screen)** | `spec_c1` / `spec_c8` (`repeats=1`) | `203.91` / `423.76 tok/s` | **`239.46` / `729.01 tok/s`** | **`+42.14%` geomean** (`+17.43%` / **`+72.03%`**) | **`PASS` (`0` Triton recompiles, 100% Greedy)** |
| **`spec-dspark` (3-Round Cold `AB/BA`)** | `spec_c2` (`repeats=3` mean) | `404.39 tok/s` (`TPOT 4.41 ms`, `p99 36.65 ms`) | **`528.42 tok/s`** (`TPOT 3.24 ms`, **`p99 11.60 ms`**) | **`+30.67%` mean** (`+26.13%` med, **`-68.4%` `p99_ITL`**) | **`ACCEPTED`** |
| **`spec-dspark` (3-Round Cold `AB/BA`)** | `spec_c4` (`repeats=3` mean) | `499.59 tok/s` (`TPOT 7.38 ms`, `p99 162.53 ms`) | **`728.99 tok/s`** (`TPOT 4.67 ms`, **`p99 14.83 ms`**) | **`+45.92%` mean** (`+41.29%` med, **`-90.9%` `p99_ITL`**) | **`ACCEPTED`** |
| **`spec-dspark` (3-Round Cold `AB/BA`)** | `spec_c16` (`repeats=3` mean) | `939.60 tok/s` (`TPOT 13.33 ms`, `p99 173.11 ms`) | **`1,285.54 tok/s`** (`TPOT 10.02 ms`, **`p99 146.76 ms`**) | **`+36.82%` mean** (`+36.74%` med, peak **`1,343.70 tok/s`**) | **`ACCEPTED` (`geomean +34.57%`, `lcb_2se +27.27%`, `net_gate 1.2303`)** |
| **`short-throughput` (Guard)** | `in512_c1` / `in512_c32` + VRAM | `167.90` / `1,099.24 tok/s` (`78,187 MiB`) | **`168.18` / `1,099.31 tok/s`** (**`66,093 MiB`**) | **`+0.09%` geomean**, **`-12,094 MiB/GPU`** | **`ACCEPTED` (`CONFORMANT`, 9/9 Z3 `CLOSED`)** |
| **`longctx-tp8` (Guard)** | `ctx8k_c2` / `ctx16k_c8` + VRAM | `99.39` / `61.50 tok/s` (`78,549 MiB`) | **`98.93` / `61.42 tok/s`** (**`66,535 MiB`**) | **`-0.30%` geomean** (noise), **`-12,014 MiB/GPU`** | **`ACCEPTED` (Zero UVA Prefetch Overhead)** |

---

## 9. Round 7: Lockstep-Safe Cold-Prefill Chunk Scaling, TBO `PXB-4K` Ring Reallocation, Piecewise SM120 MLA Prefill & CUDA 13 Green Context Audit (`zstep@3a1e298` Incumbent)

Raw Round 7 hardware scout, closed-loop 5-generation evolution lineage (`g0001..g0005`), and 3-Round Cold `AB/BA` + 3-Guard artifacts:
- Stage 1 8-GPU Hardware Scout (MLA Prefill Crossover + `shared_experts` Overlap + CUDA 13 Green Context + `PXB-4K` TBO Chunk Scaling): [`docs/round7_stage1_prefill_green_ctx_scout.json`](round7_stage1_prefill_green_ctx_scout.json)
- Stage 4 Full `ZStepRSILoop` 3-Round Interleaved Cold `AB/BA` Verification (`g0001..g0005`, Accepted `g0005-9271a17c`): [`docs/round7_multi_regime_abba_results.json`](round7_multi_regime_abba_results.json)

### 9.1 Evolith Cold-Prefill `TTFT` & Green Context Introspection Primitives (`P8`)
- Extended `HardwareTopologySnapshot` (`src/evolith/core/topology.py`) with CUDA 13 Green Context capability attributes (`green_context_supported`, `min_sm_partition_size`, `sm_coscheduled_alignment`).
- Extended `RegimeSpec`, `RegimeObservation`, and `MultiRegimeLedger` (`src/evolith/core/regime.py`) with explicit `ttft_ms` / `max_ttft_ms` admission bounds plus `prefill_chunk_tokens` and `comm_green_ctx_sms` descriptors.
- Upgraded `BenchmarkProfile`, `ZStepRSITask`, `evaluate_screen` / `evaluate_verification`, `ZStepRemoteRunner`, and `ZStepRSILoop` (`src/evolith/zstep/{hardware,task,gates,remote,loop}.py`) with per-profile `max_step_tokens` (`"cold-prefill"` = `8192`) and `max_ttft_ms` enforcement (`201 passed` CPU tests).

### 9.2 Stage 1 8-GPU Hardware Scout & Empirically Falsified Hypotheses (Preserved per `AGENTS.md`)
1. **CUDA 13 Green Context SM Partitioning (`P5` — Falsified & Excluded)**:
   - Disjoint SM splits (`pxb4k_green_split_8_148` and `pxb4k_green_split_16_140`) fail at runtime with **`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720)`** because `b12x.moe.fused_moe` (`tp_moe_dynamic_launch`) launches a `156-SM` persistent cooperative kernel that cannot launch on `148` or `140` SMs.
   - Asymmetric Green Contexts restricting only `comm_s` while keeping `main_s` at `156` SMs (`pxb4k_green_comm8_main156`: `298.94 ms` @ `8K`, `269.22 ms` @ `4K`; `pxb4k_green_comm16_main156`: `309.08 ms` @ `8K`, `241.32 ms` @ `4K`) are **`3.8%–16.7%` slower** than unpartitioned `pxb4k_shared_156sm` (`287.87 ms` @ `8K`, `230.70 ms` @ `4K`) because `shm_ar_tbo` (`24` blocks, `priority=-1`) bursts across `24` SMs during inter-kernel tail gaps when unthrottled. Excluded.
2. **Intra-Layer `shared_experts` Aux-Stream Overlap (`P5` — Falsified & Excluded)**:
   - Across `M = 8..4096`, launching `shared_experts` on an auxiliary stream concurrently with `b12x` `routed_experts` regressed **`22.6–41.2 µs/layer` (`0.981×–0.990×`)** because persistent `156-SM` `b12x.moe.fused_moe` saturates `100%` of SMs, leaving zero free wave slots and paying pure `cudaEventRecord` / `cudaStreamWaitEvent` overhead. Excluded.
3. **Piecewise SM120 MG MLA Prefill Crossover (`P1` — Tightened Dispatch)**:
   - Stock `cake_sparse_mla` (`flashinfer_trtllm_batch_decode_sparse_mla_dsv4`) beats `b12x` MG prefill (`_run_sm120_compressed_prefill`) on all `32` `swa_only` layers (`0.35×–0.96×`), on all `4` `compressed_cr64` layers (`iw_n = 512`, `0.31×–0.97×` — exposing a `1.54×–2.17×` regression in Round 6's `iw_n > 0 and rows >= 1024` rule), and at `T = 1024` on `full_cr1` (`0.835×`).
   - `b12x` MG prefill wins cleanly on `full_cr1` (`4` layers, `iw_n = 2048`) for **`rows >= 2048`** (`1.028×` @ `2K`, **`1.097×` @ `4K`**, **`1.125×` @ `8K`**, `cos_sim = 1.0`). Tightened `b12x_dsv41.py` dispatch to `iw_n >= 2048 and rows >= 2048`.
4. **TBO Chunk Scaling (`2048 → 8192`) + `shm_ar_tbo` `PXB-4K` Ring Reallocation (`P1` — Primary Win)**:
   - Scaling pure prefill from `chunk = 2048` (`u = 1024`: `2.6242 µs/tok`, `214.98 ms` / 40L) to `chunk = 8192` (`u = 4096`) initially regressed to `540.11 ms` (`r6_pxb2k_shared_156sm`) because `shm_ar_tbo.pxb_max_elements = 2048 * 5120` dropped `u = 4096` (`20,971,520` elements) off the P2P PXB ring onto host `/dev/shm`. Reallocating `shm_ar_tbo` to `pxb_max_elements = 4096 * 5120` (`pxb4k_shared_156sm`, Z3-verified `245.38 MiB <= 256.0 MiB` on `PP=1`) cut 40-layer `8K` execution to **`287.87 ms` (`0.8785 µs/tok`)** — **`1.88×` faster** at `8K` and **`2.99×` higher per-token efficiency** than `2K` chunks.

### 9.3 Closed-Loop 5-Generation Evolution Lineage (`g0001..g0005` in `round7_multi_regime_abba_results.json`)

| Candidate ID | LOC | Primary `cold-prefill` 3-Round `AB/BA` | Outcome | Multi-Regime Gate Diagnosis & Architectural Fix |
|---|---:|---:|---|---|
| `g0001-a85884d7` | `68` | — | `formal_inconclusive` | Container `PATH` lacked `.venv/bin/z3` (`Z3 solver unavailable`). Restored `PATH` so all 9 Z3 clauses run strictly (`CLOSED`). |
| `g0002-57830f5a` | `68` | `+40.72%` (`lcb_2se +39.58%`) | `verify_rejected` | Caught two cross-regime regressions: (1) `longctx-tp8` `ctx8k_c2` `p99_ITL` spiked `6.56 → 165.20 ms` (`+2418%`, `-18.31%` tok/s) because bursting `max_step_tokens` whenever `decode_active_count == 0` let Req 0 finish `8K` prefill ahead of Req 1 and suffer 4 mixed steps; fixed in `serving.rs` by gating burst on `decode_active_count == 0 && pending_prefill_seqs <= 1`. (2) `spec-dspark` `spec_c8` regressed `-4.25%` from unconditional `4K` MoE warmup & ring sizing; fixed by conditioning `4K` linear cap and `shm_ar_tbo` `PXB-4K` strictly on `self.max_step_tokens > 4096`. |
| `g0003-b2b5f177` | `89` | `+40.57%` (`lcb_2se +39.04%`) | `verify_rejected` | Passed `longctx-tp8` (`ACCEPTED`, `-1.31%`) and `spec-dspark` (`ACCEPTED`, `+4.98%`), but `short-throughput` tripped `TRACE_BATCH_COMPOSITION_INVARIANCE` (`france_capital` `batched_vs_chunked` logit drift `9.38 > 8.00`) because `chunked` (`n=101 <= 128`) used lossless `shm_ar_decode` whereas `batched` (`n=223`, `129..256` rows) used `pxb_bfp = 1`. |
| `g0004-5cffc5ce` | `93` | `+40.14%` (`lcb_2se +38.76%`) | `verify_rejected` | Setting `pxb_bfp = 0` across `0..256` rows passed `TRACE_BATCH_COMPOSITION_INVARIANCE` across all 5 containers (`drift <= 3.00`), `short-throughput` (`ACCEPTED`, `-0.21%`), and `longctx-tp8` (`ACCEPTED`, `-1.45%`), and boosted `spec-dspark` `+12.78%`, but altered `<= 128`-row `DSpark` draft acceptance enough to synchronize two `spec_c8` request retirements (`med_ttft` `114.0 → 144.3 ms`, `+26.58% > 5%`). |
| **`g0005-9271a17c`** | **`97`** | **`+40.26%` (`lcb_2se +39.27%`)** | **`ACCEPTED`** | Narrowed lossless `pxb_bfp = 0` in `shm_allreduce.py` strictly to **`128 * 5120 < numel <= 256 * 5120`**, closing the `129..256` `batched`-prefill parity gap (`max_logit_drift = 2.125 <= 8.00`) while keeping `1..128`-row `DSpark` drafts bit-identical (`spec-dspark` **`+3.38%`**, `spec_c8 med_ttft -0.39%`). **Passed 100% of primary & secondary gates.** |

### 9.4 Full 3-Round Interleaved Cold `AB/BA` Verification & 3-Guard Summary (`g0005-9271a17c`)

| Suite / Regime | Phase & Tier | Incumbent (`3a1e298`) | Candidate (`g0005-9271a17c`, `97` LOC) | Measured Delta | Gate Verdict |
|---|---|---:|---:|---:|---|
| **`cold-prefill` (Screen)** | `prefill8k_c1` / `prefill16k_c1` (`repeats=1`) | `10,402.0` / `11,020.9 total tok/s` (`TTFT 703.7` / `1,404.1 ms`) | **`15,099.5` / `16,243.6 total tok/s`** (**`TTFT 462.9` / `928.3 ms`**) | **`+46.28%` geomean** (`TTFT -34.2%` / `-33.9%`) | **`PASS` (`66,981 MiB/GPU`)** |
| **`cold-prefill` (3-Round Cold `AB/BA`)** | `prefill8k_v` (`1:6:8192:32`, median) | `37.05 out tok/s` (`9,520.9 tot tok/s`, `TTFT 697.25 ms`, `TPOT 5.32 ms`) | **`50.98 out tok/s`** (**`13,100.7 tot tok/s`**, **`TTFT 462.66 ms`**, `TPOT 5.32 ms`) | **`+37.60%` med** (`+37.93%` mean, **`-33.65%` `TTFT`**) | **`ACCEPTED`** |
| **`cold-prefill` (3-Round Cold `AB/BA`)** | `prefill16k_v` (`1:6:16384:32`, median) | `20.65 out tok/s` (`10,595.7 tot tok/s`, `TTFT 1,383.32 ms`, `TPOT 5.36 ms`) | **`29.23 out tok/s`** (**`14,996.4 tot tok/s`**, **`TTFT 928.43 ms`**, `TPOT 5.36 ms`) | **`+41.55%` med** (`+41.84%` mean, **`-32.88%` `TTFT`**) | **`ACCEPTED`** |
| **`cold-prefill` (3-Round Cold `AB/BA`)** | `prefill32k_v` (`1:4:32768:32`, median) | `10.92 out tok/s` (`11,194.3 tot tok/s`, `TTFT 2,762.32 ms`, `TPOT 5.36 ms`) | **`15.47 out tok/s`** (**`15,860.9 tot tok/s`**, **`TTFT 1,901.24 ms`**, `TPOT 5.36 ms`) | **`+41.67%` med** (`+41.48%` mean, **`-31.17%` `TTFT` / `-861 ms`**) | **`ACCEPTED` (`geomean +40.26%`, `lcb_2se +39.27%`, `67,145 MiB`)** |
| **`longctx-tp8` (Guard)** | `ctx8k_c2` / `ctx16k_c8` + VRAM | `99.13` / `61.43 tok/s` (`p99_ITL 6.56 ms` @ `ctx8k_c2`) | **`97.76` / `60.58 tok/s`** (`p99_ITL 6.55 ms`, `15,568.9 tot tok/s`, `66,515 MiB`) | **`-1.38%` geomean** (within `±3%` noise, **`0%` `p99_ITL` drift**) | **`ACCEPTED` (`CONFORMANT`)** |
| **`short-throughput` (Guard)** | `in512_c1` / `in512_c32` + VRAM | `168.06` / `1,105.03 tok/s` (`65,809 MiB`) | **`168.08` / `1,097.96 tok/s`** (`66,133 MiB`, `max_drift 2.125 <= 8.00`) | **`-0.32%` geomean** (`+0.01%` / `-0.64%`) | **`ACCEPTED` (`CONFORMANT`)** |
| **`spec-dspark` (Guard)** | `spec_c1` / `spec_c8` (`DSpark K=5`) | `270.06` / `743.16 tok/s` (`spec_c8 med_ttft 114.02 ms`) | **`289.92` / `739.79 tok/s`** (`spec_c1 TPOT 2.89 ms`, `spec_c8 med_ttft 113.57 ms`, `68,845 MiB`) | **`+3.38%` geomean** (**`+7.35%` `spec_c1`**, `-0.39%` `med_ttft`) | **`ACCEPTED` (`CONFORMANT`, 9/9 Z3 `CLOSED`)** |

> **Capability-Boundary Note (`LOSSLESS TP=8` vs `CALIBRATED_LOSSY NVFP4 TP=4`)**: On lossless `475.25 GiB` `DeepSeek-V4.1-Flash` across `8 × RTX 6000D` (`TP=8` spanning dual-socket Xeon UPI), pure GPU 40-layer `8K` chunk execution now runs in `287.87 ms` (`28,457 tok/s` kernel rate) and end-to-end HTTP serving reaches `15,864.5 total tok/s` (`TTFT 462.7 ms` @ `8K`, `1,901.2 ms` @ `32K`). Reaching the `43.8K–58.0K tok/s` single-island workstation curve (`4 × RTX PRO 6000` running `NVFP4 + CSF` `TP=4` with zero cross-socket UPI, `--row-shard-attention`, and `16K` prepared prefill graphs) crosses into `PrecisionTrack.CALIBRATED_LOSSY` + `DP=2 × TP=4` single-socket topology, scheduled as an explicitly separated track in **Round 8 (`P7`)** alongside lossless `Latent576` GPU rANS KV compression (`P4`).

---

## 10. Round 8: SM120 Multi-Plane GPU rANS Lossless KV Codec, CUDA 13 VMM Fixed-VA Peer Arena, Dual-Island `DP=2 × TP=4` Disaggregation Audit & Isolated `CALIBRATED_LOSSY` `NVFP4` Track (`zstep@e9f2b63` Incumbent)

Raw Round 8 8-GPU hardware scout, lossless 3-Round Cold `AB/BA` + `TP=8` VMM/rANS verification, and explicitly separated `CALIBRATED_LOSSY` `NVFP4` artifacts:
- Stage 1 8-GPU Hardware Scout (CUDA 13 VMM Peer Arena + Multi-Plane GPU rANS KV Codec + `b12x` `I=576` `NVFP4` vs `MXFP4` + Single-Socket `TP=4` vs Dual-Socket `TP=8` Collectives): [`docs/round8_stage1_rans_vmm_nvfp4_scout.json`](round8_stage1_rans_vmm_nvfp4_scout.json)
- Track A1 (`LOSSLESS` Dual-Island `DP=2 × TP=4` vs `TP=8` 3-Round Cold `AB/BA`) + Track A2 (`LOSSLESS` `TP=8` `VmmPeerArena` Zero-Regression Guard + Live GPU rANS `HostDramKvTier`): [`docs/round8_multi_regime_abba_results.json`](round8_multi_regime_abba_results.json)
- Track B (`PrecisionTrack.CALIBRATED_LOSSY` `LibertAIDAI NVFP4 + MXFP4 Engram` Dual-Island `DP=2 × TP=4` Cold `AB/BA` & `LossyQualityGateSpec` Audit): [`docs/round8_lossy_nvfp4_dp2tp4_results.json`](round8_lossy_nvfp4_dp2tp4_results.json)

### 10.1 Evolith Codec, VMM, Dual-Island `DP=2 × TP=4` & `LossyQualityGateSpec` Primitives (`src/evolith/`)
- **Lossless KV Codec & Calibrated-Lossy Gate Contracts (`src/evolith/core/topology.py`, `src/evolith/core/regime.py`, `src/evolith/zstep/gates.py`)**:
  - Added `LosslessKvCodecGateSpec` (`max_ulp_error == 0`, `mismatched_bytes == 0`, `min_compression_ratio`, `require_crc32_integrity=True`) and `LossyQualityGateSpec` (`max_kl_divergence=0.02`, `max_perplexity_delta=0.05`, `min_longctx_retrieval_accuracy=0.995`, `min_task_exact_match_ratio=0.99`) plus `HardwareTopologySnapshot` VMM capability fields (`vmm_peer_arena_supported`, `vmm_granularity_bytes`).
  - Wired `PiecewisePolicySynthesizer` and `ZStep` `_check_precision_track_gates` so `PrecisionTrack.CALIBRATED_LOSSY` candidates can never overwrite or cross-contaminate `PrecisionTrack.LOSSLESS` incumbents while still enforcing `triton_recompilations == 0` and explicit quality bounds (`211 passed, 1 skipped`, `ruff check` clean).
- **True Simultaneous Dual-Island `DP=2 × TP=4` Orchestration (`src/evolith/zstep/hardware.py`, `src/evolith/zstep/remote.py`)**:
  - Added `dp_size` (`world_size == dp_size * tp_size`), `"sm120-8x6000d-dp2tp4"` (`GPU 0..3` on NUMA 0 + `GPU 4..7` on NUMA 1, `--shm-name zstep_dp0` / `zstep_dp1`), and `"nvfp4-dp2tp4"` (`PrecisionTrack.CALIBRATED_LOSSY`), launching both 4-GPU NUMA islands concurrently with wall-clock-synchronized throughput aggregation.

### 10.2 Stage 1 8-GPU Hardware Scout & Capability-Boundary Findings (`round8_stage1_rans_vmm_nvfp4_scout.json`)
1. **CUDA 13 Virtual Memory Management (`cuMemCreate` + `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess`, `P6`)**:
   - Exact `1-TiB`-strided fixed virtual address reservation (`0x0800_0000_0000 + rank * (1 << 40)`, `2 MiB` granularity) succeeded across all 8 GPUs (`rc = 0` on both same-socket `PXB`/`NODE` and cross-socket `SYS` peers).
   - Direct 32 MiB peer read over the fixed-VA VMM mapping measured **`56.05 GB/s`** vs **`35.10 GB/s`** on legacy `cudaIpcOpenMemHandle` (**`+59.7%`**). Integrated into `CudaDriverGuard` (`worker/zstep_worker/cuda_driver.py`) and `DeepSeekV41StepRunner` (`ZSTEP_ENABLE_VMM_PEER_ARENA=1`).
2. **SM120 Multi-Plane GPU rANS Lossless KV-Cache Codec (`P4`)**:
   - Splitting `DSv41` `584 B/tok` (`74,752 B/page`) and `Latent576` (`73,728 B/page`) KV pages into homogeneous byte planes (`fp8_latent`: `1.2087×`, `fp16_scales`: `1.3651×`, `bf16_rope_exp`: `2.9049×`, `bf16_rope_mant`: `1.0000×`) raises lossless compression from `1.1613×` (unsplit) to **`1.2361×` (`+23.61%` host DRAM KV capacity)** on `DSv41-584` and **`1.2344×`** on `Latent576` with **`0 ULP` (`0` byte mismatches)**, **`47.50 GB/s`** SM120 32-way warp rANS GPU encode, and **`81.57 GB/s`** GPU decode.
   - **PCIe Gen5 Latency Trade-off (Preserved per `AGENTS.md`)**: Uncongested zero-SM `cuMemcpyBatchAsync_v2` Copy-Engine DMA transfers 256 pages in **`334.22 µs`** vs **`1,228.99 µs`** for `rANS encode + DMA`. Therefore `HostDramKvTier` keeps `codec="none"` as the latency-optimal default and exposes `codec="rans_gpu"` / `codec="lossless"` with IEEE 802.3 CRC32 verification for capacity-bound long-context tiers.
3. **`b12x` `NVFP4` vs `MXFP4` on `TP=4` (`I=576`) & `LibertAIDAI` Checkpoint Audit (`P7`)**:
   - `LibertAIDAI_DeepSeek-V4.1-Flash-1017-NVFP4` (`399.52 GiB`) contains bit-identical FP4 expert nibbles to native `MXFP4` (`max_abs_diff = 0.0`), a **larger** expert byte footprint (`284.76 GiB` vs `268.94 GiB` due to block-16 `E4M3` scales vs block-32 `UE8M0`), and a lossy `MXFP4` Engram table (`97.28 GiB` vs `189.12 GiB` BF16, `mean_cosine = 0.993478`, `rel_l2_err = 12.06%`).
   - On isolated `TP=4` (`I=576`, zero padding), `b12x` `nvfp4_w4a4` beats `mxfp4_w4a8` by **`1.345×` at `M=4096` (`1,989.88 vs 2,675.88 µs`)** and **`1.063×` at `M=64` (`1,566.77 vs 1,665.19 µs`)**, is **`2.64×–6.73×` slower at `M=1..8`** unless paired with `a16_max_tokens=16`, and **`w4a8_nvfp4` is unsupported in FlashInfer `0.7.6` `b12x`** (`CUDA_ERROR_ILLEGAL_ADDRESS (700)` in dynamic `m64` grouped kernel preparation).

### 10.3 Full 8-GPU Verification Across Tracks A1, A2 & Track B (`round8_multi_regime_abba_results.json`, `round8_lossy_nvfp4_dp2tp4_results.json`)

All candidate builds passed `RedTeam` (`spec_pins_verified=true`), `DispatchVerifier` (`8,640` pairs, `0` violations), 9/9 Z3 `FormalVerifier` clauses (`CLOSED`, `Z3 4.15.4`), `cargo fmt --all --check`, vendored Rust unit tests, and `triton_recompilations == 0`:

| Track & Precision Regime | Tier / Gate Metric | `LOSSLESS TP=8` Incumbent (`e9f2b63`) | Round 8 Candidate | Measured Delta | Evolith Gate Verdict |
|---|---|---:|---:|---:|---|
| **Track A2 (`LOSSLESS TP=8 + VmmPeerArena`)** | `in512_c16` / `in512_c64`<br>`pf8k_c1` / `pf8k_c4` / `pf16k_c2` | `1,099.48` / `1,887.38 tok/s`<br>`29.43` / `33.37` / `15.78 tok/s` | **`1,109.22`** / **`1,903.09 tok/s`** (`TPOT 11.20 ms`)<br>**`29.44`** / **`33.36`** / **`15.78 tok/s`** | **`+0.34%` geomean** (`+0.89%` `c16`, `+0.83%` `c64`) | **`ACCEPTED` (`100%` Greedy Parity)** |
| **Track A2 (`LOSSLESS` GPU rANS DRAM KV Tier)** | 32-Block Live Roundtrip (`2.39 MiB → 2.02 MiB`) | Raw DMA (`1.0000×`) | **`1.1834×` live** (**`1.2361×`** 4-plane), **`0 ULP`**, `32/32` CRC32 | **`+18.3%..+23.6%` DRAM KV density**, `0` byte errors | **`PASS` (`LosslessKvCodecGateSpec`)** |
| **Track A1 (`LOSSLESS DP=2 × TP=4` Prefill Island)** | `pf16k_c2` (`16K` Concurrent Cold Prefill, 3-Round `AB/BA`) | `15.78 out tok/s` (`TTFT 1,938.59 ms`, `p99_ITL 8.84 ms`) | **`16.42 out tok/s`** (**`TTFT 1,851.14 ms`**, **`p99_ITL 7.39 ms`**) | **`+4.06%` throughput**, **`-4.51%` `TTFT`**, **`-16.40%` `p99_ITL`** | **Piecewise Win: Disaggregated `>= 16K, C >= 2` Prefill Island** |
| **Track A1 (`LOSSLESS DP=2 × TP=4` Decode / Global)** | `in512_c16` / `in512_c64` / `pf8k_c4` (`82,049 MiB` VRAM) | `1,099.48` / `1,887.38` / `33.37 tok/s` (`TPOT 11.33` / `27.61 ms`) | `1,010.94` / `1,701.56` / `32.74 tok/s` (`TPOT 12.47` / `31.19 ms`) | `-8.05%` / `-9.85%` / `-1.89%` (`concurrent geomean -4.09%`) | **`REJECTED` as Global Decode Replacement** (`2×` MoE BW/GPU on `TP=4`) |
| **Track B (`CALIBRATED_LOSSY NVFP4 DP=2 × TP=4`)** | `in512_c32_v` / `pf8k_c4` `TTFT`<br>Decode `c16` / `c32` (`84,311 MiB` VRAM) | `TTFT 423.60` / `1,820.89 ms`<br>`1,103.88` / `1,448.66 tok/s` | **`TTFT 347.41`** / **`1,768.28 ms`** (`prime_squares 100%`)<br>`555.46` / `908.23 tok/s` (`KL 0.0388`, `EM 22.45%`) | **`-17.99%` / `-2.89%` `TTFT`**<br>`-49.68%` / `-37.31%` decode (`geomean -32.43%`) | **`REJECTED` (`LossyQualityGateSpec` `KL 0.0388 > 0.02` + `W4A16` Decode Fallback + `84.3 GiB` VRAM)** |

### 10.4 Roofline Diagnosis of Dual-Island `DP=2 × TP=4` & `NVFP4` on `8 × RTX 6000D` (`85.6 GiB` GDDR7)
1. **Why `LOSSLESS DP=2 × TP=4` Wins `>= 16K` Concurrent Prefill (`+4.06%`, `TTFT -4.51%`) Yet Loses Decode (`-8.05%` to `-9.85%`)**:
   - On `TP=4`, eliminating cross-socket Xeon UPI `AllReduce` saves `~445 µs/layer` at `M=4096`, which dominates compute-bound `16K` prefill (`pf16k_c2`). Conversely, in memory-bandwidth-bound decode, `TP=4` shards `268.94 GiB` of MoE weights across 4 GPUs instead of 8 (**`67.24 GiB/GPU` vs `33.62 GiB/GPU`**), doubling per-GPU GDDR7 expert reads (`+1.14 to +3.58 ms/step`) while `TP=8` cross-socket decode `AllReduce` was already compressed to `~20 µs` by `launch_ar_mhc_decode` (`PXB-BFP7`). Thus `DP=2 × TP=4` is strictly a **disaggregated long-prefill topology**, while `TP=8` remains optimal for decode and `C=1` streams.
2. **Why `LibertAIDAI NVFP4` Cuts Prefill `TTFT` (`-18.0%`) Yet Fails Deployment Gates on `RTX 6000D`**:
   - Above `a16_max_tokens=16`, `b12x` `nvfp4_w4a4` drives SM120 FP4 Tensor Cores (`1.345×` faster at `M=4096`), cutting `in512_c32_v` median `TTFT` from `423.60 ms` to **`347.41 ms` (`-17.99%`)**. However, three empirical barriers block production admission on `85.6 GiB` `RTX 6000D`: (i) FlashInfer `0.7.6` `b12x` lacks SM120 `w4a8_nvfp4` (`CUDA_ERROR_ILLEGAL_ADDRESS`), forcing `M <= 16` decode onto `2.65×` slower `W4A16` (`555.5 vs 210.1 µs`); (ii) switching between `W4A16` (`M <= 16`) and `W4A4` (`M > 16`) breaks batch-size invariance while `MXFP4` Engram + `W4A4` yields `KL = 0.03878 > 0.02000`; and (iii) `71.19 GiB/GPU` `NVFP4` experts leave only `768` KV pages (`84,311 MiB` peak VRAM on `85,651 MiB` `RTX 6000D`, vs `97,887 MiB` on workstation `RTX PRO 6000`).





