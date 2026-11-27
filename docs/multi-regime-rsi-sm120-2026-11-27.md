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

