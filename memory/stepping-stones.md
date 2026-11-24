# Stepping-Stones & Orthogonal Composition Queue (`stepping-stones.md`)

记录已通过全部正确性门控（`100%` `prime_squares_table` 单流/批处理/分块精确一致 + `0` Triton 重编译），仅因单补丁净收益略低于 `+3.00%` 门槛而惜败的**正收益踏脚石候选**，供后续打磨或正交组合（Combination Survival）。

## 1. 已晋升里程碑（Accepted Lineage Anchors）

| Candidate | Baseline / Parent | LOC | 3-Round Verify Gain | 核心机制摘要 |
| :--- | :--- | ---: | ---: | :--- |
| `g0003-b96b840f` | `2bea5a1` | 112 | **`+3.72%`** | 2-slot CUDA-IPC PXB 环（释放 `+117.5 MiB` HBM）、`shm_v5_pxb` 扩至 `M<=3072`、双态 Grid (`8×256` / `32×512`)、`MegaGate=1`、Engram AG `M>64` 走 CE3 NCCL |
| `g0011-bcaef1a3` | upstream | 195 | **`+5.19%`** | SM120 `mHC` post+collapse 融合与 `eff_blocks` 整除安全保护 |
| `g0012-243da173` | `g0011-bcaef1a3` | 156 | **`+5.93%`** | 进一步压缩混合步/预填充 `shm_v5_allreduce` 50.2% 通信瓶颈 |
| `g0015-7a608d48` | `02c0395` | 156 | **`+7.76%`** | Stage-1/2 全链路 `FP12+BFP7` 16B PXB 压缩、128B DDIO 对齐 56-CTA `shm_ar` + 修复 `~/.cache/flashinfer` `runner_hash` 丢失导致的 `M=1` TPOT 退化 |
| `g0018-dc4581de` | `g0015-7a608d48` | 253 | **`+7.83%`** | `b12x` MXFP4 MoE 窄维零填充 `I=288 -> 320`（命中 `%128==64` `N64` 特化核 + 全桶调优）+ MoE C-API 直接发射 + `M<=16` `8×512` BFP 1-Shot + `M=32` 8-CTA 保守避让 `mhc_stream` + `M>=128` 4/8-wave 全双工 PXB AR |

## 2. 高价值惜败踏脚石（Near-Miss Positive Candidates，`+2.6% ~ +2.9%`）

| Candidate | Parent / SHA | LOC | Screen | Verify | 核心机制与未过线原因 | 组合 / 打磨方向 |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| **`g0025-22be4e84`** | `13d3f4d` | 120 | `+10.32%` | **`+2.88%`** (`lcb_2se=+2.28%`) | 仅在 `is_current_stream_capturing()`（Decode CUDAGraph）启用 `b12x.attention.compressed_sparse_mla` + 全宽 `(sl=128, il=512)` Split-KV 调优；`in512_c16 +6.60%`、`in1024_c32 +4.90%`，零奥卡姆惩罚，距 `+3.00%` 仅差 **`0.12%`** | **首选合体对象 A**：纯 Decode MLA 注意力路径优化，与通信/MoE/预填充优化完全正交 |
| **`g0021-2d4c4330`** | `g0018-dc4581de` | 200 | `+2.31%` | **`+2.88%`** (`lcb_2se=+2.46%`) | 波次流水 `shm_v5_allreduce` 与 `mhc` 重叠 + PXB `shm_v5_allgather` 2:1 `fp12_bfp7` 压缩（`6.13->3.21ms`）+ `B12X_DENSE_FUSED_QUANT=1`；因 200 LOC 扣除了约 `-1.6%` 奥卡姆复杂度惩罚（`net_gate=1.0128`） | **精简提纯对象 B**：单独剥离低行数高杠杆的 **`shm_v5_allgather` `fp12_bfp7` 压缩 + `B12X_DENSE_FUSED_QUANT=1`**（控制在 `<40 LOC` 零惩罚）与 `g0025` 正交组合！ |
| **`g0026-33d1f67b`** | `13d3f4d` | 120 | `+10.56%` | **`+2.80%`** (`lcb_2se=+2.49%`) | 在 `g0025` 基础上增加 `rows>=1024, max_seq_len>=1024` Eager Prefill `_run_sm120_compressed_prefill`；`c128` 档位基本持平（`-0.02% ~ -0.49%`）未能补足最后 `0.20%` | 说明 `c128` 瓶颈不在 MLA Prefill，需搭配 `c128` 通信/Dense FP8 提速项合体 |
