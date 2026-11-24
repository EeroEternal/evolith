<!-- roots: worker/zstep_worker/kernels_sm120/shm_allreduce.py, worker/zstep_worker/p2p_probe.py, worker/zstep_worker/models/deepseek_v41.py -->
# SM120 Collectives & CX8 PXB Hardware Facts (`8× RTX 6000D`)

1. **HBM 峰值水位硬约束（`g0001`–`g0003` 实证）**：
   - `DeepSeek-V4.1-Flash`（`TP=8`，83.05 GiB/GPU）在初始化与 Engram `wkv [4096, 25600]` `all_gather`（需 `200 MiB` 连续块）阶段显存极其紧张；
   - `shm_v5_pxb` CUDA-IPC 环若设为 4-slot / `4096×5120`（`320 MiB`/GPU）会导致 Rank 6 初始化 OOM；**必须保持 2-slot 紧凑环设计（`~122.6 MiB`）**，并在 CuTeDSL warmup 后保持足够空闲块余量。
2. **CTA 数量与跨流 SM 抢占拐点（`g0013`–`g0015`、`g0018`、`g0021` 实证）**：
   - **M=1 / 小 Batch 屏障税**：`shm_v5_allgather` 若盲目扩至 56 CTAs 会在 `M=1` 增加 `+16 µs` 屏障延迟，导致 `in512_c1` 退化超 `-2%` 红线；小 token 段（`M <= 16`）最适合 `8×512` 2-barrier BFP 1-Shot。
   - **M=32 双流抢占陷阱**：当 `mhc_stream` 并发运行时，把 `M=32` 推上 56-CTA 路径会与 `mhc` 争抢 156 个 SM（每步劣化 `+0.24 ms`）；`M=32` 保持 8-CTA 2-Shot 反而更快。
   - **M >= 128 大包全双工与压缩**：Stage-1/Stage-2 全链路 `16B FP12+BFP7` 2:1 压缩 + `128B` Xeon DDIO 缓存行对齐 + `.gpu`-scope 4/8-wave 全双工流水线（56 CTAs，`control[56..63]` 留给 `small_pxb`）可将混合步通信耗时削减 `16%–43%`。
   - **PXB `shm_v5_allgather` 压缩红利（`g0021` 实证）**：为 `shm_v5_allgather` 加入 2:1 `fp12_bfp7` 压缩可将耗时从 `6.13 ms` 降至 `3.21 ms`。
3. **三端形状整除与双网格锁步守卫（`g0004`、`g0009`、`g0010` 实证）**：
   - 修改 `small_pxb` 或 `launch_shm_*` 网格映射时，Python `can_fuse_add_rmsnorm`、C++ `TORCH_CHECK` 与 CUDA `launch_shm_v5_*_cuda` 三端的 `eff_blocks` / `small_pxb` 条件必须完全一致：既要覆盖非 2 幂次中间分桶（如 `rows=24, blocks=32`，避免 `g0004`/`g0009` 的 `rows < blocks requires blocks % rows == 0` 崩溃），也要保证 `all_reduce_cuda` 与 `all_reduce_add_rmsnorm_cuda` 在默认模式下的 `num_vecs` 切分阈值严格相等（避免 `g0010` 的 `10240 vs 40960` 导致 `control` 计数器锁步断裂）。
