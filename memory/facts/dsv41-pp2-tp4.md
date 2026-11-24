<!-- roots: worker/zstep_worker/models/deepseek_v41.py, worker/zstep_worker/models/deepseek_v41_dspark.py, worker/zstep_worker/kernels_sm120/shm_allreduce.py, worker/zstep_worker/kernels_sm120/csrc/shm_allreduce_kernels.cu, worker/zstep_worker/runtime_loop.py, worker/zstep_worker/env_config.py -->
# DeepSeek-V4.1 Intra-Socket `PP=2 × TP=4` Hardware Facts (8×RTX 6000D SM120)

1. **双路 PCIe 拓扑与 `DeepSeek-V4.1-Flash` 唯一合法层切分边界（PR #75 实证）**：
   - `8 × RTX 6000D`（单卡 `85,651 MiB` / `83.64 GiB`）为双路 NUMA 拓扑：Socket 0（GPU `0–3`）与 Socket 1（GPU `4–7`）各自 100% 处于 Socket 内（`PXB` + `NODE`），跨 Socket 为 `SYS`；4 路并行跨 Socket NCCL P2P（`GPU i → GPU i+4`）聚合带宽实测 **`86 GB/s`**（`M=40: 0.147 ms`、`M=80: 0.256 ms`、`M=4096: 7.80 ms`）。
   - **唯一合法边界为 `20,20`（Layer 20）**：`L0..19` 完整包含 Engram（`L1, L14`）与 `compress_ratio=2` 压缩 KV 共享组（`L14` 为 `L15..19` 的 compressed-KV 源），`L20..39` 完整包含 `dspark` 目标层 `[37, 38, 39]` 与 `kv_source_layer_id=20`，两阶段零跨阶段 KV/aux 共享；强设 `VLLM_PP_LAYER_PARTITION=18,22` 会被 vLLM 直接拒绝（`Compressed-KV source layers.14.attn not found`）。
   - `dspark` 仅加载在 Stage 1（`ranks 4..7`），启动时须从 Stage 0 向 Stage 1 P2P 同步 `embed_tokens`（`330 MiB/GPU`），且 `rank 4`（Stage 1 `tp_rank=0`）直接向 Rust SHM 写入 `StepResult`（零 Rust 调度器改动）。

2. **Socket 内 `TP=4` `shm_v5` PXB BFP7 最优配置与禁区（PR #75 & 微基准实证）**：
   - `shm_v5_allreduce_kernel` 推广至 `world_size in (4, 8)` 后（`v31_ext`），`M=1..8192` **100% `bit_exact=True`**；`M=40` 耗时 `0.027 ms`（较 `shm_TP8` 快 `1.38×`），`M=4096` 耗时 `0.814–0.818 ms`（较 `nccl_TP4 2.534 ms` 快 **`3.11×`**），`M=8192` 耗时 `1.627 ms`（较 `nccl_TP4 5.199 ms` 快 **`3.20×`**）。
   - **扫频铁律**：`bfp=1, blocks=56` 仍是 `TP=4` 全局最优（`M=4096: 818.2 µs` vs `blocks=32: 903.8 µs` / `blocks=64: 935.2 µs`；`bfp=2` 在 `M=4096` 慢 `3.8×` 达 `3,120 µs`）；`M=40..128` 下 **2-Shot 比 1-Shot 快 `3–5×`**（`M=80`: `35.8 µs vs 139.5 µs`），严禁将 `one_shot_max_vecs` 盲目放宽超 `15,360`。

3. **按层混合 Engram 显存水位与 PCIe 隐藏规律（PR #75 实证）**：
   - `TP=4` 下单层 Engram 约 **`19.5 GiB/GPU`**；两层全驻留 Stage 0 GPU（`47.98 + 2 × 19.5 ≈ 87.0 GiB > 83.64 GiB`）必然 OOM；两层全 `cpu_offload=True`（峰值仅 `52.1 GiB`）则因 `L1` 仅在 `L0` 之后缺乏异步预取窗口，每 `4K` 微批额外阻塞 `+23.3 ms`、且 decode `TPOT` 劣化 `+2.24 ms`。
   - **最优配置 `ZSTEP_PP_ENGRAM_GPU_LAYERS=1`**：仅将无隐藏窗口的 `L1` 驻留 Stage 0 GPU、`L14` 保留 CPU offload（有 `L0..13` 14 层计算窗口 100% 隐藏 `_prefetch_stream` PCIe gather），峰值显存 **`69.8–70.9 GiB / 83.64 GiB`**（仍较 `TP=8` 省 **`12.0 GiB`**），`32K_P20_Cache90` TPOT 从 `13.60 ms` 直降至 `11.18 ms`。

4. **`PP=2` 流水线 Prefill 最优微批粒度、内联 DSpark 与已验证负向试验（PR #75–#76 实证）**：
   - **最佳微批粒度为 `~4,096 tok/mb`**（`target_k = max(min_k, (total_tokens + 2048) // 4096)`）：`total_tokens // 1024` 在 `K=32` 下会把 `32K` 单请求块切碎成 `31 × 1,055` 微批（384-expert MoE 每专家仅 `22` tokens，吞吐腰斩至 `10.8 tok/ms`，多花 `+1.4 s`）；而 `8,192 tok/mb` 在相同 chunk 下令气泡翻倍，净表现不及 `4K/mb`。
   - **流水线双缓冲 + 内联 `dspark` 上下文存储**：Stage 0 `in_flight` 深度 `3` + Stage 1 2 槽静态接收缓冲预发 `irecv` 隐藏每微批 `7.8 ms` 跨 Socket P2P；Stage 1 微批循环内直接对纯 prefill 步执行 `store_prefill_microbatch_ctx()`（复用 GPU `positions_full` / `swa_slot_map_full`），将 `578–900 ms` 的 `spec_finish` 隐藏在 Stage 1 `~15 ms/mb` 等流空窗内并省掉 `4.02 GB` `aux_out` 与 `1.34 GB` `hidden_out` 分配。
   - **实测里程碑（`main@a773c4d`，`8 × RTX 6000D`，`dspark k=3`，`36K` parity PASS）**：`PP=2 × TP=4`（`chunk=233472, K=56, SWA=2048, PDL=1`）达成 **`32K_P20_Cold = 456.15 tok/s`**（`max TTFT 29.68s`, `wall 43.84s`, 较 `TP=8` 提速 **`+11.4%`**）与 **`100K_P10_Cold = 177.73 tok/s`**（`max TTFT 46.28s`, 较 `TP=8` 提速 **`+20.8%`**）；而纯高命中短 prefill 场景（`32K_P20_Cache90 1,429.21 tok/s` / `100K_P10_Cache90 688.17 tok/s`）仍由 `PP=1 / TP=8` 领跑（因 `PP=2` 单步 decode Stage 0/1 串行，权重 HBM 读取带宽为 `TP=8` 一半）。
   - **负向试验（严禁重复踩坑）**：① `PP=2` Stage 0 内嵌 Prefill TBO（`4K→2×2K` 降效 MoE、`8K/mb K=16` 气泡翻倍）均慢于纯 `PP2 (4K/mb)`；② `ZSTEP_ENABLE_SM120_DSA_DECODE=1` 会将 `decode_graph_max_seq_len` 截断在 `16K`，长上下文必须保持 `0`；③ 单步内拆 2-microbatch verify（`B=20→10+10`）需 `T(B/2) < 0.667·T(B)` 才能回本，在 memory-BW bound 的 `B=10..20` MoE decode 下不划算，若要进一步加速 `PP=2` decode 需做跨步（inter-step）双流 `1F1B` 流水线。
