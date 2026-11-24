<!-- roots: worker/zstep_worker/models/deepseek_v41.py, worker/zstep_worker/models/deepseek_v41_dspark.py, worker/zstep_worker/kernels_sm120/b12x_dsv41.py -->
# DeepSeek-V4.1 Prefill TBO, Metadata Isolation & DSpark Facts (8×RTX 6000D SM120)

1. **8192-Token Prefill TBO 物理天花板与副流抢占禁区（PR #67–#69 实证，`+10.1%` Prefill / `+5.34%` 100K Cold）**：
   - `shm_ar_tbo`（`blocks=24`、`priority=-1` `_tbo_comm_stream`）可将 `8192` 块 `shm_v5_allreduce`（`201.6 ms`）掩盖 **99.92%**（`wait_ar = 0.15 ms`），主流 GPU 利用率达 **99.4%**；`blocks=8` 反而因通信变慢导致 `100K Cold` 净回退（`132.59 vs 134.03 tok/s`），**`24` 为最优值**。
   - **严禁在已饱和的 TBO 主流旁再叠加高优先级副流算子**：将 `mhc_shifted`（`400 MiB/次` HBM）前移至高优先级 `comm_s` 或在专流 `_tbo_engram_stream` 上异步预取 8 路 `shm_v5_allgather`（`13.99 ms`），均因与 `b12x` MoE/MLA 争抢 HBM/SM 及与 `shm_ar_tbo` 争抢 PXB PCIe 导致冷启动净回退 `-1.6% ~ -2.9%`。
   - `sm120_mhc_post_collapse_rmsnorm` 实测 `1,280–1,285 GB/s` 已贴近 RTX 6000D 持续 HBM 带宽天花板（`__launch_bounds__`/`__ldcs` 仅差 `0.4%`）。
2. **TBO 微批 `u0/u1` 元数据 Builder 原地踩踏铁律（PR #69 实证）**：
   - vLLM 0.31.0 的 `CompressorMetadataBuilder`（`slot_mapping_buffer`）、`DeepseekV4SparseMLAMetadataBuilder`（`compressed_slot_mapping_buffer`、`req_id_per_token_buffer`）与 `DeepseekSparseSWAFlashInferMetadataBuilder`（`_decode_topk_lens`、`_decode_seq_lens`）均复用 builder 实例内持久化 GPU scratch buffer；
   - **严禁在同一 `self._builder_groups` 上连续为 `u0` 和 `u1` 调用 `build()`**（会导致 `u1` 原地覆写 `u0` 的压缩 KV slot mapping 引发 36K 贪心漂移）；必须为 `u1` 持有独立 `_tbo_builder_groups_u1`，且在 `dspark` 下需用单缓冲原地写入 `aux_hidden_states` 避免 `torch.cat` 峰值显存翻倍（`84.2 GiB` 附近会触发 PyTorch allocator 同步抖动）。
3. **`dspark(k=3)` `B=10 @ 100K` 步内耗时与 CPU 发射空泡（实证）**：
   - `zs-scheduler` 在启用 `dspark` 时对所有 prefill 块均置位 `is_dspark_verify=1`，且 `DeepseekV4Model.__init__` 会分配 `dspark` 不使用的 `_mtp_hidden_buffer`；TBO 门控须在 `dspark_drafter is not None` 时放行并按 `DeepseekV4DecoderLayer.forward` 精确捕获 3 层 `aux_hidden_states`。
   - `B=10 @ 100K` 单步 `21.5 ms`（平均接受 `2.70` tok/step）：`verify_graph[10]` CUDA 核耗时 `20.5 ms`（`b12x` MoE `w4a8` `9.09 ms` 占 44.4%、`shm_v5_allreduce` 81 次 `3.04 ms` 占 14.8%、`mhc` `1.99 ms`），但 **`verify` 侧 `prep + _build_step_attn_metadata` 占 `1.10 ms` CPU、`dspark.draft()` 侧 `Self CPU = 3.40 ms` 而 `Self CUDA 仅 1.35 ms`**（`combine 0.30 ms + store_ctx 0.75 ms + dmeta 0.23 ms + sync`），每步存在约 **`~2.5 ms` 纯 CPU/发射空泡**。
   - `P10/P20` 并发下 `dspark(k=3)` 显著优于 `k=4`（`k=4` 在 `c=20` 时 verify 宽度 100 跨过带宽拐点全线回退）。
