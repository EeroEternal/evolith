<!-- roots: worker/zstep_worker/models/deepseek_v41.py, crates/zs-scheduler/src, crates/zs-kv/src -->
# DeepSeek-V4.1 MLA Attention, Autotune Cache & Scheduler Facts

1. **FlashInfer Autotune 缓存目录与 `runner_hash` 陷阱（`g0014`–`g0015` 实证，`+7.76%`）**：
   - `_load_flashinfer_autotune_cache()` 若宽泛扫描整个 `~/.cache`，会误加载 `~/.cache/vllm/flashinfer_autotune_cache/...` 到剥离了 `runner_hash` 的 `_file_configs` 中，导致 `search_cache()` 错误命中 `runners[0]` 而跳过最优 Runner，使二次启动的 `M=1` TPOT 劣化 `+0.32..0.52 ms`；
   - **必须严格限定扫描 `~/.cache/flashinfer`**，保留 `b12x` 无损 `SelectionCache` 与精确按 `runner_hash` 索引的内存 `_winner_cache()`。
2. **`b12x.attention.compressed_sparse_mla` 分派铁律（`g0024`–`g0026` 实证）**：
   - `b12x` 压缩稀疏 MLA 在 Decode `CUDAGraph` 内（配合全宽 `sl=128, il=512` Split-KV 调优）可在 `in512_c1` 提速 `+18.36%`、`in512_c16` 提速 `+6.60%`（Verify 整体 `+2.88%`）；
   - **严禁在短 Eager Prefill 直接走其通用 Python `bind+run` 包装器**（每次调用 `346 µs` CPU 开销会导致 `in512_c32` P99 ITL 暴涨 `+13.55%`）；Decode 路径应通过 `torch.cuda.is_current_stream_capturing()` 严格隔离，Eager 长上下文仅允许走零包装开销的 `_run_sm120_compressed_prefill`。
3. **混合步 Token 预算与尾部延迟平衡（`g0004`、`g0007`–`g0008`、`T1-3`/`T2-4` 实证）**：
   - 当 `decode_active_count == 0` 时若任由混合步暴冲至 `4096` tokens，会越过 `shm_v5_pxb` 的 `3072` 上限跌落到慢速路径；保持 `max_mixed_step_tokens <= 3072` 并结合在线 NNLS 步耗时模型（区分已缓存前缀 $t_c$ 二次注意力项与未缓存 $t_{uc}$）才能同时兼顾吞吐与 `c128` P99 ITL（回退红线 `<= 5%`）。
