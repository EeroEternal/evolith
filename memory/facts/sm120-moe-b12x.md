<!-- roots: worker/zstep_worker/kernels_sm120/b12x_dsv41.py, worker/zstep_worker/kernels_sm120/moe_router.py, worker/zstep_worker/kernels_sm120/norm_quant.py, worker/zstep_worker/models/deepseek_v41.py -->
# SM120 MoE (`b12x` MXFP4/W4A8) & Epilogue Facts

1. **`N64` 紧凑特化核对齐铁律（`g0018` 核心突破，`+7.83%`）**：
   - 在 `TP=8` 下，`DeepSeek-V4.1-Flash` 的 40 层动态 MoE 每卡中间维为 `I_tp8 = 288`（`288 % 128 == 32`）。默认向上取整到 `384` 会错失 `b12x` 针对 `% 128 == 64` 的紧凑 `N64` Tensor Core 特化核与 Autotuner；
   - 将权重零填充 `I = 288 -> 320`（满足 `320 % 128 == 64`，余弦相似度 `0.9999998`，100% 贪婪对齐），并补齐中间 Decode 分桶 `(24, 48, 96)` 的 `moe_warmup_counts`，可直接削减整机 `decode_c16/c32/c128` 步耗时 `-13.9% / -17.6% / -17.6%`。
2. **真实路由偏斜 vs 随机输入失真（`g0017` vs `g0018` + TKCC 铁律）**：
   - 均匀随机输入在 384 专家 MoE 层会比真实文本多激活 `13%–38%` 的专家；任何 MoE 分块或阈值决策必须基于真实专家激活分布剖析，否则会出现 `Screen` 高涨但 `Verify` 缩水。
3. **CPU Eager 发射开销陷阱与 C-API 直调（`g0016`–`g0018` 实证）**：
   - `torch.ops.b12x.tp_moe_dynamic_launch` 的 PyTorch `CustomOpDef` 装箱与 `CuTe-DSL` `jit_executor` 参数编组在 Eager 步会产生高额 Python CPU 开销；使用预校验 `capi_func` 直接发射 + 算术偏移复用 Scratch 缓冲，同时**切勿在包装对象中意外持有临时大张量引用**（`g0007` 因此引发 Warmup OOM）。
4. **并行拓扑边界警示（`g0027` 实证）**：
   - 单纯在单轮局部补丁中把 MoE 硬改为 `EP=2 × TP=4` 虽然在 `c32`/`c128` 有增益，但会在 `in512_c1`（单流 `M=1`）引入额外跨组同步税导致 `-2.85%` 退化被拒；拓扑切换需配合小 Batch 旁路或 Prefill/Decode 差异化路径。
