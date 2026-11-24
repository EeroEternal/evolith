# Human & Scout Directives (`directives.md`)

> `ZStepRSILoop` 每轮自动读取 `## Active Directives` 下的非空条目作为最高优先级演化指引。注释或移入 `## Archived` 即可停用。

## Active Directives

- **Microbench-Before-Mutate（真实路由保真）**：修改 MoE（`b12x_dsv41.py`）或 Sparse MLA 前，必须基于真实 384 专家路由偏斜分布或 `zstep-probe` 实测数据，严禁用均匀随机激活推断端到端收益（避免重蹈 `g0017` 覆辙）。
- **优先盘活正交踏脚石组合（Orthogonal Composition）**：单点改动若仅带来 `+1.5% ~ +2.9%` 增益（如 `g0021` 通信/量化 `+2.88%` 与 `g0025` `b12x` 压缩解码 MLA Graph `+2.88%`），优先考虑无文件/无流冲突的正交组合，避免超过 `complexity_free_lines` 奥卡姆行数惩罚。
- **SM120 延迟隐藏优先单流 PDL 而非跨流抢占**：在 `8× RTX 6000D`（156 SMs）上，避免用高 CTA 辅助流与 56-CTA `shm_v5_allreduce` 并发争抢 SM；优先探索同一 `CUDAGraph` 主流内的 CUDA Programmatic Dependent Launch（PDL）尾首重叠或零拷贝路径裁剪。
- **严守单档位零退化红线**：任何针对 `c32`/`c128` 吞吐或 Prefill 的改动，必须保护 `in512_c1`（单流 Decode TPOT）不发生 `> 2.0%` 退化，且保留至少 `150 MiB` Warmup HBM 空闲水位以容纳 Engram `[4096, 25600]` `all_gather`。
- **禁止无停止证据的超时回收（`veriserve` 条件不可能性铁律）**：在存在延迟物理访问（在途 GPU Step、异步 Host/DRAM KV 传输 `KvTransferOp`、跨卡 CUDA-IPC/SHM Ring 写）时，任何仅凭本地墙钟超时或异常启发式将物理页/槽位直接归还 `FREE` 的策略，在数学上不可能同时满足不串数据（`Isolation`）与最终回收（`Finality`）。**没有流/硬件完成证据（`complete(step)` / `cudaEvent` / 全 Rank 屏障），绝不允许因超时或取消将物理页/槽位直接回收到 `FREE`**——取消在途请求必须经 `DRAINING` 等待对应 `step` 完成后释放，无停止证据的超时必须永久 `QUARANTINED` 隔离或触发 `signal_abort()` 停机。
- **控制面改动遵循“锁死账本扼流点（Pinned Ledger Choke-Point）”原则**：进入 `T2-1`（异步乒乓重叠）、`T2-3`（投机回滚）与 `T2-4`（分支前缀缓存/DRAM 分层）前，双缓冲槽位 `(slot, gen)`、KV 页生命周期与 `restore_slot == -1` 哨兵必须由单一开启运行时前置条件断言（违反即致命报错）的账本对象统一管辖并用 `pinned_spec_files` 锁死；Agent 仅在受保护账本外围演化调度启发式与数据面算子。

## Archived

- *(暂无归档指令)*
