# Evolith Unified RSI Roadmap (EvolvingInfra × TKCC × Agent V)

本地持续演进主看板（整合 MIT **EvolvingInfra**、Together **TKCC** 与 Inferact **vLLM AIS / Agent V** 核心启发）。

---

## 一、 演进底座升级（Memory Ring & Multi-Agent Coordination）

- [x] **AV-1：本地零延迟复利记忆环（`memory/` + `memory_ring.py`）**
  - 按 `allowed_roots` 自动注入子系统实测事实卡（`memory/facts/*.md`）、活跃人工指引（`memory/directives.md`）与正交踏脚石提示（`memory/stepping-stones.md`），消除本地调研与后台 `zstep-evolve` 之间的上下文孤岛。
- [ ] **AV-2：两阶段 `Scout -> Patch` 探针内生化**
  - 允许演化循环在写正式源码补丁前，先提交只读微基准脚本自动调用受信任 `zstep-probe`（~35s 8×GPU 算子/通信探针），把实测 JSON 结论回填给当前轮 Builder。
- [ ] **AV-3：正交惜败踏脚石自动合体器（Orthogonal Stepping-Stone Composer）**
  - 从 `archive.sqlite3` 自动筛选同一 `upstream_sha`/父节点下修改范围正交、Verify 净增益处于 `+1.5% ~ +2.99%` 且 `100%` 贪婪对齐的候选对（如通信重叠 + 压缩 MLA Graph），自动叠加补丁送检。
- [ ] **AV-4：单调收紧型元进化（Evening Gate Promoter）**
  - 每隔 $N$ 代自动聚类远端 `Screen`/`Verify` 失败根因，合成 `< 1s` 本地静态守卫或 CPU/Smoke 回归单测（要求拦截历史失败补丁且对历史 `accepted` 补丁 100% 放行），只严不松地前移失败检测。
- [ ] **AV-5：多角色沙箱预选赛（Explorer + Fixer Pre-Tournament）**
  - 单代并行派生多个隔离 Worktree 沙箱探索不同未冻结战役；对探针收益显著但仅因对齐/类型边界报错的候选允许 1 次轻量 Fixer 修复，不立即消耗 2-Strike 熔断配额。

---

## 二、 测量吞吐与硬件保真度（Together Compiler / TKCC 借鉴）

- [ ] **TKCC-1：真实权重路由快照夹具（Real-Routing Fixtures）**
  - 导出并 `spec_pins` 锁死 `DeepSeek-V4.1-Flash` 在真实请求下的每层 `topk_ids / topk_weights` 与激活分布，杜绝随机张量多激活 `13%–38%` MoE 专家导致的微基准失真。
- [ ] **TKCC-2：分层常驻快筛引擎（Tiered Persistent Pre-Screen）**
  - 在全量 475 GiB 冷容器前增加少层数真实权重快筛层（Tier-0 Mini-Stack）或纯 `worker/zstep_worker/` 进程级热重载预筛；**最终 `Verify` 仍强制使用全新离线冷容器执行 3 轮交错测量**。
- [ ] **TKCC-3：SM120 单流 `PDL`（Programmatic Dependent Launch）重叠战役**
  - 在 `CUDAGraph` 同一主流内利用 SM120 硬件 PDL 实现 `shm_v5_allreduce` 收尾与下游 Norm/Quant/Router 前置启动重叠，并在 `redteam.py` 审计 `gdc_wait` 内存可见性边界。
- [ ] **TKCC-4：Kineto / CUPTI 全引擎算子耗时与 Launch Gap 自动归因**
  - 在 `profiling.py` 自动拆解 `Collectives`、`MoE GEMM`、`MLA/SWA`、`Epilogue` 与 `CPU/GPU Launch Gap` 五类占比注入反射卡。

---

## 三、 在线自适应调度、拓扑与正确性（EvolvingInfra & Inferact AIS 借鉴）

- [ ] **INF-1：并行拓扑种子评估与 Prefill `PP × TP` 探索**
  - 在 `perf_model.py` 建模 PCIe/PXB 8 卡上的 `TP=8` vs `EP=2×TP=4` vs **Prefill `PP=2×TP=4`（Copy-Engine 异步交接，规避 80 层同步 AllReduce 并加倍 MoE 宽度 `I=288->576`）**。
- [ ] **INF-2：异构混合 Batch 不变性金丝雀（Batch-Composition Invariance Gate）**
  - 在 `parity` 门控中构造极端异构混合步（长缓存 APC 分块 + 短 Decode + 投机验证同批），拦截依赖 Batch 构成的算子路由精度退化；支持 `zstep-debug-parity` 精度排障模式。
- [ ] **INF-3：`zs-kv` 锁步时钟（`LockstepClock`）+ 剩余空闲时间期望 $-m(\tau)$ 淘汰 + `TP=8` Host KV 单副本去重**
  - 推进 `Campaign T2-4` 与多轮因果 Agent 轨迹重放（`actual_output` 上下文模式 + `stationary` 稳态切入）。
- [ ] **VS-1（`veriserve` 设计先行扼流点）：在启动 `T2-1 / T2-3 / T2-4` 前落地锁死版 `ZsLedger`**
  - 当前 `crates/zs-kv/src/lib.rs:185` 的 `Drop for BlockLease` 直接把物理页推回 `free_ids`（`FREE`），仅在同步单步下安全；一旦开启 `T2-1` 乒乓重叠或 `T2-4` 异步 DRAM KV 传输，取消在途请求会触发跨请求串页（`DelayedAccess ∧ LocalPolicy` 不可能性陷阱）。
  - **落地动作**：先在 `zstep` 将双缓冲槽位 `(slot, gen)`、物理 KV 页 5 态（`FREE / PRIVATE / SHARED / DRAINING / QUARANTINED`）与 `restore_slot == -1` 收拢进唯一开启生产级前置条件检查（违反即抛致命 `LedgerViolation`）的 `ZsLedger` 模块，并将其加入 `pinned_spec_files` 锁死，仅允许演化循环修改账本外围调度与算子。
- [ ] **INF-4：线上影子流量校准 + `zstep` ↔ `evolith.lab (vLLM)` 双轨回流 + 自托管飞轮（> 70% Sessions）**
  - 旁路采集 `Nebula` 脱敏请求的 `(cached_len, uncached_len, output_len, concurrency)` 分位数分布动态生成 `32K–128K` Agent 评测档，避免向静态 `in512/in1024` 过拟合；
  - 将日常 `zstep-evolve` 默认切换至 `--proposer openai` 对接演进后的本地 `zstep / Nebula` 栈，并将验证成熟的 SM120 算子经 `evolith.lab` 适配回流 `vLLM`。
