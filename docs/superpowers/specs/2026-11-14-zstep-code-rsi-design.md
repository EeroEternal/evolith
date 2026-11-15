# Evolith × zstep 源码级 RSI 循环设计

日期：2026-11-14

## 目标

为 Evolith 增加一个面向 `DeepSeek-V4.1-Flash`（`deepseek_v41`、TP=8、SM120）的源码级递归自我改进循环：Kimi CLI 在隔离的 zstep 候选源码中生成补丁，Evolith 在 `10.23.51.33` 的 8 × RTX 6000D 上执行真实筛选和独立复验，只有全部护栏通过的候选才能成为下一轮基线。

本功能只生成实验结果、补丁和证据，不自动部署、不向上游推送、不修改模型权重。

## 已确认环境

- 本地源仓库：`~/github/zstep`，当前 HEAD 为 `596d4ef`，工作区已有与本任务无关的文档改动，必须保持不动。
- 远端：`node-user@10.23.51.33`，8 × RTX 6000D；开始设计时 8 卡均已空闲。
- 模型 ID 为 `DeepSeek-V4.1-Flash`，宿主机模型目录为 `/data02/DeepSeek-V4.1-Flash-1017`（指向 `/data01/ds41/models/DeepSeek-V4.1-Flash`），已确认 `model_type=deepseek_v41`；先前 `:8000` 的 vLLM 服务已停止，不作为代码生成器或必需依赖。
- 本机存在 `kimi -p`，用作非交互源码候选生成器。
- zstep 已有真实权重 benchmark、`prime_squares_table` 贪心一致性审计和 Triton 零重编译检查，优先复用。

## 隔离与目录边界

所有 zstep 源码副本放在 Evolith 自己的目录中：

```text
evolith/
├── targets/zstep/
│   ├── upstream/          # 从 ~/github/zstep 取得的独立只读基线副本
│   └── candidates/       # 每轮一个独立候选源码副本
└── .evolith/zstep-rsi/
    ├── archive.sqlite3   # 谱系、指标和当前最佳基线
    ├── patches/          # 已接受及被拒绝的 binary patch
    └── evidence/         # 配置、日志、指标、源码哈希和验收结论
```

`targets/zstep/` 和运行产物加入 Evolith 的 `.gitignore`。候选副本从固定基线及已接受补丁栈构造，不要求任何 git commit。原始 `~/github/zstep`、远端 `~/zstep-s0` 和其他 Agent 的目录永不作为候选工作区。

远端每轮使用独立目录：

```text
~/evolith-rsi/runs/<candidate-id>/screen/   # 筛选构建与运行
~/evolith-rsi/runs/<candidate-id>/verify/   # 从冻结源码重新展开的独立复验
```

## 组件

### 1. 任务合同 `ZStepRSITask`

固定记录目标模型、zstep 基线提交、远端主机、硬件、允许修改路径、benchmark 版本、性能阈值、质量护栏、每轮超时和最大代数。默认先运行 1 个完整候选闭环；连续运行必须由用户显式指定代数。

默认允许修改：

- `worker/zstep_worker/models/deepseek_v41.py`
- `worker/zstep_worker/runtime_loop.py`
- `worker/zstep_worker/kernels_sm120/`
- `crates/zs-scheduler/src/`
- `crates/zs-server/src/main.rs`
- 与上述实现对应的现有测试

默认禁止修改依赖清单、模型文件、验证器语义和 benchmark 计分逻辑。单候选默认最多修改 300 行；超出后拒绝并记录原因。

### 2. 候选生成器 `KimiCodeProposer`

每轮在独立候选目录调用 `kimi -p`。输入包括任务合同、当前最佳补丁栈、压缩谱系、最近失败原因和已公开 benchmark 结果。Agent 可以编辑候选源码并运行本地检查，但不能获得远端 SSH 密码，也不能接触隐藏复验档位与种子。

候选生成后，Evolith 而不是 Agent 决定差异是否有效：重新计算 `git diff --binary`、检查 allowlist、修改行数、禁止路径和源码哈希。没有有效 diff、修改越界或检查失败均不消耗 GPU。

### 3. 本地静态门禁

按改动内容运行现有工具链：

- Python：Ruff 与相关非 GPU 单元测试；
- Rust：`cargo fmt --all --check`、相关 `cargo test`，以及可用的 Clippy 检查；
- 通用：补丁范围、冲突、调试残留和 Evolith 现有回归测试。

这些检查只用于筛选，不等于真实性能验证。

### 4. 远端真实筛选

每轮先检查 8 张 GPU 均为空闲；发现占用、模型配置不符或已有同任务锁时立即失败，不停止其他进程。候选源码以不含 `.git`、构建目录和缓存的归档传输到独立 screen 目录。

使用 zstep 已文档化的 SM120 容器和模型挂载运行小规模真实筛选，采集输出吞吐、TTFT、TPOT、P99 ITL、显存、启动时间、贪心一致性和重编译计数。筛选失败只记录证据，不更新基线。

### 5. 独立复验与基线推进

筛选通过后冻结源码哈希，并重新展开到 verify 目录，由与候选生成器无关的 Evolith verifier 进程执行。最终验收使用当前最佳基线与候选的 round-major 交错 `n=3` 测量，以及未提供给生成器的档位和种子。

默认验收条件：

- 目标档位输出吞吐几何平均提升至少 3%；
- 任一公开或隐藏档位吞吐回退不超过 3%；
- Median TTFT、Mean TPOT、P99 ITL 回退不超过 5%，除非任务合同显式定义不同的 Pareto 目标；
- `prime_squares_table` 在 single、batched、mixed-chunked 下均保持 128/128 精确一致；
- `TRITON_RECOMPILATIONS=0`，进程正常退出，Worker、容器、共享内存和显存完成清理；
- 复验源码哈希与筛选候选完全一致。

全部通过才把候选补丁追加到最佳已验证补丁栈。原始冻结基线永久保留，用于报告累计收益；下一轮相对新的最佳已验证基线搜索。

### 6. 谱系与证据

每个候选保存：父基线、假设、补丁、修改统计、源码哈希、本地检查、公开筛选结果、隐藏复验结果、失败根因和最终结论。证据目录包含原始 stdout/stderr、JSON 指标和可重放命令，但不保存 SSH 密码或其他凭据。

CLI 增加：

- `evolith zstep-evolve`：运行指定代数；
- `evolith zstep-status`：查看当前最佳基线、候选谱系和最近失败；
- `evolith zstep-evidence <candidate-id>`：输出证据位置及验收摘要。

## 错误处理与清理

- GPU 忙碌、模型不匹配、远端锁存在：在写入候选运行目录前失败。
- 候选生成或本地门禁失败：记录并进入下一轮或达到代数后退出，不使用 GPU。
- 远端超时或崩溃：只终止本候选拥有的进程组/容器，保留日志，不更新基线。
- 复验失败：拒绝候选，保留补丁和证据，当前最佳基线不变。
- 任何情况下都不自动停止 vLLM、不清理他人 `/dev/shm` 条目、不重启服务、不推送代码。

## 测试与验收

1. 单元测试覆盖任务合同、allowlist、补丁栈、验收阈值、隐藏配置隔离、状态恢复和 GPU 忙碌拒绝。
2. 使用 fake proposer 与 fake remote runner 完整测试接受、拒绝、崩溃和下一轮基线推进，不在 CI 要求 GPU。
3. 运行 Evolith 和 zstep 的现有本地测试及静态检查。
4. 在真实服务器执行一次 Kimi 生成的源码候选：本地门禁、远端 smoke、独立复验、接受或拒绝、证据落盘均实际发生。
5. 若首轮候选无收益，只要循环正确拒绝且证据完整，即视为功能验收通过；不承诺首轮一定获得性能提升。

## 非目标

- 不在首轮实现多服务器、多模型或自动部署。
- 不自动创建上游 PR 或合并代码。
- 不用 DreamSimulator 结果替代真实 GPU 验收。
- 不将 DeepSeek API 服务作为必需组件；代码生成由本机 Kimi CLI 完成。
