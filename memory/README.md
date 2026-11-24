# Evolith Local Memory Ring (`memory/`)

零网络延迟的本地人机—多 Agent 复利记忆与异步协作目录（借鉴 Inferact **Agent V Memory Ring**、Together **TKCC** 与 MIT **EvolvingInfra**）。

不使用远程 Issue 轮询，所有战略优先级、跨代硬件实测事实（Facts）、惜败踏脚石队列与下一轮演化指令均在本目录以纯文本维护，并由 `ZStepRSILoop`（`src/evolith/zstep/memory_ring.py`）在生成 Proposer Prompt 时自动按 `allowed_roots` 挂载。

## 目录结构与职责边界

```text
memory/
├── README.md                        # 本协议说明
├── playbook.md                      # 推理优化全流程实战十律（40 轮 GPU 实测提炼，Core Laws 自动注入 Prompt）
├── directives.md                    # [人 -> Agent 控制面] 当前高优方向、待验证假设与禁飞区
├── roadmap-rsi-v3.md                # 融合 EvolvingInfra + TKCC + Agent V 的本地总路线图与 Checklist
├── stepping-stones.md               # 正收益惜败踏脚石池（如 +2.8%~+2.9%）与正交合体候选队列
└── facts/                           # 跨会话、跨 baseline rebase 持久化的子系统实测事实卡
    ├── sm120-collectives.md         # 8x RTX 6000D CX8 PXB / SHM v5 / NCCL CE3 通信铁律
    ├── sm120-moe-b12x.md            # b12x MXFP4/W4A8 N64 对齐、真实专家路由偏斜、C-API 发射铁律
    └── dsv41-mla-autotune-sched.md  # FlashInfer 缓存陷阱、b12x MLA、HBM 余量与混合步预算铁律
```

## 安全与不可变门控原则（遵循 `AGENTS.md`）

1. **只读隔离**：未受信 Proposer 运行在项目根目录之外的临时沙箱（`isolated_proposer_sandbox`），无法篡改 `memory/` 或弱化任何门控规则。
2. **APC 前缀友好**：`facts/*.md` 作为高复用静态知识置于 Prompt 前半段（`compressed_lineage_memory` 之前），保持自托管 `zstep` OpenAI 端点的自动前缀缓存（APC）高命中率。
3. **快速交接（Zero-Friction Handoff）**：
   - **人工或交互式调研会话**跑完 `evolith zstep-probe` 后：将新发现的硬件交叉点追加到对应 `facts/*.md`，并将下一轮意图写入 `directives.md` 的 `## Active Directives`；
   - **后台 `evolith zstep-evolve`** 启动时：自动读取 `directives.md` 与关联 `facts/*.md`，无需复制粘贴或同步 GitHub。
