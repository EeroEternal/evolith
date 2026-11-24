# Evolith × `zstep` 第一阶段结项报告：`8× RTX 6000D (SM120)` 全栈自进化与长文本突破

- **目标硬件**：`8 × NVIDIA RTX 6000D`（Blackwell `sm_120`，156 SMs/GPU，83.64 GiB 可用 HBM/GPU，双路 NUMA Socket 0 `GPU 0–3` / Socket 1 `GPU 4–7`，Socket 内 CX8 `PXB` ~103 GB/s，跨 Socket 4 路聚合 P2P ~86 GB/s）
- **目标模型**：`DeepSeek-V4.1-Flash`（40 层混合 MoE + Sparse MLA/SWA + Engram + mHC + `dspark` 投机解码）及 `Qwen3.5-35B-A3B` 混合 GDN 验证
- **核心结论**：在严格锁定数值不变性与尾延迟红线的前提下，完成了从**单步算子/通信自进化（累计 6 次终审阶跃晋升）**到**32K–100K 全场景拓扑重构（`PP=2 × TP=4` + 跨步 `1F1B` 流水线解码单配置 4/4 通杀）**的闭环，并在 Evolith 主干落地了 **`veriserve/vstack` 级形式化门控 + 复利记忆环**。

---

## 一、 核心量化里程碑总览

### 1. 自主源码演化谱系（35 代真机演化，`6 / 35` 终审晋升）

两轮受控源码演化（`596d4ef` 8 代 + 主归档库 27 代）全部经过冻结在位基线（Frozen Incumbent）3 轮 AB/BA 交错 `Verify` 裁决：

| 晋升代际 | 进化平面 | 核心物理机制 | 改动规模 | 3 轮交错 Verify 几何平均净增益 |
| :--- | :---: | :--- | :---: | :---: |
| **`g0008-67b2dd20`** | `L2_SCHEDULER` | 流序 `launch/finish` 拆分 + 设备端驻留 Token 传递，消除 `~1.05 ms/step` Host 气泡 | `182 LOC` | **`+5.67%`** |
| **`g0003-b96b840f`** | `L1_KERNEL` | 2-slot 紧凑 CUDA-IPC PXB 环（`122.6 MiB`）+ 双模态网格分派（`8×256` vs `32×512`），解决 Rank 6 Warmup OOM | `112 LOC` | **`+3.72%`** |
| **`g0011-bcaef1a3`** | `L1_KERNEL` | 80× `mHC` post+collapse+RMSNorm+Sinkhorn 融合与三端 `40960` `small_pxb` 锁步对齐 | `195 LOC` | **`+5.19%`** |
| **`g0012-243da173`** | `L1_KERNEL` | `shm_v5_allreduce` Stage-1/Stage-2 全链路 `FP12+BFP7` 2:1 压缩，击穿混合步 50.2% 通信瓶颈 | `156 LOC` | **`+5.93%`** |
| **`g0015-7a608d48`** | `L1_KERNEL` | 56-CTA `.gpu`-scope 4/8-wave 全双工流水 + 128B Xeon DDIO 缓存行对齐 + 根治 FlashInfer `~/.cache` 旧键污染 | `156 LOC` | **`+7.76%`** |
| **`g0018-dc4581de`** | `L1_KERNEL` | 基于真实 384 专家激活偏斜，将 `TP=8` MoE 窄维 `I=288` 零填充至 `I=320` 命中 `b12x` `N64` Tensor Core 核 + 分段 AllGather | `253 LOC` | **`+7.83%`** |

### 2. 32K–100K 长上下文全场景终局成绩（`zstep` PR #67–#78 实证）

通过将双路 PCIe 物理拓扑映射为 **Socket 内 `TP=4`（`shm_v5` PXB BFP7，较 `nccl_TP4` 快 `3.11–3.20×`）× 跨 Socket `PP=2`（`Layer 20` 唯一无共享边界）**，叠加按层混合 Engram（`L1` 驻 GPU / `L14` 异步预取省 `12 GiB`）、`4K/mb` 内联 `dspark` 上下文存储、解耦 `_SwaPagePool (2048)`、`SM120 PDL` 与**跨步双微批 `1F1B` 流水线 Decode（`ZSTEP_PP_DECODE_PIPELINE=1`）**，实现**单套配置 4/4 场景全面达标**（`PARITY_36K: PASS`，峰值显存 `70,915 MiB / 85,651 MiB`）：

| 评测场景 | 并发 × 长度 × 缓存命中 | 实测总吞吐 (`tok/s`) | 首字延迟 (`TTFT`) | 单步解码延迟 (`TPOT`) | 对比基准突破 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`32K_P20_Cold`** | `P=20`, `32K`, `0%` 命中 | **`479.82 tok/s`** | `19.98 s` | `20.18 ms` | 较 `TP=8` 提速 **`+17.2%`**，超目标 `+5.6%` |
| **`100K_P10_Cold`** | `P=10`, `100K`, `0%` 命中 | **`180.82 tok/s`** | `28.99 s` | `25.36 ms` | 较 `TP=8` 提速 **`+22.9%`**，超目标 **`+27.1%`** |
| **`32K_P20_Cache90`** | `P=20`, `32K`, `90%` 命中 | **`1,394.47 tok/s`** | — | **`10.04 ms`** (`med 9.89 ms`) | 较串行 `PP=2` 涨 **`+20.0%`**，TPOT 优于 `TP=8` (`10.29 ms`) |
| **`100K_P10_Cache90`** | `P=10`, `100K`, `90%` 命中 | **`691.22 tok/s`** | — | **`8.63 ms`** (`med 8.32 ms`) | 首次压破 `8.99 ms` 目标线（**`-4.0%`**） |

---

## 二、 分层技术沉淀：这块“SM120 肌肉”具体长出了什么？

1. **跨卡通信层（`kernels_sm120/shm_allreduce*`）**：
   - 支持 `world_size in (4, 8)` 的 2-slot PXB CUDA-IPC + Host SHM 混合环，`FP12+BFP7` 2:1 压缩、128B DDIO 对齐、双网格 `small_pxb` 锁步与 SM120 硬件 `griddepcontrol`（PDL）尾首重叠。
2. **MoE 与注意力算子层（`b12x_dsv41.py`、`candidate_mqa.py`）**：
   - `I=288 -> 320` 甜点填充激活紧凑 `N64` 核、真实专家路由快照校准、SM120 专用 `candidate_mqa` 稀疏索引核、独立 Builder 实例隔离（防 TBO `u0/u1` 原地覆写压缩 KV slot mapping）。
3. **显存与调度控制层（`_SwaPagePool`、`TBO`、`PP=2 1F1B`、`dspark`）**：
   - 14 档精细解码分桶 `(1,2,4,8,10,12,16,20,24,32,48,64,96,128)`；
   - 解耦物理 SWA 页池 `_SwaPagePool`（将全局 MLA 页容量扩展至 `10,240` 页而 SWA 仅驻留活跃窗口）；
   - `TP=8` 单阶段长分块下的 `Prefill TBO`（`blocks=24` 通信流掩盖 `99.92%` AllReduce）与 `PP=2 × TP=4` 下的跨步 `Cohort A/B` `1F1B` 流水线解码。

---

## 三、 Evolith 可信自进化底座沉淀（跨 GPU 永久复用资产）

本阶段在 Evolith 主干（`main@7334591`）建成了不依赖特定 GPU 的三道可信防线：

1. **防作弊独立评测铁笼（`gates.py` / `remote.py` / `loop.py`）**：
   - 冻结在位基线（杜绝劣化基线刷分）、3 轮交错测量 + 配对标准误（Paired SE）检验、单档回退硬上限（吞吐 `-3%` / TTFT & TPOT & P99 ITL `+5%`）、`0` Triton 重编译、36K/贪婪/分块混合步精确对齐。
2. **`veriserve/vstack` 级前置形式化验证门控（`formal_verifier.py` + `dispatch_verifier.py`）**：
   - **三态结论（`CLOSED` / `REJECTED` / `INCONCLUSIVE`）+ 最薄弱环节披露**：区分“确证有 Bug（`REJECTED`）”与“超出建模边界/超时/缺证据（`INCONCLUSIVE`）”，两者均严格按 Fail-Stop 阻断上卡，且不污染演化归档归因；
   - **三明治校验链**：AST 精确静态环字节求值 + 清洗环境 `-I` 子进程 3000 次真实 Python 函数对拍（禁 `__*` 属性，设 `5s` 超时与扰动反空壳自检）+ `.py/.cpp/.cu` 三端 Z3 SMT 蕴含证明；
   - **校准战绩**：`7/7 (100%)` 真实源码变异全部命中预期条款，`14/14` 历史正反候选树回放 `100%` 吻合。
3. **零延迟本地复利记忆环（`memory/` + `memory_ring.py`）**：
   - 提炼全生命周期《推理优化实战十律》（`playbook.md`）、5 张子系统实测事实卡（`facts/*.md`）、正交踏脚石池与 `veriserve` 条件不可能性红线（无停止证据严禁超时回收资源）。

---

## 四、 遗留收尾项与面向下一代多 GPU 环境的解耦清单

### 1. 形式化扫描发现的 `zstep` 边界加固项（建议随手收口）
- **跨语言 SWA 安全容量边界**：`serve.rs` / `cli.rs` 当前设 `min_active_swa = 2 * max_seqs`，但 4-token 投机验证跨页时 `_SwaPagePool` 每序列可钉住 `3` 页且 `dspark` 草稿越页分配未通知 Rust `SwaSafeLru`；建议上调至 `4 * max_seqs` 以严格维持 `SwaSafeLru ⊆ _SwaPagePool`。
- **`dspark` 块表缓存列上界**：Verify `_matches_gpu_bt_cache` 建议覆盖 `(p_max + num_spec_tokens) // 128` 或在 `_last_verify_bt_sig` 中记录 `verified_c_hi`。
- **`PP=1` 下双 PXB 环显存开关**：`TP=8, PP=1` 高水位下避免 `shm_ar` + `shm_ar_tbo` 同时分配双份 PXB HBM IPC 环。

### 2. 面向新 GPU 卡（`SM90 H100/H200/H800`、`SM100 B200`、`SM89 L40S/4090`）的 5 步改造准备
1. **硬件配置抽象（`HardwareProfile`）**：将 `remote.py` 的固定 8 卡检查与 `prepare_sm120_overlay.sh` 抽离为可选钩子与动态 `world_size`；
2. **30s 自动硬件建档（`evolith zstep-init-hw`）**：自动实测目标机器 `(arch, sm_count, interconnect_gbps, warmup_free_hbm)` 生成 `hardware_profile.json` 与 `memory/facts/<hw_id>.md`；
3. **动态形式化预算**：将 `formal_verifier.py` 的 `180 MiB` 环上限与 `kernels_sm120` 路径参数化为按 `HardwareProfile` 注入；
4. **通配算子平面识别**：将 `workspace.py` 与 `redteam.py` 中的 `kernels_sm120/` 升级为 `kernels_*/` 通配；
5. **按硬件指纹分仓与战役选路**：谱系归档按 `.evolith/profiles/<hw_id>-<model>/` 隔离；探针测出 NVSwitch 高带宽时自动跳过有损通信压缩，直奔 `WGMMA / tcgen05` 算子融合与大混批调度战役。
