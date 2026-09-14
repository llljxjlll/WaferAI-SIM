# 矩形 Die Mesh 端到端训练与推理负载支持完善开发计划

日期：2026-09-14  
状态：开发方案；本文新增能力均为计划项，不代表已实现或已验收。  
适用对象：Dense 推理、Dense 训练、MoE 推理、MoE 训练。  
首要执行语义：真实 NpuSim 后端上的时序执行、资源与状态生命周期验证。

## 1. 目标与交付原则

### 1.1 总体目标

用户给定模型、训练或推理负载、物理矩形 mesh、逻辑并行映射、硬件容量与内存策略后，通过统一入口完成编译、内存规划、程序生成、仿真和结果校验。执行范围覆盖：

1. HBM 容量足够时，完整多层模型在矩形阵列上执行。
2. SRAM 不足时，通过合法分块、复用和 HBM 搬运完成执行。
3. HBM 总容量足够但分配不均时，通过明确的放置或远端访问方案完成执行。
4. 全阵列 HBM 也不足时，通过显式外部内存与 offload 方案执行，并计入额外传输、排队和等待。
5. 推理保留并更新跨步 KV；训练保留参数及 optimizer state，连续执行至少两个完整 step。

“任意大小”指容量与格式预算允许的完整矩形，不指无限资源，也不意味着任意模型、任意并行因子都合法。必须公布实际验证过的形状、模型、并行配置和内存策略。

### 1.2 两个核心里程碑与一个扩展里程碑

| 里程碑 | 交付内容 | 必须覆盖的负载 |
|---|---|---|
| M1：HBM 内完整运行 | 参数化多层模型、可配置 TP/DP/EP 映射、连续多步、容量合法、统一真实运行证据 | 四类全部覆盖 |
| M2：超 HBM 运行 | 固定硬件容量下，通过外部内存和真实搬运实现 offload；包含训练 AdamW 状态路径 | 四类全部覆盖 |
| M3：形状与规模扩展 | 1..10 全形状验收、超过 10×10 的资源受限扩展、编译与仿真开销评估 | 四类分别声明验证范围 |

M1、M2 均不得只交付 Dense 推理或仅交付算子组合。训练与推理在同一总体方案内推进；同一阶段可采用先实现公共底座、再依次接入四类负载的开发顺序。

### 1.3 实现原则

- 复用现有 IR、lowering、linker、ProgramIO、finalizer 和 runtime，避免建立另一套只供测试使用的执行器。
- 完整模型在一条可观测的仿真时间线上执行；不得把独立子程序耗时相加称为真实 E2E。
- baseline 完整执行和优化路径完成分别记录；Swizzle、MeshSlice、prefetch overlap 的缺失不阻断合法 baseline。
- 固定模型与物理阵列解耦；不能只通过随 mesh 放大 hidden size、expert 数来证明通用性。
- 所有容量限制作用于实际对象、实际地址和真实生命周期；不能增大测试硬件容量、缩小逻辑负载来掩盖超限。
- 本计划不修改既有冻结 baseline；新增 schema、配置与验收产物使用独立版本。

## 2. 当前基线、复用范围与缺口

以下状态来自本次规划时检查的工作区代码与已有产物，不能据此声称后续代码自动继承旧二进制的验收结论。

| 范围 | 已有能力 | 本轮需要补齐 |
|---|---|---|
| 矩形拓扑 | `RectMeshSpec` 限定每边 1..10、row-major、X-first、一 die 一 rank | 物理形状与逻辑并行组解耦；后续资源预算化 |
| 既有发布矩阵 | 六类 family，8 种代表形状，48 cases / 96 executions；四周边界并非全部实跑 | 增加四类完整模型、多层、多步、内存模式维度 |
| 完整 Dense 推理 | 已有小规模完整前向与静态 Prefill/Decode/Mixed timing evidence | 在矩形统一入口中补齐独立实跑及跨步 KV 闭环 |
| 矩形完整推理入口 | `compile_rect_mesh()` AUTO 走 NAIVE；要求单静态 profile、TP=mesh rank 数 | 参数化并行、运行器接入；STANDARD 不作为前置条件 |
| Dense train | tiny 单层 forward/backward/DP sync/SGD/store 代表性执行 | 完整多层、可配置映射、多步、激活管理及 AdamW 扩展 |
| MoE | 静态 top-1、EP=rank 数、每 die 一个 expert 的推理/训练路径 | 完整模型组合、多 expert/rank、共享参数同步、跨步状态 |
| 持久状态 | HBM backing、显式 load/store、ProgramIO seed/probe、home die 容量检查 | 生命周期感知分配、跨步版本、外部层 residency |
| SRAM | 已有 allocation、view、部分专用复用与 spill 底座 | 审计实际覆盖后补齐通用 planner；不得把专用实现视为通用支持 |
| 大模型实验 | 有容量审计与解析性能投影 | 容量不可行投影不能直接升级为 runtime 成功 |

主要依据：

- [矩形发布落地报告](../../flexible_mesh/reports/任意矩形DieMesh全负载NpuSimRuntime闭环与100Shape发布开发落地报告.md)。
- [Dense 完整前向报告](../../reports/S2开发报告.md)、[静态推理 Profile 报告](../../reports/S3开发报告.md)。
- [主编译器](../../../../llm/frontend/wafer_frontend/compiler.py)、[矩形 workload](../../../../llm/frontend/wafer_frontend/schema/flexible_mesh_workload.py)、[并行组](../../../../llm/frontend/wafer_frontend/schema/flexible_mesh_groups.py)。
- [状态分配](../../../../llm/frontend/wafer_frontend/passes/placement.py)、[状态 schema](../../../../llm/frontend/wafer_frontend/schema/persistent_state.py)、[资源预算](../../../../llm/frontend/wafer_frontend/schema/flexible_mesh_capacity.py)。
- [大模型容量模型](../../../../exps/exp2/exp2_1/capacity_model.py)、[实验报告](../../../../exps/exp2/exp2_1/reports/experiment_report.md)。

## 3. 范围与语义合同

### 3.1 首版必需范围

| 项目 | M1 | M2 |
|---|---|---|
| 拓扑 | 完整矩形、无故障、X-first；允许计算参与子集及空闲 die | 同 M1 |
| rank/core | 一 die 至多一个 workload rank；使用显式 local core placement，不能写死 stride=16 | 同 M1；多 rank/die 另列扩展 |
| Dense 并行 | 合法 TP/DP，PP=1 | 同 M1 |
| MoE 并行 | 明确 TP/EP/DP 语义、静态 top-1；允许多个 experts/rank | 同 M1；动态 top-k 等单独扩展 |
| 推理 | 多层完整 logits 前向；Prefill 后至少 2 个 Decode step；静态 Mixed/Ragged | 加权重与 KV offload |
| 训练 | 多层完整 forward/loss/backward、同步、SGD、至少 2 step；先单 microbatch | 增加 AdamW、optimizer offload；显式激活 checkpoint/recompute 路径 |
| 内存 | SRAM + 有限 HBM；显式 placement 与容量拒绝 | 有限外部 host memory + 共享传输链路 |
| 正确性 | timing、结构、依赖、流量、搬运值与状态版本 | 同 M1；offload 的实际读写与恢复验证 |

M1 的完整训练采用有实际 lowering 的 loss 模板。CE-forward schema 或算子名字存在不能算完成；缺失的 loss/梯度 primitive 必须补齐 timing contract，或者在能力报告中明确拒绝该模型，不得将其从图中省略。

### 3.2 时序执行与数值执行

1. `timing_execution=true` 验证完整逻辑操作已执行，计算量和资源服务来自真实后端，依赖顺序正确。
2. memcpy、DMA、state restore 等具备 payload 语义的操作，应使用非零数据、哨兵和范围检查验证。
3. timing-only 的 GEMM、loss、gradient、optimizer 不能通过 checksum 或固定填充值证明数值正确。
4. 多步训练需验证 step n 的写回完成支配 step n+1 的读取，且读取相同 state 的下一版本；完整更新数值只有在 functional oracle 存在时才可声明。
5. Decode 使用外部给定的 token/request trace，MoE 使用冻结路由 trace；这是 teacher-forced/static-trace 时序工作负载，不声称模型自行生成 token 或数值 gate 决策。
6. 跨进程执行必须显式 checkpoint/reload 并标明开销；M1 优先在单次 simulator invocation 内完成连续步骤。

### 3.3 暂不作为 M1/M2 必需项

- 缺洞、故障或非矩形拓扑，自适应路由；多 rank/die。
- PP/交错 pipeline、通用自动并行搜索、在线 continuous batching、KV 抢占服务策略。
- 通用动态 top-k、随机采样、完整模型数值训练收敛、MLA 等尚无完整后端支持的模型变体。
- NVMe/多级存储、CPU optimizer 执行、ZeRO/FSDP 全套算法。
- 所有 Swizzle/MeshSlice 优化在所有形状上生效、真实性能最优或未经校准的硬件预测。

这些能力必须使用独立 feature 状态，不得静默替换为别的算法或计时模型。

## 4. 总体架构与阶段依赖

```text
ModelSpec + WorkloadSpec + PhysicalMesh + ParallelPlacement + MemoryPolicy
    -> 语义检查 / 模型算子覆盖检查 / 初步资源预算
    -> 完整模型 IR0（四类负载）
    -> 逻辑并行组与 physical placement
    -> 逐 shard state + 全局 action DAG
    -> 生命周期 / 分块 / residency / transport 规划
    -> IR2 / schedule / lowering / LinkedProgramManifest
    -> finalizer / ProgramIO / 实际 artifact 绑定
    -> NpuSim 单次调用内多层、多步执行
    -> observed evidence / 容量与流量核对 / capability report
```

```text
P0 统一合同与入口
  -> P1 拓扑/并行映射
  -> P2 SRAM/HBM 内存规划
  -> P3 四类多层多步 E2E -> M1
P2 的状态/residency 接口 + P3 的完整训练/推理链
  -> P4 外部内存与 offload -> M2
M1 + M2
  -> P5 100-shape 发布与更大规模 -> M3
```

P2 容量检查从 P0 起接入，不能等 P3 后补；P4 的接口合同可在 P2 定义，但在真实外部层实现前保持不可执行。P5 的性能统计与预算检查也从早期持续收集，最终开放范围由实测决定。

## 5. P0：统一负载入口、合同与运行证据

### 5.1 目标

四类负载都能由同一公共入口驱动，输入与输出可重建，现有专用 runner 和 tiny release adapter 不再承担用户侧模型配置职责。

### 5.2 具体开发与测试

| 子计划 | 具体开发内容 | 测试内容与退出条件 |
|---|---|---|
| P0.1 基线审计 | 记录当前源码、二进制、配置 digest；列出四类完整图、算子、状态、lowering 覆盖；区分已运行和仅 schema 支持 | 重跑四类最小可执行基线；缺能力项输出 `unsupported` 或 `not_measured`；不复制旧成功标记 |
| P0.2 输入合同 | 扩展既有 Experiment/Workload schema，包含 model、steps、parallel placement、memory policy、optimizer；统一 timing/functional 字段 | 四类序列化往返、缺字段、未知 enum、非法 model/optimizer 组合；输入变化使 case digest 变化 |
| P0.3 统一编译调度 | 复用完整 Dense compiler、train 和 MoE lowerer；建立 typed family dispatch；把测试目录内可复用 materializer 移入生产包 | 每 family 都产生真实非空 manifest；核对已有可物化范围的 op/state coverage，完整模型缺口留给 P3；不接受只有 stage 名字的计划 |
| P0.4 统一运行器 | 复用 runner/finalizer/resolver/ProgramIO；支持失败目录保留、严格 resume、双独立执行；末尾 one-shot 结束整个 workload | 编译、finalizer、resolver、runtime 各阶段注入失败；不得发布 SUCCESS；resume 拒绝不同配置或二进制 |
| P0.5 能力报告 | 分开描述 full-model、motif、baseline、optimized、runtime、functional、capacity；定义稳定 case ID | 回归证明旧 MeshSlice 成功不能自动提升完整 Dense inference；缺一步 evidence 不得置为 complete |

### 5.3 产物与验收

- 四类最小输入配置、统一 CLI/API、统一运行目录与报告 schema。
- 新 schema 对旧输入提供显式兼容路径或版本错误，不默认改变旧行为。
- 默认普通运行可单次；发布 `repeatability_verified` 需要两次独立 materialize/finalize/resolve/simulate。
- 真实命令名称在实现时固定；本文的接口与字段名称是设计草案，不是当前可执行命令。

主要改动位置：`compiler.py`、`flexible_mesh_compiler.py`、`cli.py`、`runner.py`、`schema/experiment.py`、`schema/flexible_mesh_workload.py`，以及现有 family adapter 与 runtime evidence 模块。

## 6. P1：物理拓扑与逻辑并行解耦

### 6.1 目标

同一个模型在不同矩形上使用合法并行配置，不再依赖 TP 等于 die 总数、DP 等于行数、EP 等于 die 总数。物理网格、逻辑 rank、参数 owner、通信参与者各自有明确身份。

### 6.2 并行与放置合同

- 物理 die 继续使用确定性的坐标和 ID；新增显式参与 die 列表、rank 映射、local core 和 group registry。
- 全矩形保持物理连通；空闲计算 die 可以作为路由中间节点。compute-active 与 routing-active 分开校验。
- Dense：定义 TP 内参数 shard、DP replica、梯度 SUM/MEAN 与 loss normalization；DP 复制不能当作模型参数分片来减少总容量。
- MoE：定义 TP 对 expert 的切分、EP 对 expert 集合的分配、DP 对相同 expert 的复制。显式生成 shared-parameter sync group 与 expert-gradient sync group。
- 首个 MoE 新布局支持 expert 数是 EP 的正整数倍；非均匀 expert 分配后续单独扩展。不能把不同 experts 的梯度放入同一个 AllReduce。
- 首版可使用明确的逻辑坐标 `(dp, ep, tp)`，PP 固定 1；共享 Attention/router 的复制与 TP 分片关系必须写入参数 ownership 表。
- head、hidden、FFN、expert 等维度不满足所选切分时报告具体约束；自动 padding、ragged shard 不得未经定义开启。

### 6.3 具体开发与测试

| 子计划 | 具体开发内容 | 测试内容与退出条件 |
|---|---|---|
| P1.1 Group/placement schema | 扩展组定义与逻辑 rank→die/core 映射；保留旧行列映射为默认策略之一 | 越界、重复、缺 rank、未映射 owner、错误 core；组成员正确且可重建 |
| P1.2 固定模型布局 | Dense TP/DP、MoE TP/EP/DP 与多 expert/rank；允许未使用 die | 固定同一模型在 1×4、2×2、2×3、3×2、3×3、10×10 上选合法映射；额外 die 为空闲时不得复制计算 |
| P1.3 路由与 collective | 按实际组生成 P2P、AG/RS/AR；保留单成员本地退化；有界 wave admission | 奇偶组大小、非连续 die ID、长宽转置、经过 idle die 的多跳路由；send/recv/bytes 配对与无死锁 |
| P1.4 参数/梯度 ownership | 为每个逻辑 tensor 生成 shard、replica 和同步组；解耦逻辑 identity 与物理地址 | 独立 oracle 检查参数无丢失/重复；DP 复制量正确；expert 梯度只在相同 expert replica 间同步 |
| P1.5 runtime core 绑定 | 从实际 hardware 推导 core 数和 local core 映射；清理 lowerer 内硬编码 stride | 至少两种合法每 die core 配置的映射测试及一个真实 canary；core ID 不越格式上限 |

### 6.4 产物与验收

- 参数、状态、group 与 route 的可读映射报告。
- 两类独立测量：固定 global workload 改变布局，以及随资源增加 workload 的扩展实验；二者不能混用吞吐/计算量结论。
- 固定 global workload 下核对有用计算量不变；显式复制、重计算与 padding 另计。
- 全 die 激活的 mesh-scaled smoke 与固定模型可能只用部分 die 的测试分别标记，不能互相冒充。

## 7. P2：容量审计、SRAM/HBM 分配与状态连续性底座

### 7.1 目标

在真实每 core SRAM、每 die HBM、地址和程序格式限制内执行；给后续多步与 offload 提供统一的逻辑状态、物理 backing、驻留和版本合同。

### 7.2 必须统计的内存对象

| 对象 | 推理 | 训练 | 生命周期要求 |
|---|---|---|---|
| 参数 | 只读，可能有实例副本 | 可写，按 optimizer step 更新 | 模型/step 版本 |
| KV | 按 request/layer 保留与追加 | 只在明确采用 KV 的训练语义下存在 | request/step/page |
| 激活 | 瞬时 workspace 和跨算子值 | 保存激活、反向 workspace、可选 checkpoint | op/layer/microbatch/step |
| 梯度 | 无 | parameter gradient、activation gradient、同步临时区 | backward/accumulation/update |
| optimizer state | 无 | SGD 或 AdamW 实际状态 | 跨 step |
| MoE 数据 | routing、dispatch、combine buffer | 增加反向路由与 gradient buffer | layer/forward/backward |
| 系统开销 | 通信 staging、双缓冲、对齐和保留区 | 同左 | 按真实 schedule |

容量审计必须包含临时 workspace、通信 buffer、对齐与预留，不只统计权重和 KV。逻辑 state 大小、物理驻留峰值、传输流量分别记录。

### 7.3 具体开发与测试

| 子计划 | 具体开发内容 | 测试内容与退出条件 |
|---|---|---|
| P2.1 内存能力审计 | 检查所有生产路径中的地址宽度、extent、view、ALLOC/FREE、复用与 spill；建立支持表 | 大于旧窄字段边界的地址/extent canary；超格式上限拒绝，不截断 |
| P2.2 统一 state/residency | 基于现有 PersistentState 增加 state kind、scope、generation、dirty/version、backing 与 residency；HBM/外部地址不进入逻辑 identity | 非法状态转换、旧版本读取、写只读权重、未完成传输即消费、越界 view 的负测 |
| P2.3 精确容量规划 | 按 placement 生成 HBM shard；按 DAG/schedule 推导峰值；独立累计 SRAM/HBM/外部层预算 | per-die 及全局超限、边界恰好放下、alignment padding、共享/复制权重、并发 workspace |
| P2.4 SRAM lifetime reuse | 合法不重叠 lifetime 复用；显式分块与 load/store；兼容已有专用 view/packed storage | reuse 与 no-reuse 对照；哨兵保护相邻区域；异步通信消费完之前不可释放；通量计数不丢失 |
| P2.5 HBM backing/lifetime | persistent 与 transient 分开；状态 alias 明确；完成后可回收临时 backing；大 tensor 按地址/长度合法分段 | KV 跨步保留、临时激活释放、两次训练间参数保留；释放仍被使用的 backing 必须失败 |
| P2.6 HBM 放置不足 | local-home 超限先给出诊断；实现显式 remote placement 模式或编译期重新放置并重新生成路线 | 总容量够但局部不够的 case；远端容量合法、实际有 D2D/HBM 流量；关闭策略时明确失败 |
| P2.7 预算与诊断 | 报告 tensor/state、owner、所需/可用、峰值位置、失败阶段；在大规模展开前做保守预检，finalizer 做精确检查 | SRAM/HBM/record/artifact/tag/session 分别触发超限；保守预检与精确值关系可验证 |

### 7.4 地址与后端约束

- SRAM→HBM spill 不能解决全阵列 HBM 总容量不足；P2 不声称具备 HBM→外部内存功能。
- 审计 Python schema、record literal、relocation、C++ decode、LSU/DTE length 与 backend API 的所有整数转换。
- 能以合法小块表达的大传输优先分段；必须扩宽时同步变更 ISA/format、codec、finalizer、ProgramIO 和 golden 版本。
- 若基于 schedule 回填容量影响 schedule，本轮采用确定性有界重规划；超过迭代预算报告失败，不能静默忽略峰值。
- 远端放置不得假设已有 runtime 支持任意远端 LSU；根据后端能力生成显式本地 staging + transport，并验证实际路径。

### 7.5 产物与验收

交付 `memory_plan`、分层峰值报告、state/version 表、地址映射、显式传输列表。容量足够时无额外 spill/offload；容量不足时存在合法规划或者给出稳定错误码。所有成功 case 的 peak ≤ capacity。


## 8. P3：四类完整模型的多层、多步执行

### 8.1 目标

将已有 tiny 局部能力组合成四类完整模型执行链，接入 P0 入口、P1 映射和 P2 内存规划。M1 主模型至少两层，推理执行一次 Prefill 加至少两步 Decode，训练执行至少两步完整 SGD step。

### 8.2 完整图覆盖要求

| family | 必须出现在语义图、lowered program 与 observed evidence 中的内容 |
|---|---|
| `dense_inference_e2e` | embedding、各层 norm/QKV/RoPE/attention/O/residual/MLP、final norm、LM head、logits、KV load/append |
| `dense_training_e2e` | 完整 Dense forward、loss、各可训练参数的 backward/WGRAD、梯度同步、SGD、参数写回 |
| `moe_inference_e2e` | 完整 Attention 主干、router timing、冻结路由、dispatch、各参与 expert、combine、residual、head、KV |
| `moe_training_e2e` | 完整 MoE forward/loss、Attention 与 expert backward、grad dispatch/dX combine、router/共享参数梯度、对应同步、optimizer 与写回 |

完整算子集合由冻结模型架构决定。某层没有某算子、某参数冻结或路由策略固定时必须在模型语义中明确，不能在 lowering 时直接遗漏。MoE 的硬离散 top-1 assignment 由 trace 固定；router 权重、gate score、辅助 loss 的可训练性和梯度路径需在 P3.4 明确定义。

### 8.3 具体开发与测试

| 子计划 | 具体开发内容 | 测试内容与退出条件 |
|---|---|---|
| P3.1 Dense 推理 | 接入完整前向、静态 Prefill/Decode/Mixed；每 request/layer KV backing 持续存在；terminal logits 明确 | 两层模型、多请求不同上下文、Prefill→Decode→Decode；逐层 attention pairs、KV bytes 与独立公式一致 |
| P3.2 Dense 训练 | 完整 loss/backward、saved activation、WGRAD、DP reduction、SGD 与 state store；step 内和跨步依赖明确 | 每个 trainable parameter 有梯度来源/同步/更新；norm、embedding、head 等不得只验证线性层；两步无旧版本读 |
| P3.3 MoE 推理 | 将现有 dispatch/expert/combine 嵌入多层模型；明确 expert owner、静态路由、零 token expert 与输出保留 | 多 expert/rank、local-only 与 remote 路由、零 token expert、长路径、偏斜流量；所有 token 按合同接收与合并 |
| P3.4 MoE 训练 | 完整反向路由与各参数梯度；固定 assignment 下定义 router/gate score 的 timing 梯度；shared/expert sync 分离 | 无缺失 expert/共享参数梯度；零 token expert 的更新规则明确；不同 expert 不错误规约；两步状态版本连续 |
| P3.5 跨步执行 | 在单 program 或同一模拟实例的明确执行序列中串联步骤；内部 step boundary 与最终 end 分离 | 每步只执行一次；最后一步后资源排空；第一步完成不能误触发终止或 legacy refill；KV 不重置 |
| P3.6 模型覆盖与 oracle | 建立逻辑 op、参数、state、FLOPs/ops、通信量、访存量的独立检查器 | 删除 loss、WGRAD、optimizer、KV append 或最后一层的故障样本必须失败；不能由 producer 自报完成 |
| P3.7 可运行规模阶梯 | 固定 mesh 增加 L、S、batch、experts、steps；固定模型改变 mesh/mapping | 小/中/容量边界至少三档；预算不足保留失败原因；不能将 OOM case 缩小后保留原 case ID |

### 8.4 多步与训练的特别约束

- 初版通过有界静态展开完成少量 step；更长执行采用程序复用/分段时仍需保留依赖、状态与同一时间线。
- loss reduction、梯度累计、DP SUM/MEAN 和 optimizer normalization 必须一致；数值未执行也要有明确数学合同。
- 每个 trainable state 建立 `parameter -> gradient producer -> sync group -> optimizer -> store -> next-step load` 可追溯关系。
- 普通梯度按 step 清零；显式 accumulation 模式才允许跨 microbatch 累计。首版单 microbatch，后续至少两 microbatch 的开发不得复用 step 数伪装。
- `step_commit` 表示该步骤必需写回完成。最终 drain 不等于释放所有模型状态；合法 persistent allocations 和“未完成请求残留”分别报告。
- 推理 KV 容量随实际序列增长；不能每一步始终使用固定 context 而声称验证完整生成过程。
- 冻结 router 或其他参数的模型属于显式变体；不满足完整可训练参数覆盖时不能冒称四类完整训练主验收通过。

### 8.5 M1 验收

1. 四类完整模型，至少两层，推理至少 1 Prefill + 2 Decode，训练至少 2 SGD step。
2. 八个代表矩形的主矩阵完整，且每 case 双独立执行。
3. 每 family 另有固定模型跨形状、合法非默认并行配置、容量边界测试。
4. 实际执行 marker、state lineage、逻辑 work、流量、内存峰值和 drain 全部核对。
5. 模型数值功能标志继续由功能测试决定，不能因 M1 通过自动变为 true。

## 9. P4：外部内存与超 HBM 容量执行

### 9.1 目标

当全阵列 HBM 总容量不足时，通过明确的外部 host memory 和传输操作完成四类负载运行。M2 必须包含“关闭 offload 失败、开启 offload 成功”的同模型同硬件配对证据。

### 9.2 存储层级与初版硬件模型

```text
有限 external host memory
    <-> 共享外部传输链路 / controller queue
    <-> 显式连接的 die HBM ingress
    <-> HBM resident window
    <-> LSU/DTE 与 core SRAM
    <-> compute
```

- 配置外部容量、可见地址范围、带宽、效率、首字节延迟、最大 outstanding、queue depth、方向共享规则与连接 die。
- 首版可以选择 half-duplex 共享带宽；若支持双工，必须分别定义 read/write 资源而非无意叠加带宽。
- 多 die 同时访问同一外部通道时争用同一服务资源，不能每 die 独享完整配置带宽。
- 外部数据到非直连 die 的访问需显式计入 D2D 路由及 staging。首版不假设零成本直达任意 core。
- 外部传输与 HBM/NoC/LSU 的服务边界明确：避免漏计 HBM write/read，也避免同一次 service 被重复收费。
- 仿真宿主机用于存储 backing 的 RAM，不等于被模拟的 external host memory；两者容量、耗时和统计必须区分。

### 9.3 初版内存策略

| 策略 | 行为 | 可运行条件 |
|---|---|---|
| `resident_only` | 所需持久状态留在 HBM，SRAM 使用分块与合法复用 | 按模型合同计算的峰值符合容量 |
| `offload_blocking` | 确定性按层/对象/块换入，消费完成后释放或写回；不假设传输与计算重叠 | 外部容量足够且至少一个合法工作窗口可容纳 |
| `offload_prefetch` | 在有资源与依赖许可时预取，双缓冲计入峰值 | blocking 已验收，且预取本身通过容量和队列预算 |

首版 eviction 使用静态 next-use 与稳定 tie-break；写回 dirty 对象，丢弃 clean 只读缓存。必须优先保护仍被计算、DMA、DTE 或 reduction 使用的 buffer。

即使外部容量足够，最小 compute tile、通信 staging、KV block 或 optimizer chunk 不能同时放入 HBM/SRAM 时仍应拒绝，不承诺 offload 能处理任意小内存。

### 9.4 具体开发与测试

| 子计划 | 具体开发内容 | 测试内容与退出条件 |
|---|---|---|
| P4.1 外部层 schema | 定义 external backing、连接、capacity、链路与 queue；扩展 residency transition | 零容量、地址重叠、home 越界、非法方向、缺连接、未声明外部层却开启 offload 的负测 |
| P4.2 后端传输通路 | 增加或复用版本化 DMA 描述与 runtime backend；区分 external↔HBM 与 HBM↔SRAM；实现真实队列完成事件 | 单笔非零 payload 往返、带宽/延迟公式、双请求串行/并行、跨 die 争用；无丢失、重复 completion |
| P4.3 HBM residency planner | 分块驻留、确定性换入换出、dirty writeback、pin/refcount 与 dependency；重用 P2 allocator | 读前未换入、dirty 数据被驱逐、正在传输时复用、pin 泄漏；HBM 峰值始终合法 |
| P4.4 推理 offload | Dense/MoE 权重按层/专家加载；KV 分块读取与 append；生成真实 attention KV 分块合同 | 权重单独超限、KV 单独超限、二者同时超限；KV 跨步增长；同一块多消费者；不改 attention 有用 work |
| P4.5 训练激活/参数 offload | 保存激活到外部层，反向前恢复；按 shard/chunk 加载参数与梯度；写回下一步使用的状态 | 激活生命周期跨 forward/backward；参数/梯度版本不混淆；forward/backward/optimizer 顺序完整 |
| P4.6 AdamW 可执行路径 | 增加 FP32 master、m、v、step counter、bias correction/weight decay 合同及 timing lowering；compute 在设备侧执行 | 参数/梯度/optimizer 实际 dtype 与 bytes；至少两步；缺 m/v/master load/store 必须失败；不能仅做容量计数 |
| P4.7 激活重计算 | 增加显式 per-block checkpoint policy；生成被恢复的 forward 子图，冻结随机/路由输入 | 与保存全部激活的逻辑梯度结构一致；减少的 activation bytes 和增加的 compute/traffic 可重算 |
| P4.8 预取和重叠 | blocking baseline 通过后添加 bounded prefetch/double buffer；进入相同资源模型 | 单 buffer/双 buffer、带宽饱和、预取不及时、取消/失败清理；不得凭前端直接扣除 cycles |
| P4.9 ProgramIO/状态导出 | 外部层 seed/probe、存储版本、完成排空与 checkpoint/export 合同；大状态使用受控稀疏或分块表示 | 外部 seed 不算 runtime offload；dirty 状态最终 owner 可追溯；导出失败/截断不得发布成功 |

### 9.5 Attention 与训练分块的实现要求

- KV offload 需要后端能够按块消费 KV，并保留跨块 attention 所需状态；仅在完整 Attention primitive 前补一次虚构 load 不能解决完整 KV 工作集不驻留问题。
- timing 模式可以使用 typed 分块 attention contract；必须保持 query-key pairs、KV bytes、softmax/累积工作与依赖，不重复计算整个 attention。
- AdamW 的外部状态换入后在设备侧完成更新；CPU optimizer 不是本轮默认语义。如果未来增加，应建模 CPU 算力和传输边界。
- optimizer 以 chunk 更新时，必须保证该 chunk 梯度已完成同步；新参数不能提前被旧 step 的 backward 使用。
- 激活重计算是独立优化选择，不等价于 offload。M2 同时提供 save/offload baseline 和一个显式 checkpoint/recompute 验收配置。
- MoE 若重新计算路由，必须重放同一冻结 assignment；不能因两次执行选到不同 expert 破坏状态版本。

### 9.6 M2 验收

1. 四类完整多层多步负载都具有同模型、同硬件 `resident_only` 容量失败与 `offload_blocking` 成功配对。
2. 与容量足够的 resident reference 对比：有用 work 一致；新增 external/HBM/D2D bytes、stalls 和 staging 有明确来源。
3. `useful_compute` 与 `recompute_compute` 分开记录；offload 本身不能无意改变模型 FLOPs。
4. 所有层级 peak ≤ capacity；pending external requests、dirty eviction、pin、LSU/DTE/router/credit 等未完成状态为零。
5. 推理权重/KV、训练激活/参数/optimizer，以及 MoE expert 路由对应的内存压力分别覆盖。
6. Dense 与 MoE 的 AdamW resident/offload 对照均通过；不能借用 SGD 的状态占用宣称 AdamW 可运行。
7. timing 主矩阵不宣称更新数值正确；DMA payload、state restore、版本和依赖需要独立证据。

## 10. P5：全形状验收与更大规模支持

### 10.1 目标

先补齐 10×10 envelope 内的真实全形状四类负载验收，再把尺寸限制转化为受格式、资源和实际运行预算约束的能力配置。不得只修改 `MAX_ROWS/MAX_COLUMNS` 后发布更大规模支持。

### 10.2 具体开发与测试

| 子计划 | 具体开发内容 | 测试内容与退出条件 |
|---|---|---|
| P5.1 100-shape runner | 生成确定性 canonical cases、分片、断点续跑、唯一 case 集合和工具绑定 | 覆盖缺失、重复、额外 case、失败 shard、变更二进制后 resume 均可检测 |
| P5.2 资源开销分析 | 分阶段记录构图、规划、序列化、finalizer、模拟的 wall time、RSS、record/manifest/artifact 大小 | 固定模型和 shape-scaled 模型分别画规模曲线；瓶颈归因而非仅报总时间 |
| P5.3 通信与程序展开 | 根据瓶颈引入分层 collective、流式构造、程序模板或分段；复用后仍保留完整语义 | 与小规模全展开基线比对 work/bytes/order；奇数 rank、1D、非连续组；不同算法的数值/规约合同显式声明 |
| P5.4 tag/session 复用 | tag 仅在对应 epoch 所有收发/ACK/WAIT 完成后重用；限制每 core 每 wave 并发 | 延迟包、跨 wave reuse、epoch 混淆、ID wraparound；负测必须拒绝或正确等待 |
| P5.5 超 10×10 schema | 将发布 envelope 与底层可表达能力分开；审计所有 100-rank/10-axis 假设和 core/tag 编码 | max±1 边界、乘法溢出、record/file 超限；不放宽无关 validator |
| P5.6 大矩形真实 canary | 逐步尝试 1×16、16×1、12×12、8×16、16×16；每个通过后再开放下一个 | 四类完整负载、HBM resident/offload；峰值与执行耗时符合显式运行预算；失败保留不可用状态 |

### 10.3 扩展限制的判定方式

- 当前 100 ranks、record、artifact、manifest、tag 和 runtime core ID 等限制需分别辨别是发布策略、实现预算还是 ABI 硬限制。
- 对容量策略的修改不能绕过 ABI；需要 ABI 扩展时先完成版本兼容和 codec/finalizer 测试。
- 已有 10×10 AR fallback linked manifest 约 228 MB，说明程序表示膨胀应在扩大 shape 前测量与处理。
- P5 不预先承诺 16×16 必定可运行；其真实结果进入 capability。M3 的正式验收要求至少完成并发布一个超过 10×10 既有 envelope 的新范围，受阻时不得标完成。
- 如果多个程序段分次加载，必须保持模拟时间、状态和未完成通信的合同；主机分次计时相加不算同一执行时间线。

## 11. 测试体系与精确矩阵

### 11.1 测试分层

| 层级 | 内容 | 主要证据 |
|---|---|---|
| L0 Schema/算术 | topology、sharding、state、overflow、budget、serde | 单元测试、独立公式、错误码 |
| L1 编译结构 | 完整图、参数覆盖、group、route、DAG、memory/lowering | typed manifest 和独立覆盖检查 |
| L2 后端组件 | LSU/DTE/HBM/external、allocator、queue、codec | C++/SystemC 测试、非零 payload、计数与时序 |
| L3 单 case E2E | compile→finalizer→resolver→NpuSim | 原始文件、stdout、observed markers、实际 SHA |
| L4 多步与矩阵 | 四类、多形状、两种内存模式、双跑 | case manifest、派生 completion、资源统计 |
| L5 数值/性能校准 | 小功能 oracle 与已知带宽/延迟服务 | 单独 functional/calibration 状态，不由 L4 推导 |

测试应以实际语义和失效模式为依据，避免只照抄 producer 的输出生成 expected 值。纯文档或机械重命名不新增无意义测试。

### 11.2 代表性主矩阵

固定主形状集合：

```text
1×1, 1×4, 4×1, 2×2, 2×3, 3×2, 3×3, 10×10
```

主 family 为第 8.2 节四类完整模型。推理主 case 内含 Prefill+2 Decode；训练主 case 内含 2 SGD step。所有主 case 至少两层。

| 发布节点 | 主矩阵组成 | 成功 case 目标 | 独立 execution 目标 |
|---|---|---:|---:|
| M1 | 4 families × 8 shapes × resident | 32 | 64 |
| M2 主矩阵 | 4 families × 8 shapes × resident/offload-blocking | 64 | 128 |
| M3 的 100-shape 主矩阵 | 4 families × 100 shapes × resident/offload-blocking | 800 | 1,600 |

这些是未来验收目标，不是当前通过数量。M2 的 resident 与 offload 成功主 case 可使用两种公开硬件容量 profile，但每个 offload case 还必须补一个同低容量硬件下 resident-only 拒绝的负例；三者均保持逻辑模型不变。

每个 shape 的主 profile 必须预先冻结并声明模型是否随 shape 缩放。可使用 shape-scaled 全 die 活跃 smoke 证明覆盖，但仍需下述固定模型补充矩阵。不能用只激活一个 die 的成功结果证明 100 die 通信已经通过。

### 11.3 必须的补充矩阵

| 维度 | 要求 |
|---|---|
| 固定模型 | 每 family 至少一个不随 shape 改变的模型；跨至少 3 个形状、2 种合法 mapping，明确 active die 数 |
| 形状边界 | 额外覆盖 1×10、10×1、2×10、10×2；canonical ID 与主矩阵去重 |
| AdamW | Dense/MoE train × 8 代表形状 × resident/offload；共 32 成功 case、64 executions，单列计数 |
| Mixed/Ragged | Dense/MoE 推理分别覆盖不同上下文、空闲 request slot、KV 页边界；含 resident/offload |
| MoE 路由 | local-only、长路径、均匀/偏斜、零 token expert、多个 experts/rank |
| 内存压力 | 权重、KV、激活、optimizer 单独触发压力，以及联合压力；每项有容量失败/执行成功配对 |
| 重计算 | Dense/MoE train 的保存激活与 checkpoint/recompute 对照，保持输入/路由一致 |
| 预取 | blocking 与 prefetch 的相同逻辑负载对照；不要求所有 case 必然加速 |
| 持续执行 | 至少一个 >2 step 的训练和 >2 Decode 的推理 case；最终排空与累计 work/bytes |
| 硬件 | 至少两种 core 配置；不同 SRAM/HBM、外部带宽和共享连接配置 |

补充矩阵在 P0 冻结 case manifest，正式运行前精确枚举并去重；不把尚未固定的数量混入主矩阵分母。全矩阵数量增长时采用分片和严格 resume，不能降低验收范围后继续使用原 completion 名称。

### 11.4 独立 oracle 与变形测试

- 参数 oracle：按模型维度推导参数集合、元素数、dtype、shard/replica 份数，检查实际 state coverage。
- 计算 oracle：逐 layer、step、request 计算 GEMM FLOPs、attention pairs、vector ops；重计算与 padding 单列。
- 数据 oracle：由逻辑 tensor 和通信算法合同计算 payload bytes；protocol/header/padding/flit bytes 与 payload 分开统计。
- 内存 oracle：对明确 lifetime trace 求每层级 peak；和 planner/runtime 分别核对，不直接复制 planner peak。
- 依赖 oracle：store→next-step load、gradient→sync→optimizer、prefetch→consumer 等关键边的可达性与实际时间先后。
- 拓扑变形：改变物理映射保留逻辑 work；路径与链路流量可以改变，不能错误要求所有 makespan 相同。
- 容量变形：增加容量在 baseline 策略下不应无故增加 compulsory offload；有预取时不强行要求每个策略 makespan 单调。
- 时序下界：共享链路的完成跨度满足服务时间与序列化约束；不能把可重叠阶段的下界简单相加。

### 11.5 关键失败测试

必须覆盖：错误 owner、缺 gradient、错误 loss normalization、重复执行 layer/step、KV 旧版本、错误同步组、非法地址或长度、并发 view 重叠、提前 free、session/tag 耗尽、HBM/外部容量不足、external queue 饱和、错误方向共享、缺 completion、未排空、缺 finalizer/runtime 阶段、修改 artifact 后复用旧 ProgramIO、复制双跑 evidence、错误 resume binding。

### 11.6 真实模型维度与规模阶梯

两层 tiny 矩阵用于证明拓扑、依赖和内存机制，不足以单独完成“大规模 LLM 可运行”的声明。除主矩阵外，P0 应冻结以下三个规模层次：

| 层次 | 模型定义 | 用途与验收 |
|---|---|---|
| Tiny | 至少两层，明确 H/I/head/expert/序列/batch/step 配置 | 四类代表形状和全形状矩阵，快速定位协议错误 |
| Medium | 同架构扩大层数、hidden、序列或 batch，冻结全部维度 | 触发真实 SRAM 复用、HBM 边界、KV 增长、激活/optimizer 压力 |
| Full-dimension | 从仓库已有、算子覆盖合法的 Dense 与 MoE 模型 manifest 各选一份，保持真实层数、hidden、head、expert 数与 dtype | 分别完成推理和训练的 NpuSim 实际执行；选择适配容量和程序预算的矩形，纳入 M3 必需补充项 |

- Full-dimension 不要求读取公开 checkpoint 全部数值；在 timing 模式下使用声明的初始化和 state 表示，但逻辑 tensor 大小、操作数量、访存量和全部层/step 不能缩小。
- 至少四个 full-dimension 成功 case：Dense 推理/训练、MoE 推理/训练；各独立双跑。所用阵列、batch/序列、optimizer 和内存模式逐项公开，不外推未测配置。
- 至少包含推理和训练各一个全阵列 HBM 确实不足、依赖 external offload 才成功的 full-dimension case；若预算阻塞，应记录未完成，不能用解析结果代替。
- 选择模型前先审计架构可执行性；不能将 MLA 等缺失算子改成 MHA 却保留原模型名称。完整维度保留同样适用于 expert 数，不允许只执行活跃 expert 的权重容量就声称全部模型参数已管理。
- 为每档冻结参数量、理论 state bytes、有用 work、程序/manifest 预算、宿主 RSS 与 runtime timeout；具体阈值由 P0 基线测量后给出。超过阈值是诊断与优化输入，不是自动降低负载的许可。

### 11.7 日常回归与发布运行

- 每个子计划先执行对应单元/组件测试和一个真实最小 case；修改共享 state、mapping、codec 或 runtime 时，追加四类直接受影响的代表 case。
- 日常集成覆盖 1×1、非方形、奇数 rank、多步、一个容量失败和一个 offload 压力 case；根据实际耗时确定固定子集。
- M1/M2 发布运行完整代表主矩阵及本里程碑必需补充项；P5 发布运行全形状矩阵、full-dimension 和扩展 shape。
- AdamW/offload/recompute 的补充测试从 M2 起必需；prefetch 测试仅在该功能开放时必需，未开放须标 `unsupported`，不阻断 blocking M2。
- 每个发布检查点使用同一明确 source/tool binding；旧数据只有在源码、producer 输入及实际工具绑定满足严格复用条件时才可 resume。
- 无法完成的 case 按失败阶段进入报告；不能因为其余 case 都通过而把 scope 标为完整。

## 12. 运行产物与完成状态

建议每个 case 输出下列逻辑产物；文件名在 P0 实现时固定并版本化：

```text
case_spec.json                   # 模型、负载、mesh、mapping、memory policy
input_digests.json               # 来源、源码/工具、配置绑定
parallel_placement.json          # rank/group/owner/route
model_coverage.json              # 必需 op/参数/state 与真实 producer 映射
memory_plan.json                 # backing、lifetime、peak、residency
execution_0/ 和 execution_1/
  linked.json / program.npup / finalizer.json / program_io.json
  hardware.json / mapping.spec / simulation.json
  finalizer.stdout.txt / resolver.stdout.txt / npusim.stdout.txt
  observed_runtime.json          # step、work、bytes、peak、drain、makespan
  execution_evidence.json
case_evidence.json
matrix_summary.json              # 仅聚合严格通过验证的 case
```

### 12.1 建议完成状态

以下为拟新增概念字段，不能直接冒用现有 schema：

```text
model_graph_complete
placement_validated
capacity_plan_verified
lower_link_verified
runtime_verified
multi_step_state_verified
offload_runtime_verified
repeatability_verified
compute_functional_verified
model_functional_verified
validation_scope
```

- full-model 与 motif 区分；baseline 与 optimized 区分；static/compile 与 runtime 区分。
- scope 必须区分代表形状、全 100-shape、扩展形状；全形状静态通过不等于全形状 runtime 通过。
- `runtime_verified` 要求真实 finalizer/resolver/NpuSim 成功、实际 work/state coverage、必要 marker 和资源检查。
- `offload_runtime_verified` 还要求真实 external 请求、容量约束、恢复与完成依赖，不能只根据策略字符串设置。
- 双执行比较 artifact、manifest、配置、ProgramIO、语义 marker、模拟 makespan；宿主 wall time/RSS 是统计量，不要求逐字节相等。
- 所有完成属性由 evidence 派生，缺失或不一致时保留失败/未测，不人工置 true。

### 12.2 初始化、写回和残留统计

- ProgramIO 初始化和最终 probe 的宿主开销与模型 runtime 分开；初始化不能代替模型内一次必要的换入。
- resident-only 初始状态可明确设为 HBM 已驻留；offload 初始状态必须说明哪些对象位于外部层，HBM 初始窗口占用也计入。
- dirty 参数或 optimizer state 的最终权威副本允许留在声明的 HBM/external tier；如要求 host-visible checkpoint，显式执行并计时。
- drain 指未完成请求、未归还 credit、未结束 session 等；合法 persistent state 或缓存不要求字节数为零。

## 13. 代码落点与版本迁移

### 13.1 复用的现有模块

| 层 | 主要现有位置 | 本轮职责 |
|---|---|---|
| 入口 | `llm/frontend/wafer_frontend/{cli,compiler,flexible_mesh_compiler,runner}.py` | 公共配置、四类调度、统一执行 |
| schema | `schema/{rect_mesh,flexible_mesh_groups,flexible_mesh_workload,persistent_state,flexible_mesh_capacity}.py` | mapping/state/residency/预算 |
| placement/图 | `passes/placement.py`、`passes/flexible_dense_*`、`passes/flexible_moe.py` 及现有完整前向 pass | 完整模型、ownership、多步 |
| lowering | `lowering/flexible_moe_*`、`lowering/state*`、现有 train/collective lowering | 真实计算、通信、访存程序 |
| Program/ABI | `llm/src/frontend/`、`llm/src/isa/`、对应 `include/` | schema/codec/finalizer/ProgramIO |
| runtime | `llm/src/memory/`、`llm/src/dte/`、`llm/src/workercore/`、`llm/unittest/npusim.cpp` | 分层存储、传输、step 完成与观测 |
| 验收 | `llm/test/frontend/unit/`、`llm/test/frontend/integration/`、相关 C++ selftest | 分层与矩阵证据 |

拟新增职责可命名为 `parallel_placement`、`memory_plan`、`residency`、`external_memory`、`full_workload_runtime`；实施前先确认现有模块是否可扩展。上述名称是设计建议，不表示文件已经存在。

### 13.2 迁移要求

1. 新 input schema 使用显式版本；旧 tiny case、旧 mapping 和旧 one-shot 行为保留兼容路径。
2. 修改 StateABI、地址宽度、record 或指令语义时，同步升级 codec/finalizer/ProgramIO 和测试，不单边扩 Python 字段。
3. 标记明确的 producer/schema 分支；不能为了新模型通过而放宽 generic validator。
4. 旧 baseline 保持原证据边界；新源码通过旧回归只能声明兼容，不把旧二进制结果重新绑定到新代码。
5. 原有 Swizzle/MeshSlice/intra-die 优化挂在可替换策略接口，完整负载 baseline 不依赖优化可用性。
6. 数据实际备份采用稀疏或分块表示时，不得把逻辑容量也变成无限；每页有效性、seed、probe 和被模拟容量仍独立校验。

## 14. 风险、决策与处理方式

| 风险 | 影响 | 处理方式 |
|---|---|---|
| 只有局部算子齐全，完整模型漏 loss/norm/router/head 等 | 误报训练/推理 E2E | P0 建算子覆盖清单，P3 按 trainable state 和完整图独立验收 |
| 固定模型无法整除某 TP | 无法覆盖所有 shape | mapping 合法性与 shape 支持分开；使用合法 TP 和 idle die；padding 另立合同 |
| MoE 共享参数与 expert 复制语义混淆 | 容量、梯度同步错误 | 参数分类 ownership 表，分别生成同步组与字节 oracle |
| SRAM/HBM 窄地址或长度字段 | 大模型截断、越界 | P2 端到端宽度审计；优先合法分段，必要时版本化扩宽 |
| 静态多层多步展开过大 | 编译 RSS/manifest/模拟启动耗时失控 | 从 P0 记录资源；P5 模板/流式/分段；不直接加大限额掩盖问题 |
| offload 忽略外部链路或 HBM 写入成本 | 时间过于乐观 | 明确服务边界，单资源 oracle + 多 die 争用测试 |
| 只有抽象外部 bytes，没有真实请求 | 仍然只是性能投影 | runtime 必须产生队列请求、完成事件和恢复依赖 |
| KV 只能整块 Attention 消费 | HBM 不足时仍不可执行 | P4 增加 typed 分块 Attention，保持精确 work 与状态 |
| 跨步误重置或误 refill | 重复执行/状态丢失 | 显式 step boundary；最终 one-shot；旧/新 version 负测 |
| timing 占位值被当成训练数值 | 夸大功能正确性 | state/version 与 payload 算子测试分开，模型 functional 默认 false |
| AdamW/checkpoint 只有容量假设 | 实际状态与 work 缺失 | M2 要求真实 lowering、状态访问和重计算动作 |
| 外部通道/最小工作集仍不足 | offload 不能完成 | 精确失败诊断；保留未支持状态，不自动换更大硬件 |
| 矩阵数量膨胀 | 发布运行成本高 | 固定主/补充矩阵、分片与严格 resume；抽样只用于日常回归 |

计划默认选定：PP=1、静态 top-1、SGD 先闭环、AdamW 纳入 M2、behavioral host memory 作为第一个外部层、blocking 先于 prefetch、完整数值训练独立开发。无需等待其他优化全部完成才推进这些默认路径。

## 15. 实施顺序、检查点与最终完成定义

### 15.1 推荐执行顺序

1. P0.1–P0.2：冻结能力审计、输入合同、主/补充 case manifest。
2. P0.3–P0.5：统一四类最小运行入口与证据，跑通现有合法 tiny baseline。
3. P1.1–P1.5 与 P2.1–P2.3：先打通 mapping、state、预算三者的接口，再扩展负载。
4. P2.4–P2.7：内存复用、地址、远端放置与诊断完成，保留 resident-only 基线。
5. P3.1–P3.7：逐条完成四类完整模型与多步运行；四类都通过后发布 M1。
6. P4.1–P4.3：外部硬件与 residency 底座；随后 P4.4–P4.7 接入四类及 AdamW/重计算。
7. P4.8–P4.9：预取与状态导出验证；blocking 全矩阵与必需补充项完成后发布 M2。prefetch 的缺失不能冒报优化完成，也不撤销已完成 blocking 能力。
8. P5：全 100-shape 运行、资源瓶颈改造、超过 10×10 的逐级开放，发布 M3。

预取是 M2 后可继续完善的优化子项；M2 必需项为 blocking、四类完整负载、AdamW、一个明确的 checkpoint/recompute 对照及实际 external evidence。完整 P4 阶段完成状态与 M2 里程碑状态分别记录。

### 15.2 每个实现切片的检查点

- 明确新增语义、对应代码位置、兼容影响、测试和成功/失败示例。
- 先完成聚焦单元与组件测试，再跑最小真实 E2E；共享基础变更追加直接受影响的四类回归。
- 测试通过后记录实际命令、工具 SHA、case ID、时长、RSS 和失败清单；没有数据就标未测。
- 发现额外限制时更新 capability 和本计划对应子项，不用缩小样例隐式改变目标。
- 不给未经测量的工期承诺。P0 后根据构图/内存/后端缺口评估工期，每个里程碑以退出条件验收。

### 15.3 最终完成定义

只有同时满足以下条件，才可宣称本计划的完整目标已完成：

1. 四类完整模型由统一生产入口接受参数化输入并实际运行。
2. 模型、逻辑并行和物理矩形解耦，合法配置可执行，非法配置明确诊断。
3. 推理多步 KV 和训练多步参数/optimizer 状态连续，完整 work/state coverage 通过。
4. SRAM/HBM/外部层容量、地址、生命周期与真实流量一致，无隐式无限内存。
5. HBM 不足时的 offload 有真实后端请求与完成证据，不能仅是解析投影。
6. M1、M2、M3 的主矩阵和必需补充项按明确 scope 完成，所有发布成功 case 均满足双独立执行要求。
7. 已发布至少一个超过旧 10×10 envelope 的验证范围，以及第 11.6 节四类 full-dimension 补充验收；其他形状和预算仍明确标识。
8. 数值正确性、优化收益和硬件校准分别报告，未验证部分不随运行能力自动升级。
