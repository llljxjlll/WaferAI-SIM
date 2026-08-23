# SwizzlePlanner 第一版开发方案

## 1. 目标与完成定义

本方案的目标是实现第一版可端到端运行的 `SwizzlePlanner`，使前端能够：

1. 在负载成图后自动识别可融合的 `AG+GEMM`、`GEMM+RS`和
   `GEMM+AR`；
2. 根据 IR-0 的数学/分片语义和 IR-1 的物理 Mesh/路由/带宽信息，
   生成合法 Swizzle 候选；
3. 在 `UNFUSED`、Wang 等的 1D Looped CollectiveEinsum 和 MeshSlice 2D
   Output-Stationary 之间作出确定性选择；
4. 将选中候选物化为 typed `FusionPlan`，经 N5 projection、intra-die schedule、
   GlobalAction、N6 lowering、manifest、ProgramIo 和 `npusim` 完成真实执行；
5. 通过同一负载的 `naive` / `swizzle_topo` 策略切换，支持性能对比和消融实验。

第一版的完成必须同时满足：

- 三种融合模式都有 typed candidate、semantic validator 和 negative tests；
- Wang 1D 与 MeshSlice 2D OS 都可产生确定性 action DAG；
- planner 总是保留 unfused baseline，没有收益时必须回退；
- `swizzle_topo` 在 production registry 中激活，且 context/contract 精确闭合；
- 至少一个 Dense TP 推理或训练前向负载使用真实 Swizzle plan 跑到
  `npusim`；
- `naive` 路径的 schema、artifact 和 runtime 行为无漂移；
- 所有候选、代价和选择都有 stable ID 和可重建 provenance。

## 2. 资料与算法命名

本方案依据：

- `refs/Inter-Die Swizzling优化.pdf`；
- Wang 等，*Overlap Communication with Dependent Computation via Decomposition in
  Large Deep Learning Models*；
- Nam 等，*MeshSlice: Efficient 2D Tensor Parallelism for Distributed DNN
  Training*；
- Zhang 等，*Comet: Fine-grained Computation-communication Overlapping for
  Mixture-of-Experts*。

资料中的“Google OD”指 Wang 等的 Looped CollectiveEinsum/通信与依赖计算
分解重叠方法。代码中不使用“Google OD”作为 schema 名，统一使用：

```text
WANG_1D_BIDIRECTIONAL
```

二维候选统一使用：

```text
MESHSLICE_2D_OS
```

## 3. 第一版能力边界

### 3.1 实现范围

- 非个性化 collective：`ALL_GATHER`、`REDUCE_SCATTER(SUM)`、
  `ALL_REDUCE(SUM)`；
- 直接相邻的 `AG+GEMM`、`GEMM+RS`、`GEMM+AR`；
- Dense TP 推理/训练前向图；
- 固定 parallelism 和固定 placement 之后的 Swizzle 搜索；
- 现有 X-then-Y `PairRoute`；
- 普通 Mesh 上的 1D bidirectional line；
- 具有完整矩形嵌入和语义闭合证明的 MeshSlice 2D OS；
- FP16/BF16 输入、FP32 accumulation 的 timing execution；
- 单个 profile 下确定性选择一个 plan，同时保留候选集证据。

### 3.2 不在第一版范围

- 并行切分、placement 和 Swizzle 的联合搜索；
- 非 X-then-Y adaptive routing；
- torus wrap-around 链路伪造；
- multicast tree / reduction tree 折叠；
- MeshSlice LS/RS dataflow；
- MoE Dispatch/Combine 个性化 A2A；
- 动态 token 流量与运行时自适应；
- N4 候选集和 N5 intra-die scheduler 的联合迭代；
- 优化算子的数值 functional correctness 声明；
- 任意 irregular/faulted mesh 上的 2D MeshSlice。

MoE 二阶段行列 A2A 将作为第二版的独立 pattern template，不得伪装为
Wang 1D collective。

## 4. 当前基础和必须改造的硬门禁

### 4.1 可直接复用

`PhysicalGroup.embedding` 已经包含：

- rank placement 与 logical coordinate；
- `PairRoute.die_path/hops/resource_ids`；
- resource capacity；
- canonical traffic profile；
- lane equivalent bandwidth。

可以从现有数据直接派生：

- hop distance；
- transit dies；
- resource incidence；
- 子阵列是否为完整矩形；
- 行列相邻关系；
- 候选流量的瓶颈 resource 和等效带宽。

`FusionActionKind` 已有：

```text
COMP / SEND / RECV / WAIT / REDUCE / LOCAL_COPY / BARRIER
```

因此第一版不需要新增 transport opcode。

### 4.2 当前硬编码限制

现有 pipeline 只支持 `ROW_PARALLEL GEMM -> SUM ReduceScatter`：

- `FusionPartitionContract.GEMM_RS_ALL_V1`；
- `FusedInterDieContract.DIRECT_NAIVE_V1`；
- `NaiveFusionPartition` 硬编码 GEMM+RS；
- `FusionPlan.validate()` 硬编码 `impl=NAIVE`、`ChunkDim.M`、
  GEMM->ReduceScatter 顺序；
- `InterDiePlanningContext` 只接受 `naive`；
- registry 中 `swizzle_topo` 只 declared，尚未 activate；
- N5 projection 和 downstream exact validator 假定现有 DIRECT RS action 形状。

这些是正式接入时必须同步改造的 breaking boundary，不得只放宽一个
validator。

## 5. 总体架构

```text
Logical/IR-0 graph
       |
       | discover_fusion_candidates
       v
FusionCandidate(pattern + semantic witness)
       |
       | placement / group embedding
       v
IR-1 + FusedOpSkeleton + PhysicalGroup
       |
       | build_swizzle_problem
       v
SwizzleProblem
       |
       | feasibility -> pattern shortlist -> concrete enumeration
       v
SwizzleCandidate[] + UNFUSED
       |
       | analytical resource-DAG cost model
       v
SwizzleDecision(selected + ranked evidence)
       |
       | materialize selected candidate
       v
FusionPlan
       |
       v
IR2 projection -> IntraDieSchedule -> GlobalAction -> N6 -> npusim
```

责任边界：

- discovery 只证明数学上可融合；
- planner 只在固定 placement/routing 上选择分解、通信模式与顺序；
- intra-die scheduler 负责 core/SRAM/NoC 实现；
- lowering 只消费 typed action，不猜测 pattern。

## 6. 核心 typed schema

### 6.1 Pattern 和 Algorithm

```python
class SwizzlePattern(str, Enum):
    AG_GEMM = "ag_gemm"
    GEMM_RS = "gemm_rs"
    GEMM_AR = "gemm_ar"


class SwizzleAlgorithm(str, Enum):
    UNFUSED = "unfused"
    WANG_1D_BIDIRECTIONAL = "wang_1d_bidirectional"
    MESHSLICE_2D_OS = "meshslice_2d_os"
```

### 6.2 SwizzleProblem

```python
@dataclass(frozen=True, slots=True)
class SwizzleProblem:
    schema_version: str
    id: str
    source_ir1_id: str
    fused_op_id: str
    pattern: SwizzlePattern
    gemm: SwizzleGemmDescriptor
    collective: SwizzleCollectiveDescriptor
    group: SwizzleGroupView
    hardware_profile: SwizzleHardwareProfile
    constraints: SwizzleConstraints
```

`SwizzleGemmDescriptor` 必须携带：

- M/N/K 和 batch shape；
- LHS/RHS/output layout；
- LHS/RHS/output sharding；
- 每个 logical dimension 的 `BATCH/FREE_LHS/FREE_RHS/CONTRACT` 角色；
- dtype、accumulation dtype 和 FLOPs；
- boundary value IDs。

不得只使用 `(M,N,K)` 判断 AG decomposition axis。

`SwizzleCollectiveDescriptor` 必须携带：

- collective kind 与 reduce op；
- 位于 GEMM 前或后；
- participant ranks；
- logical/rank input/rank output bytes；
- input/output layout 和 sharding transition。

### 6.3 Candidate 和 Decision

```python
@dataclass(frozen=True, slots=True)
class SwizzleCandidate:
    schema_version: str
    id: str
    problem_ref: str
    algorithm: SwizzleAlgorithm
    split_axis: SwizzleTensorAxis
    chunk_count: int
    unroll_degree: int
    rank_programs: tuple[RankProgram, ...]
    buffer_requirements: tuple[SwizzleBufferRequirement, ...]
    topology_witness: SwizzleTopologyWitness
    semantic_witness: SwizzleSemanticWitness
    cost: SwizzleCost


@dataclass(frozen=True, slots=True)
class SwizzleDecision:
    schema_version: str
    id: str
    problem: SwizzleProblem
    baseline: SwizzleCandidate
    ranked_candidates: tuple[SwizzleCandidate, ...]
    selected_candidate_ref: str
    decision_reason: SwizzleDecisionReason
```

`SwizzleDecision` 保留小候选集与选择证据，但第一版只将选中候选物化为
`FusionPlan`。第二版可将前两名候选一并交给 N5 决断。

## 7. 融合模式自动发现

新增独立纯函数：

```python
discover_fusion_candidates(graph: IR0) -> tuple[FusionCandidate, ...]
```

不再由 `logical_expand` 手写某两个 layer name 的候选，而是从直接 DATA edge 和分片状态
差量中发现。

### 7.1 GEMM+RS

```text
ROW_PARALLEL GEMM
  -> unique partial output
  -> SUM REDUCE_SCATTER
```

保留当前 exact 条件，包括直接唯一 DATA edge、唯一 consumer、同
instance/stage/phase/group、PURE、alias-safe 和 K-axis reduction。

### 7.2 AG+GEMM

第一版放行：

```text
ALL_GATHER
  -> unique gathered value
  -> COLUMN_PARALLEL GEMM
```

额外必须证明：

- AG output layout 等于 GEMM 对应 input layout；
- AG axis 能映射到 GEMM 的 non-contracting、contracting 或 batch role；
- GEMM 的其他 operand 可作为 local operand；
- AG 输出没有外部 consumer；
- boundary input/output 顺序可精确重建。

### 7.3 GEMM+AR

第一版放行：

```text
ROW_PARALLEL GEMM
  -> unique partial output
  -> SUM ALL_REDUCE
```

AR 不得当作 RS 校验，必须显式表达：

```text
ReduceScatter-like reduction phase
  + AllGather replication phase
```

边界输出是 replicated full output，不是 rank shard。

### 7.4 公共负向条件

三类 pattern 均必须拒绝：

- 多 consumer 中间值；
- 不凸依赖子图；
- effect token/alias set；
- 不同 instance/stage/phase/group；
- dtype/layout/sharding transition 不闭合；
- 多个候选共享 member；
- sequence-parallel replicated-weight 特殊语义无证明地进入融合。

## 8. 候选生成模式

### 8.1 候选空间矩阵

| Fusion pattern | UNFUSED | Wang 1D | MeshSlice 2D OS |
| --- | --- | --- | --- |
| AG+GEMM | 必须 | 支持 | 语义/矩形闭合时支持 |
| GEMM+RS | 必须 | 支持 | 语义/矩形闭合时支持 |
| GEMM+AR | 必须 | 两阶段支持 | 两阶段语义闭合时支持 |

### 8.2 Wang 1D bidirectional

参数空间：

```text
traversal/order
× split axis
× bidirectional line orientation
× unroll degree {1,2}
× legal chunk count
× initial rank/chunk rotation
```

语义：

- collective 分解为逐步 SEND/RECV/WAIT；
- GEMM 分解为与通信步数对齐的子 GEMM；
- iteration `i` 的计算与 iteration `i+1` 的通信重叠；
- AG+GEMM 根据 dimension role 选择 output slice update 或 partial addition；
- GEMM+RS 使用 loop-carried accumulator 和目标 rank rotation；
- unroll=2 建立两条独立累加链，由 epilogue 对齐并合并；
- bidirectional 同时使用两个相反方向。

物理 Mesh 不具有 wrap-around 链路时，不得伪造 ring。实现应将该候选视为
`ONE_D_PIPELINE` 族：

- 有真实 Hamiltonian cycle 时可用 ring 子变体；
- 普通矩形 Mesh 使用 bidirectional line/snake 子变体；
- ring 与 line 分别计算 prologue/epilogue。

### 8.3 MeshSlice 2D OS

参数空间：

```text
physical row/column orientation
× M/N-to-row/column mapping
× transpose mapping
× legal block size B
× legal slice count S
× row/column direction policy
```

第一版只实现 output-stationary：

- output shard 常驻；
- 行、列两组通信并行推进；
- 只有对应 K slice 的两个 input 都到达时才启动 partial GEMM；
- 使用 blocked slicing 保证连续访存；
- `S` 必须来自被 block size 和 local tensor shape 整除的 divisor；
- 行/列 collective 在 Mesh 上使用 bidirectional line，不假设 torus。

关键语义门禁：真正的 MeshSlice 会改变内部 2D tensor partition。只有以下之一成立时
才可启用：

1. IR-0/IR-1 已有与 `Pr×Pc` 一致的 2D sharding；
2. fusion plan 包含边界重分片，且能证明输入/输出 boundary layout 与原图完全一致。

否则 MeshSlice 候选必须在 Level 0 被删除，不得仅因 placement 呈现 2D 矩形就启用。

### 8.4 GEMM+AR 的两阶段物化

```text
partial GEMM
  -> reduction/ReduceScatter-like pipeline
  -> replication/AllGather pipeline
```

计算通常只能掩盖 reduction 阶段的一部分，replication 阶段进入 epilogue。
代价模型和 action DAG 必须分别统计两阶段，不得直接复制 GEMM+RS 代价。

## 9. 四级判定管线

### 9.1 Level 0：可行性剪枝

前置布尔检查，不调用代价模型：

- semantic witness 完整；
- split axis 可分；
- tile/chunk 整除；
- rank placement 和 route 覆盖完整；
- line/ring/rectangle 谓词成立；
- SRAM 能容纳所需 buffer/double buffer；
- DTE descriptor 和 synchronization flag 不超上限；
- 最小高效 GEMM tile 成立；
- action/buffer count 不超 schema/backend 上限；
- MeshSlice 边界 sharding 语义可证明。

Level 0 必须产生 `SwizzleFeasibilityWitness`，记录每条通过或拒绝理由。

### 9.2 Level 1：闭式判据短名单

使用资料中的计算/通信/距离比例 `q` 对 Wang 1D 和 MeshSlice 2D 排序：

```text
q = distance_factor × compute_intensity / effective_bandwidth
```

第一版将它作为 heuristic ordering，不作为唯一硬剪枝：

- 明显偏向某候选时，该候选排在前；
- 误差带内同时保留两者；
- 长方形必须使用真实 `(x,y)`，不假设方阵；
- 训练/推理先验只影响排序，不代替代价比较。

### 9.3 Level 2：具体切分枚举

`chunk_count`/`slice_count` 只从合法离散集合产生：

- tensor dimension divisor；
- blocked tile divisor；
- SRAM/double-buffer 上限；
- DTE/message overhead 给出的最小 payload；
- in-flight descriptor 上限；
- `eta(tile)` 崩塌点给出的最小高效 tile；
- canonical 候选数量上限。

建议第一版每个 pattern 最多物化 32 个具体候选，排序键固定为：

```text
(algorithm, orientation, split_axis, chunk_count, unroll_degree, initial_rotation)
```

### 9.4 Level 3：代价评估与决断

理论主目标：

```text
T = T_prologue + max(T_compute, T_communication) + T_epilogue
```

实现上使用小型资源约束 action-DAG evaluator：

- COMP 占用 compute resource；
- SEND/RECV 占用 route 中的 egress/link/shared-cut resource；
- WAIT/BARRIER 体现控制依赖；
- DTE 占用 descriptor/sync resource；
- buffer lifetime 统计 SRAM high-water；
- 按 earliest-start 确定性调度；
- makespan 自然包含 prologue/steady/epilogue。

计算时间：

```text
T_comp = FLOPs / (peak_flops × eta(tile))
```

通信 action 时间至少包含：

```text
T_flow = launch + sync + serialization(bytes, bottleneck_bw) + hop_latency
```

候选流量必须重算每个 `resource_id` 的工作量，不应无条件假设组内无自争用。

`SwizzleCost` 必须包含：

```text
estimated/lower/upper cycles
prologue/steady/epilogue cycles
logical bytes / byte-hops / message count
direction and port utilization
control action count / max inflight
SRAM high-water
bottleneck resources
```

决断规则：

1. 时间区间不相交：选最小 `estimated_cycles`；
2. 区间相交：先选端口方向利用率高者；
3. 再选 control action 少、SRAM 压力小者；
4. 仍并列：按 stable candidate ID 决断，并在 `SwizzleDecision` 保留前两名；
5. 所有融合候选都不优于 baseline 时选 `UNFUSED`。

## 10. `eta(tile)` 和代价 profile

第一版采用离线标定，不在每次编译时运行 npusim。

### 10.1 标定输入

- 代表性 M/N/K tile 网格；
- dtype/accumulation dtype；
- 单 Die core 数与 SRAM profile；
- 相同 lowering/primitive 路径；
- warm-up 后多次确定性重复。

### 10.2 产物

```python
SwizzleHardwareProfile(
    peak_flops=...,
    efficiency_points=(...),
    confidence_band=...,
    dte_launch_cycles=...,
    dte_sync_cycles=...,
    max_inflight_dte=...,
    efficient_tile_floor=...,
)
```

profile 必须版本化并将 digest 写入 policy registration/configuration digest。

若第一批标定尚未完成，允许使用：

```text
peak FLOPs point estimate + conservative confidence interval
```

但必须保留 unfused fallback，且不允许冻结正式性能 baseline。

## 11. 完整开发顺序

### W0：冻结 contract 和版本账本

目标：在修改 producer 前一次冻结身份、schema 和文件边界。

工作：

- 冻结 `SwizzlePattern/Algorithm/DecisionReason`；
- 决定 `FusionCandidate`、`FusionPlan`、N4/N5 carrier 的版本迁移；
- 新增 `SWIZZLE_TOPO_V1` contract；
- 冻结 policy interface 是否保持 v1；
- 冻结 cost profile schema/digest；
- 写 immediate-old-version negative tests。

出口门禁：schema 可 strict serde，stable ID 可重建，无 producer 行为变化。

### W1：纯 typed schema 和语义分析

新增建议文件：

```text
schema/swizzle.py
policies/swizzle/semantics.py
```

工作：

- `SwizzleProblem/Candidate/Cost/Decision`；
- tensor dimension-role analysis；
- 三种 pattern 的 semantic witness；
- AG 的 non-contracting/contracting/batch decomposition；
- AR 的 reduction+replication 两阶段视图；
- 所有 schema 的 independent validator。

出口门禁：合成 2-rank/4-rank 数学测试和负测全绿，不 import production compiler。

### W2：自动 fusion discovery

新增：

```text
passes/discover_fusion.py
```

改造：

- `logical_expand` 使用 discovery；
- `FusionSemanticValidator` 按 pattern 分派；
- `FusionCandidate` 携带 pattern/semantic witness；
- `NaiveFusionPartition` 仍只选旧 GEMM_RS，其他 pattern 只在 swizzle contract 下进入规划。

正向 golden：

- TP>1 Dense Transformer 每层自动产生 `AG1+QKV`、`O+RS1`、
  `AG2+GateUp`、`Down+RS2` 四个候选；
- 合成 GEMM+AR 图产生一个候选；
- TP=1 不产生 collective fusion candidate。

负测：多 consumer、effect、alias、错 axis/layout/dtype/group/order、非 SUM、不凸图。

### W3：物理 topology/resource view

新增建议文件：

```text
policies/swizzle/topology.py
policies/swizzle/problem.py
```

工作：

- 从 `PhysicalGroup.embedding` 构造 immutable topology view；
- rectangle/line/cycle 谓词；
- row/column/snake rank order；
- route/resource incidence 重建；
- candidate-specific resource load 重建；
- `SwizzleProblem` production builder；
- IR0/IR1/group/profile provenance 双向闭合。

出口门禁：1xP、2x2、2x4、非矩形、缺 route、跨组 route 测试全绿。

### W4：Wang 1D 候选生成器

新增：

```text
policies/swizzle/wang_1d.py
```

工作：

- AG+GEMM 的 Looped CollectiveEinsum；
- GEMM+RS 的 looped accumulator/rotation；
- GEMM+AR 的 reduction+replication 两阶段；
- bidirectional line；
- unroll degree 1/2；
- prologue/epilogue；
- exact rank_program/action/buffer witnesses。

golden 必须覆盖 Wang 论文中 4-way AG/RS 的 shard/rank 流转，以及双向传输。

### W5：MeshSlice 2D OS 候选生成器

新增：

```text
policies/swizzle/meshslice_2d.py
```

工作：

- Pr×Pc OS tensor view；
- blocked slicing；
- row/column parallel communication；
- K-slice dependency matching；
- rectangular orientation/transpose；
- Mesh line collective；
- boundary sharding witness；
- SRAM/tile feasibility witness。

golden：2x2、2x4、4x4；长方形行列不对称；不整除、非矩形、边界 sharding 不匹配必须拒绝。

### W6：枚举、代价模型和决断

新增：

```text
policies/swizzle/enumerate.py
policies/swizzle/cost.py
policies/swizzle/decide.py
```

工作：

- Level 0–3 完整管线；
- `eta(tile)` interpolation 和 confidence interval；
- resource-DAG evaluator；
- unfused baseline；
- Wang 自动启用收益门禁；
- deterministic ranking；
- 候选上限和缓存。

关键负测：

- 计算不足以掩盖通信时选 unfused；
- 长条 Mesh 改变 Wang/MeshSlice 排序；
- DTE overhead 使过细 chunk 变差；
- SRAM 不足剪除 double-buffer candidate；
- confidence interval 相交时使用 `(port_utilization, control_complexity)`；
- 输入顺序打乱不影响最终 stable decision。

### W7：物化 `FusionPlan` 和激活 policy

新增：

```text
policies/swizzle/materialize.py
policies/swizzle_topo.py
```

改造：

- `FusionPlan` 增加 pattern/algorithm/decision provenance；
- `FusionPlan.validate_against()` 按 pattern 分派；
- `InterDiePlanningContext` 允许精确的
  `swizzle_topo + SWIZZLE_TOPO_V1`；
- `production_registry()` activate `INTER_DIE/swizzle_topo`；
- compiler 从 ExperimentSpec 选择 naive/swizzle；
- naive 仍只允许 `DIRECT_NAIVE_V1`。

出口门禁：同一 IR1 可分别产生 naive 和 swizzle N4；策略/contract 交叉组合全拒绝。

### W8：N5 projection 和 intra-die 兼容

工作：

- 将 pattern-aware rank actions 投影为 COMP/SEND/RECV/WAIT/REDUCE/BARRIER；
- 闭合 chunk slice 到 IR2 buffer view；
- 闭合 loop-carried accumulator 和 double buffer；
- 闭合 AR 两阶段 dependency；
- 确保 `NaiveIntraDiePolicy` 能先正确调度新 action DAG；
- 在 schedule provenance 中保留 swizzle decision ID。

第一版不要同时开发新 intra-die optimized policy。必须先证明新跨 Die plan 在 naive
intra-die scheduler 下可执行，再进行两层联合优化。

### W9：N6 lowering、manifest 和 C++ trust

工作：

- 复用现有 compute/DTE/reduce lowering；
- 禁止通过 shape/opcode 猜测 Swizzle pattern；
- manifest input digest 携带 typed FusionPlan/Decision；
- finalizer 按 top schema/policy/pattern 锁定 action 形状；
- SEND/RECV/WAIT route/token/FSM 闭合；
- buffer lifecycle 与 double-buffer alias 闭合；
- AG/RS/AR 边界 output layout 闭合；
- immediate-old artifact 版本拒绝。

如现有 opcode 无法无损表达某 action，必须停止并新增 typed ABI，不得借用无关
opcode。

### W10：生产负载接入

至少接入三个小型负载：

1. Dense TP `AG+QKV GEMM`；
2. Dense TP `Down GEMM+RS`；
3. synthetic/training `GEMM+AR`。

每个 case 都必须：

- 从 production logical graph 自动发现，不手造 candidate；
- 同时编译 naive 和 swizzle；
- 生成独立 manifest/ProgramIo；
- finalizer 两次 byte-exact；
- actual-SHA ProgramIo；
- `npusim` 两次 marker digest/makespan 稳定；
- ACK/DONE/drain/residual 闭合；
- capability 仅声明 timing execution。

### W11：对比实验和证据

新增对比报告：

```text
case ID
IR0/IR1/FusionCandidate IDs
policy selection and configuration digest
selected algorithm and parameters
analytical cost interval
naive/swizzle artifact IDs
naive/swizzle runtime makespan
D2D logical bytes / byte-hops / packets
port-direction utilization
SRAM high-water / control actions
repeat stability
timing=true, functional=false
```

不要将“Swizzle 一定比 naive 快”作为 schema correctness 门禁。性能改善是实验目标；
编译器 correctness 门禁是语义、provenance、确定性和资源闭合。

## 12. 建议文件布局

```text
llm/frontend/wafer_frontend/
  schema/
    swizzle.py
  policies/
    swizzle/
      __init__.py
      semantics.py
      topology.py
      problem.py
      wang_1d.py
      meshslice_2d.py
      enumerate.py
      cost.py
      decide.py
      materialize.py
    swizzle_topo.py
  passes/
    discover_fusion.py

llm/test/frontend/
  unit/
    test_swizzle_schema.py
    test_swizzle_discovery.py
    test_swizzle_topology.py
    test_swizzle_wang_1d.py
    test_swizzle_meshslice_2d.py
    test_swizzle_cost.py
    test_swizzle_policy.py
  integration/
    swizzle_cases.py
    test_swizzle_cases.py
    run_swizzle_runtime.py
    test_run_swizzle_runtime.py
```

shared 文件修改仅限：

- `schema/ir0.py`；
- `schema/action.py`；
- `schema/n4.py`；
- 必要的 N5/IR2 carrier；
- `policies/registry.py`；
- `compiler.py`；
- `passes/logical_expand.py`；
- projection/lowering/manifest/finalizer 的 exact union/validator；
- public exports 和 CMake official test。

## 13. 测试和门禁

### 13.1 Schema/serde

- strict missing/unknown field；
- immediate-old version 拒绝；
- stable ID 重建；
- candidate/decision restable tamper 拒绝；
- policy/contract/configuration digest 交叉组合拒绝。

### 13.2 Semantic discovery

- Dense TP 每层 4 个候选；
- GEMM+AR synthetic positive；
- TP1 zero candidate；
- 多 consumer/effect/alias/layout/axis/order/group/convexity negatives；
- 旧 GEMM+RS candidate stable semantics 无漂移。

### 13.3 Wang 1D

- 2/4/8 ranks；
- AG+GEMM shard update；
- GEMM+RS accumulator/rotation；
- GEMM+AR two-phase；
- unroll 1/2；
- bidirectional flow；
- line vs real ring；
- prologue/epilogue；
- route/token/action/buffer exact closure。

### 13.4 MeshSlice 2D

- 2x2、2x4、4x4；
- rectangular orientation transpose；
- blocked contiguous slicing；
- K-slice matching；
- row/column concurrent resource use；
- invalid divisor/rectangle/route/SRAM/sharding negatives。

### 13.5 Cost/decision

- unfused fallback；
- chunk 过小的 DTE overhead；
- chunk 过大的 prologue/epilogue；
- eta 下降；
- 长方形不对称；
- resource contention；
- interval tie-break；
- deterministic ordering/cache key；
- planner 不调用 npusim。

### 13.6 End-to-end

- naive/swizzle 两套 policy 同输入；
- N4 plan pattern/algorithm 正确；
- N5 action DAG 依赖正确；
- lowering/manifest/finalizer exact；
- ProgramIo actual SHA；
- official runtime 两次确定性；
- `PROTO_WAIT` 缺失；
- all residual zero；
- naive adjacent regression 全绿。

## 14. 关键风险与预案

### 14.1 MeshSlice 与现有 1D TP sharding 不相容

风险：placement 是 2D 不等于 tensor sharding 是 2D。

预案：Level 0 要求显式 2D sharding/boundary-reshard witness；第一版对不兼容负载自动
删除 MeshSlice，仍可使用 Wang 1D/unfused。

### 14.2 代价模型误差

预案：eta confidence interval + unfused baseline + 误差带内保留两候选；不冻结跨硬件
通用性能结论。

### 14.3 当前 action/FusionPlan validator 过于 RS-specific

预案：先版本化 schema 和 pattern-specific validator，再修 producer；不做无 pattern 条件的
generic relax。

### 14.4 AR 尾部无法被 GEMM 掩盖

预案：将 replication 显式计入 epilogue，允许 planner 选 unfused；不宣称 GEMM+AR 全量重叠。

### 14.5 N4/N5 循环依赖

预案：第一版在 N4 确定性选单一 plan，但保留 `SwizzleDecision` 前两名候选；
第二版再让 N5 对小候选集做最终决断。

## 15. 第一版最终产物

1. 三种 fusion pattern 的自动 discovery；
2. `SwizzleProblem/Candidate/Cost/Decision` typed schema；
3. Wang 1D bidirectional planner；
4. MeshSlice 2D OS planner；
5. 四级判定管线和 unfused fallback；
6. 版本化 hardware cost profile；
7. `SwizzleTopoInterDiePolicy`；
8. pattern-aware `FusionPlan`、N5 projection 和 N6 lowering；
9. naive/swizzle 可配置切换；
10. 三个 production integration cases；
11. official deterministic runtime test；
12. typed comparison/evidence report。

## 16. 完成后允许的能力声明

通过全部门禁后，第一版只允许声明：

```text
AUTOMATIC_AG_GEMM_FUSION_DISCOVERY       = true
AUTOMATIC_GEMM_RS_FUSION_DISCOVERY       = true
AUTOMATIC_GEMM_AR_FUSION_DISCOVERY       = true
WANG_1D_SWIZZLE_PLANNING                 = true
MESHSLICE_2D_OS_PLANNING_WHEN_COMPATIBLE = true
UNFUSED_COST_BASELINE_FALLBACK            = true
SWIZZLE_TIMING_EXECUTION                  = true
SWIZZLE_FUNCTIONAL_EXECUTION              = false
JOINT_PLACEMENT_SWIZZLE_SEARCH            = false
JOINT_INTER_INTRA_DIE_SEARCH              = false
MOE_PERSONALIZED_A2A_SWIZZLE              = false
GENERAL_IRREGULAR_MESH_SUPPORT            = false
```

不得将本版本宣称为任意 Mesh、任意 collective、完整 MoE Swizzle 或数值正确性实现。
