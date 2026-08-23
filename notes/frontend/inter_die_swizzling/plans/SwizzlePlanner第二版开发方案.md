# SwizzlePlanner 第二版开发方案

## 1. 目标与完成定义

第二版在第一版 `AG+GEMM`、`GEMM+RS`、`GEMM+AR` 的基础上，新增
MoE 个性化通信与专家计算融合：

```text
Dispatch + GroupGEMM
GroupGEMM + Combine
```

本方案的目标不是增加两个只能生成候选的 pattern，而是让它们从真实 MoE
负载成图开始，经过发现、规划、物化、负载 action replacement、ABI、lowering、
单一 manifest、ProgramIo、C++ finalizer 和 `npusim`，完成 naive / swizzle
两分支端到端运行。

第二版完成必须同时满足：

1. `Dispatch+GroupGEMM` 和 `GroupGEMM+Combine` 均有独立 typed semantic
   witness、candidate、cost 和 negative tests；
2. candidate 使用真实 token→expert trace、expert home、`PairRoute` 和物理资源，
   不把 personalized A2A 伪装成 `ALL_GATHER` 或普通 ring；
3. Dispatch 侧按 token/M 维分解，收到可计算的 tile 后启动专家 GEMM；
4. Combine 侧按输出 N 维分解，tile 完成后立即执行组合与回传；
5. planner 始终保留 unfused personalized-A2A baseline，没有收益时自动回退；
6. 选中的两个融合 region 能替换 4-Die MoE 原 action DAG 中对应动作，未选中的
   region 保持原实现，禁止重复执行或丢动作；
7. 至少 4-Die MoE inference 和 train-forward 各完成一次 `naive` / `swizzle`
   双分支、双次重复的真实 `npusim` 运行；
8. 同一 workload 的 token assignment、专家计算量、combined outputs、train-forward
   tape 与状态初始化在两分支间 exact 相同；
9. finalizer×2、ProgramIo actual-SHA、resolver、npusim×2、ACK/DONE/drain/
   `PROTO_WAIT` 门禁全部通过；
10. 第一版 Dense 三 pattern、4-Die MoE 原 runtime matrix 和所有 legacy producer
    无未解释的行为漂移；若修复既有同工作量错误，必须同时保存旧失败证据和新 exact golden。
11. 为 MoE 建立独立的 MATMUL/GroupGEMM setup、DTE launch/sync/hop、SRAM
    lifecycle、event/control 实测 marker；marker 不完整时不得将 provisional profile
    当作收益证据。
12. `Dispatch+GroupGEMM` 和 `GroupGEMM+Combine` 各至少在一个 production
    scale point 上由 `SWIZZLE_AUTO` 选中且实测快于同工作量 executable baseline。
13. 4-Die MoE inference 和 train-forward 均至少有两个相邻 scale point 满足
    `naive_makespan / auto_makespan >= 1.10`；forced 分支不计入。
14. runtime 必须实际观测 `max_inflight_send/recv >= 2`、compute/DTE overlap
    和方向端口并行；缺 marker 或只有 profile 声明时不得验收。

因此，第二版状态固定分层为：

```text
correctness_complete = 两个 pattern 已嵌入负载并通过全链闭包
performance_complete = AUTO 分支达到相邻 scale 的 10% 实测收益门槛
v2_complete          = correctness_complete && performance_complete
```

若只得到 no-benefit 报告，这是有价值的中间证据，但不能将 MoE V2 标记为完成。
这里的“保证收益”是验收保证：开发流程不会在缺少收益时结束，也不会提升收益 capability；
它不是对任意 trace、任意 shape 的先验性能承诺。若某组 production trace 无法盈利，必须
继续 W12R，或明确缩小已验证 workload/trace 能力边界。


第二版仍以 timing execution 为交付目标：

```text
timing_execution = true
functional_execution = false
```

除非另行新增数值执行 opcode 和输出比对，本版不得声明数值 functional correctness。

## 2. 资料、术语与算法命名

本方案依据：

- `refs/Inter-Die Swizzling优化.pdf`；
- `refs/inter_die_swizzle优化思路.md`；
- Zhang 等，*Comet: Fine-grained Computation-communication Overlapping for
  Mixture-of-Experts*；
- Wang 等，*Overlap Communication with Dependent Computation via Decomposition in
  Large Deep Learning Models*；
- Nam 等，*MeshSlice: Efficient 2D Tensor Parallelism for Distributed DNN
  Training*。

Comet 提供本版的 shared-tensor 分解和重排原则：

- `Dispatch+GroupGEMM` 的 shared tensor 沿 token/M 维分解；
- `GroupGEMM+Combine` 的 shared tensor 沿输出 N 维分解；
- 计算 tile 粒度不能因逐 token 通信而退化；
- Dispatch 侧先执行已就绪的本地/近端 token tile；
- Combine 侧按列重排 GroupGEMM，使最早完成的列可立即组合并发送；
- 通信与计算资源配额需要作为 profile/candidate 参数，而不是隐式常量。

晶圆 mesh 上的二阶段行列 personalized A2A 来自本项目的 mesh 编排设计，不应
表述成 Comet 原论文直接提出的算法。代码中统一使用：

```text
COMET_MESH_PERSONALIZED_A2A
DIRECT_XY_PERSONALIZED_A2A
```

前者表示“Comet shared-tensor 分解与 tile 重排 + mesh 二阶段行列 A2A”；后者是
保留原确定性 XY route 的直接 personalized A2A 流水 baseline/候选。不得使用
`WANG_1D_BIDIRECTIONAL` 或 `MESHSLICE_2D_OS` 命名本族算法。

## 3. 第一版基础与当前硬边界

### 3.1 可直接复用的第一版能力

第一版已经具备：

- `SwizzleProblem → Candidate → Cost → Decision` 的 stable typed carrier；
- economic decision 与显式 forced deployment 分离；
- action/resource DAG 的 earliest-start 代价模型；
- `SwizzleFusionPlan`、独立 timing IR2 projection、Core/Address ABI、OperandABI；
- 标准 `CommandFragment`、`LinkedProgramManifest`、ProgramIo 和 C++ finalizer；
- 独立 typed unfused comparison chain；
- finalizer×2、actual-SHA ProgramIo、npusim×2 的 comparison runner；
- `timing=true / functional=false` 的能力报告。

第二版必须扩展这些正式接口，不再建立一套旁路的 MoE planner/runtime。

### 3.2 可直接复用的 MoE typed 真相

现有 4-Die MoE 生产链已经提供：

- `LiteMoeDp4Spec.trace.assignments`：token→expert 静态路由；
- token source die、expert home die 和 remote-token 集合；
- `LiteMoeDp4P2PBinding` 与 exact `PairRoute`；
- `MOE_DISPATCH` / `MOE_COMBINE` 两类 flow；
- per-expert token count、GEMM FLOPs、combined outputs；
- inference、train-forward tape、backward overlay；
- 4 core、12 forward flows、384B logical P2P 和现有单 manifest runtime。

这些 carrier 是 V2 的唯一 workload truth。Swizzle problem builder 只能引用并验证它们，
不得按 token index 或 die index重新手造 route、字节数和 expert home。

### 3.3 必须改造的硬门禁

当前 V1 仍有以下精确限制：

- `FusionPattern` 只有 `AG_GEMM/GEMM_RS/GEMM_AR`；
- `discover_fusion._pattern()` 只识别二节点 Dense pattern；
- `SwizzleProblem` 假定恰好一个 GEMM 和一个非个性化 collective；
- `SwizzleCollectiveDescriptor` 用 participant/rank bytes 表达均匀 collective，
  无法表达每条 token flow；
- Wang/MeshSlice generator 和 cost 都按非个性化通信闭合；
- W7/W8 projection 的 output ownership 仅区分切片、partial 与 replicated；
- V1 standard linked carrier 表达一个独立融合算子，不会替换 MoE workload 中多个
  region；
- C++ finalizer 的 SWIZZLE quotient 只承认三种 Dense pattern；
- ProgramIo terminal mapper 不理解 combined token 与 train-forward tape；
- 现有 `LiteMoeDp4ExecutionCase` 没有 swizzle overlay/decision provenance。

这些边界必须逐层精确扩展。禁止仅在 enum 中增加 pattern 后绕过下游 validator。

### 3.4 从 Dense 收益验证迁移的强制经验

Dense S1/S2 official 虽然全链通过，AUTO 分支实测仍显著更慢：

| case | naive | auto | speedup |
|---|---:|---:|---:|
| S1 AG+GEMM | 1340 | 6769 | 0.198× |
| S2 AG+GEMM | 2140 | 13420 | 0.159× |
| S2 GEMM+RS | 2689 | 7523 | 0.357× |

这些结果必须直接改变 MoE V2 的开发方式：

1. **先锁同工作量，再谈重叠。** Dense 曾将每 rank terminal 当成完整 output，导致
   output bytes 放大 `R` 倍。MoE 必须从 assignment slice 重建 logical bytes、
   GroupGEMM FLOPs 和 combined/tape terminal。
2. **大 chunk 集不等于收益。** Dense AUTO 把 4 个 baseline MATMUL 拆成 32/64 个
   MATMUL primitive，虽然 eta 仍为 0.9，固定 setup 与 records 却压过了通信隐藏量。
   MoE 必须显式计入 `setup_cycles * tile_count`。
3. **lifecycle 必须按 slot 而非 chunk 增长。** 合法复杂度是
   `O(rank * operand * physical_slots)`，不是 `O(rank * chunk_count)`。
4. **inflight 必须是 runtime 事实。** `max_inflight_dte=2` 和
   `double_buffer=true` 只是能力上界；缺 raw marker 时 capability 必须为 false。
5. **endpoint session 是独立资源。** Dense 4-rank baseline 曾以
   `SEND,SEND,SEND,RECV...` 耗尽 session。MoE personalized A2A 必须按
   capacity-safe wave 排序，SEND 和 RECV 占用的总 session 不得超过上限。
6. **多输入归约必须二元化。** top-k/combine 和跨源聚合一律物化为
   loop-carried binary FP32 reduction chain，不生成 ISA 不支持的 3-read REDUCE。
7. **先有独立 marker，再有 measured profile。** 不允许看到最终 speedup 后反调
   cost 参数。
8. **AUTO 与 forced 永久分离。** forced 只证明可执行，MoE 收益门禁只读
   economic AUTO。
9. **dedicated admission 必须 producer/rank/pattern scoped。** terminal subview、
   lifetime reuse、binary-reduce span 都不能放宽 generic validator。
10. **报告 PASS 不等于收益 PASS。** CTest 可因证据链闭合而 PASS，但 no-benefit 时
    `performance_complete` 必须为 false。

## 4. 第二版能力边界

### 4.1 本版实现范围

- 固定 EP placement 后的 Dispatch/Combine swizzle；
- 2-Die EP2 与 4-Die EP4 schema/negative coverage；
- 4-Die EP4 inference 和 train-forward 的真实端到端运行；
- 现有静态、确定性 token trace；
- top-1 production runtime；
- schema 能表达 top-k contributors、gate weight 和 combine order；
- top-k>1 若输出已在 GEMM 侧应用 gate weight，可做 timing candidate/semantic test；
- 直接 XY personalized A2A 与完整矩形上的二阶段行列 A2A；
- token/M 分块、输出 N 分块、expert wave、unroll、double buffer；
- 不均匀 per-expert token count 与 capacity 上界；
- local/one-stage/two-stage arrival class 的 tile 重排；
- 多个 fusion region 在一个 MoE workload 中的无重叠 replacement；
- timing-only `npusim` comparison 与能力证据。

### 4.2 明确不在本版范围

- gate 网络或 token assignment 算法优化；
- 动态运行时重新选择 expert placement；
- adaptive routing、torus、故障 mesh；
- placement 与 swizzle 的联合搜索；
- token dropping、capacity overflow 的运行时恢复；
- 未预乘 gate weight 的 top-k 数值归约；
- MoE backward 的 DGRAD/WGRAD 与 Dispatch/Combine 反向融合；
- 新增 GroupGEMM、weighted-combine 或 personalized-A2A ISA opcode；
- functional correctness 声明；
- 将 Comet 的 GPU thread-block specialization 原样映射为 NPU core 实现。

动态 token 流量后续可通过 typed runtime binding 扩展，但 V2 production runtime 必须
锁定现有静态 trace；若 trace 缺失则 fail closed，不按均值猜测流量。

### 4.3 MoE 收益 scale matrix

现有 8-token trace 只作为 C0 correctness control，不能作为性能完成点。新增同一
production builder 派生的参数化 trace：

| scale | tokens | 每 expert token 目标 | trace | 目的 |
|---|---:|---:|---|---|
| C0 | 8 | 2 | balanced | 固定小负载回退控制 |
| C1 | 32 | 8 | balanced | 4-Die 首个流水点 |
| C2 | 64 | 16 | balanced | Dispatch/Combine crossover |
| C3 | 128 | 32 | balanced | 相邻收益稳定点 |
| C4 | 64/128 | typed skew | skewed | expert tail 与 capacity 压力 |

具体 H、expert intermediate 和 dtype 不在 runner 中手写，必须由
`LiteMoeDp4Spec`/model profile 派生。若 C3/C4 超过真实 HBM/SRAM：

- 先保持 expert weights 不变，仅扩 tokens；
- 再减少候选 wave/chunk，而不是提高硬件预算；
- 单个 typed root 超过 region 时 fail closed；
- 不允许只保留盈利点而删除容量失败点。

每个 scale 均固定生成以下 workload：

```text
inference:     dispatch -> gate/up -> swiglu -> down -> combine
train-forward: inference actions + exact tape outputs
```

收益搜索不得只增加通信或只增加计算。对相邻 scale，必须同时报告：

```text
assignment_count
logical_dispatch_bytes / logical_combine_bytes
physical_byte_hops
gate_up/down GroupGEMM FLOPs
GroupGEMM tile shape/count
record/control/lifecycle count
SRAM high-water
```

为了防止 Dense 的小 tile 重演，production profitable candidate 还必须满足：

```text
per_expert_tile_M >= measured_efficient_tile_floor
group_gemm_primitive_count / baseline_gemm_count <= measured_setup_budget
ALLOC/FREE roots = O(rank * operand * physical_slots)
```

## 5. 两个 pattern 的语义定义

### 5.1 `MOE_DISPATCH_GEMM`

region 的成员不是简单二节点，而是：

```text
Dispatch(token/expert flows)
  -> expert gate GEMM
  -> expert up GEMM
```

如果模型的专家第一层只有一个 GEMM，则 group 可只有一个 member；现有 Lite MoE 的
gate/up fork 必须作为同一个 `MoeExpertGemmGroupDescriptor` 表达。

合法性要求：

1. 每个 dispatch output 只服务其指定 expert 的 GEMM group；
2. token payload 的 shape、dtype、bytes 与 GEMM M 行精确相等；
3. local token 不生成 DTE，但作为 arrival class 0；
4. remote token 必须绑定现有 `MOE_DISPATCH` flow 和 exact route；
5. 同一 token/expert assignment 恰好出现一次，不得丢失或复制；
6. gate/up 两个 GEMM 对同一 token view 的读取必须一致；
7. boundary output 保持原 gate/up value identity 与 consumers；
8. shared tensor 只允许沿 M/token 维分解，K/N 分解必须拒绝。

### 5.2 `MOE_GEMM_COMBINE`

region 成员为：

```text
expert down GEMM
  -> optional gate-weight application
  -> Combine(token contributors)
```

合法性要求：

1. 每个 down GEMM output 绑定一个 `(token, expert, contributor_ordinal)`；
2. combine destination 是 token 原 source die；
3. remote flow 必须是对应 dispatch flow 的 typed reverse route；
4. top-k contributors 集合、顺序与 gate weight provenance 必须完整；
5. top-1 不生成数值 reduce；
6. top-k>1 只有在 `gate_weight_applied=True` 且 reduce op 为 FP32 SUM 时才可进入
   timing lowering，否则 fail closed；
7. shared tensor 只沿输出 N 维切分，M/token 维切分会破坏 top-k combine，应拒绝；
8. 最终 combined value ID、shape、layout、owner die 与原 workload exact 相同。

### 5.3 personalized A2A 不变量

两个 pattern 共用以下不变量：

```text
每个 payload 有唯一 source、destination、token、expert、byte slice；
每条 remote payload 有唯一 PairRoute；
每个中间 pivot 只转发，不取得 value ownership；
packet aggregation 不改变 payload identity；
unpack 后的目标地址与原 consumer view exact 相同；
Combine 是 Dispatch assignment 的逆向闭包，不是按 rank 均匀 collective。
```

## 6. 总体架构

```text
LiteMoeDp4Spec/Topology/IR0/N4/Execution
                 |
                 | discover_moe_swizzle_regions
                 v
MoeFusionRegion[] + typed token/expert/flow witness
                 |
                 | build_moe_swizzle_problem
                 v
MoeSwizzleProblem
                 |
                 | baseline + direct-XY + two-stage mesh generation
                 v
SwizzleCandidate[] + personalized traffic/resource cost
                 |
                 | decide_swizzle / deployment policy
                 v
MoeSwizzleDecision + SwizzleFusionPlan
                 |
                 | materialize_moe_swizzle_overlay
                 v
original 4-Die GlobalAction DAG --exact replacement--> optimized GlobalAction DAG
                 |
                 v
IR2 projection / CoreABI / OperandABI / standard fragments
                 |
                 v
single LinkedProgramManifest + ProgramIo + C++ finalizer + npusim
```

责任边界：

- MoE graph/trace 决定 token 和 expert 语义；
- discovery 只识别可融合 region，不决定算法；
- planner 在固定 placement/route 上决定 packetization、stage、tile order 和资源配额；
- overlay 只做 typed replacement，不重建整个 MoE DAG；
- lowering 消费完整 action/ABI，不从名字或 shape 猜 dispatch/combine；
- runtime runner 只消费正式 linked workload，不拼接独立算子 artifact。

## 7. 核心 typed schema

### 7.1 Pattern 与 Algorithm

在现有单一 `FusionPattern` 真相中增加：

```python
class FusionPattern(str, Enum):
    AG_GEMM = "ag_gemm"
    GEMM_RS = "gemm_rs"
    GEMM_AR = "gemm_ar"
    MOE_DISPATCH_GEMM = "moe_dispatch_gemm"
    MOE_GEMM_COMBINE = "moe_gemm_combine"
```

在 `SwizzleAlgorithm` 增加：

```python
DIRECT_XY_PERSONALIZED_A2A = "direct_xy_personalized_a2a"
COMET_MESH_PERSONALIZED_A2A = "comet_mesh_personalized_a2a"
```

`UNFUSED` 继续始终存在。Dense algorithm 对 MoE pattern 必须拒绝，MoE algorithm 对
Dense pattern 也必须拒绝。

### 7.2 MoE descriptors

新增或以 pattern-specific union 挂入 `SwizzleProblem`：

```python
@dataclass(frozen=True, slots=True)
class MoeTokenAssignmentView:
    token_index: int
    source_rank: int
    expert_index: int
    expert_rank: int
    contributor_ordinal: int
    gate_weight_ref: str | None
    dispatch_flow_ref: str | None
    combine_flow_ref: str | None
    payload_value_ref: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class MoeExpertGemmView:
    expert_index: int
    rank: int
    member_refs: tuple[str, ...]
    m_tokens: int
    n: int
    k: int
    dtype: DType
    accumulation_dtype: DType
    flops: int


@dataclass(frozen=True, slots=True)
class MoePersonalizedTrafficView:
    assignments: tuple[MoeTokenAssignmentView, ...]
    expert_gemms: tuple[MoeExpertGemmView, ...]
    pair_routes: tuple[SwizzleRouteView, ...]
    top_k: int
    capacity_tokens_per_expert: int
    trace_digest: str
```

必须同时保留实际 trace count 和 capacity upper bound。cost 可以用 actual/p95/capacity
三点估计，但 executable candidate 只能绑定 actual trace 或显式 runtime binding。

### 7.3 Packet、stage 和 tile witness

```python
@dataclass(frozen=True, slots=True)
class MoePacketSlice:
    assignment_ref: str
    source_offset_bytes: int
    destination_offset_bytes: int
    bytes: int


@dataclass(frozen=True, slots=True)
class MoePacketWitness:
    packet_id: str
    stage: int
    source_rank: int
    destination_rank: int
    pivot_rank: int | None
    route_ref: str
    slices: tuple[MoePacketSlice, ...]
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class MoeTileWitness:
    expert_index: int
    tile_index: int
    split_axis: SwizzleTensorAxisRole
    assignment_refs: tuple[str, ...]
    arrival_class: int
    required_packet_refs: tuple[str, ...]
    output_value_refs: tuple[str, ...]
```

packet aggregation 必须能逐 slice 重建原 assignment；`sum(slice.bytes)` 必须等于
packet bytes，全部 assignment slice 必须一次且仅一次覆盖。

### 7.4 Candidate 增量字段

MoE candidate 至少携带：

```text
packetization
tile_schedule
expert_wave_count
token_block_size
output_column_block_size
unroll_degree
double_buffer
compute_core_fraction
communication_core_fraction
traffic_scenarios(actual/p95/capacity)
```

资源配额在本仿真器中先映射为 core/action scheduling witness，不声称等同 Comet 的
GPU thread-block specialization。

### 7.5 Workload overlay

新增 `MoeSwizzleOverlay`：

```python
@dataclass(frozen=True, slots=True)
class MoeSwizzleOverlay:
    source_execution_id: str
    decisions: tuple[SwizzleDecision, ...]
    deployment_selections: tuple[SwizzleDeploymentSelection, ...]
    replaced_action_refs: tuple[str, ...]
    replacement_rank_programs: tuple[SwizzleRankProgram, ...]
    preserved_action_refs: tuple[str, ...]
    boundary_value_bindings: tuple[MoeBoundaryValueBinding, ...]
```

validator 必须证明：

- 原 action 集合被 `replaced ∪ preserved` 无交覆盖；
- 每个被选择 region 的 member action 恰好被替换一次；
- 两个 region 不重叠；
- replacement 的外部 reads/writes 与原 region boundary exact；
- token/expert/combined/tape terminal IDs 不变；
- 未选择或 economic fallback 的 region 原样保留。

## 8. 融合 region 自动发现

### 8.1 不再使用二节点 `_pattern()`

MoE region 是多节点、多 flow region，必须新增独立 discovery：

```python
discover_moe_swizzle_regions(
    graph: LiteMoeDp4N4IR1,
    execution: LiteMoeDp4ExecutionCase,
) -> tuple[MoeFusionRegion, ...]
```

它从 typed transfer binding 出发：

- `MOE_DISPATCH`：沿 output consumers 收集同 expert 的 gate/up GEMM group；
- `MOE_COMBINE`：沿 input producer 反查同 token/expert 的 down GEMM；
- local assignment 由 trace 补入 region，但不伪造 P2P flow；
- region member、boundary input/output 和 flow refs 由图边导出；
- stable ID 包含 source graph/execution/trace digest。

### 8.2 发现门禁

以下任一情况必须拒绝候选：

- transfer role 或 direction 颠倒；
- expert home 与 action placement 不一致；
- route endpoint 与 token source/expert home 不一致；
- dispatch value 还有 region 外非允许 consumer；
- down GEMM output 被 combine 之外的有副作用节点消费；
- gate/up fork 缺一条或跨 expert；
- combine contributor 缺失、重复、顺序不闭合；
- dtype/layout/bytes 不相等；
- dynamic trace 未提供 concrete binding；
- 两个 candidate region 共享 member action。

## 9. 候选生成

### 9.1 始终生成 unfused baseline

unfused baseline 必须逐动作表达现有负载的：

```text
Dispatch: SEND -> RECV -> WAIT -> expert GEMM
Combine: expert GEMM -> SEND -> RECV -> WAIT -> optional SUM
```

baseline 的 flow、bytes、route、GEMM FLOPs、terminal 与原 MoE execution exact，不能
使用空 `rank_programs` 的 analytic baseline 直接充当 executable baseline。复用第一版
`UnfusedComparisonPlan/Projection` 的原则，但建立 MoE-specific typed carrier。

baseline 本身也必须满足 endpoint 容量，而不是只保证拓扑合法：

- SEND 与 RECV 共同占用 endpoint session；
- 每 rank 的 outstanding TX+RX 不得超过 typed capacity；
- 4-rank personalized exchange 使用 deterministic round-robin peer waves；
- 下一 wave 的 SEND/RECV 依赖上一 wave 的 WAIT/ACK retirement；
- per-peer FSM 不因 lane/token 复用而混淆；
- capacity-safe 排序不得改变 flow、route、bytes 或 terminal。

baseline validator 必须从 typed assignment 重算：

```text
sum(expert GEMM action FLOPs) == workload expert GEMM FLOPs
sum(rank-local combined terminal bytes) == logical combined bytes
sum(tape terminal bytes) == source train-forward tape bytes
```

禁止保留 full-output-per-rank 或 full-expert-per-rank 的放大基线。

### 9.2 Direct XY personalized pipeline

`DIRECT_XY_PERSONALIZED_A2A` 保持每条原 `PairRoute`，只改变 packetization 和 tile
调度：

- Dispatch 将同 expert、同 destination、可连续落址的 token 聚包；
- local token tile 在远端传输期间先执行；
- remote tile 按 `(arrival_stage, hop_count, source_rank, token_index)` 排序；
- Combine 按 N-block 产生，完成一个 block 即发送；
- 不允许 packet 跨 expert 或跨不连续 destination slice；
- 若聚包导致额外 copy，必须显式 `LOCAL_COPY` 并计入成本。

它既是完整矩形失败时的 fallback，也是判断二阶段转向聚合是否有收益的对照候选。

### 9.3 Comet mesh personalized A2A

适用条件：

- participant placement 构成完整 `Pr × Pc` 矩形；
- `Pr>1` 且 `Pc>1`；
- 每个 source/destination 均能映射到唯一行列坐标；
- 两阶段 route 能由现有物理 `PairRoute` 分段 exact 重建；
- pivot SRAM、DTE descriptors、inflight packets 和 event symbols 足够；
- packet 聚合后仍能逐 slice 重建 assignment。

两阶段定义固定为：

```text
stage 0: source -> (source_row, destination_col) pivot
stage 1: pivot  -> destination
```

若 source 与 pivot 或 pivot 与 destination 相同，对应 stage 退化为 local ready/copy，
不得生成零距离 SEND。

Dispatch 流水：

```text
stage0 receive/aggregate
  -> stage1 receive/unpack
  -> tile-ready event
  -> gate/up GroupGEMM tile
```

tile 顺序：

```text
local -> one-stage -> two-stage，
同一层内按 hop_count、source_rank、expert_index、tile_index 排序。
```

Combine 流水：

```text
down GroupGEMM N-block
  -> optional gate weight / contributor SUM
  -> stage0 send to pivot
  -> stage1 send to token owner
  -> combined boundary write
```

GroupGEMM 必须按 N-block 跨 expert 轮转，而不是完整算完一个 expert 再进入下一个；
只有这样最早列才能提前回传。

### 9.4 枚举维度

每个合法 algorithm 枚举：

- `token_block_size`：高效 GEMM tile 的整除因子；
- `output_column_block_size`：N 的高效 tile 整除因子；
- `expert_wave_count`：不超过专家数和 SRAM 上限；
- `unroll_degree ∈ {1,2}`；
- `double_buffer ∈ {false,true}`，与 unroll/SRAM closure 联动；
- compute/communication core fraction 的有限 profile 表；
- direct packet aggregation 粒度；
- two-stage pivot aggregation 粒度。

枚举后先执行收益前置剪枝：

- token/M 或 output-N tile 位于 measured efficiency table；
- GroupGEMM primitive 数没有超过 measured setup budget；
- packet bytes 不低于 DTE 有效载荷下限；
- unroll2 的两个 physical slot 都有地址与 lifetime witness；
- 同 slot 重写依赖最后 reader，不同 slot 不添加假依赖；
- endpoint session 的 earliest-start schedule 不超容量；
- root/ALLOC/FREE 只随 physical slot 数变化；
- barrier/event/token 数不随 token block 数线性增长。

候选总数仍受 `max_candidates` 限制，canonical sort/dedup 后截断；禁止依赖 Python
dict 插入顺序。至少保留一个低 chunk 候选，避免 cost 因只看深流水候选而被迫选择
大量小 GroupGEMM。

## 10. personalized traffic 代价模型

### 10.1 baseline 与候选必须使用同一工作量

每个 candidate 的以下总量必须与 problem exact：

```text
assignment count
logical payload bytes
expert GEMM FLOPs
combine contributor count
terminal output bytes
```

两阶段 A2A 可以改变 hop/packet/control 数，但不得改变 logical bytes。物理 transported
bytes 按每个 stage/hop 单独计数。

### 10.2 资源 DAG

沿用 V1 earliest-start evaluator，但新增资源：

- per-direction D2D link/port；
- source/pivot/destination DTE；
- pivot SRAM write/read lifetime；
- expert compute core pool；
- communication/control core pool；
- pack/unpack local-copy engine；
- combine FP32 reduction resource；
- event/barrier control slots。

每个 SEND/RECV/WAIT、COMP、REDUCE、LOCAL_COPY 都来自 candidate action witness，
不得只用 aggregate 公式估时。

### 10.3 不均衡与到达偏斜

对实际 trace 计算：

```text
per-expert token histogram
per-rank ingress/egress bytes
per-link bytes
per-pivot bytes
earliest/latest tile ready time
expert compute tail
```

cost 至少输出：

```text
prologue_cycles
steady_state_cycles
epilogue_cycles
critical_path_cycles
max_link_utilization
max_dte_utilization
compute_idle_cycles
communication_idle_cycles
sram_high_water_bytes
descriptor_count
event_count
```

### 10.4 actual/p95/capacity 三场景

候选分别在以下流量场景评估：

1. `actual`：当前静态 trace；
2. `p95`：由显式 routing profile 给出，缺 profile 时不生成；
3. `capacity`：每 expert 的容量上界。

决策以 actual 为主；如果 p95/capacity 存在，则要求候选在这些场景不违反 SRAM、
descriptor 和 inflight 上限。不能用均匀 token 分布替代缺失 profile。

### 10.5 决策规则

沿用第一版：

1. 先按置信区间是否相交比较 estimated cycles；
2. 相交时比较瓶颈 link utilization；
3. 再比较 control/descriptor count；
4. 再比较 SRAM high-water；
5. 最后按 stable candidate ID。

若最佳 fused candidate 不快于 executable unfused baseline，economic decision 必须选择
`UNFUSED`。为了覆盖 optimized runtime，可使用 typed `FORCED_BY_POLICY` deployment，
但报告必须同时保留原 economic decision，禁止把 forced 说成有收益。

### 10.6 独立标定与固定开销

在运行任何 MoE scale comparison 前，matching binary 必须输出 dedicated markers：

```text
[SWIZZLE_CALIBRATION] MATMUL/GROUP_GEMM shape,dtype,repeat,cycles
[SWIZZLE_CALIBRATION] DTE_LAUNCH / DTE_SYNC / HOP
[SWIZZLE_CALIBRATION] SRAM_ALLOC / BIND / FREE
[SWIZZLE_CALIBRATION] EVENT_SET / EVENT_WAIT / TERMINAL_DONE
```

MEASURED profile 的最低覆盖：

- 至少 3 个实际候选会经过的 GroupGEMM tile shape；
- 每个 fixed-cost kind 至少 3 个奇数样本；
- 每个样本两次 repeat exact；
- binary/tool/config SHA 完整；
- calibration scales 与 validation scales 分离。

缺 marker 时只允许生成 `PROVISIONAL` profile，且
`supports_calibrated_tile_efficiency`、`supports_calibrated_fixed_overhead` 和
`performance_complete` 均保持 false。

cost 必须显式包含：

```text
group_gemm_setup_cycles * group_gemm_primitive_count
dte_launch_cycles       * send_recv_operation_count
sram_lifecycle_cycles   * physical_root_count
event_control_cycles    * event_record_count
reduce_setup_cycles     * binary_reduce_count
```

### 10.7 预测质量与收益保障

planner 必须分解输出：

- hidden communication cycles；
- uncovered prologue/epilogue；
- GroupGEMM setup/control penalty；
- pack/pivot copy penalty；
- tile efficiency penalty；
- endpoint serialization penalty；
- predicted top-2 candidates。

在 calibration set 上冻结 profile 后，C2/C3 作为 validation set。若 actual 最优不在
predicted top-2，或者 profitable/unprofitable 分类错误，则 W7 不通过，不能继续用最终
speedup 反调本 profile；必须新增独立 microbench 或修正资源 DAG。

只有 `SWIZZLE_AUTO` 可进入收益统计。forced candidate 即使实测更快，也只能提示
cost model 失配，不能直接提升 capability。

## 11. 物化与负载嵌入

### 11.1 Action 物化

复用现有 action kind：

```text
COMP / SEND / RECV / WAIT / REDUCE / LOCAL_COPY / BARRIER
```

不新增 opcode。MoE action 需补足：

- assignment refs；
- expert/tile/N-block；
- packet/slice/stage/pivot；
- source/destination/pivot rank；
- route segment；
- operand dtype/shape/layout/extent；
- buffer slot/lifetime；
- token/event FSM；
- original action refs 与 replacement provenance。

### 11.2 Exact replacement

`materialize_moe_swizzle_overlay` 的顺序固定为：

1. 验证 source execution 与所有 decisions；
2. 按 region ID canonical 排序；
3. 检查 member action 集不相交；
4. 将 selected/forced fused region 物化为 replacement rank programs；
5. 将 UNFUSED region 保持原 action 或物化 typed executable baseline；
6. 重连外部 dependencies 和 boundary values；
7. 重算 core-order、flow/event/channel IDs；
8. 对整张 action DAG 做拓扑、ownership 和 terminal closure；
9. 生成单个 `LiteMoeDp4SwizzledExecution`。

不得把独立 Swizzle fragment 附加到旧 MoE manifest 末尾；旧 region actions 必须被删去并
由 replacement 取代，否则会双算/双发。

### 11.3 IR2、ABI 与 lowering

优先复用 V1 standard pipeline：

- projection 增加 MoE assignment/packet/stage/expert tile provenance；
- CoreABI 使用现有 4 个 logical core 与真实 SRAM region；
- OperandABI 为 pack/pivot/tile/combined/tape 建立 exact view；
- personalized REDUCE 仅用于合法 top-k contributor SUM；
- DTE FSM 按 packet stage 生成，stage0/stage1 token 不得混用；
- SEND 无接收 token relocation，RECV/WAIT 同 token；
- pivot buffer 必须有 ALLOC/use/FREE 生命周期；
- terminal combined/tape OWNED buffer 不得提前 FREE。

每个 replacement region 可以先生成 typed fragments，但 workload linker 最终必须输出
一个 `LinkedProgramManifest`，其输入 digest 同时包含：

```text
source MoE graph/execution
trace/topology/oracle
all decisions/deployment selections
overlay/projection/ABI/lowered fragments
ProgramIo source provenance
```

### 11.4 ProgramIo

ProgramIo 从整张 swizzled workload 重建，而不是把多个 operator sidecar 拼接：

- 所有 BORROWED token/weight/state 首次 READ 有初始化；
- pivot、packet staging、GEMM output、combined/tape 的 ownership 与 first use 一致；
- inference probes 仍是原 combined outputs；
- train-forward probes 为 combined outputs + tape；
- swizzle 与 naive 使用相同逻辑 terminal IDs 和 payload bytes；
- actual artifact SHA 进入 contract；
- alias/subview 只接受 dedicated producer、root layout 和 contained span 的 exact
  allowlist；
- 裸 `LinkedProgramManifest` 不能绕过 typed workload wrapper。

### 11.5 C++ finalizer

新增 MoE Swizzle dedicated trust branch，但不新增 ISA：

- exact top schema、producer 和 input digest kinds；
- 4 core 与 source workload quotient；
- pattern/algorithm/trace/overlay provenance；
- packet slice、route segment、stage token/event closure；
- COMP/transfer/copy/reduce/lifecycle opcode quotient；
- BufferABI、AddressOperandBinding、runtime/program symbol closure；
- terminal combined/tape labels；
- 非 dedicated producer 夹带 MoE Swizzle fragment 必须拒绝。

C++ 不重建 Python stable digest 的语义细节时，只能锁 typed schema、source ref、prefix
和可从 manifest 直接重建的事实；完整 semantic closure 继续由 typed wrapper exact rebuild
保证。

## 12. 负载端到端运行矩阵

### 12.1 最小 production workload

使用现有 4-Die EP4 Lite MoE：

```text
2×2 physical die mesh
4 experts / one expert per die
8 tokens / balanced static top-1 trace
12 dispatch/combine remote flows
4 logical cores
```

至少运行：

| workload | fusion regions | terminals |
|---|---|---|
| inference forward | Dispatch+gate/up GroupGEMM；down GEMM+Combine | 8 combined outputs |
| train forward | 同上 | 8 combined outputs + 8 tape outputs |

训练反向继续走现有 backward path，只做 no-drift 回归，不纳入 V2 优化声明。

### 12.2 分支矩阵

每个 workload 运行：

```text
NAIVE
SWIZZLE_AUTO
SWIZZLE_FORCED_COVERAGE
```

其中：

- `NAIVE` 是原 MoE executable chain；
- `SWIZZLE_AUTO` 尊重 economic decision，可能与 NAIVE 相同；
- `SWIZZLE_FORCED_COVERAGE` 只用于证明优化 branch 可执行，并显式报告 force reason。

正式性能表只比较相同 trace、相同 workload 和相同 timing profile。不得把 standalone
operator microbenchmark 与完整 MoE workload 混比。

### 12.3 Runtime 观测

runner 独立从 typed workload/manifest 推导期望值：

- per-pattern region count；
- assignment、packet、stage 和 flow count；
- logical/physical bytes 与 packets；
- per-link XY traffic；
- expert GEMM/tile 数；
- combined/tape probes；
- observed max inflight SEND/RECV session；
- compute、DTE、pack/copy 的 start/end marker；
- compute/DTE overlap cycles；
- row/column/direction port utilization over time；
- prologue/steady/epilogue cycles；
- MATMUL、DTE、lifecycle、event/control record breakdown；
- physical root count 与 SRAM high-water；
- core ACK/DONE；
- memory initialization/probe bytes；
- drain、residual、repeat marker。

不得接受仿真器自报 aggregate 作为唯一真相。若缺少 inflight/overlap/port-time marker，
报告必须写 `null + missing_measurements`，不能填 0、profile 上限或 planner 预测值。

### 12.4 收益验收协议

每个 scale 的 typed pair evidence 必须证明：

```text
same trace digest
same assignment/contributor set
same logical dispatch/combine bytes
same expert GroupGEMM FLOPs
same combined/tape terminal bytes
same hardware/mapping/simulation digest
same functional/timing mode
```

正式 speedup 固定为：

```text
speedup = NAIVE.actual_makespan / SWIZZLE_AUTO.actual_makespan
```

V2 performance complete 的硬条件：

1. Dispatch+GroupGEMM 与 GroupGEMM+Combine 各有一个 AUTO region 级收益点；
2. inference 和 train-forward 各有两个相邻 scale 的 workload speedup >= 1.10；
3. 两次 makespan 与 marker digest exact；
4. observed max inflight send/recv >= 2；
5. compute/DTE overlap cycles > 0；
6. physical lifecycle root 数满足 slot 复杂度；
7. `PROTO_WAIT` 缺失且全部 residual 为 0；
8. forced 分支完全排除在收益聚合之外。

若任一条件失败：

- evidence CTest 可以因“证据链完整且结论诚实”而 PASS；
- `performance_benefit=false`、`performance_complete=false`；
- MoE V2 继续进入瓶颈修复工作包，不得结束开发；
- 禁止降低 1.10 阈值、提高硬件带宽/容量或删除失败 scale。

## 13. 完整开发顺序

以下顺序按依赖执行。每个工作包首个真实门禁未通过前，不并行扩下游 schema。

### W0：冻结 V1 与 MoE 基线

1. 保存 Dense 三 pattern official report 和 exact CTest；
2. 保存 4-Die MoE inference/train-forward/backward runtime matrix；
3. 锁定现有 trace、flows、records、ProgramIo 和 terminal counts；
4. 新增 capability matrix，V2 初始全部为 false。

完成门禁：所有 baseline 通过，且没有修改任何 V2 schema。

### W0P：先补齐独立性能观测

1. 增加 GroupGEMM 多 shape exact marker；
2. 增加 DTE launch/sync/hop marker；
3. 增加 SRAM lifecycle 与 event/control marker；
4. 增加 endpoint session open/retire 与 peak marker；
5. 增加 per-direction port-time 和 compute/DTE overlap marker；
6. strict parser、repeat、tool SHA 和 missing-marker negatives。

完成门禁：能够生成 MEASURED profile，且所有 runtime 收益字段可从 raw marker 重建。
若此门禁未过，后续可继续 correctness，但不得进入 performance completion。

### W0S：参数化 MoE scale 与同工作量 oracle

1. 从 production builder 生成 C0–C4 trace；
2. balanced/skewed assignment 都有 typed digest；
3. 重建 per-expert FLOPs、dispatch/combine bytes、terminal/tape bytes；
4. 固定 calibration 与 validation scale；
5. HBM/SRAM 单root与live-set容量负测。

完成门禁：C0 固定回退；C1–C3 可执行；C4 成功或以真实容量原因 fail closed。

### W1：Pattern 与 semantic schema

1. 扩 `FusionPattern` 和 `SwizzleAlgorithm`；
2. 新增 assignment/expert/traffic/packet/tile witness；
3. 新增 Dispatch/Combine semantic validator；
4. strict serde、old-version、stable-ID 与 wrong-pattern negatives；
5. Dense 三 pattern schema 测试无漂移。

完成门禁：纯 schema/semantics 全绿，不生成 candidate。

### W2：MoE region discovery

1. 从 2-Die 与 4-Die typed graph/execution 构建 region；
2. 处理 gate/up 多 consumer group；
3. 处理 down GEMM→combine contributor；
4. local/remote assignment 合并；
5. region overlap 和 boundary closure；
6. negative：direction、expert home、route、duplicate/missing contributor。

完成门禁：4-Die 固定图发现两个 pattern family，重建稳定。

### W3：Problem 与 topology adapter

1. 从 region、trace、oracle 和 physical group 建 `MoeSwizzleProblem`；
2. 保留完整 route/resource incidence；
3. 构建完整矩形、row/column/pivot view；
4. 生成 actual/capacity traffic scenario；
5. rectangle-only 不能冒充 two-stage executable witness。

完成门禁：2×2 positive，1×4 只允许 direct，缺 route/profile fail closed。

### W4：Executable unfused baseline

1. 将原 MoE action region映射为 typed baseline rank programs；
2. exact 保留 flow、route、bytes、GEMM、terminal；
3. 重建 rank-local output，不允许 full-output×rank；
4. 生成 capacity-safe peer waves；
5. 生成 baseline cost；
6. 与原 execution action slice exact 对照。

完成门禁：baseline 可独立 project/lower，同工作量与 endpoint session 均 exact，但尚不
替换 workload。

### W5：Direct XY candidate

1. 实现合法聚包；
2. Dispatch arrival-class tile schedule；
3. Combine N-block schedule；
4. pack/unpack copy 显式化；
5. SRAM/DTE/min-payload/action-count 剪枝。

完成门禁：2-Die/4-Die 两 pattern candidate/action DAG 全绿。

### W6：Comet mesh two-stage candidate

1. 完整矩形和 pivot 构造；
2. stage0/stage1 packet/slice witness；
3. local/one-stage/two-stage退化；
4. token/M 与 output-N tile schedule；
5. expert wave、unroll、double-buffer；
6. route segment、event、buffer lifetime closure。

完成门禁：2×2 positive；route/pivot/slice/stage tamper negatives 全绿。

### W7：Cost 与 decision

1. 扩 resource-DAG evaluator；
2. per-expert/per-link/pivot load；
3. actual/p95/capacity scenarios；
4. 消费 W0P 的 measured tile/setup/fixed profile；
5. GroupGEMM primitive/setup、endpoint session、lifecycle 进入 cost；
6. unfused/direct/two-stage统一排序；
7. fallback 与 forced deployment 分离；
8. calibration/validation split 与 predicted top-2 回归。

完成门禁：输入顺序不影响决策，所有 work totals exact，区间/tie-break稳定；C1 calibration
冻结后，C2/C3 profitable 分类与 actual 一致。缺 measured profile 时本工作包不得完成。

### W8：FusionPlan 与 workload overlay

1. materialize MoE rank programs；
2. 建 `MoeSwizzleOverlay`；
3. exact replacement Dispatch/GEMM/Combine actions；
4. preserved actions、dependencies、boundary values；
5. inference/train-forward 两整图 validator；
6. double-execution、missing action、cross-region overlap negatives。

完成门禁：整图 action DAG 通过，source terminals/tape 不变。

### W9：Projection、ABI 与 standard lowering

1. MoE projection facts；
2. 4-core schedule 与 SRAM allocator；
3. packet/pivot/tile OperandABI；
4. physical slot interval coloring 与 terminal rank-local root；
5. lifecycle root 数的 chunk-invariance；
6. endpoint session/token generation closure；
7. binary combine/reduce OperandABI；
8. fragment records、runtime/event FSM、address bindings；
9. 单 manifest linker；
10. exact rebuild 和 restable tamper。

完成门禁：Python `LinkedProgramManifest.validate()` 正向与负测全绿；同 unroll 下
token block 数增加时 root/ALLOC/FREE 不增加。

### W10：ProgramIo 与 C++ finalizer

1. typed workload wrapper进入 closed union；
2. first-use ownership、alias、terminal/probe closure；
3. actual-SHA ProgramIo；
4. C++ dedicated trust branch；
5. real stdin positive×2 和 schema/producer/route/FSM/terminal negatives；
6. legacy finalizer selftest。

完成门禁：matching build 的 finalizer/resolver 全绿。

### W11：Standalone operator preflight

1. Dispatch+GroupGEMM naive/swizzle；
2. GroupGEMM+Combine naive/swizzle；
3. finalizer×2、resolver、npusim×2；
4. region-level AUTO 与 forced 分离；
5. raw evidence 和 typed report。

完成门禁：用于定位 operator 链问题，但不作为负载集成完成证据；两个 pattern 各至少有
一个 AUTO region-level speedup >= 1.10，否则返回 W5/W6/W7。

### W12：4-Die MoE workload runtime

1. inference 三分支；
2. train-forward 三分支；
3. C0–C3 scale sweep；
4. 相同 trace/ProgramIo/terminal 对照；
5. inflight/overlap/port-time marker；
6. ACK/DONE/drain/PROTO_WAIT/repeat；
7. economic 与 forced reason 报告；
8. 生成 workload comparison report。

完成门禁：两个 workload 全矩阵真实 PASS，且 inference/train-forward 各有两个相邻
scale 的 AUTO speedup >= 1.10。

### W12R：未获益时的强制修复循环

若 W11/W12 未达到收益门槛，按 typed bottleneck 分类回退：

1. GroupGEMM setup 主导：减少 tile/chunk，批量化相邻 expert tile；
2. lifecycle/control 主导：固定双槽 root，压缩每 chunk event/barrier；
3. endpoint serialization：调整 capacity-safe wave 和 packet aggregation；
4. insufficient overlap：删除假依赖，验证不同 slot 与多方向端口并行；
5. tile efficiency loss：提高 token block/N-block floor；
6. pivot/copy 主导：回退 direct XY 或实现 typed zero-copy contained view；
7. expert skew tail：调整 expert wave，不用均匀 trace 掩盖尾部。

每轮必须增加独立 marker 或修正 action/resource DAG。禁止修改收益阈值、硬件容量或
validation set。直到 W12 门禁满足，V2 保持 performance incomplete。

### W13：注册、回归与文档

1. 唯一 `moe_swizzle_runtime_comparison` CTest；
2. `RUN_SERIAL=TRUE`、明确 timeout/PASS/FAIL regex；
3. Dense Swizzle official CTest；
4. 4-Die MoE 原 runtime matrix；
5. Python unit/integration、C++ aggregate、diff-check；
6. public exports、capability report、接口笔记更新；
7. CTest PASS 与 performance PASS 分字段保存；
8. no-benefit report 不提升 V2 completion。

完成门禁：CTest 唯一发现且 official 运行通过；`v2_complete` 仅在 W12 性能门禁也满足
时为 true。

## 14. 建议文件布局

优先新增 pattern-specific 模块，避免继续膨胀 V1 通用文件：

```text
llm/frontend/wafer_frontend/schema/
  swizzle_moe.py
  swizzle_moe_plan.py
  swizzle_moe_ir2.py
  swizzle_moe_lowering.py
  swizzle_moe_standard.py
  swizzle_moe_scale.py
  swizzle_moe_calibration.py
  swizzle_moe_evidence.py

llm/frontend/wafer_frontend/policies/swizzle/
  moe_semantics.py
  moe_problem.py
  moe_direct_xy.py
  moe_comet_mesh.py
  moe_cost.py
  moe_materialize.py
  moe_calibration.py

llm/frontend/wafer_frontend/passes/
  discover_moe_swizzle.py
  build_moe_swizzle_overlay.py
  project_moe_swizzle_ir2.py
  moe_swizzle_program_io.py

llm/frontend/wafer_frontend/lowering/
  moe_swizzle_abi.py
  moe_swizzle_standard.py
  moe_swizzle_linker.py

llm/test/frontend/unit/
  test_swizzle_moe_schema.py
  test_swizzle_moe_discovery.py
  test_swizzle_moe_candidates.py
  test_swizzle_moe_cost.py
  test_swizzle_moe_overlay.py
  test_swizzle_moe_calibration.py
  test_swizzle_moe_lifecycle.py

llm/test/frontend/integration/
  moe_swizzle_cases.py
  test_moe_swizzle_lowering.py
  test_moe_swizzle_program_io.py
  run_moe_swizzle_runtime.py
  test_run_moe_swizzle_runtime.py
```
  moe_swizzle_scale_cases.py
  run_moe_swizzle_benefit_runtime.py
  test_run_moe_swizzle_benefit_runtime.py

需要窄改的 shared 文件预计包括：

```text
schema/ir0.py
schema/swizzle.py
schema/swizzle_plan.py
schema/swizzle_ir2.py
schema/artifact_manifest.py
schema/program_io.py
passes/program_io.py
policies/swizzle/enumerate.py
policies/swizzle/decide.py
program_finalizer.h/.cpp/selftest
CMakeLists.txt
各 package __init__.py
```

shared 修改必须 pattern/producer scoped，不得把 legacy validator 改成宽松 duck typing。

## 15. 测试与验收矩阵

### 15.1 Schema/semantic negatives

- old schema version；
- stable ID 与 restable tamper；
- token/expert duplicate、missing、wrong home；
- dispatch/combine direction 交换；
- route endpoint/path/pivot 篡改；
- gate/up 跨 expert；
- output N/M split 角色错误；
- top-k contributor/order/weight 缺失；
- dynamic trace 无 binding；
- wrong dtype/layout/payload bytes。

### 15.2 Candidate negatives

- placement 非完整矩形却生成 two-stage；
- stage route 不能拼回原 PairRoute；
- packet slice gap/overlap/duplicate；
- 跨 expert 非连续聚包；
- tile 未等待全部 required packets；
- Combine 在 N-block 完成前发送；
- SRAM high-water 超限；
- DTE/inflight/descriptor 超限；
- double-buffer slot alias；
- candidate action FLOPs/bytes 不守恒；
- rank-local terminal 被错误放大为 full-output×rank；
- GroupGEMM primitive 数超过 measured setup budget；
- chunk 增加导致 root/ALLOC/FREE 线性增长；
- unroll2 只有一个物理 slot 或实际 peak inflight 为 1；
- slot复用早于最后reader；
- SEND/RECV endpoint session 总数超容量；
- top-k 生成非二元 REDUCE；
- 缺 measured tile仍被标为profitable。

### 15.3 Overlay negatives

- 原 action 未被 replaced/preserved 覆盖；
- action 双重替换；
- 两个 region member 重叠；
- replacement boundary value 漂移；
- combined/tape terminal 漂移；
- preserved backward/state action 被误删；
- economic UNFUSED 却偷偷部署 fused candidate；
- forced结果进入performance聚合；
- AUTO decision与实际部署candidate不一致。

### 15.4 Manifest/ProgramIo/finalizer negatives

- wrong top schema/producer/input kind；
- 非 dedicated producer 夹带 MoE Swizzle fragment；
- packet stage token/FSM/peer mismatch；
- pivot buffer lifecycle 与 alias 错误；
- AddressOperandBinding slice gap/overlap；
- terminal label 提前 FREE；
- bare LPM 绕过 typed wrapper；
- ProgramIo source SHA 不等 actual artifact；
- missing combined/tape probe；
- restable route/terminal/ownership tamper；
- non-dedicated producer使用terminal/lifetime alias admission；
- combined rank slice gap/duplicate/full-output×rank；
- ProgramIo after-program probe覆盖仍存活的alias；
- raw 缺 inflight/overlap marker 却填 0 或 profile 上限；
- CTest PASS却将performance_complete置true；
- no-benefit report提升speedup capability。

### 15.5 Runtime acceptance

每个 branch 必须：

```text
finalizer run 0 == run 1 bytes/report/SHA
resolver PASS
npusim run 0 == run 1 makespan/marker digest
expected packets/bytes/link traffic exact
ACK/DONE exact
all residual queues == 0
PROTO_WAIT absent
ProgramIo probes exact
timing_execution == true
functional_execution == false
```

性能改善不是 correctness 门禁。若 optimized 比 naive 慢，报告保留真实结果且 economic
decision 必须回退；不得修改 golden 或强制选择来伪造收益。但性能改善是 V2 completion
门禁：correctness CTest 可以 PASS，`v2_complete` 必须保持 false。

收益 report 还必须 strict serde/reload，并分别携带：

```text
correctness_complete
measurement_complete
performance_benefit
performance_complete
v2_complete
```

## 16. 关键风险与预案

### 16.1 当前 8-token trace 太小

32B token payload 可能使 descriptor/control 开销完全压过 overlap，two-stage 聚包也难以
体现收益。

预案：

- 先用现有 trace 完成 correctness/runtime closure；
- 再增加同一 production builder 的 larger-static-trace profile；
- 大 trace 仍必须由 typed spec/trace 生成，不能在 runner 内手造 aggregate。

### 16.2 GroupGEMM 不是单个现有 action

现有 Lite MoE 将 gate/up/down 表达为多个 GEMM/SwiGLU action。

预案：使用 `MoeExpertGemmGroupDescriptor` 和 region member set，保持每个实际 GEMM
action/operand，不新增假 GroupGEMM opcode。

### 16.3 二阶段 A2A 增加中转与 copy

若 pivot 不能零拷贝聚合，两阶段可能比 direct XY 更慢。

预案：copy 必须显式进入 action DAG/cost；保留 direct candidate 与 unfused baseline，
由 planner 回退。

### 16.4 动态路由与静态 plan

生产 MoE token 数通常运行时变化。

预案：V2 锁 concrete trace；schema 同时保留 capacity/profile，为 V3 runtime binding 留接口。
缺 actual trace 的 executable lowering 一律拒绝。

### 16.5 top-k combine 数值语义

weighted reduction 不是现有 timing opcode 的完整数值实现。

预案：production runtime 先锁 top-1；top-k 只在 weight 已预乘且 FP32 SUM contract 完整时
允许 timing lowering，能力报告仍为 functional=false。

### 16.6 多 region 单 manifest

V1 主要验证 standalone pattern，V2 首次把多个 replacement region 与普通 action 混合。

预案：先做 overlay exact set algebra，再做 fragment/linker；禁止从独立 artifact 拼装结果。

### 16.7 GroupGEMM primitive/setup 压过通信

这是 Dense 实测最主要的失败原因，也最可能在 MoE 小 token block 上重现。

预案：独立标定 setup；枚举时保留低 chunk；按 expert 聚合可连续 token；同时比较
`primitive_count`、records 和 hidden communication。不能因 eta 相同就忽略 setup。

### 16.8 声明 double buffer 但没有实测 overlap

预案：将 session、slot writer/reader、compute 与 DTE start/end 写入 raw marker。若无法
重建 peak inflight 和 overlap，candidate 只算 correctness coverage，不算收益候选。

### 16.9 endpoint session 被 personalized fanout 耗尽

预案：baseline 和 fused 都用 capacity-safe peer wave；profile 中独立建模 TX/RX session；
finalizer 重建每 core 的 session 上界和 WAIT/ACK retirement。禁止提高 session capacity
来绕过调度错误。

### 16.10 balanced trace 盈利但 skewed trace 退化

预案：C4 typed skew trace 是强制 validation 点；报告 expert tail、pivot pressure 和
critical link。balanced 与 skewed 都需满足容量，收益 capability 只覆盖实际通过的
trace family。


## 17. 第二版最终产物

完成后应交付：

1. 两个 MoE fusion pattern 的 typed schema、semantics、discovery；
2. direct XY 与 Comet mesh personalized-A2A candidate generator；
3. personalized traffic/resource cost 和 deterministic decision；
4. executable unfused MoE baseline；
5. multi-region `MoeSwizzleOverlay` 与 4-Die workload exact replacement；
6. projection、Core/Operand ABI、standard lowering、single manifest；
7. ProgramIo 与 C++ finalizer dedicated trust；
8. inference/train-forward naive/auto/forced runtime comparison；
9. raw evidence、typed comparison/capability report；
10. 唯一 official CTest 与 V1/legacy no-drift 证据；
11. dedicated GroupGEMM/DTE/lifecycle/control calibration markers 与 MEASURED profile；
12. C0–C4 typed scale/trace matrix 和 calibration/validation split；
13. 两个 pattern 的 region-level AUTO 收益证据；
14. inference/train-forward 两个相邻 scale 的 >=10% workload 收益证据。

## 18. 完成后允许的能力声明

全部门禁通过后，只允许声明：

```text
supports_dispatch_gemm_discovery = true
supports_gemm_combine_discovery = true
supports_personalized_a2a_planning = true
supports_direct_xy_personalized_pipeline = true
supports_comet_mesh_personalized_pipeline = true
supports_moe_multi_region_workload_embedding = true
supports_moe_dp4_infer_timing_execution = true
supports_moe_dp4_train_forward_timing_execution = true
supports_naive_swizzle_runtime_comparison = true
supports_calibrated_group_gemm_efficiency = true
supports_calibrated_fixed_overhead = true
supports_runtime_verified_double_buffer = true
supports_runtime_verified_multi_inflight = true
supports_runtime_verified_compute_dte_overlap = true
supports_economic_moe_swizzle_auto_selection = true
supports_reproducible_moe_swizzle_speedup = true

supports_dynamic_token_routing = false
supports_joint_placement_swizzle_search = false
supports_moe_backward_swizzle = false
supports_weighted_topk_functional_execution = false
supports_functional_execution = false
```

任何 performance speedup 声明必须来自同一 workload、同一 trace、同一 build、同一
ProgramIo 输入下的 official comparison report；若 economic decision 为 UNFUSED，则只能
声明优化候选已生成并可执行，不能声明 planner 已选择或性能已提升。

上述七个 performance capability 只有在 measurement/performance completion 门禁全部
满足时才可为 true。若 correctness 已完成但收益未完成，必须保持 false，并允许前九个
correctness/timing capability 独立反映已完成事实。

## 19. 最终完成检查表

- [ ] 两个 pattern 的 semantic witness 与 strict serde 完成；
- [ ] 2-Die/4-Die region discovery 完成；
- [ ] C0–C4 typed scale 与 balanced/skewed trace 完成；
- [ ] dedicated calibration marker 与 MEASURED profile 完成；
- [ ] executable unfused/direct/two-stage candidates 完成；
- [ ] personalized cost、fallback、forced deployment 分离完成；
- [ ] cost 显式包含 GroupGEMM setup/control/lifecycle；
- [ ] 两 region 对整张 MoE DAG exact replacement 完成；
- [ ] single manifest 与 typed wrapper exact rebuild 完成；
- [ ] lifecycle roots 对 chunk 数保持不变；
- [ ] endpoint session capacity-safe wave 完成；
- [ ] runtime observed inflight >= 2；
- [ ] runtime compute/DTE overlap > 0；
- [ ] ProgramIo actual-SHA 和 terminal/tape closure 完成；
- [ ] C++ finalizer real stdin 与 tamper negatives 完成；
- [ ] 4-Die inference runtime matrix 完成；
- [ ] 4-Die train-forward runtime matrix 完成；
- [ ] 两个 pattern 各有 region-level AUTO 收益；
- [ ] inference 两个相邻 scale speedup >= 1.10；
- [ ] train-forward 两个相邻 scale speedup >= 1.10；
- [ ] Dense V1、MoE backward 和 legacy finalizer 回归完成；
- [ ] 唯一 CTest official PASS；
- [ ] capability report 严格保持 timing-only；
- [ ] `performance_complete=true` 且 `v2_complete=true`。
