# Swizzle 收益验证与优化开发方案

## 1. 目标

本方案用于把第一版 Swizzle 从“优化分支可以端到端执行”推进到“在合适负载和
真实硬件约束下，由 planner 自动选择并获得可重复收益”。

开发工作围绕七个收益条件展开：

1. 扩大 GEMM 与通信 payload；
2. 使用 4-Die 或更大 mesh 的多个方向端口；
3. 支持多于 rank 数的 chunk，且真实 `max_inflight > 1`；
4. 让 double buffer 在执行 DAG 中形成通信/计算并行；
5. 压缩 barrier、event 和中间 buffer lifecycle；
6. 保证 chunk 后的 GEMM tile 仍位于高效率区；
7. 使通信时延足够被计算覆盖，而不是人为放大延迟制造结果。

最终目标不是要求所有负载都选择 Swizzle。正确行为是：

```text
小负载、低通信占比       -> 自动选择 UNFUSED
达到 crossover 的负载    -> 自动选择 Swizzle
不满足拓扑/内存/效率约束 -> fail closed 或回退 UNFUSED
```

## 2. 当前基线与问题诊断

当前 official comparison 报告：

```text
build-debug-final/swizzle-w10-suite-official/
  swizzle-runtime-comparison-report.json
```

实际结果为：

| Pattern | Naive | Swizzle | 增幅 |
|---|---:|---:|---:|
| AG+GEMM | 309 cycles | 518 cycles | +67.6% |
| GEMM+RS | 289 cycles | 463 cycles | +60.2% |
| GEMM+AR | 378 cycles | 561 cycles | +48.4% |

当前 workload 是刻意用于闭合正确性的极小配置：

```text
M/tokens = 8
H = 16
I = 32
TP = 2
mesh = 2×1
logical communication = 256B
runtime packet_count = 16
chunk_count = 2
unroll_degree = 1
observed max_inflight = 1
```

naive 与 Swizzle 的 logical bytes、byte-hops 和 packet count 相同，Swizzle 没有减少
通信工作，却增加了控制和生命周期：

| Pattern | Naive records | Swizzle records | Naive reloc | Swizzle reloc |
|---|---:|---:|---:|---:|
| AG+GEMM | 22 | 38 | 32 | 54 |
| GEMM+RS | 32 | 44 | 46 | 68 |
| GEMM+AR | 42 | 50 | 50 | 68 |

解析 cost 在执行前也已经判断三个 fused candidate 都慢于 baseline，因此 economic
decision 正确选择 `UNFUSED`。当前 runtime 中的 Swizzle 分支来自
`FORCED_BY_POLICY`，只用于覆盖执行链，不能作为收益证据。

## 3. 收益成立的必要条件

对同一工作量，近似写成：

```text
T_unfused = T_comp + T_comm + T_base_control

T_swizzle = T_prologue
          + max(T_comp_pipeline, T_comm_pipeline)
          + T_epilogue
          + T_extra_control
          + T_tile_efficiency_loss
```

Swizzle 获益的必要条件为：

```text
被隐藏的通信时间
  > 新增控制/lifecycle时间
  + 分块导致的GEMM效率损失
  + 无法隐藏的prologue/epilogue
```

因此只扩大计算而通信不变，或只扩大通信而没有真正并发，都不保证收益。开发必须同时
改变工作量尺度和执行 DAG 的并行能力。

## 4. 完成定义

本方案完成必须同时满足：

1. 保留当前 2×1、M=8/H=16/I=32 case 作为固定 control，并继续自动回退；
2. 提供参数化 Dense TP workload，不在 runner 内手造 shape/bytes/FLOPs；
3. 4-Die 2×2 production placement 可运行 Wang 1D 和 MeshSlice 2D OS；
4. Wang chunk 数不再固定等于 rank 数；
5. 至少一个 official candidate 为 `unroll_degree=2`；
6. runtime 观测到 `max_inflight >= 2`，不是仅在 hardware profile 中声明；
7. double-buffer 两个 slot 的地址、生命周期、覆盖依赖全部 exact；
8. records/control 开销相对 chunk 数呈次线性增长；
9. `eta(tile)` 使用多点标定，不再只有 toy efficiency point；
10. planner economic decision 自动选择 fused candidate，禁止用 forced 结果完成收益门禁；
11. 至少两个相邻 scale point 的 actual makespan 相比 naive 改善不低于 10%；
12. finalizer×2、ProgramIo actual-SHA、npusim×2、ACK/DONE/drain/
    `PROTO_WAIT` 全部通过；
13. Dense V1、UNFUSED、4-Die MoE 和 legacy runtime 无行为漂移。

性能收益的最低交付目标：

```text
AG+GEMM 或 GEMM+RS 中至少一个 pattern：
  两个相邻 4-Die scale point 均 >= 10% speedup

另一个 pattern：
  至少一个 scale point 自动选择 fused，或给出可审计的 UNFUSED 原因

GEMM+AR：
  correctness/runtime 必须闭合；若 reduction+replication 结构不盈利，允许回退
```

## 5. 总体实施路径

```text
固定2-Die小负载控制组
        |
        v
参数化M/H/I/TP/mesh workload
        |
        v
4-Die 2×2 placement + Wang/MeshSlice production candidates
        |
        v
独立chunk枚举 + unroll2 + real double buffer
        |
        v
移除假串行依赖 + inflight DTE
        |
        v
lifecycle/event/barrier压缩
        |
        v
eta(tile)标定 + cost校准
        |
        v
查找真实crossover
        |
        v
economic auto-selection official runtime
```

## 6. 参数化 workload 设计

### 6.1 新增 typed scale profile

新增独立 schema，而不是复制多个 integration fixture：

```python
@dataclass(frozen=True, slots=True)
class SwizzleScalePoint:
    schema_version: str
    id: str
    name: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    tp: int
    mesh_rows: int
    mesh_columns: int
    dtype: DType
```

validator 必须证明：

- `tp == mesh_rows * mesh_columns`；
- M/N/K 可被 TP 与候选 chunk 合法整除；
- rank-local tensor shape、bytes、GEMM FLOPs 从 model/IR0 重建；
- hardware fabric 确实包含对应矩形；
- 不允许 runner 覆盖 derived bytes/FLOPs。

### 6.2 初始 scale matrix

建议第一轮使用：

| Scale | Tokens M | H | I | TP/Mesh | 目的 |
|---|---:|---:|---:|---|---|
| S0 | 8 | 16 | 32 | 2 / 2×1 | 固定回退控制组 |
| S1 | 32 | 64 | 256 | 4 / 2×2 | 验证4-Die链路 |
| S2 | 64 | 64 | 256 | 4 / 2×2 | 固定权重、放大 M 的初始流水点 |
| S3 | 128 | 64 | 256 | 4 / 2×2 | 固定权重、放大 M 的 crossover 点 |
| S4 | 256 | 512 | 2048 | 4 / 2×2 | 压力与收益稳定性 |

这些只是首轮搜索点，不是 golden。S1–S3 明确固定 `H=64/I=256`，只通过 tokens
扩大 M、payload 和工作量；不能宣称 H/I 也随 scale 递增。每轮先从 IR0/Problem 输出
实际 rank bytes、FLOPs、tile 和 SRAM，再决定是否保留。若 S4 超出 SRAM，必须由
feasibility 拒绝，不能提高预算绕过。

### 6.3 扫描纪律

- 同一 scale 的 naive/Swizzle 使用同一 graph、placement 和 hardware profile；
- 不允许只给 Swizzle 增加计算或减少通信；
- 不允许仅对成功点隐藏失败点；
- 保存所有 scale 的 economic decision 和拒绝原因；
- 至少两个相邻点出现收益才认定 crossover 稳定。

## 7. 4-Die Mesh 与候选接入

### 7.1 生产 2×2 placement

新增 `hardware_2x2` production case，要求：

- 4 ranks 对应 4 个物理 die；
- rows/columns、snake order 和全部 `PairRoute` 来自 IR1；
- resource incidence 包含 egress、link、port、cut；
- 不能使用 synthetic group 绕过 `place_ir0`；
- Wang line 和 MeshSlice rectangle 使用同一个 placement truth。

### 7.2 Wang 1D 在 4 Die 上的角色

Wang 用作低控制复杂度候选：

- 生成 bidirectional line；
- 只有真实一跳 Hamiltonian cycle 时才生成 ring；
- 不因物理 2×2 就声称同时利用横纵端口；
- cost 必须按实际 action/resource DAG 统计端口并行。

### 7.3 MeshSlice 2D OS production 接入

当前 MeshSlice 已有 candidate/unit coverage，但不在 official comparison 的 allowed
algorithms 中。需完成：

1. production planner 注册 `MESHSLICE_2D_OS`；
2. 2×2 explicit 2D sharding witness；
3. row/column AG 同时在飞；
4. 两输入 ready 后启动 COMP；
5. output-stationary loop accumulator；
6. double-buffered input slots；
7. standard projection/ABI/lowering/finalizer/runtime；
8. 2×2 positive 和 1×4 rectangle/2D-sharding negative。

4-Die 本身不是收益条件。只有 runtime evidence 证明多个方向资源在时间上重叠使用，
才可声明 Mesh 并行带来收益。

## 8. Chunk 枚举改造

### 8.1 当前问题

Wang 目前固定：

```python
chunk_count = len(ranks)
```

TP2 只能得到2块，TP4只能得到4块，无法独立平衡首尾开销、GEMM tile 效率和 DTE
流水深度。

### 8.2 新枚举规则

新增 canonical chunk factors：

```text
base = participant_count
candidate chunks = divisors(split_extent) ∩ {base, 2*base, 4*base, 8*base}
```

同时保留必要的 `chunk_count=1` 或 executable unfused baseline 作为对照。每个候选必须
通过：

- split axis exact divisibility；
- rank-local M/N/K tile exact；
- `chunk_count <= max_chunk_count`；
- transfer bytes >= `min_transfer_bytes`；
- tile >= `efficient_tile_floor`；
- actions/buffers <= constraints；
- SRAM high-water <= budget；
- loop accumulator 与 output coverage exact。

### 8.3 Candidate stable key

canonical key 至少包含：

```text
pattern
algorithm
topology kind/order
split axis
chunk_count
unroll_degree
tile shape
buffer slot count
```

candidate cap 必须在 canonical sort/dedup 后执行，禁止输入顺序影响候选集。

## 9. Double Buffer 与真实 Inflight

### 9.1 已有基础

当前已有：

- `double_buffer_supported` hardware flag；
- Wang `unroll_degree ∈ {1,2}`；
- `SwizzleBufferRequirement.double_buffered`；
- IR2 `DOUBLE_BUFFER` role；
- slot-aware Core/Address ABI；
- cost 中双倍 SRAM 计数。

缺失的是 official auto-selection 与真实并发证据。

### 9.2 执行语义

对 unroll=2，固定：

```text
slot(chunk) = chunk % 2
```

合法流水：

```text
recv(chunk i+1, slot 1) ----+
                              |
gemm(chunk i, slot 0) --------+-- 可重叠
                              |
send/reduce(chunk i-1) -------+
```

dependency 规则：

- COMP 只依赖当前 chunk 的 ready action；
- 下一个 chunk 的 SEND/RECV 不依赖当前 COMP；
- 只有复用同一 slot 时，写入依赖该 slot 上次最后一个 reader；
- WAIT 只阻塞真实 consumer，不充当全局 barrier；
- loop accumulator 依赖保持串行，但不能串行化独立通信；
- final boundary 依赖所有 required outputs，而不是所有中间控制动作。

### 9.3 Runtime 门禁

不能以 `hardware_profile.max_inflight_dte=4` 作为完成证据。必须从实际 schedule/marker
重建：

```text
observed_max_inflight_send >= 2
observed_max_inflight_recv >= 2
two buffer slots both used
no overlapping writes to same slot
no read after slot reuse
```

若 action DAG 仍得到 `max_inflight=1`，应把候选判为“double-buffer allocation exists but
pipeline not realized”，不得计入收益候选。

## 10. Barrier、Event 与 Lifecycle 压缩

### 10.1 优化原则

所有压缩必须由 dependency/liveness 证明，不能删除 validator 认为麻烦的记录。

### 10.2 Persistent ping-pong allocation

将：

```text
每chunk ALLOC -> use -> FREE
```

改为：

```text
每rank/每operand 2-slot ALLOC
  -> 所有chunk循环复用
  -> final FREE
```

要求：

- region/span/slot 地址固定；
- lifetime 覆盖全部 uses；
- terminal/borrowed buffer 不被错误 FREE；
- ProgramIo allocation closure保持不变。

### 10.3 Event/token 环形复用

- event ID 可按 `(rank, lane, phase)` 复用；
- DTE token 可按 `(flow, lane)` 复用；
- RECV/WAIT 必须保持 exact pair；
- 新一轮复用必须依赖前一轮 DONE/consumer；
- 不能跨 route、peer、direction 复用。

### 10.4 Barrier 下沉

优先使用：

```text
producer -> consumer direct dependency
```

仅在以下位置保留 barrier：

- 多 rank 最终 output ownership 收敛；
- AR reduction→replication 阶段边界；
- 程序 terminal/DONE 前的必要闭合。

禁止每 chunk 建全局 barrier。AG output-slice 模式应允许各 slice 独立完成。

### 10.5 控制开销预算

对每个候选计算：

```text
records_per_chunk
control_actions_per_chunk
events_per_chunk
alloc_free_pairs_per_rank
relocations_per_chunk
```

收益候选建议满足：

```text
alloc/free pairs = O(rank * operand_count)，不随chunk线性增长
barrier count = O(rank + phase_count)，不随chunk线性增长
event/token count = O(rank * unroll_degree)
```

## 11. GEMM Tile 效率标定

### 11.1 当前问题

现 integration profile 只有一个 toy 点：

```text
eta(4,4,4) = 0.9
```

它无法可靠判断 S2–S4 的 tile，也无法量化切成8/16块后的效率损失。

### 11.2 标定接口

扩展 `SwizzleEfficiencyPoint` 或增加 profile table：

```text
(Mtile, Ntile, Ktile, dtype, accumulation_dtype)
  -> efficiency
  -> confidence_fraction
  -> setup_cycles
```

首轮至少覆盖：

```text
Mtile: 4, 8, 16, 32, 64, 128
Ntile: 16, 32, 64, 128, 256
Ktile: 16, 32, 64, 128, 256
```

不要求完整笛卡尔积；选择实际候选会经过的 shape。

### 11.3 数据来源

优先级：

1. 同一 matching `npusim` build 的 isolated MATMUL timing microbench；
2. 已验证的硬件 profile；
3. 解析模型，只作为 provisional 数据并扩大置信区间。

每个 efficiency table 必须携 profile digest 和生成工具 SHA。不能用当前运行结果反向调参
直到恰好选择 Swizzle。

### 11.4 Planner 门禁

- tile 不在 table 内时使用明确插值/保守下界；
- 超出标定范围不能沿用最近点并声称精确；
- `eta` 下降造成的 compute loss 必须进入 cost；
- cost 区间重叠时继续按确定性 tie-break，不强选 fused。

## 12. Cost Model 校准

### 12.1 增加固定开销项

当前 analytic cost 与 actual cycles 量级不同。需要显式建模：

- MATMUL setup；
- DTE launch/sync；
- SRAM_BIND；
- ALLOC/FREE；
- EVENT_SET/WAIT；
- REDUCE setup；
- terminal/DONE control。

这些值来自微基准或 runtime marker，不从最终 speedup 反推。

### 12.2 拟合与验证分离

- calibration set：S0、S1 与部分 isolated microbench；
- validation set：S2、S3、S4；
- 不允许在 validation 结果出来后修改同一 profile 再把它当独立验证；
- 保存预测/实测误差和 candidate 排名一致率。

### 12.3 决策质量门禁

不要求 cycles 绝对预测完全一致，但要求：

- profitable/unprofitable 分类正确；
- selected candidate 排名稳定；
- actual 最优候选落在 predicted top-2；
- 若置信区间重叠，自动回退或保留并列，不能过度确信。

## 13. Runtime 证据与报告

### 13.1 Branches

每个 scale/pattern 运行：

```text
NAIVE
SWIZZLE_AUTO
SWIZZLE_FORCED_DIAGNOSTIC（仅诊断）
```

收益结论只能使用 `SWIZZLE_AUTO`。forced 分支用于证明未被选候选仍可执行，必须从性能
汇总中单独标记。

### 13.2 必须报告的字段

```text
problem shape / rank shape
logical bytes / byte-hops / packets
GEMM FLOPs / tile shape / eta
algorithm / topology / chunk_count / unroll_degree
predicted cycles interval
actual makespan ×2
observed max inflight send/recv
directional port utilization over time
compute/DTE overlap cycles
prologue/steady/epilogue
record/opcode/control counts
SRAM high-water
artifact bytes / SHA
ProgramIo init/probe counts
economic decision / deployment reason
```

### 13.3 收益判定

```text
speedup = naive_makespan / swizzle_auto_makespan
```

正式“获得收益”要求：

- `speedup >= 1.10`；
- 两次 makespan 和 marker digest exact；
- 至少两个相邻 scale point 达标；
- same work invariants 全部相等；
- economic decision 自动选择对应 fused candidate；
- 没有 `PROTO_WAIT`，所有 residual 为0。

## 14. 完整开发顺序

### W0：冻结基线与报告 schema

1. 固定当前 official report 和 S0 control；
2. 增加 predicted/actual、inflight、overlap、record breakdown 字段；
3. capability 中 `performance_benefit=false`；
4. 锁定 V1 三 pattern 和 UNFUSED counts。

门禁：不改变现有 makespan 和 economic decision。

### W1：参数化 workload

1. 新增 `SwizzleScalePoint`；
2. 从正式 ExperimentSpec 构建 S0–S4；
3. IR0/IR1/Problem exact provenance；
4. 2×1/2×2 hardware profile；
5. shape、bytes、FLOPs、SRAM negatives。

门禁：所有 scale 可发现 candidate，尚不要求 fused 盈利。

### W2：4-Die production topology

1. TP4 placement；
2. physical rows/columns/snake/routes；
3. Wang 4-rank candidate；
4. standard projection/lowering/runtime preflight。

门禁：4-Die Wang naive/forced 可运行，work totals exact。

### W3：独立 chunk 枚举

1. chunk factors；
2. tile/transfer/SRAM feasibility；
3. canonical dedup/cap；
4. per-pattern action/writer coverage；
5. invalid divisor、tiny payload、tile-floor negatives。

门禁：同一4-Die problem 至少产生4/8/16 chunk 中两个合法点。

### W4：真实 double buffer/inflight

1. slot-aware dependencies；
2. unroll2 rank programs；
3. Core/Address/Operand ABI slot closure；
4. cost resource slots；
5. runtime observed inflight markers；
6. overwrite/read-after-reuse negatives。

门禁：至少一个 4-Die candidate actual `max_inflight>=2`。

### W5：控制与生命周期压缩

1. persistent ping-pong allocations；
2. event/token reuse；
3. barrier down-scope；
4. record/relocation quotient更新；
5. finalizer dedicated negatives；
6. ProgramIo allocation/terminal回归。

门禁：相同 candidate 语义下 record/control count 下降，makespan 不回退。

### W6：MeshSlice production

1. allowed algorithm/registry；
2. 2D sharding proof；
3. row/column inflight；
4. loop accumulator/double buffer；
5. standard manifest/finalizer/runtime；
6. Wang/MeshSlice/economic decision比较。

门禁：2×2 MeshSlice actual 多方向重叠证据闭合。

### W7：eta 与 fixed-overhead 校准

1. isolated timing microbench；
2. efficiency table；
3. action fixed-cost table；
4. calibration/validation split；
5. cost interval和ranking回归。

门禁：S2–S4 profitable分类与实际一致。

### W8：Crossover sweep

1. S0–S4 全部 naive/auto/forced；
2. 收集报告；
3. 找到 crossover；
4. 若无收益，按瓶颈分类回到 W3/W4/W5/W6，不改验收阈值；
5. 固定两个相邻盈利点。

门禁：满足第13.3节，或形成明确、可审计的“不盈利原因”报告。

### W9：MoE迁移与正式回归

1. 将验证过的 chunk/double-buffer/control compaction 复用到第二版 MoE planner；
2. 4-Die MoE inference/train-forward 先 correctness，再检查收益；
3. Dense/UNFUSED/MoE backward legacy no-drift；
4. CMake 注册唯一收益 comparison test；
5. capability/evidence 文档更新。

门禁：official CTest 与全部相邻回归通过。

## 15. 建议文件布局

新增：

```text
llm/frontend/wafer_frontend/schema/
  swizzle_scale.py
  swizzle_performance_evidence.py

llm/frontend/wafer_frontend/passes/
  build_swizzle_scale_cases.py
  calibrate_swizzle_profile.py

llm/frontend/wafer_frontend/policies/swizzle/
  chunking.py
  lifecycle.py

llm/test/frontend/integration/
  swizzle_scale_cases.py
  run_swizzle_benefit_runtime.py
  test_swizzle_scale_cases.py
  test_run_swizzle_benefit_runtime.py

llm/test/frontend/unit/
  test_swizzle_chunking.py
  test_swizzle_double_buffer.py
  test_swizzle_lifecycle.py
  test_swizzle_efficiency_profile.py
```

预计窄改：

```text
policies/swizzle/wang_1d.py
policies/swizzle/meshslice_2d.py
policies/swizzle/cost.py
policies/swizzle/enumerate.py
policies/swizzle/decide.py
schema/swizzle.py
schema/swizzle_ir2.py
schema/swizzle_abi.py
lowering/swizzle_abi.py
lowering/swizzle_standard.py
passes/project_swizzle_ir2.py
schema/artifact_manifest.py
passes/program_io.py
program_finalizer.cpp/selftest
CMakeLists.txt
```

shared validator 修改必须以 pattern/algorithm/producer exact scoped，不允许为测试规模放宽
通用 schema。

## 16. 关键负测

### 16.1 Chunk/Tile

- split extent不能整除；
- chunk payload低于DTE下限；
- tile低于效率下限；
- chunk_count超限；
- candidate截断受输入顺序影响；
- GEMM action FLOPs总和漂移。

### 16.2 Double Buffer

- 两slot地址重叠；
- chunk错误绑定slot；
- slot重用早于最后reader；
- unroll2但只有一个物理slot；
- profile禁止double buffer仍生成候选；
- SRAM high-water未乘2；
- schema声称inflight2但schedule实测为1。

### 16.3 Lifecycle/Event

- ALLOC后无FREE；
- FREE早于最后use；
- terminal被FREE；
- event跨peer/route复用；
- token跨direction复用；
- 删除barrier后output ownership未闭合；
- non-Swizzle producer使用dedicated admission。

### 16.4 收益报告

- forced结果冒充economic auto；
- naive/Swizzle工作量不同；
- 只保留盈利点；
- 两次运行marker/makespan不同；
- artificial hardware profile未记录digest；
- calibration和validation使用同一测量点；
- speedup小于阈值却提升能力声明。

## 17. 风险与处置

### 17.1 4-Die Wang仍无法利用二维端口

这是可能且合理的结果。Wang是一维候选，不能仅因放在2×2上就声称二维带宽收益。
处置：完成 MeshSlice production，而不是人为提高 Wang port utilization。

### 17.2 Double buffer仍没有inflight

说明依赖 DAG 存在假串行，或DTE/compute使用同一互斥资源。处置：输出每个 action 的
earliest/start/end/resource/deps，逐边证明；不能只增加 `max_inflight_dte`。

### 17.3 大负载计算完全掩盖通信

如果 compute 远大于 communication，naive 本身的通信比例可能很低，Swizzle也无明显
收益。处置：扫描计算通信比，而不是继续单向扩大 GEMM。

### 17.4 小tile效率损失抵消收益

处置：减少 chunk、提高 tile floor，或回退。不能在 cost 中忽略效率损失。

### 17.5 控制压缩破坏fail-closed

处置：所有复用携带显式 lifetime/token generation，C++仍重建关键闭包。若无法证明，
保留原记录并接受不盈利。

### 17.6 找不到任何真实收益点

这仍是有效结论。必须输出瓶颈分解：

```text
extra control
tile efficiency loss
insufficient overlap window
resource serialization
prologue/epilogue
```

不得继续调硬件参数直到结果变正。只有基于真实目标硬件的新 profile 才能重新评估。

## 18. 最终交付物

1. 参数化 S0–S4 workload 与 2×1/2×2 production placement；
2. rank-independent chunk enumeration；
3. actual inflight double-buffer execution；
4. persistent lifecycle、event/token reuse 和 barrier下沉；
5. production MeshSlice 2D OS runtime；
6. 多点 `eta(tile)` 与 fixed-overhead profile；
7. predicted/actual crossover report；
8. 至少两个相邻盈利点，或正式不盈利根因报告；
9. economic auto-selection official runtime；
10. Dense、UNFUSED、MoE 和 legacy no-drift 证据。

## 19. 完成后允许的能力声明

只有满足全部收益门禁后才可声明：

```text
supports_rank_independent_chunk_search = true
supports_runtime_verified_double_buffer = true
supports_runtime_verified_multi_inflight = true
supports_production_meshslice_2d = true
supports_control_lifecycle_compaction = true
supports_calibrated_tile_efficiency = true
supports_economic_swizzle_auto_selection = true
supports_reproducible_swizzle_speedup = true
```

以下声明仍不能自动提升：

```text
supports_functional_execution = false
supports_arbitrary_mesh = false
supports_dynamic_routing = false
supports_all_workloads_speedup = false
```

## 20. 最终检查表

- [ ] S0小负载保持UNFUSED；
- [ ] S1–S4 typed workload可稳定重建；
- [ ] 4-Die production placement闭合；
- [ ] Wang chunk数与rank数解耦；
- [ ] MeshSlice进入official runtime；
- [ ] 至少一个candidate使用unroll2；
- [ ] actual max_inflight不少于2；
- [ ] double-buffer slot/lifetime负测齐全；
- [ ] barrier/event/lifecycle开销下降；
- [ ] eta与fixed-cost完成独立标定；
- [ ] economic auto而非forced选择fused；
- [ ] 两个相邻scale点speedup不少于1.10；
- [ ] finalizer×2、npusim×2与ProgramIo exact；
- [ ] Dense/UNFUSED/MoE/legacy回归全绿；
- [ ] capability report没有越界声明。
