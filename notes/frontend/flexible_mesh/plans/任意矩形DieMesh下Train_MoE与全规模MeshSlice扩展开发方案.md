# 任意矩形 Die Mesh 下 Train、MoE 与全规模 MeshSlice 扩展开发方案

## 1. 目标与完成定义

本方案在现有 `1≤H,W≤10`、`R=H×W≤100` 的完整矩形 Die Mesh 基础上，
进一步支持：

- Dense train 完整训练步；
- MoE inference；
- MoE train；
- MeshSlice 在全部 100 种合法矩形上的统一编排和运行；
- 每种负载均有确定性的 executable baseline；
- 优化策略不可用、无收益或超容量时自动回退；
- finalizer、ProgramIO、resolver 和 npusim timing execution 真实闭环。

“任意 Mesh”继续沿用现有定义：完整、连续、无洞的 `H×W` 矩形，
`1≤H,W≤10`，一 Die 一 rank，row-major rank，X-first XY route。它不包括
故障 Mesh、缺 Die、torus、动态绕行或在更大 Fabric 中选择子矩形。

首版执行语义继续固定为：

    timing_execution = true
    functional_execution = false

训练运行表示一个有限、可终止、状态闭合的 training step。它必须包含真实的
forward/backward/gradient synchronization/optimizer action 和状态更新顺序，
但首版不声明更新后的参数具有数值 functional correctness。

### 1.1 首版负载范围

为了尽量复用现有实现并加速开发，首个可发布版本固定以下范围：

| 负载 | 首版范围 |
|---|---|
| Dense train | 单实例，`PP=1`，`DP×TP=R`，单 microbatch，forward+backward，SGD |
| MoE inference | `EP=R`，每 Die 一个 expert，top-k=1，静态 trace，Direct-XY baseline |
| MoE train | 同一 `EP=R` 布局，forward+backward，expert/gate WGRAD，SGD |
| MeshSlice | 1×1、1×N、N×1、完整二维矩形；AG_GEMM、GEMM_RS、GEMM_AR |

后续再增加：

- 多 microbatch gradient accumulation；
- AdamW；
- `DP×EP` expert replica 和 expert-gradient DP AllReduce；
- `TP×EP` expert sharding；
- top-k=2；
- recompute、pipeline parallel 和多 stage scheduling。

这些后续项不得阻塞首版“所有 Mesh 均可运行”的完成状态。

### 1.2 分层完成状态

必须避免把 schema、candidate 或 lower/link 成功误报为负载已运行。新增状态：

    workload_contract_complete
      = 通用 workload/axis/state/capacity schema 和 typed fallback 完成

    meshslice_all_rect_complete
      = 全部 100 种 Mesh 均有合法 MeshSlice 模式或 compute-only 退化模式，
        且代表尺寸完成 standard lower/link/runtime

    dense_train_rect_complete
      = 全部 100 种 Mesh 的 tiny Dense train step 真实运行，
        代表尺寸的完整 forward/backward/SGD 状态闭合

    moe_infer_rect_complete
      = 全部 100 种 Mesh 的 static-trace MoE inference canary 真实运行，
        balanced/skewed 代表集通过

    moe_train_rect_complete
      = 全部 100 种 Mesh 的 tiny MoE train step 真实运行，
        expert/gate gradient、optimizer 和状态写回闭合

    flexible_mesh_workloads_complete
      = workload_contract_complete
        && meshslice_all_rect_complete
        && dense_train_rect_complete
        && moe_infer_rect_complete
        && moe_train_rect_complete

每个状态还必须分别记录：

    schema_verified
    candidate_verified
    lower_link_verified
    program_io_verified
    runtime_verified
    repeatability_verified

只有最后三项有真实 evidence 时，才允许使用“可运行”表述。

## 2. 当前基础与阻断点

### 2.1 可直接复用的基础

1. `RectMeshSpec` 已覆盖全部 100 种矩形、row-major、X-first 和 timing-only。
2. Fabric、HBM address space、expected group 和 PairRoute 已支持任意 `R≤100`。
3. Hamiltonian topology 已改为 O(R) 构造，不再使用指数 DFS。
4. AG、RS、AR executable baseline 已支持任意 rank，并使用 capacity-safe waves。
5. C++ collective planner 的 symmetric wave 每核每波为一次 send 和一次 receive，
   session demand 为 2，小于生产容量 3。
6. standard Swizzle IR2、Core ABI、Operand ABI、lowering、linker 和 ProgramIO
   已有 rank-generic 基础。
7. MeshSlice 已完成矩形 placement、2×3/3×2 standard lower/link、peer subview
   和 relocation addend。
8. Dense train-forward 已存在从 ExperimentSpec 到 manifest 的独立生产链。
9. MoE Swizzle 已存在 topology/problem/candidate/IR2/Core ABI/Operand ABI、
   whole-workload projection、standard lowering 和 linker。
10. Lite train/MoE backward 已证明 WGRAD、REDUCE、SGD、state ABI 和单 manifest
    可以真实 lower/link/run。

### 2.2 Dense train 当前阻断

- `TrainWorkloadSpec` 明确拒绝 `backward=true`；
- optimizer 只允许 `none`；
- 正式 runtime evidence 固定 DP2×TP2；
- Lite backward 主要覆盖 LM-head 或固定 DP2/DP4 quotient；
- 固定 action/state 数验证不能直接推广到完整模型和任意 R；
- `compile_rect_mesh` 当前只允许 Dense inference；
- standard Swizzle region 尚未合并进主 whole-workload artifact。

### 2.3 MoE 当前阻断

- `MoeSwizzleScaleSpec` 固定 expert_count=4、top-k=1、2×2、H16/I32；
- scale truth builder 直接写入 `mesh_rows=2, mesh_columns=2`；
- Lite MoE inference/train/backward carrier 固定 EP4 和四 Die；
- 部分 route fallback 仍依赖 `moe.scale.route.rX.rY` 字符串约定；
- runtime marker parser 限制 die ID≤3，并要求 2×2 session evidence；
- calibrated runtime report 假设八条 2×2 directed links；
- token/expert assignment、state ownership 和 backward tape 尚未参数化到 R；
- 当前 MoE train 只覆盖隔离的 down-projection backward，不是完整 MoE step。

### 2.4 MeshSlice 当前阻断

- production bridge 只接受 AG+GEMM；
- admission 要求 rows>1 且 columns>1，因此排除 1×N、N×1 和 1×1；
- exact OS sharding 要求两个非空、不同的 Mesh axis；
- 任意矩形 RS/AR 尚未通过 standard lower/link；
- 多 slice 只在旧 2×2 路径有完整生命周期证据；
- 当前 row/column peer action 没有形成显式 session wave carrier；
- 5×6、6×5、10×10 只有 candidate evidence，没有最终 runtime evidence。

### 2.5 共同阻断

- 100-shape finalizer/ProgramIO/npusim canary 尚未完成；
- Python 只持有 symbolic record count，精确 artifact file bytes 仍在 finalizer；
- training layer、microbatch、MoE token 数会放大 action/record/state；
- 1M records、64 MiB file、3 sessions/core/wave、U16 endpoint 和 transport tag
  必须统一预检；
- capability report 仍是 Dense-first，不能表达 train/MoE/MeshSlice 各自状态。

## 3. 核心设计决策

### 3.1 不修改冻结的旧 schema

保留以下路径及其 stable IDs：

- 旧 Dense inference RectMesh v1；
- DP2/DP4 Lite train；
- EP4/2×2 Lite MoE；
- C0-C4 MoE scale truth；
- 旧 2×2 MeshSlice fast path。

新增并列的 v2 carrier，不把 `range(4)`、EP4 或 2×2 常量直接替换为 R。
旧 case 通过 adapter 进入 v2，用于证明语义等价，而不是反向修改旧 artifact。

### 3.2 新增统一 workload envelope

新增 `FlexibleMeshWorkloadSpec`，组合而不是扩写 `RectMeshSpec`：

    mesh: RectMeshSpec
    workload_kind:
      dense_infer | dense_train | moe_infer | moe_train
    execution_mode: timing
    axis_mapping: RectMeshAxisMapping
    training: FlexibleTrainSpec | null
    moe: FlexibleMoeSpec | null
    meshslice: FlexibleMeshSliceSpec
    capacity: FlexibleMeshCapacityProfile

`RectMeshSpec` 继续描述物理 envelope。workload、逻辑并行轴和状态生命周期属于
新 schema，避免改变已冻结的 `workload_scope=DENSE_FIRST` 语义。

### 3.3 每种负载始终保留 baseline

| 负载 | 必备 baseline |
|---|---|
| Dense train | ordinary GEMM/attention + arbitrary-rank AG/RS/AR + local SGD |
| MoE inference | static bucket + Direct-XY personalized A2A + local expert compute |
| MoE train | inference baseline + reverse A2A + local WGRAD + gate AR + local SGD |
| MeshSlice | compute-only/1D/2D typed MeshSlice；失败时回退普通 GEMM+collective |

Comet、Wang、full-2D MeshSlice 和 overlap 都是优化候选，不是唯一可运行路径。

### 3.4 “任意 Mesh”不等于“任意并行因子”

首版为每种物理 Mesh 提供至少一种规范映射：

- Dense train：默认 `DP=H, TP=W, PP=EP=1`；允许显式转置为
  `DP=W, TP=H`，前提是模型维度和 token 数整除；
- MoE inference/train：默认 `EP=R, DP=TP=PP=1`，每 Die 一个 expert；
- MeshSlice：物理 Y 为 row axis，物理 X 为 column axis；尺寸为 1 的轴保留为
  typed degenerate axis。

因此 1×1、1×N、N×1 仍有合法执行模式。后续 `DP×EP`、`TP×EP` 是增量能力，
不影响首版完成状态。

### 3.5 先正确运行，再做收益门禁

- NAIVE/baseline 只要求正确完成和确定性；
- AUTO 在无校准、无收益或超容量时必须回退；
- forced optimization 只用于证明可执行，不用于收益发布；
- 所有性能声明必须来自同 workload、同 trace、同 placement 的重复运行。

## 4. 目标架构

```text
ExperimentSpec / FlexibleMeshWorkloadSpec
                 │
                 ▼
        Flexible workload validator
                 │
       ┌─────────┼───────────┐
       ▼         ▼           ▼
 Dense train   MoE graph   Dense/MoE GEMMs
   graph       + trace          │
       └─────────┼──────────────┘
                 ▼
       RectMesh axis/group registry
       full / row / column / TP / DP / EP groups
                 │
                 ▼
        ordinary + baseline collective
        + optional Wang/MeshSlice/MoE Swizzle
                 │
                 ▼
       whole-workload IR2 + state/value bridge
                 │
                 ▼
        schedule + capacity preflight
                 │
                 ▼
        standard fragments + one manifest
                 │
                 ▼
       ProgramIO → finalizer → resolver → npusim
                 │
                 ▼
          typed capability/runtime report
```

### 4.1 RectMesh group registry

从 `RectMeshSpec` 和 `RectMeshAxisMapping` 一次性派生：

- `FULL`：全部 R ranks；
- `ROW[y]`：同一物理行；
- `COLUMN[x]`：同一物理列；
- `TP[k]`、`DP[k]`、`EP[k]`：逻辑轴映射后的 canonical group；
- 每个 group 的 rank order、coordinate、PairRoute 和 digest。

group ID 必须由 `(mesh_digest, axis_kind, fixed_coordinate)` 稳定派生。禁止各
pass 自行重建 rows/columns 或假设 rank 与 x/y 相同。

### 4.2 主 compiler dispatch

新增：

    compile_flexible_mesh_workload(...)

统一执行：

1. envelope、shape、trace、state 和 capacity validation；
2. workload-specific IR0 build；
3. placement 和 group registry；
4. fusion/standalone region partition；
5. baseline 和 optimized candidate planning；
6. whole-workload projection、schedule、lower/link；
7. exact symbolic record preflight；
8. ProgramIO；
9. finalizer 精确 file-byte preflight；
10. runtime report。

`AUTO` 必须先证明 baseline 可执行，再考虑替换局部 region。任何 optimized region
失败都不能破坏 ordinary、state、standalone collective 或其他 region。

### 4.3 单 artifact 原则

每个 inference request 或 training step 只产生一个 linked manifest/artifact。
manifest 可以包含：

- ordinary compute；
- ISA fusion region；
- standalone collective；
- MeshSlice region；
- MoE Swizzle region；
- gradient synchronization；
- optimizer/state load/store。

region replacement 必须满足：

    replaced_action_ids ∩ preserved_action_ids = ∅
    replaced ∪ preserved = source_global_actions

并保持 terminal outputs、persistent states、start events 和 core streams 闭合。

## 5. Dense train 设计

### 5.1 Schema 扩展

新增 `FlexibleTrainSpec`：

    step_kind: forward_backward_update
    micro_batch_count: 1                # 首版
    recompute: none                     # 首版
    optimizer: sgd                      # 首版
    gradient_accumulation: fp32
    parameter_dtype: fp16
    master_parameter_dtype: fp32 | none
    loss: cross_entropy
    pp_degree: 1

不直接删除 `TrainWorkloadSpec` 的旧 fail-closed 校验；新增 v2 parser/adapter。

### 5.2 逻辑训练图

完整训练步至少包含：

1. parameter/state load；
2. forward embedding、attention、MLP、norm、LM head；
3. cross entropy forward；
4. CE backward；
5. LM-head、MLP、attention、norm、embedding backward；
6. data-gradient 和 weight-gradient GEMM；
7. DP gradient synchronization；
8. optimizer update；
9. updated parameter/state store；
10. loss、gradient/state probes。

每个 backward node 必须引用 exact forward activation/tape origin。不能根据名称猜测
对应关系。

### 5.3 状态 ABI

每个 trainable parameter 明确拥有：

- parameter storage；
- FP32 gradient accumulation storage；
- 可选 master parameter；
- optimizer state；
- forward activation 或 recompute witness；
- lifetime 和 alias contract。

顺序门禁：

    all local gradient writes
      → DP reduction complete
      → optimizer reads reduced gradient
      → parameter write
      → persistent store

任何 optimizer action 早于 gradient sync 都必须 schema fail。

### 5.4 DP×TP 映射

默认：

    DP = H
    TP = W

每个物理 row 是一个 TP group，每个物理 column 是一个 DP group。显式 transpose
时反向映射。

- TP group 承担 forward/backward AG、RS、AR；
- DP group 承担 weight-gradient AllReduce；
- sequence parallel 的 token divisibility 由 TP 精确校验；
- `global_batch = micro_batch × DP × micro_batch_count`。

1×N 退化为 DP1×TPN；N×1 退化为 DPN×TP1；1×1 无跨 Die collective。

### 5.5 训练 MeshSlice

新增 `MeshSliceGemmRole`：

- `FORWARD`；
- `DATA_GRAD`；
- `WEIGHT_GRAD`。

每个 role 显式定义 M/N/K 与 DP/TP axis 的映射、输入 shard、output ownership 和
terminal collective：

- forward：复用 AG_GEMM 或 GEMM_RS；
- data grad：转置权重后的 AG_GEMM/GEMM_RS；
- weight grad：token/DP partial accumulation，随后 DP RS/AR；
- 不满足整除或 SRAM 预算时回退普通 GEMM+collective。

### 5.6 首版训练边界

首版不要求：

- PP>1；
- activation recompute；
- ZeRO/sharded optimizer；
- gradient clipping；
- mixed dynamic loss scaling；
- AdamW；
- 无限 step loop。

一个有限 SGD step 真实运行即可建立 `dense_train_rect_complete`。

## 6. MoE inference 设计

### 6.1 通用 MoE spec

新增 `FlexibleMoeSpec`：

    expert_count: R                      # 首版
    expert_parallel_degree: R
    top_k: 1                             # 首版
    trace_mode: static
    trace: MoeRectStaticTrace
    capacity_policy: exact_peak | fixed
    token_drop: false
    combine_dtype: fp32
    expert_dtype: fp16
    expert_home_rank[e] = e

首版一 Die 一个 expert，最大程度复用 EP4 语义。1×1 是合法 E1 退化 case。

### 6.2 Static trace

每个 assignment 必须记录：

    token_index
    source_rank
    expert_index
    expert_home_rank
    slot_index
    gate_weight

校验：

- 每 token 恰有 top-k assignments；
- expert/slot 唯一；
- slot 小于 capacity；
- source/home rank 属于 Mesh；
- top-k=1 时 gate weight 仍显式存在；
- combine contributor count 精确；
- balanced、skewed、hot expert、empty expert 均可表示。

### 6.3 Personalized A2A baseline

先按 `(source_rank, destination_rank)` 聚合 token/expert payload，避免每 token
产生独立 DTE action。

每个 stage 最多存在 `R(R-1)` 个 remote pair packet。采用 cyclic-delta waves：

    wave δ: rank i sends to (i+δ) mod R

不存在 payload 的 pair 跳过 action，但保留 wave index。每 rank 每 wave 最多一次
send、一次 receive，session demand=2≤3。

Inference stages：

1. local gate；
2. pack dispatch；
3. remote dispatch waves；
4. local expert up/gate/down compute；
5. remote combine waves；
6. weighted FP32 combine；
7. output ownership/probe。

local assignment 不生成伪 DTE，使用 typed local view。

### 6.4 优化候选

- Direct-XY：所有完整/退化矩形的 baseline；
- Comet：只在 topology、pivot、route reversal 和 capacity witness 完整时生成；
- optional MeshSlice expert GEMM：后续 TP×EP 阶段；
- AUTO 无校准或无收益时回退 Direct-XY。

Comet 不能仅根据 `complete_rectangle=true` 入选；必须验证实际 packet、pivot、
route resource 和 endpoint wave。

### 6.5 Runtime marker 参数化

替换 2×2 假设：

- die coverage 为 `0..R-1`；
- expected directed physical links 使用
  `2[H(W-1)+W(H-1)]`；
- active route incidence 从 projection 派生，不要求每条物理 link 都有流量；
- session marker 覆盖所有 active cores；
- port/die overlap 按真实 H/W 聚合；
- marker schema 携带 mesh digest 和 workload digest。

## 7. MoE train 设计

### 7.1 首版训练语义

在 MoE inference 全链后增加：

1. 保存 dispatch assignment、slot、gate weight 和 expert activation tape；
2. output gradient 按相同 assignment 反向发送到 expert home；
3. expert down/up/gate data-gradient 和 weight-gradient；
4. expert input-gradient 返回 token source；
5. source rank 完成 weighted combine backward；
6. gate weight/logit gradient；
7. gate-gradient AllReduce；
8. expert/gate SGD；
9. persistent state writeback。

首版 `EP=R, DP=1`，每个 expert 只有一个 owner，因此 expert gradient 不需要
跨 rank AllReduce。gate 参数采用一个 typed owner 或全 rank replicated contract；
若 replicated，则使用 arbitrary-rank AR 后才能 optimizer update。

### 7.2 Backward route closure

每个 forward remote assignment 必须存在：

    forward dispatch: source → expert
    forward combine:  expert → source
    backward grad:    source → expert
    backward dx:      expert → source

四条 flow 均绑定真实 PairRoute；combine/backward 可以使用 reverse route，但必须
显式引用，不能根据 route ID 字符串推断。

### 7.3 State ownership

- expert parameters：expert home rank 独占；
- expert gradients：home rank FP32 accumulation；
- gate parameter：typed global owner 或 replicated group；
- routing tape：source 和 expert 两侧各自持有必要字段；
- optimizer state：与参数 owner 共置；
- state load/store 绑定真实 HBM address space。

expert optimizer 必须等待该 expert 的全部 token WGRAD；gate optimizer 必须等待
全部 gate-gradient contributor 和可能的 AR barrier。

### 7.4 后续 DP×EP

第二阶段允许：

    DP × EP = R

expert 在每个 DP replica 中复制。EP groups 负责 token dispatch/combine，DP groups
负责相同 expert 的 gradient AllReduce。只有完整、等尺寸 group 和 expert/DP
ownership 都可证明时才开放。

TP×EP、expert sharding 和 ZeRO 不属于首版。

## 8. 全规模 MeshSlice 设计

### 8.1 统一四种模式

MeshSlice 不再只表示“两个维度都大于 1”：

| Mesh | 模式 | 通信 |
|---|---|---|
| H>1,W>1 | `FULL_2D` | row lhs exchange + column rhs exchange |
| H=1,W>1 | `ROW_ONLY` | lhs row exchange；rhs local |
| H>1,W=1 | `COLUMN_ONLY` | lhs local；rhs column exchange |
| H=1,W=1 | `LOCAL` | compute-only，无 DTE |

逻辑 group 始终保留 `logical_shape=(H,W)`。尺寸为 1 的 axis 是 typed degenerate
axis，不应被删除或替换为 `None`。

### 8.2 通用 flow/action 公式

定义：

    R = H×W
    D = (H-1) + (W-1)
    S = slice_count

AG_GEMM panel exchange：

    row_flows    = S × R × (W-1)
    column_flows = S × R × (H-1)
    base_actions = S × R × [3D + 1]

公式对 1×1、1×N、N×1 同样成立。

RS/AR terminal collective 不再在 MeshSlice generator 中手写 all-pairs action，
而是引用通用 arbitrary-rank collective plan，并把 exact child/wave/record demand
计入 candidate preflight。

### 8.3 Session-safe waves

每 slice：

1. `W-1` 个 row cyclic-delta waves；
2. `H-1` 个 column cyclic-delta waves；
3. compute；
4. optional terminal collective。

生产 session capacity=3 时，row 和 column wave 不并发。每 rank 每 wave一次 send
和一次 receive，session demand=2。未来只有在硬件 profile 明确证明 capacity≥4
时才允许 row/column overlap。

### 8.4 Typed panel subview

每个 rank 的聚合 operand：

    lhs shape = [tile_m, tile_k]
    rhs shape = [tile_k, tile_n]

row source `j` 的 lhs subview：

    shape  = [tile_m, tile_k / W]
    offset = j × subview_bytes

column source `i` 的 rhs subview：

    shape  = [tile_k / H, tile_n]
    offset = i × subview_bytes

H 或 W 为 1 时公式自然得到完整 local view。所有 SEND/RECV relocation addend、
buffer span 和 MATMUL full view 必须闭合。

### 8.5 Slice search

合法 slice 必须同时满足：

- K、H、W、block floor 整除；
- message payload≥minimum transfer；
- input double buffer 能放入 SRAM；
- action/buffer/record/runtime-symbol/derived-byte/file-byte 预算；
- terminal collective wave 预算；
- transport tag 生命周期。

R>16 时首选 `slice=1`，再按收益和容量最多增加 1～3 个候选。禁止为大 R
枚举全部 divisor 后再构建 DAG。

### 8.6 MeshSlice fallback

typed fallback 原因至少包括：

    incompatible_axis_mapping
    nondivisible_tensor
    panel_payload_too_small
    action_budget
    buffer_budget
    sram_budget
    endpoint_session_budget
    terminal_collective_budget
    artifact_record_budget
    artifact_file_budget
    no_economic_benefit

AUTO 回退到普通 GEMM+AG/RS/AR；forced 模式在构建 artifact 前报错。

## 9. 统一容量与运行时设计

### 9.1 构建前规模预估

每个 candidate 在创建 action/value/buffer 对象前估算：

- rank、group、route 数；
- compute/transport/wait/barrier/action 数；
- fragment/core-stream/record/relocation/runtime-symbol 数；
- per-core endpoint sessions；
- per-core receive bytes；
- SRAM high-water；
- HBM parameter/gradient/optimizer/tape bytes；
- planner derived bytes；
- final artifact 上界。

估算必须与 materialized audit 精确相等，测试通过少 1 byte/action/record 的负例
证明 builder 没有被调用。

### 9.2 固定生产上限

继续保留：

    ranks <= 100
    sessions_per_core_per_wave <= 3
    symbolic_records <= 1,048,576
    artifact_file_bytes <= 64 MiB
    runtime_core_id <= U16_MAX

训练/MoE 新增：

- state count、state binding、HBM home range 上限；
- token assignment 和 expert capacity 上限；
- per-step transport tag 数；
- microbatch/layer 展开后的 action 上限。

### 9.3 有限展开与模板

首版只编译有限的一步，并允许 layer/microbatch 展开，只要不超过硬上限。
不要为了支持大模型立即修改 Program ABI 引入循环。

当真实代表模型反复触及 1M records 或 64 MiB 时，再单独设计：

- immutable plan template；
- repeat count；
- per-iteration state/transport epoch；
- finalizer/runtime 版本化支持。

模板复用不能成为 tiny canary 和首版任意 Mesh 支持的前置条件。

### 9.4 Runtime residual

每次运行结束必须检查：

    active_endpoints = 0
    active_sessions = 0
    outstanding_tags = 0
    incomplete_barriers = 0
    pending_state_writes = 0
    proto_wait_count = 0

训练还需检查 optimizer/state store 完成；MoE 还需检查 dispatch/combine/backward
packet 和 expert slot 全部消费。

## 10. Capability report

新增 `FlexibleMeshWorkloadCapabilityReport`，按 workload 分开报告：

    mesh_foundation
    baseline_plan
    optimized_plan
    lower_link
    program_io
    finalizer
    runtime
    repeatability
    performance_calibration
    functional_execution

每项状态：

    verified | fallback | not_measured | not_applicable | out_of_scope

报告必须包含：

- mesh/workload/trace/state digest；
- requested/selected mode；
- fallback reasons；
- ranks/groups/routes/waves/sessions；
- actions/records/file bytes；
- SRAM/HBM high-water；
- ProgramIO counts；
- runtime residual；
- artifact SHA、makespan 和 marker digest。

不得由 schema/candidate 测试自动把 runtime 标记为 verified。

## 11. 分阶段开发计划

### X0：冻结 v2 契约

工作项：

- `FlexibleMeshWorkloadSpec`；
- workload kind、axis mapping、train/MoE/MeshSlice spec；
- completion/capability/fallback enums；
- 旧 RectMesh、DP2/DP4、EP4 adapter；
- capacity schema。

门禁：

- 旧 stable IDs 不变；
- workload/schema roundtrip 稳定；
- 非法 Mesh、axis、trace、optimizer、state 在任何 pass 前 fail。

### X1：统一主 compiler 与 runtime provider

工作项：

- `compile_flexible_mesh_workload`；
- whole-workload region replacement；
- shared case/provider/finalizer/resolver/npusim runner；
- exact finalizer file-byte report；
- generic runtime markers。

门禁：

- Dense inference 的代表尺寸先通过新入口；
- NAIVE/AUTO 单 artifact；
- ordinary/fusion/standalone region 无重复、无缺失；
- 1×1、2×3、10×10 tiny baseline 真实运行。

### X2：MeshSlice 全 100 种 Mesh

工作项：

- 四种模式；
- degenerate axis sharding；
- row/column cyclic waves；
- AG_GEMM、GEMM_RS、GEMM_AR；
- slice search/preflight；
- train GEMM roles。

门禁：

- 100-shape candidate/lower-link sweep；
- flow/action/bytes/subview 公式闭合；
- 每波 sessions≤3；
- 代表尺寸 ProgramIO+npusim；
- no-candidate 时普通 GEMM baseline 可运行。

### X3：Dense train-forward 任意 Mesh

工作项：

- 参数化现有 train-forward placement/planning/projection/lowering/evidence；
- DP×TP group registry；
- 去除 DP2×TP2 runtime marker/evidence 假设；
- 接入主 compiler 和 ProgramIO。

门禁：

- 全部 100 种 Mesh 的 forward tiny canary；
- logical/rank FLOPs、collective bytes、state reads 闭合；
- 旧 DP2×TP2 artifact/evidence 保持。

### X4：Dense 完整 backward+SGD

工作项：

- v2 train schema 开放 backward/SGD；
- full backward graph；
- activation tape/state ABI；
- DP gradient sync；
- optimizer/state store；
- MeshSlice DATA_GRAD/WEIGHT_GRAD。

门禁：

- 单 step 有限 DAG；
- forward/backward FLOPs 和 gradient bytes 闭合；
- optimizer 严格等待 reduction；
- HBM state ownership 无别名冲突；
- 100-shape tiny runtime、代表尺寸重复两次稳定。

完成后：

    dense_train_rect_complete = true

### X5：MoE inference Direct-XY baseline

工作项：

- `FlexibleMoeSpec` 和 static trace；
- EP=R expert/state placement；
- pair bucket 和 cyclic waves；
- generic MoE IR2/ABI/workload linker adapter；
- parameterized markers/report。

门禁：

- all-local、balanced、skewed、hot/empty expert；
- assignment/slot/bytes/weighted combine 闭合；
- sessions≤3；
- 100-shape tiny canary；
- old EP4 result 可由 adapter 重建。

完成后：

    moe_infer_rect_complete = true

### X6：MoE optimized candidate

工作项：

- Direct/Comet topology admission 参数化；
- pivot/route/resource closure；
- whole-workload endpoint scheduling；
- cost/calibration 按 H/W 分桶；
- AUTO fallback。

门禁：

- linear/odd×odd/rectangle 正负例；
- forced 只证明执行，AUTO 才声明收益；
- 无校准时不选 optimized；
- baseline 始终保留。

### X7：MoE train

工作项：

- backward tape；
- reverse dispatch/combine；
- expert data/weight gradient；
- gate gradient和 AR；
- SGD/state store；
- 后续 DP×EP adapter。

门禁：

- 四向 route closure；
- 每 assignment forward/backward contributor 精确；
- expert/gate optimizer dependency 闭合；
- state bytes 和 ownership 闭合；
- 100-shape tiny canary、代表 balanced/skewed 重复运行。

完成后：

    moe_train_rect_complete = true

### X8：发布矩阵与性能

工作项：

- 400 个 release canary：100 Mesh × 4 workload family；
- representative full workload；
- NAIVE/AUTO；
- repeatability；
- planner latency/RSS；
- capability report 汇总。

门禁：

- finalizer/resolver/npusim exit=0；
- ProgramIO pass=1；
- runtime residual 全零；
- 相同输入 artifact SHA、makespan、marker digest 稳定；
- 未校准 case 不发布性能收益。

## 12. 建议代码改动

### 12.1 新增公共 schema

| 文件 | 内容 |
|---|---|
| `schema/flexible_mesh_workload.py` | workload envelope、kind、axis mapping、completion |
| `schema/flexible_mesh_capacity.py` | action/state/session/record/file preflight |
| `schema/flexible_train.py` | full-step、optimizer、gradient/state contract |
| `schema/flexible_moe.py` | arbitrary-R trace、expert placement、backward tape |
| `schema/flexible_mesh_report.py` | capability/runtime report |

### 12.2 公共 passes/compiler

| 文件 | 内容 |
|---|---|
| `compiler.py` | `compile_flexible_mesh_workload` 和 region replacement |
| `passes/group_registry.py` | FULL/ROW/COLUMN/DP/TP/EP group registry |
| `passes/flexible_mesh_capacity.py` | materialization 前统一预检 |
| `passes/program_io.py` | train/MoE state init/probe |

### 12.3 Dense train

优先扩展现有：

- `passes/train_forward.py`；
- `passes/placement.py::place_train_forward_ir0`；
- `passes/inter_die_plan.py::plan_train_forward`；
- `passes/project_to_ir2.py::project_train_forward`；
- `passes/intra_die_schedule.py::schedule_train_forward`；
- `passes/train_global_action.py`；
- `passes/train_lower_program.py`；
- `passes/train_link_program.py`。

新增 backward/optimizer pass，不在固定 Lite schema 上做机械扩维。

### 12.4 MoE

复用并参数化：

- `schema/swizzle_moe.py`；
- `passes/discover_moe_swizzle.py`；
- `passes/project_moe_scale_swizzle_ir2.py`；
- `passes/project_moe_swizzle_whole_workload.py`；
- `lowering/moe_swizzle_abi.py`；
- `lowering/moe_swizzle_workload_standard.py`；
- `lowering/moe_swizzle_workload_linker.py`。

冻结：

- `lite_moe_dp4*`；
- `swizzle_moe_scale.py` C0-C4；
- 旧 runtime evidence。

新增 adapter 从旧 EP4 truth 映射到通用 v2 carrier。

### 12.5 MeshSlice

扩展：

- `policies/swizzle/meshslice_2d.py`；
- `passes/meshslice_2d_placement.py`；
- `lowering/swizzle_meshslice_standard.py`；
- `schema/swizzle_operand_abi.py`；
- `lowering/swizzle_abi.py`；
- `lowering/swizzle_standard.py`。

建议把模块公共名逐步改为 `meshslice_rect`，但保留旧 import alias，避免一次性
重命名造成大范围 churn。

## 13. 测试矩阵

### 13.1 100-shape 静态 sweep

对每种 workload 枚举全部 H,W：

- Mesh/axis/group/rank/route digest；
- baseline availability；
- candidate scale estimate；
- lower/link core streams=R；
- ProgramIO rank/state coverage；
- capability 不越界。

MeshSlice 额外断言四种模式和公式。

### 13.2 代表 Mesh

    1×1
    1×2, 2×1
    1×10, 10×1
    2×2
    2×3, 3×2
    3×3
    2×10, 10×2
    5×6, 6×5
    9×9
    9×10, 10×9
    10×10

### 13.3 Dense train cases

- TP1/DP1 compute-only；
- DP-only、TP-only、DP×TP；
- forward-only compatibility；
- full backward+SGD；
- AG/RS/AR；
- forward/data-grad/weight-grad MeshSlice；
- zero remote traffic 的退化轴。

每个 Mesh 选择可整除的 tiny model，不用一个固定模型强行覆盖所有 TP。性能对比
另用固定模型和兼容的 Mesh 子集。

### 13.4 MoE cases

- all-local；
- balanced；
- skewed；
- single hot expert；
- empty experts；
- 每 token source 不同；
- forward inference；
- full backward+SGD；
- Direct forced、AUTO、可用时 Comet forced。

### 13.5 MeshSlice cases

- LOCAL：1×1；
- ROW_ONLY：1×2、1×10；
- COLUMN_ONLY：2×1、10×1；
- FULL_2D：所有代表二维 Mesh；
- AG_GEMM、GEMM_RS、GEMM_AR；
- slice=1、2、4 和预算回退；
- operand subview/addend；
- session waves；
- train three GEMM roles。

### 13.6 负例

- H/W 为 0 或 11、R>100；
- placement 洞、重复、越界、转置错误；
- DP×TP、EP 与 R 不闭合；
- model/token/K 不整除；
- backward 缺 forward tape；
- optimizer 早于 gradient sync；
- gradient/state alias 冲突；
- top-k>expert_count；
- assignment 缺失、重复、slot 越界；
- expert capacity overflow/token drop；
- reverse MoE route 缺失；
- 每核第 4 个 active session；
- action/buffer/SRAM/HBM/record/file/tag 超限；
- forged trace/state/plan/artifact digest；
- functional execution 请求。

### 13.7 CI 分层

PR：

- 100-shape schema/topology/group/candidate sweep；
- 代表尺寸 lower/link；
- 旧 Dense/DP2/DP4/EP4/2×2 回归；
- capacity negatives。

Nightly：

- 代表尺寸四类 workload 的 finalizer/ProgramIO/npusim；
- balanced/skewed MoE；
- Dense/MoE train full step；
- NAIVE/AUTO、重复两次；
- latency/RSS。

Release：

- 400 个 tiny canary；
- 代表 full workload；
- artifact/marker/makespan repeatability；
- capability report；
- calibrated case 的性能报告。

## 14. 兼容、上线与回退

### 14.1 兼容要求

- 不修改旧 case/schema version/stable ID；
- 不修改旧 2×2 MeshSlice operand ABI；
- 不修改 C0-C4 MoE truth；
- 不把新 80k action budget覆盖所有旧 policy；
- 新路径显式 opt-in；
- v2 adapter 必须证明旧 case 的 work/bytes/routes/state 等价。

### 14.2 Fallback reason

统一至少包含：

    invalid_mesh
    invalid_axis_mapping
    incompatible_sharding
    unsupported_workload
    unsupported_optimizer
    invalid_moe_trace
    expert_capacity
    missing_route
    candidate_action_budget
    candidate_buffer_budget
    candidate_sram_budget
    state_hbm_budget
    runtime_session_budget
    transport_tag_budget
    artifact_record_budget
    artifact_file_budget
    uncalibrated
    no_economic_benefit

### 14.3 发布声明

发布报告必须分别写明：

- 哪些 workload 的全部 100 种 Mesh 已真实运行；
- 哪些只完成 candidate 或 lower/link；
- 哪些采用 baseline fallback；
- 哪些有性能校准；
- timing-only、functional=false；
- top-k、optimizer、microbatch 和 parallel-axis 的具体范围。

## 15. 风险与控制

| 风险 | 后果 | 控制 |
|---|---|---|
| 直接放宽旧 train/MoE schema | stable ID 和 golden 大面积漂移 | 新建 v2 carrier + adapter |
| 每 token 一个 DTE action | MoE action/record 爆炸 | pair bucket + packed payload |
| row/column 同波 | sessions=4>3 | 顺序 cyclic waves |
| train 全层/多 microbatch 展开 | 1M records/64MiB | tiny 首版、精确预检、后续模板 |
| optimizer 提前执行 | 状态错误 | typed gradient barrier/state deps |
| MoE backward 根据名字反推 | route/tape provenance 错误 | exact assignment/tape refs |
| 1D Mesh 被排除 MeshSlice | 不满足任意 Mesh | ROW_ONLY/COLUMN_ONLY/LOCAL |
| 只测 candidate | 误报可运行 | 六层 completion state |
| 大 R optimized 无收益 | 性能回退 | baseline first + AUTO fallback |
| runtime marker 固定 2×2 | 大 Mesh evidence 无效 | topology-derived marker schema |

## 16. 最短开发顺序

为了简化流程并尽快得到可运行结果，建议按以下 PR 顺序：

1. v2 workload/capability/capacity schema；
2. shared group registry、compiler、runtime provider；
3. MeshSlice 四模式和 1×N/N×1；
4. Dense train-forward 100-shape；
5. Dense backward+SGD；
6. MoE inference EP=R Direct-XY；
7. MoE train backward+SGD；
8. MeshSlice RS/AR/train roles 和 MoE optimized；
9. 400-case release matrix；
10. 多 microbatch、AdamW、DP×EP 等后续能力。

每个 PR 都必须保留可执行 baseline、独立 capability 状态和容量负例。不要等待
所有优化策略一起完成，也不要为缩短代码而合并 Dense、MoE 和 MeshSlice 的语义
carrier；只复用 topology、group、wave、ABI、capacity、artifact 和 runner 基础设施。

