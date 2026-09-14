# 任意矩形 DieMesh 全负载 NpuSim Runtime 闭环与 100-Shape 发布开放计划

## 0. 2026-08-24 验证范围修订（本轮生效）

为简化收口和缩短开发周期，用户批准本轮不再执行 100 个矩形×6 个 family×2 次的
全量 NpuSim 验证。本修订在本轮验收中优先于后文“每个 Mesh 都运行 runtime”的原始
要求，但不删除原始 exhaustive 目标，便于后续需要时继续扩展。

本轮固定为：

    validation_scope = "representative"
    exhaustive_runtime = false
    tested_meshes = [
      "1x1", "1x4", "4x1", "2x2",
      "2x3", "3x2", "3x3", "10x10",
    ]

验收矩阵分两层：

1. **100-shape 静态契约层**：对 `{1..10}×{1..10}` 完成 schema、axis/group、
   placement/route、case identity、执行模式、capacity 与 deterministic compile
   覆盖；不为每个 shape 构造巨型 production artifact，也不声称这 100 个 shape
   都已实际运行 NpuSim。
2. **代表性 runtime 层**：上述 8 个 Mesh 精确覆盖 Dense Train、MoE
   inference/train、MeshSlice AG/RS/AR 六类 family，共 48 cases；每个 case
   独立运行两次 finalizer/resolver/NpuSim，共 96 executions。

完成状态仍禁止手填，必须由代表性 evidence matrix 派生。当 100-shape 静态契约
门禁和 48-case/96-execution runtime 矩阵全部通过时，本轮允许派生：

    dense_train_rect_complete = true
    moe_infer_rect_complete = true
    moe_train_rect_complete = true
    meshslice_all_rect_complete = true
    workload_contract_complete = true
    flexible_mesh_workloads_complete = true
    exhaustive_rect_runtime_complete = false

上述 `*_rect_complete=true` 的精确含义是“任意 1..10 矩形的静态契约已覆盖，并且
冻结的 8 个代表矩形已完成真实 runtime 与可重复性闭环”。它不等于“100 个矩形全部
运行过 NpuSim”；后者只能由原 600-case/1,200-execution exhaustive matrix 派生，
本轮必须保持 `exhaustive_rect_runtime_complete=false`。

## 1. 目标

本计划承接以下已完成基础：

- 1≤H,W≤10、R=H×W≤100 的完整矩形 Mesh schema、placement、route 和 group registry；
- Dense Train、MoE inference、MoE train 和 MeshSlice typed executable carrier；
- MeshSlice 1×1 AG/RS/AR 的真实 finalizer/resolver/npusim 闭环；
- MoE 1×1 inference/train 的 strict v2 lineage 和真实 npusim 闭环；
- Dense Train 1×1 complete-step artifact runtime smoke；
- 400 组合 typed compile matrix；
- 公共 runtime case、capacity、marker、residual 和 evidence schema。

本计划完成后，必须能够基于真实 evidence 自动设置：

    dense_train_rect_complete = true
    moe_infer_rect_complete = true
    moe_train_rect_complete = true
    flexible_mesh_workloads_complete = true

完成状态限定在：

    timing_execution = true
    functional_execution = false

即证明真实 action、通信、状态、资源生命周期、finalizer、resolver 和 npusim timing
执行闭合，不声明模型输出、梯度或参数更新具备数值 functional correctness。

## 2. 支持范围

物理 Mesh：

    1 <= H,W <= 10
    R = H*W <= 100
    完整、连续、无洞矩形
    one rank per Die
    row-major rank
    X-first XY route

负载：

- Dense Train：PP=1、默认 DP=H/TP=W、单 microbatch、完整有限
  forward/backward/WGRAD/DP synchronization/SGD/state store；
- MoE inference：EP=R、每 Die 一个 expert、top-k=1、static trace、
  Direct-XY baseline；
- MoE train：同一 EP=R 布局，四向 route、expert/gate gradient、gate AR、
  SGD/state store；
- MeshSlice：LOCAL、ROW_ONLY、COLUMN_ONLY、FULL_2D，覆盖 AG_GEMM、
  GEMM_RS、GEMM_AR。

不属于本计划完成前置条件：

- functional/numerical correctness；
- AdamW、ZeRO、PP>1、recompute、多 microbatch；
- MoE top-k=2、DP×EP、TP×EP；
- Comet/AUTO 性能收益；
- native MeshSlice RS/AR；
- 缺 Die、故障 Mesh、torus 或动态绕行。

## 3. 最终完成定义

### 3.1 合法 Mesh 集

定义：

    RECT_MESH_SET = {(H,W) | 1<=H<=10, 1<=W<=10}
    len(RECT_MESH_SET) = 100

每个完成状态必须由 100 个真实、唯一、可追溯的 Mesh evidence 行派生，不允许手工
写布尔值，也不允许由 compile/candidate/lower-link 成功自动推导。

### 3.2 单 case runtime 完成

一个 case 只有同时满足以下条件，才允许 runtime_verified=true：

1. workload/schema/trace/state 校验成功；
2. candidate/baseline 计划成功；
3. materialized action、state、route 和 dependency 完整；
4. standard lower/link 生成一个 LinkedProgramManifest；
5. ProgramIO 对全部 input/state/probe 精确绑定；
6. C++ finalizer exit=0；
7. finalizer exact records/relocations/symbols/file bytes 未超限；
8. actual artifact SHA 回填 ProgramIO；
9. resolver exit=0；
10. npusim timing execution exit=0；
11. ProgramIO verify pass=1；
12. rank/core coverage 与 Mesh 一致；
13. workload-specific completion marker 完整；
14. runtime residual 全零；
15. evidence digest 与 spec/plan/manifest/ProgramIO/artifact/marker 一致。

必须检查的通用 residual：

    active_endpoints = 0
    active_sessions = 0
    outstanding_tags = 0
    incomplete_barriers = 0
    pending_state_writes = 0
    proto_wait_count = 0
    lsu_residual = 0
    dte_residual = 0
    router_residual = 0
    credit_residual = 0

### 3.3 Repeatability

每个 release case 执行：

- 两次独立 materialize/finalizer；
- 两次 resolver；
- 两次 npusim；
- 比较 manifest digest、artifact SHA、file bytes、ProgramIO digest、resolver digest、
  makespan、marker digest 和 residual。

任何非版本化、未解释的漂移都使 repeatability_verified=false，并阻止对应 rect
completion。

### 3.4 Completion 公式

    dense_train_rect_complete
      = 对全部 100 Mesh：
          Dense Train runtime_verified
          && repeatability_verified
          && complete-step state closure

    moe_infer_rect_complete
      = 对全部 100 Mesh：
          MoE inference runtime_verified
          && repeatability_verified
          && dispatch/combine closure

    moe_train_rect_complete
      = 对全部 100 Mesh：
          MoE train runtime_verified
          && repeatability_verified
          && four-way route/gradient/optimizer/state closure

    meshslice_all_rect_complete
      = 对全部 100 Mesh：
          AG_GEMM runtime_verified
          && GEMM_RS baseline runtime_verified
          && GEMM_AR baseline runtime_verified
          && repeatability_verified

    workload_contract_complete
      = schema/axis/group/capacity/fallback/runtime evidence 契约通过
        && 旧 stable IDs 无回归

    flexible_mesh_workloads_complete
      = workload_contract_complete
        && meshslice_all_rect_complete
        && dense_train_rect_complete
        && moe_infer_rect_complete
        && moe_train_rect_complete

只有上述公式由 evidence matrix 计算为真时，才允许输出四个目标状态为 true。

## 4. 当前状态

| 能力 | 当前状态 | 主要缺口 |
|---|---|---|
| Workload contract | 基础完成 | completion derivation 和 100-shape report 待完成 |
| MeshSlice 1×1 | runtime+repeatability verified | 1D/2D packed view 未闭环 |
| MoE 1×1 infer/train | runtime verified | arbitrary-R DTE lowering 未完成 |
| Dense Train 1×1 | artifact runtime smoke | backward exact lineage 未完成 |
| 100-shape runtime | 未开始 | runner sharding、容量和长时任务治理 |

当前所有 rect/flexible 总完成状态必须保持 false。

## 5. 总体实施架构

    FlexibleMeshWorkloadSpec
      -> tiny release-case generator
      -> workload-specific typed plan
      -> exact IR/action/state/route materialization
      -> standard ABI/lower/link
      -> one LinkedProgramManifest
      -> ProgramIO
      -> capacity preflight
      -> finalizer
      -> exact capacity audit
      -> actual-SHA ProgramIO resolver
      -> npusim
      -> typed marker/residual/evidence
      -> repeatability comparison
      -> 100-shape matrix
      -> derived completion states

原则：

- baseline 始终可用，优化不阻塞发布；
- 每个 request/training step 只生成一个 artifact；
- frozen 旧 schema 和 stable IDs 不修改；
- finalizer/marker 增量使用 producer-scoped exact quotient，不放宽 generic 校验；
- 任何 optimized path 失败均回退 baseline；
- 任何 evidence 缺失均 fail closed。

## 6. Workstream A：MeshSlice 全 100 Mesh

### A1：Packed storage root

修复 1×N/N×1/二维 Mesh 的 panel 存储表达：

- 为通信 payload 建立 rank-major packed storage root；
- 建立 compute full-view alias；
- 建立每个 DTE packet 的 typed chunk alias；
- relocation addend 必须等于 TensorSlice dense byte addend；
- Core ABI、Operand ABI、BufferABI 和 address binding 使用同一 alias graph；
- local panel 不生成伪 DTE。

首批门禁：

    1×3 AG_GEMM
    3×1 AG_GEMM
    2×3 AG_GEMM

均完成 finalizer/resolver/npusim 和零 residual。

### A2：三种 operation

- AG_GEMM 使用 production MeshSlice standard；
- GEMM_RS 使用 generic UNFUSED fallback；
- GEMM_AR 使用 generic UNFUSED RS+AG fallback；
- fallback report 保持 STRICT_TWO_INPUT_REDUCE_ABI；
- native RS/AR 不是发布门禁。

### A3：100-shape sweep

对每个 Mesh 运行 AG/RS/AR，共 300 个 primary runtime cases，每个 case 两次。

必须验证：

- execution mode 为 LOCAL/ROW_ONLY/COLUMN_ONLY/FULL_2D；
- rank/core coverage=R；
- active route 与真实 panel flow 一致；
- 每 core 每 wave sessions≤3；
- payload、subview、addend 和 buffer span 闭合；
- LOCAL 零 route/endpoint；
- runtime residual 全零。

完成后设置 meshslice_all_rect_complete=true。

## 7. Workstream B：MoE Arbitrary-R Runtime

### B1：Production DTE adapter

扩展 FlexibleMoe production adapter 支持 R>1：

- 每个 pair bucket 物化 DTE SEND/RECV/WAIT；
- endpoint source/destination/runtime core 精确；
- transport tag 由 workload/stage/wave/pair 稳定派生；
- PairRoute 使用 typed X-first path；
- cyclic delta wave 每 rank 每波最多一 send、一 receive；
- local assignment 使用 local view，不生成伪 transport；
- fragment action coverage 与 typed plan exact；
- rank stream 保持 wave/barrier/dependency 顺序。

首批门禁：

    1×2
    2×1
    2×2
    2×3
    3×2

### B2：Inference runtime

单 artifact 顺序：

    state/input load
    -> gate
    -> pack
    -> dispatch waves
    -> expert compute
    -> combine waves
    -> weighted combine
    -> HBM retention/output timing probe

100-shape 基准 trace：

- 1×1 使用 all-local E1；
- R>1 使用确定性 balanced remote trace，保证真实 DTE；
- representative Mesh 额外运行 all-local、single-hot 和 empty-expert trace。

门禁：

- assignment/slot/pair packet contributor 精确；
- dispatch/combine route 互逆且显式；
- active endpoint/session/tag 全部 drain；
- ProgramIO probe 有效；
- 两次 artifact/runtime 稳定。

完成后设置 moe_infer_rect_complete=true。

### B3：Train runtime

在 inference 单 artifact 中加入：

    backward gradient source->expert
    -> expert DGRAD/WGRAD
    -> backward dx expert->source
    -> combine backward
    -> gate WGRAD
    -> arbitrary-R gate AR
    -> expert/gate SGD
    -> persistent state store

要求：

- 每 remote assignment 四向 route closure；
- expert optimizer 等待该 expert 全部 WGRAD；
- gate optimizer 等待 gate AR；
- R=1 gate AR 为 typed no-transport 退化；
- R>1 gate AR 使用通用 session-safe collective；
- HBM updated-state probe 覆盖全部可写参数；
- residual 增加 packet/slot/gradient/state completion。

完成 100 Mesh 两次 runtime 后设置 moe_train_rect_complete=true。

## 8. Workstream C：Dense Train Exact Runtime

### C1：Backward exact lineage

新增 Dense backward 专属、版本化：

- IR1；
- projection；
- schedule；
- global action DAG；
- lowered fragments；
- linked manifest top input。

lineage 必须逐 action 见证：

- exact forward consumer；
- backward tape origin；
- DGRAD/WGRAD；
- DP collective；
- SGD；
- state store。

禁止复用只覆盖 forward 的 trust anchor 后替换 leaf digest。C++ finalizer 使用
producer-scoped exact kind/schema/cardinality/source/fragment closure。

完成后重新运行 1×1；只有 exact lineage、ProgramIO、finalizer、resolver、npusim
全部通过，才把现有 artifact_runtime_smoke 升级为 runtime_verified。

### C2：DP/TP transport

默认映射：

    DP = H
    TP = W
    PP = 1

- TP group 承担 forward/backward AG/RS/AR；
- DP group 承担每个 parameter WGRAD AllReduce；
- DP1/TP1 使用 typed no-transport 退化；
- 每 state 的 reduction 完成后才允许 SGD；
- 全部 state store 等待对应 optimizer；
- cyclic waves 保证 sessions≤3。

首批门禁：

    1×2 TP-only
    2×1 DP-only
    2×2 DP×TP
    2×3
    3×2

### C3：100-shape tiny model

每个 Mesh 生成确定性、可整除、非零 timing model：

- hidden/intermediate/token 维度按 TP 选择合法倍数；
- global batch 与 DP 闭合；
- 所有 parameter shard 大小非零；
- GroupGEMM/collective timing 大于零；
- model/trace 足够小，不超过 artifact 容量；
- 不用一个固定模型强行覆盖所有 TP。

每个 Mesh 验证：

- 全参数 StateABI coverage；
- activation/tape provenance；
- gradient bytes/owners；
- DP collective bytes；
- optimizer dependency；
- HBM read/write 和 state completion；
- repeatability。

完成后设置 dense_train_rect_complete=true。

## 9. Workstream D：公共 Runner、容量和 Evidence

### D1：统一 release case

新增 100-shape release case generator，case ID 由以下字段稳定派生：

    mesh digest
    workload kind
    trace/model digest
    selected baseline
    operation
    runtime profile version

禁止 case 名称猜测 Mesh、workload 或 route。

### D2：工具链绑定

evidence 增加：

- finalizer binary path/version/SHA；
- resolver binary path/version/SHA；
- npusim binary path/version/SHA；
- hardware/simulation config digest；
- environment/runtime profile version。

工具不在 allowlist 或 SHA 漂移时，旧 evidence 自动失效，避免把替换执行器产生的
输出当作同一发布证据。

### D3：容量

固定上限：

    ranks <= 100
    sessions_per_core_per_wave <= 3
    records <= 1,048,576
    artifact_file_bytes <= 64 MiB
    runtime_core_id <= U16_MAX
    transport_tags <= 65,535

构建前 symbolic preflight，finalizer 后 exact audit。任何超限：

- baseline case fail closed；
- 不临时提高生产上限；
- 优先减小 tiny model/trace；
- 若最小合法 case 仍超限，单独设计模板/循环 ABI，并保持 completion=false。

### D4：Completion derivation

新增只读 completion builder：

    derive_flexible_mesh_completion(evidence_matrix)

要求：

- exact 100 Mesh coverage；
- 无重复/缺失/未知 case；
- 每 case runtime/repeatability verified；
- workload-specific residual/status 完整；
- report ID 由完整 evidence matrix 派生；
- 手工 replace true、缺 evidence 或 forged digest 全部拒绝。

## 10. 100-Shape Runtime 发布矩阵

Primary matrix：

| Family | Mesh 数 | 每 Mesh case | 首次运行数 | 含 repeat 总运行数 |
|---|---:|---:|---:|---:|
| Dense Train | 100 | 1 | 100 | 200 |
| MoE inference | 100 | 1 | 100 | 200 |
| MoE train | 100 | 1 | 100 | 200 |
| MeshSlice AG | 100 | 1 | 100 | 200 |
| MeshSlice RS fallback | 100 | 1 | 100 | 200 |
| MeshSlice AR fallback | 100 | 1 | 100 | 200 |
| 合计 | 100 | 6 | 600 | 1200 |

Representative supplemental matrix：

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

补充 trace/mode：

- MoE all-local/balanced/hot/empty；
- Dense DP-only/TP-only/DP×TP；
- MeshSlice 四种 mode；
- capacity 边界；
- forced baseline 和 AUTO fallback correctness。

## 11. CI 和执行加速

### PR 层

- schema/serde/fail-closed；
- 400 typed compile matrix；
- 1×1、1×2、2×1、2×2 runtime smoke；
- C++ finalizer selftest；
- 不运行完整 1200 次 release matrix。

### Nightly 层

- representative Mesh 全负载；
- 两次 repeatability；
- sanitizer/marker/finalizer negative cases；
- 失败保留 artifact、ProgramIO、stdout/stderr 和 evidence。

### Release 层

- 1200 次 primary runtime；
- 按 Mesh/workload shard 并行；
- 单 shard 内串行运行同一 artifact 的 repeat，避免资源竞争；
- 汇总前校验工具 SHA 和 config digest；
- 任一 case 失败则总完成状态保持 false。

加速原则：

1. MeshSlice、MoE、Dense 三条实现链在公共 evidence schema 冻结后并行；
2. 每条链先 1D，再小二维，再大二维；
3. 优先修 baseline，不并行开发非必要优化；
4. artifact build 可并行，npusim 按资源 shard；
5. 缓存只复用输入完全相同且 SHA/version 匹配的 immutable artifact；
6. 禁止复用旧 runtime stdout 伪装新 case 执行。

## 12. 分阶段开放任务

### F0：冻结 completion/evidence v2

状态：OPEN

交付：

- completion schema 和 derivation；
- tool/config SHA binding；
- release case generator；
- forged/missing/duplicate evidence 负例。

门禁：不能手工设置目标 true。

### F1：MeshSlice 1D/2D

状态：OPEN

交付：

- packed root/full alias/chunk alias；
- 1×3、3×1、2×3 三 operation runtime；
- 100 Mesh AG/RS/AR matrix。

完成：meshslice_all_rect_complete=true。

### F2：MoE multi-die inference

状态：OPEN

交付：

- arbitrary-R DTE/endpoint/tag/wave；
- 代表 Mesh；
- 100 Mesh inference runtime/repeatability。

完成：moe_infer_rect_complete=true。

### F3：MoE multi-die train

状态：OPEN

交付：

- four-way backward transport；
- gate AR；
- SGD/state completion；
- 100 Mesh train runtime/repeatability。

完成：moe_train_rect_complete=true。

### F4：Dense backward exact lineage

状态：OPEN

交付：

- backward IR1/projection/schedule/global DAG；
- producer-scoped finalizer；
- 1×1 从 smoke 升级 verified。

### F5：Dense DP/TP 与 100 Mesh

状态：OPEN

交付：

- DP/TP communication；
- 100 Mesh tiny model/runtime/repeatability。

完成：dense_train_rect_complete=true。

### F6：全矩阵发布

状态：OPEN

交付：

- 600 primary cases；
- 1200 executions；
- supplemental representative matrix；
- capacity/repeatability/failure report。

### F7：最终状态派生

状态：OPEN

仅当所有门禁通过：

    dense_train_rect_complete = true
    moe_infer_rect_complete = true
    moe_train_rect_complete = true
    meshslice_all_rect_complete = true
    workload_contract_complete = true
    flexible_mesh_workloads_complete = true

## 13. 负例门禁

必须覆盖：

- Mesh H/W 越界、placement 洞/重复/转置错误；
- DP×TP、EP 与 R 不闭合；
- tensor/model/token 不整除；
- packed alias/addend/span 错误；
- action 缺 tape、state 或 exact lineage；
- optimizer 早于 gradient sync；
- state ownership/alias/HBM 越界；
- DTE endpoint/route/tag/wave 不一致；
- 每 core 第 4 个 session；
- MoE assignment/slot/four-way route 缺失；
- packet、gradient、barrier、state residual 非零；
- artifact record/file-byte 超限；
- ProgramIO 缺 input/state/probe；
- finalizer/resolver/npusim 非零退出；
- tool/config SHA 漂移；
- forged manifest/marker/evidence digest；
- 重复运行 artifact/makespan/marker 漂移；
- functional_execution=true 请求。

所有负例必须在准确阶段 fail closed，且不能生成 completion=true。

## 14. 风险与处理

### 14.1 MeshSlice packed layout

风险：root/alias 修改可能破坏旧 2×2 ABI。

处理：新增并列 packed storage carrier；旧 2×2 stable path 保留，通过 adapter 比较
action/bytes/ProgramIO 等价性。

### 14.2 MoE R² pair 数

风险：R=100 时 pair bucket、records 和 tags 放大。

处理：只为有 payload 的 pair 物化 action；wave carrier 保留 delta index；构建前
估算；tiny balanced trace 控制每 pair payload；不逐 token 创建 DTE action。

### 14.3 Dense full-model artifact

风险：全参数 state/action 接近 1M records 或 64 MiB。

处理：release tiny model 按 Mesh 参数化；若最小合法模型仍超限，单独设计
Program ABI template/repeat，不提高硬上限。

### 14.4 NpuSim 长时运行

风险：1200 次执行耗时。

处理：release sharding、固定资源配额、失败快速终止、保留可复现输入；超时视为失败，
不能用 compile-only 代替 runtime。

### 14.5 Timing 与 functional 混淆

所有报告必须携带 timing=true、functional=false。完成状态名称的 scope 固定为
timing-v1；后续 functional correctness 使用新状态和新 schema，不复用本计划布尔值。

## 15. 建议代码改动

新增：

| 文件 | 内容 |
|---|---|
| schema/flexible_mesh_completion.py | completion matrix 和自动派生 |
| schema/flexible_mesh_release.py | 100-shape case/profile/tool binding |
| passes/meshslice_packed_storage.py | packed root 和 typed alias |
| lowering/flexible_moe_transport.py | arbitrary-R DTE lowering |
| schema/flexible_dense_backward_ir.py | Dense backward exact lineage |
| passes/flexible_dense_backward_projection.py | projection/schedule/global DAG |
| integration/run_flexible_mesh_release.py | sharded release runner |
| integration/flexible_mesh_release_report.py | evidence 汇总和 completion |

优先扩展：

- flexible_mesh_runtime_provider.py；
- flexible_mesh_runtime_markers.py；
- flexible_mesh_runtime_meshslice.py；
- flexible_moe_production.py；
- flexible_dense_backward.py；
- program_finalizer.cpp；
- ProgramIO pass 和 resolver marker。

实际文件可按现有模块边界调整，但不得混合 schema、lowering、runner 和 evidence 职责。

## 16. 最终交付物

1. MeshSlice 300 cases、600 executions 的 evidence；
2. Dense Train 100 cases、200 executions 的 evidence；
3. MoE inference 100 cases、200 executions 的 evidence；
4. MoE train 100 cases、200 executions 的 evidence；
5. representative supplemental evidence；
6. tool/config SHA 和 exact capacity report；
7. failed-case 空清单；
8. 100 Mesh coverage audit；
9. repeatability audit；
10. 自动派生的 completion report；
11. 旧 frozen schema/stable ID 回归报告；
12. 最终开发落地报告。

最终验收必须展示 completion builder 从完整 evidence matrix 派生：

    dense_train_rect_complete = true
    moe_infer_rect_complete = true
    moe_train_rect_complete = true
    flexible_mesh_workloads_complete = true

缺少任意 Mesh、任意必备 workload/operation、任意第二次运行或任意真实工具阶段时，
计划保持 OPEN，相关状态保持 false。
