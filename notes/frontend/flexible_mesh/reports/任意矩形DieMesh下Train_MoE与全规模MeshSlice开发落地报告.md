# 任意矩形 Die Mesh 下 Train、MoE 与全规模 MeshSlice 开发落地报告

## 1. 本轮结论

本轮按照《任意矩形 DieMesh 下 Train、MoE 与全规模 MeshSlice 扩展开发方案》
完成了首个加速开发切片：统一 v2 workload/capacity/group/compiler 契约、Dense Train
timing carrier、MoE inference/train timing carrier，以及 MeshSlice 四种尺寸模式。

支持范围统一为完整、连续、无洞的 `H×W` 矩形：

    1 <= H,W <= 10
    R = H*W <= 100
    one rank per Die
    row-major placement
    X-first XY route
    timing_execution = true
    functional_execution = false

本轮没有把 compile-only、typed carrier 或 forward-only lower/link 误报为完整 runtime。
`finalizer/resolver/npusim` 尚未完成的负载继续明确标记为 `NOT_MEASURED` 或
`OUT_OF_SCOPE`。

## 2. 已落地能力

### 2.1 统一 flexible workload 主干

新增：

- `schema/flexible_mesh_workload.py`：
  - `dense_infer`、`dense_train`、`moe_infer`、`moe_train`；
  - canonical DP/TP/EP axis mapping；
  - SGD/单 microbatch timing contract；
  - MeshSlice AG_GEMM/GEMM_RS/GEMM_AR 请求契约；
  - schema JSON roundtrip。
- `schema/flexible_mesh_groups.py`：
  - FULL、ROW、COLUMN、DP、TP、EP canonical groups；
  - stable group IDs；
  - transposed Dense mapping；
  - serde 结构校验和 Mesh 关联校验分层。
- `schema/flexible_mesh_capacity.py`：
  - rank/action/buffer/record/file/session/tag/state 统一预检；
  - production 100 ranks、3 sessions、1,048,576 records、64 MiB 上限。
- `flexible_mesh_compiler.py`：
  - `compile_flexible_mesh_workload()`；
  - cyclic one-send/one-receive waves；
  - workload stage 顺序；
  - capacity-safe typed timing plan；
  - workload capability report；
  - MoE 专项 executable plan 作为统一 compilation 的强制子计划。

统一 400 组合编译矩阵已通过：

    100 Mesh * (Dense infer + Dense train + MoE infer + MoE train)

该矩阵证明 schema、axis/group、wave、capacity 和 typed plan 闭合，不等价于 400 个
npusim runtime canary。

### 2.2 Dense Train

新增 `FlexibleDenseTrainSpec/Plan/ForwardCarrier` 和：

    build_flexible_dense_train_plan()
    materialize_flexible_dense_train_forward()

已经支持：

- 全部 100 种 Mesh；
- 默认 `DP=H, TP=W, PP=1`；
- 一个有限 step；
- 完整现有 train-forward IR0；
- exact reverse forward tape envelope；
- Lite CE backward、WGRAD、SGD stage contract 复用；
- 覆盖全部 forward persistent parameter shards；
- 每个参数派生 exact consumer/backward/WGRAD/FP32 gradient/DP owner；
- DP column cyclic gradient sync，sessions=2；
- optimizer 必须等待 gradient sync；
- parameter load/store 和稳定 provenance；
- 1×1 production forward placement 到 lower/link smoke。

当前边界：

- 已生成全参数 typed backward/WGRAD/SGD carrier，但不是 full-model backward
  ProgramArtifact；
- backward 尚未 lower/link；
- Dense train backward 尚无 ProgramIO/finalizer/npusim evidence；
- capability 中保持：

      full_model_backward_materialized = false
      backward_lower_link_status = OUT_OF_SCOPE
      runtime_status = NOT_MEASURED

### 2.3 MoE inference/train

新增 `FlexibleMoeSpec/StaticTrace/ExecutablePlan` 和：

    build_round_robin_flexible_moe_spec()
    compile_flexible_moe_baseline()
    adapt_ep4_scale_spec()

已经支持：

- 全部 100 种 Mesh；
- `EP=R`、每 Die 一个 expert、top-k=1；
- balanced、all-local、hot/empty expert static trace；
- `(source,destination)` pair bucket；
- Direct-XY route 和 cyclic delta waves；
- inference dispatch/combine；
- train backward gradient/backward dx 四向 route closure；
- expert DGRAD/WGRAD；
- gate WGRAD 和 arbitrary-rank AR contract；
- expert/gate SGD 与 state store dependency；
- state ownership 与 symbolic record/file capacity；
- frozen EP4/C0 adapter，不修改旧 schema/stable IDs。
- 1×1、1×2、2×2 inference/train 的公共 RecordOpcode、DTE endpoint、state ABI
  和 ProgramIO mapping plan。

当前边界：

- 已完成严格 typed executable carrier；
- `standard_mapping_verified=true`；
- `lower_link_verified=false`、`runtime_verified=false`；
- 未适配到 standard MoE lower/link；
- 没有 ProgramIO/finalizer/resolver/npusim evidence；
- 没有 calibrated AUTO/Comet 性能选择；
- top-k=2、DP×EP、TP×EP 仍未开放。

### 2.4 MeshSlice 全尺寸模式

production MeshSlice 新增：

| Mesh | 模式 | 结果 |
|---|---|---|
| H>1,W>1 | `FULL_2D` | row lhs + column rhs |
| H=1,W>1 | `ROW_ONLY` | row lhs + local rhs |
| H>1,W=1 | `COLUMN_ONLY` | local lhs + column rhs |
| H=1,W=1 | `LOCAL` | compute-only，无伪 DTE |

已经完成：

- 100-shape production estimator sweep：81/9/9/1；
- rectangular placement 1×1 到 10×10；
- degenerate local operand exact GEMM provenance；
- LOCAL one-participant、zero-payload collective；
- route-free LOCAL 显式 rank-to-Die placement validation；
- 1×1、1×3、3×1、2×3 standard lower/link + ProgramIO；
- 原 2×2 record/action/ABI 计数保持。

当前 RS/AR 边界：MeshSlice generator 的 REDUCE 仍只有一个 input ref，严格 operand
ABI 要求两读一写。因此本轮没有放宽 typed 校验；GEMM_RS/GEMM_AR 保持通用
UNFUSED executable baseline，其中 AR 使用通用 RS+AG waves。新增 typed fallback adapter，
固定报告 `selected_path=UNFUSED_FALLBACK` 和
`reason=STRICT_TWO_INPUT_REDUCE_ABI`；RS/AR × 1×1/1×3/3×1/2×3 共 8 个代表
组合已通过 lower/link+ProgramIO。单 rank 使用精确 `COMP -> LOCAL_COPY`，无伪 route。

## 3. 测试结果

本轮最终定向结果：

| 测试 | 结果 |
|---|---|
| flexible workload 400 compile matrix + serde/mode | 7 tests，PASS |
| Flexible MoE | 5 tests，PASS |
| Flexible Dense Train | 5 tests，PASS |
| Flexible Dense Train 第二切片 | 6 tests，PASS |
| Flexible MoE standard mapping | 4 tests，PASS |
| MeshSlice standard lower/link/ProgramIO | 9 tests，PASS |
| RectMesh schema/compiler | 11 tests，PASS |
| generic AG/RS/AR projection/lower/link | 20 tests，PASS |
| MeshSlice RS/AR strict fallback | 3 tests / 8 combinations，PASS |
| flexible Mesh provider/report | 6 tests，PASS |
| public API and schema `__all__` import closure | PASS |
| Python compileall | PASS |
| tracked `git diff --check` | PASS |

旧 `test_moe_swizzle_c1_selection` 仍有两个 frozen ID 断言漂移：

    selection: expected 43f..., actual 4eff...
    witness:   expected 847f..., actual b1f...

在 `/tmp` 导出的干净仓库 HEAD 上复现得到完全相同结果，因此不是本轮 flexible
workload/MoE/MeshSlice 改动引入。除两个 ID 外，该测试的 selected pair、frontier、
cycles、lifecycle counts 和算法参数均一致。本轮不修改该无关 golden。

## 4. 对原方案阶段的实际状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| X0 v2 契约 | 完成 | workload/axis/capacity/capability/adapter 基础完成 |
| X1 统一 compiler | 部分完成 | typed compiler 完成；whole-workload finalizer/runtime 未完成 |
| X2 MeshSlice | AG 完成 | 四模式 AG standard 完成；RS/AR 使用 UNFUSED fallback |
| X3 train-forward | 部分完成 | 100 plan + 1×1 production lower/link；非 100 runtime |
| X4 Dense backward+SGD | carrier 完成 | 全参数 typed carrier；backward lower-link 未完成 |
| X5 MoE Direct-XY | mapping 完成 | 200 plans + 代表 standard mapping；真实 lower/link/runtime 未完成 |
| X6 MoE optimized | 未完成 | 保持 Direct-XY baseline |
| X7 MoE train | carrier 完成 | 四向/state/SGD typed closure；runtime 未完成 |
| X8 发布矩阵 | compile 完成 | 400 compile canary，不是 400 runtime canary |

因此当前不能设置：

    dense_train_rect_complete = true
    moe_infer_rect_complete = true
    moe_train_rect_complete = true
    flexible_mesh_workloads_complete = true

下一开发切片应依次完成：

1. MoE typed plan 到 standard IR2/Core ABI/Operand ABI adapter；
2. MoE representative ProgramIO/finalizer/npusim；
3. Dense full-model backward action materialization；
4. Dense backward lower/link 与 state HBM ABI；
5. 代表 Mesh runtime 后再扩 100-shape runtime matrix；
6. 最后开放 MeshSlice native RS/AR 和 calibrated optimized selection。
