# 代表矩形 DieMesh 下 Train、MoE 与 MeshSlice 的 NpuSim Runtime 闭环开发落地报告

## 1. 本轮结论

本轮按照《代表矩形 DieMesh 下 Train、MoE 与 MeshSlice 的 NpuSim Runtime
闭环开发方案》，完成了第一组真实纵向闭环：

- MeshSlice 1×1 AG_GEMM、GEMM_RS fallback、GEMM_AR fallback；
- MoE 1×1 inference；
- MoE 1×1 train；
- Dense Train 1×1 complete-step artifact runtime smoke。

所有真实闭环均经过：

    typed plan
    -> LinkedProgramManifest
    -> ProgramIO
    -> C++ finalizer
    -> actual artifact SHA ProgramIO resolver
    -> npusim timing execution
    -> marker / residual / probe evidence

本轮保持 timing_execution=true、functional_execution=false。

MeshSlice 1×1 和 MoE 1×1 已具有真实 finalizer/resolver/npusim exit=0 evidence。
Dense 1×1 也真实运行成功，但 backward action 尚缺专属
IR1/projection/schedule/global exact quotient，因此只标记为 artifact_runtime_smoke，
不标记完整 runtime_verified。

本轮尚未覆盖 1×N、N×1 和二维代表 Mesh，因此不能设置任何
representative runtime 或 rect complete 总完成状态。

## 2. 公共 Runtime 基础

新增统一 runtime case、artifact capacity、stage status、marker、residual、evidence
和 typed stage failure schema。

runtime evidence 只有在以下条件同时满足时才允许 runtime_verified=true：

- schema、candidate、lower/link、ProgramIO、finalizer、resolver、npusim 阶段完整；
- manifest、ProgramIO、artifact、resolver 和 marker digest 闭合；
- finalizer/resolver/npusim exit code 为零；
- marker rank/core coverage 完整；
- ProgramIO probe 全部有效并通过；
- runtime residual 全零；
- timing=true、functional=false。

repeatability 不能绕过 runtime evidence。repeat_count=2 时必须有两个真实 marker。
marker parser 严格消费 router、D2D、collective、endpoint、token、event、
PROGRAM_MEMORY LSU/DTE residual 和 completed core coverage。

LOCAL/stateless MeshSlice 明确允许 active route 和 state 集为空，但仍要求 core、
ProgramIO、DONE 和 drain evidence 完整。

## 3. MeshSlice Runtime

### 3.1 1×1 三路径真实闭环

| 请求 | 路径 | Artifact SHA | records/reloc | makespan | repeatability |
|---|---|---|---|---|---|
| AG_GEMM | MeshSlice standard LOCAL | c78ba58fbb0e23f87566894060781765defc6c41070271a34068b2543be1fa0c | 8/14 | 186 | 186/186 |
| GEMM_RS | UNFUSED fallback | d488b58adbe295e0f36d012234eeb96f39da6c6995170af81b62540b61d1e3ad | 12/19 | 312 | 312/312 |
| GEMM_AR | UNFUSED fallback | ee84fb0ec06d8e0ba8ee7ed9642a9e09301037a0f946f8be12ece9eabaf53ecc | 12/19 | 125 | 125/125 |

三条路径均满足 finalizer/resolver/npusim exit=0、ProgramIO pass=1、
core/rank coverage=(0)、active routes 为空、六项 residual 全零，两次 marker digest
一致，runtime_verified=true、repeatability_verified=true。

RS/AR 保持 UNFUSED_FALLBACK 和 STRICT_TWO_INPUT_REDUCE_ABI，没有放宽 native
MeshSlice REDUCE 的两读一写约束。

### 3.2 Finalizer 和 1×3 阻断

新增显式 LOCAL 和 single-rank terminal quotient，使零通信 MeshSlice 合法通过，
同时保持旧通信 Swizzle 和 UNFUSED quotient 不变。旧 finalizer selftest 继续通过。

1×3 AG 当前 fail-closed 于：

    ABS relocation addend does not equal the dense view byte addend

根因是 legacy 32×48 K whole 2D dense root 与 linear3 使用的 rank-major 连续 K
chunk 不等价。安全修复需要 packed storage root、compute full-view alias、
per-DTE chunk alias 和 Core/Operand ABI binding closure。本轮没有放宽 addend 校验。

## 4. MoE Runtime

### 4.1 Strict Flexible MoE v2 lineage

新增专用版本化 lineage：

    FLEXIBLE_MOE_PLAN
      schema = wafer_frontend.flexible_moe_plan/v2alpha1

    FLEXIBLE_MOE_STANDARD_MAPPING
      schema = wafer_frontend.flexible_moe_standard_lowering_plan/v1alpha1

    COMMAND_FRAGMENT x 2
      kinds = STATE_IO + COARSE
      producer = flexible_moe_production_lowering

严格绑定 plan/mapping/source/schedule/producer/kind/ID，以及 plan、mapping、embedded
fragment canonical digest。旧 generic、Swizzle、MoE Swizzle 和 UNFUSED 分支保持
不变；缺 mapping、错误 schema、伪造 digest/source 均 fail closed。

### 4.2 Production action 和 state

Flexible MoE v2 已实现：

- 每个 parameter load/store 一 action 一 state ref；
- store 严格依赖对应 optimizer；
- allocation 绑定首次 consumer，free 绑定末次 consumer/store；
- compute 使用 canonical SRAM_BIND + single compute body；
- LOCAL_REDUCE 使用冻结 FP16/FP32/FP16/SUM ABI；
- gate/expert 使用各自精确 M/K/N 和 buffer extent；
- inference 使用 HBM retention witness，不声明 SRAM functional output；
- train 使用 HBM updated-state probes。

旧 EP4/C0-C4 schema 和 stable IDs 未修改。

### 4.3 1×1 inference evidence

    finalizer/resolver/npusim = 0/0/0
    artifact SHA = d74f7627b812c3137686e4a504bd60d7cb9a219622ca9f186831b13e9007bac3
    records/relocations = 16/30
    initializations/probes = 3/1
    makespan_cycles = 1538
    LSU issued/completed = 3/3
    probe valid/exact/pass = 1/1/1
    residual = 0

第二次 npusim makespan 仍为 1538。当前 evidence 可设置 runtime_verified=true，
但没有宣称 functional output correctness。

### 4.4 1×1 train evidence

    finalizer/resolver/npusim = 0/0/0
    artifact SHA = d713b79ae1b47ad493796b144a174790b96f9a2a72b918289f59504121ee805c
    records/relocations = 34/71
    initializations/probes = 3/2
    makespan_cycles = 1789
    LSU issued/completed = 4/4
    SGD_UPDATE = 2，elements = 1536 / 16
    HBM updated-state probes = 2/2 pass
    residual = 0

第二次 npusim makespan 仍为 1789。该闭环证明 timing execution、state lifecycle、
optimizer 顺序和 HBM writeback。WGRAD 当前仍是冻结 FP16 ISA timing surrogate，
不声明 gradient/SGD 数值 functional correctness。

### 4.5 当前 MoE 边界

- production adapter 仅支持 1×1；
- 1×2、2×1、2×2、2×3 在 DTE lowering 前 fail closed；
- personalized DTE endpoint、wave 和多 core runtime 尚未接入；
- full artifact SHA repeatability 尚未做两次 finalizer；
- top-k=2、DP×EP、TP×EP、Comet/AUTO 仍不在本阶段。

## 5. Dense Train Runtime Smoke

新增 full-parameter Dense backward production materializer：15 个 StateABI，
SRAM_ALLOC/BIND/FREE、parameter LOAD、backward/WGRAD MATMUL、SGD_UPDATE、
persistent STORE 和 15 initialization/probes。

    artifact SHA = 24a1f9d357a92dc5d561a9fc9aa13140924a50bd01147d2caedb5b4e83a4245d
    records = 202
    makespan_cycles = 3117
    LSU issued/completed = 30/30
    HBM read/write bytes = 808/808
    ProgramIO probes = 15/15 pass
    residual/drain/credit = 0

当前 production anchors 只覆盖 train forward，同源绑定不等于新增 backward action
的 exact quotient。因此 runtime_verified=false，状态仅为 artifact_runtime_smoke，
也不声明数值 SGD 正确性。

## 6. 回归结果

| 测试 | 结果 |
|---|---|
| runtime/Dense/MoE/400-matrix 单元回归 | 35 tests，PASS |
| runtime marker/harness/MeshSlice/UNFUSED 集成回归 | 32 tests，PASS |
| Flexible MoE production/standard/100-shape 专项 | 16 tests，PASS，已包含在单元组 |
| C++ finalizer selftest | PASS |
| finalizer/resolver/npusim build | PASS |
| public API/schema __all__ closure | PASS |

## 7. 对开发方案阶段的实际状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| R0 runtime provider/evidence | 最小闭环完成 | case/evidence/marker/residual/failure 已落地 |
| R1 MeshSlice | 1×1 完成 | AG/RS/AR runtime+repeatability；1×3 storage 阻断 |
| R2 MoE inference | 1×1 完成 | strict v2 lineage + real runtime |
| R3 MoE train | 1×1 完成 | SGD/state-store real runtime |
| R4 Dense Train | runtime smoke | npusim 成功，但 backward exact lineage 未完成 |
| R5 代表矩阵 | 未完成 | 1D/2D/大 Mesh 尚未运行 |

因此所有 representative runtime complete 和 rect complete 总状态继续保持 false。

## 8. 下一切片

1. 修复 MeshSlice packed root/alias，闭环 1×3 和 3×1；
2. 为 Flexible MoE 增加 arbitrary-R DTE endpoint/route/wave lowering；
3. 闭环 MoE 1×2、2×1、2×2、2×3；
4. 新增 Dense backward 专属 IR1/projection/schedule/global lineage；
5. 在 exact Dense lineage 下重跑 1×1，再增加 DP/TP transport；
6. 扩 nightly 代表 Mesh，最后进入 100-shape runtime 发布矩阵。
