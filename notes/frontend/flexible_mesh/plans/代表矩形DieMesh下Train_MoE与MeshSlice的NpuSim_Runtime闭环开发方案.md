# 代表矩形 DieMesh 下 Train、MoE 与 MeshSlice 的 NpuSim Runtime 闭环开发方案

## 1. 阶段目标

本阶段承接《任意矩形 DieMesh 下 Train、MoE 与全规模 MeshSlice 扩展开发方案》
已经完成的 typed workload、capacity、group registry、Dense Train carrier、MoE
Direct-XY carrier 和 MeshSlice 全尺寸编排能力，集中补齐真实执行链：

```text
FlexibleMeshWorkloadSpec
  -> workload-specific executable plan
  -> standard IR2 / Core ABI / Operand ABI
  -> lower
  -> link，生成单一 LinkedProgramManifest
  -> ProgramIO
  -> finalizer，生成最终 artifact
  -> resolver
  -> npusim timing execution
  -> runtime evidence / residual / repeatability report
```

本阶段的首要交付是代表 Mesh 上的真实 runtime 闭环，不继续扩展新的并行语义或
优化候选。完成后，应能够对 Dense Train、MoE inference、MoE train 和 MeshSlice
分别设置代表尺寸的 `runtime_verified=true`。

本阶段仍保持：

```text
1 <= H,W <= 10
R = H*W <= 100
完整、连续、无洞矩形
one rank per Die
row-major placement
X-first XY route
timing_execution = true
functional_execution = false
```

本阶段不直接宣称全部 100 种 Mesh 已完成 runtime。代表 Mesh 闭环通过后，再进入
下一阶段的 100-shape runtime 扩展和发布矩阵。

## 2. 完成定义

### 2.1 什么是 runtime 闭环

只有同一个 workload 从输入 spec 经过完整生产链并由 npusim 成功执行，才能标记
为 runtime 闭环。以下单项均不能独立代表闭环：

- schema roundtrip 成功；
- candidate 或 typed executable plan 构建成功；
- symbolic action、wave、route、state 数量闭合；
- lower/link 成功；
- ProgramIO schema 校验成功；
- finalizer 单独成功；
- 用手工构造的 manifest 或旧固定规模 artifact 运行成功。

每个闭环 case 必须同时具有：

```text
schema_verified = true
candidate_verified = true
lower_link_verified = true
program_io_verified = true
finalizer_verified = true
resolver_verified = true
runtime_verified = true
repeatability_verified = true       # 发布代表 case；PR smoke 可延后到 nightly
functional_execution = out_of_scope
```

### 2.2 Runtime 成功门禁

每次 npusim execution 必须满足：

1. process exit code 为 0；
2. 所有 rank/core stream 均被 artifact 和 ProgramIO 覆盖；
3. 所有 state load、state store、input、output 和 probe 绑定到合法 HBM 地址；
4. artifact record 数和最终文件字节数不超过生产上限；
5. runtime marker 的 mesh/workload/artifact digest 与输入一致；
6. makespan、core coverage、active route 和 state completion marker 可解析；
7. runtime 结束残留为零：

```text
active_endpoints = 0
active_sessions = 0
outstanding_tags = 0
incomplete_barriers = 0
pending_state_writes = 0
proto_wait_count = 0
```

MoE 还必须检查 dispatch、combine、backward packet 和 expert slot 全部消费；Dense
Train 还必须检查 gradient synchronization、optimizer 和 persistent store 已完成。

### 2.3 阶段完成状态

新增或在 capability report 中明确记录以下代表尺寸状态：

```text
representative_meshslice_runtime_complete
representative_dense_train_runtime_complete
representative_moe_infer_runtime_complete
representative_moe_train_runtime_complete

representative_flexible_mesh_runtime_complete
  = representative_meshslice_runtime_complete
    && representative_dense_train_runtime_complete
    && representative_moe_infer_runtime_complete
    && representative_moe_train_runtime_complete
```

这些状态不得替代原方案中的 `*_rect_complete`。只有全部 100 种 Mesh runtime
矩阵通过后，才允许设置 `dense_train_rect_complete`、`moe_*_rect_complete` 和
`flexible_mesh_workloads_complete`。

## 3. 当前基础和主要缺口

### 3.1 已有基础

- 100 种完整矩形 Mesh 的 RectMesh、rank、route、group registry 已闭合；
- 四类 workload 的 400 组合 typed compile matrix 已通过；
- Dense Train 已有覆盖全部 persistent parameter shard 的 backward/WGRAD/DP
  reduction/SGD typed carrier；
- Dense forward 已有 production placement、lower/link 和 1×1 smoke；
- MoE inference/train 已有 EP=R、top-k=1、static trace、Direct-XY、cyclic wave、
  四向 backward route、expert/gate SGD carrier；
- MoE 已有 1×1、1×2、2×2 的公共 opcode、DTE、state 和 ProgramIO mapping plan；
- MeshSlice 已有 LOCAL、ROW_ONLY、COLUMN_ONLY、FULL_2D 四种模式；
- MeshSlice AG 在代表 Mesh 已通过 standard lower/link 和 ProgramIO；
- MeshSlice RS/AR 已有严格 REDUCE ABI 下的通用 UNFUSED fallback，并在代表 Mesh
  通过 lower/link 和 ProgramIO；
- 旧 Train、Lite Train、MoE Swizzle、Lite MoE runtime runner、marker 和 evidence
  可作为生产链参考实现。

### 3.2 共同缺口

- flexible workload compiler 还没有输出统一的真实 `LinkedProgramManifest`；
- capability report 仍缺 finalizer、resolver、runtime residual 和 repeatability 实证；
- flexible runtime provider 尚未统一调用 ProgramIO、finalizer、resolver 和 npusim；
- final artifact 的 record/file-byte 还没有回填到 workload capacity report；
- runtime marker 中可能仍残留 2×2、EP4 或 die ID 小于 4 的旧假设；
- 代表 case 尚无统一 artifact SHA、makespan 和 marker digest 稳定性检查。

### 3.3 Dense Train 缺口

- full-model backward 仍是 typed carrier，不是 standard action DAG；
- activation/tape、parameter、FP32 gradient 和 optimizer state 尚未全部落到 ProgramIO；
- backward、gradient reduction、SGD、state store 尚未进入同一个 manifest；
- 当前 production smoke 主要证明 forward，不足以证明完整训练步执行。

### 3.4 MoE 缺口

- `FlexibleMoeExecutablePlan` 尚未适配到生产 MoE standard IR2/Core ABI/Operand ABI；
- 当前 `FlexibleMoeStandardLoweringPlan` 是严格映射计划，不是真实 linked manifest；
- generic EP=R state、endpoint、barrier 和 runtime marker 尚未进入 finalizer/npusim；
- train 的 reverse dispatch、expert/gate WGRAD、AR、SGD 尚未在一个 artifact 中运行。

### 3.5 MeshSlice 缺口

- AG 和 UNFUSED RS/AR 虽已 lower/link+ProgramIO，但未通过统一 flexible runtime
  provider 形成 npusim evidence；
- LOCAL 模式需要证明没有伪 DTE、伪 session 和伪 route marker；
- 1D/2D 模式需要证明真实 endpoint/session/wave 结束无残留；
- native MeshSlice RS/AR 仍受严格二输入 REDUCE ABI 阻断，但这不阻塞本阶段闭环。

## 4. 核心开发原则

### 4.1 先完成纵向闭环，再扩尺寸

每类负载先选择最小 case 打通 spec 到 npusim 的整条链：

| 负载 | 第一条纵向 case |
|---|---|
| MeshSlice | 1×1 AG_GEMM LOCAL |
| MoE inference | 1×1 all-local E1 |
| MoE train | 1×1 all-local E1 + expert/gate SGD |
| Dense Train | 1×1 TP1×DP1 single step |

1×1 闭环后再进入 1D 通信和二维通信 case。禁止在第一条 case 尚未运行时先批量
构造全部代表 artifact。

### 4.2 Baseline 优先

- MoE 固定使用 Direct-XY，不在本阶段接入 Comet/AUTO 性能选择；
- MeshSlice AG 使用现有 standard 路径；
- MeshSlice RS/AR 使用已经验证的 UNFUSED fallback；
- Dense Train 使用 ordinary GEMM/collective/local SGD；
- 不把 native MeshSlice RS/AR、top-k=2、DP×EP 或 AdamW 作为闭环前置条件。

### 4.3 单 artifact 原则

每个 inference request 或 training step 必须生成一个 manifest/artifact。不得通过
分别运行 forward、backward、collective 和 optimizer 四个 artifact 后拼接报告来
声称训练闭环。

一个 training artifact 至少包含：

```text
state load
-> forward
-> loss / output gradient seed
-> backward data gradient
-> WGRAD
-> gradient synchronization
-> optimizer
-> state store
-> probes
```

### 4.4 冻结旧链路

不修改旧 DP2/DP4 Lite Train、EP4/2×2 Lite MoE、C0-C4 scale truth 和旧 runtime
evidence 的 stable IDs。新 flexible runtime 通过 adapter 复用 opcode、ABI、runner 和
marker 逻辑，避免用任意 R 逻辑反向改写旧 frozen artifact。

## 5. 统一 Runtime Provider

### 5.1 新增运行 case 和 evidence carrier

建议新增：

```text
schema/flexible_mesh_runtime.py
test/frontend/integration/flexible_mesh_runtime_provider.py
test/frontend/integration/flexible_mesh_runtime_evidence.py
test/frontend/integration/flexible_mesh_runtime_markers.py
test/frontend/integration/run_flexible_mesh_runtime.py
```

核心 schema：

```text
FlexibleMeshRuntimeCase
  case_id
  workload_spec
  selected_baseline
  expected_rank_count
  expected_state_owners
  expected_active_routes
  repeat_count

FlexibleMeshRuntimeEvidence
  mesh_digest
  workload_digest
  manifest_digest
  artifact_sha256
  artifact_file_bytes
  program_io_digest
  resolver_digest
  npusim_exit_code
  makespan_cycles
  marker_digest
  rank/core coverage
  runtime residual
  state completion
```

evidence 必须从真实输出解析，不允许由输入 plan 反向填充 runtime 字段。

### 5.2 Provider 执行顺序

统一 provider 固定执行：

1. compile workload；
2. materialize workload-specific standard program；
3. lower/link；
4. build ProgramIO；
5. schema 和 HBM range preflight；
6. finalizer；
7. 回填 exact record/file-byte；
8. resolver；
9. npusim timing execution；
10. marker、residual、state completion 解析；
11. capability report 更新；
12. 可选第二次运行并比较 repeatability。

任一步失败都返回 typed stage failure，不得继续执行后续阶段，也不得保留旧
`runtime_verified=true`。

### 5.3 通用 marker 参数化

去除固定 2×2/EP4 假设：

- die coverage 为 `0..R-1`；
- core/rank coverage 由 manifest 和 placement 交叉验证；
- expected directed physical links 上界为
  `2[H(W-1)+W(H-1)]`，active link 集合从真实 route incidence 派生；
- endpoint/session 按真实 active core 和 cyclic wave 聚合；
- marker 同时携带 mesh、workload、manifest 和 ProgramIO digest；
- LOCAL/all-local case 允许 active route 集为空，但不允许缺 core/state marker。

### 5.4 Finalizer 精确容量回填

构建前继续使用 symbolic upper bound；finalizer 后记录：

```text
exact_record_count
exact_relocation_count
exact_runtime_symbol_count
exact_artifact_file_bytes
per_core_stream_bytes
```

若 exact 值超过 1,048,576 records、64 MiB artifact、U16 core 或其他生产上限，
case 必须 fail closed。不得通过删除 probe 或 marker 隐藏超限。

## 6. MoE Runtime 闭环

### 6.1 Flexible MoE 到生产 ABI adapter

新增从 `FlexibleMoeExecutablePlan` 到现有生产 MoE lowering/linker 输入的 adapter，
优先复用：

- `lowering/moe_swizzle_workload_standard.py`；
- `lowering/moe_swizzle_workload_linker.py`；
- `passes/build_moe_swizzle_program_io.py`；
- 现有 MoE runtime provider、marker 和 evidence parser。

adapter 必须显式转换：

- rank/expert placement；
- dispatch/combine/backward pair bucket；
- X-first `PairRoute`；
- cyclic-delta wave；
- DTE SEND/RECV/WAIT endpoint；
- local assignment view；
- expert/gate parameter、gradient、tape 和 optimizer state；
- barrier 和 dependency；
- public opcode 和 operand ABI。

禁止通过 route ID、action name 或 `range(4)` 推断 source/destination/expert。

### 6.2 MoE inference 单 artifact

真实 program 顺序：

```text
gate/input load
-> dispatch pack
-> dispatch cyclic waves
-> local expert compute
-> combine cyclic waves
-> weighted combine
-> output probe
```

验证 all-local、balanced 和 single-hot/empty-expert trace。没有 payload 的 pair 不生成
伪 DTE；empty expert 可以没有 token compute，但其合法 state ownership 仍需验证。

### 6.3 MoE train 单 artifact

在 inference 链后加入：

```text
output gradient seed
-> backward gradient source-to-expert
-> expert DGRAD/WGRAD
-> backward dx expert-to-source
-> combine backward
-> gate WGRAD
-> arbitrary-rank gate AR
-> expert/gate SGD
-> persistent state store
```

每个 remote assignment 必须在真实 program 中保留四向 closure。expert optimizer
等待该 expert 全部 WGRAD；gate optimizer 等待 gate AR barrier；state store 等待对应
optimizer。

### 6.4 MoE 完成门禁

- 1×1 all-local inference/train 闭环；
- 1×2、2×1 balanced inference/train 闭环；
- 2×2、2×3 balanced inference/train 闭环；
- 至少一个二维 single-hot/empty-expert case 闭环；
- 每波每 core active sessions 小于等于 3；
- dispatch/combine/backward packet 和 slot residual 为零；
- expert/gate state store marker 完整；
- 发布代表 case 连续运行两次，artifact SHA、makespan 和 marker digest 一致。

## 7. Dense Train Runtime 闭环

### 7.1 Materialize full backward action DAG

将 `FlexibleDenseTrainPlan` 的 typed carrier 转成真实 standard actions。每个参数 shard
必须具有 exact 链：

```text
parameter load
-> exact forward consumers
-> exact backward tape consumers
-> local WGRAD write
-> DP gradient collective
-> SGD read/write
-> persistent state store
```

不得只物化 LM head；embedding、attention、norm、MLP 和 LM-head 的全部 persistent
parameter shard 都必须进入状态闭环。

### 7.2 State 和 HBM ABI

每个 parameter shard 显式分配：

- FP16 parameter；
- FP32 gradient accumulation；
- 可选 FP32 master parameter；
- SGD state；
- activation/tape；
- input/loss/probe；
- updated parameter store target。

ProgramIO 必须验证 owner rank、address space、alignment、span、不重叠和 lifetime。
gradient 与 parameter 不允许危险 alias；optimizer 不允许读取 reduction 完成前的梯度。

### 7.3 复用现有 Train 生产链

优先扩展：

- `passes/train_lower_program.py`；
- `passes/train_link_program.py`；
- `passes/program_io.py`；
- 现有 train-forward runtime runner；
- Lite Train 中已经验证的 REDUCE、WGRAD、SGD 和 state opcode/ABI。

forward 的现有 production materialization 保持不变；backward/optimizer 作为 v2
并列 adapter 合入同一 manifest，不解除旧 `TrainWorkloadSpec` 的 fail-closed 约束。

### 7.4 Dense Train 完成门禁

- 1×1 TP1×DP1 完整 single-step 闭环；
- 1×2 TP2、2×1 DP2 退化轴闭环；
- 2×2 DP2×TP2 和 2×3 DP2×TP3 闭环；
- 参数 load/store、forward/backward/WGRAD/SGD action coverage 精确；
- DP1 不生成伪 collective，DP>1 使用 session-safe cyclic waves；
- optimizer/state store dependency 审计通过；
- runtime 结束 gradient、barrier、state residual 为零；
- 发布代表 case 连续运行两次稳定。

## 8. MeshSlice Runtime 闭环

### 8.1 本阶段路径

| 请求 | Runtime 路径 |
|---|---|
| AG_GEMM | production MeshSlice standard |
| GEMM_RS | generic UNFUSED fallback |
| GEMM_AR | generic UNFUSED RS+AG fallback |

RS/AR fallback 是本阶段的正式 executable baseline。native MeshSlice REDUCE 的单输入
ref 与严格二输入一输出 ABI 冲突继续保持 fail-closed，不通过放宽校验换取表面闭环。

### 8.2 四种模式验证

- LOCAL：1×1，无 SEND/RECV/WAIT、route、session；
- ROW_ONLY：1×N，只有 row panel exchange；
- COLUMN_ONLY：N×1，只有 column panel exchange；
- FULL_2D：H×W，row wave 和 column wave 不并发。

AG 和 fallback 的 manifest、ProgramIO、finalizer、resolver、runtime evidence 均通过
统一 provider 生成。fallback report 必须保留：

```text
selected_path = UNFUSED_FALLBACK
reason = STRICT_TWO_INPUT_REDUCE_ABI
```

### 8.3 MeshSlice 完成门禁

- AG_GEMM 在 1×1、1×3、3×1、2×3 runtime 闭环；
- GEMM_RS/GEMM_AR 在同一代表集 runtime 闭环；
- LOCAL 无伪 transport marker；
- 1D/2D 的 route、payload、subview addend 和 core coverage 闭合；
- endpoint/session/barrier residual 为零；
- fallback 原因、selected path 和 artifact digest 进入 capability report。

## 9. 代表 Mesh Runtime 矩阵

为了加速开发，将矩阵分层执行。

### 9.1 PR smoke

```text
1×1       local/all-local/DP1×TP1
1×2       row-only/TP-only/EP2
2×1       column-only/DP-only/EP2
2×2       first full-2D case
```

每次 PR 不必运行全部 trace 和 repeatability，但每类 workload 至少保留一条真实
npusim smoke，且不得替换为 mocked runner。

### 9.2 Nightly representative

```text
1×1
1×10, 10×1
2×2
2×3, 3×2
5×6, 6×5
10×10
```

覆盖：

- Dense Train single step；
- MoE inference balanced；
- MoE train balanced；
- 二维 MoE hot/empty expert 补充 case；
- MeshSlice AG_GEMM；
- MeshSlice GEMM_RS/GEMM_AR fallback。

大 Mesh 使用 capacity-safe tiny model/trace，不用固定模型强行覆盖所有 TP/EP。

### 9.3 Repeatability

发布代表集至少连续运行两次，比较：

```text
manifest digest
artifact SHA256
artifact file bytes
ProgramIO digest
resolver digest
makespan cycles
marker digest
runtime residual
```

若 npusim 存在明确、受控的非确定字段，必须从 digest schema 中版本化排除，不能在
测试代码里临时忽略任意字段。

## 10. 分阶段实施

### R0：统一 provider 和 evidence

工作项：

- 定义 runtime case/evidence/residual schema；
- 封装 ProgramIO、finalizer、resolver、npusim runner；
- 参数化 mesh/core/link/session marker；
- 回填 exact artifact capacity；
- capability report 严格按阶段更新。

门禁：

- 复用一个现有可运行旧 artifact 验证 provider 本身；
- 任一阶段故障都能返回准确 stage 和 typed reason；
- mocked npusim 不得产生 `runtime_verified=true`。

### R1：MeshSlice 纵向闭环

优先原因：现有 AG 和 fallback 已经 lower/link+ProgramIO，最适合先验证统一 provider。

门禁：

- 1×1 AG、RS fallback、AR fallback 真实运行；
- 再扩 1×3、3×1、2×3；
- 四模式 marker/residual 语义正确。

### R2：MoE inference

工作项：

- flexible plan 到 production MoE ABI adapter；
- Direct-XY dispatch/combine single manifest；
- generic EP=R ProgramIO 和 marker。

门禁：1×1、1×2、2×1、2×2、2×3 inference runtime 通过。

### R3：MoE train

工作项：

- backward route/action materialization；
- expert/gate gradient、AR、SGD、state store；
- train-specific residual。

门禁：与 R2 同一基础代表集的 complete step runtime 通过。

### R4：Dense Train

工作项：

- full-model backward action materialization；
- activation/gradient/optimizer HBM ABI；
- forward/backward/collective/optimizer 单 manifest；
- Dense Train ProgramIO 和 residual。

门禁：1×1、1×2、2×1、2×2、2×3 complete step runtime 通过。

### R5：代表矩阵和发布报告

工作项：

- 扩展 nightly representative；
- repeatability；
- artifact capacity 和 runtime residual 汇总；
- 输出 capability/evidence 报告；
- 固化下一阶段 100-shape runtime 输入矩阵。

完成后才允许设置 `representative_flexible_mesh_runtime_complete=true`。

## 11. 建议代码改动

### 11.1 新增

| 文件 | 内容 |
|---|---|
| `schema/flexible_mesh_runtime.py` | runtime case、evidence、residual、stage status |
| `passes/flexible_mesh_program_io.py` | Dense/MoE/MeshSlice 统一 ProgramIO 分派 |
| `lowering/flexible_moe_adapter.py` | flexible MoE 到 production MoE ABI adapter |
| `passes/flexible_dense_backward.py` | full-model backward/gradient/optimizer materialization |
| `integration/flexible_mesh_runtime_evidence.py` | evidence parse/validate/serde |
| `integration/flexible_mesh_runtime_markers.py` | 任意 H/W marker 解析与 residual |
| `integration/run_flexible_mesh_runtime.py` | 单 case 真实 runner |
| `integration/test_flexible_mesh_runtime_smoke.py` | PR 代表 runtime smoke |

### 11.2 优先扩展

| 文件 | 改动 |
|---|---|
| `flexible_mesh_compiler.py` | materialized program/manifest/runtime capability 分派 |
| `passes/flexible_dense_train.py` | 输出 backward materialization 所需 exact references |
| `passes/flexible_moe.py` | 输出 production adapter 所需 typed endpoint/state/dependency |
| `lowering/moe_swizzle_workload_standard.py` | 支持 adapter 的任意 R 输入，不改变旧 EP4 truth |
| `lowering/moe_swizzle_workload_linker.py` | generic core streams 和 state fragments |
| `passes/program_io.py` | flexible train/MoE state target 和 probe |
| `test/frontend/integration/flexible_mesh_runtime_provider.py` | 从 compile-only provider 升级为真实 runner |
| `test/frontend/integration/flexible_mesh_runtime_report.py` | finalizer/runtime/repeatability evidence |

实际实现时允许按现有模块边界调整文件名，但必须保持 schema、lowering、runner 和
evidence 的职责分离。

## 12. 测试和故障注入

### 12.1 单元测试

- adapter 的 action/value/state coverage；
- endpoint source/destination 与 PairRoute 一致；
- optimizer 严格晚于 gradient reduction；
- ProgramIO HBM owner/range/alignment/lifetime；
- finalizer exact capacity 回填；
- marker 对任意 H/W 的 core/link/session 解析；
- evidence serde 和 forged digest 拒绝；
- LOCAL/all-local 不产生伪 transport。

### 12.2 集成测试

- 每类 workload 的 1×1 真实 runtime；
- 1×N、N×1、二维通信；
- MeshSlice AG/RS/AR 三种请求；
- MoE all-local/balanced/hot-empty；
- Dense DP1、TP1、DP×TP；
- repeatability；
- runner/finalizer/resolver/npusim 任一失败时 capability 状态不越级。

### 12.3 负例

- ProgramIO 缺 state/input/probe；
- HBM range 越界或 state alias；
- backward 缺 tape；
- optimizer 早于 collective；
- route/endpoint/wave 不一致；
- 每 core 第 4 个 active session；
- dispatch 或 backward packet 未消费；
- final artifact 超 record/file-byte 上限；
- resolver digest 与 manifest 不一致；
- runtime 退出码非零；
- endpoint/session/tag/barrier/state residual 非零；
- forged marker 或使用旧 artifact evidence；
- 第二次运行 artifact/makespan/marker 不稳定。

负例必须 fail closed，并明确报告失败阶段。

## 13. 加速开发策略

1. 先复用现有 runtime runner 打通 MeshSlice 1×1，验证公共 provider；
2. MoE 先做 inference，再在同一个 adapter 上追加 train stages；
3. Dense 先完成 1×1 全状态闭环，再打开 DP/TP collective；
4. PR 仅运行四个小 Mesh smoke，代表大 Mesh 和 repeatability 放入 nightly；
5. RS/AR 保留 UNFUSED fallback，不在本阶段开发 native REDUCE；
6. 保持 Direct-XY，不同时开发 Comet/AUTO 校准；
7. 每打通一类 workload 立即增加真实 runtime test，避免最后集中排查；
8. 复用旧 opcode、ABI、finalizer、resolver 和 marker，不修改冻结 stable IDs；
9. 先使用 tiny model/trace 控制 artifact 大小，再单独做大模型容量治理；
10. compile/lower/link 单元测试可并行，npusim runtime 按 artifact 串行收口，减少
    模拟器资源竞争和非确定性。

## 14. 风险和回退

### 14.1 NpuSim opcode/ABI 不支持

若 full-model action 暴露 npusim 未支持的 opcode，优先映射到 Lite Train/MoE 已运行
过的 opcode 组合；必须保留真实 dependency 和 state 语义，不能改为 no-op 后声明成功。

### 14.2 Artifact 超限

先缩小 tiny model、token 和 probe 数，但不得删除完成性所需的 parameter/state；若
代表 case 仍超过 1M records 或 64 MiB，则输出 typed capacity failure，并单独立项
Program ABI 模板/循环，不在本阶段临时提高生产上限。

### 14.3 大 Mesh 模拟时间过长

PR 继续使用小 Mesh；nightly 对 10×10 使用最小合法模型和 trace。如果 10×10 超过
CI 时限，保留 artifact/finalizer gate，并在专用 runtime job 中完成 npusim evidence，
不得把 compile-only 结果标记为 runtime。

### 14.4 旧固定规模假设

发现 EP4、2×2、die 小于 4 的逻辑时，优先在 flexible adapter/provider 中参数化；若
修改公共 parser，必须先冻结并回归旧 evidence，确保 stable ID 和旧 runtime 结果不变。

## 15. 本阶段不做的内容

- 全部 100 种 Mesh 的四负载 runtime 发布矩阵；
- functional/numerical correctness；
- 多 microbatch accumulation；
- AdamW、ZeRO、PP>1、recompute；
- MoE top-k=2、DP×EP、TP×EP；
- Comet/AUTO 性能收益发布；
- native MeshSlice RS/AR；
- 故障 Mesh、缺 Die、torus 或动态绕行；
- 通过提高生产容量上限规避 artifact 设计问题。

## 16. 最终交付物

1. 统一 flexible runtime case/provider/evidence/marker；
2. MeshSlice AG 和 RS/AR fallback 的代表 Mesh npusim evidence；
3. MoE inference/train Direct-XY 的代表 Mesh单 artifact runtime；
4. Dense Train complete step 的代表 Mesh单 artifact runtime；
5. finalizer exact capacity、resolver、ProgramIO 和 residual 报告；
6. PR smoke 和 nightly representative 测试；
7. repeatability evidence；
8. 下一阶段 100-shape runtime 扩展输入矩阵和未通过 case 清单。

最终验收报告必须逐 workload、逐 Mesh 给出真实 evidence，不得由 schema、candidate、
lower/link 或 compile matrix 自动推导 runtime 完成状态。
