# 任意矩形 DieMesh 全负载 NpuSim Runtime 闭环与 100 Shape 发布开发落地报告

## 0. 文档状态

本文记录本轮批准的代表性发布验收：编译/Schema/axis/group/capacity 静态层覆盖
完整 100 shape，真实 runtime 层只覆盖精确的 8 个代表 mesh。当前口径固定为：

    validation_scope = representative
    exhaustive_runtime = false

本文只把同一正式 tool binding 下、两次独立 execution 均完成的 evidence 写入结果；
开发 canary、诊断 root、中止矩阵和旧 binding 不进入本轮代表性 completion。

最终代表性 merge 已完成：48 cases/96 executions，六个 family 各 8 cases/16 executions，
`runtime_verified=48`、`repeatability_verified=48`，failure、stage failure 和十项
residual 非零计数均为 0。完成状态由
`derive_flexible_mesh_representative_completion(...)` 从 evidence matrix 派生；
`exhaustive_runtime=false`、`exhaustive_rect_runtime_complete=false`。

## 1. 目标、范围与发布单元

本轮 envelope 是完整连续矩形、`1 <= rows, columns <= 10`、每 Die 一个 rank、
row-major rank、X-first route。验收分成两个互不混淆的层次：

1. 静态层：六类 family 的编译、Schema、axis/group 和 capacity preflight 覆盖全部
   `{1..10} x {1..10}`，每类 100 shape；
2. runtime 层：六类 family 只实跑以下精确 8 个代表 mesh，每个 case 两次独立执行：

       1x1, 1x4, 4x1, 2x2, 2x3, 3x2, 3x3, 10x10

| Family | Operation | Baseline | 静态形状数 | runtime cases | runtime executions |
|---|---|---|---:|---:|---:|
| `dense_train` | `dense_train_step` | `dense_timing_baseline` | 100 | 8 | 16 |
| `moe_inference` | `moe_inference_step` | `moe_direct_xy` | 100 | 8 | 16 |
| `moe_train` | `moe_train_step` | `moe_direct_xy` | 100 | 8 | 16 |
| `meshslice_ag` | `ag_gemm` | `meshslice_standard` | 100 | 8 | 16 |
| `meshslice_rs_fallback` | `gemm_rs` | `unfused_fallback` | 100 | 8 | 16 |
| `meshslice_ar_fallback` | `gemm_ar` | `unfused_fallback` | 100 | 8 | 16 |

本轮代表性 runtime 矩阵的精确基数是 48 cases、96 次独立 execution。每个 family
必须精确覆盖上述 8 个 mesh；缺失、重复、额外形状或 family/operation/baseline
漂移均 fail closed。600 cases/1,200 executions 的全形状真实 runtime 属于未来
`exhaustive_runtime=true` 扩展目标，不是本轮完成条件。

本轮执行语义固定为：

    timing_execution = true
    functional_execution = false

因此报告只声明 timing runtime、状态生命周期、transport 和证据闭环；不声明
Dense/MoE/collective 的数值 functional correctness 或性能最优。

## 2. 已完成的公共实现

### 2.1 矩形 Mesh 与编译入口

- `schema/rect_mesh.py`：唯一 typed `RectMeshSpec`，约束 1..10 矩形、row-major、
  X-first、单 rank/Die 和 timing-only 语义。
- `schema/rect_mesh_compile.py`、`flexible_mesh_compiler.py`：矩形参数贯通、
  capability/fallback 诊断和向后兼容入口。
- `policies/swizzle/rect_mesh_topology.py`：确定性矩形 topology、snake/tree 等
  有界构造，避免大矩形递归回溯。
- `schema/flexible_mesh_workload.py`、`flexible_mesh_groups.py`、
  `flexible_mesh_capacity.py`：workload、group/route、容量与 envelope 合同。

### 2.2 统一 release schema 与派生完成状态

- `schema/flexible_mesh_release.py`：六类 family、两次 execution、tool/config binding、
  capacity、residual、marker 和稳定 ID；仍可生成全形状 canonical cases。
- 代表性 runner/merge 层按固定 8-mesh 白名单过滤并保持 canonical case ID，不把
  过滤后的 48-case 结果伪装成 exhaustive matrix。
- `schema/flexible_mesh_completion.py`：七类 contract receipt 和只读完成视图；
  完成状态只能由 exact evidence matrix 派生。
- `run_flexible_mesh_release.py`：单 case 两次完整工具链、preflight ProgramIO、
  finalizer 后 actual artifact SHA rebind、resolver、NpuSim 和原始证据落盘。
- `run_flexible_mesh_release_merge.py`：六 family 合并、binding 一致性、exact
  cardinality 和 completion derivation。
- `flexible_mesh_release_hardware.py`：统一 checked-in P5 large template，按形状
  确定性 specialize；binding SHA 仍绑定原始模板，actual hardware SHA 进入 execution。
- `flexible_mesh_release_profiles.py`：family trace/model profile 和 adapter input
  冻结，防止算法或 tiny workload 漂移复用旧 case ID。

### 2.3 Runtime one-shot 与工具链闭环

- Program artifact runner 使用显式 `--program-one-shot`；legacy helper 默认 refill
  行为保留，仅 release runner 关闭 refill。
- terminal CONFIG 保留 `is_end=true`，one-shot 路径设置 `refill=false`，防止
  快核 `SEND_DONE` 后进入第二轮普通 prim。
- blocking LSU load/store 归入同步执行类别；地址/extent 验证仍按 transfer 处理。
- finalizer 对 Flexible Dense/MoE/MeshSlice 使用 producer-scoped exact
  schema/kind/cardinality/source/fragment/lineage closure，不放宽 generic 分支。
- NpuSim 输出必须包含 ProgramIO、PROGRAM_MEMORY、P2P/collective、router/D2D、
  credit、HOSTSIG/HOSTLANE 和 makespan 证据；observer 只消费真实输出。

## 3. Dense Train exact runtime

Dense 路径已经从 typed carrier 扩展到逐 rank production materializer：

    tiny forward train graph
      -> full-parameter typed backward/WGRAD
      -> DP gradient sync
      -> SGD_UPDATE
      -> HBM state STORE
      -> LinkedProgramManifest
      -> ProgramIO

关键实现包括：

- `schema/flexible_dense_train.py`、`schema/flexible_dense_backward.py`、
  `schema/flexible_dense_backward_ir.py`；
- `passes/flexible_dense_train.py`、`passes/flexible_dense_backward_projection.py`、
  `passes/flexible_dense_backward.py`、`passes/flexible_dense_backward_multi.py`；
- `flexible_mesh_release_dense.py`、`run_flexible_mesh_release_dense.py`。

Dense release workload 显式绑定 `release_layers=1`，默认 `DP=rows`、`TP=columns`、
`PP=1`。DP sync 使用 row-major、root0 的 binary-tree reduce+broadcast：DP=1
typed no-transport；DP>1 物化 SEND/RECV/WAIT/LOCAL_REDUCE。每个 parameter/state
保持 owner、HBM address、initialization、probe、optimizer-after-reduction 和
store-after-optimizer 闭环。

Dense observer 对有 transport 的形状精确核对每 core typed tx/rx completion 和
P5 drain；对 DP=1 要求 per-core P5 STATS/DRAIN 均为空且全局 timing drain 恰一条 0。
伪造额外 STATS/DRAIN、缺失/非零/重复 timing drain 均拒绝。


Dense 最终代表矩阵已在 fresh root 完成 8/8 cases、16/16 独立 executions：

- binding：`flexible_mesh_release_binding_c9159583ad0c5853`；
- root：`/workspace/build-release-final/flexible-dense-release-representative-binding-c1635d56`；
- public strict schema/binding 校验 8/8 通过，`runtime_verified` 与
  `repeatability_verified` 均为 true；
- stage 非零计数 0、residual 非零计数 0、repeatability mismatch 0；
- 最大 makespan 61,125 cycles，最大 linked manifest 47,639,480 bytes，
  最大 artifact 2,021,297 bytes；
- 最大 exact records 18,470，最大 transport tags 1,620。

这些值与最终代表性 completion 中的 Dense 子矩阵一致。
## 4. MoE exact runtime

MoE 路径包含 inference/train 两类 primary family，使用专用 typed plan、standard
mapping 和 production lowering：

- `schema/flexible_moe.py`、`schema/flexible_moe_standard.py`；
- `passes/flexible_moe.py`；
- `lowering/flexible_moe_standard.py`、`flexible_moe_production.py`、
  `flexible_moe_multi_production.py`；
- `flexible_mesh_release_moe.py`、`run_flexible_mesh_release_moe.py`。

Inference 必须闭合 dispatch/combine 与 HBM output retention；train 还必须闭合
four-way route、gradient、optimizer、state writeback 和每 owner state probe。
多 rank route/endpoint/wave、state ownership、ProgramIO actual artifact SHA 与
完整 residual 均进入正式 evidence，不能由 expected marker 代填。

本轮 MoE 代表矩阵已经在 fresh root
`/workspace/build-release-final/flexible-moe-release-representative-binding-c1635d56`
完整落盘，结果如下：

- 代表 mesh：`1x1, 1x4, 4x1, 2x2, 2x3, 3x2, 3x3, 10x10`；
- inference 8 cases/16 executions，train 8 cases/16 executions；
- finalizer/resolver/NpuSim stage failure 为 0，ProgramIO 三阶段不匹配为 0；
- 十项 residual 的非零 execution 为 0；
- artifact/manifest/ProgramIO/marker/makespan/plan/spec 双执行差异为 0；
- inference marker 为 `dispatch_combine`；train marker 为
  `four_way_route/gradient/optimizer/state`；
- 最大 makespan 为 180,865 cycles（train 10x10）；
- 最大 linked manifest 13,846,670 bytes，最大 artifact 570,947 bytes；
- 最大 exact/symbolic records 均为 5,294，最大 transport tags 为 598；
- 最大 rank count 为 100，最大 runtime core ID 为 1,584，
  最大 session/core/wave 为 2。

这些值与最终代表性 completion 中的 MoE 两类子矩阵一致。

## 5. MeshSlice exact runtime

MeshSlice 发布包含 AG_GEMM standard 及 GEMM_RS/GEMM_AR unfused fallback：

- `flexible_mesh_runtime_meshslice.py`、`flexible_mesh_release_meshslice.py`；
- `run_flexible_mesh_release_meshslice.py`；
- standard/fallback lineage、packed storage/view/operand ABI、DTE/collective
  lowering 和 strict finalizer producer 分支。

三类 operation 均必须保持其冻结 baseline，不允许因 optimized candidate 失败而
变成无程序；fallback 必须是真实可执行 program，而不是 capability 文本。

MeshSlice 最终代表矩阵位于
`/workspace/build-release-final/flexible-meshslice-release-representative-binding-c1635d56`，
AG、RS fallback、AR fallback 各完成 8 cases/16 executions；三阶段、十项 residual
和双执行重复性均通过。10x10 指标如下：

| Family | Makespan | Exact records | Linked bytes | Artifact bytes | Max sessions/core/wave |
|---|---:|---:|---:|---:|---:|
| MeshSlice AG | 11,157 | 6,200 | 18,230,440 | 600,217 | 2 |
| MeshSlice RS fallback | 150,128 | 50,900 | 132,302,879 | 4,756,717 | 2 |
| MeshSlice AR fallback | 180,885 | 90,696 | 228,283,906 | 7,517,245 | 2 |

## 6. 静态 100-shape 验证

静态验证与代表性 runtime 分开计数，且不调用真实工具：

- Dense：dense train 100 shape 与 backward lineage 100 shape 两项测试精确覆盖，
  校验 `DP=rows`、`TP=columns`、state/gradient wave 与 session/core/wave 上界；
  结果为 2 tests PASS，170.649s。
- MoE：命令
  `env PYTHONPATH=. python3 -m unittest -v llm.test.frontend.unit.test_flexible_moe`
  为 6/6 PASS，10.591s，peak RSS 54,400 KiB。100 mesh 生成
  inference/train 共 200 typed cases；X-first flows 23,994、adjacent XY hops
  58,150、最大 hop 18、最大 session/core/wave 2。10x10 train 静态值为
  3,394 actions、598 flows、400 states、16,170 symbolic records、
  801,760 symbolic bytes。
- MeshSlice：
  `llm/test/frontend/integration/test_swizzle_meshslice_standard.py::`
  `test_all_100_shapes_have_one_exact_execution_mode` 覆盖 100 shape，
  模式分布为 FULL_2D=81、ROW_ONLY=9、COLUMN_ONLY=9、LOCAL=1；
  既有完整 suite 9/9、IR0/Swizzle/IR2/projection 29/29、
  100-shape+四 mode lower/link+冻结 2x2 3/3 均通过，精确耗时未留存。

## 7. 严格验收口径

单个 case 只有同时满足以下条件才允许 `runtime_verified`：

1. case/binding/schema/stable ID 均通过 public strict validate；

2. 两次 execution 各自独立 materialize、finalize、resolve、simulate；
3. linked manifest、hardware、mapping、simulation、artifact 和 ProgramIO SHA 闭合；
4. ProgramIO 使用 actual artifact SHA，所有 initialization/probe 均精确覆盖；
5. finalizer、resolver、NpuSim exit code 均为 0；
6. rank/core coverage 精确，无重复或缺失；
7. family completion markers 来自真实 stdout，marker digest 可重算；
8. endpoint、session、tag、barrier、state write、protocol wait、LSU、DTE、router、
   credit residual 全零；
9. record、manifest bytes、runtime core ID、session/wave 和 transport tag 不越界；
10. timing=true、functional=false。

`repeatability_verified` 还要求两次 execution 的 artifact、spec/plan/manifest、
ProgramIO、hardware/mapping/simulation、marker、makespan 和 capacity 满足 schema 的
精确重复性合同。一次成功、复制 evidence 或手填 repeatability 都不能通过。

本轮代表性 merge 还必须验证：

- 唯一 release binding；
- 六 family 各精确 8 case、总计 48；
- 每 case 恰两次 execution、总计 96；
- schema/axis/group/capacity/fallback/runtime/stable-ID 七类 receipt 可从矩阵重算；
- `derive_flexible_mesh_representative_completion(...)` 成功，所有目标状态由派生属性给出。

## 8. 代表性发布证据

### 8.1 最终统一 binding

| 项目 | 最终值 |
|---|---|
| runtime profile | `flexible-mesh-timing-v3-one-shot` |
| environment profile | `p5-rect-release-v1` |
| finalizer SHA-256 | `d39bb8c69349f8492e77862e28324eb9621fb472982dc5ba87c3c1715f971ed9` |
| resolver SHA-256 | `36bdc79c3d13338285ccf70e6949fc46f2f71369123e2afb1620c925b311c980` |
| NpuSim SHA-256 | `c1635d56a99e54c8bd62307a8a7cc51280ff3817989c6c7eae9d0fc446cfae29` |
| hardware template SHA-256 | `2e040e5b6b80fc2e0869f26f9cf771c7e0714e9b04347c1be2b74d8ece6cc732` |
| simulation SHA-256 | `337f5bc3f195489e8c6ef8830d83ce709a3e79a434401bd545504bf543420afb` |
| mapping SHA-256 | `99a357b646bc6d0d81ac188c8bfffcbf6ab8f8f72a5d262fe81624f6f9a9a66c` |
| release binding ID | `flexible_mesh_release_binding_c9159583ad0c5853` |
| 全局 binding digest | `03976915393b028b40acd97f04c669b838ad3185c6db762208722786401769a8` |

### 8.2 最终代表矩阵

| Family | Cases | Executions | Runtime verified | Repeatability verified |
|---|---:|---:|---:|---:|
| Dense train | 8 | 16 | 8 | 8 |
| MoE inference | 8 | 16 | 8 | 8 |
| MoE train | 8 | 16 | 8 | 8 |
| MeshSlice AG | 8 | 16 | 8 | 8 |
| MeshSlice RS fallback | 8 | 16 | 8 | 8 |
| MeshSlice AR fallback | 8 | 16 | 8 | 8 |
| 合计 | 48 | 96 | 48 | 48 |

failure count、stage failure count、repeatability mismatch 和十项 residual 非零
execution 均为 0；finalizer/resolver/NpuSim 各 96 次成功，ProgramIO
resolved/applied-actual-artifact-SHA/verify-pass 三阶段各 96 次成功。

### 8.3 派生完成状态与合同

`dense_train_rect_complete`、`moe_infer_rect_complete`、
`moe_train_rect_complete`、`meshslice_all_rect_complete`、
`workload_contract_complete`、`flexible_mesh_workloads_complete` 均由最终矩阵
派生为 `true`；`exhaustive_runtime=false`、
`exhaustive_rect_runtime_complete=false`。

- completion ID：`flexible_mesh_representative_completion_53c896fd38d74290`
- evidence matrix ID：`flexible_mesh_representative_evidence_matrix_58dd4781ca716df6`
- output root：`/workspace/build-release-final/flexible-mesh-release-representative-completion-c1635d56`

| Contract | Evidence ID |
|---|---|
| schema | `flexible_mesh_contract_evidence_19fd8753fa3820be` |
| axis | `flexible_mesh_contract_evidence_6ea0927f66caf162` |
| group | `flexible_mesh_contract_evidence_039d75d8a548a8a5` |
| capacity | `flexible_mesh_contract_evidence_a851c0e6a670bf40` |
| fallback | `flexible_mesh_contract_evidence_57a66dbb51715aed` |
| runtime_evidence | `flexible_mesh_contract_evidence_9301f35a1f7aeed7` |
| stable_ids | `flexible_mesh_contract_evidence_957c915b11fab0e7` |

## 9. 最终 merge 与聚焦回归

最终 merge 命令：

    PYTHONPATH=/workspace python3 llm/test/frontend/integration/run_flexible_mesh_release_merge.py \
      --binding /workspace/build-release-final/flexible-dense-release-representative-binding-c1635d56/release_binding.json \
      --runtime-root /workspace/build-release-final/flexible-dense-release-representative-binding-c1635d56 \
      --runtime-root /workspace/build-release-final/flexible-moe-release-representative-binding-c1635d56 \
      --runtime-root /workspace/build-release-final/flexible-meshslice-release-representative-binding-c1635d56 \
      --output-dir /workspace/build-release-final/flexible-mesh-release-representative-completion-c1635d56 \
      --validation-scope representative

命令 exit 0，输出为 `scope=representative cases=48 executions=96`。三个 family
runner 根分别贡献 Dense 8/16、MoE 16/32、MeshSlice 24/48，且 binding 完全一致。

| 聚焦回归 | 结果 |
|---|---|
| finalizer selftest | PASS |
| ProgramIO selftest | 7/7 PASS |
| MeshSlice profile focused | 13/13 PASS，18.625s |
| P2P lifetime focused | 1/1 PASS |
| odd-rank barrier leader | 1/1 PASS |
| 2x2 canary | finalizer/resolver/NpuSim exit 0，residual 0 |

以上仅列有独立证据的聚焦回归，不据此推导或伪造总 suite 计数。

## 10. 证据隔离与不声明内容

- 旧 tool SHA、旧 binding、debug build、被中止的 shard、单次 canary 和诊断 root
  不进入正式 completion matrix。
- 缺 `case_evidence.json` 的半成品目录不算完成；resume 必须重新校验 raw
  manifest/artifact/ProgramIO/stdout/config 和当前 producer 输出。
- 任何 capacity/finalizer/runtime 失败都保留为诊断，不通过放宽 C++ validator、
  复制 marker、忽略 residual 或手填 flag 解决。
- timing-only 不等于数值正确性；本报告不声明 loss、gradient、optimizer 数值，
  也不声明性能收益或最优 mapping。

## 11. 最终结论

本轮完成六类 family 的完整 100-shape 静态覆盖，以及精确
`1x1, 1x4, 4x1, 2x2, 2x3, 3x2, 3x3, 10x10` 八个 mesh 的代表性真实 runtime：
48 cases/96 executions、各 family 8/16、runtime/repeatability 48/48，
failure/stage/residual 均为 0；统一 binding、七类 contract receipt 和 completion
均已闭环。发布范围仍为 `validation_scope=representative`，
`exhaustive_runtime=false`、`exhaustive_rect_runtime_complete=false`，不声明
600/1,200 全形状真实 runtime，也不声明数值 functional correctness。
