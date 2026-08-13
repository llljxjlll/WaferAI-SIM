# P3 开发记录：计算指令接入与成本模型验证

阶段：P3——计算指令接入与成本模型验证

阶段状态：完成；代表性 JSON/program 精确差分、P8-B 跨阶段 region 生命周期以及 fresh Release/Debug 63/63 均已闭环

记录日期：2026-08-11（UTC）

提交/变更集：工作区实现，尚未提交

## 计划确认与阶段边界

- 本阶段对照《编译产物指令集开发计划》9.1～9.4，实现 public NPU compute 的注册、lowering、one-shot `SRAM_BIND`、统一语义计算、成本快照和精确 trace。
- P0 D-24 已冻结：public `SRAM_BIND` 跳过中间 MEM/SYNC，只由下一条成功开始的 COMPUTE 消费并立即失效；legacy JSON `Set_addr` 继续保持 persistent，二者使用独立状态。
- P0 D-29 已冻结：`MATMUL_MLA`、`MATMUL_PD`、`ATTENTION_PD`、`ROPE_PD` 虽有稳定编号，但在 program context/adapter 完成前 capability gate 保持关闭。本阶段没有用静态 shape 或通用 helper 强转绕过 gate。
- P3 只记录计算语义、绑定生命周期、成本和相关 program fixture。P4 的 region/sync/event、P5 的 endpoint/payload 不计入 P3 完成项；释放/重命名后的跨能力绑定以及 output region 跨阶段存活仍由 P4/P8 验收。
- 开发计划 9.1 要求的代表性短模型片段已由同一硬件配置下的 `MATMUL -> ATTENTION -> LAYERNORM -> GELU` program/legacy JSON 差分闭合；比较项包括 opcode、参数、地址、标签、exact ops/cycle、完成顺序和 RegionTable 生命周期。

## 按计划完成的开发

### 17 个 public compute 的统一语义入口

以下 17 个已发布计算 opcode 已纳入同一共享 evaluator，并分别覆盖最小 shape 与典型 shape：

- 矩阵、卷积和注意力：`MATMUL`、`CONV`、`MAXPOOL`、`ATTENTION`；
- MoE：`GATE`、`MOE_MATMUL`；
- 激活和归一化：`GELU`、`SILU`、`SWIGLU`、`RELU`、`RESIDUAL`、`LAYERNORM`、`RMSNORM`；
- 位置编码和拆并：`ROPE`、`SPLIT_MATMUL`、`MERGE_MATMUL`；
- 辅助计算：`DUMMY`。

共享 evaluator 对外返回规范化的 EXU/SFU/VEC ops，统一检查必需参数、负值、非法零除数、整数乘法/累加溢出和硬件配置。record schema、lowering 与 production Prim 复用同一组 public 语义定义，避免在 ISA 层复制另一套公式。

`BATCHNORM`、`SPLIT_CONV`、`MERGE_CONV` 保持 known-but-unsupported，`GEMM_REDUCE_SCATTER` 保持 experimental；GPU primitive 不进入 public NPU compute 集合。MLA/PD 四项按 D-29 保持 capability disabled，加载期拒绝而不是以零周期成功。

### SRAM_BIND：72-byte payload、PrimId 52 与 one-shot RAII

- 外部 `SRAM_BIND` record 使用固定 72-byte payload；该数字不包含 8-byte external record header。
- lowering 生成独立内部 `Sram_bind_oneshot`，固定 `PrimId=52 (SRAM_BIND_ONESHOT)`，strict-only 线格式为 8 个 128-bit segment：metadata、6 个输入标签段和 1 个输出标签段；每段低 8 位均为自身 PrimId。
- 输入标签数量限制为 1～16；空输出、非法标签、段数、segment ID、reserved/padding、截断和重复 decode 均严格校验。
- `PrimCoreContext` 为 program mode 维护独立 pending binding，不复用 legacy `Set_addr` 的 active/persistent 状态。重复 bind、compute 缺 bind、arity 不匹配和程序末 dangling bind 均 fail-fast。
- `NpuBase` 在下一条 NPU compute 成功开始时以 RAII 交换 active labels，并立即清除 pending；中间 MEM/SYNC/COMM 不消费 binding。计算成功或后续执行抛异常时，RAII 析构都恢复原 active labels，避免 one-shot 状态泄漏到下一条 compute。
- helper 的候选构造、label intern、serialize 和 program 提交保持事务式；pre/post-intern 失败测试确认旧程序状态不变，失败的候选不会污染全局标签表。
- legacy JSON helper 不生成该 Prim，旧 `Set_addr` 的 persistent 位串、段数和执行语义保持不变。

### 共享成本模型、快照和 exact trace

- 新增共享 `EvaluatePublishedNpuOps`，作为 17 个 public compute 的参数语义与 ops 计算唯一入口；production Prim 和独立 selftest 使用同一结果结构。
- `CalculateNpuCost` 保留已冻结的单位换算：EXU 的历史浮点截断、SFU/VEC 整数除法、计算周期取三单元最大值，以及 DRAM overlap 的 `max(compute-dram, 0)`。
- `NpuBase::writeOutputData` 只消费一次共享结果并保存 `NpuCostSnapshot`；ISA/lowering 不重复计费，Worker 也不再次推导成本。
- Worker 在实际 Prim 完成后输出 exact `Compute_cost` trace，字段包括 `compute_cycle_ns`、`dram_time_ns`、`exu_cycle_ns`、`exu_ops`、`sfu_cycle_ns`、`sfu_ops`、`vec_cycle_ns`、`vec_ops` 和 `overlap_delay_ns`。
- 独立成本 oracle 覆盖 EXU/SFU/VEC 最大值选择、DRAM overlap、历史截断、无效硬件参数和溢出；同时核对一组 production Prim 与共享公式一致。

### DUMMY 与 MoE 修复

- `DUMMY` 不再是零成本 NOP；production `Dummy_p` 通过共享 evaluator 固定产生 `exu_ops=10`，进入正常成本快照和 trace。
- `MOE_MATMUL` 检查 `0 <= K <= E_N`、expert 下标范围以及外部选择时 `selected_experts.size()==K`。
- 需要内部选择时，旧选择先清空，再生成唯一 expert；`selected_freq` 扩展至 `E_N` 并更新频次。原有 expert 静态数据访问和选择副作用继续保留，不被成本模型抽取丢失。

## 测试设计与实际证据

### 逐指令与成本 selftest

- 17 个 public compute 均有最小/典型参数向量；负例覆盖参数缺失、负数、非法零值、运算溢出、无效硬件和未发布 operation。
- `SRAM_BIND` wire 覆盖 golden、最大标签数、逐段 PrimId 破坏、padding/reserved、截断、重复 decode，以及 legacy mode 拒绝。
- binding 生命周期覆盖：bind 后直接 compute、bind 后穿插 LSU load/fence 再 compute、两次 bind 配两条 compute、一次 bind 配两条 compute、缺 bind、double bind、wrong arity、dangling bind。
- cost oracle 覆盖三计算单元的独立/组合选择、`max` 周期、DRAM overlap 只计一次、截断和溢出；`DUMMY=10` 作为固定 golden。
- MoE 覆盖合法/非法 K、expert 范围、外部选择数量、内部唯一选择和 frequency 副作用。

最近一次统一 ISA 输出中已获得以下分项证据：

| 自测模块 | 最近证据 | P3 相关说明 |
|---|---:|---|
| manifest/factory | 942 PASS | public compute 状态、PrimId 52 和 factory 集合纳入统一校验 |
| external record codec | 892 PASS | 包含 `SRAM_BIND` 72-byte payload schema/边界 |
| Prim wire | 526 PASS | 包含 one-shot 8-segment strict wire 与破坏矩阵 |
| Program Format | 632 PASS | artifact 容器基础回归 |
| record lowering | 244 PASS | compute/`SRAM_BIND` lowering 与 capability gate |
| program helper | 183 PASS | helper、transaction、whole-artifact 与诊断矩阵通过 |
| published NPU ops | 41 PASS | 含 17×min/typical 与参数负例 |
| NPU cost model | 22 PASS | 含独立 cost oracle、overlap、截断与溢出 |

program helper 的旧断言失败属于 P4 `GROUP_SYNC` unknown-group 诊断文案匹配，不是 P3 compute、one-shot binding 或成本语义失败；后续 latest-source 运行已获得 helper 183、published ops 41、cost model 22 全绿和 full CTest 63/63。由于工作树在该历史门后仍有源码变化，本文不把它虚报为最终 clean Release/Debug 结论。

### Program fixture、构建与回归

- 正向 fixture `BOUND_DUMMY` 通过真实 artifact 路径执行 `SRAM_BIND -> DUMMY`，worker 必须出现 `Core 0 end compute primitive Dummy_p` 并完成 START/DONE。
- 负向 fixture 覆盖缺 bind 和错误 arity，并要求在打印 `Loaded Program Format` 前拒绝；helper 级测试另覆盖 double bind、一次 bind 后连续两条 compute 和 dangling bind。
- P3 收口时 CTest 为 27/27 通过。
- 后续 P3/P4 定向 program 集合为 22/22 通过；其中只把上述 P3 compute/bind fixture 作为本阶段证据，GROUP_SYNC/EVENT/SRAM lifecycle 等 P4 用例不计入 P3。
- latest-source 历史构建已获得 full CTest 63/63；最新工作树的最终 clean Release/Debug 仍需统一复跑，二者证据层级不混用。
- 新增 `program_legacy_compute_diff`：单核真实 SRAM 路径执行 `MATMUL -> ATTENTION -> LAYERNORM -> GELU`，program 侧每条 compute 前使用 one-shot bind，legacy 侧使用等价 persistent 标签配置。两路 manifest 的 opcode、参数、DRAM offset 和输入/输出标签完全一致；四条 exact `(exu_ops,sfu_ops,vec_ops,compute_cycle_ns)` 依次为 `(0,0,32768,512)`、`(4096,64,128,5)`、`(0,4,2060,32)`、`(0,256,1024,16)`。归一化首条 CONFIG 开销后，完整 B/E span 和完成顺序一致；input 与三个中间标签均释放，仅静态 layer 权重与最终 result 存活。该 CTest 已实际 1/1 PASS，原 program smoke/missing-bind/wrong-arity 子集 6/6 PASS。

## 开发、评审、修改、复测闭环

1. 按 P0/P3 计划先冻结 public compute 集合、D-24 one-shot 与 D-29 gate，再实现独立 Prim/wire，未修改 legacy `Set_addr` 语义。
2. 开发 shared evaluator，把分散的 public 参数语义和 ops 计算收敛到公共入口，并让 production Prim、成本快照和 selftest 使用同一结果结构。
3. 评审确认 ISA 层不能重复 `NpuBase::writeOutputData` 计费；修改为一次生产计算、一次 snapshot、一次 exact trace。
4. 评审发现 `DUMMY` 零成本与 P3 golden 冲突后，改为固定 `exu_ops=10`；MoE 同时补齐 K/expert 验证并保留选择频次副作用。
5. 评审 one-shot 异常路径后，以 RAII 保证 active labels 恢复，并增加 MEM/SYNC 穿插、double/missing/dangling、arity 和 post-intern rollback 负例。
6. 复测已获得 P3 27/27、P3/P4 定向 program 22/22、代表性 compute differential 1/1、原 program 子集 6/6、helper 183、published ops 41、cost model 22，以及 latest-source 历史 full CTest 63/63；最新源码的 clean Release/Debug 统一复跑仍是阶段关闭动作。

## 与开发计划 9.1～9.4 的差距

- 9.1 已完成：program 可承载并执行 public compute，参数/ops/成本/副作用均有共享语义来源；代表性四算子片段已在同一硬件配置下与 JSON 入口完成全链精确差分。
- 9.2 的 P3-owned 实现任务已完成：17 个 published compute、one-shot `SRAM_BIND`、单次成本换算、精确 trace、MoE 副作用和 unavailable/capability gate 均已落地。
- 9.3 的 unit、wire、helper、program 正负例及代表性多算子 JSON/program 对照均已覆盖；释放/重命名后的跨能力 binding 仍按计划留给 P4/P8 联合验收。
- 9.4 的代码评审与修改闭环已完成；退出判定仅保留当前 P4/P5 集成源码统一重建后的最终全量回归，以及按计划归属 P4/P8 的跨能力 region 联合验证。

## 阶段退出条件

- [x] 计划 9.2 的 P3-owned 代码实现完成。
- [x] 17 个已发布 compute 具有 codec/lowering、最小/典型功能语义和成本 oracle 覆盖。
- [x] `SRAM_BIND` 的 D-24 one-shot helper/runtime/wire 规则一致，异常路径恢复且 legacy `Set_addr` 不变。
- [x] 成本只结算一次，并有 shared evaluator、snapshot 与 exact trace 支撑。
- [x] `DUMMY=10` 与 MoE 参数/选择频次副作用已修复并纳入测试。
- [x] stub、experimental、GPU 和 D-29 MLA/PD capability 均确定性拒绝，未以零成本或不完整上下文执行。
- [x] P3-owned pending/active label 临时状态在成功与异常路径均回收，无 one-shot residual。
- [x] 完成代表性 MATMUL/ATTENTION/归一化/激活短片段的 JSON/program 指令数、ops、周期和输出标签生命周期精确对照。
- [x] P8-B 已完成 ALLOC/BIND、跨 P2P/collective、RENAME 8→9、rebind、后 compute、FREE 的联合验证。
- [x] latest-source 历史统一证据已补入：program helper 183、published NPU ops 41、NPU cost model 22、full CTest 63/63。
- [x] fresh Release 与 Debug 均完成统一 ISA 与 full CTest 63/63，确认最终工作树没有回退。

是否允许继续后续集成：允许。

结论：P3 完成；阶段正负例、代表性 JSON/program 精确差分、P8-B 跨阶段 lifecycle 与 fresh Release/Debug 63/63 均通过。
