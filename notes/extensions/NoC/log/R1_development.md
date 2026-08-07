# NoC 集合通信重构 R1 开发日志

日期：2026-08-03
状态：已完成

## 1. 阶段范围

R1 完成配置正交化、规范展开、能力门禁和 backend 驱动的建树选择。新 DCA 计算池、
stream-v2 codec 和 Router production wiring 分别属于 R2～R4，因此本阶段接受并校验
`dca_offload` 配置，但遇到实际 reduction workload 时明确启动失败，禁止使用旧 V5
实现生成伪装成新 DCA 的结果。

## 2. Canonical 配置

新增 `llm/include/dte/coll_config.h` 与 `llm/src/dte/coll_config.cpp`。规范展开为：

| profile | broadcast backend | reduce backend | reduce wire |
|---|---|---|---|
| baseline | unicast | endpoint | stream_v2 |
| broadcast_only | multicast | endpoint | stream_v2 |
| reduce_only | unicast | dca_offload | stream_v2 |
| reduce_broadcast | multicast | dca_offload | stream_v2 |

兼容 alias `tier=0/1/2` 分别映射 baseline、broadcast_only、reduce_broadcast。profile、
tier 和显式 backend 同时出现时必须一致。评审确认旧全局
`SPEC_NOC_COLL_TIER` 已无消费者后将其删除；兼容性只保留在 JSON `tier` alias
到 canonical profile 的单向解析，不再维护有损的 profile→tier 反向映射。

配置层和协议契约层现在共同引用 `coll_wire.h` 中唯一的
`coll_refactor::ReduceWireVersion`：`LEGACY_TWO_SEGMENT=0`、
`STREAM_V2=1`。配置层的 `NocCollReduceWire` 只是同一类型的别名，并以
`static_assert` 冻结数值，消除了后续 codec 桥接时 legacy/stream 对调的风险。

旧 V5/V6 只允许以下显式 debug 配置：

```json
{
  "broadcast_backend": "multicast",
  "reduce_backend": "legacy_router_alu",
  "reduce_wire": "legacy_two_segment",
  "allow_legacy_backend": true
}
```

legacy backend 不得与 profile/tier 混用；缺 gate、错误 wire 或 unicast+legacy 均启动期
失败。V4/V5/V6 runner、三档 legacy 实验和压力 runner 已迁移到该显式配置。

## 3. DCA 与 transport 配置门禁

R1 解析 DCA 字段，但只有
`collective.enabled=true && reduce_backend=dca_offload` 才执行结构和 value-mode
语义校验。baseline、endpoint 与 disabled 配置可以携带 dormant DCA 参数而不影响
启动；未知字段、未知枚举等 JSON 拼写错误仍严格拒绝。

实际启用 DCA 时校验：

- `vector_bits`、`slice_bits`、`slices_per_tile`，要求
  `vector_bits=slice_bits×slices_per_tile`；
- `vector_bits%dtype_bits==0` 在 reduction descriptor 的实际 dtype 已知后校验，
  不再用 `%8 && %32 && %64` 等价的最大位宽硬编码；
- per-dtype/per-op `latency` 与 `initiation_interval`，必须为正；
- header/operand/result FIFO depth，必须为正；
- arbitration：round_robin、core_priority、dca_priority；
- value mode：integer_exact、timing_only；fp_exact 在确定性 helper 完成前拒绝；
- `noc.transport` 仅接受 conventional；smart 和未知值明确拒绝；
- multicast 或 DCA tree 遇到跨 die group 启动期拒绝。

结构默认值为 512-bit vector、8×64-bit slices、三个 FIFO depth=8、round_robin、
integer_exact、L=1/II=1。由于 R1 不执行新 DCA，这些值不构成性能结果；R2 自测和后续
性能实验应显式给出 L/II。

## 4. Backend 驱动的建树和执行

`config_helper_core.cpp` 不再按 `tier>=1/tier==2` 分支：

- baseline 不建立硬件 collective tree；
- broadcast_only 仅为 standalone Broadcast 建 multicast tree；endpoint AllReduce 的
  gather/result broadcast 均为 Tier0 unicast，不建 tree；
- reduce_only 的 reduction 只要求 reduce tree，不建立 multicast tree；standalone
  Broadcast 仍为 unicast；
- reduce_broadcast 的 DCA AllReduce 同时要求 reduce tree 与结果 multicast tree；
- legacy reduction 保留 V5/V6 的两个 registry 和生产数据面。

本阶段发现并修复一个集成问题：最初的规则会给
`broadcast_only + endpoint AllReduce` 误建 multicast tree，但 Tier0 AllReduce 不会
release 它，导致 9 点实验结束时 `tree_entries=4`。条件已收紧为只有
`dca_offload AllReduce + multicast backend` 才建立结果树，并增加专门 selftest。

## 5. 测试结果

新增：

- `npusim --coll-r1-selftest`：42/42；
- 独立 Python profile/tree-use oracle：通过；
- `run_test_coll_r1.py`：16/16，覆盖四 profile 实际 Broadcast 数据面、tier=2 不回退、
  inactive DCA 不影响 endpoint、按 workload dtype 校验 lanes、legacy gate、
  冲突/SMART/DCA 几何负例和跨 die multicast/DCA 拒绝。

冻结 collective 回归：

- V0～V6 selftest：32/32、12/12、14/14、14/14、13/13、20/20、11/11；
- R0 selftest/runner：19/19、2/2；
- V0～V6 runner：2/2、24/24、8/8、19/19、V4/V5 9/9、V6 6/6；
- 三档 legacy 实验：9/9，全部时间与 flit-hop 与 R0 基线相同；
- legacy 压力：32 chunks 6328 ns；40/64 chunks 保持预期 watchdog。

共享路径回归：

- NoC congestion：4/4，14781/29109 与 14833/45441 ns；
- D2D V0：67/67 test groups，pure-function 308/308；
- DTE V0/V3b/V4：64/64、21/21、19/19。

## 6. 已知边界

- `dca_offload` reduction production execution 要等 R4/R6；R1 的明确拒绝是安全门禁；
- stream-v2 当前只有 R0 逻辑 header contract，bit layout/codec 在 R3；
- DCA timing 参数尚未驱动事件，ComputePool 在 R2；
- FP32 reduction 和 fp_exact value mode 尚未实现；
- OpenSMART transport 明确拒绝，本轮不建模。
