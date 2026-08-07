# NoC 集合通信重构 R2 开发日志

日期：2026-08-03
状态：已完成

## 1. 阶段范围

R2 实现并验证独立的 per-tile DCA ComputePool，证明 DCA 是可流水化的共享 vector/FPU
资源，而不是旧 V5/V6 的 Router 内逐 element 串行 ALU。该模型尚未接入 Router、
collective stream 或生产 compute path；实际 `dca_offload` reduction 仍由 R1 gate
在启动期明确拒绝。

### 1.1 R0 reduce-wire 冻结契约修订

R2 评审要求将此前的枚举统一作为正式契约修订留痕。原 R0
`coll_refactor::ReduceWireVersion` 已冻结
`LEGACY_TWO_SEGMENT=0, STREAM_V2=1`；R1 配置层一度另有数值相反的
`NocCollReduceWire`。虽然当时所有引用都使用枚举名且 R3 codec 尚不存在，没有触发
运行时错误，但任何跨层数值转换都存在反转风险。

修订后 `coll_wire.h` 持有唯一 `ReduceWireVersion` 定义；
`coll_refactor_contract.h` 只 include 它，`NocCollReduceWire` 是同一类型的别名，
并以 `static_assert` 冻结 0/1。R0 的名称、数值和语义均未改变；改变的是声明归属和
配置层重复枚举。该修订同时记录在重构计划 §12 与 R0 开发日志。

### 1.2 DCA 校验路径复核

R1 评审指出的“只要存在 collective 就无条件 `dca.Validate()`”已经修复，R2 保持并
复核该行为：

- `ParseNocCollectiveConfig` 始终严格解析已知 DCA 字段/枚举，但只有
  `result.enabled && result.UsesDcaOffload()` 才执行资源几何、FIFO、L/II 和
  value-mode 语义校验；
- baseline/endpoint 或 `enabled=false` 可以携带 dormant 的零宽度、零 FIFO、零
  L/II、`fp_exact` 参数而不阻止启动，R1 selftest 和 runner 已覆盖；
- `DcaComputePool` 一旦构造就代表 DCA 资源已实际实例化，因此构造函数无条件
  `config.Validate()`；R2 新增无效 FIFO 配置拒绝测试固定该边界。

## 2. 有限资源模型

新增：

- `llm/include/dte/coll_compute_pool.h`
- `llm/src/dte/coll_compute_pool.cpp`

配置资源映射为：

| 配置项 | R2 ComputePool 状态 |
|---|---|
| `operand_fifo_depth` | core/DCA 两类 pending request 的共享总容量 |
| `header_fifo_depth` | 已 issue、尚未进入 result queue 的 inflight context 容量 |
| `result_fifo_depth` | 已完成、尚未由消费者取走的 result 容量 |

每个 `Tick(cycle)` 最多发射一个二输入 vector request，但流水线中可同时存在多个 tag。
同周期先处理 completion，再尝试 issue，因此 cycle C 释放的 inflight context 可在 C
立即复用。`Residual()` 统计 pending、inflight、result 对象；`Drained()` 还要求
live-tag registry 为空，既不重复计数也不会漏掉 tag 生命周期泄漏。

## 3. L/II、仲裁与背压

每个 request 根据自身 dtype/op 查表：

```text
completion_cycle = issue_cycle + latency[dtype][op]
next_issue_cycle = issue_cycle + initiation_interval[dtype][op]
```

发射不等待前一个 completion。不同 latency 的 request 可以乱序完成，result 依靠完整
tag/key 恢复上下文。completion 按 `(scheduled cycle, submit sequence)` 确定性选择。

实现三种仲裁：

- `round_robin`：core 与 DCA 都等待时，从 core 开始交替；
- `core_priority`：core 优先，累计 DCA wait cycles；
- `dca_priority`：DCA 优先，累计 core wait cycles。

统计包括 submit/issue/completion/consume 数、pending/inflight/result 峰值、issue queue
backpressure、II stall、inflight stall、result backpressure 和两类 wait cycles。三处
有限容量解除后均可继续推进且最终 drain。

## 4. 值语义与 vector issue 语义

`ReduceIntegerVector` 只负责值：

- UINT8 SUM/MAX，SUM 按 8-bit wrap；
- INT32/INT64 SUM/MAX，MAX 按有符号二补码比较，SUM 按 dtype width wrap；
- tail mask 外的 lane 原样保留，不参与运算；
- FP32 不进入 integer-exact helper。

`integer_exact` request 必须携带两个完整 lane vector；`timing_only` request 禁止携带
值，避免性能测试误报 value verification。一次 pool issue 始终只有两个等宽 operand。
多输入工作量继续使用 R0 冻结公式
`vector_beats × (input_count-1)`，不会退化为 element×source cycles。

512-bit UINT8 对应 64 lanes，128-bit UINT8 对应 16 lanes。FP8 与 UINT8 具有相同的
8-bit lane 几何，但正式 FP8 dtype、timing profile 和值语义属于 R6，不在 R2 扩展
冻结 dtype 枚举。

## 5. Tag 生命周期与完成完整性

tag 从成功 submit 起保持 live，直到 result 被 `PopResult()` 消费。实现：

- live tag collision 拒绝；
- 有限 tag 空间自动分配、耗尽拒绝和 wrap；
- 只有已消费 tag 才可复用；
- pending/unissued、unknown、early 和 duplicate completion 分别确定性拒绝；
- result queue 满时 completion 返回 backpressure，后续 cycle 只重试一次状态，不生成
  重复 result。

## 6. 测试结果

新增：

- `npusim --coll-r2-selftest`：37/37；
- `r2_oracle.py`：独立验证多 dtype width/count/fan-in 的 vector work、L/II timeline、
  混合 latency 乱序完成、三种仲裁及整数 wrap；
- `run_test_coll_r2.py`：2/2。

冻结 collective 回归：

- R0/R1 selftest：19/19、42/42；runner：2/2、16/16；
- V0～V6 selftest：32/32、12/12、14/14、14/14、13/13、20/20、11/11；
- V1 runner：24/24；V4/V5：9/9；V6：6/6。

共享路径回归：

- NoC congestion：4/4，保持 14781/29109 与 14833/45441 ns；
- D2D V0：隔离运行 67/67 test groups，pure-function 308/308。

`git diff --check` 通过。

## 7. 已知边界

- R2 不接 Router、Sync/Match、stream-v2 codec 或 result reinjection；
- 任意 fan-in 的 binary stage 与 local feedback 属于 R3；
- Router production wiring 和真实 output/credit 竞争属于 R4；
- core 请求当前为 synthetic request，只验证共享仲裁契约；
- FP32/FP16/FP8 timing/value 支持按计划在 R6 扩展；
- R2 统计是 ComputePool 局部可观测量，production trace 接线属于 R4。
