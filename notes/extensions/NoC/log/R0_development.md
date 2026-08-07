# NoC 集合通信重构 R0 开发日志

日期：2026-08-03
状态：已完成

## 1. 阶段范围

R0 只冻结旧 Tier2 异常和论文对齐的新公共契约，不修改生产 Router、planner、runtime、
multicast 或 in-network reduce 数据通路。旧 V5/V6 backend、wire 和 trace 继续原样运行，
后续 R1～R8 必须通过显式的新 generation/backend 接入。

## 2. 新冻结的契约

新增 `llm/include/dte/coll_refactor_contract.h`，冻结以下接口：

- `ReduceStreamKey/ReduceBeatKey`：用 collective、phase、stream、stage、vector beat
  唯一标识归约工作；
- `DcaTiming`：pipeline latency 与 initiation interval 独立；第 `i` 次 issue 为
  `first+i×II`，completion 为 `issue+L`；
- `VectorWork/VectorLaneMask`：`lanes=vector_bits/dtype_bits`，尾 beat 使用连续前缀
  lane mask；
- `DcaRequest/DcaResult`：每个请求严格为两个等宽 vector operands，使用非零 tag 和
  完整 beat key 对齐返回；
- `ReduceStreamHeader`：冻结 stream wire v2 的逻辑 header 及 physical flit/vector beat
  几何校验；
- `AsyncSessionProgress`：只有 TX、RX 均完成且 header/operand/issue/inflight/result
  全部清零时 session 才完成；
- `TraceContract`：`LEGACY_V5+LEGACY_TWO_SEGMENT` 与 `REFACTOR+STREAM_V2` 双向绑定，
  禁止跨 generation 混记 header/data/DCA issue/completion/inflight/stall。

向量工作量使用：

```text
B = ceil(total_elements / lanes)
pairwise_issues_per_beat = input_count - 1
total_issues = B × (input_count - 1)
```

因此单输入为 0 issue，双输入为 `B`，三输入为 `2B`。C++ 逐 element helper 未来只
负责值语义，不代表串行时序。

`CollDcaServiceCycles(max(comp,ceil(payload/128))+54)` 保留，但已在代码、需求和旧计划
中标为 V5/V6 legacy-only；新生产实现不得调用该公式。

### 2.1 R0 冻结契约修订（R2 评审补记，2026-08-03）

R1 配置正交化时曾形成两个同义但底层数值相反的枚举：

- R0 协议契约：`ReduceWireVersion{LEGACY_TWO_SEGMENT=0, STREAM_V2=1}`；
- 配置层：`NocCollReduceWire{STREAM_V2=0, LEGACY_TWO_SEGMENT=1}`。

当时全部代码按枚举名引用，且 R3 codec 尚未实现，因此没有运行时错误或历史 packet
迁移；但未来若跨层 `static_cast<uint8_t>` 会反转 wire 语义。正式修订为：

- `coll_wire.h` 是 `coll_refactor::ReduceWireVersion` 的唯一声明位置；
- R0 冻结名称与数值 `LEGACY_TWO_SEGMENT=0, STREAM_V2=1` 不变；
- `coll_refactor_contract.h` 改为 include 公共头；
- `NocCollReduceWire` 改为同一类型的别名；
- 两个冻结数值由 `static_assert` 固定。

该修订改变声明归属并删除重复配置枚举，不改变 R0 的协议语义、逻辑 header 或测试
结果。R0 19/19、R1 42/42 及全部 frozen regression 均通过。

## 3. Legacy 性能参考

负载为 2×2 mesh、4 ranks、root=0，依次执行 Broadcast 和 UINT8/SUM AllReduce。
实测时间及 flit-hop 与 `run_experiment_tiers.py` 的独立 XY oracle 完全相等：

| payload | chunks | Tier | time (ns) | normal hops | collective hops | total hops |
|---:|---:|---:|---:|---:|---:|---:|
| 128 bit | 1 | 0 | 912 | 12 | 0 | 12 |
| 128 bit | 1 | 1 | 780 | 8 | 3 | 11 |
| 128 bit | 1 | 2 | 740 | 0 | 12 | 12 |
| 512 bit | 4 | 0 | 1108 | 48 | 0 | 48 |
| 512 bit | 4 | 1 | 930 | 32 | 12 | 44 |
| 512 bit | 4 | 2 | 1288 | 0 | 48 | 48 |
| 2048 bit | 16 | 0 | 1894 | 192 | 0 | 192 |
| 2048 bit | 16 | 1 | 1524 | 128 | 48 | 176 |
| 2048 bit | 16 | 2 | 3448 | 0 | 192 | 192 |

新增 `run_test_coll_legacy_pressure.py` 固定旧 Tier2 的可扩展性边界：

- 4096 bit / 32 chunks：6328 ns 完成，全部状态 drain；
- 5120 bit / 40 chunks：退出码 3，稳定触发 `[PROTO_WAIT]`；
- 8192 bit / 64 chunks：退出码 3，稳定触发 `[PROTO_WAIT]`。

40/64 chunks 是 legacy 缺陷的参考，不是期望新实现继续失败。R4 的异步 TX/RX 必须
消除该边界，R8 才能用新结果替换性能结论。

## 4. 自动化测试

新增：

- `npusim --coll-r0-selftest`：19/19；
- `run_test_coll_r0.py`：Python oracle + C++ contract，2/2；
- `run_test_coll_legacy_pressure.py`：3/3；
- 三档实验：9/9。

冻结集合通信回归：

- V0～V6 selftest：32/32、12/12、14/14、14/14、13/13、20/20、11/11；
- V0～V6 runner：2/2、24/24、8/8、19/19、V4/V5 9/9、V6 6/6。

相邻和共享数据通路回归：

- NoC congestion：4/4，冻结时间 14781/29109 与 14833/45441 ns；
- D2D V0 runner：67/67 test groups，底层 pure-function 308/308；
- DTE V0/V3b/V4：64/64、21/21、19/19。

## 5. 已知边界

- R0 只有 contract 和测试，尚无 `DcaComputePool`、stream wire 编解码、binary stage
  展开或 Router production wiring；
- `ReduceStreamHeader` 是逻辑契约，具体 bit layout/codec 在 R3 冻结；
- `TraceContract` 定义 generation 隔离，生产 trace 的实际发射在 R4 接通；
- OpenSMART transport 本轮重构不建模；
- legacy 40/64 chunks watchdog 在 R0 中被保存而非修复。
