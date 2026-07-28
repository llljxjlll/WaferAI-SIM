# NoC 集合通信 V0 开发日志

日期：2026-07-25

## 交付范围

V0 只冻结并实现无运行时副作用的公共契约，不接 workload、DTE 执行器或 router：

- `CollTxKind`、`CollRxKind`、完整 `CollOp`、固定首版算法、reduce op 和 dtype；
- `CollectiveKey`、`PacketKey` 与实例隔离比较语义；
- `CollDescriptor` 及启动前校验；
- count 不能整除 rank 数时的 quotient/remainder 分片；
- behavioral/oracle 专用时延纯函数，含取整、非法输入和 uint64 overflow；
- 128-bit 多段 descriptor codec，含 magic/version/段数/enum/group 校验；
- C++ self-test 命令行入口和独立 Python oracle/runner。

## 冻结决策

- ReduceScatter 固定为 Reduce-to-root + Scatter。
- AllReduce 固定为 Reduce-to-root + Broadcast。
- `tag_id` 不能替代 `(group_id, collective_id, epoch)`。
- cycle-accurate backend 后续不得叠加本 V0 闭式网络时延。
- Gather/Reduce 的来源到齐是完成条件；V0 endpoint 公式只表达 commit/alignment service。
- group wire 成员使用 16-bit core ID；超过仿真器 endpoint wire 能力的拓扑应在后续配置期拒绝。
- `gather_reorder_depth=0` 表示理想无限深；有限容量和背压在 V2 实现。

## 文件

- `llm/include/dte/coll_types.h`
- `llm/include/dte/coll_latency.h`
- `llm/include/dte/coll_codec.h`
- `llm/src/dte/coll_v0_selftest.cpp`
- `llm/include/dte/dte_unit.h`
- `llm/unittest/npusim.cpp`
- `llm/test/noc_collective/oracle.py`
- `llm/test/noc_collective/run_test_coll_v0.py`

V0 没有运行时配置项或示例配置：功能尚未接入仿真路径，提前增加开关会形成无效果配置。配置 schema 在 V1 与 workload 展开一起加入。

## 验收结果

- CMake configure：PASS。
- `cmake --build build --target npusim --parallel 2`：PASS。
- `./build/npusim --coll-v0-selftest`：32/32，PASS。
- `python3 llm/test/noc_collective/oracle.py`：PASS。
- `python3 llm/test/noc_collective/run_test_coll_v0.py`：2/2，PASS。
- `./build/npusim --dte-v0-selftest`：64/64，PASS。
- `./build/npusim --d2d-v0-selftest`：308/308，PASS。
- `python3 llm/test/noc_congestion/run_test_noc_congestion.py`：4/4，冻结值保持 14781/29109、14833/45441。

## V0 未实现项

以下属于后续版本而非 V0 遗漏：全局 workload 声明和每 core 展开、Collective primitive、DTE TX/RX、Gather reorder 状态机、multicast router、in-network reduce、运行时 tier/backend 配置。
