# NoC 集合通信重构 R6 开发记录

日期：2026-08-03

## 阶段定位

R6 首次交付计划要求的 `profile=reduce_only`：只加速 Reduce 家族，不改变 Broadcast
家族。它用于把 in-network reduce 的收益与 multicast broadcast 的收益拆开测量，也避免沿用
旧 Tier2 “reduce 与 broadcast 必须同时开启”的耦合。

## profile 语义

| collective op | reduce backend | result/distribution backend |
|---|---|---|
| Broadcast | 无 | Tier0 ordinary unicast |
| AllGather | 无 | 每 source 的 Tier0 ordinary unicast |
| Reduce | stream DCA | root 本地完成 |
| ReduceScatter | stream DCA | quotient/remainder ordinary Scatter |
| AllReduce | stream DCA | root 到其余 rank 的 ordinary unicast |

该 profile 只为 reduction op 编程 reduce tree，不创建 multicast tree。尤其 AllReduce 的
result distribution 不能因为历史 Tier2 实现而进入 `AtomicMulticastFork`；运行时
`COLL_V4_TX/RX` 必须为零。

## 开发过程

### 1. workload 展开

- Reduce/ReduceScatter/AllReduce 为所有 rank 创建 stream TX，root 提前创建异步 RX
  session；N=1 仍走 CENTER-to-CENTER stream bypass，但 DCA issue 为零。
- reduce 完成 barrier 保证任何 result distribution 都看见最终值。
- ReduceScatter 用 `CollRankCountOffset` 做 quotient/remainder 划分，非整除 count 无 tail
  丢失；AllReduce 用普通 root-to-peer flows。
- Broadcast/AllGather 继续使用已冻结的 source-phased Tier0 planner，不编程无用 tree。

### 2. 数值、dtype 和 root 组合

整数 exact 覆盖 UINT8/INT32/INT64 的 SUM/MAX。FP32 按 R5 分成 `fp_exact` 与
`timing_only`；FP16/FP8 只允许 timing-only。group 可以非连续，root 只要求在 group 内；
tree node 按实际 mesh rank 而不是 group 下标建立。

### 3. 普通 unicast 混合流量补强

初版 runner 已覆盖 V1 的 tag namespace gate 和 R5 CORE+DCA pool contention，但评审指出
缺少“新 collective DCA 与普通 Send/Recv 在 production Router 同时出现”的直接证据。

补强用例使用 2×2 mesh：group `[0,2]` 执行 UINT8/SUM Reduce 到 root 0；与此同时 host
向 core 3 注入普通工作，core 3 用合法普通 tag 0 向 core 0 发送 DATA。core 3 不属于
collective group，避免 host-fed sender 的 terminal 生命周期被集合工作覆盖。两条业务流在
Router 2 的 root-bound output 共享现有仲裁和 credit。

用例不仅检查同一运行中出现两种日志，还断言：

- `[COLL_SHARED] router=2 output=3 normal_flits=4 collective_flits=18`，同一真实 output
  的两类计数都非零；
- 2 个 rank 各产生一条 `COLL_STREAM_TX`，root 只产生一个 result 且 `value=verified`；
- regular tag 位于普通 namespace，collective stream 使用独立完整 key；
- normal/collective data、control、tree、stream、DCA、barrier、endpoint、DTE token 全 drain。

开发时曾尝试让 host-fed core 同时成为 collective rank；该安排会使普通 source 的 DONE
生命周期与 terminal collective 叠加并触发 protocol watchdog。最终测试把普通 sender 放在
group 外，同时保留同 output 竞争，因此覆盖的是预期并发语义，而非测试终止协议的歧义。

## 独立自测

新增 `npusim --coll-r6-selftest`，7/7：

1. profile 选择 DCA、拒绝 multicast；
2. Broadcast/AllGather 保持 unicast；
3. Reduce/ReduceScatter 只创建 reduce tree；
4. AllReduce result 不进入 multicast；
5. N=1 bypass 与 N=4 issue 公式；
6. 67 elements/4 ranks 的 `17/17/17/16` slice 和 offset 全覆盖；
7. profile/backend 可观测字符串与实际选择一致。

该入口把 R6 的 profile composition 合同从综合 runner 中独立出来；runner 仍负责验证真实
Router、数值和生命周期。

## 端到端验收证据

- R6 op 矩阵：Broadcast、AllGather、N=1/N=2 Reduce、N=4 ReduceScatter、N=4
  AllReduce 全部通过；含非连续 group、不同 root 和 tail count。
- R5 value-mode/CORE contention 与 R7 multicast 组合后，统一
  `run_test_coll_r6_r7.py` 为 19/19。
- mixed production case 明确获得同 output `4 normal / 18 collective flits`，数值验证和
  drain 同时通过。
- standalone Broadcast/AllGather/AllReduce result 的 `COLL_V4_TX/RX=0`。

## 已知限制

- Tier0 distribution 保持 source-phased 串行，不是 ring/tree 并行算法；这正是 reduce-only
  配置要保留的对照边界。
- 当前同 output mixed 回归覆盖一条普通 flow 和一个 reduce instance；不宣称穷举任意多个
  regular/collective 流的公平性组合。
- FP16/FP8 仍为 timing-only；跨 die collective 和 SMART transport 均不支持。
- ReduceScatter 验证 slice 元数据和 exactly-once 时序，仿真器不搬运完整真实 tensor 存储。

R6 退出条件：存在一个可独立运行和测量的“只改变 reduce、不改变 broadcast”配置，且其
DCA 流量能与普通 unicast 在 production Router 共存并最终 drain。
