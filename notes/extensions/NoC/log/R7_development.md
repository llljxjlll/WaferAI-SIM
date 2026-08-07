# NoC 集合通信重构 R7 开发记录

日期：2026-08-03

## 阶段定位

R7 交付改进后的 `profile=reduce_broadcast`：归约使用 R3～R6 的 stream DCA，分发使用已经
验证的 Router atomic multicast。它替换旧 Tier2 two-segment/逐 element 延迟模型，同时保留
V4～V6 的 multicast exactly-once 和 tree lifecycle 合同。

## op 到 backend 的组合

| collective op | reduce tree | multicast tree |
|---|---:|---:|
| Broadcast | 否 | 是，source 单注入 |
| AllGather | 否 | 是，每 source 独立单注入 |
| Reduce | 是 | 否 |
| ReduceScatter | 是 | 否，结果仍为普通 Scatter |
| AllReduce | 是 | 是，root result 单注入 |

这个映射防止两类常见误配：为纯 Reduce 创建永远不会消费的 multicast tree，以及把
ReduceScatter 的不同 slice 错当成一份可广播 payload。

## 开发过程

### 1. stream reduce 与 atomic fork 组合

Reduce 前半段完全复用 R6：每 source 一个 stream，Router 节点做 binary vector reduction，
root 得到唯一 result。需要广播的 op 在 reduce-complete barrier 后才开始 single injection，
因此 multicast 不可能读取半完成结果。

`AtomicMulticastFork` 仍执行 `CanCommit(all branches) -> Commit(all branches)`；任一输出无
credit 或被另一个完整 collective instance lock 占用时，所有副本均不提交。lock key 包含
tree id、CollectiveKey、phase、chunk，不复用普通 unicast tag。

### 2. tree 编程和释放

- Broadcast：一个 tree、一次 source TX、N-1 个 endpoint RX。
- AllGather：每个 source 使用独立派生 tree id 和 barrier/release，避免多个 source 共享
  branch lock 或过早释放。
- AllReduce：reduce tree 和 result multicast tree 属于同一 collective lifecycle；最终
  barrier 的最后离开 rank 才释放。
- Reduce/ReduceScatter：只释放 reduce registry，不要求存在 multicast entry。

tree-id 稳定 hash 冲突、release mismatch、unknown release 继续由 V6 gate 拒绝。最终 residual
包含 tree table、reduce nodes、fork refs、pending headers、stream/DCA 和 endpoint session。

### 3. 背压组合

stream egress 与 multicast/normal wire 都进入 production `buffer_o`。DCA result FIFO 满时
stream engine 停止接收后续 completion；multicast 任一 branch 阻塞时整 fork 停止；普通
DATA 仍按原 output 仲裁前进。三者没有额外旁路或无限队列。

R6 新增的 mixed production 回归同时成为 R7 的相邻防线：它证明打开集合路径不会让 normal
DATA 被 wire magic 误判，且 Router shared-output accounting 能同时观察两类 flit。

## 独立自测

新增 `npusim --coll-r7-selftest`，6/6：

1. combined profile 同时选择 stream DCA 和 multicast；
2. Broadcast/AllGather 只使用 multicast；
3. Reduce 只使用 reduce tree；
4. ReduceScatter 不错误广播 slice；
5. AllReduce 同时拥有 reduce/result-multicast tree；
6. production profile 固定 `stream_v2`，不能静默回退 legacy wire。

合同自测专门验证 profile composition；V4/V6 selftest 继续负责 fork 原子性、tree collision 和
release 负例，production runner 负责 exactly-once 与 drain。

## 端到端验收证据

- 四 rank Broadcast：`COLL_V4_TX/RX=1/3`。
- 四 rank AllGather：4 个 source 各单注入，合计 `4/12`，tree 分别释放。
- Reduce/ReduceScatter：`COLL_V4_TX/RX=0/0`，仍有 4 条 stream TX。
- AllReduce：reduce issue/value 与 R6 一致，result multicast `1/3`。
- R5～R7 production runner 在加入两个独立 selftest 和 mixed traffic 后为 19/19；所有 case
  的 tree、fork、stream、DCA、barrier、endpoint 和 token 最终为零。
- V0～V6 合同及 frozen NoC/D2D 回归保持不变，legacy backend 只能经显式 debug gate 使用。

## 已知限制

- AllGather 沿原 planner 对 source 分阶段，尚未模拟多个 source 同时注入 multicast 的更激进
  调度；每 source exactly-once 是当前合同。
- Atomic fork 是 all-or-nothing，因此单慢分支会拖慢所有分支；这属于 VCT 风格背压语义，
  不是独立 branch buffering 模型。
- conventional 同 die tree 已验证；跨 die multicast/reduce 与 SMART bypass 未实现。
- ReduceScatter 仍使用普通 Scatter，这是 payload 各异的有意边界，不作为 R7 性能缺陷。

R7 退出条件：旧 Tier2 的 reduce 计算被流式、vector-lane、共享资源 DCA 替换，同时单注入
multicast 和最终 tree release 的既有正确性保持不变。
