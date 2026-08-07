# NoC 集合通信 Tier0/Tier1/Tier2 对照实验

日期：2026-07-28

## 1. 目的

使用完全相同的 workload 比较三档 NoC 集合通信配置：

- Tier0：集合操作全部展开成普通单播；
- Tier1：Broadcast 使用 Router multicast，AllReduce 仍使用 Tier0
  root-reduce + broadcast；
- Tier2：Broadcast 使用 Router multicast，AllReduce 使用 Router
  in-network reduce + multicast result distribution。

实验同时检查总时间、理论/实测 flit-hop、实际启用的数据面以及所有运行时状态
最终归零。总时间采用 cycle-accurate NoC 下仿真结束时的
`sc_time_stamp`；当前 `CYCLE=2`，即一个模型周期为 2 ns。

## 2. 负载与配置

- 2×2 mesh，core/rank 为 `[0,1,2,3]`，root 为 core 0；
- XY routing，`noc_payload_per_cycle=1`；
- 每轮依次执行：
  1. Broadcast；
  2. UINT8/SUM AllReduce；
- 两个集合操作使用相同 payload，分别测试 128、512、2048 bit，即
  1、4、16 个 128-bit chunk；
- 第一个 Broadcast 非 terminal，第二个 AllReduce terminal，保证两项工作
  全部完成后仿真才结束；
- 三档仅改变 `noc.collective.tier`，硬件、拓扑、group、root、payload 和
  collective 顺序均保持不变。

测试脚本：
`llm/test/noc_collective/run_experiment_tiers.py`

运行方式：

```bash
python3 llm/test/noc_collective/run_experiment_tiers.py
```

## 3. 理论分析

令 `F = ceil(payload_bits / 128)`。

在 2×2 mesh 中，从 root 0 到其他三个 rank 的 XY 单播路径长度分别为
1、1、2，因此逐目标单播 Broadcast 需要 `4F` flit-hop。三条路径的有向边
并集为 `{0→1, 0→2, 1→3}`，Router multicast tree 只需要 `3F`
flit-hop。

本 workload 的理论数据流量为：

| Tier | Broadcast | AllReduce | 普通 flit-hop | collective flit-hop | 合计 |
|---|---:|---:|---:|---:|---:|
| 0 | `4F` 单播 | `4F` gather + `4F` broadcast | `12F` | 0 | `12F` |
| 1 | `3F` multicast | Tier0 `8F` | `8F` | `3F` | `11F` |
| 2 | `3F` multicast | `6F` reduce operand + `3F` result | 0 | `12F` | `12F` |

Tier2 reduce 的 `6F` 来自每条 tree edge 上的两段 operand wire
（header + payload）。因此，当前两段 framing 下 Tier2 并不比 Tier1
减少总 flit-hop。

每个完整 UINT8 chunk 有 16 个元素。按
`max(count × (children−1), ceil(payload/128)) + 54`：

- leaf Router：`max(0,1)+54 = 55` cycles；
- Router 1（local + Router 3）：`max(16,1)+54 = 70` cycles；
- root Router 0（local + Router 1 + Router 2）：
  `max(32,1)+54 = 86` cycles。

最深 reduce path 的首 chunk DCA 服务和为 `55+70+86=211` cycles，
且每个 Router 只有一个串行 DCA 服务端。由此预期：

- 单 chunk 时，Tier2 可能因消除多组 REQUEST/ACK/DATA 和 endpoint
  phase 而获益；
- chunk 增多后，逐 chunk DCA `+54` pipeline 和两段 operand framing
  会超过其减少的 endpoint 协议开销；
- Tier1 应稳定优于 Tier0，因为它既减少 Broadcast tree 流量，又把三条
  逐目标 Broadcast flow 变为一次注入。

## 4. 实测结果

| payload | chunks | Tier | time (ns) | normal hops | collective hops | total hops | 相对 Tier0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 bit | 1 | 0 | 912 | 12 | 0 | 12 | 1.000× |
| 128 bit | 1 | 1 | 780 | 8 | 3 | 11 | 1.169× |
| 128 bit | 1 | 2 | 740 | 0 | 12 | 12 | 1.232× |
| 512 bit | 4 | 0 | 1108 | 48 | 0 | 48 | 1.000× |
| 512 bit | 4 | 1 | 930 | 32 | 12 | 44 | 1.191× |
| 512 bit | 4 | 2 | 1288 | 0 | 48 | 48 | 0.860× |
| 2048 bit | 16 | 0 | 1894 | 192 | 0 | 192 | 1.000× |
| 2048 bit | 16 | 1 | 1524 | 128 | 48 | 176 | 1.243× |
| 2048 bit | 16 | 2 | 3448 | 0 | 192 | 192 | 0.549× |

所有 9 次运行都满足：

- 理论 normal/collective flit-hop 与 `COLL_SHARED` 实测逐项精确相等；
- Tier0 未进入 V4/V5 数据面；
- Tier1 恰有一次 multicast Broadcast；
- Tier2 恰有两次 multicast Broadcast 和一次 value-verified reduce；
- `router_residual=0`、credit balanced；
- `COLL_DRAIN` 的 tree、reduce、barrier、reorder、endpoint、DTE token
  全部为零。

## 5. 对比结论

1. Tier1 是当前实现中最稳定的流量/周期优化。它将总 flit-hop 从 `12F`
   降到 `11F`，三个尺寸相对 Tier0 分别加速 1.169×、1.191×、1.243×。
   周期收益大于 8.3% 的流量降幅，是因为单份 multicast 注入还消除了
   三条逐目标 flow 的串行握手和 phase 开销。
2. Tier2 对单 chunk 有效：128 bit 时相对 Tier0 加速 1.232×，也比
   Tier1 快 40 cycles。此时减少 endpoint flow/phase 的收益大于一轮
   Router DCA 固定开销。
3. Tier2 当前不是大消息加速路径。4/16 chunks 时，性能分别只有 Tier0
   的 0.860×/0.549×。实测趋势与逐 chunk `+54` DCA pipeline、串行服务
   和两段 operand framing 的理论分析一致。
4. Tier2 的数值正确性与性能结论应分开：实验中的 UINT8/SUM 结果仍由
   root bit-accurate 验证，性能下降不代表归约结果错误。

## 6. 压力扫描发现的边界

默认通过矩阵之外还进行了 Tier2 payload 压力扫描：

- 4096 bit（32 chunks）可完成，`sc_time_stamp=6328 ns`；
- 5120 bit（40 chunks）及 8192 bit（64 chunks）触发 protocol progress
  watchdog，不能完成最终 drain。

当前 root 的 `REDUCE_TX` 会先注入所有 chunk，之后才执行 `REDUCE_RX`；
每个结果包含两段 256-bit wire，而 endpoint raw queue 的实际流控阈值是
`MAX_BUFFER_PACKET_SIZE=3` 个 wire。随着 payload 增大，已归约结果可能在
root 仍执行 TX 时返回并填满该队列，向网络传播背压；root 又尚未进入 RX
消费，形成 TX 等待网络、网络等待 root RX 的循环等待。Router Match
Buffer 的 operand capacity 才是 64，不能与 endpoint queue 混为一谈。

这不是三档对照公式的问题，而是 Tier2 大 payload 的实际可扩展性缺口。
修复方向应是按 chunk 交错 root TX/RX，或增加独立并发结果 drain 状态机；
不应简单扩大有限队列或提高 watchdog 阈值。32 chunks 只是本拓扑和本负载
下已经实测通过的点，不是协议保证的容量上限；完成修复前不能宣称 Tier2
支持任意大 payload。
