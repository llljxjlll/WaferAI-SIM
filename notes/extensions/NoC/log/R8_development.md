# NoC 集合通信重构 R8 开发记录

日期：2026-08-03

## 阶段定位

R8 不再增加 backend，而是用统一 workload 收口四 profile 的性能、流量、压力、回归和文档。
核心问题是验证重构是否消除了旧 Tier2 的多 chunk 性能伪影：DCA 固定 latency 应只形成一次
pipeline fill，吞吐斜率应由 vector issue II/链路/真实争用决定。

## 实验设计

`run_experiment_profiles.py` 对同一个 2×2 mesh workload 运行：

- 一个 root=0、group `[0,1,2,3]` 的 Broadcast；
- 一个相同 group 的 UINT8/SUM AllReduce；
- payload 为 640 B、1 KiB、8 KiB、32 KiB；
- DCA `vector_bits=512`、`L=7`、`II=1`；物理 flit 为 128 bit。

四 profile 是 `baseline`、`broadcast_only`、`reduce_only`、
`reduce_broadcast`。所有配置使用相同硬件、映射、payload 和仿真模式；脚本不从仿真总时间
反推理论量，而是独立计算 flit-hop、vector beats、pairwise issues、fill 和 steady 项，再逐项
匹配 production trace。

## 理论 oracle

设 `F=ceil(payload_bits/128)`、`B=ceil(payload_bits/512)`。2×2 topology 中，root fan-in
为 3，router 2 的局部 fan-in 为 2，因此全树 pairwise issue 数为 `3B`。有 DCA 时计算流水
项为：

```text
T_dca_pipeline = L + (3B - 1) * II
```

不是 `3B*L`，也不是每个 physical flit/chunk 单独加一次 L。32 KiB 时 `F=2048`、`B=512`、
issues=1536，理论为 `fill=7`、`steady=1535`、合计 1542 个 pipeline cycles。

mesh-link oracle 使用固定 XY 路由逐边计数：普通 root-to-3-peers 的总边数为 4，tree 边数为
3。stream reduce 每条 tree edge 有 1 header + F data；multicast result 每条 edge 有 F data。
脚本按 profile 分开累加 normal 和 collective flit-hops，避免总流量相同但 backend 错误仍通过。

## 独立自测补强

评审指出 R8 只有 Python runner、没有独立 `--coll-rN-selftest`。新增
`npusim --coll-r8-selftest`，6/6：

- 5120/8192/65536/262144 bit 分别得到 10/16/128/512 vector beats 和
  30/48/384/1536 全树 issues；
- 32 KiB stream framing 为每 edge `1+2048` flits、三条边 6147 flit-hops；
- `L+(issues-1)II=1542 < issues*L`，显式防止固定 latency 回归为 per-chunk 计费。

四 profile 实验在启动时先执行该 C++ oracle selftest；合同失败时不会继续输出性能表。

## 实测结果

16 个 profile/payload case 全部通过。代表性 32 KiB 结果：

| profile | time (ns) | normal hops | collective hops | DCA issues |
|---|---:|---:|---:|---:|
| baseline | 134990 | 24576 | 0 | 0 |
| broadcast_only | 102108 | 16384 | 6144 | 0 |
| reduce_only | 90750 | 16384 | 6147 | 1536 |
| reduce_broadcast | 24986 | 0 | 18435 | 1536 |

所有大小的 actual normal/collective hop、DCA issue/completion 与 oracle 精确一致；backend
trace 分别为 baseline 无硬件 tree、broadcast_only 仅 multicast、reduce_only 仅 stream DCA、
combined 两者都有。32 KiB 无 progress watchdog，data/control balanced，tree/stream/DCA/
barrier/endpoint/DTE token 全零。

评审补强后，R6/R7 production runner 另为 19/19，包含同 Router output 的 normal unicast +
stream-DCA 混合流量；R6/R7/R8 独立 selftest 分别 7/7、6/6、6/6。

## 回归收口

- 未启用 collective 的 frozen NoC 周期保持 14781/29109 和 14833/45441 ns。
- D2D V0 为 67/67；V0～V6 与 R0～R5 合同/runner 保持通过。
- `git diff --check` 必须无 whitespace error；仓库根目录不得产生 trace 图片或事件转储。
- 配置期继续明确拒绝 `transport=smart`，不会用 conventional 结果冒充 OpenSMART。

## 已知限制

- 性能表是 2×2、固定 XY、单 Broadcast + 单 AllReduce 的可解释基准，不代表所有 topology、
  congestion 或消息分布。
- 目前没有 OpenSMART 单周期多跳、跨 die collective、RTL 功耗/面积建模。
- `reduce_broadcast` 在本组 workload 明显更快不构成“所有小消息/拥塞场景必然更快”的承诺；
  其他场景必须继续用 link、DCA 和 contention trace 分解。
- FP16/FP8 数值仍为 timing-only，四 profile 主实验使用 UINT8 exact。

R8 退出条件：大 payload 时间随 physical/vector beats 线性增长，固定 L 只出现一次；四 profile
backend、流量、DCA issue 和最终 drain 都能由独立 oracle 解释。
