# 跨 die 与片上混合流量 NoC 拥塞实验报告

## 结论

在同一套 2×1 die、相同的两个 GEMM、相同 D2D 链路参数和相同 32 个 DATA
包下，仅把跨 die 流的 C2C 出口从不相交的第 2 行移到本地流所在的第 1
行，就产生了 **11 次 NoC 发送阻塞**。跨 die flow
完成时间增加 **36 cycle = 72 ns**，仿真总时间
增加 **72 ns（11.25%）**。

## 实验设计

- 周期精确模式：`use_beha_noc=false`，`CYCLE=2 ns`。
- 两个场景都执行两个完全相同的 `Matmul_f(B=1,T=4,C=64,OC=512)`。
- 本地 flow 固定为 `core5 -> core7`，NoC 路径为 `5->6->7`。
- shared：跨 die flow `core4 -> core20`，die0 路径
  `4->5->6->7->D2D`，与本地流共享 [(5, 6), (6, 7)]。
- disjoint：跨 die flow `core8 -> core24`，die0 路径
  `8->9->10->11->D2D`，与本地流共享 []。
- 两个场景的 C2C hop 数、link latency/rate、SAF 容量、HOST 距离均一致。

## 周期精确结果

| 指标 | 不拥塞（disjoint） | 拥塞（shared） | 差值 |
|---|---:|---:|---:|
| 仿真完成时间 | 640 ns | 712 ns | +72 ns |
| 跨 die flow 完成 | cycle 311 | cycle 347 | +36 cycle |
| 本地 flow 完成 | cycle 260 | cycle 260 | +0 |
| die0 NoC 成功发送 | 230 | 230 | +0 |
| die0 NoC stall | 0 | 11 | +11 |
| D2D DATA in/out | 32/32 | 32/32 | 0 |
| DATA seqhash/checksum | match | match | — |
| D2D source/端口/链路 stall | 0 | 0 | 0 |
| 最终残留 | 0 | 0 | 0 |

## 静态负载模型与仿真对照

忽略 REQUEST/ACK 和流水线相位，只统计 32 个 DATA 包在源 die mesh 上的
packet-hop，可得到：

| 理论量 | 不拥塞（disjoint） | 拥塞（shared） |
|---|---:|---:|
| 本地 flow packet-hop | `32×2 = 64` | `32×2 = 64` |
| 跨 die flow 源 mesh packet-hop | `32×3 = 96` | `32×3 = 96` |
| DATA 总 packet-hop | 160 | 160 |
| 单条有向链路最大 DATA 负载 | 32 | 64 |
| 1 packet/cycle 下的瓶颈服务下界 | 32 cycle | 64 cycle |

因此两种编排的通信工作总量相同，但 shared 把两条 flow 的负载集中到
`5->6` 和 `6->7`，瓶颈链路负载翻倍。若本地 flow 先获得输出锁，简单容量
模型给出的额外串行化下界是 **32 cycle**。

周期仿真在 D2D 边界观测到：

| 周期观测 | 不拥塞（disjoint） | 拥塞（shared） | 差值 |
|---|---:|---:|---:|
| D2D DATA 输入窗口 | 275–306 | 311–342 | 整体 +36 cycle |
| D2D DATA 输出窗口 | 276–307 | 312–343 | 整体 +36 cycle |
| 输入/输出窗口宽度 | 31/31 cycle | 31/31 cycle | 0 |
| D2D 首包固定延迟 | 1 cycle | 1 cycle | 0 |
| 跨 die flow 完成 | 311 | 347 | +36 cycle |

实测额外 **36 cycle** 比最简容量下界多 4 cycle（相对实测差
11.11%）。这 4 cycle 是静态容量模型未覆盖的动态流水线项，候选来源包括有限
router buffer、registered ready、输入仲裁和 output-lock 释放相位；没有逐 router
事件 trace 时，不能把它进一步确定地分摊到某一个机制。
`NOC_ACT.stalls=11` 表示“输出有包但下游 input buffer 满”的累计
blocked-output 事件，约为每 100 次成功 NoC 发送对应 4.78 次；它不是
flow 被推迟的 cycle 数，所以不能把 11 stall 直接换算成 22 ns。跨 die flow
完成时间增加 11.58%，总运行时间增加 11.25%。

## 归因与边界

两个场景成功发送数相同、D2D DATA/ACK 数相同、D2D 端口和链路 stall
均为 0，且本地 flow 完成周期相同；差异只在 shared 场景的 die0 NoC
路径竞争。因此，本实验观测到的额外延迟可归因于跨 die 出口流量与片上
流量共享 NoC 物理链路，而不是计算量、D2D hop 数、HOST 距离或 D2D
限速差异。

本报告证明的是该确定性双 flow 工作负载下的周期精确拥塞效应；它不是对
所有路由、流量分布或有限缓冲参数的统计性能结论。
