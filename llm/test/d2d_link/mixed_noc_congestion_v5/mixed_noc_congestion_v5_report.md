# V5 跨 die striping 与片上流量 NoC 拥塞实验报告

## 结论

在同一套 2×1 dies、两个相同 GEMM、32 个跨 die DATA 包、2-way striping
及相同 D2D 参数下，仅平移跨 die 源核和两条 C2C 端口，使其在 shared
场景与本地 flow 共享两条源 die NoC 链路，便在 die0 产生
**11 次 blocked-output 事件**。跨 die flow 比不相交场景
晚 **28 cycle = 56 ns** 完成，仿真总时间增加
**56 ns（8.54%）**。

理想单链路容量模型预测的拥塞服务项为
**64−32 = 32 cycle**；
周期精确结果为 28 cycle，绝对偏差
4 cycle、相对偏差 12.50%。

## 实验设计

- 周期精确模式：`use_beha_noc=false`，`CYCLE=2 ns`。
- 两个场景均执行两个 `Matmul_f(B=1,T=4,C=64,OC=512)`。
- 本地 flow 固定为 `core5→core7`，路径
  `5→6→7`，发送 32 个包。
- 跨 die flow 使用 V5 `multi_port=true`、`hybrid` 选择和 2-way striping，
  配额为 [16, 16]。
- shared：`core4→core20`，C2C rows=[0, 1]；
  subflow 0（16 包）: 4→5→6→7→3；subflow 1（16 包）: 4→5→6→7。与本地 flow 共享 [(5, 6), (6, 7)]。
- disjoint：`core8→core24`，C2C rows=[3, 2]；
  subflow 0（16 包）: 8→9→10→11→15；subflow 1（16 包）: 8→9→10→11。与本地 flow 共享 []。
- 两个场景的 GEMM、总 DATA、stripe 配额、源/目的 mesh packet-hop、
  D2D link latency/rate/capacity 和 HOST 布局完全相同。

## 周期精确仿真结果

| 指标 | 无源 die 混合争用（disjoint） | 有源 die 混合争用（shared） | 差值 |
|---|---:|---:|---:|
| 仿真完成时间 | 656 ns | 712 ns | +56 ns |
| 跨 die flow 完成 | cycle 319 | cycle 347 | +28 cycle |
| 本地 flow 完成 | cycle 260 | cycle 260 | +0 |
| die0 NoC 成功发送 | 246 | 246 | +0 |
| die0 NoC stall | 0 | 11 | +11 |
| die1 NoC stall（共同的 stripe 汇聚） | 7 | 7 | +0 |
| D2D REQUEST/ACK/DATA | (2, 2, 2, 2, 32, 32) | (2, 2, 2, 2, 32, 32) | 相同 |
| 每 subflow DATA | 16/16，无损/有序 | 16/16，无损/有序 | 相同 |
| D2D source/port/link/inflight/group stall | 0 | 0 | 0 |
| SAF/credit/router/link 残留 | 0 | 0 | 0 |

两次独立运行的所有已解析指标均完全一致。全局 `[D2D_DATA]` 的两个 stripe
允许合法交织，因此完整性使用 `[V5_SUBFLOW]` 分桶验证：每个 subflow 的
包数、顺序 hash、完整 payload checksum、序号范围和唯一 tail 均匹配。

## 理论路径负载

只按 DATA 统计有向 NoC `packet-hop`：

| 理论量 | disjoint | shared |
|---|---:|---:|
| 本地 flow 源 die packet-hop | 64 | 64 |
| 跨 die flow 源 die packet-hop | 112 | 112 |
| 跨 die flow 目的 die packet-hop | 16 | 16 |
| 总 DATA mesh packet-hop | 192 | 192 |
| 源 die 单条有向链路最大负载 | 32 | 64 |
| 1 packet/cycle 理想瓶颈服务项 | 32 cycle | 64 cycle |

总通信工作量相同，但 shared 的两个 stripe 在到达不同 C2C 端口前仍共享
源核的行向 NoC cut；`5→6`、`6→7` 各承载本地 32 包和跨 die 32 包，
负载达到 64。disjoint 把本地和跨 die 流分散到不同链路，最大负载只有
32。这也说明 V5 的两条 D2D lane 不等于两条独立的源 NoC 注入 cut。

## 理论值与仿真对比

| 对照量 | disjoint | shared | 差值 |
|---|---:|---:|---:|
| 理论瓶颈服务项 | 32 cycle | 64 cycle | +32 |
| D2D DATA 输入窗口 | 283–298 | 311–326 | 首包 +28 |
| D2D DATA 输出窗口 | 284–306 | 312–334 | 首包 +28 |
| 跨 die flow 完成 | 319 | 347 | +28 |

拥塞延迟已经在 D2D 输入边界出现：首个 DATA 进入 D2D 的时间整体后移
28 cycle，恰好等于跨 die flow 完成周期差。实测差值比理想容量
模型少 4 cycle（12.50%）。容量模型只计算稳态
瓶颈负载，不包含两条 GEMM 的启动相位、router pipeline、仲裁相位和
output-lock 的瞬态重叠，因此不应期待逐周期完全相等；误差仅 4 cycle，
且方向和量级一致。

`NOC_ACT.stalls=11` 是“输出有包但下游输入满”的累计事件，
约为每 100 次成功发送对应 4.47 次，不等价于 flow 延迟 cycle。
本地 flow 仍在 cycle 260 完成，表明本次确定性仲裁中它先取得共享
输出资源，额外等待主要落在跨 die flow 上；后者完成时间增加
8.78%。

## 归因与适用边界

两个场景的成功 NoC 发送数、per-die 活动、D2D 消息数、每 link 包数、
SAF admission、D2D backpressure 总量以及最终排空状态均一致；只有 die0
共享路径产生额外 stall。因而差异可归因于跨 die 流量和片内流量对源 die
NoC 物理链路的争用，而不是计算量、D2D 带宽、stripe 不均、丢包或残留状态。

本实验验证的是一个确定性的双 GEMM、单源 2-way striping 场景。它不是对
任意流量分布的统计性能预测；理论模型也是容量近似，不包含完整动态仲裁。
