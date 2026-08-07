# DTE 能力实验报告

> 本报告由 `llm/test/dte/experiment/run_experiment.py` 自动生成。
> 仿真数值全部从 `events.json` trace 提取；理论值由脚本内的闭式模型独立计算，二者对照。

## 硬件与时序模型

- 时钟周期 `CYCLE = 2 ns`；DTE 启动 `γ = 4 ns`，`τ_launch = 2 ns`。
- DTE 启动延迟 `T_launch = (⌈γ/CYCLE⌉+⌈τ/CYCLE⌉)·CYCLE = 6 ns`。
- DTE 传输延迟 `T_xfer = ⌈payload_bits / dte_bit_width⌉·CYCLE`。
- 单端点 DTE 服务 `T_endpoint = T_launch + T_xfer`。
- 网络（NoC）延迟与 DTE 位宽无关，`noc_payload_per_cycle = 4`。

## 实验 A：异步通信能力（compute/DMA 重叠）

固定一条 Matmul 计算，扫描 DTE 传输量。阻塞调度 `issue→wait→compute` 串行支付两者；异步调度 `issue→compute→wait` 让计算与 DTE 传输并发。

- DTE 位宽固定 2048 bit；计算实测 `compute = 209 ns`（常量）。
- 调度间隙 `g = 4 ns`：core 派发 issue 与 compute 之间的固定 dispatch 开销，计算比传输晚 g 起步。
- 直接、可精确验证的异步指标是 **trace 中 compute 区间与 `DTE_transmit` 区间的实际相交时长**；理论并发量 `= min(compute, T_xfer − g)`。

| payload (bit) | T_xfer (ns) | 实测并发 (ns) | 理论并发 (ns) | 阻塞并发 (ns) | 阻塞 finish | 异步 finish | 端到端受益 (ns) | 规律 | 一致 |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 8 | 4 | 4 | 0 | 351 | 345 | 6 | DTE 藏进计算 | ✅ |
| 65536 | 64 | 60 | 60 | 0 | 407 | 345 | 62 | DTE 藏进计算 | ✅ |
| 393216 | 384 | 209 | 209 | 0 | 727 | 514 | 213 | 计算藏进 DTE | ✅ |
| 786432 | 768 | 209 | 209 | 0 | 1111 | 898 | 213 | 计算藏进 DTE | ✅ |

**分析**：阻塞调度里 compute 与传输零相交（严格串行）；异步调度里两者真实并发。并发时长随传输量增大而增大，并在 `T_xfer ≥ compute` 后**饱和于计算时长**——异步只能把两段中较短的一段藏进较长的一段。端到端 finish 也随之降低（受益 ≈ 并发量，含个位数原语边界开销）。实测并发与 `min(compute, T_xfer − g)` 逐行精确一致，证明 DTE 具备异步通信能力。

## 实验 B：DTE 占总通信时延的比例

单条 16384-bit 跨核 flow，store-and-forward 模式下端到端时延可加性分解为`源 DTE + 网络 + 目的 DTE`。

- DTE 位宽 = 256 bit，payload = 16384 bit。
- 源 DTE `t_src = 134 ns`（理论 134 ns）
- 网络 `t_net = 138 ns`
- 目的 DTE `t_dst = 134 ns`（理论 134 ns）
- 端到端 `= 406 ns`

**DTE 占总通信时延 = (t_src + t_dst) / 端到端 = 268 / 406 = 66.0%**（理论 66.0%）。

## 实验 C：DTE 带宽是否为通信瓶颈

固定网络，扫描 DTE 位宽（即 DTE 带宽 = 位宽/CYCLE）。观察端到端时延与其中 DTE / 网络两部分的变化。

说明：`单端 T_xfer = ⌈payload/位宽⌉·CYCLE` 是一次 DTE 传输的时间，与网络时间 `t_net` 直接比较即回答“DTE 带宽是否为瓶颈”。

| DTE 位宽 (bit) | DTE 带宽 (bit/ns) | 单端 T_xfer (ns) | t_net (ns) | 端到端 (ns) | DTE 部分 t_src+t_dst (ns) | 瓶颈 | 理论一致 |
|---|---|---|---|---|---|---|---|
| 64 | 32 | 512 | 138 | 1174 | 1036 | DTE-bound | ✅ |
| 128 | 64 | 256 | 138 | 662 | 524 | DTE-bound | ✅ |
| 256 | 128 | 128 | 138 | 406 | 268 | network-bound | ✅ |
| 512 | 256 | 64 | 138 | 278 | 140 | network-bound | ✅ |
| 1024 | 512 | 32 | 138 | 214 | 76 | network-bound | ✅ |
| 2048 | 1024 | 16 | 138 | 182 | 44 | network-bound | ✅ |
| 4096 | 2048 | 8 | 138 | 166 | 28 | network-bound | ✅ |
| 8192 | 4096 | 4 | 138 | 158 | 20 | network-bound | ✅ |

- 网络延迟在所有位宽下恒为 **138 ns**，证明它与 DTE 位宽无关，扫描确实只改变 DTE 一侧。
- 有效网络带宽 = payload / t_net = 16384 / 138 ≈ **118.7 bit/ns**。
- 理论瓶颈翻转位宽 = payload·CYCLE / t_net ≈ **237 bit**：
  - 位宽低于该值时 DTE 传输时间 > 网络时间 → **DTE 带宽是瓶颈**，端到端时延随位宽减半而近似翻倍（∝ 1/位宽）；
  - 位宽高于该值时 DTE 传输时间被网络掩盖 → **网络成为瓶颈**，端到端时延趋于由 `t_net` 决定的地板。

## 结论

1. **异步能力**：DTE 传输与计算可真实并发，并发时长 `= min(compute, T_xfer − g)`（g=4 ns 为固定派发间隙），阻塞调度并发为 0，仿真与理论逐行一致。
2. **DTE 时延占比**：在 256-bit 位宽下 DTE 占单 flow 通信时延的 66.0%，源/目的两端点各 134 ns，与理论一致。
3. **瓶颈带宽**：DTE 带宽低于 ~237 bit/2ns（有效网络带宽 118.7 bit/ns）时，DTE 是通信瓶颈；高于该点网络接管。仿真的瓶颈翻转与理论吻合。

**总体：仿真与理论 完全一致 ✅。**
