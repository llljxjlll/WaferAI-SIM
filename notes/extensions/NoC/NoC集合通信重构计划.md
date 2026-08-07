# NoC 集合通信重构计划

日期：2026-08-03

## 1. 背景与结论

现有 NoC 集合通信 V0～V6 已经完成 Tier0 多单播、Tier1 Router multicast、
Tier2 Router in-network reduce，以及 tree 生命周期、有限 buffer、背压、数值
验证和 drain 检查。现有实现对冻结需求是正确的，但后续三档实验表明，旧 Tier2
只在单 chunk 时有效，payload 增大后反而显著慢于 Tier0/Tier1，并在更大 payload
下出现循环等待。

结合以下论文重新分析后，旧 Tier2 的 DCA 抽象需要重构，而不是只调整一个固定
延时常数：

1. Stefan Mach et al., *FPnew: An Open-Source Multi-Format Floating-Point Unit
   Architecture for Energy-Proportional Transprecision Computing*；
2. Hyoukjun Kwon and Tushar Krishna, *OpenSMART: Single-Cycle Multi-hop NoC
   Generator in BSV and Chisel*；
3. Luca Colagrande et al., *A Lightweight High-Throughput Collective-Capable NoC
   for Large-Scale ML Accelerators*。

本轮重构交付两个新配置：

- **Reduce-only**：只使用 in-network reduce；Broadcast、AllGather 的分发和
  AllReduce 的结果分发仍走普通单播，不启用 Router multicast。
- **Reduce+Broadcast**：使用新的 in-network DCA reduce，同时保留 Router
  multicast 加速 Broadcast 和 AllReduce 结果分发，替换旧 Tier2 实现。

本轮明确不建模 OpenSMART。所有 flit 继续按当前逐 Router、逐 link、真实 buffer/
credit/仲裁路径运行；不加入 SSR、HPCmax 或 multi-hop bypass。OpenSMART 论文只用于
澄清“网络传输优化”和“DCA 计算优化”必须分层，不能再把 hop latency、DCA latency
和 DCA throughput 合并到一个公式中。

## 2. 目标与非目标

### 2.1 目标

- 消除旧 Tier2 将 SIMD elements 当成串行 ALU 次数的错误。
- 分离 DCA/FPU 的 pipeline latency 与 initiation interval（II）。
- 将 DCA 建模为 Router 向所在 tile/cluster 的计算资源发起二输入向量运算，而不是
  Router 内部无竞争、不可流水的专用标量 ALU。
- 每个 DCA issue 最多接收两个 vector operands；多输入归约确定性分解为多次
  pairwise vector reduction。
- 单输入节点直接转发，不收取 DCA 计算时间。
- 支持多个 DCA request 同时处于流水线，header/context 通过有限 FIFO 和 tag 对齐。
- root 从 collective 启动时就持续 drain reduce result，源端 TX 和 root RX 可并行，
  消除大 payload 的 TX/RX 循环等待。
- 将完整 metadata 与数据流解耦，避免每个 128-bit payload 都携带一个独立 256-bit
  header wire。
- 使用独立配置表达 Broadcast backend 和 Reduce backend，支持四种组合并保持旧
  tier alias 的兼容性。
- cycle backend 由真实事件计时；behavioral/oracle 使用与流水数据面一致的吞吐模型。
- 未启用 collective 时，现有 NoC、DTE、D2D 冻结回归必须逐项不变。

### 2.2 非目标

- 本轮不实现 OpenSMART/SSR/HPCmax/lookahead multi-hop bypass。
- 本轮不实现跨 die Tier1/Tier2；in-network reduce 和 Router multicast 仍只支持
  same-die group。
- 本轮不增加 ring、recursive-halving 或 halving-doubling。
- 本轮不把 ReduceScatter 的 scatter 阶段错误地改成 multicast；不同目标的数据不同，
  Router fork 不能加速 scatter。
- 本轮不把 elementwise collective reduction 改成 vector 内部的 horizontal reduction。
- 在确定性 IEEE-754 helper、舍入和 NaN 规则完成前，不宣称 FP reduction bit-accurate；
  timing-only FP 模式必须在 trace 中显式标记。

## 3. 旧 Tier2 性能异常分析

### 3.1 已观察到的异常

现有 2×2 mesh 实验由一次 Broadcast 和一次 UINT8/SUM AllReduce 组成，模型周期为
2 ns：

| payload | chunks | Tier0 | Tier1 | 旧 Tier2 | 旧 Tier2/Tier0 |
|---:|---:|---:|---:|---:|---:|
| 128 bit | 1 | 912 ns | 780 ns | 740 ns | 1.232× |
| 512 bit | 4 | 1108 ns | 930 ns | 1288 ns | 0.860× |
| 2048 bit | 16 | 1894 ns | 1524 ns | 3448 ns | 0.549× |

压力扫描中，4096 bit/32 chunks 可以完成；5120 bit/40 chunks 和
8192 bit/64 chunks 触发 progress watchdog。结果数值仍正确，异常集中在数据流和
计时模型。

### 3.2 原因一：把 SIMD elements 当成串行计算周期

旧实现使用：

```text
compute_cycles = valid_elements × (expected_inputs - 1)
```

例如 128-bit UINT8 chunk 有 16 个 elements，三个输入在 root 汇合时被计算为
`16 × 2 = 32 cycles`。但论文中的 DCA 是两个宽 vector operands 的逐 lane 运算：

```text
T[j] = A[j] + B[j]，所有有效 j 并行
R[j] = T[j] + C[j]，所有有效 j 并行
```

因此三个输入需要两次 vector issue，不是 32 次 scalar issue。C++ 中逐 element
循环可以继续用于计算 bit-accurate 结果，但不能再用循环次数生成 cycle。

### 3.3 原因二：把 pipeline latency 当成 initiation interval

旧公式为：

```text
service = max(compute_cycles, ceil(payload_bits / 128)) + 54
```

Router 再用 `reduce_dca_available=previous_completion` 串行化所有 chunk，等价于：

```text
DCA_II = DCA_latency = 整个 service
```

FPnew 和 collective-capable NoC 的关键设计恰好相反：算术单元可以有多周期 latency，
但 header FIFO 足够深时，独立 vector beat 可以背靠背 issue，稳态 II 可以为 1。
固定流水延迟只影响首结果和 drain，不应对每个 chunk 重复收费。

### 3.4 原因三：单输入节点也支付 DCA 开销

旧路径中 leaf/单输入 Router 即使没有发生任何加法，仍支付
`ceil(payload/128)+54`。论文 RTL 对 single-member reduction 明确直接转发。单输入
节点的 vector issue 数必须为 0。

### 3.5 原因四：一次 Match 吞入任意多个输入

旧 Match Buffer 等待 expected bitmap 全部到齐，然后一次调用
`ReduceIntegerOperands(values)`。论文的 wide reduction offload 只有两个 operand
端口；三个或更多输入必须分成确定性的 pairwise stages。旧实现虽然最终值正确，
但没有表达二输入限制、partial result 依赖、每 beat 的 issue 数和确定的 FP 结合顺序。

### 3.6 原因五：operand framing 将数据流量翻倍

旧 Tier2 每个 128-bit payload 使用两个 256-bit wire：一个完整 header segment 和
一个 payload segment。2×2 实验中 reduce operand 因此需要 `6F` tree-edge flit-hop，
加上 result multicast 后，Tier2 总 flit-hop 与 Tier0 相同。固定 metadata 被按 chunk
重复发送，掩盖了 in-network aggregation 的流量优势。

### 3.7 原因六：root 先完成全部 TX，之后才进入 RX

root 的 primitive 顺序为完整 `REDUCE_TX` 后再执行 `REDUCE_RX`。结果在 TX 期间返回
并填满只有 3 个 wire 流控阈值的 endpoint raw queue 后，会向 Router 传播背压；root
同时等待 TX 网络前进，形成“TX 等网络、网络等 RX”的循环等待。扩大 queue 或 watchdog
只能推迟问题，不能证明任意 payload 的 progress。

### 3.8 原因七：Tier0 与 Tier2 使用了不公平的计算抽象

Tier0 root compute 已按 endpoint vector lanes 计费；旧 Tier2 却按 element 数串行计费，
并对每个 chunk增加固定 54 cycles。这使配置对比同时改变通信算法和计算单元吞吐，
不能代表论文中的 DCA 加速。

### 3.9 原因八：缺少 DCA 与 core 的资源共享

论文中的 DCA 借用 cluster 已有 FPU，并与 core 请求仲裁。旧模型既把 DCA 当成 Router
专用资源，又把它建模得极慢：空闲时性能被低估，core 忙时争用又被遗漏。重构后必须
显式建模共享资源、仲裁和统计。

## 4. 论文约束如何进入新模型

### 4.1 FPnew 约束

- FPU/vector unit 的 datapath width、支持格式、lanes、pipeline latency 和 II 均为配置。
- `lanes(dtype) = vector_bits / dtype_bits`；尾 beat 使用 valid-lane mask，但仍只占一个
  vector issue。
- value semantics 与 timing semantics 分离：逐 element helper 计算值，vector issue
  scheduler 计算时间。
- 不同 dtype/op 可以具有不同 latency/II。
- DCA request 与 core vector/FPU request 进入同一个 compute pool 仲裁。
- request 带 tag，tag 随流水线返回并定位 header/context/result destination。

### 4.2 OpenSMART 约束及本轮边界

- OpenSMART 优化的是一个 flit 沿既定方向一次跨多少 Router，不改变 reduction 的
  vector issue 数。
- 本轮保持 conventional hop-by-hop Router；理论模型继续包含逐 hop/link latency、
  buffer、credit 和仲裁。
- 配置中预留 `noc.transport=conventional`，但本轮若配置为 `smart` 必须启动期拒绝，
  不能静默使用普通 Router 却报告 SMART 结果。

### 4.3 Collective-capable NoC/DCA 约束

- 每个 Router 只有一个集中式 wide-reduction issue port，所有输入/输出共享；
  “单 issue port”不等于“流水线中只能有一个 request”。
- 每次 offload 只处理两个 vector operands，返回一个等宽 vector result。
- single-member reduction 直接转发。
- Sync/Match 确保两个 operand 的 key、beat、dtype、op 和 valid lanes 一致。
- Header/context FIFO 保存已 issue 但尚未完成的请求；深度必须覆盖流水 latency，满时
  通过 ready/valid 向上游背压。
- result 按 tag 找回 context，恢复 header 后作为普通归约 partial result 或最终 unicast
  数据重注入。
- 参考 profile 使用 `2×512-bit input + 1×512-bit output`、`8×64-bit` compute slices；
  仿真器允许缩放，但必须显式配置，不能把当前 128-bit 物理 payload 当成固定 DCA
  架构事实。
- 三输入汇合需要每个 beat 两次 pairwise vector issue；论文报告的合法稳态瓶颈是约
  1 个 fully-reduced beat/2 cycles，而不是 `elements×2+固定延迟`。

## 5. 新配置模型

现有 `tier=0|1|2` 无法表达“只加速 reduce”，因此 backend 组合成为唯一规范配置：

```text
noc.collective.enabled = true | false
noc.collective.profile = baseline | broadcast_only | reduce_only | reduce_broadcast
noc.collective.broadcast_backend = unicast | multicast
noc.collective.reduce_backend = endpoint | dca_offload | legacy_router_alu
noc.collective.reduce_wire = stream_v2 | legacy_two_segment
noc.transport = conventional
```

四种 profile 的规范展开为：

| profile | broadcast_backend | reduce_backend | 用途 |
|---|---|---|---|
| `baseline` | `unicast` | `endpoint` | 原 Tier0 |
| `broadcast_only` | `multicast` | `endpoint` | 原 Tier1 |
| `reduce_only` | `unicast` | `dca_offload` | 本轮新增 |
| `reduce_broadcast` | `multicast` | `dca_offload` | 改进后的 Tier2 |

兼容规则：

- 旧 `tier=0/1/2` 分别映射到 `baseline/broadcast_only/reduce_broadcast`。
- profile、tier、显式 backend 同时出现时必须完全一致，否则启动期失败。
- `legacy_router_alu + legacy_two_segment` 只用于复现旧 V5/V6 自测和旧实验，不作为新
  workload 的默认生产配置；必须同时显式设置
  `noc.collective.allow_legacy_backend=true`，并且不得与 profile/tier 混用。
- 未配置 profile/backend 时保持当前默认关闭行为，不改变普通 NoC。
- `dca_offload` 或 `multicast` 遇到跨 die group 继续启动期拒绝。
- `noc.transport=smart` 在本轮启动期明确报“尚未实现”，禁止静默降级。

新增 DCA 配置：

```text
noc.collective.dca.vector_bits = 512
noc.collective.dca.slice_bits = 64
noc.collective.dca.slices_per_tile = 8
noc.collective.dca.latency[dtype][op]
noc.collective.dca.initiation_interval[dtype][op]
noc.collective.dca.header_fifo_depth
noc.collective.dca.operand_fifo_depth
noc.collective.dca.result_fifo_depth
noc.collective.dca.arbitration = round_robin | core_priority | dca_priority
noc.collective.dca.value_mode = integer_exact | fp_exact | timing_only
```

配置校验：

- 只有 `collective.enabled=true && reduce_backend=dca_offload` 时才执行 DCA
  结构/value-mode 语义校验；baseline、endpoint 或 disabled 配置中的 dormant DCA
  参数不得阻止启动，但 JSON 字段名和枚举拼写仍严格解析。
- `vector_bits == slice_bits × slices_per_tile`，或显式允许 time-multiplexed width conversion。
- `vector_bits % dtype_bits == 0` 在实际 reduction workload 的 dtype 已知后校验，
  不再以“当前最大 dtype 位宽”近似全部 dtype。
- FIFO depth、latency、II 必须为正；若希望无结构性 pipeline stall，要求
  `header_fifo_depth >= ceil(latency/II)+2`，否则允许配置但必须产生容量背压统计。
- `value_mode=fp_exact` 时必须存在对应 dtype/op 的确定性 IEEE-754 helper。
- timing-only 不得运行 value-verified runner。

## 6. 新的 DCA/SIMD 计算模型

设：

```text
W       = dca.vector_bits
w       = dtype_bits
lanes   = W / w
B       = ceil(total_elements / lanes)
k_v     = Router v 某 reduction stage 的输入数
q_v     = max(k_v - 1, 0)
II      = dca.initiation_interval[dtype][op]
L       = dca.latency[dtype][op]
```

则 Router v 的 vector issue 工作量为：

```text
issues_v = B × q_v
```

不是：

```text
total_elements × q_v
```

关键时序契约：

- `k_v=1`：`issues_v=0`，直接转发。
- `k_v=2`：每 beat 一个二输入 vector issue；无争用时结果稳态间隔为 `II`。
- `k_v=3`：确定性分为两个 pairwise stages；单 issue port 下 fully-reduced result 的稳态
  间隔为 `2×II`，流水 fill/dependency 只影响启动部分。
- 一个 request 的 completion 为 `issue_time+L`；下一个独立 request 最早可在
  `previous_issue_time+II` 发射，不等待 previous completion。
- 多个 request 在流水线中并存；result tag 必须与 header/context FIFO 一一对应。
- partial tail beat 的无效 lanes 被 mask，不参与值运算，但占一个 vector issue。

Cycle backend 不额外等待闭式 DCA 时间。`DcaComputePool` 生成 issue/completion event，
Router、FIFO、backpressure 和 result reinjection 自然决定完成时间。Behavioral backend 和
Python oracle 才使用上述公式。

## 7. 二输入 reduction tree 与 Match 设计

### 7.1 Tree/Stage key

现有 `(CollectiveKey,phase_id,chunk_id)` 扩展为：

```text
ReduceStreamKey = (CollectiveKey, phase_id, stream_id)
ReduceBeatKey   = (ReduceStreamKey, reduce_stage_id, vector_beat_id)
```

`reduce_stage_id` 明确表示 pairwise 结合顺序，不能把三个输入隐藏在一个 Match state
里。配置期为每个物理 Router 构造确定性 binary schedule：

```text
stage 0: input A + input B -> partial P0
stage 1: partial P0 + input C -> partial P1
...
```

每个 stage 的 expected-input bitmap 恰好有两个 bit；只有 single-member bypass 可以有
一个 bit且不创建 DCA request。FP SUM 的结合顺序由 stage 顺序冻结，保证重复运行结果
确定。

### 7.2 状态拆分

旧 Match Buffer 拆为：

```text
ReduceHeaderTable     // 活跃 stream、dtype/op、总 beat、下一序号
ReduceOperandMatch    // 每个 stage/beat 最多两个 operands
DcaIssueQueue         // 已匹配、等待共享 compute pool
DcaInflightTable      // tag -> header/context/stage/route
DcaResultQueue        // 已完成、等待 local feedback 或网络重注入
```

所有结构有限；满载必须返回 backpressure，不抛异常、不丢 operand、不重复 issue。Residual
统计必须覆盖五类状态以及 stream RX/TX assembler。

### 7.3 Local feedback

同一 Router 上的下一 pairwise stage 使用 local feedback queue 接收 partial result，避免
结果绕网络一圈。Local feedback 与网络输入共同进入公平仲裁，不能永久饿死任一方向。

## 8. Reduce stream wire v2

### 8.1 目标

将旧的“每个 128-bit chunk 一个完整 header + 一个 payload”改成“每个 stream 一个
header + 连续 data flits”：

```text
REDUCE_STREAM_HDR
REDUCE_STREAM_DATA seq=0
REDUCE_STREAM_DATA seq=1
...
REDUCE_STREAM_DATA seq=F-1, tail=1
```

header 至少包含：wire magic/version、CollectiveKey、tree/stage/stream ID、source、dtype、
reduce op、总 element/physical flit/vector beat 数、尾 beat valid lanes。Data flit 只携
payload、紧凑 stream ID、seq/tail 和必要的防误判字段。

在 physical payload 为 128 bit、DCA vector 为 512 bit 时，assembler 每收到四个连续
data flits 形成一个 DCA vector beat；最后不足四个时补零并生成 lane mask。Result
splitter 将 512-bit result 重新拆为 128-bit data flits。

### 8.2 正确性和流控

- stream header 建立 per-input context；同一 VC/stream 内 data 顺序递增，非法 seq、
  重复 tail、截断或额外 data 确定性报错。
- header/context 满、assembler 满、operand match 满、DCA queue 满和 result queue 满均
  逐级传播 backpressure。
- normal Msg、COLL_DATA v1、REDUCE_STREAM v2 使用严格 magic/version/segment 检测，
  保持普通流量零误判。
- legacy wire 只在显式 legacy backend 下解码；同一 CollectiveKey 禁止混用 wire 版本。
- stream 完成和 tree release 前必须证明所有 header/data/assembler/tag/result 状态归零。

数据量从旧的约 `2F` wires/edge 改为 `F+1` wires/edge；小 payload 可能仍受 header
影响，但大 payload 不再重复支付完整 metadata。

## 9. 并发执行与 progress

Collective session 启动时同时建立 TX producer 和 RX consumer：

```text
start collective session
  ├─ source/root TX stream producer
  ├─ root/final-destination RX drain（立即 armed）
  └─ completion tracker
```

root 不能等所有 TX 完成后才开始 RX。最终 barrier 的进入条件为：本 rank TX 完成、
预期 RX/result 完成、DCA/local feedback 与 endpoint context 全部完成。这样返回结果始终
有可消费端，打破旧 root queue 的循环等待。

若 future/worker primitive 接口仍是顺序列表，则 `Collective_data_prim` 只负责创建异步
session，随后以 token/fence 等待 session 完成；不能用两个阻断式 REDUCE_TX/REDUCE_RX
primitive 表达并发。

## 10. 两种新加速方案

### 10.1 Reduce-only

配置：

```text
profile=reduce_only
broadcast_backend=unicast
reduce_backend=dca_offload
transport=conventional
```

操作映射：

| collective | Reduce-only 数据面 |
|---|---|
| P2P/Scatter/Gather/AllToAll | 保持 Tier0 |
| Broadcast | root 向 N-1 目标发送普通 unicast |
| AllGather | 每个 source 的分发展开为普通 unicast，RX 仍用 Gather reorder |
| Reduce | source stream 沿 reduce tree 逐级 DCA，最终结果普通 unicast/local delivery 到 root |
| ReduceScatter | DCA Reduce-to-root，随后按不同 slice 做普通 unicast Scatter |
| AllReduce | DCA Reduce-to-root，随后 root 用 N-1 条普通 unicast 分发相同结果 |

Reduce-only 仍需要 reduce tree/stage table，但不得编程或进入 multicast fork datapath。
验收 trace 中 standalone Broadcast 和 AllReduce 结果阶段的 `COLL_V4_TX/RX` 必须为 0；
reduce tree、DCA、普通 unicast 可以共享 link 并产生真实竞争。

该方案用于隔离评价 in-network reduce 的收益：与 baseline 相比只改变 reduction；与
reduce_broadcast 相比 DCA 完全相同，差异只来自 broadcast result distribution。

### 10.2 改进后的 Reduce+Broadcast

配置：

```text
profile=reduce_broadcast
broadcast_backend=multicast
reduce_backend=dca_offload
transport=conventional
```

操作映射：

| collective | Reduce+Broadcast 数据面 |
|---|---|
| P2P/Scatter/Gather/AllToAll | 保持 Tier0；Scatter 不用 multicast |
| Broadcast | root 单份注入，Router atomic multicast fork |
| AllGather | 每个 source 单份 multicast，RX 使用 Gather reorder |
| Reduce | 与 Reduce-only 使用完全相同的 DCA reduce tree |
| ReduceScatter | 与 Reduce-only 相同：DCA reduce + 普通 Scatter |
| AllReduce | DCA Reduce-to-root，root 单份 multicast result stream |

现有 AtomicMulticastFork、CollectiveTreeTable、每目标 exactly-once、tree release 和
mixed-traffic 仲裁可以保留。改进重点是用新的二输入流式 DCA 替换旧 Tier2 reduce；
multicast 不能掩盖或旁路 DCA backpressure。

## 11. 理论与 oracle

### 11.1 逻辑工作量

对 Router `v`：

```text
vector_beats       B = ceil(count / lanes(dtype))
pairwise_per_beat  q_v = expected_inputs_v - 1
dca_issues_v       = B × q_v
```

physical link flits：

```text
data_flits_per_stream = ceil(count × dtype_bits / physical_payload_bits)
wire_flits_per_stream = 1 header + data_flits_per_stream
```

### 11.2 无争用时序边界

两输入 stage、`II=1`：

```text
first completion = first issue + L
last completion  = first issue + L + (B-1)
```

一个物理 Router 上有 `q_v` 次 pairwise work/beat 且共享一个 issue port时，稳态 fully
reduced beat 间隔近似为：

```text
q_v × II
```

整体 behavioral 估算使用：

```text
T = T_setup
  + T_first_data_transport
  + T_pipeline_fill_on_critical_path
  + (B-1) × max(II_link_per_vector, max_v(q_v × II_dca_v))
  + T_core_dca_queueing
  + T_tail_transport_and_drain
```

其中 `II_link_per_vector` 由一个 DCA vector 需要的 physical flit 数和 link rate 推导。
禁止重新引入 `B×(L+work)` 或逐 chunk `+54`。

### 11.3 公平对比

Tier0 endpoint reduction 也使用同一套 dtype/lanes/latency/II 定义，但资源位于 root
endpoint；DCA 模式的资源由沿途 tile 借用并逐级计算。比较中只改变数据移动位置和资源
竞争，不再让 Tier0 使用 vector 模型、Tier2 使用 scalar 模型。

## 12. 分阶段开发计划

每个阶段必须同时提交代码、自动化测试、oracle 更新、开发日志和已知限制；前一阶段
未通过不得进入下一阶段的 production wiring。

### R0 — 冻结旧异常与新契约

**状态：已完成（2026-08-03）**。公共契约、独立 oracle、19 项 C++ selftest、9 点
legacy 实验和 32/40/64 chunks 压力边界均已冻结；本阶段未接入生产 Router 数据通路。
详见 `log/R0_development.md`。

**R0 冻结契约修订（R2 评审补记，2026-08-03）**：原
`coll_refactor::ReduceWireVersion` 的名称和数值
`LEGACY_TWO_SEGMENT=0, STREAM_V2=1` 保持不变，但定义从
`coll_refactor_contract.h` 移到唯一公共头 `coll_wire.h`；配置层删除数值相反的
重复枚举，`NocCollReduceWire` 改为同一类型的别名，并用 `static_assert` 冻结数值。
这是声明归属/唯一真源修订，不改变 R0 wire 语义；R0 时尚无 wire codec，所有调用均按
枚举名引用，因此没有历史 packet 或运行时结果迁移。

**开发内容**

- 将现有 9 点 Tier0/Tier1/Tier2 实验及 32/40/64 chunks 压力结果保存为 legacy
  reference，不再把旧 Tier2 性能当作新实现目标。
- 冻结 `DcaRequest/DcaResult`、vector beat、lane mask、latency/II、tag、pairwise stage、
  stream wire v2 和 async collective session 契约。
- 在需求和旧计划中记录：`max(comp,p/128)+54` 只属于 legacy contract，不是论文 DCA。
- 定义旧/new trace 字段，禁止新旧计数混在同一统计中。

**必须通过的测试**

- 旧 V0～V6 selftest/runner 全绿。
- 现有三档实验数值、flit-hop 和压力边界可重复。
- 新的纯函数 selftest 检查 `lanes/B/issues`：单输入 0 issue、二输入 B issue、三输入
  `2B` issue、tail mask 正确。

**阶段目标**

- 形成可审查的新 contract，明确值语义与时序语义分离。
- 后续性能变化可归因于新 backend，而不是无记录地修改 legacy 路径。

### R1 — 配置正交化与能力门禁

**状态：已完成（2026-08-03）**。四 profile、显式 backend/wire、tier alias、legacy
debug gate、DCA 参数、transport/cross-die 门禁和 backend 驱动的建树规则均已接入；
新 `dca_offload` production datapath 尚未接通，reduce workload 会在启动期明确拒绝，
不会静默回退到 V5。评审修订后配置层与协议层共用唯一 reduce-wire 枚举，DCA
语义校验仅对实际启用的 DCA 后端生效，dtype lane 校验延后到 workload dtype 已知时，
且已删除无消费者的 `SPEC_NOC_COLL_TIER` 全局量。详见
`log/R1_development.md`。

**开发内容**

- 实现四种 profile 和显式 broadcast/reduce backend。
- 实现 tier alias、冲突配置拒绝、legacy backend debug gate。
- 增加 DCA width/slices/latency/II/FIFO/arbitration/value-mode 配置解析和校验。
- 明确拒绝 `noc.transport=smart`、跨 die DCA/multicast 和不支持的 dtype/op。

**必须通过的测试**

- 四种 profile 的正向解析和规范展开。
- tier/profile/backend 一致与冲突的正负例。
- 实际启用 DCA 时非法 vector width、latency、II、FIFO depth、value mode 启动期失败；
  dtype lane 整除性按 workload dtype 检查；inactive DCA 参数不影响 baseline/disabled。
- `reduce_only` 不编程 multicast tree；`reduce_broadcast` 同时编程 reduce 与 multicast
  tree。
- collective disabled 的 frozen configuration 输出不变。

**阶段目标**

- 用户可以不依赖含糊的 tier 数字准确选择两种新方案。
- 所有不支持组合在启动期失败，不允许静默降级。

### R2 — DCA ComputePool 与流水契约（隔离实现）

**状态：已完成（2026-08-03）**。已实现隔离的有限 `DcaComputePool`：共享
core/DCA pending issue 容量、有限 inflight context/result queue、单二输入 issue port、
多 inflight tag、per-dtype/op L/II、三种仲裁、整数 SIMD value helper、tag wrap 与完整
背压/排空统计。R2 仍不连接 Router production datapath，`dca_offload` workload
继续由 R1 gate 明确拒绝。本阶段同时正式登记上述 R0 reduce-wire 契约修订，并复核
inactive DCA 配置门禁。详见 `log/R2_development.md`。

**开发内容**

- 新增 per-tile `DcaComputePool`：一个二输入 vector issue port、有限 issue queue、多个
  inflight tags、result queue。
- 实现 `issue_time/II` 与 `completion_time/latency` 分离。
- 实现 dtype lanes、tail mask、整数 SUM/MAX value helper；C++ element loop 只计算值。
- 实现 round-robin/core-priority/DCA-priority 仲裁和资源占用统计。
- 提供 synthetic core requests，用于在尚未接生产 compute path 前验证争用。

**必须通过的测试**

- `L>1, II=1` 时连续 B 个 request 每周期 issue，最后完成时间为 `L+B-1`。
- `II>1`、不同 dtype/op latency、queue full backpressure、result backpressure。
- 512-bit FP8/UINT8 等价 64 lanes，128-bit UINT8 等价 16 lanes；tail lane 不修改。
- 两输入一次 vector issue；三输入调度为两次 pairwise issue，不出现 element×source cycles。
- core/DCA 同时请求时仲裁顺序、stall 计数和最终 drain 正确。
- tag wrap/collision、重复 completion、未知 tag 确定性报错。

**完成证据**

- `npusim --coll-r2-selftest`：37/37。
- 独立 `r2_oracle.py` 覆盖多 dtype width/count/fan-in 的
  `lanes/beats/issues/tail`、L/II timeline、混合 latency 乱序完成和三种仲裁；
  `run_test_coll_r2.py`：2/2。
- 512-bit UINT8 验证 64 lanes，128-bit UINT8 验证 16 lanes；FP8 使用相同 8-bit
  lane 几何，但 FP8 dtype/格式接入按计划留在 R6。
- R1 配置解析继续只在 `enabled && dca_offload` 时执行 DCA 语义校验；已经实例化的
  ComputePool 则无条件强校验资源配置，二者分别有正负例。
- R0/R1、V0～V6 selftest、V1/V4-V6 runner、NoC congestion 和 D2D 冻结回归均通过。

**阶段目标**

- 在不连接 Router 的情况下证明 DCA 可以流水化，latency 不再等于 II。
- 对任意 dtype/count/fan-in，vector issue 数与 oracle 精确一致。

### R3 — 二输入 tree/stage、stream wire v2 与有限状态

**状态：已完成（2026-08-03）**。已新增隔离的 `coll_reduce_stream.h/.cpp`：
任意 fan-in 被确定性展开为二输入 left-fold stages，single-member 直接 bypass；冻结
256-bit stream-v2 header/data wire，实现 128-bit physical flit 到 512-bit DCA vector 的
4:1 assembler/splitter；header、assembler-ready、operand、issue、inflight、result、
network-input 和 local-feedback 均为有限结构且容量满可重试。R3 不接入生产 Router，
现有 `coll_innetwork_reduce` 和 V4～V6 datapath 保持不变。详见
`log/R3_development.md`。

**开发内容**

- 配置期把任意 fan-in 展开为确定性 binary pairwise stages。
- 实现 `ReduceStreamKey/ReduceBeatKey`、single-member bypass 和 local feedback。
- 实现 stream header + data wire v2、assembler/splitter、seq/tail/lane-mask 校验。
- 将 Match、issue、inflight、result、header context 拆成有限结构并接通背压。
- wire version 与 legacy two-segment 严格隔离。

**必须通过的测试**

- wire round-trip、截断、额外 data、乱序/重复 seq、错误 tail、dtype/op/key mismatch。
- 1/2/3/5 输入的 stage 数分别为 0/1/2/4，每 stage 最多两个输入。
- single-member 只转发且 DCA issue=0。
- 128-bit physical payload 到 512-bit DCA vector 的 4:1 assemble/split 和尾部不足。
- header/operand/issue/inflight/result/local-feedback 任一容量满时可恢复 backpressure。
- stream 完成后所有 state/residual 为零。

**完成证据**

- `npusim --coll-r3-selftest`：39/39；覆盖 wire 合法/非法路径、1/2/3/5 输入
  stage、single-member bypass、4:1 assemble/split、48-bit tail、截断/额外/乱序 data、
  全部有限容量背压恢复、tag/geometry 完整性和全 drain。
- 独立 `r3_oracle.py` 验证 left-fold stage、`F/B/lanes/tail`、split seq，以及
  stream-v2 的 `1+F` wire framing；`run_test_coll_r3.py`：2/2。
- R0～R2、V0～V6 selftest 和 R0～R3/V1/V4～V6 runner 全部通过；NoC congestion
  保持 14781/29109 与 14833/45441 ns，D2D V0 保持 67/67 test groups。

**阶段目标**

- 新 wire 的大消息 metadata 开销从每 data flit 一个完整 header 降为每 stream 一个
  header。
- Router-facing 合约不再允许一次 DCA request 接收超过两个 operands。

### R4 — Router production wiring 与异步 TX/RX progress

**状态：已完成（2026-08-03）**。`STREAM_V2` 已接入 production Router input、有限
assembler/operand/issue/inflight/result/egress、真实 output arbitration 和 endpoint
异步 RX session；2/3 输入、tail、L/II、64-flit 及全 drain 均通过。详见
`log/R4_development.md`。

**开发内容**

- 将 Router Sync/Match 接到所在 tile 的 `DcaComputePool`，result 根据 tag 进入 local
  feedback 或 parent output。
- 一个 Router 的所有 input/output 共享一个公平 issue port，但允许多个 inflight。
- 将阻断式 REDUCE_TX/REDUCE_RX 改为异步 collective session；root RX 从 session 启动
  起始终 armed。
- 接通真实 Router buffer/credit/output arbitration 和 normal/unicast/multicast 竞争。
- 扩展 `COLL_DCA/COLL_STREAM/COLL_DRAIN` trace 和 residual accounting。

**必须通过的测试**

- end-to-end 两输入 1D stream：issue/completion 周期符合 `L/II` oracle。
- 三输入汇合：每 beat 两次 issue，稳态接近 2×II，值正确。
- DCA result 与普通 DATA 争用同一 output，二者都能前进且 credit balanced。
- root 同时 TX/RX；40、64 chunks 不再触发旧循环等待。
- header/result queue 长期背压解除后继续完成，无重复 result。
- completion 前状态非零、completion/release 后全 drain。

**阶段目标**

- 新 DCA 路径真实穿过 production Router，不使用测试捷径或额外闭式 wait。
- 大 payload progress 不依赖扩大 endpoint raw queue。

### R5 — FPnew 风格共享向量资源与 FP 语义

**状态：已完成（2026-08-03）**。endpoint CORE request 与 Router DCA request 使用
同一 per-tile `DcaComputePool` 和仲裁器；FP32 exact 已冻结 NaN/Inf/signed-zero/结合
顺序，FP16/FP8 提供 timing-only wire/lanes/L/II，且 production CORE+DCA 并发验证
12/9 issues、21 completions、tag 不串流。详见 `log/R5_development.md`。

**开发内容**

- 将 synthetic core request 替换为 endpoint compute path 的真实共享资源请求；DCA 与
  core vector/FPU operation 使用同一个 per-tile arbiter。
- 让 Tier0 endpoint compute 和 DCA compute 读取同一份 dtype/lanes/latency/II 配置。
- 支持 FP timing profile；引入确定性 IEEE-754 helper 后逐步开启 FP32/FP16/FP8
  value mode，冻结舍入、NaN、Inf、signed zero 和结合顺序。
- trace 区分 core issue、DCA issue、两者 stall 和利用率。

**必须通过的测试**

- core idle 时 DCA 达到配置 II；core 饱和时按仲裁策略产生可解释 queueing。
- core 与 DCA 并发结果 tag 不串流。
- Tier0/Tier2 对相同 vector operation 使用一致 lanes/latency/II。
- FP timing-only 明确拒绝 value assertion；FP exact 模式与独立软件 oracle 对齐。
- FP SUM 在固定 pairwise stage 下重复运行 bit-identical。

**阶段目标**

- DCA 被建模为“借用 tile compute resource”，不再是 Router 免费专用 ALU。
- 配置对比同时报告通信收益和对 core 计算资源的影响。

### R6 — Reduce-only 端到端交付

**状态：已完成（2026-08-03）**。Reduce/ReduceScatter/AllReduce 使用 stream DCA；
Broadcast、AllGather、AllReduce-result 均为普通 unicast，端到端矩阵覆盖 N=1/2/4、
非连续 group、不同 root、tail、整数 exact 与 FP timing/exact。独立 R6 合同自测 7/7；
新增 production mixed case 在同一 Router output 实测 4 个 normal 与 18 个 collective
flits，DCA value 和全 drain 同时通过。详见
`log/R6_development.md`。

**开发内容**

- 实现 `profile=reduce_only` 对 Reduce、ReduceScatter、AllReduce 的完整展开。
- Broadcast/AllGather/AllReduce-result 强制走普通 unicast；不得进入 multicast fork。
- 完成 cycle/behavioral/oracle 三方计数、值、时间和 completion 对齐。
- 增加 reduce-only 与 baseline 的独立实验脚本和报告表格。

**必须通过的测试**

- Reduce/ReduceScatter/AllReduce：N=1/2/4、非连续 group、不同 root、tail count。
- SUM/MAX 和已支持 dtype 的 value verification。
- standalone Broadcast、AllGather、AllReduce result 的 `COLL_V4_TX/RX=0`。
- AllReduce result 对 N-1 个目标 exactly once；ReduceScatter 每目标收到正确不同 slice。
- reduce tree/DCA 存在真实流量，最终 tree、stream、DCA、barrier、DTE token 全清零。
- 与普通 Send/Recv、两个不同 CollectiveKey 并发。

**阶段目标**

- 首次提供“只改变 reduce、不改变 broadcast”的可运行配置。
- 能单独量化 in-network reduce 对流量、延迟和 core/FPU 竞争的影响。

### R7 — 改进 Reduce+Broadcast 端到端交付

**状态：已完成（2026-08-03）**。新 DCA reduce 与 AtomicMulticastFork 组合；
Broadcast 单注入、AllGather 每 source 单注入并独立 release、AllReduce result 单注入，
ReduceScatter 保持普通 Scatter。独立 R7 合同自测 6/6；加入 R6/R7 独立入口和
normal-unicast/stream-DCA 混合流量后，R5～R7 production runner 为 19/19。详见
`log/R7_development.md`。

**开发内容**

- 将新 DCA 路径与现有 AtomicMulticastFork/CollectiveTreeTable 组合。
- Broadcast/AllGather 使用单份 multicast stream；AllReduce result 使用单份 multicast。
- ReduceScatter 保持普通 Scatter。
- 保证 DCA backpressure、multicast atomic fork backpressure 和普通 traffic 仲裁可以组合，
  tree 生命周期只在最终 barrier 完成后释放。

**必须通过的测试**

- Broadcast 每目标 exactly once、source single injection。
- Reduce/ReduceScatter 与 R6 的 DCA issue/value/time 一致。
- AllReduce 比 R6 只少 result-distribution unicast flows，不改变 reduce issue 数和结果。
- 单 multicast 分支长期背压、DCA result queue 背压、普通 unicast 竞争组合后仍 progress。
- 多 tree、连续 epoch、tree-ID collision、release mismatch/unknown tree 负例。
- 全部 V0～V6 legacy tests 在显式 legacy mode 下继续通过。

**阶段目标**

- 用论文对齐的流式 DCA 替换旧 Tier2，同时保留已经验证的 Router multicast。
- 在无额外争用的同一 workload 下，`reduce_broadcast` 的 AllReduce result 流量必须小于
  `reduce_only`，且性能差异能由 multicast tree oracle 解释。

### R8 — 性能、压力、回归与文档收口

**状态：已完成（2026-08-03）**。四 profile 的 640 B/1 KiB/8 KiB/32 KiB 共 16 项
实验全部通过，mesh-link flit-hop、DCA issue/completion 与独立 oracle 精确一致；
独立 R8 C++ oracle 自测 6/6，32 KiB 的固定 latency 只计一次 pipeline fill。
OpenSMART/SMART transport 仍明确拒绝，
不在本轮范围。详见 `log/R8_development.md` 与 `NoC集合通信四档配置实验.md`。

**开发内容**

- 将原三档实验扩展为四 profile：baseline、broadcast_only、reduce_only、
  reduce_broadcast。
- 增加 1D 两输入、2D 三输入汇合、1～32 KiB payload、不同 dtype/lanes、core contention
  和 buffer depth 扫描。
- behavioral/oracle 输出 setup、network、DCA fill、DCA steady state、contention、tail/drain
  分项，而不是只给总时间。
- 更新需求、主计划、配置示例、开发日志和实验报告；明确 OpenSMART 未实现。

**必须通过的测试**

- 1D 两输入：无争用稳态为配置 II；2D 三输入：稳态为约 `2×II`。
- pipeline fixed latency 只进入启动/尾部，不随 chunk 数逐次重复。
- 5120/8192 bit 以及 32 KiB workload 全部完成，无 progress watchdog、无残留状态。
- 新 stream 的理论/实测 header/data/link flit-hop 精确一致。
- 四 profile 的实际 backend trace 与配置一致，无静默 fallback。
- 未启用 collective 的 NoC 冻结周期、DTE、D2D 全量回归 bit-identical。
- `git diff --check`、构建、自测、runner、oracle 和端到端矩阵全部通过。

**阶段目标**

- 消除旧 Tier2 在多 chunk 下由“逐 element 串行 + 每 chunk 固定 latency + 双段
  framing”造成的性能伪影。
- 大 payload 完成时间随 data/vector beat 数线性增长，斜率由 link II、pairwise DCA II
  或真实 core contention 中的最大者解释。
- 不要求所有小消息或拥塞场景都严格快于 baseline，但每个性能拐点必须能由 trace 和
  oracle 分项解释。

## 13. 总体验收矩阵

除各阶段测试外，最终至少覆盖：

| 维度 | 覆盖 |
|---|---|
| profile | baseline / broadcast_only / reduce_only / reduce_broadcast |
| operation | Broadcast / AllGather / Reduce / ReduceScatter / AllReduce |
| group | N=1/2/4、非连续、不同 root、并发 group |
| topology | 1D path、2×2、至少一个三输入汇合的更大 2D mesh |
| payload | tail、1/4/16/32/40/64 chunks、1～32 KiB |
| DCA | single-member、2-input、3+-input stages、L>1、II=1/2、FIFO 满 |
| dtype | UINT/INT exact；FP timing；完成 helper 后 FP exact |
| contention | normal unicast、multicast、core FPU、DCA、result reinjection |
| lifecycle | 连续 epoch、多 tree、collision、release、全 drain |
| backend | cycle 真实事件；behavioral/oracle 无重复计费 |

最终测试不仅检查总时间，还必须检查：

- source stream/header/data 注入数；
- 每 link header/data flit-hop；
- 每 stage expected/received bitmap；
- DCA issue/completion/tag、inflight 峰值、latency/II；
- vector lanes、tail mask、pairwise stage 数；
- core/DCA arbitration 和 stall；
- result per-target exactly once；
- header/operand/issue/inflight/result/local-feedback occupancy；
- endpoint TX/RX 并发和 progress；
- Router lock/refcount、tree registry、barrier、DTE token 最终清零。

## 14. 关键文件和预计改动

- 配置：`llm/include/defs/spec.h`、`llm/src/defs/spec.cpp`、
  `llm/src/utils/config_utils.cpp`
- 描述符/契约：`llm/include/dte/coll_types.h`、`coll_latency.h`、
  `coll_innetwork_reduce.h`
- 新计算资源：建议新增 `llm/include/dte/dca_compute_pool.h`、
  `llm/src/dte/dca_compute_pool.cpp`
- tree/stage/wire：`llm/include/dte/coll_innetwork_reduce.h`，建议拆出
  `coll_reduce_stream.h/.cpp`
- 配置展开：`llm/src/monitor/config_helper_core.cpp`
- Router：`llm/include/router/router.h`、`llm/src/router/router.cpp`
- endpoint/session：`llm/include/workercore/workercore.h`、
  `llm/src/workercore/workercore.cpp`、`llm/src/workercore/logic.cpp`
- multicast 复用：`llm/include/dte/coll_multicast.h`、
  `llm/src/dte/coll_multicast.cpp`
- 测试：`llm/src/dte/coll_*_selftest.cpp`、`llm/test/noc_collective/`
- 文档：`notes/extensions/NoC/NoC集合通信建模需求.md`、
  `NoC集合通信建模计划.md`、`NoC集合通信三档配置实验.md`、`log/`

## 15. 兼容性、迁移和退出条件

- 普通 NoC 默认路径不增加 DCA/stream 判断的行为变化；collective gate 关闭时 Router
  冻结回归必须完全一致。
- 旧 `tier=2` 在重构后映射到新的 `reduce_broadcast`；需要复现旧结果时必须显式使用
  `legacy_router_alu + legacy_two_segment`。
- legacy backend 至少保留一个发布周期，只运行合同回归，不继续增加功能；新四 profile
  验收完成后再单独评审是否删除。
- 配置、trace 和报告必须输出实际 backend、wire version、DCA width/lanes/L/II 和
  value mode，避免把 timing-only 结果误认为论文的 bit-accurate FP 实现。
- **退出条件已满足（2026-08-03）**：R0～R8 全部完成，四 profile 矩阵与 32 KiB
  压力通过，新时序由 oracle 分解；R6/R7/R8 独立自测为 7/7、6/6、6/6，production
  mixed runner 为 19/19；普通 NoC/DTE/D2D 与 legacy collective 回归保持通过。当前唯一
  显式范围外项目是 OpenSMART transport。
