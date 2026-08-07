主要目的是建模出集合通信原语，可能要配合其他部件（比如DTE）的修改
要能：
- 支持顶层原语封装，能直接调用集合通信原语发动noc集合通信
- 能兼容现有的通信机制
- 可以选择开启哪一级集合通信加速方案

背景
主流芯片普遍采用"原语分解"思路
把高层集合通信（All-Reduce、All-Gather、All-to-All等）分解为更底层的发送/接收原语，是业界普遍做法。原因很实际：
- 硬件面积有限，不可能为每个集合操作都做一套专用电路
- 集合通信算法本身就在不断演进（ring、tree、halving-doubling、2D-ring、hierarchical...），固化到硬件反而失去灵活性
- 组合式原语可以覆盖未来新出现的通信模式
但具体分解粒度，各家差异很大
你描述的方式——发送端 {unicast, scatter, broadcast, shuffle} × 接收端 {unicast, gather}——这种明确的"TX原语集 × RX原语集"矩阵式设计，在业界属于偏硬件导向的一种做法，常见于自研NoC且通信引擎（DMA/CCE）和路由器耦合紧密的芯片。
类似风格的有：
- Tesla Dojo：每个节点的DMA引擎支持multi-cast send和gather receive，通过组合实现各种集合模式
- Graphcore IPU：exchange阶段把通信拆成send pattern和receive pattern，编译器来组合
- 一些国产训练芯片（包括寒武纪、壁仞、燧原等的部分产品）：都采用类似的"发送模式+接收模式"可编程组合
为什么要这样设计
1. 集合通信本质上可以被分解
2. 避免为每种集合通信单独做硬件导致硬件结构数量爆炸
3. 适合编译器做映射
4. 适合NoC/router内融合，比如：broadcast原语中 router 直接 packet replication、reduce原语中 router 可以边转发边规约
5. 天然支持流水化：通信可以chunk化
6. 支持未来不确定的workload

建模行为
原语发起的单位（重要）
认为每个核上有一个数据搬运硬件DTE，每个核上的每个DTE可以发起TX/RX原语。
所以通信发起的单位是核，而不是一张SoC。
需支持的原语组合
RX / TX
Unicast TX
Scatter TX
Broadcast TX
Unicast RX
Send/Recv (P2P)
Scatter
Broadcast
Gather RX
Gather
AllToAll
AllGather
Reduce RX
Reduce
ReduceScatter
AllReduce
TX原语
（仅供参考，注意通信单位的粒度是核）
Unicast TX：一个发送端，数据发往单个目标。语义上是完整数据的点对点传输，不做任何切分或复制。
Scatter TX：一个发送端，将数据切片后分发到多个目标，每个目标收到不同的片段。
Broadcast TX：一个发送端，将完整数据复制发往多个目标，每个目标收到相同的副本。依赖 NoC 多播树硬件支持。
RX原语
Unicast RX：接收单个来源的单片数据，写入指定 dst_addr，完成后置位 completion。最简单，无乱序处理需求。
Gather RX：接收来自多个源节点的数据，按 src_rank 或预设 offset 写入目标 buffer 的不同位置。硬件难点在于包的乱序到达——NoC 不保证多条 Unicast 的到达顺序，因此 Gather RX 引擎需要一个 reorder buffer（或基于 sequence number 的重组逻辑），把乱序的包放到正确的 slot 里。这个 buffer 的深度直接影响面积和延迟的权衡。
组合集合通信原语
1. P2P — Unicast TX × Unicast RX
最基础的单发单收。硬件上对应一条描述符：src_addr / dst_node_id / size。NoC 路由器按目标 ID 做单播路由，无需多播树或 reduction 引擎参与。延迟最低，是其他所有原语的构建基础。
2. Scatter — Scatter TX × Unicast RX
根节点将一块大 buffer 按 rank 切成 N 份，每份以 Unicast TX 独立发往对应节点。TX 侧描述符链：stride_offset = rank × chunk_size，dst_id = rank。接收端每个节点只 Unicast RX 自己的那一片。硬件上等价于 N 条并发 Unicast 描述符，或支持 scatter list 的 DMA engine 一次性提交。
3. Broadcast — Broadcast TX × Unicast RX
发送端走多播树复制数据，但接收端只关心自己的副本，等价于普通 Broadcast。这种描述方式在某些芯片中只是"接收方不需要 gather"的特例，本质与标准 Broadcast 相同，无额外硬件区分。
4. Gather — Unicast TX × Gather RX
各节点各持一片数据，均以 Unicast Out 发往根节点；根节点开 Gather InStream，按 src_rank 将到来的 N 片按序写入连续 buffer（offset = rank × chunk_size）。硬件 Gather In 需要 reorder buffer 处理乱序到达。是 Scatter 的逆操作。
5. AllToAll — Scatter TX × Gather RX
每个节点持有 N 片，第 j 片以 Scatter Out 发往 rank-j；同时每个节点开 Gather In 接收来自所有节点发给自己的那一片。等价于矩阵转置通信：第 i 节点的第 j 片 → 第 j 节点的第 i 位置。带宽压力最大（全交换，流量 = N²×chunk），最需要 QoS 保障。
6. AllGather — Broadcast TX × Gather RX
每个节点把自己持有的那一片以 Broadcast Out 广播出去，同时所有节点开 Gather In 拼装来自 N 个节点的 N 片，得到完整数组。图中这个映射很清晰：Broadcast Out（每个节点广播自己的片）+ Gather In（每个节点聚合所有片）= AllGather。Ring-AllGather 是其高效实现变体。

如何理解原语的行为
用延时建模理解原语的行为
TX原语
1. 单次Unicast TX（传输数据量为P）耗时计算：
时间轴示意（H=3 跳，P 占 L 个 flit）：

t=0
│
│  源端注入 flit:  [f1][f2][f3]...[fL]
│                  ↑头 flit                    ↑尾 flit
│                  │                             │
│   头 flit 在网络中流水穿越 ───→ 到达目标       │
│                  T_Network                    │
│                                               │
│                            尾 flit 也需要走完 T_Network
│
└──→ 最终完成时刻 = (L-1)·t_flit + T_Network + t_flit
                  ≈ P/bw + T_Network
T = P/bw + T_Network
暂时无法在飞书文档外展示此内容

2. Scatter/Broadcast TX耗时计算：
broadcast：(N-1)个包，包大小为m
scatter：(N-1)个包，包大小为m/N

T(broadcast) = (N-1)*m / bw + T_Network
T(scatter) = m/bw + T_Network

RX原语
router弹出到DTE的耗时

1. Unicast RX
  router弹出到DTE：1 cycle
  （包在DTE中处理的延时、写入SRAM的延时在DTE、SRAM有关地方另算；完成通知消费方的时间忽略）

2. Gather RX
  Unicast RX时间+(N-1)个源到齐的同步时间+拼接时间（假设1 cycle）

3. Reduce RX
  Unicast RX时间+(N-1)个源到齐的同步时间+对齐时间（假设1 cycle）
  （ALU计算时间作为计算操作，在计算原语中算）

三层次集合通信实现
基础方案：拆成多个unicast
基本复用基础的仿真器设置，只是添加集合通信原语，底层由多个unicast实现。

进阶1，只加速broadcast
新增三类硬件：VCT table（组播树表）、advanced request / bypass 控制、VC partition / multicast control。
- 做法：源只注入 1 个 multicast 包（带 tree_id）→ router 在分叉点单周期把包复制（fork）出去（架构图里的 mXbar + redpath: broadcast single-cycle fork datapath）。
- 效果：消掉了初始方案里 (N-1)·m/bw 这个源侧串行注入的开销，一次注入就能在网内树状分发（参考 Krishna《Towards the Ideal On-Chip Fabric for 1-to-Many...》、OpenSMART 单周期多跳）。
- 注意（文档加粗）：broadcast 的硬件对 scatter 没用。scatter 虽是 one-to-many，但发给每个目标的是不同的 flit，"复制同一份内容"的 fork 逻辑起不了作用。

进阶2，加速broadcast和reduce
在加速broadcast的基础上再加速reduce
VCT-guided + DCA-assisted In-network Reduce Router 在 router 内部加 Operand/Hdr Buffer + Sync/Match（用 expected-input bitmap 匹配各源是否到齐）+ DCA（旧 V5/V6 将其解释为 Router 专用计算单元），让包在经过 router 的途中就做加法。旧版延时 `t = max(comp, p/128) + 54`。不再由 root 一点逐个 ALU，而是沿树边走边聚合。

> **重构边界（R0 冻结）**：上述公式和“Router 专用计算单元”解释仅属于 V5/V6
> legacy contract，用于重放旧测试与实验，不代表论文中的 DCA。论文对齐的新模型见
> `NoC集合通信重构计划.md`：DCA 借用 tile/cluster 的向量/FPU 资源，每次 issue 处理
> 两个等宽 vector operands，lane 内元素并行，多输入通过确定性的 pairwise stage
> 分解；pipeline latency 与 initiation interval 分开建模。新生产路径不得调用 legacy
> `max(comp,p/128)+54` 公式。

R1 起配置不再用一个 `tier` 同时代表 Broadcast 和 Reduce 硬件。规范选择为
`baseline`、`broadcast_only`、`reduce_only`、`reduce_broadcast` 四个 profile，并可
显式指定 `broadcast_backend=unicast|multicast`、
`reduce_backend=endpoint|dca_offload|legacy_router_alu` 与
`reduce_wire=stream_v2|legacy_two_segment`。旧 `tier=0/1/2` 仅作为规范 profile 的
兼容 alias；旧 V5/V6 数据面必须使用 `legacy_router_alu + legacy_two_segment +
allow_legacy_backend=true`，不得由新 DCA 配置静默回退。

配置与协议必须共用唯一的 reduce-wire 枚举定义，冻结
`legacy_two_segment=0`、`stream_v2=1`，禁止跨层复制出数值不同的同义枚举。
将定义从 `coll_refactor_contract.h` 提取到 `coll_wire.h`、并令配置类型成为其别名，
按 R0 冻结契约的“唯一真源”修订管理：协议名称和冻结数值不得改变，修订必须在计划及
R0/R2 开发日志中留痕。
DCA 结构/value-mode 语义校验只在 collective 已启用且实际选择
`dca_offload` 时执行；inactive DCA 配置不得使 baseline/endpoint 启动失败。
`vector_bits%dtype_bits==0` 必须按实际 reduction workload dtype 校验。

R2 起 DCA 计算资源必须建模为每 tile 一个共享 ComputePool：core 与 DCA 请求进入有限
pending queue，经一个二输入 vector issue port 仲裁；已发射请求占用有限 inflight
context，完成结果进入有限 result queue。一个请求的完成周期为 `issue+latency`，
下一请求最早发射周期为 `previous_issue+II`，不得等待上一请求完成后才发射。tag 从
submit 保持 live 直到 result 被消费，重复/未知/提前 completion 必须确定性报错。R2
仅冻结隔离资源模型，不得绕过 R1 gate 接入 Router 或宣称端到端 DCA 性能。
配置文件中的 dormant DCA 块与已实例化 ComputePool 必须区分：前者仅严格解析字段和
枚举，在 `enabled && dca_offload` 时才执行资源/value-mode 语义校验；后者代表真实
DCA 资源，构造时必须无条件执行完整 `Validate()`。
方案流程展示：
                                +--------------------------------------+
Incoming Links / Local Ingress  |   CollectiveTreeTable (shared)       |
(N / S / E / W / Local)         |   - broadcast tree info              |
        |                       |   - reduce tree info                 |
        v                       +------------------+-------------------+
+-------------------------+                        |
| Input Ports + VC Buffers|<-----------------------+
| p·v·A_VC,coll           |   tree_id / mode / seq / dtype / control
| - multicast VC state    |
| - reduce metadata       |
+------------+------------+
             |
             v
+-------------------------+
| RouteUnit_coll          |
| - XY route              |
| - VCT/Whirl decode      |
| - reduce tree decode    |
+------------+------------+
             |
             v
+-------------------------+
| Arbiter_inport_coll     |
| - multicast req gen     |
| - reduce operand match  |
+------------+------------+
             |
             +--------------------------------------------------+
             |                                                  |
             | broadcast / multicast path                       | reduce path
             v                                                  v
+-------------------------+                     +-------------------------------+
| AR / bypass             |                     | Sync / Match                  |
| - Advanced Request      |                     | - expected-input bitmap       |
| - bypass mux/control    |                     | - flow/seq matching           |
+------------+------------+                     | - LZC / priority select       |
             |                                  +---------------+---------------+
             v                                                  |
+-------------------------+                                     v
| Arbiter_outport_coll    |                     +-------------------------------+
| - mSA                   |<--------------------| Operand / Hdr Buffer          |
| - result reinjection    |                     | - operand staging             |
+------------+------------+                     | - header buffer               |
             |                                  | - result staging              |
             v                                  +---------------+---------------+
+-------------------------+                                     |
| mXbar + redpath         |<------------------------------------+
| - broadcast single-cycle|                                     |
|   fork datapath         |                                     v
| - DCA operand mux       |                     +-------------------------------+
| - DCA result mux        |-------------------->| DCA_Ctrl                      |
+------------+------------+                     | - offload / reduction control |
             |                                  | - issue to DCA compute path   |
             |                                  +---------------+---------------+
             |                                                  |
             |<---------------- result reinjection -------------+
             |
             v
+-------------------------+
| Output Links / Local    |
| N / S / E / W / Local   |
+-------------------------+

Misc blocks:
  [misc]
  - opcode/dtype/config/debug/clock-gating

可供参考的参考文献：
1. Stefan Mach, et al., “FPnew: An Open-Source Multiformat Floating-Point Unit Architecture for Energy-Proportional Transprecision Computing”.
2. Mingran Wang et al., “OpenSMART: Single-Cycle Multi-Hop NoC Generator in BSV and Chisel”.
3. Luca Colagrande, Lorenzo Leone, Chen Wu, Tim Fischer, Raphael Roth, and Luca Benini. “A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators.”
4. Tushar Krishna, et al., "Towards the Ideal On-Chip Fabric for 1-to-Many and Many-to-1 Communication".
5. Si Qing Zheng, et al., "Algorithm-Hardware Codesign of Fast Parallel Round-Robin Arbiters".

## 重构实现对齐状态（2026-08-03）

- VCT-guided many-to-one 路径采用每 stream 一个 header、128-bit physical flit 到
  configurable vector beat 的 assembler、确定性 binary pairwise stages 和有限背压。
- DCA 被建模为 tile 上 CORE/DCA 共用的 vector/FPU compute pool，latency 与 II 分离，
  允许多个 tagged inflight；固定 latency 只进入 pipeline fill，不随 chunk 重复。
- `reduce_only` 仅加速 Reduce 族，Broadcast/AllGather/AllReduce-result 为普通 unicast；
  `reduce_broadcast` 在相同 DCA reduce 上组合单注入 multicast。
- integer SUM/MAX 与 FP32 exact 进行值验证；FP16/FP8 目前仅 timing-only。SMART
  single-cycle multi-hop 与跨 die hierarchical collective 尚未实现，配置期明确拒绝。
