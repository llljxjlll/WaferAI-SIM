# D2D Link 开发版本与测试计划

## 1. 文档目标

本文档用于规划 D2D Link 建模的分版本实现范围、测试层级、测试用例和每一版的准入标准。

整体开发遵循以下原则：

1. 先解决寻址、拓扑和消息格式，再加入跨 die 传输。
2. 先实现单端口、静态、确定性的最小闭环，再增加多跳、拥塞和高级策略。
3. 每一版只引入一类主要复杂性，并能独立运行和验证。
4. 暂不支持的配置必须在启动阶段明确报错，不能静默降级或给出不准确结果。
5. 先用可解析的微基准证明模型正确，再使用真实 workload 评估性能。
6. 测试不能只观察最终 `sc_time_stamp()`，还要能解释延迟来自路由、序列化、排队还是背压。

### 范围与前置说明

- **`mem` 端口 role 本计划不 exercise（点 6）**：端口框架统一支持 `HOST/MEM/C2C` 三种 role，但「内存控制器挂边缘端口」是独立于 D2D 的语义迁移（现有内存是 per-core `dram_bw`）。本测试计划只在 V0 覆盖 `MEM` role 的**结构与配置校验**（数据结构、非法配置报错），**不测 mem-at-edge 的真实内存流量**；该迁移若推进，另立版本与测试，不并入本路线。
- **尽早验证 cycle 全链路可跑通（点 8）**：D2D 的 V1–V3 建立在**周期精确路径**（`use_beha_noc=false`）之上，Behavioral 排到 V4。而现有工程默认几乎都走 beha、cycle 路径平时较少被使用、稳定性未知（见 `noc_congestion` 落地记录）。因此 **V0/V1 的 smoke 必须先用「无拥塞单流 + cycle」验证 cycle 全链路（含 SRAM/DRAM 均在 cycle 模式）能正常结束不死锁**，把这一风险前移，避免堆到 V3 才暴露。

## 2. 总体版本路线

```text
V0 多 die 基础设施重构
  ↓
V1 相邻 die 单链路 MVP
  ↓
V2 多跳路由与协议级活性
  ↓
V3 周期精确拥塞与背压
  ↓
V4 Behavioral 模型与定量校准
  ↓
V5 多端口、条带化和高级策略
```

| 版本 | 核心目标 | 主要新增能力 | 主要不支持项 |
| --- | --- | --- | --- |
| V0 | 消除单 die 基础假设 | 全局寻址、Die/Port/Link 数据结构、拓扑与配置校验 | 不传输 D2D 流量 |
| V1 | 跑通第一条完整 D2D 通信 | 相邻 die、单端口、REQ/ACK/DATA、固定延迟 | 多跳、多端口、复杂拥塞 |
| V2 | 支持任意规则 die mesh | die 级 XY、多跳接力、per-die pinning、协议级活性监测 | 有限资源网络死锁安全、多端口、动态选路 |
| V3 | 建模真实资源争用 | 有限缓冲、带宽、仲裁、背压、网络死锁安全、混合流量拥塞 | Behavioral 快速近似、多端口聚合 |
| V4 | 提供快速且可解释的模型 | Behavioral 公式、oracle、与 cycle 模型校准 | 跨流争用的行为级近似 |
| V5 | 支持复杂物理组织与优化 | 多端口、link group、striping、hash/hybrid/dynamic | 无；实验性策略需单独标注 |

## 3. 测试所需的可观测性

在 V1 前应准备好最基本的统计接口；V3 前应补齐拥塞和背压统计。否则只能知道“运行变慢”，无法判断模型是否按预期工作。

### 3.1 Flow 级指标

每条 flow 至少记录：

- `flow_id = (source_global_id, tag, epoch/subflow_id)`；
- source、destination、消息类型和有效 payload；
- 选择的出口端口及经过的 die/port 序列；
- REQUEST 发出、ACK 返回、首包注入、首包到达、尾包到达时间；
- 发送包数、接收包数、重复包数、丢包数和乱序数；
- NoC stall、port stall、D2D stall 和目的端 stall 周期。

### 3.2 Port 和 Link 级指标

每个 C2C port 和 D2D link 至少记录：

- 输入/输出包数和字节数；
- busy cycles、idle cycles 和理论容量；
- 最大、平均 buffer depth；
- buffer full cycles；
- 仲裁等待周期；
- 正向、反向的独立统计；
- link group 的聚合利用率。

### 3.3 仿真结束不变量

仿真正常结束时必须断言：

```text
所有已发送数据均被接收
所有 flow 均完成
所有 port/link/router buffer 均排空
所有 output lock、VC 和 credit 均归零
不存在未匹配的 subflow 或未完成重组
```

### 3.4 推荐性能指标

```text
goodput       = 有效 payload bytes / DATA 完成时间
link_util     = 实际传输 flit / (link_bw × 活跃周期)
efficiency    = goodput / 理论可用峰值带宽
queue_delay   = 总延迟 - 无争用理论延迟
handshake_lat = DATA 首包注入时间 - REQUEST 发出时间
```

同优先级多流的公平性可使用 Jain fairness：

```text
J = (Σ throughput_i)^2 / (n × Σ throughput_i^2)
```

## 4. 分版本实现与准入测试

### 4.1 V0：多 die 基础设施重构版

#### 实现目标

在不引入真实 D2D 流量的前提下，拆除当前代码中的单 die、方形 mesh 和 `GRID_SIZE` 多义假设。

#### 实现内容

- 拆分尺寸和 endpoint 常量：

```text
CORES_PER_DIE
DIE_X / DIE_Y / DIE_COUNT
TOTAL_CORES
HOST_ENDPOINT_ID
```

- 实现以下无歧义转换：

```text
global_id ↔ (die_id, local_id)
die_id ↔ (die_x, die_y)
local_id ↔ (local_x, local_y)
```

- 消息头支持全局 source/destination，并修正所有字段的初始化与序列化。
- 引入 `Die`、`Port`、`D2DLink`、`Endpoint` 等基础数据结构。
- 重构 topology builder，去掉边界方向的取模连接（torus→开边 mesh）。**时序不变性论证（点 7）**：现有 `GetNextHop` 是目的导向的纯 XY 比较，**从不选择 wrap 方向**——环绕连接虽物理存在但从不被任何路由使用，故拆除后**不改变任何实际路由**。V0 需显式验证「无既有测例依赖环绕路径」（即 T00/noc_congestion 时序确实不变），并把该论证写入 README。
- 固定 N/S 与坐标增减方向的定义（`NORTH=y+`：`S=row 0`、`N=row Y-1`、`W=col 0`、`E=col X-1`）。
- 支持矩形 die 内 mesh 和矩形 die mesh。
- 实现端口、peer 和参数的启动期配置校验。
- **HOST/mem 端点统一化（点 5）**：HOST 不再硬编西边 margin，而是 `role=HOST` 的端口；需预计算 `port_for_host[core]`（去 HOST 走哪个端口，类比 `port_for[core][dir]`），并对 `des_` 落在保留区时解码 endpoint 类型（core/host/mem）。
- 旧配置未指定 die 字段时解释为单 die。
- 建立 flow/link 统计的基础接口。

#### 本版不实现

- 跨 die 包传输；
- D2D latency/bandwidth；
- 多端口策略与 striping。

#### 必须通过的测试

1. 地址编解码边界测试。
2. 消息序列化/反序列化逐字段往返测试。
3. 1×1、2×1、1×2、2×2 die topology 构造测试。
4. 方形和矩形 die 内 mesh 测试。
5. 非法 peer、重复端口、越界端口和缺失方向配置必须报错。
6. **HOST 可达性校验（点 5）**：断言至少 1 个 `role=HOST` 端口存在且每核可达；构造「所有 margin 端口都被 C2C override」的配置，必须在启动阶段报「HOST 不可达」而非静默运行。
7. `port_for_host[core]` 对每核有定义；endpoint 类型解码对 core/host/mem 边界值正确。
8. 现有单 die workload 全部能够运行。
9. 现有 `llm/test/noc_congestion` 的功能和趋势保持不变（验证拆环绕不改变时序，见点 7）。
10. 单 die 下所有 D2D port/link 活动计数为 0。

#### 完成标准

只改变内部结构，不改变旧配置下的功能和网络时序；所有旧回归测试通过。

> **实现状态（2026-07，逐轮 review 后更新）**：单 die **仿真时序（ns）精确一致**全程通过
> （noc_congestion 四场景 14781/29109、14833/45441 不变）。测试见 `llm/test/d2d_link/`：
> **纯函数自测 165/165 + 端到端 23/23 组**（含 3 个多 die 实例化组：2×1、1×2、2×2；
> 165 含 V1-pre 2a..3b-2a 的 HOST 挂载表结构/路由收敛/跨 die 拒绝/config 挂载表/RouterUnit 接口/egress anchor 自测；端到端 23 组含 W/S/E/N 四方向 HOST 闭环（方阵 4×4 + 矩形 4×2），签名 + per-lane 加固）。
> 验证口径 = 退出码 + sim-time(ns) + 独立层级模块计数 + 运行后 D2D 统计（非完整 stdout/trace 比对）。
> **V0a 已完成**：编址常量/helper、die 配置解析、端口结构与配置校验（参数范围、
> HOST 可达性、core/host/mem endpoint 解码）、flow/link 统计接口、消息全字段序列化往返、
> `GetInputSource` 矩形修正、N/S 定义、Msg 类内默认值、endpoint 容量校验。
> **V0b 已完成（增量）**：V0b-1 编址/消息语义收口（每 die HOST endpoint 区间 `IsHostEndpoint`、
> DONE/ACK 发本 die host、DATA 携带真实全局 `source_=cid`）；V0b-3 live torus 拆除
> （`OpenMeshNeighbor` 开边 mesh + 终结通道）；V0b-2 前置 hardening（地址防溢出、`delete[]`、
> g_dram_kvtable UAF/double-free）；**V0b-2A 运行时多 die 模块实例化**（核级数组/`RouterMonitor`/
> `WorkerCore`/cache 按 `TOTAL_CORES`，`GetCoreHWConfig` 同构映射，die>0 HOST 端口终结）——
> 2×1/1×2 建 32、2×2 建 64 个 router+worker（**独立层级计数**核对），die>0 空闲，die0 workload
> 仍 29109ns，D2D 活动 0，无 unbound/multi-writer。
> **V0b-2C0 已完成（可靠跨 die 拒绝）**：`config_helper_core::PreflightValidateWorkload` 在绘图/
> 构造/elaboration 前扫描**原始 JSON** 的 core/cast/source id，拒绝跨 die cast、越界、die>0 目标；
> `plot_dataflow` 多 die 跳过（不再段错误）；`id_space`（缺省 die0-local / `global`）消歧。runner
> 加 `core0→core16` 拒绝用例（10s 超时保护）：非零退出、报 `cross-die traffic requires D2D Link`、
> **不进入仿真、不永久等待**；单 die 逐位不变。
> **✔ 4 增量全部完成（每轮一片、单 die 逐位不变）**：
> - **2C1** global-id 归一化（`monitor/workload_normalize`：`NormalizeWorkloadJson` +
>   `ValidateWorkloadStructure`；`o2r/r2o` 生产路径本就 map 化）。
> - **2B0** `HostEnvelope` + `LegacyHostEnqueue`（`config_helper_core::BuildConfigMessages/
>   BuildStartMessages` 返回信封，解耦物理 HOST lane）。
> - **2B1** per-die HOST lane（`HOST_LANES=GRID_Y*DIE_COUNT`；lane i↔全局行 i↔router i*GRID_X↔
>   write_buffer[i]，`dest/GRID_X` 天然泛化多 die；`config.cpp` `id>=GRID_SIZE`→`TOTAL_CORES`；
>   移除 die>0-unrunnable guard）。
> **✔ 里程碑达成——workload 真正在 die>0 运行**：
> - **T3** die1 平移 workload → 16 个 die1 核执行、HOST1 收齐 DONE、sim-time=29109（die0 基线等价）、D2D=0。
> - **T2** die0+die1 合并 → 两 die 各 16 核并行、sim-time=29109（零干扰）、D2D=0。
> 单 die 四场景 sim-time 精确一致；自测 165/165、runner 23/23。
> **V1-pre / V0-exit ✔ 全部完成**：配置驱动的 HOST attachment 是 V1 前置里程碑（**非 polish**——V1 要测
> E/W/N/S 四方向 D2D，西边缘可能同时有 HOST 与 C2C，须由配置驱动且不冲突）。已完成 **Inc 1**
> （HOST 挂载表 `g_host_attach` 接缝，仅 legacy 构造器）、**Inc 2a**（补全 core↔lane 映射、
> `HOST_LANES←n_lanes` 强一致、独立 `ValidateHostAttach()`、入队防 `q[-1]`）、**Inc 2b**
> （`GetNextHop/Reverse` 朝本 die 挂载 tile 收敛 + **显式拒绝跨 die HOST** + 逐核距离严格下降自测）、
> **Inc 3a**（config 驱动挂载表构造器：`role=HOST` 端口建 lane/tile/core_lane；W-host==legacy 故
> e2e 仍 29109；**不改路由**）、**Inc 3b-1**（RouterUnit HOST 接口 `IsMarginCore`→`IsHostAttachTile`
> + `ValidateHostAttach` 拒绝重复 lane_tile）、**Inc 3b-2a**（`GetNextHop` 加显式 `anchor_core`，HOST
> 挂载 tile 取自消息 `source_` 而非 `pos`，router 传 `m.source_`，divergence 单元测试证明用固定
> source 不退回 pos）、**Inc 3b-2b/3c**（W/S/E/N 四方向 HOST 端到端，**方阵 4×4 + 矩形 4×2 die 内 mesh**
> 都覆盖，每方向连续两次均校验 mismatch=0、`DONE(source)` 签名 + ACK 计数与 west 一致、预期 per-lane
> （读 GRID_X/Y：4×4 行 `2,2,2,2`/列 `4,0,4,0`；4×2 行 `2,2`/列 `2,0,2,0`）、D2D=0、绑定无误、
> `GRID_X*GRID_Y` 核；sim-time 两次一致、差异符合各自 hop 变化——**非西侧 HOST 四方向 × 方阵/矩形运行时
> 闭环达成**。附带修复：CONFIG ACK tag 来自 `Recv_prim(RECV_CONF)` 未初始化 `tag_id`（UB，4×2 暴露、
> 4×4 巧为 0）——不影响 HOST lane 选择但 V1 flow-key 会用 tag，已修为类内默认值 + 显式 tag=0 契约，
> 现严格 (source,tag) 比对 + ACK 结构断言）；
> 均单 die sim-time 精确一致。
> **Inc 4 ✔ V0 基线冻结**：阻塞性准入门 = 自测 165/165 + D2D runner 23/23 + NoC 四场景精确
> （14781/29109、14833/45441）+ 全部负例启动报错；冻结范围 = dataflow + 配置驱动 HOST（四方向 ×
> 方阵/矩形）+ 多 die 实例化 + 跨 die 拒绝；非阻塞后续 = ASan/teardown/非 dataflow 多 die/ACK ack_seq。
> **V1-pre / V0-exit 全部完成，可正式进入 V1。** 详见 `llm/test/d2d_link/README.md` 基线冻结节。
> **V0 纯 polish（不影响功能）**：ASan/UBSan、WorkerCore teardown、非 dataflow（pd/gpu）多 die 化
> （V0 不要求）；cache/DUMMY 已随默认构建的多 die 用例覆盖。详见 `llm/test/d2d_link/README.md`。

### 4.2 V1：相邻 die 单链路 MVP

#### 实现目标

跑通第一条完整跨 die 通信链：

```text
SEND_REQ → RECV/ACK → SEND_DATA
源 NoC → D2D Link → 目的 NoC
```

#### 当前增量状态

- **c0 ✔**：`FlowKey` + 消息内 16-bit pinned `exit_port_`，未 pin 的现有消息 wire 保留位仍为零。
- **c1a ✔**：相邻/多跳/实际双向 peer-link 的 preflight helper 已可测；c1/c2 期间生产 gate 保持
  关闭，避免“校验接受但运行挂死”的中间版本；c3 闭环后已显式放行相邻 peer-link。
- **c1b ✔**：REQUEST（串行/并行路径）和反向 ACK 在各自源核选择一次出口，Router 控制路径消费
  固定出口；双向逐跳测试证明各跨一次正确有向 link，并覆盖序列化、same-die 等价与缺 pin 拒绝。
- **c2 ✔**：每条 `SEND_DATA` 原语在源核选一次 C2C 出口，所有 DATA 包携带相同 pin；Router 数据
  dispatch 消费固定出口，进入目标 die 后恢复片内 XY，尾包按同一 `out` 解锁。覆盖 3 包同 pin/序列化、
  双向 link、same-die 等价、缺 pin 与非 DATA 跨 die 拒绝；`Send_prim::deserialize` 清空运行态 pin，
  并先解析 `type` 后读取 DATA 字段。
- **c3 ✔**：生产 preflight 放行“相邻 + 精确双向 peer link”的 cast。两核 workload
  `core0(die0) → core16(die1)` 连续两次完成 REQUEST → 反向 ACK → 4 个 DATA → DONE，均为
  **398 ns**；按类型 Link capture/delivery 为 REQUEST=`1/1`、ACK=`1/1`、DATA=`4/4`，总计=`6/6`，
  HOST mismatch=0、唯一 DONE=`{16:1}`、五个生产协议阶段日志齐全、无绑定错误。3×1 die0→die2
  生产负例仍在仿真前拒绝，多跳边界未被 gate 放宽。
- **d1 ✔**：2×1 正反向覆盖 E/W，1×2 正反向覆盖 N/S；每方向连续两次均为 398 ns，
  REQUEST/ACK/DATA 全闭环、反向 ACK 回 producer、consumer 唯一 DONE，Router/Link
  drain=`0/0`。非法 `side!=dir` 以精确诊断在仿真前拒绝。
- **d2 ✔**：DATA capture/delivery 两侧记录包数、顺序敏感 seq hash、完整 payload checksum、
  连续序号区间、唯一尾包/尾长和 cycle span。覆盖 1/2/5(partial tail)/7/8/9/32 包；
  5 包尾长 96 bit，其余 128 bit，全部指纹守恒、顺序正确、完成时间严格递增并排空。
  7/8/9 是消息尺寸边界覆盖，不把 V1 无限功能 FIFO 误表述为有限缓冲测试。
- **d3 ✔**：生产端到端 `link_latency=0/1/7/20` 各跑两次，得到
  278/284/320/398 ns。单次 Link DATA 交付相对 capture 增量为 `L cycle`，DATA span
  恒定；完整 REQUEST→ACK→DATA 事务满足
  `T(L)-T(0)=3*L*CYCLE=6L ns`（`CYCLE=2 ns`）。
- **当前门 / V1 完成**：218/218、Link 18/18、runner **48/48**、NoC 冻结值全绿。
  V1 支持范围（相邻 die、每方向单端口、1 packet/cycle、固定 latency、功能性无限 FIFO）
  已全部闭合；多跳属于 V2，有限缓冲/背压属于 V3。

#### 实现内容

- 支持两个相邻 die。
- 每个存在的邻居方向最多一个 C2C 端口。
- 第一版限制 `side == dir`。
- 使用 global destination 自动识别是否跨 die，保留现有 `SEND_REQ/SEND_DATA` 协议阶段。
- REQUEST、ACK 和 DATA 全部能够经过 D2D Link。
- 静态、确定性的端口选择。
- flow 进入一个 die 时选择一次出口，离开该 die 前不允许重选。
- **flow 标识与 output_lock 语义（点 3，V1 审查修正）**：本工程 `tag == 接收核 recv_tag == 全局核 id`，即 tag 就是**接收端聚合槽**。故 `output_lock` **有意按 tag 锁**——同 tag（=同接收核）的多个源流本应**共享锁、交错通过**（「多发一」聚合，接收端按包内地址重组）；给锁加 source 维反而会把多发一错误拆成串行。**防别名的正确不变量是 tag→dest 唯一性**（同一 tag 只能指向一个接收核，preflight 校验，禁止两接收核撞 tag），而非「按 source 分流」。`FlowKey(source,tag,subflow)` 类型**保留给 V5 subflow striping**（同 `(source,tag)` 拆多条子流时用三元组区分），**不用于 output_lock**。
- D2D Link 支持固定 latency、`1 packet/cycle` 和功能性 FIFO；V1 测试将其配置为不饱和，不在本版赋予有限容量背压或网络死锁语义。
- 输出 flow 路径、端口和时间戳。

#### 本版不实现

- 多跳 die；
- 每方向多个端口；
- `port_width/link_bw > 1 packet/cycle`；
- striping、hash、hybrid 和 dynamic；
- 精细的跨流拥塞建模。

若 Behavioral D2D 尚未实现，应明确拒绝相应配置，不能绕过 D2D 或返回零延迟。

#### 必须通过的测试

1. 东、西、南、北四个方向的相邻 die 单流。
2. 1 包、2 包、buffer 边界前后和长消息。
3. REQUEST、ACK、DATA 均经过预期 link。
4. 接收包数、序号和 checksum 完全正确。
5. 无丢包、重复包和乱序。
6. `link_latency = 0/1/7/20` 扫描。
7. 分层验证 latency：独立单包 Link 的 delivery-capture 增量为 `ΔL cycle`；完整
   REQUEST→ACK→DATA 事务含三个因果串联跨链阶段，故
   `ΔT=3*ΔL*CYCLE`（当前 `CYCLE=2 ns`），不能误写成完整事务 `ΔT=ΔL`。
8. 反向 ACK 使用正确的反向 endpoint。
9. 仿真结束 Router buffer/lock 与 D2D data/control FIFO 直接统计归零；V1 无有限缓冲
   credit 状态，credit/背压归零测试从 V3 开始。
10. V0 的全部单 die 回归继续通过。

#### 完成标准

获得第一个可解释、可重复的相邻 die 端到端结果，并能证明控制和数据均真实经过 D2D Link。

### 4.3 V2：多跳路由与协议级活性版

#### 实现目标

支持任意规则 die mesh，保证包在中间 die 正确接力，并验证 REQUEST/ACK/DATA 握手在受支持的通信模式下能够持续推进。

本版只讨论协议级等待，例如双方都先发送再接收造成的 rendezvous 环形等待。有限 port/link buffer 的资源约束、背压和由此产生的 hold-and-wait 网络死锁尚未进入模型，因此不在 V2 声称或验证 buffer 死锁安全。

#### 当前进展（V2-a / V2-b / V2-b2 / V2-c / V2-d 全部完成）

- **V2-a ✔ 多跳路由纯函数**：源端只 pin 首跳，`exit_port` 只对当前 die 有效；`CrossDieStep` 现算
  `md` 并校验携带值（携带上一跳出口 → 抛错而非错误路由）。新增 `CrossDieIngressTile` 由
  `g_d2d_links` 精确 peer 给出落点。自测覆盖 3×1 `E,E` 与 2×2 对角 `E→N`（含反向）：
  hop 数 == `DieManhattan`、逐跳 die 距离严格 −1、片内距离严格下降、egress 序列；并直测生产
  包装 `SelectCoreMsgExit` 与 peer 映射的精确性/非法输入。
- **V2-b ✔ 中间 die 运行时转发**：`RepinOnC2CIngress` 在 C2C 入口重写 pinned exit（到目的 die
  清 −1，否则按入口 tile 重新 pin），数据/控制两路都接入，REQUEST/ACK/DATA 均中继；preflight
  逐跳要求双向 peer link。**同 port id 的巧合必须排除**：3×1 相邻两 die 的 E 出口是同一模板
  port id，送达成功无法区分「重新 pin」与「沿用旧值」，故以 `[D2D_REPIN] total/changed/same`
  计数为准，断言 `same>0` 且 `total == 跨 link 包数`。
- **V2-b2 ✔ HOST lane 缺口 + 方向变化证据**：修复潜伏的 V1 dataflow 缺口——权重下发
  `fill_queue_data` 仍用 `config.id / GRID_X` 索引 write_buffer，在 config 驱动 HOST
  （每 die lane 数 ≠ `GRID_Y`）下越界段错误（2×2 die3：`48/4=12` 而 `HOST_LANES=12`）；
  改走 `HostEnvelope + LegacyHostEnqueue`（`HostLaneOfCore` + 范围校验）。补 die3 本地回归
  （D2D 活动恒 0，与多跳解耦）与 2×2 对角 e2e。
- 参数 `allow_adjacent_d2d` 更名 `allow_d2d`（现放行任意已连通多跳路径）。
- **实测**：3×1 两跳 `typed=(2,2,2,2,8,8)`、`repin=(12,6,6)`；2×2 对角 `typed=(2,2,2,2,8,8)`、
  `repin=(12,12,0)`（方向真的改变 → `changed==total`、`same=0`）；两者均只有终点 DONE、
  ACK 源不含中间 die、router 与 link drain=0。自测 241/241、runner 51/51、NoC 冻结值不变。
- **V2-c ✔ 多跳端到端闭环**：新增逐条有向 link 归因 `[D2D_LINK]` 与每 die 活动 `[DIE_ACT]`
  （`router_pkts` 全部输入 / `mesh_pkts` 仅片内 router→router）。精确断言：DATA、REQUEST、ACK
  各自承载的有向 link 集合**恰好**等于期望路径（方向与每条包数精确、`in==out`）、每包 hop 数
  == 路径长度、入口重写数 == 总跨链包数、中间 die `mesh_pkts>0`（`router_pkts>0` 不足以排除
  零片内 hop）、只有终点 DONE 且 ACK 源恰为首尾、mismatch=0、drain=0。覆盖 3×1 正/反向与
  2×2 对角；对角**正反路径不对称**（维序两向都先走 X，往返成矩形）已固化。
- **V2-d ✔ 延迟标定与活性验收**：多跳 latency 律推广为 **`T(L)-T(0)=3*H*L*CYCLE`**（两跳实测
  增量 `0/12/84/240` == `3*2*L*2`，逐点精确）；**NoC 与 D2D 分离**——多出一跳的 NoC 开销
  `T2(0)-T1(0)=54 ns`（与 L 无关），而 `(T2(L)-T1(L))-54 = 0/6/42/120` 精确等于 `3*L*CYCLE`；
  latency 只平移固定延迟（各 L 下路径/包数/repin/mesh 不变）；watchdog 把超时哨兵判为失败
  （16 次扫描 0 超时）；多流两条 2 跳流共享同一对 link，每条计数恰为单流两倍、均 drain=0。
- **V2-d2 ✔ 仿真器内部协议 watchdog**：`ProtocolWatchdog` 维护「最后一次协议进展时间」，
  超阈值无进展即 dump `[PROTO_WAIT]`（wait_cycle / stalled_for / router+link residual /
  output lock / 队首 `(source,tag,dest,phase,wait_reason)`）并由 npusim 以**退出码 3** 主动结束；
  rendezvous 依赖环用例实测 `exit=3`、residual 全 0、判定为原语层等待；健康用例不误触发。
  Python `timeout=` 退居测试框架保险。
- 拓扑覆盖补齐：除 3×1 与 2×2 外，新增 **1×3 纵向两跳**（正向 `N,N`、反向 `S,S`）。
  逐 die 片内 hop 数改为**精确期望**（如 3×1/1×3 `[18,18,0]`、2×2 对角 `[52,15,3,9]`），
  不再只断言 >0；目的 die 为 0 是因入口 tile 恰为目的核本身（几何决定），已固化说明。
- **当前门 / V2 完成**：自测 241/241、Link 18/18、runner **62/62**、NoC 冻结值不变。
  V2 范围（多跳接力、per-die 重新 pin、精确路径证据、延迟标定、协议级活性）已闭合；
  **有限缓冲/带宽/背压与网络死锁安全属 V3，V2 不声称也未验证**。

#### 实现内容

- 支持 `DIE_X × DIE_Y`。
- 实现 die 级 XY 路由。
- 实现中间 die 的真实路径：

```text
D2D 入口 → 中间 die NoC → 下一 D2D 出口
```

- 每进入一个 die 时选择一次出口。
- 包头或 route context 保存当前 die 的选定出口或 ingress anchor。
- 显式 D2D peer mapping。
- REQUEST、ACK、DATA 全部支持多跳。
- 规定并测试受支持的握手调度，例如接收方先 post receive、奇发偶收等无协议依赖环模式。
- 引入进展 watchdog 和协议等待状态 dump，用于区分路由错误与 REQUEST/ACK 层环形等待。

#### 本版不实现

- 每方向多个候选端口；
- 动态端口选择；
- 多 lane 或多包每周期；
- 复杂的聚合带宽模型；
- 有限 port/link buffer 产生的背压；
- escape VC、whole-flow SAF 等网络 buffer 死锁规避机制。

#### 必须通过的测试

1. 3×1、1×3、2×2 die 多跳。
2. D2D hop 数等于 die 级 Manhattan 距离。
3. 每个 die hop 后剩余 die 距离减少 1。
4. 中间入口和出口不重合时，中间 NoC counters 必须增加。
5. 总片内 hop 数等于：

```text
源核→源出口
+ Σ(中间入口→中间出口)
+ 目的入口→目的核
```

6. 奇发偶收、接收方先 post receive 等受支持握手模式在多跳下完成。
7. 构造一个已知的协议依赖环，验证 watchdog 能将其报告为协议级等待，而不是路由丢包或网络 buffer 死锁；该用例是预期失败诊断测试。
8. watchdog 时间内所有合法 flow 完成。
9. 仿真结束所有网络状态排空和归零。
10. 连续多次运行的路径和周期一致。
11. 缺少中间方向端口时在启动阶段报错。
12. **tag=接收槽 语义（点 3，V1 审查修正；V1-c 已覆盖）**：`tag` 是接收端聚合槽，故 output_lock 按 tag 锁是正确的。已在 V1-c 验证：(a) **tag→dest 唯一性**——preflight 拒绝「同 tag 指向不同接收核」；(b) **many-to-one**——两源同 tag 发一个接收核（recv_cnt=2）**共享同一把锁**（`max_output_ref>=2`）、正确聚合、结束态归零；(c) **distinct-tag**——两条不同 tag 流经同一 C2C 端口各自独占锁（`max_output_ref==1`）、串行、数据不串。（原「同 tag 不同 source 应被视为不同 flow」的表述与「tag=接收槽 / 多发一」直接矛盾，已更正。）

#### 完成标准

多跳传输功能正确、路径可解释；合法协议调度能够完成，已知协议依赖环能够被 watchdog 正确诊断。本版不对有限缓冲网络死锁作出保证。

### 4.3b V3 实现进展（V3-a～V3-e 已完成）

- **V3-a 配置契约**：默认 `functional_v2` 与显式 `bounded_saf` 严格分区；whole-flow SAF、
  四类独立容量、整数有理数速率、`rate<=1` 与保守 inflight 窗口均在启动期校验。
- **V3-b 独立 Link**：有限 DATA/CTRL FIFO、token bucket、显式 pulse credit、BDP 边界和下游 stall
  在冻结 tag 由 Link SystemC self-test **32/32** 验证；该 BDP-1/BDP 扫描针对 standalone 模型。
- **V3-c 流大小与 admission**：REQUEST 携 `flow_packets`；`FlowKey(source,tag,subflow)` 原子预留；
  `F/F-1/BDP<F`、重复 key、并发守恒与未知释放均覆盖。
- **V3-d 生产接线**：Monitor 的 bounded gate 已移除；每条有向 link 实例化
  `SAF -> port limiter -> link limiter/inflight -> RX` 有限流水线。REQUEST 前原子预留整条多跳路径；
  DATA/CTRL 都用真实回程信用，结束时两类信用、SAF 预留、Router/Link residual 全归零。
  Post-freeze 新增 5 项直接 production probe：完整 flow gate、公式深度充分性、toggle credit 连续边沿、
  `SAF/inflight/RX=64/2/1` 背压和真实 reservation drain；当前 Link self-test **37/37**。
- **V3-e 定量与压力**：独立 runner **16/16** 覆盖三类瓶颈、共享/独立/full-duplex、源 die 与
  中间 die 混合拥塞、生产背压链、多跳、双向对角、2×2 四流置换、overbook 拒绝、RX/CTRL 小缓冲。
- **当前总门**：自测 **284/284**、Link self-test **37/37**（冻结 tag 为 32/32）、历史 runner **67/67**、
  V3 runner **16/16**、NoC 冻结值 **14781/29109、14833/45441**。

### 4.4 V3：周期精确拥塞与背压版

#### 实现目标

让 D2D Link 成为真实的有限速率、有限缓冲资源，并能够建模跨 die 流量与本地 NoC 流量之间的相互影响。

#### 实现内容

- 有限 port input/output buffer。
- D2D 在途队列和远端接收 buffer。
- 支持：

```text
port_width
link_bw
link_latency
buffer_depth
```

- 明确 `>1 packet/cycle` 的实现方式：多 lane、flit 聚合或 token bucket。
- 目的端拥塞能够逐级回压到 RX/inflight/SAF 边界；采用 whole-flow SAF 时，完整 flow 已落地后
  必须在该边界切断源 NoC 锁依赖，后续 flow 容量不足则在 REQUEST 注入前 admission 拒绝，
  **不**要求已完成整流的源 flow 继续持锁等待。
- 明确全双工、半双工以及正反向是否共享容量。
- 明确多个 endpoint 是否属于同一个物理 link group。
- 实现端口仲裁和公平性统计。
- 实现 stall 分类、buffer occupancy 和 link utilization。
- 在有限缓冲和背压进入模型的同一版本引入并验证网络 buffer 死锁规避机制。实现前必须固定采用的安全契约，不能把 whole-flow SAF 和 escape VC 当作无差别的实现选项：

  - **Whole-flow SAF**：开始向下一段注入前，必须为整条 flow 原子预留足够空间。若 flow 含 `F` 个网络传输单元，则可用/可预留容量必须至少为 `F`；存在多个并发预留时还要计算预留总量。`buffer_depth ≥ BDP` 只关系到流水利用率，不能替代 `buffer_depth ≥ F` 的 whole-flow SAF 正确性条件。
  - **Chunked SAF**：若不准备容纳完整 flow，必须明确最大 `chunk_packets`，每个 chunk 在注入前原子预留，且 `chunk_packets ≤ reserved_buffer_depth`；前一 chunk 释放其上游资源后，下一 chunk 才能建立新的依赖。
  - **Escape VC**：不要求 buffer 容纳整条 flow，但必须给出无环的 channel dependency/VC 转换规则，并保证 escape VC 至少具备实现所需的 buffer/credit。

- SAF 模式发现容量不足时，必须在任何部分注入和上游资源占用之前执行以下显式行为之一：拒绝配置/传输、按已配置的 chunk 大小拆分，或切换到明确启用且已经验证的 escape VC。禁止静默退化为部分注入后等待空间。
- 网络死锁自由必须同时具备设计层的结构性论证（SAF 的原子容量预留不变量，或 escape VC 的无环 channel dependency graph）。压力测试用于验证实现是否遵守该论证，不能单独证明不存在所有可能的死锁。
- **V3 瓶颈测试需要解析基线（点 4）**：瓶颈隔离验收「goodput 接近理论最小带宽」依赖一个理论参考值，但完整的 `oracle.py` 排在 V4。故 V3 须先提供一个**最小解析基线**——即瓶颈段 `min(noc, port, link)` 速率的内联/手算值——用于 goodput 对照；V4 的 `oracle.py` 再将其一般化为含多跳路径的完整模型。V3 不得因缺 oracle 而跳过瓶颈精度验收。

#### 必须通过的测试

##### 瓶颈隔离

| 场景 | NoC | Port | D2D | 预期瓶颈 |
| --- | ---: | ---: | ---: | --- |
| A | 1 | 4 | 4 | NoC |
| B | 4 | 1 | 4 | Port |
| C | 4 | 4 | 1 | D2D Link |

验收：

- 长消息 goodput 接近理论最小带宽，建议误差小于 1%；
- 对应瓶颈利用率接近 100%；
- 增大非瓶颈参数不改变吞吐；
- 增大当前瓶颈后，瓶颈转移到下一段。

##### D2D Link 争用

- 单流基线；
- 两流共享同一 link；
- 两流使用独立 link。

验收：共享 link 的总吞吐不超过物理容量；独立 link 能够并行；同优先级流量不会永久饥饿。

##### Local 与 D2D 混合流量

运行以下四组：

1. Local flow 单独运行；
2. D2D flow 单独运行；
3. Local+D2D 同时运行，片内路径不相交；
4. Local+D2D 同时运行，共享源 die 或中间 die 的片内链路。

验收：

- 不相交场景只有很小的额外 queue delay；
- 共享场景的 local latency 或 D2D latency 明显增加；
- 被共享 NoC link 的 busy、queue 和 stall counters 增加；
- 未共享链路不能产生同等程度的拥塞。

##### 背压

设置 `NoC bw > port bw > link bw`、小 buffer 和长消息。

验收：

- D2D buffer 达到满状态；
- stall 传播到 RX、inflight 和 SAF drain；若采用 wormhole/VC 才继续传播到源 port/NoC/核；
  whole-flow SAF 则应在完整整流后切断源侧依赖，容量不足的新 flow 在注入前拒绝；
- 无丢包且最终排空；
- 增大 buffer 改变瞬态行为，但不应改变长时间稳态瓶颈吞吐。

##### 有限缓冲网络死锁安全

所有实现都需要在其安全契约允许的最小合法 buffer/credit 下运行多方向、hotspot、对角和随机置换流量，并检查网络最终排空。对于 `buffer_depth=1/2`，若 SAF admission 条件不满足，预期结果应是首包注入前明确拒绝，而不是强行要求该 flow 完成。除此之外，根据采用的机制执行对应测试：

**Whole-flow SAF：**

1. 对大小为 `F` 的 flow，`buffer_depth/reserved_capacity = F` 时能够完成。
2. `buffer_depth/reserved_capacity = F-1` 时，在首包注入前明确拒绝或走配置指定的安全 fallback，不能进入部分注入后永久等待。
3. 构造 `BDP < F` 且 `buffer_depth = BDP` 的用例，确认其不能被误判为满足 whole-flow SAF 容量条件。
4. 多 flow 并发时检查预留容量守恒，不允许超额预留或两个 flow 同时占用同一份容量。
5. flow 大小只能在运行时确定时，容量不足必须产生可诊断的 admission 结果，而不是仿真超时。

**Chunked SAF：**

1. `chunk_packets = buffer_depth` 的边界用例能够完成。
2. `chunk_packets > buffer_depth` 时启动或 admission 阶段报错。
3. 检查未完成预留前没有任何 chunk 的部分包进入下游。
4. 多个 chunk 之间不保留能够形成跨 chunk 依赖环的上游资源。

**Escape VC：**

1. 使用最小合法 escape buffer/credit 运行多方向循环压力流量。
2. 检查阻塞流量最终进入 escape VC 并持续取得进展。
3. 检查 VC 转换符合规定顺序，禁止从逃逸资源返回可能重新形成依赖环的普通 VC。
4. 流量完成后所有 VC、credit、buffer 和 output lock 归零。

#### 已完成的实现与验收（V3-d / V3-e）

- 生产 DATA 路径：`whole-flow SAF -> port token -> link token/inflight -> finite RX`；CTRL 独立有限 FIFO。
- DATA SAF 槽位与 CTRL FIFO 都由显式回程信用保护，信用 event 使用翻转位避免连续归还合并；
  每个成功用例要求 `data_balanced=ctrl_balanced=1`。
- 128 包长流 goodput：NoC/port/link 瓶颈实测约 `1 / 0.501976 / 0.250493`，相对理论值误差 <1%；
  非瓶颈变化不改吞吐，stall 分类指向实际限制器。
- 源 die shared/disjoint：NoC stall `11/0`；中间 die shared/disjoint：stall `7/0`，D2D 完成
  cycle `397/380`，证明中间 die 混合拥塞真实发生。
- 生产背压热点（`rx=1,inflight=4`）：`inflight_full=85、rx_full=95、inflight_stall=60、
  rx_stall=63、downstream_stall=63`，最终全部排空。`source_stalls=0` 符合 whole-flow SAF
  在完整整流后切断源锁依赖的安全设计。
- 最小合法安全压力：`SAF=F=4、inflight=BDP=4、rx=1、ctrl=1` 的 2×2 四流固定置换覆盖
  八条有向 link，均完成；`SAF=F-1` 与 concurrent overbook 在首包前明确拒绝。
- 2×2 双向对角覆盖 E/N/W/S；3×1 两跳证明每个中间 SAF stage 都完整存 flow 后再排空。

#### 完成标准

能够定量回答“瓶颈在哪里”“跨 die 流量是否使本地或中间 die NoC 变慢”“背压是否真实传播”，并且有限缓冲下采用的 SAF/escape VC 安全契约已通过对应边界和压力测试。

### 4.5 V4：Behavioral 模型与定量校准版

#### 实现目标

在周期精确模型稳定后，实现快速、无跨流争用但具备正确距离、延迟和带宽语义的 Behavioral D2D 模型。

#### 实现内容

- Behavioral D2D 延迟公式。
- 正确计入：

```text
源 die 片内 hops
所有中间 die 片内 hops
目的 die 片内 hops
D2D hop 数 × link latency
序列化时间
```

- 明确多跳链路是流水还是逐段 store-and-forward。
- 避免 router hop latency 和 `wait()` 重复记账。
- 支持多跳路径的最小瓶颈带宽。
- 明确 Behavioral 不建模跨 flow 的端口和链路争用。
- 建立 `oracle.py`，根据 topology、路径和消息大小计算理论时间。

#### 必须通过的测试

1. 单包 fixed latency 扫描。
2. 消息大小扫描：`1/2/B-1/B/B+1/100/1000` 包。
3. NoC、port、D2D 三类带宽扫描。
4. 单跳与多跳理论公式测试。
5. Behavioral 与 oracle 完全一致或误差不超过 1 cycle。
6. 无争用长消息下，Behavioral 和 cycle 的 goodput 接近。
7. shared/disjoint 对照：

```text
Behavioral：结果基本相同
Cycle：shared 明显慢于 disjoint
```

8. 增加 `link_latency` 只改变固定/首包延迟，不改变稳态吞吐。
9. 改变 `link_bw` 只改变序列化时间和稳态吞吐，不被重复计算。

#### 完成标准

Behavioral 与 cycle 模型的适用范围清晰；无争用时能够通过统一 oracle 相互校准，有争用时差异符合模型定义。

### 4.6 V5：多端口、条带化和高级策略版

#### 实现目标

支持多个物理端口、共享或独立 D2D Link，以及利用多端口聚合带宽的高级流量策略。

#### 实现内容

- 每方向多个 C2C 端口。
- 显式 link group，区分：

```text
多个独立物理 D2D Link
多个端口共享同一个物理 D2D Link
```

- 静态选择策略：

```text
nearest
banded_nearest
tag_hash
hybrid
```

- 严格 per-flow pinning。
- tensor striping 和 subflow。
- subflow 重组、统一完成事件和非整除分片。
- dynamic 最少负载策略作为最后引入的实验能力。
- 固定 seed 和可复现机制。

#### 必须通过的测试

1. 非 stripe 单流不超过单端口带宽。
2. `stripe=2/4` 时 subflow 数、包数和 checksum 正确。
3. `N_pkt % stripe != 0` 时尾部分配正确。
4. 所有 subflow 完成后只报告一次逻辑传输完成。
5. 独立端口的聚合带宽随 stripe 增加。
6. 遇到源注入或共享 NoC cut 后，继续增加 stripe 不再错误提速。
7. 共享 link group 的总吞吐不超过物理 link 容量。
8. 同一 flow 的所有包在每个 die 内使用固定出口。
9. 同一 `(source, tag)` 被 striping 拆成的多条 subflow 用 subflow 号区分、正确重组，不被误判为乱序/重复；（注：`tag` 本身是接收端聚合槽，同 tag 多源发一个接收核属「多发一」聚合，见 V1 审查修正与 T22。）
10. static/hash 在固定 seed 下完全可复现。
11. dynamic 不产生乱序、锁泄漏或死锁。
12. 多流公平性达到预定阈值，或测试明确记录固定优先级仲裁的预期偏差。
13. 1/2/4/8 dies 规模下仿真器运行时间和内存增长可接受。

#### 完成标准

多端口性能收益来自真实独立物理资源，并受源注入、共享 NoC cut、link group 和目的端接收能力约束，而不是简单乘以端口数量。

## 5. 独立于版本的测试层级

版本定义“什么时候实现什么”，测试层级定义“从哪些角度证明实现可信”。每一版应选取与其功能相符的层级，不应只写端到端 workload。

### 5.1 L0：纯函数和静态配置测试

覆盖：

- 地址和坐标转换；
- 消息编码；
- die 级第一跳；
- port/peer 查找；
- flow key；
- 带宽和延迟的整数取整；
- 配置合法性与错误信息。

这一级应给出精确结果，不能只判断“没有崩溃”。

### 5.2 L1：单 die 回归测试

覆盖：

- 旧配置兼容；
- 核间通信路径不变；
- 原 NoC 拥塞趋势不变；
- D2D 模块不活动；
- trace 开关不改变仿真结果。

这是所有版本共同的第一道回归门槛。

### 5.3 L2：端到端功能测试

覆盖：

- REQUEST/ACK/DATA 完整闭环；
- 四方向相邻 die；
- 单跳、多跳和中间 die；
- 单向、双向；
- 数据完整性、顺序和最终网络排空。

### 5.4 L3：无争用时序与吞吐测试

覆盖：

- 首包固定延迟；
- 长消息稳态吞吐；
- 消息大小、latency、bandwidth 扫描；
- 单跳、多跳；
- cycle、Behavioral 和 analytical oracle 对照。

对无争用串联流水线，应分别验证：

```text
T_first = 固定路由延迟 + D2D 固定延迟 + 首包服务时间
T_last  = T_first + 后续包的稳态服务时间
```

实际公式必须以最终确定的 pipeline/service contract 为准。

### 5.5 L4：拥塞与混合流量测试

覆盖：

- 同一端口多流；
- 同一物理 link 多流；
- 独立端口/link 对照；
- D2D 与源 die 本地流量；
- D2D 与中间 die 本地流量；
- 多入口汇入目的 die；
- shared/disjoint 路径对照。

测试场景间应保持计算量、消息数、接收端负载和尽可能相同的跳数，只改变是否共享目标资源。

### 5.6 L5：背压、死锁与活性测试

这一层分两个阶段：V2 验证协议级活性和 watchdog 诊断；V3 在有限缓冲、背压和资源持有真实存在后，验证网络 buffer 死锁安全。不得用 V2 的无限/非饱和 buffer 结果替代 V3 的有限资源压力测试。

V2 协议级覆盖：

- 接收方先 post receive；
- 奇发偶收等无协议依赖环模式；
- 已知 REQUEST/ACK 环形等待的预期失败诊断；
- watchdog 能区分协议等待、路由不可达和无进展。

V3 网络级覆盖：

- buffer depth=1；
- 低带宽 D2D Link；
- 多方向并发；
- hotspot、对角、随机置换；
- REQUEST/ACK/DATA 交错；
- whole-flow/chunked SAF 的容量 admission 边界，或 VC/escape path 的最小 buffer/credit；
- SAF 容量不足时禁止部分注入后等待；
- watchdog 和死锁现场输出。

超时时至少输出：

```text
所有非空 buffer
所有未释放 output lock/VC/credit
所有未完成 flow
每条 flow 的最后进展时间
```

### 5.7 L6：模型一致性、确定性与规模测试

覆盖：

- Behavioral 和 cycle 无争用结果校准；
- cycle 在拥塞下表现出额外 queue delay；
- 静态策略重复运行结果一致；
- dynamic 在固定 seed 下可复现；
- 1/2/4/8 dies 的运行时间、内存和 trace 大小；
- 大规模结束时的资源清理。

## 6. 核心测试用例矩阵

| ID | 测试名称 | 最早版本 | 主要目的 | 核心验收标准 |
| --- | --- | --- | --- | --- |
| T00 | 单 die 精确回归 | V0 | 旧功能兼容 | 输出、路径和时序不变，D2D 计数为 0 |
| T01 | 地址/消息往返 | V0 | 基础数据正确 | 所有边界值逐字段一致 |
| T02 | 非法 topology/config | V0 | 防止运行时挂死 | 启动阶段明确报错 |
| T03 | 四方向相邻 die 单流 | V1 | 基本 D2D 功能 | REQ/ACK/DATA 正确闭环 |
| T04 | 单跳/多跳 latency 扫描 | V1/V2 | 固定延迟准确性 + 归因 | V1：Link `Δdelivery=ΔL cycle`、事务 `ΔT=3ΔL·CYCLE`；**V2-d**：多跳 `ΔT=3·H·ΔL·CYCLE`（两跳实测精确），且分离出「每多一跳的 NoC 开销 54 ns（与 L 无关）+ D2D 部分 3·L·CYCLE」 |
| T05 | 消息大小边界 | V1 | 包化与尾包正确 | 1/2/5/7/8/9/32 包的 seq/hash/checksum/尾长正确，完成时间单调 |
| T06 | 3-die 多跳 | V2 | 中间 die 接力 | **V2-b/V2-c 已覆盖**：3×1 正向 E,E 与反向 W,W；承载 DATA/REQUEST/ACK 的有向 link 集合恰好等于期望路径且每条 in==out；每包 hop 数==2；repin total==跨链包数且 same>0（排除同 port id 巧合）；中间 die mesh_pkts=18>0（真片内 hop）；仅终点 DONE、drain=0 |
| T07 | 2×2 对角路径 | V2 | die 级 XY | **V2-b2/V2-c 已覆盖**：正向 die0-(E)->die1-(N)->die3，ACK die3-(W)->die2-(S)->die0——**正反不对称**（维序两向都先走 X，往返成矩形）已固化为期望；repin=(12,12,0)（每次入口重写都改方向）；mesh_pkts=[52,15,3,9] 示 die1 承载正向、die2 只承载 ACK；路径收敛无绕圈、仅 core48 DONE、drain=0 |
| T08 | 多跳协议活性与诊断 | V2 | 握手调度和 watchdog | **V2-d/V2-d2 已完整覆盖**：合法模式——latency 扫描 16 次 0 超时、多流两条 2 跳流均完成 drain=0；**已知协议环**——`cross_die_rendezvous_cycle.json` 由**仿真器内部** `ProtocolWatchdog` 主动诊断，dump `protocol_wait_cycle`/`stalled_for`/residual/output lock/`(source,tag,dest,phase,wait_reason)` 并以**退出码 3** 结束（非框架 124 超时），且明确判定「等待在原语/rendezvous 层而非网络层」；健康用例不误触发 |
| T09 | 三类瓶颈隔离 | V3 | 带宽准确性 | **已覆盖**：128 包 goodput 在 1% 内等于 `min(NoC,port,link)`，非瓶颈不敏感，stall 精确归因 |
| T10 | 同 link/独立 link | V3 | D2D 争用 | **已覆盖**：共享总量守恒且两 flow 完成；独立 link 并行；双向容量独立 |
| T11 | Local+D2D shared/disjoint | V3 | 混合 NoC 拥塞 | **已覆盖**：源 die shared/disjoint stall=11/0，D2D completion 增加 |
| T12 | 中间 die 混合拥塞 | V3 | 诱发中间 NoC 流量 | **已覆盖**：shared/disjoint stall=7/0、D2D cycle=397/380 |
| T13 | 小 buffer 背压 | V3 | 有限流水线回压 | **已覆盖**：remote NoC→RX→inflight→SAF stall 全触发并排空；whole-flow SAF 按设计切断源锁依赖 |
| T14 | Behavioral oracle | V4 | 快速模型准确性 | 理论值误差不超过 1 cycle |
| T15 | Behavioral/cycle 对照 | V4 | 两档语义一致 | 无争用接近；仅 cycle 响应争用 |
| T16 | Striping 非整除 | V5 | 子流正确性 | 数据守恒、一次逻辑完成 |
| T17 | 多端口聚合与共享 cut | V5 | 防止虚假加速 | 聚合吞吐不超过所有共享瓶颈 |
| T18 | Dynamic 可复现与活性 | V5 | 高级策略安全 | 固定 seed 可复现、无乱序和死锁 |
| T19 | SAF 容量 admission 边界（采用 SAF 时） | V3 | 防止 SAF 保证静默失效 | **已覆盖**：`F` 完成、`F-1` 注入前拒绝、later-hop 失败原子回滚、并发预留守恒 |
| T20 | Escape VC 有限缓冲压力（采用 escape VC 时） | V3 | 验证网络 buffer 死锁安全 | 最小合法 buffer/credit 下持续进展并最终排空 |
| T21 | HOST 可达性与 endpoint 解码 | V0 | 统一端口框架校验（点 5） | 无 HOST 端口/全被 C2C override 时启动报错；`port_for_host` 每核有定义 |
| T22 | tag=接收槽 语义 | V1 | tag→dest 唯一 + 多发一聚合（点 3，审查修正） | tag→dest 唯一性拒绝撞槽；同 tag 多源共享锁聚合（maxref≥2）；distinct-tag 串行不串 |

## 7. 测试目录建议

沿用现有 `llm/test/noc_congestion` 的配置驱动和运行器组织方式：

```text
llm/test/d2d_link/
├── hardware/
│   ├── die_1x1.json
│   ├── die_2x1.json
│   ├── die_1x2.json
│   ├── die_3x1.json
│   └── die_2x2.json
├── sim/
│   ├── sim_cycle.json
│   └── sim_beha.json
├── workload/
│   ├── smoke_single_flow.json
│   ├── multihop.json
│   ├── bidirectional.json
│   ├── mixed_disjoint.json
│   ├── mixed_shared.json
│   ├── same_port.json
│   ├── different_ports.json
│   └── deadlock_stress.json
├── invalid_config/
├── golden/
├── gen_workloads.py
├── oracle.py
├── run_test_d2d_link.py
└── README.md
```

运行器建议输出机器可读 CSV/JSON，而不只输出最终总时间：

```text
case
model
packets/bytes
d2d_hops/noc_hops
first_packet_latency
completion_latency
goodput
link_util
max_queue
stall_cycles
completed
```

## 8. 测试分组与执行频率

### Smoke

- 地址、序列化、配置校验；
- 单 die 回归；
- 相邻 die 单流。

每次提交执行，目标是数秒到数十秒完成。

### Regression

- 四方向；
- 多跳；
- latency/bandwidth 边界；
- REQUEST/ACK/DATA 完整闭环；
- Behavioral oracle。

每个 PR 必须执行。

### Congestion

- 同 link/独立 link；
- shared/disjoint；
- local+D2D；
- 中间 die 拥塞；
- 背压。

合并前或模型相关 PR 执行。

### Stress

- buffer depth=1；
- 多方向随机流量；
- hotspot/all-to-all；
- SAF 容量不足和并发预留；
- escape VC 最小 buffer/credit；
- dynamic；
- 大规模 die mesh。

建议 nightly 执行，并保存失败时的网络状态 dump。

### Accuracy Sweep

- 消息大小；
- link latency；
- NoC/port/link bandwidth；
- stripe 数；
- die hop 数。

模型公式、带宽实现或 pipeline 时序改变时执行。

## 9. 每一版共同的发布门槛

每个版本交付前都必须满足：

1. 上一版本全部测试继续通过。
2. 新功能同时具有正向、边界和非法配置测试。
3. 包/字节发送数与接收数守恒。
4. 所有 flow 均有完整时间戳和路径记录。
5. 仿真结束时网络排空且所有 lock/credit/VC 归零。
6. 不支持的配置明确报错。
7. 静态模式可复现；随机和动态模式支持固定 seed。
8. trace 开关不能改变仿真时序和结果。
9. README 明确记录本版支持项、限制和参数单位。
10. 测试运行器给出机器可读汇总和清晰的失败原因。

## 10. 第一阶段最小准入集合

如果先交付 V0～V3，至少应将以下测试设为阻塞性准入测试：

1. 单 die 精确回归；
2. 地址、消息和配置校验；
3. 四方向相邻 die 单流；
4. REQUEST/ACK/DATA 端到端及数据完整性；
5. 3-die 多跳和中间 NoC 计数；
6. latency 和消息大小扫描；
7. NoC/port/D2D 三类瓶颈隔离；
8. 同 link 与独立 link 争用对照；
9. Local+D2D 的 shared/disjoint 对照；
10. 小 buffer 背压传播；
11. V2 合法握手模式完成，已知协议依赖环能够被 watchdog 正确诊断；
12. V3 有限缓冲、多方向压力测试无网络 buffer 死锁；
13. 若采用 whole-flow/chunked SAF，容量等于和小于 flow/chunk 的边界测试必须通过；若采用 escape VC，最小合法 buffer/credit 压力测试必须通过；
14. 所有非法端口、link 和 SAF 容量配置启动或 admission 失败。

这组测试分别回答：D2D 能否工作、包是否真实经过各层网络、路径和延迟是否正确、拥塞是否由预期资源引起、背压是否能够传播，以及模型在压力下是否仍能完成。

#### V4 开发进展

- **V4-a 已完成**：新增 `backend=cycle|behavioral` 契约，默认 cycle；Behavioral 的
  `port_rate/link_rate/link_latency` 显式必填，并拒绝 cycle mode、SAF/depth 与 legacy
  字段，避免静默忽略。
- V4-a 仅建立接缝，生产 Behavioral 尚未接线；合法配置会明确报 `runtime is not wired yet`。
- V4-a 验证：纯函数 **293/293**、Link **37/37**、历史 D2D **67/67**、V3 production
  **16/16**，NoC 冻结值 **14781/29109、14833/45441**。

下一步 V4-b：独立 `oracle.py` 与 C++ 纯函数 estimate，先固定 min-cut、延迟唯一记账点
和多跳 forwarding contract，再进入运行时。

- **V4-b 已完成**：C++ 纯函数与独立 Python oracle 均实现 X-first 单/多跳路径、
  `R=min(1,port,link)`、`S(F)=ceil(F/R)` 和 `T_D2D=3*H*L+S(F)`。
- 多跳明确采用 pipelined min-cut；bulk service 只算一次。`intra_die_hops` 只作路径解释，
  因代表包仍穿 Router 而不再写入 estimate。
- 覆盖 2×1、3×1、2×2 diagonal、missing peer、同 die、非法 packet；C++ 总门
  **300/300**、Python **8/8**。

下一步 V4-c：让 REQUEST/ACK/DATA 代表包进入 Behavioral runtime，并使其可观测值逐项匹配 oracle。

- **V4-c 已完成**：生产 D2DLinkUnit 增加独立 Behavioral dispatch；wire 上每 flow 保留
  REQUEST/ACK/DATA 各一个代表消息，DATA 的逻辑 `F` 不丢失，bulk 只在第一条 link 计一次。
- `[D2D_BEHA]` 与独立 oracle 对 `data_flows/logical packets/service/fixed/total` 逐项比较。
  一跳 `F=4,L=7,R=1` 两边均为 `service=4,fixed=21,total=25`，两次运行均 `286 ns`；
  代表消息计数、repin、drain 和错误 backend/NoC 组合拒绝均已覆盖（V4 **4/4**）。
- 自审抓出并修复 Behavioral dispatch 误嵌套在 bounded 分支导致的静默退化；完整 V3/NoC
  冻结门保持精确全绿。

下一步 V4-d：多跳、消息大小、latency/rate 扫描，逐项校验 oracle 斜率与边界。
