# D2D Link 测试 — V0（多 die 基础设施重构版）

对应 `notes/extensions/D2D_link/D2D_link_test.md` 的 V0，扩展逻辑依据
`notes/extensions/D2D_link/D2D通信建模计划.md`。

## 运行

```bash
python3 llm/test/d2d_link/run_test_d2d_v0.py     # 需先 build 出 build/npusim
# 或单独跑纯函数自测：
cd build && ./npusim --d2d-v0-selftest
```

## 当前状态（逐增量推进）

- **V0a 完成**：基础设施 + 纯函数 + 配置校验 + 统计接口。
- **V0b 已完成增量**：V0b-1（编址/消息语义收口）、V0b-3（live torus 拆除）、前置 hardening、
  **V0b-2A（运行时多 die 模块实例化——真正建 `TOTAL_CORES` 个 router/worker，die>0 存在但空闲）**、
  V0b-4（D2D peer/link 结构级构造+校验）、**V0b-2C0（可靠跨 die 拒绝，raw-JSON preflight）**、
  **V0b-2C1（global-id 归一化 + 结构校验，构造级）**、**V0b-2B0（`HostEnvelope` + LegacyHostEnqueue，
  解耦 config_helper 与物理 HOST lane）**、**V0b-2B1（per-die HOST lane，workload 真正在 die>0 运行）**。
  均**单 die sim-time / 核集合 / 计数精确一致**。
- **✔ 里程碑：workload 已能在 die>0 运行且多 die 并行**——
  - **T3**：die1 平移 workload（ids+16, `id_space:global`）→ 16 个 die1 核执行、HOST1 收齐 DONE、
    sim-time=29109、D2D=0。
  - **T2**：die0+die1 合并 workload → 两 die 各 16 核并行执行、sim-time=29109（**零干扰**）、D2D=0。
- **V1-pre / V0-exit ✔ 全部完成**：非西侧 HOST 路由**不再算 polish**，是 V1 的前置里程碑（V1 要测
  E/W/N/S 四方向 D2D，西边缘可能同时有 HOST 与 C2C，必须由配置驱动、二者不冲突）。分增量推进：
  - **Inc 1 ✔**：HOST 挂载表 `g_host_attach` 接缝——routing/enqueue/binding 统一查表；仅 legacy
    构造器（西边缘每全局行一 lane），返回值与旧硬编码相等，sim-time 精确一致。
  - **Inc 2a ✔**：补全 `core_lane` 全映射、`HostLaneOfCore` 查表、`HOST_LANES ← n_lanes` 强一致、
    独立 `ValidateHostAttach()`、入队前 lane 校验（杜绝 `q[-1]`）；结构性自测。
  - **Inc 2b ✔**：`GetNextHop/GetNextHopReverse` 朝挂载表 tile 路由（legacy=本行西边缘）；
    显式拒绝跨 die HOST（`DieOfHostEndpoint(des)!=DieOfGlobal(pos)` 抛错，不静默发 HOST）。
  - **Inc 3a ✔**：config 驱动挂载表构造器（`role=HOST` 端口建 lane/tile/core_lane，同构 die 模板）
    + 同物理端口 host/c2c 冲突拒绝；W-host 配置产出表 == legacy（e2e 仍 29109）。**不改路由**。
  - **Inc 3b-1 ✔**：RouterUnit HOST 接口 `IsMarginCore`→`IsHostAttachTile`（Monitor 才能按表绑定
    非西侧 tile 而不解引用空指针）+ `ValidateHostAttach` 拒绝重复 `lane_tile`（单 tile 单 HOST 接口）。
  - **Inc 3b-2a ✔**：`GetNextHop` 加显式 `anchor_core`，HOST 挂载 tile 取自消息 `source_`（非 `pos`）
    + router 传 `m.source_` + divergence 单元测试证明用固定 source。
  - **Inc 3b-2b ✔**：S-host 单 die 端到端——`[HOSTLANE]` mismatch=0（DONE/ACK 到达正确 lane）。
    非西侧 HOST 运行时闭环达成。
  - **Inc 3c ✔**：W/S/E/N 四方向 HOST 端到端(**W 也连续两次**),**方阵 4×4 + 矩形 4×2 die 内 mesh**
    都覆盖。每方向两次都校验 mismatch=0、`DONE(source)`+`ACK(source,tag)` **严格签名**与 west 一致、
    **ACK 结构**(每 source 1 个 CONFIG ACK@tag0 + 1 个 WEIGHT ACK@tag=id)、预期 per-lane(读
    `GRID_X/GRID_Y`:4×4 行=`2,2,2,2`/列=`4,0,4,0`;4×2 行=`2,2`/列=`2,0,2,0`)、D2D=0、无绑定错误、
    `GRID_X*GRID_Y` 核;sim-time 两次一致、差异符合各自 hop 变化。
  - **CONFIG ACK tag UB 修复**:`Recv_prim(RECV_CONF)` 曾留 `tag_id` 未初始化,CONFIG ACK 读到非确定值
    (4×2 暴露 30535/28719...,4×4 恰为 0 只是内存布局巧合)。此前它**不影响当前 HOST lane 选择**(路由
    anchor 用 `source_`),但属 UB 且 V1 flow-key 会用 tag,不能带入 V1。已修:`Recv_prim` 全字段类内
    默认值 + 显式 CONFIG ACK tag 契约=0。现 4×2/4×4 均**严格 (source,tag)** 比对、多次运行一致。
- **V0 剩余（纯 polish，不影响功能）**：ASan/UBSan 干净性、WorkerCore 完整 teardown、
  pd/gpu 等非 dataflow 模式的多 die 化（V0 不要求）。cache/DUMMY 已随默认构建的多 die 用例覆盖。

## V1 开发进展（feat/d2d-v1）

- **V1-c0 ✔**：`FlowKey(source_global_id,tag,subflow)` 与消息内 16-bit
  `exit_port_`；`0=unpinned`、有效端口编码为 `port_id+1`，现有未 pin 消息的保留位仍为零。
- **V1-c1a ✔（能力就绪、生产 gate 保持关闭）**：preflight 可精确验证相邻 die 的实际双向
  `g_d2d_links`，拒绝无 link 与多跳；但 `ValidateWorkloadStructure` 的生产调用默认
  `allow_adjacent_d2d=false`，在 c3 的 REQUEST/ACK/DATA 全链闭环前仍启动期拒绝，避免接受后挂死。
- **V1-c1b ✔（控制路由接线）**：串行/并行 REQUEST 与接收端返回 ACK 均在源核调用
  `PinControlMsgExit`，相邻跨 die 时只选一次出口并随包携带；Router 控制路径统一调用
  `ControlMsgNextHop`，在源 die 朝固定端口收敛，过 link 后在目标 die 退回片内 XY。
  当前证据为双向 REQUEST/ACK 逐跳 walk + 序列化 + Router 生产调用点；生产 workload 仍由 c1a gate
  拒绝，故尚不声称真实 workload 控制包已经运行时穿链。
- **当前验证**：纯函数/路由自测 **209/209**、Link SystemC 自测 **18/18**、D2D runner
  **27/27**（含“有 peer link 但 c3 前仍拒绝”的真实启动负例）、NoC 冻结值
  **14781/29109、14833/45441**。
- **下一步**：V1-c2 接 DATA 源端 pin 与数据路由；V1-c3 再显式打开生产 preflight，并增加第一条
  REQUEST → ACK → DATA 的真实跨 die 端到端用例。

## V0 基线冻结（V0-exit / Inc 4）—— 进 V1 前的准入门

**一键跑三道门（自动阻塞、汇总退出码）**：`python3 llm/test/run_v0_exit.py`（任一门非零即整体非零）。
各门也可单独运行——每个脚本自身即断言门（数值漂移 / 进程非零 / 结束时间缺失均 `sys.exit(1)`）：

1. **纯函数自测 165/165**：`cd build && ./npusim --d2d-v0-selftest`（编址/端点/拓扑/端口校验、HOST 挂载表
   结构、路由收敛、跨 die 拒绝、config 挂载表、egress anchor）。（也作为门 2 的第 1 组）
2. **D2D 端到端 23/23**：`python3 llm/test/d2d_link/run_test_d2d_v0.py`——含单 die 回归(29109)、多 die
   实例化(2×1/1×2/2×2)、跨 die dest 拒绝、四方向 HOST e2e(**方阵 4×4 + 矩形 4×2**，mismatch=0 +
   `(source,tag)` 严格签名 + ACK 结构 + per-lane 分布 + 两次确定)。
3. **NoC 四场景精确数值**：`python3 llm/test/noc_congestion/run_test_noc_congestion.py`（**内置期望字典 +
   `sys.exit(1)`**）→ no_congestion **beha 14781 / cycle 29109**、congestion **beha 14833 / cycle 45441**。

**负例分两类**（准确区分证据形态）：
- **进程级启动负例**（独立 npusim 进程、非零退出，在 D2D runner 内）：非法 die_ports(缺方向)、非法 c2c
  参数(link_bw/latency/buffer_depth)、跨 die dest。
- **纯函数负例**（`--d2d-v0-selftest` 内的异常断言，非独立进程退出）：HOST 不可达、重复 (side,idx)
  override、同 tile 重复挂载(S+W 角点)、缺 egress anchor、越界/非法映射、跨 die HOST。
  （后续可补对应的进程级集成用例，把「纯函数负例」提升为独立启动负例。）

**冻结范围（V0 支持并保证）**：**dataflow 模式** + **配置驱动 HOST attachment**（`role=HOST` 端口驱动、
W/S/E/N 四方向、方阵 + 矩形 die 内 mesh、per-message egress anchor）、多 die 模块实例化与 die>0 运行、
跨 die 流量的启动期拒绝、单 die sim-time 精确一致。

**非阻塞后续项（不在 V0 冻结范围）**：ASan/UBSan 干净性、WorkerCore 完整 teardown、PD/GPU/PDS 非 dataflow
模式的多 die 化、ACK 逐消息唯一序号(`ack_seq`)、mem-role at-edge 真实内存流量、非默认编译组合。

**版本标识（已冻结）**：V0 已固化为 tag **`d2d-v0-baseline`**，指向 freeze 提交
**`b6334397e43f704739692fcdc9e9ca450a469320`**（`test(d2d): freeze V0 regression baseline`，其父
`d260b60` = `feat(d2d): add V0 multi-die and config-driven HOST support`，上游 `7940afa`）。
该提交在**干净工作树**上 `python3 llm/test/run_v0_exit.py` 全绿（165/165 + 23/23 + NoC 4/4）。
随时可 `git checkout d2d-v0-baseline` 精确恢复到「V0 完成、全绿」的基线，用于 V1 回归对照。
（本 SHA 说明由紧随 tag 之后的文档提交回填，故 tag 指向的提交里本节仍为占位——提交无法包含自身哈希。）

## V0a 已实现（本版）

- **编址常量拆分**：`CORES_PER_DIE / DIE_X / DIE_Y / DIE_COUNT / TOTAL_CORES /
  HOST_ENDPOINT_ID`（`defs/spec.{h,cpp}`）。单 die 时全部退化为旧 `GRID_SIZE`。
- **全局编址 helper**：`GlobalId / DieOfGlobal / LocalOfGlobal / Die{X,Y}Of /
  Local{X,Y}Of / IsDieEdge / DecodeEndpointType`（`utils/router_utils.{h,cpp}`）。
- **die 配置解析**：hardware config 支持 `"die":{x,y}`（缺省单 die）与矩形 die 内
  mesh（可选 `"y"`）（`utils/config_utils.cpp`）。
- **端口数据结构 + 配置解析 + 校验**：`die/port.h`、`die/port_config.cpp`
  （`D2DPort/D2DLink/D2DPortTable`、`ParseDiePorts/ValidateDiePorts`、
  `port_for_host`、endpoint 解码）。校验含：MVP `side==dir`、重复端口、越界 idx、
  缺方向 C2C、**HOST 可达性**。
- **矩形拓扑修正**：`GetInputSource` 的 Y 轴改用 `GRID_Y`（方阵时无变化）。
- **N/S 坐标固定**：`NORTH=y+` → `S=row 0`、`N=row(Y-1)`、`W=col 0`、`E=col(X-1)`。
- **消息头 bug 修复**：DATA 构造函数原漏初始化 `source_` 却仍被序列化，已给定值
  （`common/msg.h`）。注意 `source_/des_` 序列化为无符号 16 bit，取值应为合法 endpoint
  （0..65535）。**DATA 携带真实全局 source（发送核 cid）已在 V0b-1 落地**（见下）。
- **参数范围校验（第二轮补）**：`die_ports.c2c` 的 `link_bw>=1 / latency>=0 /
  buffer_depth>=1`，非法即启动报错（`die/port_config.cpp`）。
- **endpoint 三分解码（第二轮补）**：`DecodeEndpointType` 现返回 core/host/mem——
  core `[0,TOTAL_CORES)`、host `[TOTAL_CORES, +DIE_COUNT)`、mem `[+DIE_COUNT, +2*DIE_COUNT)`
  （provisional 布局，`utils/router_utils.cpp`）。
- **flow/link 统计接口（第二轮补）**：`D2DPortStats` + `D2DPortTable::TotalActivity()`；
  单 die 恒为 0（`die/port.h`）。
- **消息逐字段序列化往返测试（第二轮补）**：`SerializeMsg/DeserializeMsg` 往返 +
  16-bit 边界（见自测第 8 组）。

## 测试（全部通过：自测 165/165，端到端 23/23 组）

> **验证口径（对比范围）**：回归/多 die 用例对比的是 (a) 进程退出码、(b) 仿真结束
> **sim-time（ns）**、(c) **独立层级模块计数**（`dynamic_cast` 遍历 SystemC 层级，非自报
> `TOTAL_CORES`）、(d) **运行结束后 D2D 统计**（`[D2D] in/out/busy/stall`）。**并未**逐字节
> 比对完整 stdout / trace / 内部状态文件。故下文「一致/不变」均指**仿真时序（ns）精确一致**，
> 而非 “byte-identical”。

| 组 | 覆盖 | 结果 |
| --- | --- | --- |
| L0 纯函数自测（`--d2d-v0-selftest`，**165 项**） | 编址 round-trip、core/host/mem endpoint 解码、**每 die HOST endpoint 语义（`IsHostEndpoint`/`HostEndpointOfDie`/`GetNextHop` 分类）**、矩形 `GetInputSource`、`IsDieEdge`、**`OpenMeshNeighbor`（无 wrap/不跨 die）**、1x1/2x1/1x2/2x2 die 构造、端口校验（合法/缺方向/HOST 不可达/重复/side≠dir/非法参数/无端口 legacy）、消息全字段序列化往返（refill_/config_end_ 用 true、默认构造确定性、16-bit 边界）、endpoint 容量校验（>65536 及非正维度报错）、单 die D2D 活动计数=0、**HOST 挂载表结构 + 路由收敛（V1-pre 2a/2b）：5 拓扑 × core→lane 范围/legacy 精确值/lane→tile 合法且同 die/非法→-1/`ValidateHostAttach` 正反例/非法 HostEnvelope 拒绝/逐核走向 HOST 距离严格下降/跨 die HOST 正反向拒绝、config 驱动挂载表(W-host==legacy/S-host/2×1 c2c+host)+重复 override 拒绝 + S+W 角点重复挂载拒绝 + egress anchor 稳定性(固定 source 不退回 pos)/缺 anchor 正反向报错/非法映射报错** | 165/165 |
| 有效 `die_ports` 端到端 | 真实 `ParseHardwareConfig` 路径解析并运行，sim-time 不变 | 29109 ns ✔ |
| 非法 `die_ports`（缺方向） | 2x1 缺 E 向 C2C → 启动期报错、非零退出 | ✔ |
| 非法参数 ×3（参数化） | `link_bw=0` / `latency=-1` / `buffer_depth=0` 各自单独经真实配置路径被拒 | ✔✔✔ |
| **V0b-2A 多 die 实例化 ×3** | **2×1 / 1×2**（层级计数 router==worker==**32**）、**2×2**（==**64**）；die0-only workload sim-time 仍 29109；运行后 **D2D 活动 0** | ✔✔✔ |
| 单 die 回归 | noc_congestion no_congestion cycle sim-time 不变 | 29109 ns ✔ |

运行 `python3 llm/test/d2d_link/run_test_d2d_v0.py` → **23/23 组通过**。单 die 完整回归
（`llm/test/noc_congestion`）四场景 sim-time 与基线**精确一致**：no_congestion 14781/29109、
congestion 14833/45441 ns。

> **条件编译现状**：默认构建即 `USE_L1L2_CACHE=1`、`DUMMY=1`、`DCACHE`（`macros.h`）——这些路径
> **已在所有多 die 用例中编译并运行**（die1/dual-die 跑通时 cache system 按 `TOTAL_CORES` 实例化）。
> 仍未做：**ASan/UBSan** 干净性验证；非默认编译组合（关闭 cache 等）未单独验。

## V0b 进展（分增量，每步单 die 逐位回归）

V0b 按建议顺序分增量推进,**全部增量已完成**(1、3、前置 hardening、2A、4、2C0、2C1、2B0、2B1),
每步单 die 逐位回归通过(14781/29109、14833/45441)。**workload 已能在 die>0 运行并多 die 并行**
(见 V0b-2B1)。以下按增量记录。

### V0b-1 ✔ 编址 / 消息语义收口

- `HostEndpointOfDie() / MemEndpointOfDie() / DieOfHostEndpoint()` helper（`router_utils.h`）。
- HOST 语义 sentinel `des_/source_ == GRID_SIZE` → **每 die HOST endpoint 区间**。
  新增 `IsHostEndpoint(ep)` 识别整段 `[TOTAL_CORES, TOTAL_CORES+DIE_COUNT)`（每 die 一个
  host），用于 `GetNextHop`/`GetNextHopReverse`（`router_utils.cpp`）与 `router.cpp` 的
  output_lock 判断——不再只认单一 `HOST_ENDPOINT_ID`（否则 `HostEndpointOfDie(1)` 会被当
  普通 core 坐标误算）。`config_helper_*` 的 host 注入用 `HOST_ENDPOINT_ID`，`logic.cpp` 的
  **DONE/ACK 改发本核所在 die 的 host**（`HostEndpointOfDie(DieOfGlobal(cid))`）。单 die 下
  `DIE_COUNT==1`，该区间只含 `HOST_ENDPOINT_ID==GRID_SIZE`，逐位不变。
- **DATA 现携带真实全局 `source_ = cid`**（`logic.cpp` 三处发送点），取代原 -1。
- **保留** `GRID_SIZE` 的几何/容量语义（`min(sms,GRID_SIZE)`、`GRID_SIZE/GRID_X`、
  `recv_tag<GRID_SIZE`）不动——不机械替换。

### V0b-3 ✔ live torus 拆除（开边 mesh）

- 新增纯函数 `OpenMeshNeighbor(global, dir)`：无环绕、不跨 die，边缘返回 -1（`router_utils.cpp`）。
- `monitor.cpp` router-router 绑定改为开边 mesh：**边缘方向输入侧接共享终结通道**
  （`term_channel/avail/sent` + ctrl），**运行时取模环绕连接已物理移除**。
- 关键验证：单 die noc_congestion 四场景**逐位不变**——这从运行时**证明** wrap 链路本就
  inert，现已真正拆除（非「未使用」）。自测 `OpenMeshNeighbor` 9 项（无 wrap、不跨 die、
  边界⇔-1 遍历一致）。

### V0b-2 前置 hardening（扩容到 TOTAL_CORES 前的安全修复）

> **定位**：本节修的是**具体的确定性 UB/溢出 bug**（`delete[]`、g_dram_kvtable UAF/double-free、
> 地址溢出），**不是完整的资源生命周期修复**。退出期仍有已知泄漏（见末条），属退出时回收、
> 不影响正确性与扩容安全性。

- **地址校验防 int 溢出**：`config_utils.cpp` 现**先**用宽整数 `ValidateAddressSpace()`
  校验维度正数 + endpoint 空间 ≤ 65536，**再**计算 `GRID_SIZE/DIE_COUNT/TOTAL_CORES` 等
  int 全局量——极大非法配置（如 `x=100000`）在赋值前即被拒（实测报 `10000000002 exceeds
  65536`，证明无有符号溢出），退出码 1。
- **数组释放用 `delete[]`**：`RouterMonitor::~`（`router.cpp`）、`Monitor::~`
  （`workerCores`、`g_dram_kvtable`）的数组释放由 `delete` 改为 `delete[]`。
- **消除 g_dram_kvtable use-after-free / double-free**：`WorkerCoreExecutor::~`
  原先 `delete` 整个数组后再访问其元素（UAF），且多核会重复 delete 该数组（double-free）。
  现改为**只 `delete g_dram_kvtable[cid]`（本核元素）**；数组由 `Monitor` 统一 `delete[]`。
- **WorkerCore 对象仍有意保留（不逐个析构）**：实测启用逐个 `delete workerCores[i]` 会在
  退出时**段错误**——其析构链存在既有 SystemC teardown 隐患（成员释放/拆解顺序），与
  g_dram_kvtable 无关。故当前退出码 0（正常），代价是 Worker 对象泄漏（进程退出即回收）。
  由于不析构 Worker，扩容到 `TOTAL_CORES` 也**不会**触发 g_dram_kvtable 多核 double-free。
  **完整 Worker teardown（或改 `unique_ptr`）是独立清理项**，留待专门修复。
- **已知退出期泄漏（同类，留待 V0b-2B）**：die>0 HOST 终结的独立 sink、开边 mesh 的终结信号
  均以裸 `new` 分配且未保存指针（进程退出即回收）。V0b-2B 应改由 `Monitor` 成员容器统一持有，
  与 HOST attachment 一并规整。故本节**不宣称**完整生命周期修复。

### V0b-2A ✔ 运行时多 die 模块实例化（die>0 存在但空闲）

- 核级数组/遍历/模块数量按 **`TOTAL_CORES`**：`monitor.cpp`（workerCores、g_dram_kvtable、
  channel/ctrl 信号、core&router、router-router 循环）、`router.cpp`（`RouterMonitor` 建
  `TOTAL_CORES` 个 router）、`system_utils.cpp`（dram/dcache 数组）、cache system processor 数。
  **保留** `GRID_X/GRID_Y`（单 die 几何）、`CORES_PER_DIE`（die 内规模）、HOST 信道仍 `GRID_X`。
- **`GetCoreHWConfig` 同构映射**：全局核 id → `id % CORES_PER_DIE` 的 die 内模板（含
  `GetCoreHWConfigForGlobal` 显式入口），避免 die>0 executor 取不到 HW 配置导致的除零。
- **die>0 HOST 端口终结**：die>0 西边缘 router 仍创建 HOST 端口，V0b-2A 只接 die0 HOST；
  对 die>0 的 HOST 端口做终结绑定——**输入共享只读终结信号，每个输出各自独立 sink**
  （避免 multi-writer）。
- 验收（全部通过，见 runner `V0b-2A * die` 组）：**2×1/1×2 建 32、2×2 建 64 个 router+worker**；
  die>0 空闲；die0-only workload 仍精确 **29109 ns**；无 unbound-port / multi-writer / 崩溃；
  D2D 活动 0。单 die 逐位回归不变。

### V0b-4 ✔ D2D peer/link 结构级构造与配对校验（含收口）

- `BuildD2DLinks()`（`die/port_config.cpp`，`ParseDiePorts` 末尾调用）：从同构端口模板 +
  die-mesh 构造 `g_d2d_links`——每个 C2C 端口找**方向邻 die 上对侧同垂直坐标的镜像端口**成对。
- **收口（本轮）**：`ParseDiePorts` 开头 `g_d2d_links.clear()`（消除重复解析残留）；**精确四元组
  互反校验**——link `(a,b)->(c,d)` 必存在反向 `(c,d)->(a,b)`（原弱判断只查远端端点是否出现，
  会漏「远端指向第三端点」）；端口唯一性（每 (die,port) 至多一条 link）。
- **带宽/延迟兼容 = 同构保证**：V0 所有 C2C 端口共享全局 `die_ports.c2c` 参数，镜像端口天然相等，
  故该 check **恒不触发**——是同构保证、**非**已测的不兼容拒绝路径（per-port override 属后续）。
- 验收（自测 6h，全通过）：2×1 满边 → **8 条有向 link**、精确四元组互反、单 die 0 link、
  **负向：镜像坐标不匹配（E@row0/W@row1）报错**、**re-parse 清空 g_d2d_links**。

### V0b-2C0 ✔ 跨 die destination 可靠拒绝（raw-JSON preflight，已验证契约）

- **绘图健壮化**：`plot_dataflow` 遇 `DIE_COUNT>1` 跳过（`display_utils`）——单 die 调试辅助且有
  `GRID_SIZE` 网格假设，跳过后不会因 die>0 id 段错误（绘图失败不阻断解析）。单 die 逐位不变。
- **raw-JSON preflight（`config_helper_core::PreflightValidateWorkload`）**：在 `plot_dataflow` /
  `CoreConfigRemap` / o2r / SystemC elaboration **之前**，直接扫描**原始 JSON** 的
  `core.id / cast.dest / source.dest / prim_copy / send_global_mem`，拒绝：
  - **跨 die cast**（`cross-die traffic requires D2D Link (V1)`）；
  - 越界 id（`>= TOTAL_CORES`）；
  - die0-local `id_space` 下引用 die>0 核（提示改 `"id_space":"global"`）；
  - 目标 die>0 的核（`running on die>0 needs V0b-2B`，避免配置分发不到 die>0 后运行时永久等待）。
  这修好了此前 guard 无效的根因——不再依赖尚未填充的 `work.cast`。
- **id_space 约定**：缺省 = die0 local（旧配置兼容）；`"global"` = ids 落在 `[0,TOTAL_CORES)`。
- **验收（runner `V0b-2C0`，10s 超时保护，已通过）**：`core0 → core16` **启动期非零退出、
  报 `cross-die traffic requires D2D Link`、不进入 SystemC 仿真、不永久等待**；单 die 四场景
  逐位不变；V0b-2A die0-only 用例不受影响。**不要求 die1 workload 能运行**（符合 2C0 范围）。
- 遗留小项：`plot_dataflow` 仍各自 reopen 文件（preflight 已单独解析一次），「plot 接受已解析
  json、全局只解析一次」的清理留待后续。

### V0b-2C1 ✔ global-id 归一化 + 结构校验（构造级，不跑仿真）

- 新模块 **`monitor/workload_normalize.{h,cpp}`**：
  - `NormalizeWorkloadJson(j, die_id)`：JSON 层把所有 core-reference id（`core.id`/`prim_copy`/
    `send_global_mem`/`cast.dest`/`source.dest`）从 die 内 local **平移到 global**（`+= die_id*CORES_PER_DIE`）
    并置 `id_space:"global"`；`die_id==0` 恒等（旧配置兼容）。这是方案推荐的**低风险 JSON 层归一化**。
  - `ValidateWorkloadStructure(j)`：**纯函数、可独立测试**的结构校验（bounds + 跨 die cast），
    从 2C0 的 preflight 拆出——与「die>0 能否运行」的运行时判断分离（后者留在 `config_helper` ctor，
    接入 2B 后移除）。
- **`o2r/r2o` 收口**：生产路径的 `CoreConfigRemap`（`config_utils.cpp`）本就用 `g_core_remap` **map**
  （天然支持 global id），用 `tag==dest` 判 core-ref（非 `tag<GRID_SIZE`）；仅（停用的）`random_core`
  里的 `int o2r[GRID_SIZE]` 改为 `vector(TOTAL_CORES)`、`tag<TOTAL_CORES`。
- 验收（自测 6i，全通过，**不跑仿真**）：logical 0/1 → global 16/17 各字段正确、`id_space=global`；
  **die1-local(16→17) 结构合法**；**cross-die core0→core16 拒绝**；越界/die0-local 引用 die>0 拒绝；
  `die_id=0` 归一化恒等。单 die 逐位不变。

### V0b-2B0 ✔ HostEnvelope 接口 + LegacyHostEnqueue（解耦 config_helper 与物理 HOST lane）

- 新 `monitor/host_envelope.{h,cpp}`：`struct HostEnvelope{int dest_global_id; Msg msg;}` +
  `LegacyHostEnqueue(envs, q)`（按 die0 西边 row `dest/GRID_X` 落入 `write_buffer`）。
- `config_helper_core` 新增 **`BuildConfigMessages()` / `BuildStartMessages()`** 返回信封——只决定
  「发给哪个全局核 + 什么 Msg」，**不含物理 HOST lane/write_buffer 知识**；`fill_queue_config/start`
  接口不变，内部 = `Build*Messages()` + `LegacyHostEnqueue`。
- **验收（关键：单 die 逐位不变）**：信封按原顺序生成、legacy backend 按 `dest/GRID_X` 落队（与原
  `config.id/GRID_X` 一致）→ 消息 dest/顺序/数量完全一致；noc_congestion 四场景 14781/29109、
  14833/45441 **逐位不变**。（2B0 时 die1 HOST 尚走 legacy die0；2B1 已接入 per-die lane。）
- 这是 **2B1 的接口地基**：per-die HostAttachment 将消费这些信封、按 `DieOfGlobal(dest)` 路由到对应
  die 的 HOST lane，取代 `LegacyHostEnqueue`。

### V0b-2B1 ✔ per-die HOST lane —— workload 真正在 die>0 运行

- **HOST lane 数 `GRID_X` → `HOST_LANES = GRID_Y * DIE_COUNT`**（`spec` 全局 + `config_utils`）：
  MemInterface 的 host 端口/`write_buffer`、Monitor 的 host 信号/绑定循环全部按 `HOST_LANES`。
- **优雅的通用映射**：lane `i` ↔ 全局行 `i` ↔ 西边缘 router `i*GRID_X` ↔ `write_buffer[i]`，而
  `dest/GRID_X` 恰好就是 lane——所以 `LegacyHostEnqueue`、Monitor 绑定循环**天然泛化到多 die**
  （单 die 方阵 `HOST_LANES==GRID_X`，逐位不变）。die>0 的 HOST 终结块删除，改由 lane 循环统一接入。
- **DONE/config 路由**：`GetNextHop` 用 `IsHostEndpoint`（每 die host 区间），die>0 核的 DONE 走
  `HostEndpointOfDie(die)` → 本 die 西边缘（开边 mesh 不跨 die）→ 本 die HOST lane → MemInterface。
- **2C-main 审计（本轮触发修复）**：`config.cpp` 的 `c.id >= GRID_SIZE` → `>= TOTAL_CORES`（否则 die>0
  核 id 被误判越界、`send_global_mem` 未初始化触发断言）。
- **移除 die>0-unrunnable guard**（2C0 加的运行时限制）；跨 die cast 仍被结构校验拒绝。
- **验收（runner，已通过）**：
  - **T3 die1 等价运行**：die1 平移 workload → 16 个 die1 核全部执行、sim-time=29109（die0 基线
    等价）、D2D 活动 0、退出码 0。
  - **T2 dual-die 同时运行**：die0(0-15)+die1(16-31) 合并 workload → **两 die 各 16 核全部执行、
    sim-time=29109（完全并行、零干扰）、D2D 活动 0、退出码 0**。
  - **布局泛化**：die1@1×2（纵向）、**die3@2×2**（更高 die + 2D 布局，64 核，die3 核 48-63 全执行、
    sim-time=29109）——验证 `HOST_LANES=GRID_Y*DIE_COUNT`、`lane=dest/GRID_X` 对任意 die/布局成立。
  - 单 die 四场景 sim-time 精确一致；自测 87/87、runner **15/15**（均为 2B1 当时快照；
    V1-pre 2a..3b-2a 后自测升至 165/165，见文末 V1-pre 段）。

## V1-pre / V0-exit 进展（分增量，每步单 die sim-time 精确一致）

V0 功能性目标已全部达成。进入 V1 前，先把「配置驱动的 HOST attachment」作为前置里程碑落地
（原被列为 polish，实为 V1 正确性所需——见下）。每步仍以 **sim-time / 核集合 / D2D 计数 /
层级模块数** 精确一致为回归门（**非**逐字节内部状态比对）。

### V1-pre Inc 1 ✔ HOST 挂载表接缝（legacy 构造器，sim-time 精确一致）

- 新增 `HostAttachTable g_host_attach`（`die/port.h`、`die/port_config.cpp`）作为 routing / enqueue /
  binding 的**统一真源**；`BuildHostAttach()` 在 `ParseDiePorts` 后由 `config_utils` 调用。
- 目前仅 **legacy 构造器**：西边缘、每全局行一条 lane，`lane_tile[i]=i*GRID_X`。三处消费点改为查表
  （Monitor 绑定 `HostTileOfLane(i)`、`LegacyHostEnqueue` 用 `HostLaneOfCore(dest)`），返回值与旧
  硬编码 `i*GRID_X`/`dest/GRID_X` 相等 → 所有布局 sim-time 不变。

### V1-pre Inc 2a ✔ 挂载表加固 + 强一致 + 结构性自测

- **补全 core→lane 映射**：`core_lane`（size `TOTAL_CORES`），`HostLaneOfCore` 改为查表、非法核返回
  `-1`；表现在持有 `lane_tile` 与 `core_lane` 双向完整映射。
- **强一致**：`HOST_LANES = g_host_attach.n_lanes`——**表为 lane 数唯一真源**，Monitor/MemInterface
  数组与循环都以 `HOST_LANES` 为界，杜绝 config 改变 lane 数后二者分叉。
- **独立 `ValidateHostAttach()`**：`lane_tile/core_lane` 尺寸、每 lane tile 合法、每 core 映射合法且
  **HOST tile 与该核同 die**、生产路径 `HOST_LANES==n_lanes`；启动期调用，非法即抛。
- **入队防御**：`LegacyHostEnqueue` 入队前校验 lane ∈ [0,HOST_LANES)，非法 dest 抛异常而非写 `q[-1]`。
- **范围边界**：仅 dataflow 的 HOST 入队经挂载表；PD/GPU/PDS 的 `fill_queue_*` 仍用 `id/GRID_X`
  （legacy 下与表相等），config 驱动 HOST 仅 V1 dataflow 生效——已在 `die/port.h` 标注。
- 自测 **87 → 142**（5 拓扑含矩形 die 内 mesh；每拓扑 core→lane 范围/legacy 精确值/lane→tile
  合法且同 die/非法→-1/`ValidateHostAttach` 正反例）；runner 15/15；所有布局 29109 ns。

### V1-pre Inc 2b ✔ HOST 路由收敛到挂载 tile（sim-time 精确一致）

- `GetNextHop/GetNextHopReverse`（`router_utils.cpp`）改为:HOST 目的时朝
  `HostTileOfLane(HostLaneOfCore(pos))` 做 XY 收敛,`pos==tile` 才发 HOST。删除旧的
  `IsMarginCore(pos)` 硬编码西边分支(及 `CTODO: fix this`)。legacy 下 tile=本行西边缘,
  「朝 tile 做 XY」≡ 旧的「一路向西到 col0 再 HOST」——所有布局 sim-time 不变。
- **路由层保证(显式拒绝跨 die HOST)**:两函数在 HOST 分支先比 `DieOfHostEndpoint(des)` 与
  `DieOfGlobal(pos)`,**不同 die 即抛 `cross-die HOST endpoint requires D2D routing`**——绝不
  静默返回 HOST。这修好了旧自测 `GetNextHop(die1-host, die0-core)==HOST` 暴露的真实缺陷
  (仅靠 `IsHostEndpoint` 判断、且 die0 core0 恰是本 die 挂载 tile 时会误发)。
- 自测 **144 → 156**:5 拓扑 × 正/反向逐核走向本 die HOST(每步 Manhattan 距离严格下降、不跨
  die、无环、终于挂载 tile 发 HOST)+ 跨 die HOST 正反向拒绝(die0 核→die1 HOST 必抛错、
  错误文本含 `cross-die HOST`;die1 核→die1 HOST 正常收敛)。runner 15/15;所有布局 29109 ns。
- **已知边界**:legacy 下沿路 `pos` 同行,`HostLaneOfCore(pos)` 沿途不变故 tile 稳定;非西侧
  (config)HOST 需 per-message ingress anchor(不能只靠 `pos` 推导),留给 Inc 3。

### V1-pre Inc 3a ✔ config 驱动挂载表构造器 + 冲突校验（结构级，不改路由）

- `BuildHostAttach`（`die/port_config.cpp`）新增 **config 分支**:`die_ports` 提供 `role=HOST` 端口时,
  从这些端口建挂载表(同构 die 模板每 die 复制一份):每个 HOST 端口 = 一条 lane,`lane_tile` = 该
  端口全局 tile,`core_lane` = 核就近 HOST 端口(`port_for_host`)对应的 lane。无 die_ports / 无 HOST
  端口时仍走 legacy。
- **关键等价**:W 全边 host 配置产出的表与 legacy 西边缘**完全相同**——故现有 `core_4x4_ports_ok`
  e2e 走 config 分支仍精确 **29109 ns**。
- **一个物理端口只能一个 role**:两个 override 命中同一 `(side,idx)` → 拒绝(**复用已有的重复
  override 校验路径**,保证同一端口不能既 host 又 c2c;非新增专用校验)。
- **legacy 触发条件更正**:只有「无 die_ports」才走 legacy;有 die_ports 但无 HOST 端口会**先**被
  `ValidateDiePorts` 以「HOST 不可达」拒绝,不会落到 legacy。
- 自测 **156 → 160**:W-host==legacy、S-host(每列一 lane / tile=南边缘 / core_lane=列)、
  2×1 die(E/W-c2c + S-host,`hp*DIE_COUNT` lane、tile 同 die、`ValidateHostAttach` 通过)、
  重复 override 拒绝。runner 15/15;所有布局 29109 ns。
- **本段不改路由**:`GetNextHop` 仍 pos 推导(legacy 语义)。西边缘配置下 pos 推导 ≡ config 表故一致;
  **非西侧 HOST 的路由 + egress anchor + e2e 属 Inc 3b**（下）。

### V1-pre Inc 3b-1 ✔ RouterUnit HOST 接口按挂载表实例化 + 单 tile 单 lane 校验（sim-time 精确一致）

- **RouterUnit HOST 接口改由挂载表驱动**:`router.cpp` 的 HOST 接口创建、sensitivity、
  `end_of_elaboration` 三处 `IsMarginCore(rid)` → **`IsHostAttachTile(rid)`**(其余 HOST 收发/析构
  本就按指针非空守卫,自动适配)。legacy 下挂载 tile==西边缘,`IsHostAttachTile` 匹配同一集合 →
  sim-time 精确一致。这是让 Monitor 能按挂载表绑定 S/N/E tile 而不解引用空指针的**硬前置**。
- **单 tile 单 HOST lane 校验**:`HostAttachTable` 加反向 `tile_lane`(O(1) 的 `IsHostAttachTile`/
  `HostLaneOfTile`);`ValidateHostAttach` 拒绝**重复 `lane_tile`**——RouterUnit 每 tile 仅一套 HOST
  接口,S+W 全边 host 会让西南角 tile 被两条 lane 挂载、Monitor 重复绑定同组 port,故启动期拒绝。
- 自测 **160 → 161**:S+W 角点重复挂载被拒。runner 15/15;所有布局 29109 ns。
- **3b-2 前置兜底**:`router.cpp` 控制/数据路由在 `out==HOST` 分支**直接**访问 `host_ctrl_buffer_o`/
  `host_buffer_o`(无非空守卫)。加显式检查 `out==HOST && (!IsHostAttachTile(rid) || 指针==nullptr)
  → throw`(同时查表与指针),让 3b-2 改路由时的错误变成明确报错而非空指针崩溃。

### V1-pre Inc 3b-2a ✔ egress anchor 取自消息 source（路由不再用 pos 推导挂载 tile）

- **`GetNextHop/GetNextHopReverse` 加显式 `anchor_core` 参数**(`router_utils.{h,cpp}`):HOST 目的时
  挂载 tile = `HostTileOfLane(HostLaneOfCore(anchor_core))`,**不再用 `pos`**——否则多非西侧 HOST 端口
  间会沿途重选 tile。校验:anchor 为合法 core、与目标 HOST **同 die**、缺 anchor(-1)即报错(杜绝隐式
  退回 pos)。跨 die HOST 仍拒绝。非 HOST 路由忽略该参数。
- **router 首次路由传 `m.source_` 作 anchor**(控制 [router.cpp:319]、数据 [router.cpp:355] 两处
  HOST-dest 站点)。DONE/ACK 的 `source_` 是真实全局 core id,可直接用;host→core 注入与 DATA 尾包
  解锁重算目的均非 HOST,忽略 anchor。
- **divergence 单元测试**:固定 `anchor=A`(行0)、`pos=P`(行1,恰是 lane1 挂载 tile),断言路由
  = `SOUTH`(朝 A 的 tile),而 `anchor=P` 时才 = `HOST`——**证明路由始终用固定 source,绝不退回
  pos**。walk 测试也改为 anchor=起点核固定、pos 沿途变。
- 自测 **161 → 165**;runner 15/15;所有布局 29109 ns。**注意**:现有 e2e 仍用**西侧** HOST,
  它只证明生产路径正确传递合法 `source_`,**不能单独证明非西侧路由正确**;非西侧正确性此时主要由
  divergence 单测证明,真正的运行时闭环见 3b-2b。
  （另补两低成本单测:`GetNextHopReverse` 缺 anchor 报错、anchor 映射非法(lane/tile=-1)明确报错。）

### V1-pre Inc 3b-2b ✔ S-host 单 die 端到端 —— 非西侧 HOST 真正运行时闭环

- **HOST lane 接收统计**(`mem_interface.cpp` `recv_helper`):每 lane 的 DONE/ACK 计数 +
  **mismatch**(消息到达 lane != `HostLaneOfCore(source_)` 的次数)+ **接收签名**——DONE 按 `source`、
  **ACK 按 `(source, tag_id)`**(`g_host_done_src`/`g_host_ack_sig`);运行后 `npusim` dump
  `[HOSTLANE] ...` 与 `[HOSTSIG] done=source:count ack=source:tag:count`。
- **新配置** `hardware/core_4x4_ports_{s,e,n}host.json`(单 die,`edges.{S,E,N}=host`)。

### V1-pre Inc 3c ✔（四方向）W/S/E/N HOST 端到端 + 证据加固

runner 组 **V1-pre 3c {W,S,E,N}-host**。每方向**连续跑两次,两次都**断言:
- **mismatch=0**——每条 DONE/ACK 到达 `HostLaneOfCore(source_)` 对应的正确边缘 lane;
- **接收签名与 west 一致**——DONE `(source,count)` + ACK `(source,tag_id,count)` **严格签名**,并断言
  **ACK 结构**:每 source 恰 1 个 CONFIG ACK(tag 0)+ 1 个 WEIGHT ACK(tag=core_id);source 0 两者
  tag 都为 0 → `(0,0):2`。这比只比总数强(能发现「同 source 丢一条 + 重一条**不同 tag**」),但**仍不是
  逐消息全等**:**同 source 同 tag 的丢+重会抵消、无法发现**(如 source 0 的 `(0,0):2`),且签名未含
  逐消息唯一序号/事件轨迹。故准确表述为 **「每 (source,tag) 接收计数与 west 一致」**,不写「严格无丢包
  且无重复」。
  - **逐消息身份的正确做法(未做)**:当前 ACK 构造未设 `seq_id_`,默认恒为 -1(`msg.h`、`logic.cpp`
    的 `Msg(ACK, host, tag, cid)`),故仅把 `seq_id` 塞进签名**不够**。需发送端为每条 ACK 分配
    **稳定且唯一**的序号(如 `(source, ack_seq)`),再纳入签名,方能逐消息去重/查漏。
- **预期 per-lane 分布**(不只总数):由 west 的 DONE 源按该方向 lane 函数推导——runner **读 `GRID_X/GRID_Y`**
  (不硬编码 4):W/E lane=`source//GRID_X`(数=GRID_Y)、S/N lane=`source%GRID_X`(数=GRID_X);
  4×4 下 W/E=`2,2,2,2`、S/N=`4,0,4,0`,实测吻合;
- **D2D=0**、无 unbound/multi-bind、16 核完成;
- **连续两次「所有已检查指标」一致**(sim-time / mismatch / 接收签名 / per-lane DONE / D2D / 绑定 /
  核集合)——**不含**内部消息时序、router 状态、完整事件轨迹的比对。
- sim-time 各方向:W 29109 / S 29123 / E 29063 / N 29081,两次一致,**差异符合各自 hop 变化的预期**
  (非严格因果证明)。**W 也走两次**(用 `edges.W=host` 配置分支,挂载同 legacy 但非真 legacy 路径;
  真 legacy 无 die_ports 路径由「单 die 回归」组覆盖)。

**矩形 4×2 die 内 mesh(8 核)**:独立小型 dataflow workload `workload/gemm_4x2.json`(取 gemm 前 8 核,
核 0–7 自成 4 对不引用他核;8 核全产 ACK)+ 配置 `hardware/core_4x2_ports_{w,s,e,n}host.json`
(`x=4,y=2` 单 die)。**独立 west 参考签名、期望核数 = `GRID_X*GRID_Y = 8`**;DONE 源 `{0,2,4,6}` 使
**行分布 `[2,2]`(2 lane) vs 列分布 `[2,0,2,0]`(4 lane)明显不同**,错误映射难碰巧通过。至少一个
水平挂载(W/E)+ 一个垂直挂载(S/N),四方向全覆盖;各方向两次一致(W/E 29109/29063、S/N 24921/24909)。
**曾发现并已修复的 UB**:CONFIG ACK 的 tag 来自 `Recv_prim(RECV_CONF)` 的**未初始化** `tag_id`
(4×2 暴露 run-to-run 变化 30535/28719...;4×4 恰为 0 是内存布局巧合)。它**不影响当前 HOST lane 选择**
(路由 anchor 用 `source_`,mismatch 恒 0),但属 UB 且 V1 的 flow-key 会用 tag。**已修**:`Recv_prim`
全字段类内默认值(`norm_prims.h`)+ 显式 CONFIG ACK tag 契约=0(`workercore.cpp` 构造、`logic.cpp` 注释)。
修后 4×2/4×4 的 ACK tag 均确定,runner 恢复**严格 (source,tag)** 比对 + ACK 结构断言。

**这是非西侧 HOST 路由正确性的运行时证明**(divergence 单测之外的四方向 × 方阵/矩形闭环)。
runner **15 → 23 组**(4×4 W/S/E/N + 4×2 W/S/E/N);self-test 165/165。

### 非阻塞后续项（不在 V0 冻结范围）

- **证据精度残留**(CONFIG ACK tag UB 已修):ACK **逐消息**身份(区分 source 0 的 `(0,0):2`)
  需**发送端分配稳定唯一 ACK 序号**(如 `(source,ack_seq)`)再入签名——当前 `seq_id_` 默认 -1、未设置,
  仅塞 `seq_id` 不够;两次一致仅限已检查指标(不含事件轨迹)。

### V0 剩余（纯 polish，独立且低优先）

1. **ASan/UBSan 干净性**：可能牵出既有（SystemC/DRAMSys）问题，需专门一轮。cache/DUMMY 已随默认
   构建的多 die 用例覆盖，但非默认编译组合（关 cache 等）未单独验。
2. **完整 WorkerCore teardown**：既有 SystemC 拆解段错误（与本工作无关），现有意泄漏 Worker 对象
   + 开边 mesh 终结信号（进程退出即回收，不影响正确性）；独立清理项。
3. **非 dataflow 模式（pd/gpu/pds）多 die 化**：其 `config_helper` 仍按 `GRID_SIZE`——**V0 不要求**
   （D2D 走 dataflow 路径），若需再单独做。
