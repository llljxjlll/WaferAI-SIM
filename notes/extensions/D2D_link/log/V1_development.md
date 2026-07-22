# V1 开发总结（相邻 die 单链路 MVP：第一条真实跨 die 通信）

> 记录 WaferAI-SIM D2D Link 建模 **V1** 的开发全过程：目标、增量顺序、每步的设计与验证、
> 途中发现并修复的 bug、以及最终状态。规划见 `../D2D_link_test.md` 与 `../D2D通信建模计划.md`，
> 测试与逐增量说明见 `llm/test/d2d_link/README.md`，上一阶段见 `V0_development.md`。
>
> **一句话**：V0 让仿真器能实例化多 die、在 die>0 运行 die-local workload 但**不跨 die**；
> V1 打通**第一条真实跨 die 通信**——源核 NoC → D2D Link（固定延迟）→ 邻 die NoC，
> 完整 `REQUEST → ACK → DATA` 协议闭环，相邻 die、每方向单链路、1 packet/cycle、固定延迟。

## 1. 目标与约束

**目标**：让一条跨 die dataflow 真正跑起来——包从源核经片内 XY 路由到边缘 C2C 端口，
穿过带固定延迟的 D2D Link，落到邻 die 入口 tile，再经邻 die 片内 XY 到目的核；控制握手
（REQUEST/ACK）与数据（DATA）都真实穿链。**范围刻意收窄到 MVP**：

- **相邻 die**（`DieManhattan==1`）——多跳留 V2；
- **每个 die 级方向恰好一个 C2C 端口**（`ValidateV1MvpTopology`）——多端口/条带留 V5；
- **1 packet/cycle、功能性无限 FIFO**——有限缓冲/带宽/背压留 V3；
- **固定 latency**（无争用建模）。

**贯穿全程的铁律**（承袭 V0）：
- **单 die 仿真时序（ns）逐位不变**——`llm/test/noc_congestion` 四场景
  （14781/29109、14833/45441）每步回归，精确一致；无 C2C 端口配置退化为今天。
- **一次一个纵向增量**、**禁止一次同时改 preflight/控制路由/数据锁/完整 workload**。
- **诚实**——只声称验证过的；区分「结构接通」与「运行时闭环」。

## 2. 增量顺序（最终形态）

```
V1-a   固定单跳跨 die 路由（per-flow exit pinning，尚无 Link 单元）
  ↓
V1-b   link-site seam（识别 peer-connected C2C 出口边）
V1-b2  运行时 D2DLinkUnit（固定延迟 FIFO，honor 下游背压）+ MVP 拓扑契约校验
  ↓
V1-c0  FlowKey 类型 + 消息 16-bit pinned exit_port 字段
V1-c1  接通 REQUEST/ACK 控制路由过 Link
V1-c2  接通 DATA 数据路由过 Link
V1-c3  生产准入放行相邻 die + 按类型统计 + 第一条完整协议闭环 e2e
V1-c   drain 不变量（router 归零）
V1-c   aggregation-tag 锁语义（tag-only + tag→dest 唯一性 + 多发一）
  ↓
V1-d1  四方向相邻 die e2e（2×1 的 E/W、1×2 的 N/S）
V1-d2  DATA 逐包完整性探针 + 消息大小边界
V1-d3  latency 标定（0/1/7/20，建立并固化 3·L 公式）  ← V1 完成
```

> 与 V0 一样，这个切分是**多轮 review 后收敛**的（见 §5）。V1-c 尤其被要求拆成
> c0→c1→c2→c3 四小步，「不要一次同时改 preflight、控制路由、数据锁和完整 workload」。

## 3. 各增量：做了什么 / 怎么验证

### V1-a 固定单跳跨 die 路由
- die 级维序首跳方向 `DieFirstHopDir(my_die, des_die)`（die 级 XY，先 X 后 Y）、`DieManhattan`。
- **per-flow exit pinning（铁律）**：`CrossDieSelectExit(at_core, des)` 在流启动（SEND_DATA）时
  选**一次**出口 C2C 端口，钉进 flow context，之后该流每个包复用；`CrossDieStep(des, pos,
  exit_port)` 收敛到钉死的出口——**不信任包里携带的值**，每跳按 `des_die!=my_die` 重新判定并
  朝钉死出口的 tile 走。`ValidatePinnedExit` 校验携带的 exit 对「从 md 去 dd」仍合法
  （范围/ROLE_C2C/dir==die 首跳方向/MVP side==dir/tile 合法）。
- **验证**：纯函数自测（单 die 同 die→原片内 XY 零变化、相邻 die→选唯一出口、多跳→抛错）。

### V1-b link-site seam
- `IsC2CEgressEdge(global_tile, dir)`：该 tile 局部位置有一个 `side==dir` 的 C2C 端口且该 die
  在 dir 方向存在邻 die。V1-b 拓扑接线据此把该边接到 D2D Link（取代 V0 的开边终结）；
  单 die / 无 C2C 时恒 false，拓扑不变。
- **验证**：2×1 满 E/W 配置识别出 2 条 peer-connected 出口边；单 die 0 条。

### V1-b2 运行时 D2DLinkUnit
- 新模块 `die/d2d_link.{h,cpp}`：`D2DLinkUnit`（SC_MODULE + SC_THREAD）。功能 FIFO
  `{ready_cycle, payload}`：每 cycle 采集上游真实包入队、记 `ready_cycle=capture+latency`；
  队首成熟（`ready<=cyc`）且**下游 `out_avail=true`** 时出队一包（FIFO 序）。
  **honor 下游背压**：下游不 ready 时成熟包留队首等待（不丢/不重/不越容量）；
  data/ctrl 各一条独立 FIFO；V1 对上游恒 avail=true（无限功能队列，不施背压）。
- `monitor.cpp`：`if (!g_d2d_links.empty()) ValidateV1MvpTopology()` 在**任何 deferred binding 和
  Link 实例化之前**调用；按 `IsC2CEgressEdge` 在出口边插入 D2DLinkUnit 取代终结；
  `[D2D] link_sites= link_units=`（独立层级计数，非 `g_d2d_links.size()` 自报）。
- `ValidateV1MvpTopology`：每方向 C2C 端口 ≤1、存在邻 die 的方向恰好一个 peer-connected C2C、
  `link_bw==1`、`port_id` 可编码、多 die 必须有 die_ports。
- **验证**：独立 link 自测 `--d2d-link-selftest`（latency 0/1/7/20、顺序、无丢/重、下游 stall、
  drain、data/ctrl 独立）**18/18**；生产启动负例（`link_bw=2` 拒绝、`side!=dir` 拒绝）。

### V1-c0 FlowKey + 消息 pinned exit_port 字段
- 新 `common/flow.h`：`FlowKey(source, tag, subflow)`（含 `==/!=/<`）。**明确不用于 output_lock**
  （见 c-lock 节），**预留给 V5 subflow striping**。
- `Msg` 加 `int exit_port_`；`macros.h` `M_D_EXIT_PORT = 16`（总计 255 bit）；
  **编码 0=unpinned，合法端口=port_id+1**（8-bit 曾不足以容纳 port_id，见 §5）。
- **验证**：`-1/0/254/255` 等边界序列化往返自测。

### V1-c1 / V1-c2 控制 + 数据路由过 Link
- `ControlMsgNextHop(m, rid)`（REQUEST/ACK）与 `DataMsgNextHop(m, rid)`（DATA）在 router 的
  ctrl/data 路由点接入；跨 die 时 `SelectCoreMsgExit`（同 die→-1、多跳→抛、相邻→
  `CrossDieSelectExit`）+ `PinControlMsgExit` 把选定出口写进包头，data 复用同一钉死出口。
- **验证**：c1 后 REQUEST/ACK 真实穿链；c2 后 DATA 真实穿链（`[D2D_TYPE]` 分类型计数守恒）。

### V1-c3 生产准入 + 第一条完整闭环
- preflight `ValidateWorkloadStructure(j, chip_id, allow_adjacent_d2d)`：放行**相邻**且有精确
  **双向 peer link**（`HasD2DLink(a,b)&&HasD2DLink(b,a)`）的跨 die cast，多跳仍拒绝；生产
  dataflow 路径（`config_helper_core`）显式传 `true`。
- `[D2D_TYPE] request/ack/data in/out` 分类型统计（`d2d_link.cpp` 按 wire 类型计数）。
- workload `cross_die_2core.json`：core0(die0) → core16(die1) tag16。
- **验证（第一条闭环）**：`ns=398`、`typed=(1,1,1,1,4,4)`、`done={16:1}`、
  phases（Core0 SEND_REQ/RECV_ACK/SEND_DATA、Core16 RECV_DATA/SEND_DONE）、两次确定一致；
  生产负例：3×1 `die0→die2` 即使逐段物理 link 存在，多跳仍在**进仿真前**拒绝。

### V1-c drain 不变量
- `RouterUnit::residual()`（未释放 in/out lock ref + 各方向 data/ctrl buffer + host buffer）；
  `npusim` 遍历 SystemC 层级累加 `[DRAIN] router_residual`。仿真正常结束应为 **0**——
  非 0 = 锁泄漏 / 尾包丢失 / 别名致 ref 未归零。c3 e2e 断言 `drain=0`。

### V1-c aggregation-tag 锁语义（重要审查修正）
- **结论**：本工程 `tag == 接收核 recv_tag == 全局核 id`，即 tag 就是**接收端聚合槽**。故
  `output_lock` **有意按 tag 锁**（非缺陷）：多个源用同 tag 发一个核（**多发一**）本就该
  **共享锁、交错通过**，接收端按包内地址重组。**若误改成 `(source,tag)` 会把多发一错误拆成
  串行**（见 §5）。防别名的正确不变量是 **tag→dest 唯一性**（preflight：同 tag 指向不同接收核 →
  拒绝）。`flow.h`/`router.cpp` 就地注释固化此结论；FlowKey 留给 V5。
- **验证（判别性）**：instrument `g_max_output_lock_ref` 峰值——
  - **多发一**（core0/core1 同 tag16 → core16，`recv_cnt=2`）：`maxref>=2`（共享锁聚合）、
    `done={16:1}`、数据不串、`drain=0`；
  - **distinct-tag**（core0→16 tag16、core1→17 tag17 同端口）：`maxref==1`（各自独占串行）、
    `done={16:1,17:1}`；
  - 单 die：`maxref==1`。
- **验证（唯一性）**：自测正反例（同 tag 不同 dest 拒绝、同 tag 同 dest 允许、不同 tag 允许）。

### V1-d1 四方向相邻 die e2e
- 同一对 workload（正向 `cross_die_2core.json` die0→die1 / 反向 `cross_die_rev.json`
  die1→die0）分别跑在 **2×1（横向 E/W）** 与 **1×2（纵向 N/S）** 两个 C2C 布局，覆盖 die
  首跳四方向。新 hw 配置 `core_4x4_die1x2_c2c.json`。
- 每方向连续两次，均校验：REQUEST/ACK/DATA 类型计数（REQ 正向、ACK 反向都真实穿链）、
  agg 守恒、consumer 恰好 DONE 一次、**反向 ACK 落回 producer**（phases 的 RECV_ACK）、
  `link_sites==link_units==2`、无绑定错误、`mism==0`、`drain=0`、两次 sim-time/typed 确定一致。
- **验证**：E/W/N/S 均 `ns=398/398`、`typed=(1,1,1,1,4,4)`；E/N `done={16:1}`、W/S `done={0:1}`；
  拓扑负例：1×2 的 `side=N,dir=E` 以 `MVP requires C2C dir == side` 在仿真前拒绝。

### V1-d2 DATA 逐包完整性 + 消息大小边界
- 新 **DATA 完整性探针** `D2DDataProbe`（`port.h`）：capture(in)/delivery(out) 两侧各一份，
  仅对 DATA 型包累加——`pkts`、顺序敏感 `seqhash`（吸收 seq_id）、`csum`（吸收完整 256-bit
  payload）、`minseq/maxseq/endseq/end_count/end_length`、`inorder`（交付序严格 +1，base-agnostic）、
  `first/last_cycle`。`d2d_link.cpp` 在 in/out 两处 `ProbeData`；`D2DLinkUnit::residual()` +
  `[DRAIN] d2d_link_residual` 证明 Link FIFO 也归零。
- **判据**：`in==out` 的 `pkts/seqhash/csum` ⇒ 链路无丢/重/乱序/损坏（比「类型总数守恒」强）；
  canonical 形状 `maxseq-minseq+1==pkts`、`inorder==1`、唯一 `is_end` 且 `endseq==maxseq`、
  尾包 `length` 正确；`drain=0 && link_drain=0`。
- **验证**：`OC=16*N` 精确产生 N 个 DATA 包（int8、`noc_payload_per_cycle=4`）；覆盖
  **1/2/5(partial-tail 尾长 96)/7/8/9/32** 包（7/8/9 跨过 `buffer_depth=8`，但 V1 FIFO 仍是
  **无限功能队列，不赋予 V3 背压语义**，仅证尺寸不影响正确性）；完成时间严格单调递增。
- **注意（探针作用域）**：`g_d2d_data_*` 是**全局累加器**，seq/inorder 语义只对**单方向单条
  DATA 消息**良定义（d2 用例均为正向 E 单流满足）；双向/多流 DATA 会混叠，属后续 per-link 探针。

### V1-d3 latency 标定 —— V1 完成
- 生产端到端扫描 `link_latency=0/1/7/20`，每点两次。**先测量后固化**公式（不臆断 ΔT=ΔL）：
  完整事务有 **REQUEST 正向、ACK 反向、DATA 正向**三个严格因果串联的跨链阶段，`CYCLE=2ns`，故
  **`T(L) − T(0) = 3 · L · CYCLE = 6L ns`**。
- 分层校验：单个 Link 的 DATA `delivery−capture` 相对增量精确为 **`L cycle`**；DATA in/out
  **包间 span 恒定**（latency 不改稳态间距）；counts/seq/checksum（`data[:12]`）latency 无关。
- **验证**：`ns = 278/284/320/398`（Δ=6/42/120 = 3·L·2 精确）；per-packet 链路增量 = 0/1/7/20；
  span 恒为 6 cycles；两次确定一致。

## 4. 验证方法学

- **单 die 逐位一致**：每增量跑 noc_congestion 四场景，比 sim-time + 独立层级模块计数 +
  运行后 D2D 统计（口径同 V0，非 stdout byte-identical）。
- **分类型链路计数** `[D2D_TYPE]`：REQUEST/ACK/DATA 各自 capture==delivery 守恒。
- **两侧完整性指纹** `[D2D_DATA]`：in/out 的 pkts/seqhash/csum 相等 = 链路透明。
- **drain 双不变量**：`[DRAIN] router_residual` + `[DRAIN] d2d_link_residual` 结束态均为 0。
- **锁语义判别** `[LOCK] max_output_ref`：多发一 ≥2 vs distinct-tag ==1，判别 tag-only 是否被
  错误改成 per-source。
- **确定性**：跨 die 每组连续两次，断言 sim-time/typed/data 完全一致。
- **每方向两次 + 反向 ACK endpoint + 拓扑负例**：功能正确性 + 启动期拒绝一起验。
- 载体：`run_test_d2d_v0.py`（端到端 **48 组**）+ `--d2d-v0-selftest`（纯函数 **218 项**）+
  `--d2d-link-selftest`（Link SystemC **18 项**）+ `run_v0_exit.py`（三道阻塞门汇总）。

## 5. 关键教训 / 踩过的坑

1. **tag-only 锁是特性不是 bug（最重要的审查修正）**：一度把「output_lock 未迁移到 FlowKey」
   当缺口，深入分析发现迁到 `(source,tag)` 会**破坏多发一聚合**（tag=接收槽，同 tag 多源本该
   共享锁）。改为固化 tag-only + 补 tag→dest 唯一性护栏 + 补**判别性**的多发一测试
   （`maxref>=2`）——distinct-tag 测试并不能覆盖这个核心语义。
2. **8-bit 字段不够**：`exit_port_` 起初 8-bit，但 `port_id` 可超 254/255。改 16-bit +
   `0=unpinned/port_id+1` 编码 + port_id 容量护栏 + 边界往返测试。
3. **校验必须进生产路径**：`ValidateV1MvpTopology` 一度只在自测调用，没进 Monitor 的 Link
   实例化路径。改为在任何 deferred binding / Link 创建之前调用 + 补真实启动负例。
4. **不臆断端到端公式**：完整协议不能直接断言 `ΔT=ΔL`——REQ/ACK/DATA 三个因果串联跨链阶段，
   端到端增量是 `3·L`。应先测量建立公式再固化为断言（V1-d3 即如此）。
5. **DATA seq 是 1-based**：探针最初假设 `0..N-1`，实测生产从 **1** 开始。改成 base-agnostic
   连续区间判据（`maxseq-minseq+1==pkts`），不硬编码基址。
6. **测试要用对配置**：手动验证时用了默认 SIM/MAP，得到与 runner 不同的包数/时序；runner 用
   `noc_congestion/sim/sim_cycle.json` + `identity.spec`——排障时先对齐配置再下结论。
7. **措辞要精确**：「空闲 link 等同终结」不对（idle link 对上游 avail=true ≠ 终结 avail=false）；
   改为「无 C2C-bound 流量时 idle link 对 die-local workload 无可观测影响」。
8. **诚实报告**：「三次运行」等不准确表述被就地纠正；只声称实际验证口径。

## 6. 最终状态

- **V1 目标达成并多布局/多方向验证**：第一条真实跨 die 通信闭环（NoC→D2D Link→NoC，
  REQUEST→ACK→DATA），四方向（2×1 E/W、1×2 N/S）、逐包完整性、latency 标定全绿。
- **范围闭合**：相邻 die、每方向单端口、1 packet/cycle、固定 latency、功能性无限 FIFO。
- **最终测试数值**：纯函数/路由自测 **218/218**、Link SystemC 自测 **18/18**、
  D2D 端到端 runner **48/48**、noc_congestion 四场景**精确不变**
  （no_congestion 14781/29109、congestion 14833/45441）。
- **分支状态**：15 个 V1 提交在 `feat/d2d-v1`，已推送 `origin/feat/d2d-v1`；工作树 clean。
  V0 tag `d2d-v0-baseline` 保持不可变。
- **明确留待后续**：多跳跨 die → **V2**；有限缓冲 / 带宽 / 背压（credit / SAF）→ **V3**；
  多端口 / tensor 条带 / subflow（FlowKey 的 subflow 位）→ **V5**；behavioral 双档 min-cut
  记账、反向链路 BDP、全双工 tx/rx 物理参数等亦属后续。

## 7. 主要改动文件

| 层 | 文件 |
| --- | --- |
| 跨 die 路由 | `include/utils/router_utils.h`、`src/utils/router_utils.cpp`（DieFirstHopDir/CrossDieSelectExit/CrossDieStep/Control|DataMsgNextHop） |
| flow / 消息字段 | `include/common/flow.h`（新）、`include/common/msg.h`、`include/macros/macros.h`、`src/utils/msg_utils.cpp`（exit_port 16-bit 编解码） |
| D2D Link 单元 | `include/die/d2d_link.h`（新）、`src/die/d2d_link.cpp`（新，含 DATA 探针）、`src/die/v1_link_selftest.cpp`（新） |
| 端口 / 拓扑 / 探针 | `include/die/port.h`、`src/die/port_config.cpp`（ValidateV1MvpTopology、by_type 统计、D2DDataProbe）、`src/die/v0_selftest.cpp` |
| router egress / 锁 | `src/router/router.cpp`（Control|DataMsgNextHop 接入、tag-only 注释、residual、max_output_ref）、`include/router/router.h` |
| 生产准入 / 归一化 | `src/monitor/workload_normalize.{h,cpp}`（allow_adjacent_d2d + tag→dest 唯一性）、`src/monitor/config_helper_core.cpp` |
| 链路实例化 / dump | `src/monitor/monitor.cpp`、`llm/unittest/npusim.cpp`（[D2D_TYPE]/[D2D_DATA]/[DRAIN]×2/[LOCK]） |
| 测试 | `llm/test/d2d_link/`（`run_test_d2d_v0.py` 48 组 + `workload/cross_die_*.json` + `hardware/core_4x4_die{2x1,1x2}_c2c.json` + README） |

---
_下一阶段：V2（多跳跨 die）/ V3（有限缓冲 + 带宽 + 背压）——从固定延迟走向真实争用建模。_
