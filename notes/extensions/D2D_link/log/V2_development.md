# V2 开发总结（多跳跨 die：die 级 XY 接力与协议级活性）

> 记录 WaferAI-SIM D2D Link 建模 **V2** 的开发全过程：目标、增量顺序、每步的设计与验证、
> 途中发现并修复的 bug、以及最终状态。规划见 `../D2D_link_test.md` 与 `../D2D通信建模计划.md`，
> 测试说明见 `llm/test/d2d_link/README.md`，上一阶段见 `V1_development.md`。
>
> **一句话**：V1 打通了**相邻** die 的第一条真实跨 die 通信；V2 让包能穿过**任意多个中间 die**
> ——每进入一个 die 重新选出口、在中间 die 真实穿越其 NoC、并给出「经过哪几条 link、方向序列、
> 每包几跳」的精确证据，同时把协议级活性从「测试不卡住」升级为「仿真器主动诊断依赖环」。

## 1. 目标与约束

**目标**：支持任意规则 die mesh 上的多跳跨 die 通信（die 级维序 XY 接力），并验证
REQUEST/ACK/DATA 在受支持模式下持续推进。

**范围（刻意不变的部分）**：
- 继续沿用 V1 的**功能性无限 FIFO**，**不**引入有限缓冲/带宽/背压——那属 V3；
- 因此 V2 **不声称也未验证** buffer/网络死锁安全，只覆盖**协议级**等待；
- 仍是每方向单 C2C 端口、1 packet/cycle、固定 latency。

**铁律**（承袭 V0/V1）：单 die 时序逐位不变（NoC 四场景 14781/29109、14833/45441）；
一次一个纵向增量；诚实——区分「已验证」与「未验证」。

## 2. 增量顺序

```
V2-a   多跳路由纯函数（去掉仅相邻限制、每 die 重新 pin、落点 peer 映射）
  ↓
V2-b   中间 die 运行时转发（C2C 入口重写 pinned exit + repin 计数）
V2-b2  HOST lane 缺口修复 + 方向真改变的多跳证据（2×2 对角）
  ↓
V2-c   精确路径证据（逐条有向 link 归因 + 每 die 片内 hop）
  ↓
V2-d   延迟标定（多跳 latency 律、固定开销/可编程延迟分离）+ 多流
V2-d2  仿真器内部协议 watchdog + 已知依赖环诊断 + 拓扑/精度补齐  ← V2 完成
```

## 3. 各增量：做了什么 / 怎么验证

### V2-a 多跳路由纯函数
- 去掉 `SelectCoreMsgExit` 的「仅相邻」拒绝：源端只 pin **第一跳**。
- **核心语义**：`exit_port` 只对**当前 die** 有效。`CrossDieStep` 用 `DieOfGlobal(pos)` 现算
  `md`，并校验携带值是否仍等于 `DieFirstHopDir(md,dd)`——携带上一跳出口会被**拒绝**而非错误路由。
- 新增 `CrossDieIngressTile(die, port)`：由 `g_d2d_links` 精确 peer 元组给出跨 link 落点 tile，
  即中间 die 重新 pin 的锚点。
- **验证**：自测走完整多跳路径，断言 hop 数 == `DieManhattan`、每跨 link die 距离严格 −1、
  片内距离严格下降、egress 方向序列；直测生产包装 `SelectCoreMsgExit` 与 peer 映射精确性/非法输入。

### V2-b 中间 die 运行时转发
- `RouterUnit::RepinOnC2CIngress`：包跨 link 进入本 die 时在入口重写 pinned exit——目的在本 die
  清为 −1，否则按 `CrossDieSelectExit(入口 tile, des)` 重新 pin。数据与控制两条入口路径都接入，
  故 REQUEST/ACK/DATA 均可中继。入口判定 = `IsC2CEgressEdge(rid, dir)`。
- preflight 沿 die 级维序路径**逐跳**要求双向 peer link。
- **关键证据设计（避免巧合通过）**：3×1 直线相邻两 die 的 E 出口是**同一个模板 port id**，
  送达成功**无法**区分「已重新 pin」与「沿用旧值」。故新增 `[D2D_REPIN] total/changed/same`，
  断言 `same>0` 且 `total == 跨 link 包数`。实测两跳 `typed=(2,2,2,2,8,8)`、`repin=(12,6,6)`。

### V2-b2 HOST lane 缺口 + 方向真改变的证据
- **修复潜伏的 V1 dataflow 缺口**：config 与 START 早已走 `HostLaneOfCore`，唯独权重下发
  `fill_queue_data` 仍用 `config.id / GRID_X` 索引 `write_buffer`。该式只在「每 die lane 数
  == `GRID_Y`」的 legacy 布局下等于 lane；config 驱动 HOST 时（2×2 每 die 3 lane →
  `HOST_LANES=12`）die3 的 core48 算出 12，越界 `q[12]` **段错误**。此前配置恰好每 die 4 lane
  故未暴露。改走 `HostEnvelope + HostEnqueue`（`HostLaneOfCore` + 范围校验，非法即抛）。
- **副作用（预期、已核实）**：垂直挂载（S/N）的 HOST 用例时间变化——权重此前按**行**索引选 lane，
  而 S/N 的正确 lane 是**列**索引，故权重曾投到错误 lane。4×4 S `29123→29117`、4×4 N
  `29081→29063`、4×2 S `24921→24909`、4×2 N `24909→24891`；水平挂载与 NoC 冻结值不变。
- **内存**：权重逐条生成即逐条入队（新增单消息入口 `HostEnqueue`），避免 vector 与
  write_buffer 同时持有全部权重包导致峰值内存翻倍。
- **2×2 对角 e2e**：`die0-(E)->die1-(N)->die3`，方向**真的改变**，故 `repin=(12,12,0)`
  （`changed==total`、`same=0`）——与 3×1 直线的 `same=6` 互补，构成完整证据。
- `allow_adjacent_d2d` 改名 `allow_d2d`（现放行任意已连通多跳路径）。

### V2-c 精确路径证据
- 新增**逐条有向 link** 归因 `[D2D_LINK]`（`g_d2d_link_stats` 与 `g_d2d_links` 下标一一对应，
  每个 `D2DLinkUnit` 带 link 下标）与**每 die 活动** `[DIE_ACT]`。
- 断言：承载 DATA / REQUEST / ACK 的有向 link 集合**分别恰好**等于期望路径（方向与每条包数精确、
  `in==out`、active-set 用 `in or out`）、每包 hop 数 == 路径长度、入口重写数 == 总跨链包数、
  只有终点 DONE 且 ACK 源**恰好**是首尾、mismatch=0、drain=0。
- **发现并固化**：2×2 对角的**正反路径不对称**——die 级维序两个方向都先走 X，故正向
  `die0→die1→die3` 而 ACK 走 `die3→die2→die0`，往返构成矩形。
- **附带修复**：`DIRNAME` 的 N/E 与 `Directions` 枚举顺序不符，否则方向断言会对着错误标签通过。

### V2-d 延迟标定与多流
- **多跳 latency 律**：V1-d3 的单跳 `T(L)-T(0)=3*L*CYCLE` 推广为 **`3*H*L*CYCLE`**
  （REQUEST/ACK/DATA 三个因果串联阶段各跨 H 条 link）。两跳实测 `T2=332/344/416/572`
  （L=0/1/7/20），增量 `0/12/84/240` 与 `3*2*L*2` 逐点精确相等；各 L 下路径/包数/repin/mesh 不变。
- **固定开销与可编程延迟分离**：`T(H,L) = T_fixed(H) + 3*H*L*CYCLE`。对比 1 跳与 2 跳得
  **每多一跳的 L-independent 固定开销 = 54 ns**，而 `(T2(L)-T1(L)) - 54 = 0/6/42/120`
  精确等于 `3*L*CYCLE`。**措辞边界**：54 ns 是当前 workload/拓扑/端口配置下的固定成本，
  其中同时含中间 die NoC/router traversal、ingress re-pin、D2D 接口固定 pipeline 与两组
  实验端点位置差异；本测试**未**把这些分项拆开，故**不**称其为纯 NoC 开销。
- **多流**（仍无有限缓冲）：两条 2 跳流共享同一对 link，每条 link 计数**恰为单流两倍**，
  `repin=(24,12,12)`、两接收核各 DONE、drain=0。

### V2-d2 协议 watchdog 与依赖环诊断 —— V2 完成
- **动机（审查指出的冻结阻塞项）**：此前的「watchdog」只是 runner 的 subprocess timeout，
  能避免测试卡住，但**不能**区分协议依赖环 / 路由丢包 / 网络残留，拿不到等待状态，也无法在
  wall-clock 超时前由仿真器主动报告。故当时只能声称「合法用例在外部超时内完成」。
- **实现**：`ProtocolWatchdog`（`monitor/watchdog.{h,cpp}`）在**仿真内部**维护「最后一次协议
  进展时间」（进展 = router 入口收包 / link 搬运 / HOST 收到 DONE）；仍有未完成流量却连续超阈值
  （默认 20000 cycle）无进展时，dump `[PROTO_WAIT]`：`protocol_wait_cycle`、
  `last_progress_cycle`、`stalled_for`、router/link residual、各 router 持有的 output lock(tag)、
  各方向队首消息的 `(source, tag, dest, phase, wait_reason)`；随后 `sc_stop()`，npusim 以
  **退出码 3 主动结束**。Python `timeout=` 退居最后保险。
- **已知依赖环用例**：`cross_die_rendezvous_cycle.json`——core0 等 core16 的 tag0、core16 等
  core0 的 tag16。实测 `exit=3`、`stalled_for=20001`、`router_residual=0 d2d_link_residual=0`，
  并明确报告**「等待在原语/rendezvous 层而非网络层」**（residual 全 0 ⇒ 网络已排空）。
  这正是外部超时给不出的判别。健康的两跳多流运行 `exit=0`、无任何 `PROTO_WAIT`，不误触发。
- **拓扑补齐**：新增计划要求的 **1×3 纵向两跳**（正向 `N,N`、反向 `S,S`）。
- **精度补齐**：逐 die 片内 hop 数改为**精确期望**（3×1/1×3 `[18,18,0]`、2×2 对角
  `[52,15,3,9]`），不再只断言 >0。目的 die 为 0 是因入口 tile 恰为目的核本身（几何决定）。

## 4. 验证方法学（V2 新增）

- **逐条有向 link 归因**：全局 `[D2D_TYPE]` 只有总数，无法回答路径问题；`[D2D_LINK]` 才能
  精确断言「经过哪几条 link、方向、每条几包」。
- **片内 hop 与跨链到达分离**：`router_pkts` 含「跨 link 到达那一拍」，`>0` **不能**排除
  「入口 tile 恰为下一条 link 出口 tile、零片内 hop」；`mesh_pkts` 只统计同 die router→router
  输入，才是穿越 NoC 的直接证据。
- **排除数值巧合**：同模板 port id 下「重新 pin」与「沿用旧值」路由结果相同，必须靠
  `[D2D_REPIN] same/changed` 计数区分，不能靠端到端是否送达推断。
- **诊断优于超时**：活性问题由仿真器主动 dump 等待状态并非零退出；测试框架超时只作保险。

## 5. 关键教训 / 踩过的坑

1. **「能送达」不等于「实现正确」**：3×1 直线两 die 的 E 出口 port id 相同，即使运行时忘记
   重新 pin 也会碰巧正确。必须设计**判别性**证据（repin same/changed），并用**方向会改变**的
   2×2 对角作为互补用例。
2. **旧配置掩盖的缺口**：权重下发的 `id/GRID_X` 索引在「每 die 4 lane」下恰好等于 lane，
   直到 config 驱动 HOST 改变 lane 数才暴露成段错误。**恰好通过**不等于正确。
3. **先隔离再归因**：die3 段错误最初疑似多跳所致；用「die3 纯 die 内 workload、D2D 活动恒 0」
   与「legacy host 下 die3 正常」两个对照，才确认是 HOST lane 缺口而非 V2-b。
4. **不要过度归因数值**：`T2(0)-T1(0)=54 ns` 是 L-independent 固定开销，不等于纯 NoC 开销；
   未拆开的分项必须在措辞里如实标注。
5. **超时不是诊断**：外部 wall-clock 超时只能说明「卡住了」；要闭合活性验收，必须由仿真器
   维护进展状态、输出等待身份并主动失败。
6. **枚举顺序要核对**：`DIRNAME` 的 N/E 写反会让方向断言对着错误标签通过——测试自身也会说谎。

## 6. 最终状态

- **V2 范围闭合**：多跳接力、每 die 重新 pin、精确路径/方向/hop 证据、中间 die 片内 hop、
  延迟标定与归因、协议级活性与依赖环诊断。
- **最终测试数值**：纯函数/路由自测 **241/241**、Link SystemC 自测 **18/18**、
  D2D runner **62/62**、noc_congestion 四场景**精确不变**（14781/29109、14833/45441）。
- **明确留待后续**：有限缓冲 / 带宽 / 背压与由此产生的**网络 buffer 死锁安全** → **V3**
  （V2 既不声称也未验证）；Behavioral 双档与校准 → V4；多端口 / 条带 / subflow → V5。

## 7. 主要改动文件

| 层 | 文件 |
| --- | --- |
| 多跳路由 | `include/utils/router_utils.h`、`src/utils/router_utils.cpp`（CrossDieIngressTile、去相邻限制） |
| 中间 die 转发 | `src/router/router.cpp`、`include/router/router.h`（RepinOnC2CIngress、CountDieRouterPkt） |
| 统计/归因 | `include/die/port.h`、`src/die/port_config.cpp`、`src/die/d2d_link.cpp`（D2DLinkStat、die_mesh_pkts、repin 计数） |
| 协议 watchdog | `include/monitor/watchdog.h`、`src/monitor/watchdog.cpp`（新） |
| HOST lane 修复 | `src/monitor/config_helper_base.cpp`、`src/monitor/host_envelope.{h,cpp}` |
| 生产准入 | `src/monitor/workload_normalize.{h,cpp}`（逐跳 link 校验、allow_d2d） |
| dump | `llm/unittest/npusim.cpp`（[D2D_LINK]/[DIE_ACT]/[D2D_REPIN]/[PROTO_WAIT] 退出码 3） |
| 测试 | `llm/test/d2d_link/`（runner 62 组 + 1×3/2×2 配置 + 多跳/多流/rendezvous workload） |

---
_下一阶段：V3（有限缓冲 + 带宽 + 背压 + 网络死锁安全）——从固定延迟走向真实资源争用。_
