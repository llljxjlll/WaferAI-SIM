# D2D Link 测试 — V0（多 die 基础设施重构版）

对应 `notes/extensions/D2D_link/D2D_link_test.md` 的 V0，扩展逻辑依据
`notes/extensions/D2D_link/D2D通信建模计划.md`。

## 运行

```bash
python3 llm/test/d2d_link/run_test_d2d_v0.py     # 需先 build 出 build/npusim
# 或单独跑纯函数自测：
cd build && ./npusim --d2d-v0-selftest
```

## V5 开发进展（feat/d2d-v5）

- **V5-a ✔ 配置/编码/选择接缝**：`die_ports.c2c.multi_port=true` 必须显式给出
  `select_policy=nearest|banded_nearest|tag_hash|hybrid|dynamic`，`select_seed` 为固定非负整数；
  端口可声明 `link_group`（缺省 `-1` 表示独立物理链路），对端镜像的 group 必须一致。
- `Msg.subflow_` 使用 REQUEST/ACK/DATA 专属的 2-bit tagged-union，不加宽 256-bit wire；
  支持本版 `stripe=1/2/4`，旧消息 `subflow=0` 的编码保持为零。`Send_prim.stripe_count` 与
  `Recv_prim.stripe_count` 分别占原语 spare bits，旧配置的全零字段解码为 1。
- workload 的 `cast.stripe` 与 `work.recv_stripe` 缺省均为 1，非法值启动期拒绝；配置生成器已把
  stripe 同步到 REQUEST/RECV_ACK/SEND_DATA 和 grouped RECV。静态选择固定在
  `(source,tag,subflow,seed)`，中间 die re-pin 保留同一 key。
- V5-a 验证：纯函数 **305/305**，V4 聚合门 `AGGREGATE EXIT=0`，NoC 冻结值精确不变。
  V5-a 时 `dynamic` 只有选择算法接缝；active-flow 记账、尾包释放与运行时验收现已由 V5-f 补齐。
- **V5-b/c ✔ 多端口生产接线 + sender striping + grouped receive**：总包数按
  `q=F/k,r=F%k` 拆分，前 `r` 条子流各 `q+1` 包；REQUEST/ACK/DATA 都携带 subflow，
  每条子流固定一个出口，接收端收齐全部 ACK 和 `(source,subflow)` 尾包后才完成一次逻辑 flow。
- 新增 `[V5_SUBFLOW]` 按 `(link,source,tag,subflow)` 分桶，避免旧单序列探针把合法交织误判
  为乱序；逐子流核对 in/out、顺序 hash、完整 payload checksum、连续 seq 和唯一尾包。
- 四端口一跳生产测试 `F=7`：k=1/2/4 配额分别为 `[7]`、`[4,3]`、`[2,2,2,1]`，
  k=4 精确使用四条 link；两次运行选择/统计一致，router/link/SAF 全部 drain-to-zero；
  `F<k` 在 DATA 注入前明确拒绝。V5 runner 当前 **7/7**，V4 runner **13/13**。
- Behavioral 代表包仍满足 `seq=1 && is_end=true`，既获取又释放 Router 锁；V4 cycle ledger、
  时序与 NoC Behavioral 回归因此保持原契约。
- **V5-d ✔ 条带化 whole-flow SAF 与共享 link group**：sender 在第一个 REQUEST 前一次性
  计算全部 subflow 配额和实际逐跳端口路径；先聚合检查每条物理 link 与每个共享 group 的总需求，
  任一容量不足均在修改账本和发 REQUEST 前拒绝，成功则统一提交。每个 subflow 尾包逐 link
  排空后成对释放 link/group 账本，结束时两者都必须为 0。
- `link_group` 不仅共享 admission 容量，也通过按有向 die-pair/group 建立的确定性 round-robin
  仲裁器共享一个 DATA cut；CTRL 独立、不被 DATA 限速。`F=31,k=4` 实测独立 group 692 ns、
  共享 group 702 ns，四个正向成员均出现仲裁等待 `19/9/17/16` 且配额 `[8,8,8,7]` 全部
  按序排空；容量负例在首个 REQUEST 前失败。V5 runner **12/12**，Link 自测 **37/37**。
- **V5-e ✔ Behavioral 多端口 min-cut**：每一 die-hop 分别求所选端口速率和、独立 link
  速率和；同一 `(local_die,remote_die,link_group)` 只计一次，最后与源/目的 NoC 的 `1/1`
  cut 取最小。禁止 `k*min(single-lane)` 式虚假加速。完整逻辑 flow 元数据只注册一次，由
  `subflow=0` 首链代表包消费并承担一次 `ceil(F/R_eff)` bulk 服务；其余代表包只加固定延迟。
- 独立 Python oracle 自行展开端口、静态选择、逐 hop 资源和共享 group，不读取 C++ 统计。
  `F=31,k=4,port=link=1/4`：独立四链 `R=1,S=31`，共享 group `R=1/4,S=124`；
  C++ ledger 分别精确为 `(1,31,31,21,52)` 与 `(1,31,124,21,145)`，完成时间
  572/742 ns 且重复运行一致。未消费元数据计入 D2D drain/watchdog。V5 runner **15/15**，
  V4 Behavioral **13/13**。

- **V5-f ✔ dynamic active-flow pin**：动态策略按 `(die,dir,source,tag,subflow)` 缓存一次选择，
  负载定义为物理端口上的活跃 flow 数；重复 REQUEST/DATA 复用 pin，不逐包换口。DATA 尾包和 ACK
  真正从 C2C egress 发出后分别释放正向/反向 pin；重复/缺失释放硬报错，active pin 同时计入
  drain/watchdog。多跳采用 REQUEST 控制面在每个 die 首次到达时惰性建立固定 pin，等价于在 DATA
  到来前完成逐 die 路径钉死，细化了早期计划中的“中间 die 不得逐包动态重选”约束。
- 生产证据：单流 `F=31,k=4` 为 selection/release=`8/8`；两条并发流为 `16/16`，每流四端口
  配额 `[8,8,8,7]`、负载最终全 0；3×1 两跳单流逐 die 建立/释放 `16/16` 个 pin，8 条正向
  DATA link 均按序完整，`typed=(8,8,8,8,62,62)`、repin=`78/78/0`。所有场景连续两次
  选择、统计和 sim-time 一致且无 watchdog。纯函数 **307/307**，V5 runner **21/21**，Link **37/37**。

- **V5-g ✔ 规模门与聚合门**：真实生产配置覆盖 1/2/4/8 dies，核层级为 16/32/64/128，
  有向 link units 为 0/8/24/56；多 die workload 均从 die0 发送到最远 die，逐 subflow-hop
  完整且 dynamic selection==release、全部 residual=0。冻结运行峰值 RSS 为
  135256/247808/467328/906624 KiB，wall time 为 0.138/0.190/0.402/0.929 s；门同时保留
  1.5 GiB、30 s 绝对上限和相对增长护栏。它是当前容器的回归 smoke bound，不是跨机器性能承诺。
- 统一入口 `python3 llm/test/run_v5_exit.py`：嵌套执行 V0–V4 冻结聚合门及 V5 **23/23**；
  纯函数 **307/307**、Link **37/37**、V3 **16/16**、V4 oracle **8/8**、V4 **13/13**，
  NoC 冻结值 **14781/29109、14833/45441**，最终 `AGGREGATE EXIT=0`。

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
- **V1-c1a ✔（能力就绪；当时生产 gate 保持关闭）**：preflight 可精确验证相邻 die 的实际双向
  `g_d2d_links`，拒绝无 link 与多跳；c1/c2 期间生产调用保持 `allow_d2d=false`，避免接受后
  挂死；该 gate 已在 c3 闭环后显式打开。
- **V1-c1b ✔（控制路由接线）**：串行/并行 REQUEST 与接收端返回 ACK 均在源核调用
  `PinControlMsgExit`，相邻跨 die 时只选一次出口并随包携带；Router 控制路径统一调用
  `ControlMsgNextHop`，在源 die 朝固定端口收敛，过 link 后在目标 die 退回片内 XY。本增量当时的证据为
  双向 REQUEST/ACK 逐跳 walk + 序列化 + Router 生产调用点；真实运行时证据见 c3。
- **V1-c2 ✔（DATA 路由接线；当时生产 gate 保持关闭）**：每条 `SEND_DATA` 原语在源核调用
  `SelectCoreMsgExit` 一次，并把同一 `exit_port_` 复制到该 flow 的所有 DATA 包；Router 数据路径统一调用
  `DataMsgNextHop`，在源 die 使用固定出口、进入目标 die 后退回片内 XY。尾包解锁复用该包本轮已经解析的
  `out`，不再用全局目的核重新做片内路由。覆盖 3 包同 pin/序列化、E→W 与 W→E、same-die 等价、
  缺 pin 与非 DATA 跨 die 拒绝。`Send_prim::deserialize` 同时先解析 `type` 再消费 DATA 字段，并清空
  运行态 pin，消除旧的未初始化读取与复用污染风险。
- **V1-c3 ✔（第一条真实跨 die 协议闭环）**：生产 dataflow preflight 显式放行“相邻且存在精确双向
  peer link”的 cast；最小 workload `core0(die0) → core16(die1)` 连续两次完成
  `SEND_REQ → RECV/ACK → 4-packet SEND_DATA → DONE`，sim-time 均为 **398 ns**。Link 新增按消息类型
  capture/delivery 统计，实测 REQUEST=`1/1`、ACK=`1/1`、DATA=`4/4`、总计=`6/6`，HOST mismatch=0、
  唯一 DONE=`{16:1}`、五个生产协议阶段日志齐全、无绑定错误。3×1 die0→die2 生产负例证明开 gate
  后多跳仍在仿真前拒绝。
- **V1-d1 ✔（四方向相邻 die）**：2×1 的 die0→die1 / die1→die0 覆盖 E/W，1×2 覆盖 N/S；
  每方向连续两次均为 **398/398 ns**，REQUEST/ACK/DATA=`1/1,1/1,4/4`、consumer 唯一 DONE、
  反向 ACK 回到 producer、HOST mismatch=0、Router/Link FIFO drain=`0/0`。非法
  `side=N,dir=E` 在仿真前以精确 `MVP requires C2C dir == side` 诊断拒绝。
- **V1-d2 ✔（逐包完整性与大小边界）**：新增 DATA capture/delivery 探针，比较包数、
  顺序敏感 seq hash 和完整 256-bit payload checksum，并检查 base-agnostic 连续序列、
  唯一尾包、尾包长度及 Link cycle span。覆盖 **1/2/5(partial tail)/7/8/9/32** 包；
  5 包用例尾长 96 bit，其余尾长 128 bit；全部 in/out 指纹一致、完成时间严格递增、
  Router/Link 均排空。7/8/9 只覆盖消息尺寸跨过配置的 `buffer_depth=8`，V1 FIFO 仍是
  无限功能队列，不赋予 V3 有限缓冲/背压语义。
- **V1-d3 ✔（latency 标定）**：生产端到端扫描 `L=0/1/7/20`，每点连续两次，完成时间
  分别为 **278/284/320/398 ns**。Link DATA 的 delivery-capture 相对增量精确为 `L cycle`，
  DATA span 恒为 6 cycles；完整协议因 REQUEST→ACK→DATA 三个因果串联跨链阶段，满足
  **`T(L)-T(0)=3*L*CYCLE=6L ns`**（`CYCLE=2 ns`），不是错误的 `ΔT=ΔL`。
- **V1 完成（tag `d2d-v1-baseline`）**：V1 的相邻 die、单端口、1 packet/cycle、固定 latency、
  功能性无限 FIFO 范围已闭合；多跳进入 V2，有限缓冲/带宽/背压进入 V3。

## V2（多跳跨 die）—— 完成并冻结

V2 继续沿用 V1 的**功能性无限 FIFO**，**不**引入有限缓冲/背压（属 V3），因此 V2 **不**声称
解决了 buffer deadlock。

- **V2-a ✔（多跳路由纯函数）**：去掉 `SelectCoreMsgExit` 的「仅相邻」拒绝——源端只 pin
  **第一跳**，每进入一个 die 再重新 pin。`exit_port` 只对**当前 die** 有效：`CrossDieStep`
  用 `DieOfGlobal(pos)` 现算 `md` 并校验携带值，携带上一跳出口会被拒（而非错误路由）。
  新增 `CrossDieIngressTile(die, port)` 由 `g_d2d_links` 精确 peer 元组给出落点 tile。
  自测走完整多跳路径（3×1 `E,E`；2×2 对角 `E` 然后 `N`，含两个反向），断言
  hop 数 == `DieManhattan`、每跨 link die 距离严格 −1、片内距离严格下降、egress 方向序列；
  并直测生产包装 `SelectCoreMsgExit`（== `CrossDieSelectExit`、只 pin 首跳、可进包头）
  与 peer 映射（== `remote_die/remote_port.tile`、反向回原 egress tile、非法/无 peer → −1）。
- **V2-b ✔（中间 die 运行时转发）**：`RouterUnit::RepinOnC2CIngress` 在 C2C 入口重写 pinned
  exit——目的在本 die 清为 −1，否则按 `CrossDieSelectExit(入口 tile, des)` 重新 pin；数据与
  控制两条入口路径都接入，故 REQUEST/ACK/DATA 均可中继。preflight 沿 die 级维序路径逐跳要求
  双向 peer link。**关键证据**：3×1 直线相邻两 die 的 E 出口是**同一个模板 port id**，送达成功
  无法区分「已重新 pin」与「沿用旧值」，故新增 `[D2D_REPIN] total/changed/same` 计数，
  断言 `same>0` 且 `total == 跨 link 包数`。3×1 两跳实测 `typed=(2,2,2,2,8,8)`、
  `repin=(12,6,6)`、仅终点 DONE、ACK 源 `{0,32}`（中间 die 未提前产生 ACK/DONE）、
  router 与 link 均 drain=0。
- **V2-b2 ✔（HOST lane 缺口修复 + 方向变化的多跳证据）**：
  - **修复潜伏的 V1 dataflow 缺口**：config 与 START 早已走 `HostLaneOfCore`，唯独权重下发
    `config_helper_base::fill_queue_data` 仍用 `config.id / GRID_X` 直接索引 `write_buffer`。
    该式只在「每 die lane 数 == `GRID_Y`」的 legacy 布局下才等于 lane；config 驱动 HOST 时
    （2×2 每 die 3 lane → `HOST_LANES=12`）die3 的 core48 算出 12，越界 `q[12]` **段错误**。
    此前所有配置恰好每 die 4 lane，故未暴露。现统一走 `HostEnvelope + LegacyHostEnqueue`
    （内部 `HostLaneOfCore` + lane 范围校验，非法 dest 抛异常而非静默越界）。
  - **die3 本地回归**：2×2 config 驱动 HOST、workload 只在 die3 内跑，**D2D 活动恒为 0**，
    正常完成、drain=0 —— 把 HOST lane 缺陷与多跳彻底解耦。
  - **2×2 对角多跳 e2e**：`die0 -(E)-> die1 -(N)-> die3`（反向 ACK `W` 然后 `S`）。方向**真的
    改变**，故每次入口重写都必须改值：实测 `typed=(2,2,2,2,8,8)`、**`repin=(12,12,0)`**
    （`changed==total`、`same=0`）、仅 core48 DONE、ACK 源 `{0,48}`、drain=0。这与 3×1 直线
    互补，构成「运行时确实没有沿用 stale exit」的完整证据。
  - **副作用（预期且已核实）**：垂直挂载（S/N）的 HOST 用例确定性时间变化——权重此前按**行**
    索引 `config.id / GRID_X` 选 lane，而 S/N 的正确 lane 是**列**索引，故权重曾被投到错误 lane；
    修复后 hop 距离改变。4×4 S `29123→29117`、4×4 N `29081→29063`、4×2 S `24921→24909`、
    4×2 N `24909→24891`；水平挂载（W/E）与 NoC 冻结值不变（行索引恰等于正确 lane）。
    详见上文「V1-pre 3c」小节的历史/当前对照表。
  - **内存**：权重下发**逐条生成即逐条入队**（新增单消息入口 `HostEnqueue`），不再先缓存整批
    信封再复制，避免 vector 与 write_buffer 同时持有全部权重包导致峰值内存接近翻倍。
  - `allow_adjacent_d2d` 改名 **`allow_d2d`**（它现在放行任意已连通的多跳路径，不只相邻）。
- **V2-c ✔（多跳端到端闭环：精确路径证据）**：新增**逐条有向 link** 归因 `[D2D_LINK]`
  （`g_d2d_link_stats` 与 `g_d2d_links` 下标一一对应）与**每 die 活动** `[DIE_ACT]`
  （`router_pkts` = 所有输入；`mesh_pkts` = **仅片内 router→router**）。全局 `[D2D_TYPE]`
  只有总数，无法回答「经过哪几条 link、方向序列、每包几跳」，故 runner 现精确断言：
  - 承载 DATA 的有向 link 集合**恰好**等于期望正向路径（方向与每条包数精确），REQUEST 也
    **恰好**只出现在该正向路径上，承载 ACK 的集合**恰好**等于期望反向路径；集合判定用
    `in or out`（避免漏掉 out-only 异常）；每条 link `in==out`；
  - 每包 hop 数：DATA 总跨链次数 == 包数 × hop 数；入口重写次数 == 总跨链包数；
  - 中间 die **`mesh_pkts>0`**——`router_pkts` 含「跨 link 到达那一拍」，单看它 >0 **不能**
    排除「入口 tile 恰好就是下一条 link 的出口 tile、零片内 hop」，故以 `mesh_pkts` 为准；
  - 只有终点 DONE、ACK 源**恰好**是首尾两端（`==` 而非 `⊆`）、`mismatch=0`、无绑定错误、
    router 与 link 均 drain=0；每例连续两次且 link/die 活动完全一致。
  - 覆盖：3×1 正向 `E,E`(ACK `W,W`)、3×1 **反向** `W,W`(ACK `E,E`)、2×2 对角 `E,N`(ACK `W,S`)。
  - **对角正反路径不对称**（已固化为期望行为）：die 级维序两个方向都先走 X，故正向
    `die0-E->die1-N->die3`，而 ACK 走 `die3-W->die2-S->die0`，往返构成矩形；其 `die_act`
    `[60,20,4,18]`/`mesh_pkts` `[52,15,3,9]` 显示 die1 承载正向、die2 只承载 ACK。
  - 附带修复：`DIRNAME` 的 N/E 与 `Directions` 枚举（`WEST=0,EAST=1,NORTH=2,SOUTH=3`）顺序不符，
    否则方向断言会对着错误标签通过。
- **V2-d ✔（延迟标定与活性验收）**：
  - **多跳 latency 律**：V1-d3 在单跳标定了 `T(L)-T(0)=3*L*CYCLE`；多跳时 REQUEST/ACK/DATA
    三个因果串联阶段各跨 `H` 条 link，故推广为 **`T(L)-T(0) = 3*H*L*CYCLE`**。两跳实测
    `T2 = 332/344/416/572 ns`（L=0/1/7/20），增量 `0/12/84/240` 与 `3*2*L*2` **逐点精确相等**。
    且各 L 下 link 路径/包数/repin/mesh 活动完全不变——latency 只平移固定延迟。
  - **固定开销与可编程延迟分离**（可归因，而非笼统「变慢」）：`T(H,L) = T_fixed(H) + 3*H*L*CYCLE`。
    对比 1 跳与 2 跳：`T1 = 278/284/320/398`、`T2 = 332/344/416/572`，得
    **每多一跳的 L-independent 固定开销 = `T2(0)-T1(0)` = 54 ns**，而
    `(T2(L)-T1(L)) - 54 = 0/6/42/120` **精确等于 `3*L*CYCLE`**（可编程 D2D latency 增量）。
    **措辞边界**：这 54 ns 是「当前 workload/拓扑/端口配置下每增加一跳的固定开销」，其中同时
    含中间 die 的 NoC/router traversal、ingress re-pin、D2D 接口固定 pipeline 以及两组实验
    端点位置差异；本测试**未**把这些分项进一步拆开，故**不**称其为纯 NoC 开销。
  - **watchdog / 活性**：V2 用功能性无限 FIFO，合法多跳模式必须始终推进到完成；16 次扫描 0 超时。
- **V2-d2 ✔（仿真器内部协议 watchdog + 依赖环诊断）**：Python 的 subprocess timeout 只能把
  「永久挂起」变成测试失败，无法区分协议依赖环 / 路由丢包 / 网络残留，也拿不到等待状态。
  新增 `ProtocolWatchdog`（`monitor/watchdog.{h,cpp}`）在**仿真内部**维护「最后一次协议进展
  时间」（进展 = router 入口收包 / link 搬运 / HOST 收到 DONE），仍有未完成流量却连续超阈值
  （默认 20000 cycle）无进展时，dump `[PROTO_WAIT]`：`protocol_wait_cycle`、
  `last_progress_cycle`、`stalled_for`、router/link residual、各 router 持有的 output lock
  （tag）与各方向队首消息的 `(source, tag, dest, phase, wait_reason)`，随后 `sc_stop()` 并由
  npusim **主动以退出码 3 结束**（不依赖测试框架的 124 超时）。
  - **已知依赖环用例**：`cross_die_rendezvous_cycle.json` 构造 core0 等 core16 的 tag0、
    core16 等 core0 的 tag16 的 rendezvous 环。实测 `exit=3`、`stalled_for=20001`、
    `router_residual=0 d2d_link_residual=0`，并明确报告
    **「等待发生在原语/rendezvous 层而非网络层」**——这正是外部 wall-clock 超时给不出的判别。
  - **不误伤**：健康的两跳多流运行 `exit=0`、无任何 `PROTO_WAIT` 输出。
  - Python 的 `timeout=` 仅保留为测试框架最后一道保险。
  - **多流**（仍**不**引入有限缓冲/背压，那属 V3）：两条 2 跳流（`core0→core32`、`core1→core33`，
    不同 tag）共享同一对 link，每条 link 计数**恰为单流两倍**（`req 2/2`、`data 8/8`、`ack 2/2`），
    `repin=(24,12,12)`、两接收核各 DONE 一次、中间 die `mesh_pkts=36>0`、drain=0。
- **当前验证 / V2 完成**：纯函数/路由自测 **241/241**、Link SystemC 自测 **18/18**、
  D2D runner **62/62**、NoC 冻结值 **14781/29109、14833/45441**。
  V2 的多跳接力、per-die 重新 pin、精确路径证据、延迟标定与协议级活性范围已闭合；
  **有限缓冲 / 带宽 / 背压与由此产生的网络死锁安全属 V3，V2 不声称也未验证**。

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

  > **⚠ 上面这组数值是 V1 冻结时(tag `d2d-v1-baseline`)的历史快照**,在该 tag 上仍然正确。
  > **V2-b2 修复 HOST weight lane 后,垂直挂载(S/N)的确定性时间发生了合理变化**——原因见下:
  > 权重下发过去用 `config.id / GRID_X`(**行**索引)选 lane,而 S/N 挂载的正确 lane 是**列**索引
  > (`HostLaneOfCore`)。故垂直挂载下权重此前被投递到**错误的 lane**(config/START 走
  > `HostLaneOfCore` 一直是对的,只有权重这一路残留旧式索引);改为统一入口后 lane 正确,hop 距离
  > 随之改变。水平挂载(W/E)行索引恰等于正确 lane,故**不变**。
  >
  > | 用例 | V1 冻结(历史) | V2-b2 修复后(当前) |
  > | --- | --- | --- |
  > | 4×4 W / E | 29109 / 29063 | 29109 / 29063(不变) |
  > | 4×4 S | 29123 | **29117** |
  > | 4×4 N | 29081 | **29063** |
  > | 4×2 W / E | 29109 / 29063 | 29109 / 29063(不变) |
  > | 4×2 S | 24921 | **24909** |
  > | 4×2 N | 24909 | **24891** |
  >
  > runner 只断言「同一配置两次运行一致 + 签名/per-lane/mismatch 等指标」,不钉死绝对 ns,
  > 因此该修复不会表现为回归失败;这里显式记录,避免读者误以为输出前后不一致是 bug。
  > NoC 四场景冻结值(14781/29109、14833/45441)**不受影响**——它们走真 legacy(无 `die_ports`)
  > 西边挂载,行索引与 `HostLaneOfCore` 恒等。

**矩形 4×2 die 内 mesh(8 核)**:独立小型 dataflow workload `workload/gemm_4x2.json`(取 gemm 前 8 核,
核 0–7 自成 4 对不引用他核;8 核全产 ACK)+ 配置 `hardware/core_4x2_ports_{w,s,e,n}host.json`
(`x=4,y=2` 单 die)。**独立 west 参考签名、期望核数 = `GRID_X*GRID_Y = 8`**;DONE 源 `{0,2,4,6}` 使
**行分布 `[2,2]`(2 lane) vs 列分布 `[2,0,2,0]`(4 lane)明显不同**,错误映射难碰巧通过。至少一个
水平挂载(W/E)+ 一个垂直挂载(S/N),四方向全覆盖;各方向两次一致(W/E 29109/29063、S/N 24921/24909
——**S/N 为 V1 冻结历史值,V2-b2 修 HOST weight lane 后为 24909/24891,见上方对照表**)。
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

## V3（有限缓冲、背压与 whole-flow SAF）—— 完成

> **最终状态**：`mode=bounded_saf` 已接入生产周期精确路径，不再被 Monitor gate 拒绝。
> `functional_v2` 仍是默认值并保持 V1/V2 冻结行为；有限缓冲模式采用明确的
> `safety=whole_flow_saf`，不声称实现 escape VC、chunked SAF 或多 lane。

### V3-a：配置与兼容契约

- 两种模式：默认 `functional_v2` 与显式 `bounded_saf`。两类字段严格分区，避免“配置被接受但忽略”。
- `bounded_saf` 必须给出 `safety=whole_flow_saf`、整数有理数 `port_rate/link_rate`，且
  `0 < rate <= 1 packet/cycle`。单信道无法表达 `rate>1`，因此明确拒绝；多 lane 留给 V5。
- 四种容量独立：`saf_buffer_depth`、`link_inflight_depth`、`rx_buffer_depth`、
  `ctrl_buffer_depth`。SAF 容量要求 `>= flow_packets(F)`；inflight 继续执行 V3-b credit 模型验证过的
  保守窗口下界 `ceil((2*max(L,1)+2)*link_rate)`。二者不可互相替代。
- `ValidateD2DTopology` 按模式校验；每方向仍限一个 C2C 端口。

### V3-b：独立 Link 有限 FIFO 与速率模型

- `D2DLinkUnit::forward_bounded()` 在 standalone SystemC 测试中验证有限 DATA/CTRL FIFO、显式信用、
  下游长 stall、FIFO 顺序、无丢/重、占用不越 depth 和最终信用恢复。
- token bucket 使用整数 token，支持 `1、1/2、1/4、2/3`；`2/3` 以 1/2 拍间隔交替且 token 守恒，
  不能用单一 modal gap 近似。
- `L=0/1/7` 与多种速率验证 `BDP-1` 欠速、`BDP` 达到目标速率。
- V3 冻结时 Link self-test 从 V2 的 18 项扩展到 **32/32**；这些细粒度 credit/BDP 扫描验证
  standalone `forward_bounded()`，与生产 whole-flow SAF 分支彼此隔离。

### V3-c：REQUEST 流大小与原子 admission

- 256-bit 消息不扩宽：REQUEST 以 tagged union 复用既有 24-bit roofline 字段携带
  `flow_packets`，范围 `1..2^24-1`；非 REQUEST 的原语义不变。
- dataflow 配置把 `SEND_REQ` 与后续同 `(dest,tag)` 的 `SEND_DATA` 配对，缺失、错配和不可编码均报错。
- `WholeFlowSafAdmission` 按 `FlowKey(source,tag,subflow)` 原子记账；无效大小、重复 active key、
  容量不足和未知释放均明确失败且不破坏账本。
- 已验证 `capacity=F`、`F-1`、`BDP<F`、并发预留守恒和完整释放。

### V3-d：生产 bounded SAF 数据路径

生产每条**有向** D2D Link 的 DATA 流水线为：

```text
source router
  -> whole-flow SAF stage
  -> port token bucket
  -> link token bucket + finite inflight/latency FIFO
  -> finite RX stage
  -> remote router / remote NoC
```

- REQUEST 注入前，沿确定性 die-level XY 路径对**每条有向 link**一次性预留 `F` 个 SAF 槽位。
  任一跳不足会回滚此前所有预留并在 REQUEST/DATA 注入前报错；不会出现只占部分路径后等待。
- 每跳 REQUEST 先记录 `F`；DATA 按 FlowKey 收齐并核对尾包/包数，只有完整 flow 才进入物理 link。
  每条 link 的 SAF stage 排空该 flow 后释放该跳预留；多跳最终 `reserved_packets=0`。
- DATA 与 CTRL 都使用真实回程信用。DATA 初始信用为 SAF 深度，包离开 SAF 时归还；CTRL 初始信用为
  `ctrl_buffer_depth`，控制包交付时归还。信用事件采用**翻转位**而非单周期高电平，连续交付不会合并。
  Router 对信用做 0/上界保护，结束时 `[CREDIT] data_balanced=1 ctrl_balanced=1`。
- `saf/inflight/rx/ctrl` 均有独立有限容量；RX 满会依次阻塞 inflight 到达和 SAF drain。控制网络独立，
  不消耗 DATA token。全双工由两个相反方向的 Link 单元提供，容量互不共享。
- whole-flow SAF 的依赖切断点在 SAF stage：flow 完整落地后源 NoC 锁已释放。因此远端拥塞传播为
  `remote NoC -> RX -> inflight -> SAF drain`，**不会要求已完成整流的源 flow 继续持锁等待**。
  新 flow 容量不足时在源端 admission 阶段拒绝，而非进入网络后形成 hold-and-wait。

#### Post-freeze 生产路径证据加固

评审指出 V3-b 的 BDP/credit 细粒度证据原本只执行 standalone `forward_bounded()`。当前 Link self-test
新增 5 项、达到 **37/37**，直接实例化 `whole_flow_saf=true` 的生产 `forward_bounded_saf()`：

- 真实 REQUEST 声明 `F=64`，完整 flow 到齐前不允许任何 DATA 离开 SAF；
- `L=3,rate=1` 时按保守公式得到 `inflight=8`，生产流水连续交付 64 包，输出跨度 63 cycle；
- toggle DATA/CTRL credit 对连续归还逐次边沿计数，两个方向均精确恢复 64 次 DATA / 1 次 CTRL；
- `SAF=F=64,inflight=2,RX=1` 加长下游停顿，峰值严格为 `64/2/1`，各级背压计数触发后按序排空；
- 使用真实 2×1 admission 账本，最终 FIFO、SAF expectation 与 `reserved_packets` 全为 0。

边界口径：standalone 的 `BDP-1/BDP` 扫描证明其 pulse-credit RTT 模型中的最小窗口；生产探针证明
同一公式给出的深度在生产编码上**足够维持配置速率**，不声称该保守深度是生产流水线的最小值。

### V3-e：拥塞、背压与安全证据

独立 runner `run_test_d2d_v3.py` 提供 **16/16** 组生产级门禁：

- **瓶颈解析基线**：128 包长流 goodput 在 1% 内等于
  `min(NoC=1, port_rate, link_rate)`：实测 1、0.501976、0.250493；增大非瓶颈不改变吞吐，
  改变当前瓶颈后瓶颈随之转移；port/link stall 分类只在对应限制器上触发。
- **共享与独立资源**：两流共享 link 时总包数守恒且都完成；独立 link 并行；相反方向全双工独立。
- **Local+D2D 混合拥塞**：源 die shared/disjoint 对照为 stall `11/0`；中间 die 对照为
  stall `7/0`，且 shared 的两跳 D2D 完成 cycle `397 > 380`，证明拥塞出现在实际共享的片内路径。
  固定配置、双次重复运行和自动报告见
  [`mixed_noc_congestion/`](mixed_noc_congestion/README.md)。
- **真实生产背压链**：中间 die 热点在 `rx=1、inflight=4` 下稳定触发
  `inflight_full=85、rx_full=95、inflight_stall=60、rx_stall=63、downstream_stall=63`，
  全部 flow 完成并排空；`source_stalls=0` 是 whole-flow SAF 已切断源侧依赖的预期，不是漏接背压。
- **SAF/multi-hop**：3×1 两跳的两个 forward SAF stage 均完整存 32 包并释放；2×2 双向对角覆盖
  E/N/W/S 与最小 `rx=1/ctrl=1`；四流固定置换在 `SAF=F=4` 下让 2×2 的八条有向 link
  各承载 4 DATA 包并最终排空。
- **容量边界**：`SAF=F` 完成，`F-1` 在 DATA 前拒绝；并发 overbook 明确拒绝且原子回滚。
- **控制浅缓冲**：`ctrl_depth=1` 下 8 个并发 flow 的 8 REQUEST、8 ACK、32 DATA 全部按序完成，
  DATA/CTRL 信用均恢复，避免 registered-ready 的一拍滞后造成越界。
- **结束态**：每个成功用例要求 router residual、link residual、SAF reservation 全为 0，
  DATA/CTRL 信用平衡，watchdog 不触发。

### V3 完成范围与边界

V3 已支持：dataflow 周期精确模式、每方向单 C2C 端口、`rate<=1`、多跳、全双工、有限
SAF/inflight/RX/CTRL、whole-flow SAF 安全契约、端口/链路限速、Local+D2D 混合拥塞和可归因统计。

不在 V3 范围：Behavioral 快速模型（V4）、多 lane/多端口聚合与 striping（V5）、escape VC、
chunked SAF、`rate>1`、非 dataflow 多 die，以及完整逐事务 trace/oracle。对同一 `(source,tag,subflow)`
的 active flow 重入会明确拒绝；当前协议不为 active key 分配额外 flow-instance 序号。

### V3 冻结准入门

统一命令：

```bash
python3 llm/test/run_v0_exit.py
```

必须同时满足：

- 纯函数/路由自测 **284/284**；
- Link SystemC 自测：冻结 tag 为 **32/32**；当前 post-freeze hardening 为 **37/37**；
- 历史 D2D runner **67/67**；
- V3 production runner **16/16**；
- NoC 四场景精确保持 **14781/29109、14833/45441**；
- `git diff --check` 干净。

## V4 开发状态

V4 从 `e1a8c02` 建立独立分支 `feat/d2d-v4`。V3 的 `d2d-v3-baseline` tag 保持不动；
冻结后新增的 mixed-NoC 与 production-SAF probe 作为 V4 的 cycle 校准基线。

### V4-a：Behavioral backend 配置接缝（完成）

- 缺省 `backend=cycle`，历史配置和全部 cycle 门精确不变；
- `backend=behavioral` 显式要求 `port_rate/link_rate/link_latency`；
- Behavioral 禁止 cycle-only mode、有限 buffer/SAF 字段和 legacy 字段，不允许接受后忽略；
- `rate>1` 仍拒绝；V4-a 尚未接运行时，合法配置在生产 topology gate 明确报错。

当前门：纯函数 **293/293**、Link **37/37**、历史 D2D **67/67**、V3 production
**16/16**，NoC **14781/29109、14833/45441**。

### V4-b：Behavioral oracle 与解析式（完成）

- C++ `die/behavioral.h` 与独立 Python `behavioral/oracle.py` 分别实现 topology/path 与
  fixed/service 分解；
- 单 flow `R=min(NoC cut=1,port_rate,link_rate)`；多跳采用 pipelined min-cut；
- `T_D2D transaction=3*H*link_latency+ceil(F/R)`；Router hops 由代表包实际穿越，
  estimate 不重复计费；
- Behavioral 明确无跨-flow争用/有限 FIFO/credit/SAF 语义。

门：C++ pure-function **300/300**，Python oracle **8/8**。详细公式见 `behavioral/README.md`。

### V4-c：Behavioral 单流运行时（完成）

- REQUEST/ACK/DATA 代表消息走真实 Router；源 die 首 link 对 DATA 一次计入 `S(F)`，
  每阶段每 hop 计 `L`，接收核不重复计 bulk；
- `[D2D_BEHA]` 保存逻辑包数和周期分解，`run_test_d2d_v4.py` 用独立 oracle 检查；
- 一跳 `F=4,L=7,R=1` 实测 wire 计数 `1/1/1`、账本 `4+21=25 cycle`、两次
  `286 ns`、drain=0；cycle NoC 误配被启动期拒绝。

门：V4 **4/4**、纯函数 **300/300**、Link **37/37**、V3 **16/16**，NoC 冻结值不变。

### V4-d：解析参数扫描（完成）

- `F=1/7/8/9/128`、`R=1,2/3,1/2,1/4`、`L=0/1/7/20`；
- 2×1 一跳与 3×1 两跳；fixed=`3HL`，pipelined bulk 只计一次；
- runtime/oracle 账本完全一致，并以真实 sim-time 验证
  `ΔT=CYCLE·ΔS(F)`、`ΔT=CYCLE·3HΔL`。

V4 runner：**9/9**。

### V4-e：Behavioral/cycle 校准（完成）

- 无争用 `F=128`：cycle goodput 对 `R=1,1/2,1/4` 在 oracle 1% 内；Behavioral
  `S(F)` 精确等于相同 oracle；
- mixed shared/disjoint：cycle stall=`11/0`、cross-flow 相差 36 cycle；Behavioral
  stall=`0/0`、service/fixed 相同、总时长仅差一个固定路由 cycle；
- 同一 D2D link 上两条 Behavioral flow 与单 flow 同时完成，证明 V4 不建模跨-flow争用。

V4 runner：**13/13**。需要资源争用、背压和死锁安全时必须使用 cycle backend。

### V4-f：冻结门（完成）

```bash
python3 llm/test/run_v4_exit.py
```

- 历史 D2D **67/67**（pure **300/300**、Link **37/37**）；
- V3 production **16/16**；
- Python oracle **8/8**；V4 **13/13**；
- NoC 精确 **14781/29109、14833/45441**；
- 聚合 `AGGREGATE EXIT=0`。

冻结 tag：`d2d-v4-baseline`。V4 保证的是无跨-flow争用的快速解析 D2D；有限资源、
背压、拥塞和死锁安全仍必须选 cycle backend。V4-a 段中的“尚未接运行时”是当时的阶段
快照，已由 V4-c 完成状态取代。

V4 冻结后评审收口：`run_v0_exit.py` 仅是保留旧 CI 的 V0–V3 兼容门；当时全量门为
`run_v4_exit.py`，V5 完成后当前全量门已升级为 `run_v5_exit.py`。
Behavioral C++/Python 的单 lane NoC cut=`1/1` 均标记 `TODO(V5)`：
未来多 lane/striping 必须按真实共享 cut 建模，不能简单按 lane 数放大单 lane 最小速率。
该收口不移动 `d2d-v4-baseline` tag。
