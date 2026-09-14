# motivation_exp —— 跨 die 流量与片上运算流量的 NoC 争用

验证「跨 die 流量会和片上算子的通信在同一批 mesh 物理链路上竞争，从而吃掉
计算/通信重叠本该省下的时间」。实测结果见
[`motivation_exp_report.md`](motivation_exp_report.md)（由脚本自动生成）。

## 运行

需要先构建 `build/npusim`。然后在仓库根目录：

```bash
python3 exps/motivation_exp/run_experiment.py
```

脚本会生成全部 workload、每个配置连续跑两次并校验确定性与 drain-to-zero，
然后写出 `results/results.json`、`figures/normalized_time.svg` 和实验报告。
一次完整 sweep 约 30 组仿真，几十秒量级。

## 目录

```
exps/motivation_exp/
├── hardware/mesh_3die_4x4.json    # 3x1 dies，每 die 4x4 核，端口只在四个角
├── sim/cycle.json                 # use_beha_noc=false —— 周期精确 router
├── mapping/identity.spec          # 恒等映射（空文件）
├── gen_workloads.py               # 五种场景的 workload 生成器
├── run_experiment.py              # sweep 运行器 + 静态路径模型 + 出图 + 出报告
├── report_template.md             # 报告模板（<<占位符>> 由运行器回填）
├── workload/                      # 运行时生成的 workload（每次重跑覆盖）
├── results/results.json           # 全部原始指标
├── figures/normalized_time.svg    # 归一化时间柱状图
└── motivation_exp_report.md       # 实验报告
```

## 设计要点

**拓扑。** 3×1 dies，每 die 4×4 核；`die_ports` 只在四个角上开：
`W(0,0)`、`W(0,3)` 作为入口，`E(3,0)`、`E(3,3)` 作为出口。被观测的是中间的
**die1**（全局核 16..31）。

**跨 die 流量是「过境」的。** 两条 flow 由 die0 发出、die2 接收，路由上必须
穿过 die1：从西侧角端口进来 → 沿本行走 3 跳 mesh → 从东侧角端口出去。
die1 的核完全不参与这两条 flow，所以它和 die1 的片上任务在时间上天然并行，
唯一的相遇点就是 mesh 链路。这一点是整个实验能干净归因的关键 —— 如果让
die1 的核自己去发跨 die 数据，核内 worklist 是串行的，测到的就变成人为串行
而不是链路争用了。

**片上负载。** die1 的 16 个核各做一个 `Matmul_f` 分片（二维切分 GEMM 的
本核部分），然后做一次 4-rank 的一维 AllGather：

* **x 方向**：4 个组 = 4 行，组内通信只走东西向链路 → 与过境流量在第 0/3 行
  相交（拥塞重叠）；
* **y 方向**：4 个组 = 4 列，组内通信只走南北向链路 → 与过境流量零相交
  （无拥塞重叠）。

两个方向的消息条数、每条大小、每核计算量逐项相同，只有物理链路不同。

**AllGather 为什么要「按 rank 分相」。** 核间传输是 `REQ → ACK → DATA` 的
阻塞握手，接收方必须先 post 接收才能回 ACK。如果一组里所有核同时发，就会
形成环形等待而死锁。展开成 4 相、第 p 相只有 rank p 发送、其余 3 个 rank 接收，
既无环也保持了 AllGather 的通信量和方向。

**计时口径：必须扣掉 host 配置前导。** 仿真开始时 host 有一道串行屏障 ——
给 workload 里声明的每个核发 CONFIG + WEIGHT 并收齐 ACK，之后才放行所有核。
这道屏障的长度只取决于声明了多少核（4 核的 `xdie` 是 142 ns，20 核的
`overlap_*` 是 366 ns），与它们要干什么无关。直接拿总时间相减会把这 224 ns
的常数差当成「重叠的额外开销」记到重叠场景头上。脚本从日志里读每次运行自己的
`Config helper start START data distribution` 时刻，一律用
`工作时间 = 总时间 − 本次前导` 做对比。

**为什么 host 注入要保持小。** `source` 注入和片上 flow 争同一个核的输入端口。
如果注入量大，靠近 HOST lane 的核会先拿到输入并开始发 AG 数据，占住下游核的
输入口，而下游核还在等自己的 START data —— 实测会真的死锁。脚本里
`SRC_BYTES=64`，注入相在几十 cycle 内结束，GEMM 的规模由 `OC` 单独控制。

**不能用内置 collective。** `noc.collective.enabled=true` 要求
`dte.use_beha_dte=true`，而 `use_beha_dte=true` 下跨 die 的 `SEND_DATA` 会挂在
DTE 上不推进（本仓库当前状态）。所以片上 AllGather 用普通 cast 手工展开。

## 改参数

`run_experiment.py` 顶部：

* `GEMM`：每核 GEMM 分片的 `(B, T, C, OC)`，决定计算相长度；
* `AG_BYTES`：AllGather 每条 point-to-point 消息的字节数（片上通信量）；
* `RATIOS`：inter-die / intra-die 流量比的 sweep 点；
* `SRC_BYTES`：host 每核注入字节数（见上，建议保持小）。

跨 die 每条流的大小由 `xdie_bytes_for(ratio)` 自动算出：
`总跨 die 字节 = ratio × 48 × AG_BYTES`，两条过境流平分。
