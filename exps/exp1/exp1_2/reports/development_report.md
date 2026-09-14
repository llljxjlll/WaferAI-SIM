# exp1-2 开发与执行报告

更新日期：2026-08-27

## 1. 执行状态

本轮已实现并运行双算力剖面的探索性实验：

| 结果集 | H128 | H2000 | 合计 |
|---|---:|---:|---:|
| isolated architecture | 16 | 16 | 32 |
| isolated HBM-free | 16 | 16 | 32 |
| loaded-groups architecture | 4 | 4 | 8 |
| 总计 | 36 | 36 | 72 |

72/72 条解析 case 成功生成。回归测试 11/11 通过，包含 assignment/FLOPs 守恒、
per-expert padding、remote-only D2D、local NoC K-loop、四 stack 容量与地址、双 profile
配对、loaded 9-group placement、图表负收益保留和周期烟测 provenance。

状态仍是 exploratory，而不是 formal quantitative release。H128/H2000 目标 hardware
binding 的 unit closure、16 个 capped motif、4 对直接反事实和 replay 误差门禁尚未完成；
结果文件因此统一标记 analytical_physical_resource_replay、
simulator_unit_closure=false 和 cycle_accurate_calibration_pending。

## 2. 实现内容

### 2.1 分 expert workload

run_experiment.py 保存完整整数 assignment 矩阵 A[source, expert]。每个 expert 独立执行
M 维 padding，再按 home rank 聚合工作量；不再把多个 expert 展平成一个普通 GEMM。

Dispatch 的 logical/runtime tensor work 同时包含 gate 和 up：

~~~text
F_dispatch = 4 × S × top-k × H × I
~~~

Combine 只包含 down GEMM：

~~~text
F_combine = 2 × S × top-k × H × I
~~~

SwiGLU 与 weighted combine 使用独立 vector work。DeepSeek-V3 输出标记
routed_expert_only。

### 2.2 物理 placement、D2D 与 loaded groups

physical_model.py 统一采用 6×6 row-major die id 和 (x,y) 坐标，route 为 X-first。
isolated 只放置中心 focus group；loaded 显式放置 9 个互不重叠的 EP group，并为全部
group 生成 assignment、remote flow 和 route。

D2D 指标来自所有 flow 的逐有向边聚合。compact loaded 的 group 位于互不重叠的 2×2
区域，D2D max-link 相对 isolated 为 1 倍；noncompact 的九组 route 真实重叠，max-link
为 3 倍。该差异是流量聚合结果，不是 contention 常数。

### 2.3 四个边缘 HBM stack

四个 16 GiB stack 固定在：

| stack | home die | coordinate | address range |
|---:|---:|---|---|
| 0 | 1 | (1,0) | [0,16 GiB) |
| 1 | 4 | (4,0) | [16,32 GiB) |
| 2 | 31 | (1,5) | [32,48 GiB) |
| 3 | 34 | (4,5) | [48,64 GiB) |

每个 expert 的 gate/up/down root 都有唯一 stack/address owner。Mixtral 每 stack 分配
704,643,072 bytes（4.10%）；DeepSeek 每 stack 分配 5,637,144,576 bytes（32.81%）。
所有 case capacity feasible，地址区间无重叠。

HBM read 从 stack home die 到 compute die 逐链路路由，再与 MoE all-to-all 的链路负载
合并。architecture 不再使用“每 die 独立本地 256 GB/s HBM”的旧抽象。

### 2.4 Grouped-GEMM 与 local NoC

调度搜索 PE×PM×PN×PK≤16，合法任务不足时允许 underfill。H128 architecture 实际
active cores 为 8 或 16；H2000 architecture 为 2 或 16，说明模型没有强制 16 核满载。

tile 为 128×512×256。Dispatch 的保守 live set 为 2,228,224 bytes，包含 gate/up 的第二
weight stream 和 512 KiB reserve，低于 3 MiB/core。A/B broadcast 对每个 output tile
遍历完整 Tk，Split-K reduction 单独计入 FP32 bytes，并在 4×4 core mesh 上逐有向链路
聚合。

### 2.5 四状态与绘图

结果保存 T00/T10/T01/T11，不对 speedup 做正值钳制。图使用 T00/T11 并列柱，测试用
人工负收益样本确认 T11 较慢时仍能正常显示。H128/H2000 和 HBM-free 均分别输出图，
不会混用 profile 或 memory mode。

当前 inter wave replay 仍把一个 wave 内已经解析出的 tensor/HBM/local-NoC 时间压缩为
复合 compute resource；它保留 operator dependency、DTE stage、候选 unroll、
double-buffer slot 和 physical route，但尚未达到 development_plan 要求的完整 production
resource 粒度。这是 formal gate 未通过的另一原因。

## 3. 周期精确复用结果

本轮复用了当前 release build 的现有 Flexible Mesh MoE 入口和 GroupGEMM artifact：

| 项目 | 结果 | 闭合情况 |
|---|---:|---|
| Flexible MoE inference 1×2，双跑 | 2727 / 2727 cycles | ProgramIO、residual、重复性通过 |
| Flexible MoE train 1×2，双跑 | 3566 / 3566 cycles | ProgramIO、residual、重复性通过 |
| GroupGEMM (1,8,32) | primitive 6、program 70 cycles | ProgramIO/drain 通过 |

证据保存在 calibration/source_evidence.json 及其引用文件。Flexible fixture 是
H16/I32/top-1、每 rank 1 token，使用每 die 本地 1 MiB/8 GB/s HBM；它既不是 H128/H2000，
也不是目标四边缘 stack 拓扑。因此只作为执行链路和固定结构 prior。

仓库中的旧 168-sample profile 虽然完整 measured，但绑定的 npusim SHA 与当前 binary
不同；当前 preflight 会拒绝。旧 profile 没有被静默复用为目标 duration。

## 4. 验证结果

执行命令：

~~~bash
PYTHONPATH=/workspace/exps/exp1/exp1_2:/workspace \
  python3 -B -m unittest discover \
  -s exps/exp1/exp1_2/tests -v
~~~

结果为 11 tests passed。另一个临时全量烟测使用 profile=all、memory-mode=all、
network-scenario=all，得到 72/72 successful；正式结果随后用相同参数写入 results/。

## 5. 尚未通过的正式门禁

- H128/H2000 各自的 tensor、vector、HBM/DTE、NoC unit closure；
- 2 operators × 2 placements × 2 payload regimes × 2 profiles 的 16 个目标 motif；
- 4 对直接 T_base/T_overlap 周期精确反事实；
- target top-k=2/8、64 local experts、skew/capacity 边界；
- leave-one-signature-out 的 median≤8%、p95≤15%、单点≤20%；
- speedup bootstrap 95% CI；
- inter action 的完整 production 资源拆分。

在这些门禁完成前，报告中的绝对 cycles、TFLOP/s、speedup 和 theory attainment 只应作为
同一解析模型内的探索性对比。architecture 的 attainment 接近 1 主要表示同源 HBM 下界
成为关键路径，不是对真实硬件误差的证明。
