# exp1-1 开发与结果报告

更新日期：2026-08-26

## 1. 实验目标

实验评估两类计算通信融合算子：

- AG+GEMM：AllGather 与 GEMM 融合；
- GEMM+RS：GEMM 与 ReduceScatter 融合。

实验同时考察两层优化：

- inter-die：collective 与 GEMM 的跨 tile 流水和重叠；
- intra-die：每 die 16 核的 `PM × PN × PK = 16` 自适应编排。

四种反事实调度用 `Txy` 表示：

| 结果 | Inter-die | Intra-die |
|---|---|---|
| T00 | naive | naive |
| T10 | optimized | naive |
| T01 | naive | optimized |
| T11 | optimized | optimized |

由四个周期数计算 inter/intra 单独收益、组合收益和协同系数，避免用不同工作量或
不同 padding 的结果相除。

## 2. 实验空间

- Mesh：`1×4`、`2×3`、`3×3`、`6×6`；
- 算子：AG+GEMM、GEMM+RS；
- 模型：LLaMA-2-7B、GPT-3-175B、LLaMA-3-8B、LLaMA-3.1-405B、
  Mixtral-8×7B 单专家、DeepSeek-V3 单路由专家；
- 层：Attention 与 MLP，DeepSeek-V3 不包含 Attention；
- 每个模型/层有两档 seq_len；
- 总计：`4 mesh × 2 operator × 22 = 176` 个 case。

固定硬件参数包括每 die 4×4 核、每核 3 MiB SRAM、8 TFLOP/s、256 GB/s
本地目标带宽和两个 DTE channel。完整定义见 `plans/plan.md`。

## 3. 最终实现结构

| 文件 | 职责 |
|---|---|
| `run_experiment.py` | 定义模型、矩阵、mesh、理论算法和 176 个逻辑 case |
| `runtime_adapter.py` | 接入 production frontend、lowering、finalizer、resolver 和 npusim |
| `estimate_trace_replay.py` | shape 规范化、16 核调度搜索、解析阶段模型和可选 trace 校准 |
| `result_adapter.py` | 统一结果 schema，并只从 T00/T10/T01/T11 推导所有比值 |
| `plot_results.py` | 生成基础性能和指标图 |
| `plot_dual_axis.py` | 生成 8 张最终层次化双轴图 |
| `calibration/` | 已保留周期精确证据、prior 提取和 retained artifact replay |
| `tests/` | shape、SRAM、调度、理论口径、结果 schema 和绘图回归测试 |

## 4. Unsupported case 的规范化

### 4.1 先分片，再对每个 die 独立 tiling

最初若先把全局矩阵 padding 到 TP 与 16 核 Split-K 的共同倍数，会在 36 dies 上
产生过度 padding。最终实现先按 inter-die 维度得到 rank-local shard，再只将本地
维度对齐到 tile：

```text
AG+GEMM: N 按 D 分片，rank_N 对齐 512；K 对齐 256
GEMM+RS: K 按 D 分片，rank_K 对齐 256；N 对齐 512
M: 对齐 128
```

理论和实际估算都使用同一份 padded runtime shape；logical FLOPs 只作为有效性能的
分子。这样同时解决 TP 不能整除和理论/实际口径不一致的问题。

### 4.2 SRAM-safe streaming tile

固定 tile 为 `128 × 512 × 256`，采用 FP16 输入、FP32 accumulator/reduction
scratch、A/B 双缓冲和 512 KiB runtime reserve。保守 live set 为：

```text
1,703,936 bytes = 1.625 MiB
SRAM capacity    = 3,145,728 bytes = 3 MiB
占用率           = 54.17%
```

因此不再把完整 weight、activation 或 partial sum 常驻 SRAM，而是通过 HBM
streaming 和 tile 生命周期复用处理完整矩阵。

HBM 流量现显式标记为 `fused_boundary_tile_replay_v1`，并与 exp1-2 共用同一规则：
AG/Dispatch 将网络到达的 activation 排除出 HBM，权重按 M tile 重放；RS/Combine
按 output-stationary 顺序计入 A/B tile 重放。新增 `hbm_bytes_per_die`、
`hbm_cycles` 和 `hbm_traffic_model` 字段。重建前后 176 个 case 的 T00/T10/T01/T11、
总加速和理论加速逐字段比较，漂移为 0。

### 4.3 每 die 16 核自适应编排

搜索所有合法 `PM × PN × PK = 16` 候选。代价同时包含：

- compute cycle 与空间利用率；
- A/B 广播；
- Split-K FP32 partial reduction；
- local NoC transport；
- HBM 与计算/transport 的关键路径。

当前 176 个 case 的选择分布为：

| PM×PN×PK | case 数 |
|---|---:|
| 2×8×1 | 94 |
| 2×4×2 | 38 |
| 4×2×2 | 22 |
| 4×1×4 | 22 |

这避免了所有算子固定使用 16-way Split-K，也把 reduction 成本纳入了核行为。

## 5. 两层性能模型

### 5.1 Architecture-aware upper

理论上限使用相同 runtime shape、相同 tile 数和相同 local transport：

```text
theory_naive = T_compute_single_core + T_HBM + T_local + T_comm
ideal_intra  = max(T_compute_16core_peak, T_HBM, T_local)
theory_upper = max(ideal_intra, T_comm) + min(ideal_intra, T_comm) / Qfull
theory_speedup = theory_naive / theory_upper
```

`Qfull = Tm × Tn`，最后一项是不可隐藏的边界 wave。这个 upper 比只写
`max(T_compute, T_comm)` 更接近 16 核 die 的真实行为，因为它包含 HBM、广播、
reduction 和本地 transport。

### 5.2 实际 fused 估算

实际路径额外计入：

- 分段 collective 效率；
- mesh 拓扑拥塞效率；
- wave fill；
- Split-K penalty；
- PM/PN 广播扇出；
- spatial tail；
- collective 与 intra stage 共享注入端口/SRAM port 的 contention tail。

早期模型在 `Qfull ≥ 256` 后把 `core_schedule_efficiency` 固定为 0.82，导致 6×6
两张图几乎是直线。最终改成随 `Qfull`、PM/PN/PK 和 spatial utilization 连续变化，
并保留非零 overlap contention tail，避免大 Qfull 被错误视作完全无争用。

### 5.3 周期精确校准接口

若提供匹配 `mesh/operator` 的 Q=8 trace 摘要，估算器用：

```text
T11 = Tfill + (Qfull - 1) × II × K_scale + Tdrain
```

并将 T00/T10/T01 的解析边际比例锚定到实测 T11。输入支持直接给出
`tfill_cycles/ii_cycles/tdrain_cycles`，也支持 tile completion marker。

当前默认运行没有传入这类校准文件，因此最终 176 个 case 均为解析 fallback。
`calibration/existing_calibration.json` 是来自已检入小规模仿真的 prior；
`retained_replay.json` 只证明 retained artifact 能稳定 drain，二者都不能冒充当前
8 个 mesh/operator 签名的直接校准。

## 6. 当前结果摘要

下表依次给出每组 22 个 case 的总加速比最小值、平均值、最大值，以及理论达成率
范围：

| Mesh | Operator | Total speedup min/avg/max | Actual/Theory range |
|---|---|---:|---:|
| 1×4 | AG+GEMM | 3.953 / 4.141 / 4.289 | 75.00%–81.53% |
| 1×4 | GEMM+RS | 3.428 / 3.600 / 4.050 | 77.39%–81.59% |
| 2×3 | AG+GEMM | 3.932 / 4.145 / 4.343 | 74.96%–81.40% |
| 2×3 | GEMM+RS | 3.449 / 3.668 / 4.061 | 77.33%–81.59% |
| 3×3 | AG+GEMM | 3.972 / 4.163 / 4.347 | 74.91%–81.27% |
| 3×3 | GEMM+RS | 2.406 / 3.613 / 4.146 | 77.25%–81.58% |
| 6×6 | AG+GEMM | 4.149 / 4.382 / 4.544 | 77.69%–80.87% |
| 6×6 | GEMM+RS | 2.509 / 3.613 / 4.115 | 80.41%–81.59% |

全局结果：

- 总加速比范围：2.406×–4.544×；
- architecture-aware upper 达成率：74.91%–81.59%；
- synergy：1.0032–1.1365，当前模型下所有 case 均为正协同；
- 176/176 case 状态为 `estimated_via_tiling_and_padding`；
- 176/176 `estimate_source` 为 `analytical_resource_replay_fallback`；
- 所有实际加速都不超过 architecture-aware upper。

GEMM+RS 的部分 case 较低，主要由矩阵形状、rank-local K、collective 输出量和
2×8×1 核内广播路径共同决定。AG+GEMM 与 GEMM+RS 不应强制使用同一种核内模式。

## 7. 绘图输出

最终双轴图共 8 张，每个 mesh/operator 一张，每张 22 根柱：

- 左轴柱高：每张图内按最高 TFLOP/s 归一化；
- 单根柱深色段：naive T00；
- 单根柱同色浅色段：从 naive 到 optimized T11 的增量；
- 两个 seq_len 使用蓝/绿色系；
- 右轴折线：`actual_speedup / theory_speedup`；
- Attention 和 MLP 使用两段独立折线，不跨层连接。

绘图数据保留 theoretical/actual speedup 原值和 attainment rate，避免只能从 SVG
反推数字。

## 8. 验证状态

当前 15 项单元测试全部通过，覆盖：

- 176 个 case 完整性与唯一性；
- shard-first padding；
- tile 对齐和 3 MiB SRAM 上限；
- 两类算子的 HBM tile-replay 公式；
- `PM×PN×PK=16` 以及 PK 不超过实际 K tile；
- T00/T10/T01/T11 顺序关系；
- theory 与 replay 使用相同 runtime shape；
- 实际加速不超过理论上限；
- 6×6 达成率随调度/工作量变化；
- 结果 schema 和由 cycle 派生的比值一致；
- 8 张图、22 根柱、Attn/MLP 两段折线及最终样式。

## 9. 复现与升级路径

默认解析回放命令见根目录 `README.md`。若获得当前硬件 Q=8 trace，可运行：

```bash
python3 -B exps/exp1/exp1_1/estimate_trace_replay.py \
  --calibration /path/to/current_q8_trace_summary.json
python3 -B exps/exp1/exp1_1/result_adapter.py
python3 -B exps/exp1/exp1_1/plot_results.py
python3 -B exps/exp1/exp1_1/plot_dual_axis.py
```

升级优先级：

1. 为 4 mesh × 2 operator 生成当前硬件的 staggered Q=8 trace；
2. 再生成 synchronized injection trace 校准 congestion factor；
3. 直接仿真 2–4 个 T00/T10/T01 代表点，量化 replay 误差；
4. 若误差超过 20%，按调度签名增加校准点，而不是回退到 176 次完整仿真。

## 10. 不能从当前结果推出的结论

- 不能声称 176 个完整模型都经过端到端周期精确仿真；
- 不能将 retained 小 case 的 6391 cycles 当作当前 exp1 的校准；
- 不能将解析 congestion factor 当作真实 router queue occupancy trace；
- 不能用本结果验证未建模的 runtime、host、跨层依赖或完整模型调度开销；
- MoE case 是 single-expert GEMM+collective，不代表多专家路由、负载不均和 all-to-all。
