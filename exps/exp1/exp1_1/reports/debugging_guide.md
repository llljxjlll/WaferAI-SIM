# exp1-1 调试经验与排障手册

更新日期：2026-08-26

## 1. 推荐排查顺序

遇到 unsupported、周期异常或图形异常时，按下面顺序检查：

1. logical shape 是否由模型定义正确生成；
2. inter-die shard 是否先于 tile padding；
3. rank-local N/K 是否按 512/256 对齐；
4. tile live set 是否低于 3 MiB；
5. 是否存在合法 `PM×PN×PK=16`；
6. compute、HBM、collective、local transport、reduction bytes 是否来自同一 runtime shape；
7. T00/T10/T01/T11 是否只改变调度，而没有改变工作量；
8. source 字段到底是 simulation、trace calibration 还是 analytical fallback；
9. 所有比值是否由最终整数 cycles 重新推导；
10. 最后再检查绘图排序、归一化和坐标轴。

先查 shape 和来源，通常比先盯着最终加速比更快定位问题。

## 2. TP 无法整除

### 现象

- rank-local shape 被整数除法截断；
- frontend 生成的 GEMM shape 与预期不同；
- 某些 6×6 case 无 candidate；
- 理论 FLOPs 与 simulator 工作量不一致。

### 错误做法

先对全局矩阵做大倍数 padding，或者为了 16-way Split-K 强制把 K pad 到
`TP×16×Kt`。这会在 36 dies 上放大无效工作。

### 修正

先分片，再对 rank-local 维度做最小 tile 对齐：AG 分 N，RS 分 K。全局 runtime
维度由对齐后的 rank-local shard 乘 D 恢复。logical/runtime 两套 shape 都必须输出。

### 检查字段

`logical_M/N/K`、`runtime_M/N/K`、`rank_N`、`rank_K`、`padding_M/N/K`、
`runtime_flops_over_logical_flops`。

## 3. 3 MiB SRAM 超限

### 现象

- lowering 尝试分配完整 weight/activation/partial；
- SRAM allocator 报容量不足；
- 大 seq_len 或大 hidden size 完全不能运行。

### 根因

问题不是模型本身一定无法运行，而是 lowering 粒度错误：完整矩阵被当成 SRAM
resident object，没有 GEMM tiling、HBM streaming 和跨核分布。

### 修正

固定 SRAM-safe tile，显式建模 A/B 双缓冲、FP32 accumulator、reduce scratch 和
runtime reserve。当前 live set 1.625 MiB，仅占 3 MiB 的 54.17%。完整 shape 只
影响 tile/wave 数量，不影响单 tile live set。

### 检查字段

`tile_M/N/K`、`tile_live_bytes`、`sram_capacity_bytes`、`Qfull`、`intra_k_waves`。

## 4. “16 核”不等于固定 16-way Split-K

### 现象

- K 较小或 rank_K=256 时仍出现 PK=16；
- reduction bytes 极大；
- 36 dies 上出现 `36 × 16-way Split-K` 的错误解释；
- M/N 方向有大量 tile，却没有被用来做空间并行。

### 根因

把“每 die 有 16 核”错误等同为“每个输出 tile 的 K 维固定切 16 份”。16 核应在
M、N、K 三个方向间自适应分配。

### 修正

枚举 `PM×PN×PK=16`，要求 PM≤Tm、PN≤Tn、PK≤rank_K/Kt。候选代价必须包含
空间尾波、A/B 广播、FP32 partial reduction 和 HBM/compute/local transport
关键路径。当前实际选择了四种模式，而不是固定模式。

## 5. Swizzling 或 production lowering 不支持完整 shape

### 现象

- Wang-1D candidate 缺失；
- frontend shape mismatch；
- production projection 能识别 fused op，但 lowering/linker 无法处理完整矩阵；
- 编译 DAG 过大，耗时远超实验本身。

### 处理原则

- production adapter 保留，用于 smoke 和代表点周期精确验证；
- 完整实验使用固定 Q=8 tile motif/解析回放，不为 176 个完整 shape 重复编译；
- planner 对 tile shape 仍无候选时，允许 topology 合法的 canonical fallback，
  但必须输出 `schedule_source`；
- 缺 opcode、拓扑不合法或不能 drain 时应输出 algorithm_unavailable，不能伪造周期。

## 6. 理论上限异常偏高

### 现象

- 实际/理论比值远低于 GPU 论文的合理区间；
- 理论只包含 `max(Tcomp,Tcomm)`，实际却包含 reduction/HBM/local NoC；
- padding 后实际工作量变大，但理论仍使用 logical shape。

### 根因

理论和实际描述的架构行为、shape 或数据搬运范围不同。

### 修正

使用 architecture-aware upper：理论包含 16 核 peak compute、HBM、选定
PM/PN/PK 的 local transport、reduction 和一个边界 wave；理论与 replay 使用同一
padded runtime shape。旧 algorithmic upper 只保留作辅助字段。

### 必须保持的断言

```text
actual_speedup <= theory_speedup
theory runtime shape == replay runtime shape
logical FLOPs 只用于报告有效吞吐
```

## 7. 6×6 达成率曲线过直

### 现象

AG+GEMM 和 GEMM+RS 的 `actual/theory` 几乎全部等于 82%。

### 根因

旧公式为：

```text
core_schedule_efficiency = 0.76 + 0.06 × min(1, Qfull/256)
```

当 6×6 case 的 Qfull≥256 后，所有 case 都硬饱和在 0.82；同时 overlap 中的
`min(stage)/Qfull` 趋近于零，拓扑拥塞又因不在关键路径而被隐藏。

### 修正

- wave fill 改为连续的 `Qfull/(Qfull+256)`；
- 显式加入 Split-K、PM/PN fanout 和 spatial utilization penalty；
- overlap 保留随 stage balance 和 mesh span 变化的 contention tail。

修正后 6×6 AG 达成率为 77.69%–80.87%，GEMM+RS 为 80.41%–81.59%。RS
仍较平滑，因为其多数 case 确实选择同一个 2×8×1 调度，这属于模型结果而非绘图错误。

## 8. AG+GEMM 与 GEMM+RS 不应强制同一种编排

AG 路径和 RS 路径的分片维度、collective payload、广播方向与 reduction 时机不同。
如果两者固定使用相同 Split-K 或相同 PM/PN，会出现：

- AG+GEMM 普遍被高估或低估；
- LLaMA-3/Mixtral Attention 的 GEMM+RS 异常偏低；
- DeepSeek-V3 MLP 的尾块和空间利用率惩罚被隐藏；
- 不同模型得到不合理的同一达成率。

调试时直接比较 `intra_pm/pn/pk`、`intra_spatial_utilization`、
`a_broadcast_bytes`、`b_broadcast_bytes` 和 `reduction_bytes`，不要只看 GEMM FLOPs。

## 9. 结果来源字段容易误导

### 风险

schema 为缺失字段设置默认值时，可能把 fallback 结果误标成
`cycle_accurate_simulation`。

### 规则

- 先看 `estimate_source`；
- 再看 `T11_source` 和 `T00_T10_T01_source`；
- `analytical_resource_replay_fallback` 必须按解析估算报告；
- `cycle_accurate_trace_calibrated` 只表示被 trace 锚定，不等于完整 shape 端到端仿真；
- retained artifact replay 只验证 simulator/drain 稳定性。

当前 176 个结果全部是 analytical fallback。论文或报告中必须保留这一说明。

## 10. 比值必须由周期派生

不要在估算器、适配器和绘图脚本中分别维护一份 speedup。标准定义为：

```text
inter_without_intra = T00/T10
inter_with_intra    = T01/T11
intra_without_inter = T00/T01
intra_with_inter    = T10/T11
total_speedup       = T00/T11
synergy             = T10*T01/(T00*T11)
attainment          = total_speedup/theory_speedup
```

`result_adapter.py` 会对已有比值做 fail-closed 一致性检查，避免 CSV 中残留旧公式结果。

## 11. 周期精确运行的 drain 检查

一次 npusim 返回零退出码并不够。至少检查：

- program I/O resolved/applied/verify 全部通过；
- DONE/ACK 数量匹配；
- hostlane mismatch=0；
- router residual=0；
- D2D link residual=0；
- data/control credit balanced；
- completion marker 与 makespan 对齐。

`calibration/replay_retained.py` 提供了 retained artifact 的检查范例。

## 12. 绘图调试经验

- 每张图独立按最高 TFLOP/s 归一化柱高，不能跨图比较绝对柱高；绝对值保留在 JSON；
- 深色段是 naive，浅色段是到 optimized T11 的增量，两段宽度必须相同；
- 两个 seq_len 用不同色系，不同模型复用相同色系；
- 右轴是 actual/theory 百分比，不是 speedup 倍数；
- Attention 和 MLP 折线应分开，不能跨层连线；
- SVG 应保留 data-role/data-layer，便于自动验证 22 根柱和两段折线；
- 曲线平滑时先检查原始 attainment span，再判断是否为坐标轴视觉压缩。

## 13. 最小诊断命令

```bash
# 检查 case 数、来源和状态
python3 -B -c 'import csv,collections; r=list(csv.DictReader(open("exps/exp1/exp1_1/results/results.csv"))); print(len(r)); print(collections.Counter(x["estimate_source"] for x in r)); print(collections.Counter(x["status"] for x in r))'

# 运行全部回归测试
PYTHONPATH=/workspace/exps/exp1/exp1_1:/workspace \
  python3 -B -m unittest discover -s exps/exp1/exp1_1/tests -v

# 只重新生成最终 8 张图
python3 -B exps/exp1/exp1_1/plot_dual_axis.py
```

## 14. 提交结果前检查清单

- [ ] 176 个 case，无重复 case_id；
- [ ] 所有成功记录有四个正周期数；
- [ ] T00>T10>T11 且 T00>T01>T11；
- [ ] tile live set≤3 MiB；
- [ ] runtime shape 不小于 logical shape；
- [ ] 理论与实际使用同一 runtime shape；
- [ ] actual speedup≤theory speedup；
- [ ] source/status 与实际方法一致；
- [ ] 8 张双轴图，每张 22 根柱、两段折线；
- [ ] 测试全部通过；
- [ ] 报告明确说明 MoE 是 single-expert。

