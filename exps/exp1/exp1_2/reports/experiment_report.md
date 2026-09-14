# 实验 1-2：MoE Dispatch/Combine 双算力实验报告

更新日期：2026-08-27

## 1. 结论摘要

实验成功生成 32 条 architecture 主结果、32 条 HBM-free 配对消融和 8 条 loaded-groups
敏感性，共 72 条解析结果；H128/H2000 各有一份。所有 assignment、FLOPs、padding、
D2D、local NoC、HBM owner/capacity 和 profile 配对检查均通过。

主要观察：

1. architecture 模式在两套 tensor profile 下都由四个 256 GB/s HBM stack 的 per-expert
   replay 主导，因此 H128 与 H2000 的 optimized throughput 范围相同；提升 tensor peak
   没有突破 memory roof。
2. HBM-free 消融暴露了算力敏感性：H2000 的 focus-group optimized throughput 达
   3.58–6.39 PFLOP/s，显著高于 H128 的 0.23–0.41 PFLOP/s，但这不是可部署硬件结果。
3. loaded noncompact 的 D2D max-link 是 isolated 的 3 倍，来自 9 个显式 group 的实际
   route 重叠；compact 为 1 倍。四 stack 同时服务 9 个 group 时，loaded T11 约为
   isolated 的 9 倍，表明主瓶颈是共享 HBM service。
4. 当前周期精确 smoke 只验证执行链路和资源释放，不覆盖目标 workload/hardware。
   所以下列数字是 analytical physical replay 的探索性结果，不是已校准的绝对预测。

## 2. 实验矩阵

主矩阵：

~~~text
2 operators × 2 placements × 2 models × 2 seq_len × 2 profiles
= 32 architecture cases
~~~

另有完全配对的 32 条 HBM-free 消融。loaded-groups 选择 DeepSeek-V3、S=36864，
覆盖 2 operators × 2 placements × 2 profiles，共 8 条。

| Model | H | Expert I | top-k | Routed experts |
|---|---:|---:|---:|---:|
| Mixtral-8×7B | 4096 | 14336 | 2 | 8 |
| DeepSeek-V3 | 7168 | 2048 | 8 | 256 |

seq_len 为 2304 和 36864，S 表示整个 EP=4 group 的 token 数。DeepSeek 仅统计 routed
expert，不含 shared expert。主路径采用确定性 balanced routing、无 capacity drop。

## 3. 资源模型

- H128：128 TFLOP/s tensor/die；
- H2000：2000 TFLOP/s tensor/die；
- 两者 vector 均暂用 60 TFLOP/s/die，状态为 provisional_vector_rate；
- 6×6 wafer，D2D 每条有向 link 1 TB/s，X-first route；
- 四个 16 GiB HBM stack，每 stack 256 GB/s；
- 16 compute cores/die，local NoC 每条有向 link 256 GB/s；
- 3 MiB SRAM/core，tile 为 128×512×256；
- source 为 analytical_physical_resource_replay 或其 HBM-free 变体。

T00 是 unfused inter + baseline grouped-intra，T11 是 fused inter + optimized
grouped-intra。actual speedup=T00/T11，没有强制大于 1。理论下界使用同口径 padded work
与 resource/precedence floor。

## 4. isolated 结果

| Profile | Mode | Actual speedup | Focus-group optimized TFLOP/s |
|---|---|---:|---:|
| H128 | architecture | 1.417×–1.933× | 58.978–131.061 |
| H2000 | architecture | 1.118×–1.638× | 58.978–131.061 |
| H128 | HBM-free | 1.303×–3.646× | 230.305–409.589 |
| H2000 | HBM-free | 5.710×–38.882× | 3577.058–6393.968 |

本轮 72 条结果中没有观察到负 speedup，但这不是测试前提；绘图和回归允许负收益。

### 4.1 architecture 的关键路径

H128 architecture 的主要 cycle 范围：

- tensor ideal：1.17M–33.82M；
- max-stack HBM service：5.73M–132.12M；
- HBM route：1.47M–33.82M；
- D2D communication ideal：0.007M–0.793M；
- T11：5.73M–132.14M。

H2000 tensor ideal 降为 0.075M–2.16M，但 HBM service 不变，T11 仍为
5.73M–132.14M。因此 H128/H2000 architecture 的 optimized throughput 完全落在同一
58.978–131.061 TFLOP/s 范围。这里的 conclusion 是“四 stack replay 限制了 tensor
scale-up”，而不是“两个 tensor profile 等价”。

architecture 的 theory attainment 四舍五入后接近 1，是因为理论 resource floor 与实际
T11 都被同一 max-stack HBM service 主导。它不提供外部准确性证据。

### 4.2 HBM-free 的含义

HBM-free 仅令 modeled_hbm_cycles=0，仍保存原 HBM bytes/cycles 供审计，workload、
placement 和 route 不变。H2000 的最高 38.882× speedup 很大，主要因为 baseline 与
optimized grouped schedule 在高 tensor peak、provisional 0.8 efficiency prior 下差异被
放大。没有目标 profile 的周期锚点或置信区间，不能把该数值当作可实现收益。

## 5. loaded-groups 敏感性

loaded 模式显式放置 9 个 EP group，并用相同 DeepSeek 长序列 assignment 为全部 group
生成 route。下表比较 loaded 与对应 isolated T11：

| Operator | Placement | loaded/isolated T11 | D2D max-link ratio |
|---|---|---:|---:|
| Dispatch | compact | 8.9993× | 1× |
| Dispatch | noncompact | 8.9993× | 3× |
| Combine | compact | 8.9999× | 1× |
| Combine | noncompact | 8.9999× | 3× |

H128/H2000 的 loaded T11 相同，因为两者都被共享 HBM service 主导。compact 九组占据
互不重叠的 2×2 区域，所以 D2D max-link 不增长；noncompact route 在 wafer 中央重叠，
max-link 增长到 3 倍。合并 HBM+D2D 的最忙链路相对 isolated，compact 为 6 倍，
noncompact 约为 3.03–3.05 倍；最终约 9 倍 latency 来自四个 stack 服务九组权重流量。

loaded 结果中的 optimized_tflops 是 focus group 的有效吞吐；scenario_optimized_tflops
是 9 个 group 的全场景总吞吐，避免将两个分子混淆。

## 6. 周期精确证据

当前 build 实际运行结果：

| Workload | Cycles | 用途 |
|---|---:|---|
| Flexible MoE inference 1×2，run 1/2 | 2727 / 2727 | 执行链路与重复性 |
| Flexible MoE train 1×2，run 1/2 | 3566 / 3566 | 执行链路与重复性 |
| GroupGEMM (1,8,32) | primitive 6，program 70 | isolated primitive smoke |

ProgramIO、drain、credit/residual 检查通过。fixture 是 H16/I32/top-1、每 rank 1 token，
且 HBM 为每 die 本地 1 MiB/8 GB/s，不能校准目标 H128/H2000 或四边缘 stack。旧
168-sample profile 绑定不同的 npusim SHA，当前 preflight 拒绝，故未用于 duration。

## 7. 结果可靠性与发布边界

已经闭合：

- assignment 与 per-expert padding；
- Dispatch/Combine FLOPs；
- local assignment 排除 D2D；
- D2D/NoC/HBM 逐有向链路；
- 四 stack 唯一 owner、地址、容量；
- loaded 9-group 真实 placement/route；
- H128/H2000 workload/bytes 配对；
- 11/11 本地回归。

尚未闭合：

- 目标 H128/H2000 的 unit microbenchmark；
- top-k=2/8 capped motif 和直接四状态反事实；
- replay median/p95 error 与 bootstrap CI；
- vector rate；
- skew/capacity/drop；
- inter wave 的完整 production resource 粒度。

因此实验完成的是“可复现的探索性物理资源回放”。在正式定量发布前，必须补齐上述周期
精确门禁；当前报告不使用“周期精确大 shape”或“已验证达到真实硬件误差阈值”等表述。

## 8. 产物

- results/{h128,h2000}/{architecture,hbm_free_compute_comm}/results.{csv,json}
- results/loaded_groups/{h128,h2000}/architecture/results.{csv,json}
- figures/{h128,h2000}/{architecture,hbm_free_compute_comm}/
- calibration/source_evidence.json
- reports/development_report.md
- tests/test_experiment.py 与 tests/test_physical_model.py
