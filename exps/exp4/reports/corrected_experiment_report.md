# Exp4 软硬件协同优化实验更正报告

## 结论与证据等级

此前的 Exp4 结果作废。旧实现把 Exp-2 的 `resource_service_cycles` 当成可按候选带宽缩放的
byte/FLOP 计数；这些字段实际是 action duration 在其占用资源上的重复记账。Prefill 还误用
训练 forward 账本做代理，D2D 也把 DTE 限速时间按 1 TB/s 反推 bytes，因而得到
`sw+hw < sw_only` 的错误图形。

更正实现从 Exp-2 的真实 action DAG 重建 36 个工作负载的 FLOPs、runtime work、HBM/SRAM/
NoC/DTE/D2D bytes、route 和 naive/sw_opt 依赖最长路，再对 383 个候选逐资源重算。旧的
`duration-ledger-scaling-v1` 已在 manifest 中显式标记为失效，标准 `run_experiment.py` 入口也
已切换到新模型。

更正后的解析 sweep 已通过自动门禁，但本次 build 的 168 个短周期精确 shadow anchor 仍未
执行，完成数为 0。因此当前结论的证据等级仍是
`resource_explicit_analytical_extrapolation`，不能标为周期精确校准后的最终发布结果。

## 更正后的规模与门禁

- 候选硬件 383 个，primitive workload 36 个，软件状态 2 个。
- 生成 27,576 条 primitive 结果和 9,192 条请求级派生结果。
- 11,964 条为容量/拓扑可行的解析结果，15,612 条为显式标记的不可行性能投影。
- 72 个 `H_exp2 × {naive, sw_opt}` 参考状态全部精确复现，失败 0 个。
- 非有限/非正周期 0 个，理论下界违规 0 个。
- 48 条 `sw_opt_only` 控制组均连接回 Exp-2 的原始 `result_digest`。
- 383 个候选均满足 500 MHz、`P_core=2*N_PE*f`、`DTE_channel=ceil(B/128)` 和
  `D2D_edge=min(d_d2d*B,512)` 语义。
- 13 项 Exp4 单元测试通过；标准入口的单候选 smoke test 产生 72 条正确模型结果。
- 两张 SVG 均有白色背景。

规划明确要求不使用正收益 clamp。候选上共有 3,450/13,788 对 naive/sw_opt 出现软件调度
变慢，最小 speedup 为 0.7633；这些点被保留而没有改写为 1。Figure 1 使用各 shape 上的
最优硬件，因此它不等同于任意固定候选都必须从软件优化获益。

## Figure 1：三类优化收益

共同分母均为候选集外的 `H_exp2 + naive`。下表给出六模型几何平均点的范围：

|负载|sw_opt_only|hw_opt_only|sw+hw|
|---|---:|---:|---:|
|training|1.039--1.071x|2.000--2.044x|2.365--3.127x|
|prefill|1.116--1.227x|2.044--2.627x|2.767--3.679x|
|decode|1.000--1.025x|1.596--1.995x|1.732--1.997x|

更正后六个 prefill 模型全部满足 `sw+hw > sw_only`。这也说明先前把异常归因于
`H_exp2` 的 1 TB/s D2D 是错误解释：外部基准不同可以影响倍率，但不能修复错误的工作量
账本和资源换算。

其中 training 的 GPT-3-175B、LLaMA-3.1-405B、DeepSeek-V3，以及全部 decode 点使用容量
不可行投影；图中以空心菱形标记。Prefill 的 18 个 Figure 1 点均由容量可行候选得到。

## 硬件选择与 Pareto

严格可行的训练—请求 Pareto 仍为空，因为主请求指标要求同一候选同时覆盖 B64 和 B512，
而 B512 的容量可行请求行数为 0。因此 `hardware_pareto.svg` 只展示明确标注的 capacity-only
性能投影，不能解释为可部署的严格 Pareto 前沿。

六模型平均投影的训练最优与 balanced 最优均为 `cand-ecb228e2d444c8e4`：

~~~text
n_ctrl=2, router=broadcast, B=384 GB/s, B_s=768 GB/s/core/direction,
K=2 MiB/core, N_PE=8192, N=8, DTE_channel=3,
d_d2d=2, D2D edge=512 GB/s, HBM=6x16 GB/module,
wafer=8x6 modules, P_core=8.192 TFLOP/s @ 500 MHz
~~~

六模型平均投影的 inference 最优为 `cand-eefdeafbc7962204`：

~~~text
n_ctrl=1, router=base, B=512 GB/s, B_s=1024 GB/s/core/direction,
K=4 MiB/core, N_PE=12288, N=4, DTE_channel=4,
d_d2d=2, D2D edge=512 GB/s, HBM=2x16 GB/module,
wafer=13x8 modules, P_core=12.288 TFLOP/s @ 500 MHz
~~~

六模型平均的 naive/sw_opt balanced argmax 没有切换；DeepSeek-V3 单模型的 balanced argmax
从 `cand-ecb228e2d444c8e4` 切换为 `cand-eefdeafbc7962204`。

## 当前不能越过的证据边界

`calibration/calibration_summary.json` 仍记录：planned 168、completed 0、
`current_build_direct_gate_passed=false`。这不是解析 sweep 的数值一致性失败，而是开发规划中
周期精确校准阶段尚未实现/执行。`results/validation_summary.json` 因此报告：

~~~text
analytical_sweep_gate_passed = true
cycle_anchor_gate_passed     = false
release_gate_passed          = false
~~~

所以当前可以用于检查趋势、筛选 shadow hardware 和安排后续锚点，但若论文或报告需要声称
“少量周期精确校准 + 大量解析外推已经完成”，还必须实现并跑完 96 个 motif 双跑与 72 个
holdout window 双跑，达到规划第 6.2 节的误差阈值后再发布。

## 可复现标识

- corrected run digest：`f5acfa2f90d91907cb4c809e84062de52556f092a0234cc4077d346d38eb495a`
- corrected Pareto digest：`1aa5daa55f4683da60c1066b6e6a049fd4e8d3e3d0d6637b12584d693841f035`
- 机器验证摘要：`results/validation_summary.json`

上述 digest 对应本次已生成的数据文件；若重新运行，以上述 JSON 内的新 digest 为准。
