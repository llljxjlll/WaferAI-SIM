# DTE channel_count 对通信耗时影响的实验

> 代码：`channel_sweep.cpp`（直接驱动真实的 `DTEUnit`，链接项目自身编译出的 `dte_unit.cpp.o`，不是重新实现的模型）。理论：`theory.py`，从 `dte_unit.cpp` 的调度逻辑（`admitPending`/`chooseReadySlot`/`finishPortServices`）独立推导的闭式递推。运行：`build_and_run.sh` 生成 `results.csv`，`build_report.py` 生成本报告。

## 实验设置

单核在 t=0 时刻背靠背发起 **N=12** 个方向相同、大小相同的 `SPM_TO_REMOTE` 传输（模拟 `send_para_logic()` 连续发起一串 SEND_DATA 而不中间等待的场景）。固定：
- `dte_bit_width = 256` bit，单次传输 payload = 768 bit → 传输阶段 `T = ⌈payload/width⌉ = 3` cycle。
- `gamma_cycles=4`, `tau_launch_cycles=2` → 启动阶段 `L = gamma+tau = 6` cycle。
- `CYCLE = 2 ns`（`llm/include/macros/macros.h`）。
- 唯一变量：`dte_channel_count ∈ {1, 2, 3, 4, 6, 8, 12, 16}`。

## 结果：仿真 vs. 理论

| channel_count | 总耗时 makespan（仿真, cycle） | 理论 | 一致 | 总耗时（ns） | 相对 channel=1 的加速比 |
|---|---|---|---|---|---|
| 1 | 108 | 108 | ✅ | 216 | 1.00x |
| 2 | 57 | 57 | ✅ | 114 | 1.89x |
| 3 | 42 | 42 | ✅ | 84 | 2.57x |
| 4 | 42 | 42 | ✅ | 84 | 2.57x |
| 6 | 42 | 42 | ✅ | 84 | 2.57x |
| 8 | 42 | 42 | ✅ | 84 | 2.57x |
| 12 | 42 | 42 | ✅ | 84 | 2.57x |
| 16 | 42 | 42 | ✅ | 84 | 2.57x |

逐条传输的完成时刻（不止 makespan）也逐条核对，全部与理论精确一致 ✅（详见 `results.csv` 与 `theory.py` 的比对，`build_report.py` 内实现）。

## 分析

- **channel_count=1**：完全串行，makespan = N·(L+T) = 12·(6+3) = 108 cycle （216 ns）。每条传输必须等前一条彻底完成（含总线传输）才能开始下一条的启动延迟。
- **channel_count 增大**：多条传输可以同时处于 LAUNCHING/BUS_WAIT，启动延迟 L 被并行掉；但所有传输仍争用同一条 `LEGACY_BUS`（round-robin 仲裁，一次只有一条在真正传数据），所以吞吐的下限由总线决定：`makespan → L + N·T = 6+12×3 = 42` cycle。
- **饱和点**：本实验中 `channel_count=3` 时已经达到总线瓶颈（makespan=42 cycle），继续加到 4/6/8/12/16 个 channel **完全没有额外收益**——仿真数据证实了这一点（channel_count ≥ 3 的 makespan 全部相等）。 饱和所需的最小 channel 数可由 `⌈L/T⌉+1` 估算（本例 `⌈6/3⌉+1=3`），物理含义是：只要新准入的传输能在总线腾出空档之前完成自己的启动延迟，总线就不会因为等待启动而空闲。
- **结论**：在当前的 legacy（非 `fine_grained_resources`）DTE 模型下，`dte_channel_count` 只影响启动延迟能被流水线掩盖的程度，**不会带来真正的数据带宽并行**——因为所有 channel 共享同一条 `LEGACY_BUS`。增大 channel_count 的收益有一个由 `启动延迟/传输时间` 决定的硬上限，超过该点后再增加 channel 数量对通信耗时没有任何帮助，只会线性增加 DTE 的面积开销（`computeAreaUm2()` 中 `channel_count * channel_area_um2` 项）。 若要让 channel 数带来真正的带宽并行，需要打开 `fine_grained_resources`（V4）模型，让不同方向的传输占用不同的 SPM/AXI 端口而不是共享一条总线。
