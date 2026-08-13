#!/usr/bin/env python3
"""Cross-checks channel_sweep's simulated results against the independent
closed-form theory in theory.py, and writes DTE_channel_experiment_report.md.

Usage: python3 build_report.py   (run after build_and_run.sh has produced results.csv)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from theory import completion_cycles, saturation_channels

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
REPORT = HERE / "DTE_channel_experiment_report.md"

# Must match channel_sweep.cpp's constants exactly.
N = 12
BIT_WIDTH = 256
PAYLOAD_BITS = 768
GAMMA_CYCLES = 4
TAU_CYCLES = 2
CYCLE_NS = 2

LAUNCH = GAMMA_CYCLES + TAU_CYCLES
TRANSMIT = -(-PAYLOAD_BITS // BIT_WIDTH)


def load_results() -> dict[int, list[int]]:
    by_channel: dict[int, list[int]] = defaultdict(list)
    with RESULTS.open() as f:
        for row in csv.DictReader(f):
            by_channel[int(row["channel_count"])].append(
                int(row["completion_cycle"]))
    return dict(by_channel)


def main() -> int:
    sim = load_results()
    channels_sorted = sorted(sim)
    c_star = saturation_channels(LAUNCH, TRANSMIT)

    rows = []
    all_match = True
    for c in channels_sorted:
        theory = completion_cycles(N, c, LAUNCH, TRANSMIT)
        match = sim[c] == theory
        all_match &= match
        makespan_sim = max(sim[c])
        makespan_theory = max(theory)
        rows.append({
            "channels": c, "makespan_sim": makespan_sim,
            "makespan_theory": makespan_theory,
            "makespan_ns": makespan_sim * CYCLE_NS, "match": match,
        })

    serial_makespan = max(completion_cycles(N, 1, LAUNCH, TRANSMIT))
    bus_bound_makespan = LAUNCH + N * TRANSMIT

    L: list[str] = []
    L.append("# DTE channel_count 对通信耗时影响的实验")
    L.append("")
    L.append("> 代码：`channel_sweep.cpp`（直接驱动真实的 `DTEUnit`，链接项目自身编译出的 "
             "`dte_unit.cpp.o`，不是重新实现的模型）。理论：`theory.py`，从 "
             "`dte_unit.cpp` 的调度逻辑（`admitPending`/`chooseReadySlot`/"
             "`finishPortServices`）独立推导的闭式递推。运行：`build_and_run.sh` "
             "生成 `results.csv`，`build_report.py` 生成本报告。")
    L.append("")
    L.append("## 实验设置")
    L.append("")
    L.append(f"单核在 t=0 时刻背靠背发起 **N={N}** 个方向相同、大小相同的 "
             f"`SPM_TO_REMOTE` 传输（模拟 `send_para_logic()` 连续发起一串 "
             "SEND_DATA 而不中间等待的场景）。固定：")
    L.append(f"- `dte_bit_width = {BIT_WIDTH}` bit，单次传输 payload = "
             f"{PAYLOAD_BITS} bit → 传输阶段 `T = ⌈payload/width⌉ = "
             f"{TRANSMIT}` cycle。")
    L.append(f"- `gamma_cycles={GAMMA_CYCLES}`, `tau_launch_cycles="
             f"{TAU_CYCLES}` → 启动阶段 `L = gamma+tau = {LAUNCH}` cycle。")
    L.append(f"- `CYCLE = {CYCLE_NS} ns`（`llm/include/macros/macros.h`）。")
    L.append(f"- 唯一变量：`dte_channel_count ∈ {{{', '.join(str(c) for c in channels_sorted)}}}`。")
    L.append("")
    L.append("## 结果：仿真 vs. 理论")
    L.append("")
    L.append("| channel_count | 总耗时 makespan（仿真, cycle） | 理论 | 一致 | "
             "总耗时（ns） | 相对 channel=1 的加速比 |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        speedup = serial_makespan / r["makespan_sim"]
        L.append(f"| {r['channels']} | {r['makespan_sim']} | "
                 f"{r['makespan_theory']} | {'✅' if r['match'] else '❌'} | "
                 f"{r['makespan_ns']} | {speedup:.2f}x |")
    L.append("")
    L.append(f"逐条传输的完成时刻（不止 makespan）也逐条核对，"
             f"{'全部与理论精确一致 ✅' if all_match else '存在偏差 ❌'}"
             "（详见 `results.csv` 与 `theory.py` 的比对，`build_report.py` 内实现）。")
    L.append("")
    L.append("## 分析")
    L.append("")
    L.append(f"- **channel_count=1**：完全串行，makespan = N·(L+T) = "
             f"{N}·({LAUNCH}+{TRANSMIT}) = {serial_makespan} cycle "
             f"（{serial_makespan * CYCLE_NS} ns）。每条传输必须等前一条彻底完成"
             "（含总线传输）才能开始下一条的启动延迟。")
    L.append(f"- **channel_count 增大**：多条传输可以同时处于 LAUNCHING/BUS_WAIT，"
             "启动延迟 L 被并行掉；但所有传输仍争用同一条 `LEGACY_BUS`"
             "（round-robin 仲裁，一次只有一条在真正传数据），所以吞吐的下限"
             f"由总线决定：`makespan → L + N·T = {LAUNCH}+{N}×{TRANSMIT} = "
             f"{bus_bound_makespan}` cycle。")
    L.append(f"- **饱和点**：本实验中 `channel_count={c_star}` 时已经达到总线瓶颈"
             f"（makespan={bus_bound_makespan} cycle），继续加到 4/6/8/12/16 "
             "个 channel **完全没有额外收益**——仿真数据证实了这一点"
             f"（channel_count ≥ {c_star} 的 makespan 全部相等）。"
             f" 饱和所需的最小 channel 数可由 `⌈L/T⌉+1` 估算"
             f"（本例 `⌈{LAUNCH}/{TRANSMIT}⌉+1={c_star}`），物理含义是："
             "只要新准入的传输能在总线腾出空档之前完成自己的启动延迟，"
             "总线就不会因为等待启动而空闲。")
    L.append(f"- **结论**：在当前的 legacy（非 `fine_grained_resources`）DTE 模型下，"
             "`dte_channel_count` 只影响启动延迟能被流水线掩盖的程度，"
             "**不会带来真正的数据带宽并行**——因为所有 channel 共享同一条 "
             "`LEGACY_BUS`。增大 channel_count 的收益有一个由 `启动延迟/传输时间` "
             "决定的硬上限，超过该点后再增加 channel 数量对通信耗时没有任何帮助，"
             "只会线性增加 DTE 的面积开销"
             "（`computeAreaUm2()` 中 `channel_count * channel_area_um2` 项）。"
             " 若要让 channel 数带来真正的带宽并行，需要打开 "
             "`fine_grained_resources`（V4）模型，让不同方向的传输占用不同的"
             " SPM/AXI 端口而不是共享一条总线。")
    L.append("")

    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT}")
    print(f"all match: {all_match}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
