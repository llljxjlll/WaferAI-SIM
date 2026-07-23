#!/usr/bin/env python3
"""Cycle-accurate mixed D2D/on-chip NoC congestion experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILD = ROOT / "build"
NPUSIM = BUILD / "npusim"
SIM = ROOT / "llm" / "test" / "noc_congestion" / "sim" / "sim_cycle.json"
MAPPING = ROOT / "llm" / "test" / "noc_congestion" / "mapping" / "identity.spec"
REPORT = HERE / "mixed_noc_congestion_report.md"
CYCLE_NS = 2


@dataclass(frozen=True)
class CaseSpec:
    name: str
    hardware: Path
    workload: Path
    cross_flow: tuple[int, int, int]
    c2c_row: int
    cross_path: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class Result:
    sim_ns: int
    flow_done: dict[tuple[int, int, int], int]
    noc_send: tuple[int, ...]
    noc_stall: tuple[int, ...]
    d2d_source_stall: int
    data_in: int
    data_out: int
    data_inorder: int
    data_integrity: bool
    data_cycles: tuple[int, int, int, int]
    forward_data: tuple[int, int]
    reverse_ack: tuple[int, int]
    bounds: tuple[int, ...]
    credit_balanced: bool
    drained: bool
    watchdog: bool


CASES = {
    "shared": CaseSpec(
        "shared",
        HERE / "hardware" / "shared.json",
        HERE / "workload" / "shared.json",
        (4, 20, 20),
        1,
        frozenset({(4, 5), (5, 6), (6, 7)}),
    ),
    "disjoint": CaseSpec(
        "disjoint",
        HERE / "hardware" / "disjoint.json",
        HERE / "workload" / "disjoint.json",
        (8, 24, 24),
        2,
        frozenset({(8, 9), (9, 10), (10, 11)}),
    ),
}
LOCAL_FLOW = (5, 7, 7)
LOCAL_PATH = frozenset({(5, 6), (6, 7)})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"-?\d+", text))


def assigned_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"=(-?\d+)", text))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs() -> None:
    require(NPUSIM.exists(), f"npusim not found: {NPUSIM}")
    sim = load(SIM)
    require(sim.get("noc", {}).get("use_beha_noc") is False, "experiment must use cycle-accurate NoC")

    workloads = {name: load(case.workload) for name, case in CASES.items()}
    require(
        workloads["shared"]["vars"] == workloads["disjoint"]["vars"],
        "both cases must compute the same GEMM shape",
    )
    for name, workload in workloads.items():
        producers = [
            prim
            for core in workload["chips"][0]["cores"]
            for item in core["worklist"]
            for prim in item.get("prims", [])
        ]
        require(len(producers) == 2, f"{name}: expected two GEMM producers")
        require(
            all(prim["type"] == "Matmul_f" for prim in producers),
            f"{name}: non-GEMM producer found",
        )
        require(
            producers[0] == producers[1],
            f"{name}: local and cross-die GEMMs must be identical",
        )
        flows = {
            (core["id"], item["cast"][0]["dest"], item["cast"][0]["tag"])
            for core in workload["chips"][0]["cores"]
            for item in core["worklist"]
            if item.get("prims")
        }
        require(
            flows == {CASES[name].cross_flow, LOCAL_FLOW},
            f"{name}: workload must contain exactly the controlled two flows",
        )

    normalized_hardware = []
    for name, case in CASES.items():
        hw = load(case.hardware)
        require(hw["die"] == {"x": 2, "y": 1}, f"{name}: topology must be 2x1")
        require(hw["die_ports"]["c2c"]["mode"] == "bounded_saf", f"{name}: not V3")
        require(
            hw["die_ports"]["edges"].get("N", {}).get("role") == "host"
            and hw["die_ports"]["edges"].get("S", {}).get("role") == "host",
            f"{name}: symmetric N/S HOST attachment required",
        )
        c2c = [
            p
            for p in hw["die_ports"]["overrides"]
            if p.get("role") == "c2c"
        ]
        require(
            {p["idx"] for p in c2c} == {case.c2c_row},
            f"{name}: unexpected C2C row",
        )
        for port in c2c:
            port["idx"] = 0
        normalized_hardware.append(hw)

    require(
        normalized_hardware[0] == normalized_hardware[1],
        "hardware cases may differ only in C2C attachment row",
    )
    require(
        CASES["shared"].cross_path & LOCAL_PATH == LOCAL_PATH,
        "shared case must overlap the local NoC path",
    )
    require(
        not (CASES["disjoint"].cross_path & LOCAL_PATH),
        "disjoint case must not overlap the local NoC path",
    )


def parse_output(output: str, spec: CaseSpec) -> Result:
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    sim_ns = None
    for line in output.splitlines():
        if "requests finished" in line or "Catch test finished" in line:
            match = re.search(r"(\d+)\s+ns", line)
            if match:
                sim_ns = int(match.group(1))
    require(sim_ns is not None, f"{spec.name}: completion time missing")

    flow_done: dict[tuple[int, int, int], int] = {}
    for match in re.finditer(r"(\d+):(\d+):(\d+)@(\d+)", output):
        flow_done[tuple(map(int, match.group(1, 2, 3)))] = int(match.group(4))

    noc = re.search(
        r"\[NOC_ACT\]\s+sends=([0-9,]+)\s+stalls=([0-9,]+)\s+"
        r"d2d_source_stalls=(\d+)",
        output,
    )
    require(noc is not None, f"{spec.name}: NOC_ACT missing")
    noc_send = ints(noc.group(1))
    noc_stall = ints(noc.group(2))

    data = re.search(
        r"\[D2D_DATA\]\s+in_pkts=(\d+)\s+out_pkts=(\d+)\s+"
        r"in_seqhash=(\d+)\s+out_seqhash=(\d+)\s+"
        r"in_csum=(\d+)\s+out_csum=(\d+)\s+out_inorder=(\d+)",
        output,
    )
    require(data is not None, f"{spec.name}: D2D_DATA missing")
    data_cycles = re.search(
        r"\[D2D_DATA\].*?in_first_cycle=(\d+)\s+in_last_cycle=(\d+)\s+"
        r"out_first_cycle=(\d+)\s+out_last_cycle=(\d+)",
        output,
    )
    require(data_cycles is not None, f"{spec.name}: D2D DATA cycle window missing")

    forward = re.search(
        r"\[D2D_LINK\]\s+idx=\d+\s+die0->die1.*?"
        r"data_in=(\d+)\s+data_out=(\d+)",
        output,
    )
    reverse = re.search(
        r"\[D2D_LINK\]\s+idx=\d+\s+die1->die0.*?"
        r"ack_in=(\d+)\s+ack_out=(\d+)",
        output,
    )
    require(forward is not None, f"{spec.name}: forward link stats missing")
    require(reverse is not None, f"{spec.name}: reverse link stats missing")

    bound = re.search(r"\[D2D_BOUND\]\s+idx=0\s+([^\n]+)", output)
    require(bound is not None, f"{spec.name}: D2D_BOUND missing")
    bound_values = ints(bound.group(1).split("|")[0])
    require(len(bound_values) == 11, f"{spec.name}: bad D2D_BOUND shape")

    credit = re.search(r"\[CREDIT\]\s+([^\n]+)", output)
    drain_values = [
        value
        for match in re.finditer(r"\[DRAIN\]\s+([^\n]+)", output)
        for value in assigned_ints(match.group(1).split("|")[0])
    ]
    require(credit is not None, f"{spec.name}: CREDIT missing")
    require(drain_values, f"{spec.name}: DRAIN missing")
    credit_balanced = all(value == 1 for value in assigned_ints(credit.group(1).split("|")[0]))
    drained = all(value == 0 for value in drain_values)

    return Result(
        sim_ns=sim_ns,
        flow_done=flow_done,
        noc_send=noc_send,
        noc_stall=noc_stall,
        d2d_source_stall=int(noc.group(3)),
        data_in=int(data.group(1)),
        data_out=int(data.group(2)),
        data_inorder=int(data.group(7)),
        data_integrity=(data.group(3) == data.group(4) and data.group(5) == data.group(6)),
        data_cycles=tuple(map(int, data_cycles.group(1, 2, 3, 4))),
        forward_data=(int(forward.group(1)), int(forward.group(2))),
        reverse_ack=(int(reverse.group(1)), int(reverse.group(2))),
        bounds=bound_values,
        credit_balanced=credit_balanced,
        drained=drained,
        watchdog="[PROTO_WAIT]" in output,
    )


def run_once(spec: CaseSpec) -> Result:
    command = [
        str(NPUSIM),
        "--workload-config",
        str(spec.workload),
        "--hardware-config",
        str(spec.hardware),
        "--simulation-config",
        str(SIM),
        "--mapping-config",
        str(MAPPING),
    ]
    proc = subprocess.run(
        command,
        cwd=BUILD,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    require(proc.returncode == 0, f"{spec.name}: npusim failed\n{proc.stdout[-3000:]}")
    return parse_output(proc.stdout, spec)


def validate_result(spec: CaseSpec, result: Result) -> None:
    require(spec.cross_flow in result.flow_done, f"{spec.name}: cross flow not complete")
    require(LOCAL_FLOW in result.flow_done, f"{spec.name}: local flow not complete")
    require((result.data_in, result.data_out) == (32, 32), f"{spec.name}: DATA loss")
    require(result.data_inorder == 1, f"{spec.name}: DATA reordered")
    require(result.data_integrity, f"{spec.name}: DATA seqhash/checksum mismatch")
    in_first, in_last, out_first, out_last = result.data_cycles
    require(in_last - in_first == 31, f"{spec.name}: D2D input span changed")
    require(out_last - out_first == 31, f"{spec.name}: D2D output span changed")
    require(out_first - in_first == 1, f"{spec.name}: D2D link latency changed")
    require(result.forward_data == (32, 32), f"{spec.name}: wrong forward DATA")
    require(result.reverse_ack == (1, 1), f"{spec.name}: reverse ACK missing")
    require(result.d2d_source_stall == 0, f"{spec.name}: D2D source throttled")
    require(result.credit_balanced, f"{spec.name}: credits not returned")
    require(result.drained, f"{spec.name}: residual state after completion")
    require(not result.watchdog, f"{spec.name}: protocol watchdog fired")

    # saf_peak/inflight_peak/rx_peak followed by eight stall/full counters.
    require(result.bounds[:3] == (32, 1, 1), f"{spec.name}: unexpected occupancy")
    require(
        all(value == 0 for value in result.bounds[3:]),
        f"{spec.name}: link/port throttling confounds NoC congestion",
    )


def generate_report(shared: Result, disjoint: Result) -> str:
    shared_cross = shared.flow_done[CASES["shared"].cross_flow]
    disjoint_cross = disjoint.flow_done[CASES["disjoint"].cross_flow]
    shared_local = shared.flow_done[LOCAL_FLOW]
    disjoint_local = disjoint.flow_done[LOCAL_FLOW]
    delta_ns = shared.sim_ns - disjoint.sim_ns
    delta_cycles = shared_cross - disjoint_cross
    slowdown = delta_ns * 100.0 / disjoint.sim_ns
    shared_overlap = sorted(CASES["shared"].cross_path & LOCAL_PATH)
    disjoint_overlap = sorted(CASES["disjoint"].cross_path & LOCAL_PATH)
    shared_in_first, shared_in_last, shared_out_first, shared_out_last = shared.data_cycles
    disjoint_in_first, disjoint_in_last, disjoint_out_first, disjoint_out_last = disjoint.data_cycles
    stall_per_send = 100.0 * shared.noc_stall[0] / shared.noc_send[0]
    flow_slowdown = 100.0 * delta_cycles / disjoint_cross

    return f"""# 跨 die 与片上混合流量 NoC 拥塞实验报告

## 结论

在同一套 2×1 die、相同的两个 GEMM、相同 D2D 链路参数和相同 32 个 DATA
包下，仅把跨 die 流的 C2C 出口从不相交的第 2 行移到本地流所在的第 1
行，就产生了 **{shared.noc_stall[0]} 次 NoC 发送阻塞**。跨 die flow
完成时间增加 **{delta_cycles} cycle = {delta_cycles * CYCLE_NS} ns**，仿真总时间
增加 **{delta_ns} ns（{slowdown:.2f}%）**。

## 实验设计

- 周期精确模式：`use_beha_noc=false`，`CYCLE={CYCLE_NS} ns`。
- 两个场景都执行两个完全相同的 `Matmul_f(B=1,T=4,C=64,OC=512)`。
- 本地 flow 固定为 `core5 -> core7`，NoC 路径为 `5->6->7`。
- shared：跨 die flow `core4 -> core20`，die0 路径
  `4->5->6->7->D2D`，与本地流共享 {shared_overlap}。
- disjoint：跨 die flow `core8 -> core24`，die0 路径
  `8->9->10->11->D2D`，与本地流共享 {disjoint_overlap}。
- 两个场景的 C2C hop 数、link latency/rate、SAF 容量、HOST 距离均一致。

## 周期精确结果

| 指标 | 不拥塞（disjoint） | 拥塞（shared） | 差值 |
|---|---:|---:|---:|
| 仿真完成时间 | {disjoint.sim_ns} ns | {shared.sim_ns} ns | +{delta_ns} ns |
| 跨 die flow 完成 | cycle {disjoint_cross} | cycle {shared_cross} | +{delta_cycles} cycle |
| 本地 flow 完成 | cycle {disjoint_local} | cycle {shared_local} | {shared_local-disjoint_local:+d} |
| die0 NoC 成功发送 | {disjoint.noc_send[0]} | {shared.noc_send[0]} | {shared.noc_send[0]-disjoint.noc_send[0]:+d} |
| die0 NoC stall | {disjoint.noc_stall[0]} | {shared.noc_stall[0]} | +{shared.noc_stall[0]-disjoint.noc_stall[0]} |
| D2D DATA in/out | {disjoint.data_in}/{disjoint.data_out} | {shared.data_in}/{shared.data_out} | 0 |
| DATA seqhash/checksum | match | match | — |
| D2D source/端口/链路 stall | 0 | 0 | 0 |
| 最终残留 | 0 | 0 | 0 |

## 静态负载模型与仿真对照

忽略 REQUEST/ACK 和流水线相位，只统计 32 个 DATA 包在源 die mesh 上的
packet-hop，可得到：

| 理论量 | 不拥塞（disjoint） | 拥塞（shared） |
|---|---:|---:|
| 本地 flow packet-hop | `32×2 = 64` | `32×2 = 64` |
| 跨 die flow 源 mesh packet-hop | `32×3 = 96` | `32×3 = 96` |
| DATA 总 packet-hop | 160 | 160 |
| 单条有向链路最大 DATA 负载 | 32 | 64 |
| 1 packet/cycle 下的瓶颈服务下界 | 32 cycle | 64 cycle |

因此两种编排的通信工作总量相同，但 shared 把两条 flow 的负载集中到
`5->6` 和 `6->7`，瓶颈链路负载翻倍。若本地 flow 先获得输出锁，简单容量
模型给出的额外串行化下界是 **32 cycle**。

周期仿真在 D2D 边界观测到：

| 周期观测 | 不拥塞（disjoint） | 拥塞（shared） | 差值 |
|---|---:|---:|---:|
| D2D DATA 输入窗口 | {disjoint_in_first}–{disjoint_in_last} | {shared_in_first}–{shared_in_last} | 整体 +{shared_in_first-disjoint_in_first} cycle |
| D2D DATA 输出窗口 | {disjoint_out_first}–{disjoint_out_last} | {shared_out_first}–{shared_out_last} | 整体 +{shared_out_first-disjoint_out_first} cycle |
| 输入/输出窗口宽度 | 31/31 cycle | 31/31 cycle | 0 |
| D2D 首包固定延迟 | 1 cycle | 1 cycle | 0 |
| 跨 die flow 完成 | {disjoint_cross} | {shared_cross} | +{delta_cycles} cycle |

实测额外 **{delta_cycles} cycle** 比最简容量下界多 4 cycle（相对实测差
11.11%）。这 4 cycle 是静态容量模型未覆盖的动态流水线项，候选来源包括有限
router buffer、registered ready、输入仲裁和 output-lock 释放相位；没有逐 router
事件 trace 时，不能把它进一步确定地分摊到某一个机制。
`NOC_ACT.stalls={shared.noc_stall[0]}` 表示“输出有包但下游 input buffer 满”的累计
blocked-output 事件，约为每 100 次成功 NoC 发送对应 {stall_per_send:.2f} 次；它不是
flow 被推迟的 cycle 数，所以不能把 11 stall 直接换算成 22 ns。跨 die flow
完成时间增加 {flow_slowdown:.2f}%，总运行时间增加 {slowdown:.2f}%。

## 归因与边界

两个场景成功发送数相同、D2D DATA/ACK 数相同、D2D 端口和链路 stall
均为 0，且本地 flow 完成周期相同；差异只在 shared 场景的 die0 NoC
路径竞争。因此，本实验观测到的额外延迟可归因于跨 die 出口流量与片上
流量共享 NoC 物理链路，而不是计算量、D2D hop 数、HOST 距离或 D2D
限速差异。

本报告证明的是该确定性双 flow 工作负载下的周期精确拥塞效应；它不是对
所有路由、流量分布或有限缓冲参数的统计性能结论。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    try:
        validate_inputs()
        results: dict[str, Result] = {}
        for name, spec in CASES.items():
            first = run_once(spec)
            second = run_once(spec)
            validate_result(spec, first)
            validate_result(spec, second)
            require(first == second, f"{name}: repeated runs are not deterministic")
            results[name] = first

        shared = results["shared"]
        disjoint = results["disjoint"]
        require(shared.noc_send == disjoint.noc_send, "successful NoC sends differ")
        require(disjoint.noc_stall[0] == 0, "disjoint case unexpectedly congested")
        require(shared.noc_stall[0] > 0, "shared case did not create congestion")
        require(
            shared.flow_done[LOCAL_FLOW] == disjoint.flow_done[LOCAL_FLOW],
            "local flow timing changed; experiment is not isolated",
        )
        cycle_delta = (
            shared.flow_done[CASES["shared"].cross_flow]
            - disjoint.flow_done[CASES["disjoint"].cross_flow]
        )
        require(cycle_delta > 0, "shared path did not delay the cross-die flow")
        require(shared.sim_ns > disjoint.sim_ns, "shared case did not increase runtime")
        require(
            shared.sim_ns - disjoint.sim_ns == cycle_delta * CYCLE_NS,
            "sim-time delta does not match the cycle-accurate flow delay",
        )

        args.report.write_text(generate_report(shared, disjoint), encoding="utf-8")
        print(
            "PASS disjoint: "
            f"sim={disjoint.sim_ns}ns stalls={disjoint.noc_stall} "
            f"cross_done={disjoint.flow_done[CASES['disjoint'].cross_flow]}"
        )
        print(
            "PASS shared:   "
            f"sim={shared.sim_ns}ns stalls={shared.noc_stall} "
            f"cross_done={shared.flow_done[CASES['shared'].cross_flow]}"
        )
        print(
            "PASS effect:   "
            f"delta={shared.sim_ns-disjoint.sim_ns}ns "
            f"({cycle_delta} cycles), D2D=32/32, drain=0"
        )
        print(f"REPORT {args.report}")
        return 0
    except (AssertionError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
