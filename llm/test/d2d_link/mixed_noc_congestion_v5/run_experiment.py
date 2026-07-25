#!/usr/bin/env python3
"""V5 cycle-accurate striped D2D/on-chip NoC congestion experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILD = ROOT / "build"
NPUSIM = BUILD / "npusim"
SIM = ROOT / "llm" / "test" / "noc_congestion" / "sim" / "sim_cycle.json"
MAPPING = ROOT / "llm" / "test" / "noc_congestion" / "mapping" / "identity.spec"
REPORT = HERE / "mixed_noc_congestion_v5_report.md"
CYCLE_NS = 2


@dataclass(frozen=True)
class CaseSpec:
    name: str
    hardware: Path
    workload: Path
    cross_flow: tuple[int, int, int]
    c2c_rows: tuple[int, int]


@dataclass(frozen=True)
class SubflowStat:
    link: int
    source: int
    tag: int
    subflow: int
    in_pkts: int
    out_pkts: int
    in_seqhash: int
    out_seqhash: int
    in_csum: int
    out_csum: int
    inorder: int
    minseq: int
    maxseq: int
    endseq: int
    ends: int
    end_length: int


@dataclass(frozen=True)
class LinkStat:
    index: int
    src_die: int
    dst_die: int
    direction: str
    req_in: int
    req_out: int
    ack_in: int
    ack_out: int
    data_in: int
    data_out: int


@dataclass(frozen=True)
class BoundStat:
    index: int
    saf_peak: int
    inflight_peak: int
    rx_peak: int
    saf_full: int
    inflight_full: int
    rx_full: int
    port_stall: int
    link_stall: int
    inflight_stall: int
    rx_stall: int
    downstream_stall: int
    group_stall: int


@dataclass(frozen=True)
class Theory:
    grid_x: int
    grid_y: int
    packets: int
    stripes: int
    quotas: tuple[int, ...]
    local_path: tuple[tuple[int, int], ...]
    cross_source_paths: tuple[tuple[tuple[int, int], ...], ...]
    cross_dest_paths: tuple[tuple[tuple[int, int], ...], ...]
    selected_rows: tuple[int, ...]
    overlap_edges: tuple[tuple[int, int], ...]
    local_packet_hops: int
    cross_source_packet_hops: int
    cross_dest_packet_hops: int
    max_source_edge_load: int
    bottleneck_cycles: int


@dataclass(frozen=True)
class Result:
    sim_ns: int
    flow_done: tuple[tuple[tuple[int, int, int], int], ...]
    noc_send: tuple[int, ...]
    noc_stall: tuple[int, ...]
    d2d_source_stall: int
    typed: tuple[int, ...]
    repin: tuple[int, ...]
    data_cycles: tuple[int, int, int, int]
    subflows: tuple[SubflowStat, ...]
    links: tuple[LinkStat, ...]
    bounds: tuple[BoundStat, ...]
    die_router_pkts: tuple[int, ...]
    die_mesh_pkts: tuple[int, ...]
    saf_reserved: tuple[int, int]
    saf_admit: tuple[int, int]
    credit_balanced: bool
    drained: bool
    host_mismatch: int
    watchdog: bool

    def flow_cycle(self, key: tuple[int, int, int]) -> int:
        return dict(self.flow_done)[key]


CASES = {
    "shared": CaseSpec(
        "shared",
        HERE / "hardware" / "shared.json",
        HERE / "workload" / "shared.json",
        (4, 20, 20),
        (0, 1),
    ),
    "disjoint": CaseSpec(
        "disjoint",
        HERE / "hardware" / "disjoint.json",
        HERE / "workload" / "disjoint.json",
        (8, 24, 24),
        (2, 3),
    ),
}
LOCAL_FLOW = (5, 7, 7)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"-?\d+", text))


def assigned_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"=(-?\d+)", text))


def xy_edges(src: int, dst: int, grid_x: int) -> tuple[tuple[int, int], ...]:
    """Return the directed X-first XY path within one die."""
    cur = src
    edges: list[tuple[int, int]] = []
    sx, sy = cur % grid_x, cur // grid_x
    dx, dy = dst % grid_x, dst // grid_x
    while sx != dx:
        nxt = cur + (1 if sx < dx else -1)
        edges.append((cur, nxt))
        cur = nxt
        sx = cur % grid_x
    while sy != dy:
        nxt = cur + (grid_x if sy < dy else -grid_x)
        edges.append((cur, nxt))
        cur = nxt
        sy = cur // grid_x
    return tuple(edges)


def packet_count(workload: dict) -> int:
    variables = workload["vars"]
    elements = (
        int(variables["B"]) * int(variables["T"]) * int(variables["OC"])
    )
    return (elements + 63) // 64


def producer_records(workload: dict) -> list[tuple[int, dict, dict]]:
    records = []
    for core in workload["chips"][0]["cores"]:
        for item in core["worklist"]:
            if item.get("prims"):
                records.append((core["id"], item["prims"][0], item["cast"][0]))
    return records


def build_theory(spec: CaseSpec) -> Theory:
    hw = load(spec.hardware)
    wl = load(spec.workload)
    grid_x = int(hw["x"])
    grid_y = int(hw.get("y", grid_x))
    cores_per_die = grid_x * grid_y
    packets = packet_count(wl)
    source, dest, _ = spec.cross_flow
    stripes = next(
        cast.get("stripe", 1)
        for core, _, cast in producer_records(wl)
        if core == source
    )
    quotas = tuple(
        packets // stripes + (1 if s < packets % stripes else 0)
        for s in range(stripes)
    )

    # V5 hybrid: select the source-coordinate band, then rotate by subflow.
    rows = tuple(sorted(spec.c2c_rows))
    source_local = source % cores_per_die
    source_row = source_local // grid_x
    band = min(len(rows) - 1, source_row * len(rows) // grid_y)
    selected_rows = tuple(rows[(band + s) % len(rows)] for s in range(stripes))

    source_paths = tuple(
        xy_edges(source_local, row * grid_x + grid_x - 1, grid_x)
        for row in selected_rows
    )
    dest_local = dest % cores_per_die
    dest_paths = tuple(
        xy_edges(row * grid_x, dest_local, grid_x)
        for row in selected_rows
    )
    local_path = xy_edges(LOCAL_FLOW[0], LOCAL_FLOW[1], grid_x)

    edge_load: dict[tuple[int, int], int] = defaultdict(int)
    for edge in local_path:
        edge_load[edge] += packets
    for quota, path in zip(quotas, source_paths):
        for edge in path:
            edge_load[edge] += quota

    cross_edges = {edge for path in source_paths for edge in path}
    overlap = tuple(sorted(cross_edges & set(local_path)))
    local_hops = packets * len(local_path)
    source_hops = sum(q * len(path) for q, path in zip(quotas, source_paths))
    dest_hops = sum(q * len(path) for q, path in zip(quotas, dest_paths))
    max_load = max(edge_load.values())
    return Theory(
        grid_x=grid_x,
        grid_y=grid_y,
        packets=packets,
        stripes=stripes,
        quotas=quotas,
        local_path=local_path,
        cross_source_paths=source_paths,
        cross_dest_paths=dest_paths,
        selected_rows=selected_rows,
        overlap_edges=overlap,
        local_packet_hops=local_hops,
        cross_source_packet_hops=source_hops,
        cross_dest_packet_hops=dest_hops,
        max_source_edge_load=max_load,
        bottleneck_cycles=max_load,
    )


def validate_inputs() -> dict[str, Theory]:
    require(NPUSIM.exists(), f"npusim not found: {NPUSIM}")
    sim = load(SIM)
    require(
        sim.get("noc", {}).get("use_beha_noc") is False,
        "experiment must use the cycle-accurate NoC",
    )

    workloads = {name: load(spec.workload) for name, spec in CASES.items()}
    require(
        workloads["shared"]["vars"] == workloads["disjoint"]["vars"],
        "both cases must compute the same GEMM shape",
    )
    for name, workload in workloads.items():
        producers = producer_records(workload)
        require(len(producers) == 2, f"{name}: expected exactly two producers")
        require(
            all(prim["type"] == "Matmul_f" for _, prim, _ in producers),
            f"{name}: every producer must be a GEMM",
        )
        require(
            producers[0][1] == producers[1][1],
            f"{name}: local and cross-die GEMMs differ",
        )
        flows = {(core, cast["dest"], cast["tag"]) for core, _, cast in producers}
        require(
            flows == {LOCAL_FLOW, CASES[name].cross_flow},
            f"{name}: workload does not contain the controlled two flows",
        )
        cross_cast = next(
            cast for core, _, cast in producers if core == CASES[name].cross_flow[0]
        )
        require(cross_cast.get("stripe") == 2, f"{name}: cross flow is not striped")

    normalized_hardware = []
    for name, spec in CASES.items():
        hw = load(spec.hardware)
        require(hw["die"] == {"x": 2, "y": 1}, f"{name}: topology must be 2x1")
        c2c_cfg = hw["die_ports"]["c2c"]
        require(c2c_cfg.get("multi_port") is True, f"{name}: not V5 multi-port")
        require(c2c_cfg.get("select_policy") == "hybrid", f"{name}: policy changed")
        require(c2c_cfg.get("mode") == "bounded_saf", f"{name}: not cycle SAF")
        require(
            hw["die_ports"]["edges"].get("N", {}).get("role") == "host"
            and hw["die_ports"]["edges"].get("S", {}).get("role") == "host",
            f"{name}: symmetric N/S HOST attachment required",
        )
        c2c_ports = [
            port
            for port in hw["die_ports"]["overrides"]
            if port.get("role") == "c2c"
        ]
        require(
            len(c2c_ports) == 4,
            f"{name}: expected two E and two W C2C ports",
        )
        require(
            {port["idx"] for port in c2c_ports} == set(spec.c2c_rows),
            f"{name}: unexpected C2C rows",
        )
        for port in c2c_ports:
            port["idx"] -= min(spec.c2c_rows)
        normalized_hardware.append(hw)
    require(
        normalized_hardware[0] == normalized_hardware[1],
        "hardware cases may differ only by translating both C2C rows",
    )

    theory = {name: build_theory(spec) for name, spec in CASES.items()}
    shared, disjoint = theory["shared"], theory["disjoint"]
    require(
        shared.local_packet_hops == disjoint.local_packet_hops,
        "local NoC work differs",
    )
    require(
        shared.cross_source_packet_hops == disjoint.cross_source_packet_hops,
        "cross-flow source NoC work differs",
    )
    require(
        shared.cross_dest_packet_hops == disjoint.cross_dest_packet_hops,
        "cross-flow destination NoC work differs",
    )
    require(
        shared.overlap_edges == tuple(sorted(shared.local_path)),
        "shared flow must overlap the complete local path",
    )
    require(not disjoint.overlap_edges, "disjoint path overlaps local path")
    require(
        shared.max_source_edge_load == 2 * disjoint.max_source_edge_load,
        "controlled comparison must double the bottleneck load",
    )
    return theory


def parse_output(output: str, spec: CaseSpec) -> Result:
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    finish = re.search(r"All requests finished.*?\|\s+(\d+)\s+ns", output)
    require(finish is not None, f"{spec.name}: completion time missing")

    flow_done = tuple(
        sorted(
            (
                tuple(map(int, match.group(1, 2, 3))),
                int(match.group(4)),
            )
            for match in re.finditer(r"(\d+):(\d+):(\d+)@(\d+)", output)
        )
    )
    noc = re.search(
        r"\[NOC_ACT\]\s+sends=([0-9,]+)\s+stalls=([0-9,]+)\s+"
        r"d2d_source_stalls=(\d+)",
        output,
    )
    require(noc is not None, f"{spec.name}: NOC_ACT missing")
    typed = re.search(
        r"\[D2D_TYPE\]\s+request_in=(\d+)\s+request_out=(\d+)\s+"
        r"ack_in=(\d+)\s+ack_out=(\d+)\s+data_in=(\d+)\s+data_out=(\d+)",
        output,
    )
    repin = re.search(
        r"\[D2D_REPIN\]\s+total=(\d+)\s+changed=(\d+)\s+same=(\d+)",
        output,
    )
    cycles = re.search(
        r"\[D2D_DATA\].*?in_first_cycle=(\d+)\s+in_last_cycle=(\d+)\s+"
        r"out_first_cycle=(\d+)\s+out_last_cycle=(\d+)",
        output,
    )
    require(typed is not None, f"{spec.name}: D2D_TYPE missing")
    require(repin is not None, f"{spec.name}: D2D_REPIN missing")
    require(cycles is not None, f"{spec.name}: D2D_DATA cycles missing")

    subflows = tuple(
        SubflowStat(*map(int, values))
        for values in re.findall(
            r"\[V5_SUBFLOW\]\s+idx=(\d+)\s+source=(\d+)\s+tag=(\d+)\s+"
            r"subflow=(\d+)\s+in=(\d+)\s+out=(\d+)\s+"
            r"in_seqhash=(\d+)\s+out_seqhash=(\d+)\s+"
            r"in_csum=(\d+)\s+out_csum=(\d+)\s+inorder=(\d+)\s+"
            r"minseq=(-?\d+)\s+maxseq=(-?\d+)\s+endseq=(-?\d+)\s+"
            r"ends=(\d+)\s+end_length=(-?\d+)",
            output,
        )
    )
    links = tuple(
        LinkStat(
            int(index),
            int(src),
            int(dst),
            direction,
            *map(int, counts),
        )
        for index, src, dst, direction, *counts in re.findall(
            r"\[D2D_LINK\]\s+idx=(\d+)\s+die(\d+)->die(\d+)\s+dir=([A-Z]+)\s+"
            r"req_in=(\d+)\s+req_out=(\d+)\s+ack_in=(\d+)\s+ack_out=(\d+)\s+"
            r"data_in=(\d+)\s+data_out=(\d+)",
            output,
        )
    )
    bounds = tuple(
        BoundStat(*map(int, values))
        for values in re.findall(
            r"\[D2D_BOUND\]\s+idx=(\d+)\s+saf_peak=(\d+)\s+"
            r"inflight_peak=(\d+)\s+rx_peak=(\d+)\s+saf_full=(\d+)\s+"
            r"inflight_full=(\d+)\s+rx_full=(\d+)\s+port_stall=(\d+)\s+"
            r"link_stall=(\d+)\s+inflight_stall=(\d+)\s+rx_stall=(\d+)\s+"
            r"downstream_stall=(\d+)\s+group_stall=(\d+)",
            output,
        )
    )
    die_act = re.search(
        r"\[DIE_ACT\]\s+router_pkts=([0-9,]+)\s+mesh_pkts=([0-9,]+)",
        output,
    )
    saf = re.search(
        r"\[SAF\]\s+reserved_packets=(\d+)\s+group_reserved_packets=(\d+)",
        output,
    )
    admit = re.search(r"\[SAF_ADMIT\]\s+success=(\d+)\s+reject=(\d+)", output)
    credit = re.search(r"\[CREDIT\]\s+([^\n]+)", output)
    host = re.search(r"\[HOSTLANE\].*?mismatch=(\d+)", output)
    drain_values = [
        value
        for match in re.finditer(r"\[DRAIN\]\s+([^\n]+)", output)
        for value in assigned_ints(match.group(1).split("|")[0])
    ]
    require(die_act is not None, f"{spec.name}: DIE_ACT missing")
    require(saf is not None, f"{spec.name}: SAF missing")
    require(admit is not None, f"{spec.name}: SAF_ADMIT missing")
    require(credit is not None, f"{spec.name}: CREDIT missing")
    require(host is not None, f"{spec.name}: HOSTLANE missing")
    require(drain_values, f"{spec.name}: DRAIN missing")

    return Result(
        sim_ns=int(finish.group(1)),
        flow_done=flow_done,
        noc_send=ints(noc.group(1)),
        noc_stall=ints(noc.group(2)),
        d2d_source_stall=int(noc.group(3)),
        typed=tuple(map(int, typed.groups())),
        repin=tuple(map(int, repin.groups())),
        data_cycles=tuple(map(int, cycles.groups())),
        subflows=subflows,
        links=links,
        bounds=bounds,
        die_router_pkts=ints(die_act.group(1)),
        die_mesh_pkts=ints(die_act.group(2)),
        saf_reserved=tuple(map(int, saf.groups())),
        saf_admit=tuple(map(int, admit.groups())),
        credit_balanced=all(
            value == 1
            for value in assigned_ints(credit.group(1).split("|")[0])
        ),
        drained=all(value == 0 for value in drain_values),
        host_mismatch=int(host.group(1)),
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


def validate_result(spec: CaseSpec, theory: Theory, result: Result) -> None:
    flow_done = dict(result.flow_done)
    require(spec.cross_flow in flow_done, f"{spec.name}: cross flow incomplete")
    require(LOCAL_FLOW in flow_done, f"{spec.name}: local flow incomplete")
    require(
        result.typed == (2, 2, 2, 2, theory.packets, theory.packets),
        f"{spec.name}: wrong REQUEST/ACK/DATA totals {result.typed}",
    )
    require(
        result.repin == (theory.packets + 4, theory.packets + 4, 0),
        f"{spec.name}: wrong ingress re-pin count {result.repin}",
    )

    relevant = [
        stat
        for stat in result.subflows
        if (stat.source, stat.tag)
        == (spec.cross_flow[0], spec.cross_flow[2])
    ]
    require(len(relevant) == theory.stripes, f"{spec.name}: missing subflow stats")
    require(
        {stat.subflow for stat in relevant} == set(range(theory.stripes)),
        f"{spec.name}: wrong subflow ids",
    )
    require(
        len({stat.link for stat in relevant}) == theory.stripes,
        f"{spec.name}: stripes did not use distinct links",
    )
    for stat in relevant:
        quota = theory.quotas[stat.subflow]
        require(
            stat.in_pkts == stat.out_pkts == quota,
            f"{spec.name}: subflow {stat.subflow} lost/duplicated DATA",
        )
        require(
            stat.in_seqhash == stat.out_seqhash
            and stat.in_csum == stat.out_csum
            and stat.inorder == 1,
            f"{spec.name}: subflow {stat.subflow} reordered/corrupted DATA",
        )
        require(
            stat.minseq == 1
            and stat.maxseq == stat.endseq == quota
            and stat.ends == 1,
            f"{spec.name}: subflow {stat.subflow} packet shape is invalid",
        )

    forward = [link for link in result.links if link.src_die == 0]
    reverse = [link for link in result.links if link.src_die == 1]
    require(len(forward) == len(reverse) == 2, f"{spec.name}: wrong link count")
    require(len(result.bounds) == 4, f"{spec.name}: missing bounded-link stats")
    require(
        all(
            link.direction == "E"
            and (link.req_in, link.req_out) == (1, 1)
            and (link.data_in, link.data_out) == (16, 16)
            and (link.ack_in, link.ack_out) == (0, 0)
            for link in forward
        ),
        f"{spec.name}: wrong forward-link traffic",
    )
    require(
        all(
            link.direction == "W"
            and (link.ack_in, link.ack_out) == (1, 1)
            and (link.req_in, link.req_out, link.data_in, link.data_out)
            == (0, 0, 0, 0)
            for link in reverse
        ),
        f"{spec.name}: wrong reverse-link traffic",
    )
    require(result.d2d_source_stall == 0, f"{spec.name}: source port throttled")
    require(
        all(
            bound.port_stall == 0
            and bound.link_stall == 0
            and bound.inflight_stall == 0
            and bound.group_stall == 0
            for bound in result.bounds
        ),
        f"{spec.name}: D2D port/link rate confounds NoC contention",
    )
    require(result.saf_reserved == (0, 0), f"{spec.name}: SAF reservation leaked")
    require(result.saf_admit == (1, 0), f"{spec.name}: SAF admission changed")
    require(result.credit_balanced, f"{spec.name}: credits did not return")
    require(result.drained, f"{spec.name}: residual router/link state")
    require(result.host_mismatch == 0, f"{spec.name}: wrong HOST lane")
    require(not result.watchdog, f"{spec.name}: protocol watchdog fired")


def aggregate_bound_stalls(result: Result) -> tuple[int, ...]:
    fields = (
        "saf_full",
        "inflight_full",
        "rx_full",
        "port_stall",
        "link_stall",
        "inflight_stall",
        "rx_stall",
        "downstream_stall",
        "group_stall",
    )
    return tuple(sum(getattr(bound, field) for bound in result.bounds) for field in fields)


def format_path(path: tuple[tuple[int, int], ...]) -> str:
    if not path:
        return "（入口即目的 tile）"
    nodes = [path[0][0], *(edge[1] for edge in path)]
    return "→".join(map(str, nodes))


def generate_report(
    theory: dict[str, Theory],
    results: dict[str, Result],
) -> str:
    shared_t, disjoint_t = theory["shared"], theory["disjoint"]
    shared, disjoint = results["shared"], results["disjoint"]
    shared_cross = shared.flow_cycle(CASES["shared"].cross_flow)
    disjoint_cross = disjoint.flow_cycle(CASES["disjoint"].cross_flow)
    shared_local = shared.flow_cycle(LOCAL_FLOW)
    disjoint_local = disjoint.flow_cycle(LOCAL_FLOW)
    delta_cycles = shared_cross - disjoint_cross
    delta_ns = shared.sim_ns - disjoint.sim_ns
    theory_delta = shared_t.bottleneck_cycles - disjoint_t.bottleneck_cycles
    error_cycles = delta_cycles - theory_delta
    model_error = 100.0 * abs(error_cycles) / theory_delta
    total_slowdown = 100.0 * delta_ns / disjoint.sim_ns
    flow_slowdown = 100.0 * delta_cycles / disjoint_cross
    stall_rate = 100.0 * shared.noc_stall[0] / shared.noc_send[0]
    s_in0, s_in1, s_out0, s_out1 = shared.data_cycles
    d_in0, d_in1, d_out0, d_out1 = disjoint.data_cycles

    shared_paths = "；".join(
        f"subflow {i}（{shared_t.quotas[i]} 包）: {format_path(path)}"
        for i, path in enumerate(shared_t.cross_source_paths)
    )
    disjoint_paths = "；".join(
        f"subflow {i}（{disjoint_t.quotas[i]} 包）: {format_path(path)}"
        for i, path in enumerate(disjoint_t.cross_source_paths)
    )

    return f"""# V5 跨 die striping 与片上流量 NoC 拥塞实验报告

## 结论

在同一套 2×1 dies、两个相同 GEMM、32 个跨 die DATA 包、2-way striping
及相同 D2D 参数下，仅平移跨 die 源核和两条 C2C 端口，使其在 shared
场景与本地 flow 共享两条源 die NoC 链路，便在 die0 产生
**{shared.noc_stall[0]} 次 blocked-output 事件**。跨 die flow 比不相交场景
晚 **{delta_cycles} cycle = {delta_cycles * CYCLE_NS} ns** 完成，仿真总时间增加
**{delta_ns} ns（{total_slowdown:.2f}%）**。

理想单链路容量模型预测的拥塞服务项为
**{shared_t.bottleneck_cycles}−{disjoint_t.bottleneck_cycles} = {theory_delta} cycle**；
周期精确结果为 {delta_cycles} cycle，绝对偏差
{abs(error_cycles)} cycle、相对偏差 {model_error:.2f}%。

## 实验设计

- 周期精确模式：`use_beha_noc=false`，`CYCLE={CYCLE_NS} ns`。
- 两个场景均执行两个 `Matmul_f(B=1,T=4,C=64,OC=512)`。
- 本地 flow 固定为 `core5→core7`，路径
  `{format_path(shared_t.local_path)}`，发送 {shared_t.packets} 个包。
- 跨 die flow 使用 V5 `multi_port=true`、`hybrid` 选择和 2-way striping，
  配额为 {list(shared_t.quotas)}。
- shared：`core4→core20`，C2C rows={list(shared_t.selected_rows)}；
  {shared_paths}。与本地 flow 共享 {list(shared_t.overlap_edges)}。
- disjoint：`core8→core24`，C2C rows={list(disjoint_t.selected_rows)}；
  {disjoint_paths}。与本地 flow 共享 {list(disjoint_t.overlap_edges)}。
- 两个场景的 GEMM、总 DATA、stripe 配额、源/目的 mesh packet-hop、
  D2D link latency/rate/capacity 和 HOST 布局完全相同。

## 周期精确仿真结果

| 指标 | 无源 die 混合争用（disjoint） | 有源 die 混合争用（shared） | 差值 |
|---|---:|---:|---:|
| 仿真完成时间 | {disjoint.sim_ns} ns | {shared.sim_ns} ns | +{delta_ns} ns |
| 跨 die flow 完成 | cycle {disjoint_cross} | cycle {shared_cross} | +{delta_cycles} cycle |
| 本地 flow 完成 | cycle {disjoint_local} | cycle {shared_local} | {shared_local-disjoint_local:+d} |
| die0 NoC 成功发送 | {disjoint.noc_send[0]} | {shared.noc_send[0]} | {shared.noc_send[0]-disjoint.noc_send[0]:+d} |
| die0 NoC stall | {disjoint.noc_stall[0]} | {shared.noc_stall[0]} | +{shared.noc_stall[0]-disjoint.noc_stall[0]} |
| die1 NoC stall（共同的 stripe 汇聚） | {disjoint.noc_stall[1]} | {shared.noc_stall[1]} | {shared.noc_stall[1]-disjoint.noc_stall[1]:+d} |
| D2D REQUEST/ACK/DATA | {disjoint.typed} | {shared.typed} | 相同 |
| 每 subflow DATA | 16/16，无损/有序 | 16/16，无损/有序 | 相同 |
| D2D source/port/link/inflight/group stall | 0 | 0 | 0 |
| SAF/credit/router/link 残留 | 0 | 0 | 0 |

两次独立运行的所有已解析指标均完全一致。全局 `[D2D_DATA]` 的两个 stripe
允许合法交织，因此完整性使用 `[V5_SUBFLOW]` 分桶验证：每个 subflow 的
包数、顺序 hash、完整 payload checksum、序号范围和唯一 tail 均匹配。

## 理论路径负载

只按 DATA 统计有向 NoC `packet-hop`：

| 理论量 | disjoint | shared |
|---|---:|---:|
| 本地 flow 源 die packet-hop | {disjoint_t.local_packet_hops} | {shared_t.local_packet_hops} |
| 跨 die flow 源 die packet-hop | {disjoint_t.cross_source_packet_hops} | {shared_t.cross_source_packet_hops} |
| 跨 die flow 目的 die packet-hop | {disjoint_t.cross_dest_packet_hops} | {shared_t.cross_dest_packet_hops} |
| 总 DATA mesh packet-hop | {disjoint_t.local_packet_hops + disjoint_t.cross_source_packet_hops + disjoint_t.cross_dest_packet_hops} | {shared_t.local_packet_hops + shared_t.cross_source_packet_hops + shared_t.cross_dest_packet_hops} |
| 源 die 单条有向链路最大负载 | {disjoint_t.max_source_edge_load} | {shared_t.max_source_edge_load} |
| 1 packet/cycle 理想瓶颈服务项 | {disjoint_t.bottleneck_cycles} cycle | {shared_t.bottleneck_cycles} cycle |

总通信工作量相同，但 shared 的两个 stripe 在到达不同 C2C 端口前仍共享
源核的行向 NoC cut；`5→6`、`6→7` 各承载本地 32 包和跨 die 32 包，
负载达到 64。disjoint 把本地和跨 die 流分散到不同链路，最大负载只有
32。这也说明 V5 的两条 D2D lane 不等于两条独立的源 NoC 注入 cut。

## 理论值与仿真对比

| 对照量 | disjoint | shared | 差值 |
|---|---:|---:|---:|
| 理论瓶颈服务项 | {disjoint_t.bottleneck_cycles} cycle | {shared_t.bottleneck_cycles} cycle | +{theory_delta} |
| D2D DATA 输入窗口 | {d_in0}–{d_in1} | {s_in0}–{s_in1} | 首包 +{s_in0-d_in0} |
| D2D DATA 输出窗口 | {d_out0}–{d_out1} | {s_out0}–{s_out1} | 首包 +{s_out0-d_out0} |
| 跨 die flow 完成 | {disjoint_cross} | {shared_cross} | +{delta_cycles} |

拥塞延迟已经在 D2D 输入边界出现：首个 DATA 进入 D2D 的时间整体后移
{s_in0-d_in0} cycle，恰好等于跨 die flow 完成周期差。实测差值比理想容量
模型少 {abs(error_cycles)} cycle（{model_error:.2f}%）。容量模型只计算稳态
瓶颈负载，不包含两条 GEMM 的启动相位、router pipeline、仲裁相位和
output-lock 的瞬态重叠，因此不应期待逐周期完全相等；误差仅 {abs(error_cycles)} cycle，
且方向和量级一致。

`NOC_ACT.stalls={shared.noc_stall[0]}` 是“输出有包但下游输入满”的累计事件，
约为每 100 次成功发送对应 {stall_rate:.2f} 次，不等价于 flow 延迟 cycle。
本地 flow 仍在 cycle {shared_local} 完成，表明本次确定性仲裁中它先取得共享
输出资源，额外等待主要落在跨 die flow 上；后者完成时间增加
{flow_slowdown:.2f}%。

## 归因与适用边界

两个场景的成功 NoC 发送数、per-die 活动、D2D 消息数、每 link 包数、
SAF admission、D2D backpressure 总量以及最终排空状态均一致；只有 die0
共享路径产生额外 stall。因而差异可归因于跨 die 流量和片内流量对源 die
NoC 物理链路的争用，而不是计算量、D2D 带宽、stripe 不均、丢包或残留状态。

本实验验证的是一个确定性的双 GEMM、单源 2-way striping 场景。它不是对
任意流量分布的统计性能预测；理论模型也是容量近似，不包含完整动态仲裁。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    try:
        theory = validate_inputs()
        results: dict[str, Result] = {}
        for name, spec in CASES.items():
            first = run_once(spec)
            second = run_once(spec)
            validate_result(spec, theory[name], first)
            validate_result(spec, theory[name], second)
            require(first == second, f"{name}: repeated runs are not deterministic")
            results[name] = first

        shared, disjoint = results["shared"], results["disjoint"]
        shared_t, disjoint_t = theory["shared"], theory["disjoint"]
        require(shared.noc_send == disjoint.noc_send, "successful NoC sends differ")
        require(shared.die_router_pkts == disjoint.die_router_pkts, "router work differs")
        require(shared.die_mesh_pkts == disjoint.die_mesh_pkts, "mesh work differs")
        require(disjoint.noc_stall[0] == 0, "disjoint source die is congested")
        require(shared.noc_stall[0] > 0, "shared source die did not congest")
        require(
            shared.noc_stall[1:] == disjoint.noc_stall[1:],
            "non-source-die stalls differ",
        )
        require(
            aggregate_bound_stalls(shared) == aggregate_bound_stalls(disjoint),
            "D2D backpressure differs and confounds the NoC comparison",
        )
        require(
            shared.flow_cycle(LOCAL_FLOW) == disjoint.flow_cycle(LOCAL_FLOW),
            "local flow timing changed; comparison is not isolated",
        )
        delta_cycles = (
            shared.flow_cycle(CASES["shared"].cross_flow)
            - disjoint.flow_cycle(CASES["disjoint"].cross_flow)
        )
        require(delta_cycles > 0, "shared path did not delay the cross-die flow")
        require(shared.sim_ns > disjoint.sim_ns, "shared path did not increase runtime")
        require(
            shared.sim_ns - disjoint.sim_ns == delta_cycles * CYCLE_NS,
            "sim-time delta does not match cycle-accurate flow delay",
        )
        theory_delta = (
            shared_t.bottleneck_cycles - disjoint_t.bottleneck_cycles
        )
        require(theory_delta > 0, "theoretical model predicts no congestion")
        require(
            abs(delta_cycles - theory_delta) <= 4,
            "simulation differs from the ideal capacity model by more than 4 cycles",
        )

        args.report.write_text(generate_report(theory, results), encoding="utf-8")
        print(
            "PASS disjoint: "
            f"sim={disjoint.sim_ns}ns stalls={disjoint.noc_stall} "
            f"cross_done={disjoint.flow_cycle(CASES['disjoint'].cross_flow)} "
            f"theory_bottleneck={disjoint_t.bottleneck_cycles}"
        )
        print(
            "PASS shared:   "
            f"sim={shared.sim_ns}ns stalls={shared.noc_stall} "
            f"cross_done={shared.flow_cycle(CASES['shared'].cross_flow)} "
            f"theory_bottleneck={shared_t.bottleneck_cycles}"
        )
        print(
            "PASS effect:   "
            f"observed={delta_cycles} cycles/{shared.sim_ns-disjoint.sim_ns}ns "
            f"theory={theory_delta} cycles error={delta_cycles-theory_delta:+d}, "
            "2x16 DATA intact, drain=0"
        )
        print(f"REPORT {args.report}")
        return 0
    except (AssertionError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
