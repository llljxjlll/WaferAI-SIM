#!/usr/bin/env python3
"""Serial C5 sweep for the configurable DTE control core."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_LOG = ROOT / "notes/extensions/DTE_control_core/log"
BASE_HARDWARE = HERE / "hardware/v2_channel4.json"
WORKLOAD = HERE / "workload/control_core_parallel_four.json"
SIMULATION = HERE / "simulation/v2_parallel_on.json"
MAPPING = ROOT / "llm/test/noc_congestion/mapping/identity.spec"

FINISH_RE = re.compile(r"All requests finished.*?(\d+)\s*ns")
CYCLES_RE = re.compile(r"\[SIM_RESULT\]\s+makespan_cycles=(\d+)")
CTRL_RE = re.compile(
    r"^(DTE_CTRL_(?:queue_wait|dispatch|notify))\s+.*?\b"
    r"(?:command_id|handle)=(\d+)\b"
)
DTE_RE = re.compile(
    r"^(DTE_(?:pending|launch|bus_wait|transmit))\s+"
    r"xfer=(\d+)\s+core=(\d+)\s+channel=(-?\d+)\s+"
    r"dir=([A-Z_]+)\s+bits=(\d+)$"
)
OCCUPANCY_RE = re.compile(r"\boccupancy=(\d+)\b")
KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([a-zA-Z0-9_.-]+)")


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    mode: str
    queue_depth: int = 16
    dispatch_width: int = 1
    dispatch_latency_ns: int = 0
    notify_latency_ns: int = 0
    heterogeneous: bool = False


@dataclass
class Span:
    resource: str
    stage: str
    start_ns: float
    end_ns: float
    identity: int
    core: int | None = None
    bits: int | None = None

    @property
    def duration_ns(self) -> float:
        return max(0.0, self.end_ns - self.start_ns)


def parse_csv(value: str, label: str, allow_zero: bool = False) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        try:
            number = int(part.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{label} must be comma-separated integers"
            ) from error
        if number < 0 or (number == 0 and not allow_zero):
            relation = "non-negative" if allow_zero else "positive"
            raise argparse.ArgumentTypeError(
                f"{label} values must be {relation}"
            )
        if number not in result:
            result.append(number)
    if not result:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return result


def event_ns(event: dict) -> float:
    # Event_engine exports Chrome trace timestamps in microseconds.
    return float(event["ts"]) * 1000.0


def pair_spans(events: list[dict]) -> tuple[list[Span], list[Span]]:
    ctrl_open: dict[tuple, list[float]] = defaultdict(list)
    dte_open: dict[tuple, list[float]] = defaultdict(list)
    ctrl: list[Span] = []
    dte: list[Span] = []

    for event in events:
        phase = event.get("ph")
        if phase not in {"B", "E"}:
            continue
        name = event.get("name", "")
        resource = event.get("cat", "")
        ctrl_match = CTRL_RE.match(name)
        if ctrl_match:
            stage, identity_text = ctrl_match.groups()
            identity = int(identity_text)
            key = (resource, stage, identity)
            if phase == "B":
                ctrl_open[key].append(event_ns(event))
            elif ctrl_open[key]:
                ctrl.append(
                    Span(
                        resource,
                        stage,
                        ctrl_open[key].pop(0),
                        event_ns(event),
                        identity,
                    )
                )
            continue

        dte_match = DTE_RE.match(name)
        if dte_match:
            stage, identity_text, core_text, _, _, bits_text = (
                dte_match.groups()
            )
            identity = int(identity_text)
            core = int(core_text)
            key = (resource, stage, identity, core)
            if phase == "B":
                dte_open[key].append(event_ns(event))
            elif dte_open[key]:
                dte.append(
                    Span(
                        resource,
                        stage,
                        dte_open[key].pop(0),
                        event_ns(event),
                        identity,
                        core=core,
                        bits=int(bits_text),
                    )
                )

    unmatched_ctrl = sum(len(items) for items in ctrl_open.values())
    unmatched_dte = sum(len(items) for items in dte_open.values())
    if unmatched_ctrl or unmatched_dte:
        raise RuntimeError(
            "unmatched trace spans: "
            f"controller={unmatched_ctrl}, DTE={unmatched_dte}"
        )
    return ctrl, dte


def merged_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def busy_core_time(spans: Iterable[Span]) -> float:
    by_resource: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for span in spans:
        by_resource[span.resource].append((span.start_ns, span.end_ns))
    return sum(merged_duration(items) for items in by_resource.values())


def effective_modes(hardware: dict) -> list[str]:
    grid_size = int(hardware["x"]) * int(
        hardware.get("y", hardware["x"])
    )
    global_mode = hardware.get("control_cores", {}).get(
        "mode", "legacy_shared"
    )
    explicit = sorted(hardware["cores"], key=lambda item: int(item["id"]))
    result = [global_mode] * grid_size
    previous_id = -1
    previous_mode = global_mode
    for core in explicit:
        core_id = int(core["id"])
        for missing in range(previous_id + 1, min(core_id, grid_size)):
            result[missing] = previous_mode
        mode = core.get("control_cores", {}).get("mode", global_mode)
        if core_id < grid_size:
            result[core_id] = mode
        previous_id = core_id
        previous_mode = mode
    for missing in range(previous_id + 1, grid_size):
        result[missing] = previous_mode
    return result


def controller_config(case: Case) -> dict:
    if case.mode == "legacy_shared":
        return {"mode": "legacy_shared"}
    return {
        "mode": "dual_dte_dedicated",
        "dte": {
            "command_queue_depth": case.queue_depth,
            "dispatch_width": case.dispatch_width,
            "dispatch_latency_ns": case.dispatch_latency_ns,
            "completion_notify_latency_ns": case.notify_latency_ns,
        },
    }


def hardware_for(case: Case, base: dict) -> dict:
    hardware = copy.deepcopy(base)
    hardware["control_cores"] = controller_config(case)
    if not case.heterogeneous:
        return hardware

    template = copy.deepcopy(hardware["cores"][0])
    cores = []
    # The active source core is legacy; core 1 restores the global dedicated
    # mode and terminates the legacy inheritance interval.
    for core_id, mode in (
        (0, "legacy_shared"),
        (1, "dual_dte_dedicated"),
    ):
        core = copy.deepcopy(template)
        core["id"] = core_id
        core["control_cores"] = {"mode": mode}
        cores.append(core)
    hardware["cores"] = cores
    return hardware


def make_cases(args: argparse.Namespace) -> list[Case]:
    depths = parse_csv(args.queue_depths, "queue depths")
    widths = parse_csv(args.dispatch_widths, "dispatch widths")
    dispatch_latencies = parse_csv(
        args.dispatch_latencies_ns, "dispatch latencies", allow_zero=True
    )
    notify_latencies = parse_csv(
        args.notify_latencies_ns, "notify latencies", allow_zero=True
    )
    common_depth = max(max(depths), max(widths), 4)
    cases = [
        Case("legacy_off", "mode", "legacy_shared"),
        Case("dedicated_on", "mode", "dual_dte_dedicated", 4, 1, 0, 0),
    ]
    cases.extend(
        Case(
            f"queue_depth_{depth}",
            "queue_depth",
            "dual_dte_dedicated",
            depth,
            1,
            args.depth_sweep_dispatch_latency_ns,
            0,
        )
        for depth in depths
    )
    cases.extend(
        Case(
            f"dispatch_width_{width}",
            "dispatch_width",
            "dual_dte_dedicated",
            common_depth,
            width,
            args.width_sweep_dispatch_latency_ns,
            0,
        )
        for width in widths
    )
    cases.extend(
        Case(
            f"dispatch_latency_{latency}ns",
            "dispatch_latency",
            "dual_dte_dedicated",
            common_depth,
            max(widths),
            latency,
            0,
        )
        for latency in dispatch_latencies
    )
    cases.extend(
        Case(
            f"notify_latency_{latency}ns",
            "notify_latency",
            "dual_dte_dedicated",
            common_depth,
            max(widths),
            0,
            latency,
        )
        for latency in notify_latencies
    )
    cases.append(
        Case(
            "heterogeneous_core_0_legacy",
            "heterogeneous",
            "dual_dte_dedicated",
            common_depth,
            max(widths),
            2,
            2,
            heterogeneous=True,
        )
    )
    names = set()
    unique = []
    for case in cases:
        if case.name not in names:
            names.add(case.name)
            unique.append(case)
    return unique


def production_stats(stdout: str, marker: str) -> list[dict]:
    result = []
    for line in stdout.splitlines():
        if marker not in line:
            continue
        fields = dict(KV_RE.findall(line))
        normalized = {}
        for key, value in fields.items():
            if key == "mode":
                normalized[key] = value
            else:
                try:
                    normalized[key] = int(value)
                except ValueError:
                    normalized[key] = value
        result.append(normalized)
    return result


def summarize(case: Case, hardware: dict, stdout: str, events: list[dict]) -> dict:
    finish = FINISH_RE.findall(stdout)
    cycles = CYCLES_RE.findall(stdout)
    if not finish:
        raise RuntimeError(f"{case.name}: no simulation finish time")
    sim_time_ns = int(finish[-1])
    sim_cycles = int(cycles[-1]) if cycles else None
    ctrl_spans, dte_spans = pair_spans(events)
    dispatch = [
        item for item in ctrl_spans if item.stage == "DTE_CTRL_dispatch"
    ]
    notify = [
        item for item in ctrl_spans if item.stage == "DTE_CTRL_notify"
    ]
    stalls = [
        item for item in ctrl_spans if item.stage == "DTE_CTRL_queue_wait"
    ]
    transmits = [
        item for item in dte_spans if item.stage == "DTE_transmit"
    ]
    modes = effective_modes(hardware)
    total_cores = len(modes)
    dedicated_cores = modes.count("dual_dte_dedicated")

    ctrl_busy = busy_core_time(dispatch + notify)
    dispatch_busy = busy_core_time(dispatch)
    notify_busy = busy_core_time(notify)
    queue_stall_ns = sum(item.duration_ns for item in stalls)
    dte_busy = busy_core_time(transmits)
    active_dte = len({item.resource for item in transmits})

    dispatch_batches: dict[tuple[str, float], int] = defaultdict(int)
    occupancies = []
    for event in events:
        name = event.get("name", "")
        match = CTRL_RE.match(name)
        if event.get("ph") == "B" and match:
            if match.group(1) == "DTE_CTRL_dispatch":
                dispatch_batches[
                    (event.get("cat", ""), event_ns(event))
                ] += 1
        occupancy = OCCUPANCY_RE.search(name)
        if occupancy and "DTE_CTRL_" in name:
            occupancies.append(int(occupancy.group(1)))

    ctrl_denominator = sim_time_ns * dedicated_cores
    dte_denominator = sim_time_ns * total_cores
    active_dte_denominator = sim_time_ns * active_dte
    ctrl_logs = production_stats(stdout, "[DTE_CTRL_STATS]")
    dte_logs = production_stats(stdout, "[DTE_STATS]")
    return {
        **asdict(case),
        "total_cores": total_cores,
        "dedicated_cores": dedicated_cores,
        "sim_time_ns": sim_time_ns,
        "sim_cycles": sim_cycles,
        "controller_commands": len(dispatch),
        "controller_completed": len(notify),
        "controller_busy_ns": round(ctrl_busy, 3),
        "controller_dispatch_busy_ns": round(dispatch_busy, 3),
        "controller_notify_busy_ns": round(notify_busy, 3),
        "controller_utilization": (
            ctrl_busy / ctrl_denominator if ctrl_denominator else 0.0
        ),
        "queue_stall_count": len(stalls),
        "queue_stall_ns": round(queue_stall_ns, 3),
        "queue_stall_ratio": (
            queue_stall_ns / ctrl_denominator if ctrl_denominator else 0.0
        ),
        "max_trace_queue_occupancy": max(occupancies, default=0),
        "max_dispatch_batch_per_core": max(
            dispatch_batches.values(), default=0
        ),
        "observed_dispatch_latency_ns": (
            round(mean(item.duration_ns for item in dispatch), 3)
            if dispatch
            else 0.0
        ),
        "observed_notify_latency_ns": (
            round(mean(item.duration_ns for item in notify), 3)
            if notify
            else 0.0
        ),
        "dte_transfer_count": len(transmits),
        "dte_transmit_bits": sum(item.bits or 0 for item in transmits),
        "dte_busy_ns": round(dte_busy, 3),
        "dte_utilization": (
            dte_busy / dte_denominator if dte_denominator else 0.0
        ),
        "active_dte_resources": active_dte,
        "active_dte_utilization": (
            dte_busy / active_dte_denominator
            if active_dte_denominator
            else 0.0
        ),
        "production_controller_stats": ctrl_logs,
        "production_dte_stats": dte_logs,
    }


def run_case(
    npusim: Path,
    case: Case,
    hardware: dict,
    artifacts: Path,
    timeout: int,
    keep_traces: bool,
) -> dict:
    cases_dir = artifacts / "cases"
    stdout_dir = artifacts / "stdout"
    traces_dir = artifacts / "traces"
    cases_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    if keep_traces:
        traces_dir.mkdir(parents=True, exist_ok=True)

    hardware_path = cases_dir / f"{case.name}.json"
    hardware_path.write_text(
        json.dumps(hardware, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    trace_path = npusim.parent / "events.json"
    trace_path.unlink(missing_ok=True)
    command = [
        str(npusim),
        "--trace-window",
        "1000000",
        "--workload-config",
        str(WORKLOAD),
        "--hardware-config",
        str(hardware_path),
        "--simulation-config",
        str(SIMULATION),
        "--mapping-config",
        str(MAPPING),
    ]
    proc = subprocess.run(
        command,
        cwd=npusim.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    stdout_path = stdout_dir / f"{case.name}.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{case.name}: exit={proc.returncode}; see {stdout_path}"
        )
    if not trace_path.exists():
        raise RuntimeError(f"{case.name}: no {trace_path}")
    trace_document = json.loads(trace_path.read_text(encoding="utf-8"))
    if keep_traces:
        (traces_dir / f"{case.name}.json").write_text(
            json.dumps(trace_document, separators=(",", ":")),
            encoding="utf-8",
        )
    trace_path.unlink(missing_ok=True)
    result = summarize(
        case, hardware, proc.stdout, trace_document.get("traceEvents", [])
    )
    result["hardware_config"] = str(hardware_path)
    result["stdout_log"] = str(stdout_path)
    return result


def acceptance_gates(results: list[dict]) -> list[dict]:
    by_name = {item["name"]: item for item in results}
    legacy = by_name["legacy_off"]
    dedicated = by_name["dedicated_on"]
    hetero = by_name["heterogeneous_core_0_legacy"]
    transfer_count = legacy["dte_transfer_count"]
    depth_rows = [
        item for item in results if item["family"] == "queue_depth"
    ]
    width_rows = sorted(
        (
            item
            for item in results
            if item["family"] == "dispatch_width"
        ),
        key=lambda item: item["dispatch_width"],
    )
    stats_rows = [
        item
        for item in results
        if item["production_controller_stats"]
    ]
    gates = [
        (
            "legacy has no controller activity",
            legacy["dedicated_cores"] == 0
            and legacy["controller_commands"] == 0
            and legacy["controller_utilization"] == 0,
        ),
        (
            "dedicated mode emits and completes controller commands",
            dedicated["controller_commands"] > 0
            and dedicated["controller_completed"]
            == dedicated["controller_commands"],
        ),
        (
            "all cases preserve the DTE transfer count",
            transfer_count > 0
            and all(
                item["dte_transfer_count"] == transfer_count
                for item in results
            ),
        ),
        (
            "depth=1 exercises queue backpressure",
            any(
                item["queue_depth"] == 1
                and item["queue_stall_count"] > 0
                for item in depth_rows
            ),
        ),
        (
            "dispatch batches respect configured width",
            all(
                item["max_dispatch_batch_per_core"]
                <= item["dispatch_width"]
                for item in results
                if item["mode"] == "dual_dte_dedicated"
            ),
        ),
        (
            "dispatch batch capacity is non-decreasing with width",
            all(
                left["max_dispatch_batch_per_core"]
                <= right["max_dispatch_batch_per_core"]
                for left, right in zip(width_rows, width_rows[1:])
            ),
        ),
        (
            "heterogeneous sample contains both modes",
            0 < hetero["dedicated_cores"] < hetero["total_cores"]
            and hetero["controller_commands"]
            < dedicated["controller_commands"],
        ),
        (
            "production controller stats are drained when present",
            not stats_rows
            or all(
                all(
                    stat.get("queued", 0) == 0
                    and stat.get("outstanding", 0) == 0
                    and stat.get("submitted", 0)
                    == stat.get("completed", 0)
                    for stat in item["production_controller_stats"]
                )
                for item in stats_rows
            ),
        ),
    ]
    return [{"name": name, "passed": passed} for name, passed in gates]


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def markdown_report(document: dict, command: str) -> str:
    results = document["results"]
    lines = [
        "# C5 实验与最终验收",
        "",
        f"> 生成时间：{document['generated_at_utc']}  ",
        f"> binary：'{document['npusim']}'  ",
        f"> 总体结果：**{'PASS' if document['passed'] else 'FAIL'}**",
        "",
        "## 1. 单命令复现",
        "",
        f"    {command}",
        "",
        "runner 串行执行所有 case，并对 binary 目录中的 events.json 使用",
        "非阻塞进程锁，避免多个 sweep 互相覆盖 trace。",
        "",
        "## 2. 指标口径",
        "",
        "- controller 利用率：逐 dedicated core 合并 dispatch/notify 区间，",
        "  除以 dedicated core 数乘以 sim-time。",
        "- queue stall：配对 DTE_CTRL_queue_wait 的 B/E，统计次数与累计",
        "  core-time。",
        "- DTE 利用率：逐 core 合并 DTE_transmit 区间，除以总 core 数乘以",
        "  sim-time；active DTE 利用率只按出现传输的 DTE 资源归一化。",
        "- Chrome trace 微秒时间戳乘 1000 转换为 ns。controller 与 DTE",
        "  transmit 独立统计，不重复 DTE launch、NoC 或 memory 延迟。",
        "",
        "## 3. Sweep 结果",
        "",
        "| case | family | mode | depth | width | dispatch/notify ns | dedicated cores | ctrl util | queue stalls/ns | DTE util | active DTE util | sim-time ns |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {name} | {family} | {mode} | {queue_depth} | "
            "{dispatch_width} | {dispatch_latency_ns}/{notify_latency_ns} | "
            "{dedicated_cores} | {ctrl} | "
            "{queue_stall_count}/{queue_stall_ns:g} | {dte} | {active} | "
            "{sim_time_ns} |".format(
                **item,
                ctrl=pct(item["controller_utilization"]),
                dte=pct(item["dte_utilization"]),
                active=pct(item["active_dte_utilization"]),
            )
        )
    lines.extend(
        [
            "",
            "完整 commands、observed latency、max batch、busy core-time 与",
            f"生产统计行见 '{document['result_json']}'。",
            "",
            "## 4. 验收门禁",
            "",
        ]
    )
    for gate in document["acceptance_gates"]:
        lines.append(f"- [{'x' if gate['passed'] else ' '}] {gate['name']}")

    depth_rows = [
        item for item in results if item["family"] == "queue_depth"
    ]
    width_rows = [
        item for item in results if item["family"] == "dispatch_width"
    ]
    dispatch_rows = [
        item for item in results if item["family"] == "dispatch_latency"
    ]
    notify_rows = [
        item for item in results if item["family"] == "notify_latency"
    ]
    lines.extend(
        [
            "",
            "## 5. 固定样例",
            "",
            "- legacy_off：专用控制核关闭。",
            "- dedicated_on：所有 core 使用 dedicated controller。",
            "- heterogeneous_core_0_legacy：活跃 source core 0 为 legacy；",
            "  core 1 显式恢复 dedicated 并终止继承区间。",
            "",
            "## 6. 阶段结论",
            "",
            "- queue-depth sweep stall 次数范围："
            f"{min(item['queue_stall_count'] for item in depth_rows)}–"
            f"{max(item['queue_stall_count'] for item in depth_rows)}。",
            "- dispatch-width sweep 每 core 最大批量范围："
            f"{min(item['max_dispatch_batch_per_core'] for item in width_rows)}–"
            f"{max(item['max_dispatch_batch_per_core'] for item in width_rows)}。",
            "- dispatch latency sweep sim-time："
            f"{min(item['sim_time_ns'] for item in dispatch_rows)}–"
            f"{max(item['sim_time_ns'] for item in dispatch_rows)} ns。",
            "- notify latency sweep sim-time："
            f"{min(item['sim_time_ns'] for item in notify_rows)}–"
            f"{max(item['sim_time_ns'] for item in notify_rows)} ns。",
            f"- 所有 case 均完成 {results[0]['dte_transfer_count']} 条 DTE transfer。",
            "",
            "## 7. 产物",
            "",
            f"- JSON：'{document['result_json']}'",
            f"- case 配置与 stdout：'{document['artifacts_dir']}'",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npusim",
        type=Path,
        default=ROOT / "build/npusim",
        help="path to npusim",
    )
    parser.add_argument("--queue-depths", default="1,2,4,8")
    parser.add_argument("--dispatch-widths", default="1,2,4")
    parser.add_argument("--dispatch-latencies-ns", default="0,2,4")
    parser.add_argument("--notify-latencies-ns", default="0,2,4")
    parser.add_argument(
        "--depth-sweep-dispatch-latency-ns", type=int, default=2
    )
    parser.add_argument(
        "--width-sweep-dispatch-latency-ns", type=int, default=2
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_LOG / "C5_sweep_results.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_LOG / "C5实验与最终验收.md",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_LOG / "C5_sweep_artifacts",
    )
    parser.add_argument("--keep-traces", action="store_true")
    args = parser.parse_args()

    if args.depth_sweep_dispatch_latency_ns < 0:
        parser.error("depth sweep latency must be non-negative")
    if args.width_sweep_dispatch_latency_ns < 0:
        parser.error("width sweep latency must be non-negative")
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    npusim = args.npusim.resolve()
    if not npusim.is_file():
        parser.error(f"npusim does not exist: {npusim}")
    output_json = args.output_json.resolve()
    output_markdown = args.output_markdown.resolve()
    artifacts = args.artifacts_dir.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    base = json.loads(BASE_HARDWARE.read_text(encoding="utf-8"))
    cases = make_cases(args)
    results = []
    lock_path = npusim.parent / ".dte_control_core_sweep.lock"
    try:
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"another sweep owns {lock_path}"
                ) from error
            for index, case in enumerate(cases, 1):
                print(
                    f"[{index:02d}/{len(cases):02d}] {case.name}",
                    flush=True,
                )
                results.append(
                    run_case(
                        npusim,
                        case,
                        hardware_for(case, base),
                        artifacts,
                        args.timeout,
                        args.keep_traces,
                    )
                )
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"DTE control-core sweep: FAIL: {error}", file=sys.stderr)
        return 1

    gates = acceptance_gates(results)
    try:
        binary_argument = str(npusim.relative_to(ROOT))
    except ValueError:
        binary_argument = str(npusim)
    command = (
        "python3 llm/test/dte/run_dte_control_core_sweep.py "
        f"--npusim {binary_argument}"
    )
    document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "npusim": str(npusim),
        "binary_size_bytes": npusim.stat().st_size,
        "binary_mtime_ns": npusim.stat().st_mtime_ns,
        "workload": str(WORKLOAD),
        "simulation": str(SIMULATION),
        "mapping": str(MAPPING),
        "serial_execution": True,
        "result_json": str(output_json),
        "result_markdown": str(output_markdown),
        "artifacts_dir": str(artifacts),
        "results": results,
        "acceptance_gates": gates,
        "passed": all(gate["passed"] for gate in gates),
    }
    output_json.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(
        markdown_report(document, command), encoding="utf-8"
    )
    print(
        "DTE control-core sweep: "
        f"{'PASS' if document['passed'] else 'FAIL'} "
        f"({len(results)} cases)"
    )
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_markdown}")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

