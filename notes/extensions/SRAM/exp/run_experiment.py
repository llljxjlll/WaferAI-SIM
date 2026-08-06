#!/usr/bin/env python3
"""Run and analyze the single-die manually partitioned SRAM GEMM+RS experiment."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILD = ROOT / "build"
NPUSIM = BUILD / "npusim"
TRACE = BUILD / "events.json"
RESULTS = HERE / "results"
MAPPING = ROOT / "llm/test/noc_collective/mapping/identity.spec"

TILE_RE = re.compile(
    r"\[GEMM_RS_TILE\] core=(\d+) chunk=(\d+) owner=(\d+) "
    r"slot=(\S+) hbm_store_ns=([0-9.eE+-]+) "
    r"hbm_load_ns=([0-9.eE+-]+) weight_ns=([0-9.eE+-]+) "
    r"gemm_ns=([0-9.eE+-]+) swizzle_ns=([0-9.eE+-]+) "
    r"checksum=(\d+) ready_ns=([0-9.eE+-]+)"
)
REDUCE_RE = re.compile(
    r"\[GEMM_RS_REDUCE\] core=(\d+) chunk=(\d+) contributors=(\d+) "
    r"reduce_ns=([0-9.eE+-]+) checksum=(\d+) done_ns=([0-9.eE+-]+)"
)
WINDOW_RE = re.compile(
    r"\[GEMM_RS_OWNER_WINDOW\] core=(\d+) chunk=(\d+) "
    r"wait_ns=([0-9.eE+-]+)"
)

REGIONS = [
    ("gemm_input_a", 0, 8192, "even GEMM input/prefetch"),
    ("gemm_input_b", 8192, 8192, "odd GEMM input/prefetch"),
    ("gemm_weight_b", 16384, 8192, "resident weight"),
    ("gemm_accum", 24576, 8192, "2 intermediate chunks"),
    ("rs_send_a", 32768, 512, "even communication buffer"),
    ("rs_send_b", 33280, 512, "odd communication buffer"),
    ("rs_recv", 33792, 512, "owner receive/reduce buffer"),
    ("scratch", 34304, 512, "final local RS chunk"),
]
CAPACITY = 40960
PARTICIPANTS = 2


def primitive(core: int, chunk: int, mode: int = 0) -> dict:
    label = f"rs_core{core}_chunk{chunk}" if mode == 0 else f"rs_final_{core}"
    return {
        "type": "Gemm_rs_swizzle",
        "mode": mode,
        "chunk": chunk,
        "tile_bytes": 4096,
        "comm_bytes": 512,
        "compute_cycles": 512,
        "reduce_cycles": 32,
        "hbm_base": 0x10000 + core * 0x10000,
        "participants": PARTICIPANTS,
        "dram_address": {"input": 0, "data": 0, "output": 0},
        "sram_address": {"indata": "_manual_input", "outdata": label},
    }


def workload() -> dict:
    cores = []
    for core in range(PARTICIPANTS):
        worklist = []
        if core == 0:
            startup = primitive(core, core, mode=3)
            startup["compute_cycles"] = 128
            worklist.append({
                "recv_cnt": 0, "prims": [startup], "cast": [],
            })
        for chunk in range(PARTICIPANTS):
            work = {"recv_cnt": 0, "prims": [primitive(core, chunk)]}
            if core != chunk:
                work["cast"] = [{"dest": chunk, "tag": 7000 + chunk}]
            else:
                work["cast"] = []
            worklist.append(work)
            if core == chunk:
                arrival = primitive(core, core, mode=3)
                if core == 0:
                    arrival["compute_cycles"] = 0
                worklist.append({
                    "recv_cnt": 0, "prims": [arrival], "cast": [],
                })
                worklist.append({
                    "recv_cnt": PARTICIPANTS - 1,
                    "recv_tag": 7000 + core,
                    "prims": [primitive(core, core, mode=1)],
                    "cast": [],
                })
        if core == 0:
            worklist.append({
                "recv_cnt": 0,
                "prims": [primitive(core, core, mode=2)],
                "cast": [{"dest": 1, "tag": 8000}],
            })
            park = primitive(core, core, mode=3)
            park["compute_cycles"] = 8192
            worklist.append({"recv_cnt": 0, "prims": [park], "cast": []})
            worklist.append({"recv_cnt": 0,
                             "prims": [primitive(core, core, mode=2)],
                             "cast": [{"dest": -1, "loopout": "true"}]})
        else:
            worklist.append({
                "recv_cnt": 1, "recv_tag": 8000,
                "prims": [primitive(core, core, mode=2)], "cast": [],
            })
            worklist.append({"recv_cnt": 0,
                             "prims": [primitive(core, core, mode=2)],
                             "cast": [{"dest": -1, "loopout": "true"}]})
        cores.append({"id": core, "loop": 1, "worklist": worklist})
    return {
        "vars": {"B": 1, "T": 1},
        "pipeline": 1,
        "source": [],
        "chips": [{"chip_id": 0, "cores": cores}],
    }


def spans(events: list[dict]) -> list[dict]:
    opened: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    result = []
    for event in events:
        key = (str(event.get("cat", "")), int(event.get("tid", -1)),
               str(event.get("name", "")))
        if event.get("ph") == "B":
            opened[key].append(float(event["ts"]))
        elif event.get("ph") == "E" and opened[key]:
            begin = opened[key].pop(0)
            result.append({"cat": key[0], "tid": key[1], "name": key[2],
                           "begin": begin * 1000.0,
                           "end": float(event["ts"]) * 1000.0})
    return result


def overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["begin"], b["begin"]))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return ordered[index]


def markdown(summary: dict) -> str:
    op = summary["operation_breakdown_ns"]
    comm = summary["communication_breakdown_ns"]
    occupancy = summary["occupancy"]
    rows = "\n".join(
        f"| `{r['name']}` | {r['base']}–{r['end']} | {r['bytes']} | "
        f"{r['reserved_pct']:.3f}% | {r['purpose']} |"
        for r in occupancy["regions"]
    )
    op_rows = "\n".join(
        f"| {name} | {stats['count']} | {stats['mean']:.3f} | "
        f"{stats['p95']:.3f} | {stats['total']:.3f} |"
        for name, stats in op.items()
    )
    comm_rows = "\n".join(
        f"| {name} | {stats['count']} | {stats['mean']:.3f} | "
        f"{stats['p95']:.3f} | {stats['total']:.3f} |"
        for name, stats in comm.items()
    )
    core_rows = "\n".join(
        f"| {core} | {item['eligible_flows']} | {item['overlapped_flows']} | "
        f"{item['payload_overlapped_flows']} | {item['overlap_ns']:.3f} |"
        for core, item in summary["overlap_by_core"].items()
    )
    return f"""# 单 die GEMM + Reduce-Scatter SRAM 实验报告

## 1. 结论

实验 **PASS**。单个 2×2 mesh die 上启用 2 个核，完成 {summary['tile_markers']} 个 GEMM chunk、{summary['data_flows']} 个 2-way direct Reduce-Scatter DATA flow 和 {summary['reduce_markers']} 个 owner reduce；真实 SRAM 数据、LSU HBM↔SRAM checksum、trace phase 与网络 drain 均通过。{summary['overlap']['eligible_flows']} 个具备后继 GEMM 的 flow 检测到 {summary['overlap']['overlapped_flows']} 个完整通信事务重叠和 {summary['overlap']['payload_overlapped_flows']} 个 DATA-only 重叠；累计事务交叠 {summary['overlap']['total_overlap_ns']:.3f} ns。

运行命令：`python3 notes/extensions/SRAM/exp/run_experiment.py`

## 2. 实验与验收摘要

- 运行完成时间：{summary['makespan_ns']:.3f} ns
- GEMM tile marker：{summary['tile_markers']}（期望 {PARTICIPANTS ** 2}）
- owner reduce marker：{summary['reduce_markers']}（期望 {PARTICIPANTS}）
- NoC RS DATA flow：{summary['data_flows']}（期望 {PARTICIPANTS * (PARTICIPANTS - 1)}）；completion DATA flow：{summary['completion_flows']}
- `Compute_tile` span：{summary['compute_spans']}（期望 {PARTICIPANTS * (PARTICIPANTS + 1)}）
- drain：`{summary['drain_marker']}`
- SRAM/LSU/Compute trace：B/E phase 平衡
- checksum：{summary['tile_markers']} 个 swizzle 与 {summary['reduce_markers']} 个 reduce 均非零

## 3. SRAM 占用分析（每核）

| 功能区 | byte 地址 | 预留容量/B | SRAM 占比 | 用途 |
|---|---:|---:|---:|---|
{rows}

- 静态预留：{occupancy['reserved_bytes']} B / {CAPACITY} B = {occupancy['reserved_pct']:.3f}%
- 未划分 headroom：{occupancy['free_bytes']} B = {occupancy['free_pct']:.3f}%
- 估算峰值 live payload：{occupancy['peak_live_bytes']} B = {occupancy['peak_live_pct']:.3f}% SRAM；占已预留空间 {occupancy['peak_of_reserved_pct']:.3f}%
- live 峰值组成：双 input buffer 8192 B + weight 4096 B + 两个 accum chunk 8192 B + 双 send payload 1024 B + recv 512 B + scratch 512 B。预留容量大于 live 数据是有意的：保留 tile/协议扩展空间，并用 fixed non-spillable 区域保证角色不互相侵占。

## 4. 数据与计算操作时间 breakdown

单位为 ns；`total` 是跨 2 个活跃核/全部 chunk 的累计服务区间，不等于 makespan（核间并行且不同阶段会重叠）。

| 操作 | 次数 | mean/ns | p95/ns | total/ns |
|---|---:|---:|---:|---:|
{op_rows}

`HBM store` 在每核 chunk 0 初始化两个 tile 的 backing，`HBM load/prefetch` 包括当前 tile load 与计算期间的下一 tile 异步预取；`weight round-trip` 每核执行一次；`GEMM` 包括 SRAM 输入读、512-cycle 计算、权重读和 accumulator 写；`swizzle+label` 包括双 buffer 写及生产标签绑定；`owner reduce` 包括 `rs_recv` 写/读、32-cycle reduce、local accumulator 读和 scratch 写；`schedule windows` 包括 128-cycle startup guard、0-cycle core0 arrival、512-cycle core1 arrival 和 8192-cycle core0 park。

## 5. 通信时间 breakdown

| NoC/接收操作 | 次数 | mean/ns | p95/ns | total/ns |
|---|---:|---:|---:|---:|
{comm_rows}

这些区间取自 `events.json` 中 WorkerCore 的 REQUEST、ACK、DATA 与接收原语 B/E trace。ACK 包含 owner arrival window 的等待；DATA 为 512 B、32 packet 的发送区间。二者分别列出，事务重叠按 REQUEST 开始到 DATA 完成计算，DATA-only 重叠另行验收。OWNER_RECV 只统计持续时间大于 20 ns 的真实阻塞接收，过滤 recv_cnt=0 的调度原语。RS_* 与 COMPLETION_* 分别统计 2 条 RS flow 和 1 条 core0→core1 completion flow；OWNER_RECV 包含两次 RS owner receive 与一次 completion receive。重叠分析只使用 RS flow。

## 6. Swizzling 重叠证据

| core | 有后继 GEMM 的发送 | 事务重叠 | DATA-only 重叠 | 累计 overlap/ns |
|---:|---:|---:|---:|---:|
{core_rows}

匹配规则按每核发送顺序，把 chunk `c` 的 REQUEST→DATA 完整事务及 DATA 子区间分别与同核 chunk `c+1` 的真实 `Compute_tile` 求交。最终 chunk 1 没有后继 GEMM，因此不作为“应重叠”分母。core0 的 128-cycle startup guard 与 0-cycle arrival 调整两核相位，使 core1 先进入后继 GEMM，32-packet DATA 的后半段与之重叠。双 buffer 映射为偶数 chunk→`rs_send_a`、奇数 chunk→`rs_send_b`；执行器在提交下一发送批次前等待上一批完成，所以复用同一 slot 前有顺序保障。

## 7. 能力边界

本实验确认的是：真实 SRAM storage/access、LSU HBM 双向搬运、手工功能区、发送前真实 SRAM 读取，以及真实 NoC 控制/数据时序的计算通信重叠。普通 NoC DATA 目前仍只携带元数据，owner 端远程 tensor bytes 由确定性 payload 建模后写入 `rs_recv`；因此报告不宣称完成了跨核逐字节 GEMM tensor 的数值 Reduce-Scatter。若要验证端到端数值 RS，需要后续扩展 `Msg`/DTE remote path 传递真实 payload 并由 `Recv_prim` 写入 `rs_recv`。

## 8. 产物

- `results/workload.generated.json`：本次运行的完整 worklist
- `results/events.json`：Chrome trace 原始事件
- `results/run.log`：仿真器标准输出
- `results/summary.json`：本报告的机器可读统计
"""


def stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": mean(values),
            "p95": percentile(values, 0.95), "total": sum(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    generated = RESULTS / "workload.generated.json"
    generated.write_text(json.dumps(workload(), indent=2) + "\n")

    if not args.skip_build:
        build = subprocess.run(["cmake", "--build", str(BUILD), "-j2"],
                               cwd=ROOT, text=True)
        if build.returncode:
            return build.returncode
    TRACE.unlink(missing_ok=True)
    command = [
        str(NPUSIM), "--trace-window", "1000000",
        "--workload-config", str(generated),
        "--hardware-config", str(HERE / "hardware.json"),
        "--simulation-config", str(HERE / "simulation.json"),
        "--mapping-config", str(MAPPING),
    ]
    try:
        proc = subprocess.run(command, cwd=BUILD, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=90)
    except subprocess.TimeoutExpired as error:
        (RESULTS / "run.log").write_text(error.stdout or "")
        print("FAIL: npusim timed out; see results/run.log")
        return 1
    (RESULTS / "run.log").write_text(proc.stdout)
    if proc.returncode or not TRACE.exists():
        print(proc.stdout[-3000:])
        print(f"FAIL: npusim rc={proc.returncode}, trace={TRACE.exists()}")
        return 1
    shutil.copy2(TRACE, RESULTS / "events.json")

    tile_rows = [m.groups() for m in TILE_RE.finditer(proc.stdout)]
    reduce_rows = [m.groups() for m in REDUCE_RE.finditer(proc.stdout)]
    window_rows = [m.groups() for m in WINDOW_RE.finditer(proc.stdout)]
    if len(tile_rows) != PARTICIPANTS ** 2 or len(reduce_rows) != PARTICIPANTS or len(window_rows) != 4:
        print(f"FAIL: marker counts tile/reduce={len(tile_rows)}/{len(reduce_rows)}")
        return 1
    if any(int(r[9]) <= 0 for r in tile_rows) or any(int(r[4]) <= 0 for r in reduce_rows):
        print("FAIL: zero checksum")
        return 1

    events = json.loads(TRACE.read_text())["traceEvents"]
    all_spans = spans(events)
    for prefix in ("SRAM_", "LSU_", "Compute_"):
        staged = [e for e in events if str(e.get("cat", "")).startswith(prefix)
                  and e.get("ph") in ("B", "E")]
        if not staged or sum(e["ph"] == "B" for e in staged) != sum(
                e["ph"] == "E" for e in staged):
            print(f"FAIL: absent or unbalanced {prefix} trace")
            return 1
    compute_by_core = {}
    all_compute_by_core = {}
    for core in range(PARTICIPANTS):
        all_compute_by_core[core] = sorted(
            [s for s in all_spans if s["cat"] == f"Compute_{core}" and
             s["tid"] == 0], key=lambda s: s["begin"])
        compute_by_core[core] = [
            s for s in all_compute_by_core[core] if "bytes=4096" in s["name"]]
    compute_count = sum(map(len, all_compute_by_core.values()))

    def core_from_cat(category: str) -> int | None:
        match = re.search(r"Core\s+(?:0x)?([0-9a-fA-F]+)$", category)
        return int(match.group(1), 16) if match else None

    network = defaultdict(list)
    for span in all_spans:
        core = core_from_cat(span["cat"])
        if core is None:
            continue
        name = span["name"]
        if "SEND_REQ" in name:
            network[(core, "REQUEST")].append(span)
        elif "RECV_ACK" in name:
            network[(core, "ACK")].append(span)
        elif "SEND_DATA" in name:
            network[(core, "DATA")].append(span)
        elif name == "Receive_primRECV_DATA":
            network[(core, "RECV")].append(span)
    for value in network.values():
        value.sort(key=lambda s: s["begin"])

    overlap_by_core = {}
    eligible = overlapped = 0
    total_overlap = 0.0
    payload_overlapped = 0
    data_flows = 0
    for core in range(PARTICIPANTS):
        sent_chunks = [chunk for chunk in range(PARTICIPANTS) if chunk != core]
        all_data = network[(core, "DATA")]
        expected_data = len(sent_chunks) + (1 if core == 0 else 0)
        data = all_data[:len(sent_chunks)]
        data_flows += len(data)
        if (len(all_data) != expected_data or len(compute_by_core[core]) != PARTICIPANTS or
                len(all_compute_by_core[core]) != PARTICIPANTS + 1):
            print(f"FAIL: core {core} data/gemm/all-compute="
                  f"{len(data)}/{len(compute_by_core[core])}/"
                  f"{len(all_compute_by_core[core])}")
            return 1
        core_eligible = core_overlapped = core_payload_overlapped = 0
        core_overlap = 0.0
        for chunk, flow in zip(sent_chunks, data):
            if chunk == PARTICIPANTS - 1:
                continue
            core_eligible += 1
            request = network[(core, "REQUEST")][sent_chunks.index(chunk)]
            transaction = {"begin": request["begin"], "end": flow["end"]}
            amount = overlap(transaction, compute_by_core[core][chunk + 1])
            if overlap(flow, compute_by_core[core][chunk + 1]) > 0:
                core_payload_overlapped += 1
            if amount > 0:
                core_overlapped += 1
                core_overlap += amount
        eligible += core_eligible
        overlapped += core_overlapped
        total_overlap += core_overlap
        payload_overlapped += core_payload_overlapped
        overlap_by_core[str(core)] = {
            "eligible_flows": core_eligible,
            "overlapped_flows": core_overlapped,
            "overlap_ns": core_overlap,
            "payload_overlapped_flows": core_payload_overlapped,
        }
    if data_flows != PARTICIPANTS * (PARTICIPANTS - 1) or compute_count != PARTICIPANTS * (PARTICIPANTS + 1) or overlapped != eligible or payload_overlapped != eligible or eligible < 1:
        print(f"FAIL: flows/compute/overlap={data_flows}/{compute_count}/{overlapped}/{eligible}")
        return 1

    tile_fields = {
        "HBM store": [float(r[4]) for r in tile_rows if float(r[4]) > 0],
        "HBM load/prefetch": [float(r[5]) for r in tile_rows if float(r[5]) > 0],
        "weight round-trip": [float(r[6]) for r in tile_rows if float(r[6]) > 0],
        "GEMM": [float(r[7]) for r in tile_rows],
        "swizzle+label": [float(r[8]) for r in tile_rows],
        "owner reduce": [float(r[3]) for r in reduce_rows],
        "schedule windows": [float(r[2]) for r in window_rows],
    }
    comm_fields = {}
    for kind in ("REQUEST", "ACK", "DATA"):
        rs_spans = []
        for core in range(PARTICIPANTS):
            rs_count = PARTICIPANTS - 1
            rs_spans.extend(network[(core, kind)][:rs_count])
        comm_fields["RS_" + kind] = stats(
            [s["end"] - s["begin"] for s in rs_spans])
        completion = network[(0, kind)][PARTICIPANTS - 1:]
        comm_fields["COMPLETION_" + kind] = stats(
            [s["end"] - s["begin"] for s in completion])
    owner_recv = [s["end"] - s["begin"] for core in range(PARTICIPANTS)
                  for s in network[(core, "RECV")]
                  if s["end"] - s["begin"] > 20.0]
    comm_fields["OWNER_RECV"] = stats(owner_recv)

    reserved = sum(row[2] for row in REGIONS)
    peak_live = 8192 + 4096 + 8192 + 1024 + 512 + 512
    region_rows = [
        {"name": name, "base": base, "end": base + size - 1,
         "bytes": size, "reserved_pct": size / CAPACITY * 100, "purpose": purpose}
        for name, base, size, purpose in REGIONS
    ]
    makespan = max((float(e.get("ts", 0)) * 1000.0 for e in events), default=0.0)
    drain = "router_residual=0; credits balanced" if (
        "[DRAIN] router_residual=0" in proc.stdout and
        "data_balanced=1 ctrl_balanced=1" in proc.stdout) else "missing"
    if drain == "missing" or "Catch test finished" not in proc.stdout:
        print("FAIL: simulation did not report completion/drain")
        return 1
    summary = {
        "status": "PASS",
        "makespan_ns": makespan,
        "tile_markers": len(tile_rows),
        "reduce_markers": len(reduce_rows),
        "data_flows": data_flows,
        "completion_flows": 1,
        "compute_spans": compute_count,
        "drain_marker": drain,
        "trace_balanced": True,
        "occupancy": {
            "capacity_bytes": CAPACITY, "reserved_bytes": reserved,
            "reserved_pct": reserved / CAPACITY * 100,
            "free_bytes": CAPACITY - reserved,
            "free_pct": (CAPACITY - reserved) / CAPACITY * 100,
            "peak_live_bytes": peak_live,
            "peak_live_pct": peak_live / CAPACITY * 100,
            "peak_of_reserved_pct": peak_live / reserved * 100,
            "regions": region_rows,
        },
        "operation_breakdown_ns": {k: stats(v) for k, v in tile_fields.items()},
        "communication_breakdown_ns": comm_fields,
        "overlap": {"eligible_flows": eligible,
                    "overlapped_flows": overlapped,
                    "total_overlap_ns": total_overlap,
                    "payload_overlapped_flows": payload_overlapped},
        "overlap_by_core": overlap_by_core,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "实验报告.md").write_text(markdown(summary))
    print("PASS: manually partitioned SRAM GEMM+RS experiment")
    print(f"  tile/reduce/data={len(tile_rows)}/{len(reduce_rows)}/{data_flows}")
    print(f"  overlap={overlapped}/{eligible}, total={total_overlap:.3f} ns")
    print(f"  reserved/peak-live={reserved}/{peak_live} B")
    print(f"  report={HERE / '实验报告.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
