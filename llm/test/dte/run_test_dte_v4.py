#!/usr/bin/env python3
"""Run DTE V4 endpoint-resource integration tests."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict

from oracle import (V4_DIRECTION_PORTS, v4_area_um2,
                    v4_dynamic_energy_pj, v4_port_service_cycles)
from run_test_dte_v1 import BUILD, NPUSIM, run_sim
from run_test_dte_v3 import async_events

SIM = "../llm/test/dte/simulation/v4_on.json"
SIM_OFF = "../llm/test/dte/simulation/v4_off.json"
SIM_DRAM_OFF = "../llm/test/dte/simulation/v4_on_dram_off.json"
SIM_NO_ASYNC = "../llm/test/dte/simulation/v4_invalid_without_async.json"
HARDWARE = "../llm/test/dte/hardware/v4.json"
BAD_SLOTS = "../llm/test/dte/hardware/v4_invalid_slots.json"
V3_SIM = "../llm/test/dte/simulation/v3_async_on.json"
V3_HW = "../llm/test/dte/hardware/v2_channel2.json"

DIRECTION_RE = re.compile(
    r"^(DTE_(?:pending|launch|bus_wait|transmit)) xfer=(\d+) "
    r"core=(\d+) channel=(-?\d+) dir=(\w+) bits=(\d+)$"
)
PORT_RE = re.compile(
    r"^DTE_port_service xfer=(\d+) core=(\d+) channel=(\d+) "
    r"slot=(\d+) port=(\w+) dir=(\w+) bits=(\d+) width=(\d+)$"
)
STATS_RE = re.compile(
    r"^DTE_stats core=(\d+) completed=(\d+) issued=(\d+) "
    r"energy_pj=([0-9.]+) area_um2=([0-9.]+) "
    r"average_power_mw=([0-9.]+) backpressure_stalls=(\d+)$"
)

WIDTHS = {"SPM_READ": 64, "SPM_WRITE": 32,
          "AXI_READ": 16, "AXI_WRITE": 128}
PAYLOAD = 4096
CYCLE_NS = 2


def workload(name: str) -> str:
    return f"../llm/test/dte/workload/{name}.json"


def direction_events(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        match = DIRECTION_RE.match(str(event.get("name", "")))
        if not match or event.get("ph") not in {"B", "E"}:
            continue
        stage, xfer, core, channel, direction, bits = match.groups()
        result.append({"stage": stage, "xfer": int(xfer),
                       "core": int(core), "channel": int(channel),
                       "direction": direction, "bits": int(bits),
                       "phase": event["ph"],
                       "ns": round(float(event["ts"]) * 1000)})
    return result


def port_events(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        match = PORT_RE.match(str(event.get("name", "")))
        if not match or event.get("ph") not in {"B", "E"}:
            continue
        xfer, core, channel, slot, port, direction, bits, width = match.groups()
        result.append({"xfer": int(xfer), "core": int(core),
                       "channel": int(channel), "slot": int(slot),
                       "port": port, "direction": direction,
                       "bits": int(bits), "width": int(width),
                       "phase": event["ph"],
                       "ns": round(float(event["ts"]) * 1000)})
    return result


def spans(items: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, tuple[int, int]]:
    grouped: dict[tuple, dict[str, int]] = defaultdict(dict)
    for item in items:
        key = tuple(item[field] for field in key_fields)
        grouped[key][item["phase"]] = item["ns"]
    return {key: (times["B"], times["E"])
            for key, times in grouped.items() if set(times) == {"B", "E"}}


def final_stats_by_core(events: list[dict]) -> dict[int, dict]:
    parsed: dict[int, dict] = {}
    for event in events:
        match = STATS_RE.match(str(event.get("name", "")))
        if match and event.get("ph") == "E":
            core, completed, issued, energy, area, power, stalls = match.groups()
            core_id = int(core)
            parsed[core_id] = {
                "core": core_id, "completed": int(completed),
                "issued": int(issued), "energy": float(energy),
                "area": float(area), "power": float(power),
                "stalls": int(stalls),
            }
    return parsed


def final_stats(events: list[dict]) -> dict | None:
    parsed = final_stats_by_core(events)
    return list(parsed.values())[-1] if parsed else None


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    selftest = subprocess.run(
        [str(NPUSIM), "--dte-v4-selftest"], cwd=BUILD, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    tests.append(("V4 SystemC resource selftest",
                  selftest.returncode == 0 and
                  "PASS (19/19 checks)" in selftest.stdout,
                  f"exit={selftest.returncode}"))

    directions = run_sim(SIM, workload("v4_directions"), HARDWARE)
    d_events = direction_events(directions.trace_events)
    p_events = port_events(directions.trace_events)
    d_spans = spans([e for e in d_events if e["stage"] == "DTE_transmit"],
                    ("xfer",))
    p_spans = spans(p_events, ("xfer", "port"))
    expected_dirs = list(V4_DIRECTION_PORTS)
    observed_dirs = [next(e["direction"] for e in d_events
                          if e["stage"] == "DTE_pending" and
                          e["phase"] == "B" and e["xfer"] == xfer)
                     for xfer in range(6)] if len(d_spans) == 6 else []
    tests.append(("all six endpoint directions execute through WorkerCore",
                  directions.returncode == 0 and observed_dirs == expected_dirs,
                  f"finish={directions.finish_ns}, dirs={observed_dirs}"))

    exact_ports = True
    composite = True
    for xfer, direction in enumerate(expected_dirs):
        oracle = v4_port_service_cycles(direction, PAYLOAD, WIDTHS)
        actual_ports = {port for (actual_xfer, port) in p_spans
                        if actual_xfer == xfer}
        exact_ports &= actual_ports == set(oracle)
        for port, cycles in oracle.items():
            begin, end = p_spans[(xfer, port)]
            exact_ports &= end - begin == cycles * CYCLE_NS
        begin, end = d_spans[(xfer,)]
        composite &= end - begin == max(oracle.values()) * CYCLE_NS
        composite &= all(p_spans[(xfer, port)][0] == begin
                         for port in oracle)
    tests.append(("each direction uses exactly its frozen endpoint ports",
                  exact_ports, f"port_spans={p_spans}"))
    tests.append(("composite completion equals the slowest required port",
                  composite, f"transmit_spans={d_spans}"))

    duplex = run_sim(SIM, workload("v4_duplex"), HARDWARE)
    duplex_spans = spans([e for e in direction_events(duplex.trace_events)
                          if e["stage"] == "DTE_transmit"], ("xfer",))
    duplex_ports = port_events(duplex.trace_events)
    tests.append(("independent SPM read/write directions overlap",
                  duplex.returncode == 0 and len(duplex_spans) == 2 and
                  duplex_spans[(0,)][0] < duplex_spans[(1,)][1] and
                  duplex_spans[(1,)][0] < duplex_spans[(0,)][1],
                  f"spans={duplex_spans}"))
    tests.append(("one channel exposes command slots 0 and 1",
                  {e["slot"] for e in duplex_ports} == {0, 1} and
                  {e["channel"] for e in duplex_ports} == {0},
                  "slots=" + str(sorted({e["slot"] for e in duplex_ports}))))

    contention = run_sim(SIM, workload("v4_contention"), HARDWARE)
    contention_spans = spans(port_events(contention.trace_events),
                             ("xfer", "port"))
    c0 = contention_spans.get((0, "SPM_READ"))
    c1 = contention_spans.get((1, "SPM_READ"))
    tests.append(("same-port readers serialize without overlap",
                  contention.returncode == 0 and c0 is not None and
                  c1 is not None and c0[1] <= c1[0],
                  f"read_spans={c0}/{c1}"))

    backpressure = run_sim(SIM, workload("v4_backpressure"), HARDWARE)
    bp_async = async_events(backpressure.trace_events)
    issue_times = {event["token"]: event["ns"] for event in bp_async
                   if event["stage"] == "DTE_async_issue" and
                   event["phase"] == "B"}
    bp_transmit = spans([e for e in direction_events(backpressure.trace_events)
                         if e["stage"] == "DTE_transmit"], ("xfer",))
    bp_stats = final_stats(backpressure.trace_events)
    tests.append(("finite descriptor credit applies real issue backpressure",
                  backpressure.returncode == 0 and len(issue_times) == 4 and
                  issue_times[4] >= bp_transmit[(0,)][1] and
                  bp_stats is not None and bp_stats["stalls"] >= 1,
                  f"issues={issue_times}, first={bp_transmit.get((0,))}, stats={bp_stats}"))

    hazard = run_sim(SIM, workload("v4_hazard"), HARDWARE)
    hazard_async = async_events(hazard.trace_events)
    hazard_transmit = spans([e for e in direction_events(hazard.trace_events)
                             if e["stage"] == "DTE_transmit"], ("xfer",))
    hazard_seen = any(e["stage"] == "DTE_async_hazard" and
                      e["phase"] == "B" and e["token"] == 2
                      for e in hazard_async)
    tests.append(("SPM_TO_SPM destination participates in RAW hazards",
                  hazard.returncode == 0 and hazard_seen and
                  hazard_transmit[(1,)][0] >= hazard_transmit[(0,)][1],
                  f"hazard={hazard_seen}, spans={hazard_transmit}"))

    mixed = run_sim(SIM, workload("v4_mixed_send_recv_credit"), HARDWARE)
    mixed_dte = direction_events(mixed.trace_events)
    mixed_transmit = spans(
        [e for e in mixed_dte if e["stage"] == "DTE_transmit"],
        ("core", "xfer"))
    mixed_pending = {
        (e["core"], e["xfer"]): e["ns"] for e in mixed_dte
        if e["stage"] == "DTE_pending" and e["phase"] == "B"
    }
    mixed_stats = final_stats_by_core(mixed.trace_events)
    source_credit_wait = (
        mixed_pending.get((0, 3), -1) >= mixed_transmit.get((0, 0), (0, 0))[1]
    )
    destination_credit_wait = (
        mixed_pending.get((1, 3), -1) >= mixed_transmit.get((1, 0), (0, 0))[1]
    )
    tests.append((
        "blocking SEND/RECV share bounded credits with async descriptors",
        mixed.returncode == 0 and mixed.finish_ns == 24742 and
        "credits exhausted" not in mixed.stdout and
        source_credit_wait and destination_credit_wait and
        set(mixed_stats) == {0, 1} and
        all(stats["issued"] == 4 and stats["completed"] == 4 and
            stats["stalls"] >= 1 for stats in mixed_stats.values()),
        f"finish={mixed.finish_ns}, pending={mixed_pending}, "
        f"transmit={mixed_transmit}, stats={mixed_stats}"))

    stats = final_stats(directions.trace_events)
    expected_energy = sum(v4_dynamic_energy_pj(
        direction, PAYLOAD, 10.0, 0.01, 0.02)
        for direction in expected_dirs)
    expected_area = v4_area_um2(
        channels=1, command_slots=2, widths=WIDTHS,
        base_area=1000.0, channel_area=100.0, slot_area=10.0,
        port_bit_area=0.5)
    tests.append(("dynamic energy matches the independent per-port oracle",
                  stats is not None and stats["completed"] == 6 and
                  stats["issued"] == 6 and
                  abs(stats["energy"] - expected_energy) < 1e-6,
                  f"stats={stats}, expected={expected_energy}"))
    tests.append(("area and average power statistics are observable",
                  stats is not None and abs(stats["area"] - expected_area) < 1e-6
                  and stats["power"] > 0.0,
                  f"stats={stats}, expected_area={expected_area}"))
    non_dte_dram = [event for event in directions.trace_events
                    if "dram" in (str(event.get("cat", "")) +
                                   str(event.get("name", ""))).lower()
                    and "DTE_" not in str(event.get("name", ""))]
    tests.append(("endpoint directions do not duplicate DRAM media service",
                  not non_dte_dram, f"extra_dram_events={len(non_dte_dram)}"))
    dram_off = run_sim(SIM_DRAM_OFF, workload("v4_directions"), HARDWARE)
    dram_off_ports = spans(port_events(dram_off.trace_events),
                           ("xfer", "port"))
    dram_off_stats = final_stats(dram_off.trace_events)
    tests.append(("behavioral/DRAMSys switches do not double-charge endpoint service",
                  dram_off.returncode == 0 and
                  dram_off_ports == p_spans and
                  dram_off.finish_ns == directions.finish_ns and
                  dram_off_stats is not None and stats is not None and
                  dram_off_stats["energy"] == stats["energy"],
                  f"on/off={directions.finish_ns}/{dram_off.finish_ns}"))

    off_direction = run_sim(SIM_OFF, workload("v4_invalid_direction_off"),
                            HARDWARE)
    tests.append(("V4 direction requires the fine-resource gate",
                  off_direction.returncode != 0 and
                  "requires fine_grained_resources=true" in off_direction.stdout,
                  f"exit={off_direction.returncode}"))
    no_async = run_sim(SIM_NO_ASYNC, workload("v4_duplex"), HARDWARE)
    tests.append(("fine resources require async mode",
                  no_async.returncode != 0 and
                  "requires dte.async=true" in no_async.stdout,
                  f"exit={no_async.returncode}"))
    bad_slots = run_sim(SIM, workload("v4_duplex"), BAD_SLOTS)
    tests.append(("V4 hardware requires exactly two command slots",
                  bad_slots.returncode != 0 and
                  "command_slots_per_channel=2" in bad_slots.stdout,
                  f"exit={bad_slots.returncode}"))

    legacy = run_sim(V3_SIM, workload("v3_overlap"), V3_HW)
    tests.append(("V4 disabled preserves the frozen V3a timing",
                  legacy.returncode == 0 and legacy.finish_ns == 345 and
                  not port_events(legacy.trace_events),
                  f"finish={legacy.finish_ns}"))

    passed = 0
    for name, ok, detail in tests:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        passed += int(ok)
    print(f"DTE V4 integration: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
