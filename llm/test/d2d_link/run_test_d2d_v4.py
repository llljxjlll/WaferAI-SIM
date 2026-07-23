#!/usr/bin/env python3
"""V4 Behavioral D2D production regression and independent-oracle comparison."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BUILD = os.path.join(ROOT, "build")
NPUSIM = os.path.join(BUILD, "npusim")
SIM_BEHA = "../llm/test/noc_congestion/sim/sim_beha.json"
SIM_CYCLE = "../llm/test/noc_congestion/sim/sim_cycle.json"
MAP = "../llm/test/noc_congestion/mapping/identity.spec"
WL = os.path.join(HERE, "workload", "cross_die_2core.json")
HW21 = os.path.join(HERE, "hardware", "core_4x4_die2x1_c2c.json")
HW31 = os.path.join(HERE, "hardware", "core_4x4_die3x1_c2c.json")
HW22 = os.path.join(HERE, "hardware", "core_4x4_die2x2_c2c.json")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

sys.path.insert(0, os.path.join(HERE, "behavioral"))
from oracle import estimate  # noqa: E402

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def behavioral_hw(path=HW21, port=(1, 1), link=(1, 1), latency=7):
    with open(path) as f:
        hw = json.load(f)
    hw["die_ports"]["c2c"] = {
        "backend": "behavioral",
        "port_rate": {"num": port[0], "den": port[1]},
        "link_rate": {"num": link[0], "den": link[1]},
        "link_latency": latency,
    }
    return hw


def workload(packets=4, source=0, dest=16, tag=None):
    with open(WL) as f:
        wl = json.load(f)
    wl["vars"]["OC"] = packets * 16
    tag = dest if tag is None else tag
    wl["source"][0]["dest"] = source
    wl["chips"][0]["cores"][0]["id"] = source
    wl["chips"][0]["cores"][0]["worklist"][0]["cast"] = [{"dest": dest, "tag": tag}]
    wl["chips"][0]["cores"][1]["id"] = dest
    wl["chips"][0]["cores"][1]["worklist"][0]["recv_tag"] = tag
    return wl


def temp_json(obj):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def run_case(wl, hw, sim=SIM_BEHA, timeout=60):
    wp, hp = temp_json(wl), temp_json(hw)
    try:
        p = subprocess.run(
            [NPUSIM, "--workload-config", wp, "--hardware-config", hp,
             "--simulation-config", sim, "--mapping-config", MAP],
            cwd=BUILD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout)
        return parse(p.returncode, ANSI.sub("", p.stdout))
    except subprocess.TimeoutExpired as e:
        out = e.output if isinstance(e.output, str) else "<timeout>"
        return parse(124, ANSI.sub("", out))
    finally:
        os.remove(wp)
        os.remove(hp)


def fields(pattern, out):
    m = re.search(pattern, out)
    return tuple(map(int, m.groups())) if m else None


def parse(rc, out):
    ns = None
    for line in out.splitlines():
        if "requests finished" in line or "Catch test finished" in line:
            m = re.search(r"(\d+) ns", line)
            if m:
                ns = int(m.group(1))
    beha = fields(r"\[D2D_BEHA\] data_flows=(\d+) logical_data_packets=(\d+) "
                  r"service_cycles=(\d+) fixed_cycles=(\d+) "
                  r"total_d2d_cycles=(\d+)", out)
    typed = fields(r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
                   r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)", out)
    data = fields(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)", out)
    repin = fields(r"\[D2D_REPIN\] total=(\d+) changed=(\d+) same=(\d+)", out)
    return {
        "rc": rc, "out": out, "ns": ns, "beha": beha, "typed": typed,
        "data": data, "repin": repin,
        "drained": "[DRAIN] router_residual=0" in out and
                   "[DRAIN] d2d_link_residual=0" in out,
        "watchdog": "[PROTO_WAIT]" in out,
    }


def main():
    if not os.path.exists(NPUSIM):
        sys.exit(f"npusim not found at {NPUSIM}; build it first")

    # V4-c: production branch carries one representative per phase while retaining F.
    packets = 4
    hw = behavioral_hw(latency=7)
    expected = estimate(hw, 0, 16, packets)
    a = run_case(workload(packets), hw)
    b = run_case(workload(packets), hw)
    expected_stats = (1, packets, expected["bulk_service_cycles"],
                      expected["transaction_link_latency_cycles"],
                      expected["transaction_d2d_cycles"])
    record("V4-c production Behavioral e2e completes and drains",
           a["rc"] == b["rc"] == 0 and a["drained"] and b["drained"] and
           not a["watchdog"] and not b["watchdog"],
           f"ns={a['ns']}/{b['ns']} residual={a['drained']}/{b['drained']}")
    record("V4-c REQUEST/ACK/DATA use one representative message per link",
           a["typed"] == b["typed"] == (1, 1, 1, 1, 1, 1) and
           a["data"] == b["data"] == (1, 1) and
           a["repin"] == b["repin"] == (3, 3, 0),
           f"typed={a['typed']} data={a['data']} repin={a['repin']}")
    record("V4-c runtime cycle ledger equals independent oracle",
           a["beha"] == b["beha"] == expected_stats,
           f"runtime={a['beha']} oracle={expected_stats}")

    bad = run_case(workload(packets), hw, sim=SIM_CYCLE)
    record("V4-c Behavioral backend rejects cycle NoC before simulation",
           bad["rc"] != 0 and "requires noc.use_beha_noc=true" in bad["out"] and
           bad["ns"] is None,
           f"rc={bad['rc']}")

    # V4-d1: F boundary sweep. The physical wire remains three representatives;
    # only logical count/service changes, including boundaries around old depth=8.
    sizes = (1, 7, 8, 9, 128)
    size_runs = {}
    for f in sizes:
        h = behavioral_hw(latency=7)
        r = run_case(workload(f), h)
        e = estimate(h, 0, 16, f)
        want = (1, f, e["bulk_service_cycles"],
                e["transaction_link_latency_cycles"], e["transaction_d2d_cycles"])
        size_runs[f] = (r, want)
    record("V4-d message-size boundaries preserve representatives and exact S(F)",
           all(r["rc"] == 0 and r["drained"] and r["typed"] == (1, 1, 1, 1, 1, 1)
               and r["beha"] == want for r, want in size_runs.values()),
           " ".join(f"F={f}:{r['beha']}" for f, (r, _) in size_runs.items()))

    # V4-d2: rational min-cut. Non-limiting port/link changes must not alter service;
    # non-integer 2/3 verifies ceil rather than float/truncation.
    rates = {
        "full": ((1, 1), (1, 1)),
        "port_half": ((1, 2), (1, 1)),
        "link_half": ((1, 1), (1, 2)),
        "both_half": ((1, 2), (3, 4)),
        "two_thirds": ((2, 3), (1, 1)),
        "quarter": ((1, 1), (1, 4)),
    }
    rate_runs = {}
    for name, (pr, lr) in rates.items():
        h = behavioral_hw(port=pr, link=lr, latency=7)
        r = run_case(workload(7), h)
        e = estimate(h, 0, 16, 7)
        rate_runs[name] = (r, e)
    record("V4-d rational port/link min-cut matches oracle exactly",
           all(r["rc"] == 0 and r["drained"] and r["beha"] ==
               (1, 7, e["bulk_service_cycles"], e["transaction_link_latency_cycles"],
                e["transaction_d2d_cycles"]) for r, e in rate_runs.values()) and
           rate_runs["port_half"][0]["beha"] == rate_runs["link_half"][0]["beha"] ==
           rate_runs["both_half"][0]["beha"],
           " ".join(f"{n}:S={r['beha'][2] if r['beha'] else None}"
                    for n, (r, _) in rate_runs.items()))
    base_rate = rate_runs["full"][0]
    timing_rate_ok = all(
        r["ns"] - base_rate["ns"] ==
        2 * (e["bulk_service_cycles"] -
             rate_runs["full"][1]["bulk_service_cycles"])
        for r, e in rate_runs.values())
    record("V4-d measured completion shifts by CYCLE*delta(S(F))",
           timing_rate_ok,
           " ".join(f"{n}:{r['ns']}ns" for n, (r, _) in rate_runs.items()))

    # V4-d3: latency and hop slopes. A two-hop route charges fixed delay twice but
    # never repeats bulk service; representative crossing/re-pin counts scale with H.
    latencies = (0, 1, 7, 20)
    latency_runs = {}
    for lat in latencies:
        h1 = behavioral_hw(HW21, latency=lat)
        h2 = behavioral_hw(HW31, latency=lat)
        r1 = run_case(workload(7, 0, 16), h1)
        r2 = run_case(workload(7, 0, 32), h2)
        latency_runs[lat] = (r1, r2, estimate(h1, 0, 16, 7),
                             estimate(h2, 0, 32, 7))
    slope_ok = all(
        r1["beha"] == (1, 7, e1["bulk_service_cycles"], 3 * lat,
                        e1["transaction_d2d_cycles"]) and
        r2["beha"] == (1, 7, e2["bulk_service_cycles"], 6 * lat,
                        e2["transaction_d2d_cycles"]) and
        r1["typed"] == (1, 1, 1, 1, 1, 1) and r1["repin"] == (3, 3, 0) and
        r2["typed"] == (2, 2, 2, 2, 2, 2) and r2["repin"][0] == 6
        for lat, (r1, r2, e1, e2) in latency_runs.items())
    record("V4-d latency/hop law: fixed=3*H*L and bulk service once",
           slope_ok,
           " ".join(f"L={l}:H1={v[0]['beha']},H2={v[1]['beha']}"
                    for l, v in latency_runs.items()))
    base1, base2 = latency_runs[0][0]["ns"], latency_runs[0][1]["ns"]
    timing_hop_ok = all(
        r1["ns"] - base1 == 2 * 3 * lat and
        r2["ns"] - base2 == 2 * 3 * 2 * lat
        for lat, (r1, r2, _, _) in latency_runs.items())
    record("V4-d measured completion slope is CYCLE*3*H*L",
           timing_hop_ok,
           " ".join(f"L={l}:H1={v[0]['ns']}ns,H2={v[1]['ns']}ns"
                    for l, v in latency_runs.items()))


    passed = sum(ok for _, ok, _ in results)
    print(f"V4 runner: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
