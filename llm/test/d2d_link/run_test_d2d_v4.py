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


def workload(packets=4):
    with open(WL) as f:
        wl = json.load(f)
    wl["vars"]["OC"] = packets * 16
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

    passed = sum(ok for _, ok, _ in results)
    print(f"V4 runner: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
