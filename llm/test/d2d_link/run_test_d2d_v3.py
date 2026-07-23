#!/usr/bin/env python3
"""V3 production bounded_saf regression: bottleneck, contention, mixed NoC, SAF safety."""
import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BUILD = os.path.join(ROOT, "build")
NPUSIM = os.path.join(BUILD, "npusim")
SIM = "../llm/test/noc_congestion/sim/sim_cycle.json"
MAP = "../llm/test/noc_congestion/mapping/identity.spec"
WL_BASE = os.path.join(HERE, "workload", "cross_die_2core.json")
HW21 = os.path.join(HERE, "hardware", "core_4x4_die2x1_c2c.json")
HW31 = os.path.join(HERE, "hardware", "core_4x4_die3x1_c2c.json")
HW22 = os.path.join(HERE, "hardware", "core_4x4_die2x2_c2c.json")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def temp_json(obj):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def bounded_hw(path, port=(1, 1), link=(1, 1), latency=1, saf=32,
               inflight=64, rx=2, ctrl=4):
    h = json.load(open(path))
    h["die_ports"]["c2c"] = {
        "mode": "bounded_saf", "safety": "whole_flow_saf",
        "port_rate": {"num": port[0], "den": port[1]},
        "link_rate": {"num": link[0], "den": link[1]},
        "link_latency": latency, "saf_buffer_depth": saf,
        "link_inflight_depth": inflight, "rx_buffer_depth": rx,
        "ctrl_buffer_depth": ctrl,
    }
    return h


def make_workload(flows, oc=512):
    """flows=(source,dest,tag); OC=16*network packet count for this Matmul fixture."""
    t = json.load(open(WL_BASE))
    src_t = t["chips"][0]["cores"][0]
    dst_t = t["chips"][0]["cores"][1]
    t["vars"]["OC"] = oc
    t["source"] = []
    t["chips"][0]["cores"] = []
    for source, dest, tag in flows:
        s = copy.deepcopy(src_t)
        s["id"] = source
        s["worklist"][0]["cast"] = [{"dest": dest, "tag": tag}]
        d = copy.deepcopy(dst_t)
        d["id"] = dest
        d["worklist"][0]["recv_tag"] = tag
        t["source"].append({"dest": source, "size": "BTP"})
        t["chips"][0]["cores"].extend((s, d))
    return t


def run_case(workload, hardware, timeout=90):
    wp, hp = temp_json(workload), temp_json(hardware)
    try:
        p = subprocess.run(
            [NPUSIM, "--workload-config", wp, "--hardware-config", hp,
             "--simulation-config", SIM, "--mapping-config", MAP],
            cwd=BUILD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout)
        out = ANSI.sub("", p.stdout)
        return parse_output(p.returncode, out)
    except subprocess.TimeoutExpired as e:
        out = e.output if isinstance(e.output, str) else "<timeout>"
        return parse_output(124, ANSI.sub("", out))
    finally:
        os.remove(wp)
        os.remove(hp)


def parse_csv(s):
    return [int(x) for x in s.split(",") if x != ""]


def parse_output(rc, out):
    ns = None
    for line in out.splitlines():
        if "requests finished" in line or "Catch test finished" in line:
            m = re.search(r"(\d+) ns", line)
            if m:
                ns = int(m.group(1))
    dm = re.search(r"\[D2D_DATA\].*in_pkts=(\d+) out_pkts=(\d+).*"
                   r"out_inorder=(\d+).*in_first_cycle=(-?\d+) in_last_cycle=(-?\d+) "
                   r"out_first_cycle=(-?\d+) out_last_cycle=(-?\d+)", out)
    bounds = {}
    for m in re.finditer(
            r"\[D2D_BOUND\] idx=(\d+) saf_peak=(\d+) inflight_peak=(\d+) rx_peak=(\d+) "
            r"saf_full=(\d+) inflight_full=(\d+) rx_full=(\d+) port_stall=(\d+) "
            r"link_stall=(\d+) inflight_stall=(\d+) rx_stall=(\d+) downstream_stall=(\d+)", out):
        bounds[int(m.group(1))] = tuple(map(int, m.groups()[1:]))
    links = {}
    for m in re.finditer(
            r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=([WENS]) "
            r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
            r"data_in=(\d+) data_out=(\d+)", out):
        links[(int(m.group(2)), int(m.group(3)))] = {
            "idx": int(m.group(1)), "dir": m.group(4),
            "req": (int(m.group(5)), int(m.group(6))),
            "ack": (int(m.group(7)), int(m.group(8))),
            "data": (int(m.group(9)), int(m.group(10))),
        }
    nm = re.search(r"\[NOC_ACT\] sends=([0-9,]*) stalls=([0-9,]*) d2d_source_stalls=(\d+)", out)
    flows = {}
    fm = re.search(r"\[FLOW_DONE\]\s*([^\n]*)", out)
    if fm:
        for source, tag, dest, cyc in re.findall(r"(\d+):(\d+):(\d+)@(\d+)", fm.group(1)):
            flows[(int(source), int(tag), int(dest))] = int(cyc)
    sm = re.search(r"\[SAF_ADMIT\] success=(\d+) reject=(\d+)", out)
    cm = re.search(r"\[CREDIT\] data_balanced=(\d+) ctrl_balanced=(\d+)", out)
    return {
        "rc": rc, "out": out, "ns": ns,
        "data": tuple(map(int, dm.groups())) if dm else None,
        "bounds": bounds, "links": links, "flows": flows,
        "noc_sends": parse_csv(nm.group(1)) if nm else None,
        "noc_stalls": parse_csv(nm.group(2)) if nm else None,
        "d2d_source_stalls": int(nm.group(3)) if nm else None,
        "admit": tuple(map(int, sm.groups())) if sm else None,
        "credit_balanced": bool(cm and cm.group(1) == cm.group(2) == "1"),
        "drained": ("[DRAIN] router_residual=0" in out and
                    "[DRAIN] d2d_link_residual=0" in out and
                    "[SAF] reserved_packets=0" in out and
                    bool(cm and cm.group(1) == cm.group(2) == "1")),
        "watchdog": "[PROTO_WAIT]" in out,
    }


def flow_done(r, source, tag, dest):
    return r["flows"].get((source, tag, dest))


def single_integrity(r, packets):
    return (r["rc"] == 0 and r["drained"] and not r["watchdog"] and
            r["data"] is not None and r["data"][0] == r["data"][1] == packets and
            r["data"][2] == 1)


def main():
    if not os.path.exists(NPUSIM):
        sys.exit(f"npusim not found at {NPUSIM}; build it first")

    # V3-e1: inline analytic bottleneck oracle. Single signal fixes NoC ceiling at 1 packet/cycle.
    # For one long flow, measured service goodput=(N-1)/(last_in-first_in); expected=min(1,port,link).
    packets = 128
    long_wl = make_workload([(0, 16, 16)], oc=packets * 16)
    sweeps = {
        "noc": ((1, 1), (1, 1)),
        "port": ((1, 2), (1, 1)),
        "link": ((1, 1), (1, 2)),
        "port_nonbottleneck": ((1, 2), (3, 4)),
        "link_nonbottleneck": ((3, 4), (1, 2)),
        "link_slow": ((1, 1), (1, 4)),
    }
    swept = {}
    for name, (pr, lr) in sweeps.items():
        swept[name] = run_case(long_wl, bounded_hw(HW21, pr, lr, saf=packets, inflight=64, rx=4))
    gp = {}
    for name, r in swept.items():
        if r["data"]:
            span = r["data"][4] - r["data"][3]
            gp[name] = (packets - 1) / span if span > 0 else 0.0
    expected = {name: min(1.0, pr[0]/pr[1], lr[0]/lr[1])
                for name, (pr, lr) in sweeps.items()}
    accuracy = all(single_integrity(r, packets) and name in gp and
                   abs(gp[name] - expected[name]) / expected[name] < 0.01
                   for name, r in swept.items())
    record("V3-e bottleneck oracle: long-flow goodput within 1% of min(NoC,port,link)",
           accuracy, f"goodput={gp} expected={expected}")

    pb = swept["port"]["bounds"].get(0)
    lb = swept["link"]["bounds"].get(0)
    stalls_ok = (pb is not None and lb is not None and pb[6] > 0 and pb[7] == 0 and
                 lb[6] == 0 and lb[7] > 0)
    record("V3-e bottleneck attribution: port/link stall counters identify the active limiter",
           stalls_ok, f"port_bound={pb} link_bound={lb}")
    record("V3-e non-bottleneck invariance + bottleneck transfer",
           gp.get("port") == gp.get("port_nonbottleneck") and
           gp.get("link") == gp.get("link_nonbottleneck") and
           gp.get("link_slow", 0) < gp.get("link", 0) < gp.get("noc", 0),
           f"goodput={gp}")

    # V3-e2: contention and physical-link independence/full duplex.
    one = run_case(make_workload([(0, 16, 16)]), bounded_hw(HW21, saf=64))
    shared = run_case(make_workload([(0, 16, 16), (1, 17, 17)]),
                      bounded_hw(HW21, saf=64))
    independent = run_case(make_workload([(0, 16, 16), (32, 48, 48)]),
                           bounded_hw(HW22, saf=32))
    duplex = run_case(make_workload([(0, 16, 16), (17, 1, 1)]),
                      bounded_hw(HW21, saf=32))
    shared_link = shared["links"].get((0, 1), {}).get("data") == (64, 64)
    fair = (flow_done(shared, 0, 16, 16) is not None and
            flow_done(shared, 1, 17, 17) is not None)
    record("V3-e shared link: two flows conserve 64 packets, both complete, no starvation",
           shared["rc"] == 0 and shared["drained"] and shared_link and fair and
           shared["ns"] > one["ns"],
           f"single_ns={one['ns']} shared_ns={shared['ns']} flows={shared['flows']}")
    indep_links = (independent["links"].get((0, 1), {}).get("data") == (32, 32) and
                   independent["links"].get((2, 3), {}).get("data") == (32, 32))
    record("V3-e independent links: two flows use disjoint physical units and complete in parallel",
           independent["rc"] == 0 and independent["drained"] and indep_links and
           independent["ns"] < shared["ns"],
           f"independent_ns={independent['ns']} shared_ns={shared['ns']} links={independent['links']}")
    duplex_ok = (duplex["links"].get((0, 1), {}).get("data") == (32, 32) and
                 duplex["links"].get((1, 0), {}).get("data") == (32, 32))
    record("V3-e full duplex: opposite directions have independent capacity",
           duplex["rc"] == 0 and duplex["drained"] and duplex_ok and
           duplex["ns"] <= shared["ns"],
           f"duplex_ns={duplex['ns']} links={duplex['links']}")

    # V3-e3: Local+D2D, same hop count for local reference (1->3 vs 5->7).
    local = run_case(make_workload([(1, 3, 3)]), bounded_hw(HW21, saf=32))
    d2d = run_case(make_workload([(0, 16, 16)]), bounded_hw(HW21, saf=32))
    disjoint = run_case(make_workload([(0, 16, 16), (5, 7, 7)]), bounded_hw(HW21, saf=32))
    mixed = run_case(make_workload([(0, 16, 16), (1, 3, 3)]), bounded_hw(HW21, saf=32))
    record("V3-e mixed disjoint: Local+D2D completes with zero source-die NoC stall",
           all(r["rc"] == 0 and r["drained"] for r in (local, d2d, disjoint)) and
           disjoint["noc_stalls"][0] == 0,
           f"local={local['flows']} d2d={d2d['flows']} disjoint={disjoint['flows']} "
           f"stalls={disjoint['noc_stalls']}")
    record("V3-e mixed shared: shared on-die path raises D2D latency and NoC stall only there",
           mixed["rc"] == 0 and mixed["drained"] and mixed["noc_stalls"][0] > 0 and
           mixed["noc_stalls"][0] > disjoint["noc_stalls"][0] and
           flow_done(mixed, 0, 16, 16) > flow_done(disjoint, 0, 16, 16),
           f"shared_flow={mixed['flows']} disjoint_flow={disjoint['flows']} "
           f"stalls={mixed['noc_stalls']}/{disjoint['noc_stalls']}")

    # V3-e4: Local traffic on the intermediate die, sharing the exact row traversed by D2D.
    # Duplicate the local source compute once so its DATA overlaps the relayed flow deterministically.
    shared_mid_wl = make_workload([(0, 32, 32), (17, 19, 19)])
    disjoint_mid_wl = make_workload([(0, 32, 32), (21, 23, 23)])
    for w in (shared_mid_wl, disjoint_mid_wl):
        prim = w["chips"][0]["cores"][2]["worklist"][0]["prims"][0]
        w["chips"][0]["cores"][2]["worklist"][0]["prims"] = [
            copy.deepcopy(prim), copy.deepcopy(prim)]
    mid_hw = bounded_hw(HW31, saf=64, inflight=8, rx=1)
    mid_shared = run_case(shared_mid_wl, mid_hw)
    mid_disjoint = run_case(disjoint_mid_wl, mid_hw)
    record("V3-e intermediate-die mixed traffic: shared mesh row stalls and delays relayed D2D only",
           mid_shared["rc"] == mid_disjoint["rc"] == 0 and
           mid_shared["drained"] and mid_disjoint["drained"] and
           mid_shared["noc_stalls"][1] > 0 and mid_disjoint["noc_stalls"][1] == 0 and
           flow_done(mid_shared, 0, 32, 32) > flow_done(mid_disjoint, 0, 32, 32),
           f"shared={mid_shared['flows']} disjoint={mid_disjoint['flows']} "
           f"stalls={mid_shared['noc_stalls']}/{mid_disjoint['noc_stalls']}")

    # A stronger intermediate hotspot fills the destination-facing router input. With rx=1 and
    # inflight=4 this must propagate backward through RX and inflight into SAF drain. Whole-flow SAF
    # intentionally cuts the dependency before the source NoC, so source stall remains zero after store.
    bp_wl = make_workload([(0, 32, 32), (16, 19, 19),
                           (17, 19, 19), (18, 19, 19)])
    bp_cores = bp_wl["chips"][0]["cores"]
    for idx in (2, 4, 6):
        prim = bp_cores[idx]["worklist"][0]["prims"][0]
        bp_cores[idx]["worklist"][0]["prims"] = [
            copy.deepcopy(prim), copy.deepcopy(prim)]
    bp_cores[3]["worklist"][0]["recv_cnt"] = 3
    bp_wl["chips"][0]["cores"] = [bp_cores[i] for i in (0, 1, 2, 3, 4, 6)]
    bp_runs = [run_case(bp_wl, bounded_hw(HW31, saf=128, inflight=4, rx=1))
               for _ in range(2)]
    bp = bp_runs[0]["bounds"].get(0)
    record("V3-e production backpressure: remote NoC -> RX -> inflight -> SAF, then drains",
           all(r["rc"] == 0 and r["drained"] and len(r["flows"]) == 4
               for r in bp_runs) and bp is not None and
           bp[4] > 0 and bp[5] > 0 and bp[8] > 0 and bp[9] > 0 and bp[10] > 0 and
           bp_runs[0]["d2d_source_stalls"] == 0 and
           bp_runs[0]["ns"] == bp_runs[1]["ns"] and
           bp_runs[0]["bounds"] == bp_runs[1]["bounds"],
           f"ns={bp_runs[0]['ns']}/{bp_runs[1]['ns']} bound={bp} "
           f"noc_stalls={bp_runs[0]['noc_stalls']} "
           f"source_stalls={bp_runs[0]['d2d_source_stalls']}")

    # Multi-hop SAF + capacity safety.
    hop = run_case(make_workload([(0, 32, 32)]), bounded_hw(HW31, saf=32))
    hop_data = (hop["links"].get((0, 1), {}).get("data") == (32, 32) and
                hop["links"].get((1, 2), {}).get("data") == (32, 32))
    hop_peaks = [b[0] for i, b in hop["bounds"].items() if i in
                 {hop["links"].get((0, 1), {}).get("idx"),
                  hop["links"].get((1, 2), {}).get("idx")}]
    record("V3-e multi-hop SAF: every forward boundary stores the whole flow and releases to zero",
           hop["rc"] == 0 and hop["drained"] and hop_data and hop["admit"] == (1, 0) and
           sorted(hop_peaks) == [32, 32],
           f"ns={hop['ns']} peaks={hop_peaks} links={hop['links']} admit={hop['admit']}")

    diagonal = run_case(make_workload([(0, 48, 48), (49, 1, 1)]),
                        bounded_hw(HW22, saf=64, inflight=8, rx=1, ctrl=1))
    diag_expected = {(0, 1): "E", (1, 3): "N", (3, 2): "W", (2, 0): "S"}
    diag_links_ok = all(
        diagonal["links"].get(edge, {}).get("dir") == direction and
        diagonal["links"].get(edge, {}).get("data") == (32, 32)
        for edge, direction in diag_expected.items())
    record("V3-e 2x2 bidirectional diagonal: four directions, two-hop SAF, min RX/CTRL drain",
           diagonal["rc"] == 0 and diagonal["drained"] and not diagonal["watchdog"] and
           diagonal["credit_balanced"] and len(diagonal["flows"]) == 2 and diag_links_ok,
           f"ns={diagonal['ns']} flows={diagonal['flows']} links={diagonal['links']}")

    # Fixed four-flow permutation: every one of the 2x2 mesh's eight directed links carries exactly
    # one F=4 flow, at the minimum legal SAF capacity and depth-1 RX/CTRL endpoints.
    perm_flows = [(0, 49, 49), (16, 33, 33), (32, 17, 17), (48, 1, 1)]
    permutation = run_case(make_workload(perm_flows, oc=64),
                           bounded_hw(HW22, saf=4, inflight=4, rx=1, ctrl=1))
    perm_links_ok = len(permutation["links"]) == 8 and all(
        st["req"] == st["ack"] == (1, 1) and st["data"] == (4, 4)
        for st in permutation["links"].values())
    record("V3-e minimum-buffer 2x2 permutation: all directed links progress and drain",
           permutation["rc"] == 0 and permutation["drained"] and
           permutation["credit_balanced"] and not permutation["watchdog"] and
           len(permutation["flows"]) == 4 and perm_links_ok,
           f"ns={permutation['ns']} flows={permutation['flows']} "
           f"links={permutation['links']}")

    overbook = run_case(make_workload([(0, 16, 16), (1, 17, 17)]),
                        bounded_hw(HW21, saf=32))
    record("V3-e concurrent SAF overbook: rejected before rejected flow REQUEST/DATA injection",
           overbook["rc"] not in (0, 124) and
           "whole-flow SAF path admission rejected before REQUEST/DATA" in overbook["out"] and
           "insufficient_capacity" in overbook["out"],
           f"exit={overbook['rc']} timeout={overbook['rc']==124}")

    # V3-e5: finite RX depth changes transient full-state only, not steady bottleneck throughput.
    rx1 = run_case(make_workload([(0, 16, 16), (1, 17, 17)], oc=2048),
                   bounded_hw(HW21, saf=256, inflight=4, rx=1))
    rx8 = run_case(make_workload([(0, 16, 16), (1, 17, 17)], oc=2048),
                   bounded_hw(HW21, saf=256, inflight=4, rx=8))
    b1 = rx1["bounds"].get(0)
    b8 = rx8["bounds"].get(0)
    record("V3-e finite RX buffer: depth=1 reaches full; depth=8 does not; steady completion unchanged",
           rx1["rc"] == rx8["rc"] == 0 and rx1["drained"] and rx8["drained"] and
           b1 is not None and b8 is not None and b1[5] > 0 and b8[5] == 0 and
           rx1["ns"] == rx8["ns"],
           f"ns={rx1['ns']}/{rx8['ns']} rx1={b1} rx8={b8}")

    # V3-e6: the registered ready signal is not sufficient for a depth-1 control FIFO.
    # Eight concurrent flows force REQUEST and ACK bursts through the real credit path.
    ctrl_flows = [(i, 16 + i, 16 + i) for i in range(8)]
    ctrl_pressure = run_case(make_workload(ctrl_flows, oc=64),
                             bounded_hw(HW21, latency=20, saf=32,
                                        inflight=42, rx=1, ctrl=1))
    fwd = ctrl_pressure["links"].get((0, 1), {})
    rev = ctrl_pressure["links"].get((1, 0), {})
    record("V3-e depth-1 control FIFO: credit flow handles 8 concurrent REQUEST/ACK bursts",
           ctrl_pressure["rc"] == 0 and ctrl_pressure["drained"] and
           ctrl_pressure["credit_balanced"] and not ctrl_pressure["watchdog"] and
           len(ctrl_pressure["flows"]) == 8 and
           fwd.get("req") == (8, 8) and fwd.get("data") == (32, 32) and
           rev.get("ack") == (8, 8),
           f"ns={ctrl_pressure['ns']} flows={len(ctrl_pressure['flows'])} "
           f"credit_balanced={ctrl_pressure['credit_balanced']} links={ctrl_pressure['links']}")

    passed = sum(ok for _, ok, _ in results)
    print(f"\n==== D2D V3: {passed}/{len(results)} test groups passed ====")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
