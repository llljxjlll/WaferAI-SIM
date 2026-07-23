#!/usr/bin/env python3
"""V5 multi-port striping production regression."""
from __future__ import annotations

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
WL = os.path.join(HERE, "workload", "cross_die_2core.json")
WL2 = os.path.join(HERE, "workload", "cross_die_2flow.json")
HW = os.path.join(HERE, "hardware", "core_4x4_die2x1_c2c_multi4.json")
SIM = "../llm/test/noc_congestion/sim/sim_cycle.json"
SIM_BEHA = "../llm/test/noc_congestion/sim/sim_beha.json"
MAP = "../llm/test/noc_congestion/mapping/identity.spec"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
sys.path.insert(0, os.path.join(HERE, "behavioral"))
from oracle import estimate_striped  # noqa: E402
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def load(path):
    with open(path) as f:
        return json.load(f)


def bounded_hw(saf_depth, shared_group=False, policy="tag_hash"):
    h = load(HW)
    c = h["die_ports"]["c2c"]
    c["select_policy"] = policy
    for field in ("link_bw", "latency", "buffer_depth"):
        c.pop(field, None)
    c.update({
        "mode": "bounded_saf", "safety": "whole_flow_saf",
        "port_rate": {"num": 1, "den": 1},
        "link_rate": {"num": 1, "den": 1},
        "link_latency": 7,
        "saf_buffer_depth": saf_depth,
        "link_inflight_depth": 32,
        "rx_buffer_depth": 4,
        "ctrl_buffer_depth": 4,
    })
    if shared_group:
        for p in h["die_ports"]["overrides"]:
            if p["role"] == "c2c":
                p["link_group"] = 0
    return h


def behavioral_hw(shared_group=False):
    h = load(HW)
    c = h["die_ports"]["c2c"]
    for field in ("link_bw", "latency", "buffer_depth"):
        c.pop(field, None)
    c.update({
        "backend": "behavioral", "select_policy": "hybrid",
        "port_rate": {"num": 1, "den": 4},
        "link_rate": {"num": 1, "den": 4},
        "link_latency": 7,
    })
    if shared_group:
        for p in h["die_ports"]["overrides"]:
            if p["role"] == "c2c":
                p["link_group"] = 0
    return h


def workload(packets, stripes, dest=16):
    w = load(WL)
    w["vars"]["OC"] = packets * 16
    cast = w["chips"][0]["cores"][0]["worklist"][0]["cast"][0]
    cast["dest"] = dest
    cast["tag"] = dest
    cast["stripe"] = stripes
    w["chips"][0]["cores"][1]["id"] = dest
    recv = w["chips"][0]["cores"][1]["worklist"][0]
    recv["recv_tag"] = dest
    recv["recv_stripe"] = stripes
    return w


def workload_two(packets, stripes):
    w = load(WL2)
    w["vars"]["OC"] = packets * 16
    for core in w["chips"][0]["cores"]:
        if core["id"] in (0, 1):
            core["worklist"][0]["cast"][0]["stripe"] = stripes
        elif core["id"] in (16, 17):
            core["worklist"][0]["recv_stripe"] = stripes
    return w


def temp_json(obj):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def run_case(packets, stripes, timeout=60, hw=None, sim=SIM, wl=None):
    wp = temp_json(wl if wl is not None else workload(packets, stripes))
    hp = temp_json(hw) if hw is not None else None
    try:
        p = subprocess.run(
            [NPUSIM, "--workload-config", wp, "--hardware-config", hp or HW,
             "--simulation-config", sim, "--mapping-config", MAP],
            cwd=BUILD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout)
        return parse(p.returncode, ANSI.sub("", p.stdout))
    except subprocess.TimeoutExpired as e:
        out = e.output if isinstance(e.output, str) else "<timeout>"
        return parse(124, ANSI.sub("", out))
    finally:
        os.remove(wp)

        if hp:
            os.remove(hp)

def fields(pattern, out):
    m = re.search(pattern, out)
    return tuple(map(int, m.groups())) if m else None


def parse(rc, out):
    subflows = {}
    pat = re.compile(
        r"\[V5_SUBFLOW\] idx=(\d+) source=(\d+) tag=(\d+) subflow=(\d+) "
        r"in=(\d+) out=(\d+) in_seqhash=(\d+) out_seqhash=(\d+) "
        r"in_csum=(\d+) out_csum=(\d+) inorder=(\d+) minseq=(-?\d+) "
        r"maxseq=(-?\d+) endseq=(-?\d+) ends=(\d+) end_length=(-?\d+)")
    for vals in pat.findall(out):
        nums = tuple(map(int, vals))
        idx, source, tag, subflow = nums[:4]
        subflows[(idx, source, tag, subflow)] = nums[4:]
    typed = fields(
        r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
        r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)", out)
    repin = fields(r"\[D2D_REPIN\] total=(\d+) changed=(\d+) same=(\d+)", out)
    admit = fields(r"\[SAF_ADMIT\] success=(\d+) reject=(\d+)", out)
    beha = fields(r"\[D2D_BEHA\] data_flows=(\d+) logical_data_packets=(\d+) "
                  r"service_cycles=(\d+) fixed_cycles=(\d+) "
                  r"total_d2d_cycles=(\d+)", out)
    finish = re.search(r"All requests finished.*?\| (\d+) ns", out)
    group_stalls = {
        int(idx): int(stall)
        for idx, stall in re.findall(
            r"\[D2D_BOUND\] idx=(\d+).*?group_stall=(\d+)", out)
    }
    dynamic_match = re.search(
        r"\[V5_DYNAMIC\] selections=(\d+) releases=(\d+) "
        r"active=(\d+) loads=([0-9,]*)", out)
    dynamic = None
    if dynamic_match:
        dynamic = (int(dynamic_match.group(1)), int(dynamic_match.group(2)),
                   int(dynamic_match.group(3)),
                   [int(x) for x in dynamic_match.group(4).split(",") if x])
    return {
        "rc": rc, "out": out, "subflows": subflows, "typed": typed,
        "repin": repin, "admit": admit, "beha": beha,
        "ns": int(finish.group(1)) if finish else None,
        "group_stalls": group_stalls, "dynamic": dynamic,
        "drained": "[DRAIN] router_residual=0" in out and
                   "[DRAIN] d2d_link_residual=0" in out,
        "saf_zero": "[SAF] reserved_packets=0" in out,
        "watchdog": "[PROTO_WAIT]" in out,
        "group_zero": "group_reserved_packets=0" in out,
    }


def expected_quotas(packets, stripes):
    q, r = divmod(packets, stripes)
    return [q + (1 if s < r else 0) for s in range(stripes)]


def check_run(run, packets, stripes):
    buckets = {}
    for (idx, source, tag, subflow), stat in run["subflows"].items():
        if source == 0 and tag == 16:
            buckets[subflow] = (idx, stat)
    quotas = expected_quotas(packets, stripes)
    if set(buckets) != set(range(stripes)):
        return False, f"subflow keys={sorted(buckets)}"
    for s, quota in enumerate(quotas):
        _, st = buckets[s]
        (inp, outp, inh, outh, inc, outc, inorder, minseq, maxseq,
         endseq, ends, _) = st
        if not (inp == outp == quota and inh == outh and inc == outc and
                inorder == 1 and minseq == 1 and maxseq == endseq == quota and
                ends == 1):
            return False, f"sf{s}={st} quota={quota}"
    distinct_links = len({buckets[s][0] for s in buckets})
    if distinct_links != stripes:
        return False, f"links={sorted(buckets[s][0] for s in buckets)}"
    want_typed = (stripes, stripes, stripes, stripes, packets, packets)
    ok = (run["rc"] == 0 and run["typed"] == want_typed and run["drained"] and
          run["saf_zero"] and not run["watchdog"])
    return ok, (f"quotas={quotas} links={sorted(buckets[s][0] for s in buckets)} "
                f"typed={run['typed']} repin={run['repin']}")


def check_flow(run, packets, stripes, source, tag):
    buckets = {}
    for (idx, src, flow_tag, subflow), stat in run["subflows"].items():
        if src == source and flow_tag == tag:
            buckets[subflow] = (idx, stat)
    quotas = expected_quotas(packets, stripes)
    if set(buckets) != set(range(stripes)):
        return False, [], f"{source}:{tag} keys={sorted(buckets)}"
    for subflow, quota in enumerate(quotas):
        _, st = buckets[subflow]
        (inp, outp, inh, outh, inc, outc, inorder, minseq, maxseq,
         endseq, ends, _) = st
        if not (inp == outp == quota and inh == outh and inc == outc and
                inorder == 1 and minseq == 1 and maxseq == endseq == quota and
                ends == 1):
            return False, [], f"{source}:{tag}:sf{subflow}={st}"
    links = [buckets[s][0] for s in range(stripes)]
    return len(set(links)) == stripes, links, f"links={links} quotas={quotas}"


def check_multihop_flow(run, packets, stripes, source, tag, hops):
    buckets = {s: [] for s in range(stripes)}
    for (idx, src, flow_tag, subflow), stat in run["subflows"].items():
        if src == source and flow_tag == tag and subflow in buckets:
            buckets[subflow].append((idx, stat))
    quotas = expected_quotas(packets, stripes)
    for subflow, quota in enumerate(quotas):
        entries = buckets[subflow]
        if len(entries) != hops:
            return False, f"sf{subflow} links={len(entries)} expected={hops}"
        for idx, st in entries:
            (inp, outp, inh, outh, inc, outc, inorder, minseq, maxseq,
             endseq, ends, _) = st
            if not (inp == outp == quota and inh == outh and inc == outc and
                    inorder == 1 and minseq == 1 and
                    maxseq == endseq == quota and ends == 1):
                return False, f"sf{subflow}={st} quota={quota}"
    links = [idx for entries in buckets.values() for idx, _ in entries]
    want_typed = (stripes * hops, stripes * hops,
                  stripes * hops, stripes * hops, packets * hops, packets * hops)
    ok = (run["rc"] == 0 and run["typed"] == want_typed and run["drained"] and
          run["saf_zero"] and not run["watchdog"] and
          len(links) == stripes * hops and len(set(links)) == len(links))
    return ok, "quotas={} links={} typed={}".format(
        quotas, sorted(links), run["typed"])


def main():
    if not os.path.exists(NPUSIM):
        sys.exit(f"npusim not found at {NPUSIM}; build it first")
    for stripes in (1, 2, 4):
        a, b = run_case(7, stripes), run_case(7, stripes)
        oka, da = check_run(a, 7, stripes)
        okb, db = check_run(b, 7, stripes)
        record(f"V5-b/c k={stripes}: q/r split, per-subflow integrity and drain",
               oka and okb, da if oka else da + " | " + a["out"][-1200:])
        record(f"V5-b/c k={stripes}: static selection deterministic across runs",
               a["subflows"] == b["subflows"] and a["typed"] == b["typed"],
               f"A={da} B={db}")

    bad = run_case(3, 4)
    record("V5-b/c rejects F<stripe_count before DATA injection",
           bad["rc"] != 0 and "at least one packet per subflow" in bad["out"],
           f"rc={bad['rc']}")

    bounded = run_case(7, 4, hw=bounded_hw(4))
    ok, detail = check_run(bounded, 7, 4)
    record("V5-d striped SAF admits all four independent subflows atomically",
           ok and bounded["admit"] == (1, 0) and bounded["group_zero"],
           f"{detail} admit={bounded['admit']} group0={bounded['group_zero']}")

    no_partial = run_case(7, 2, hw=bounded_hw(4, policy="nearest"))
    record("V5-d per-link capacity failure rejects before every REQUEST",
           no_partial["rc"] != 0 and no_partial["typed"] is None and
           "before any REQUEST: link capacity" in no_partial["out"],
           f"rc={no_partial['rc']} typed={no_partial['typed']}")

    group_bad = run_case(7, 4, hw=bounded_hw(4, shared_group=True))
    record("V5-d shared link_group capacity is checked across all subflows",
           group_bad["rc"] != 0 and group_bad["typed"] is None and
           "shared link_group capacity" in group_bad["out"],
           f"rc={group_bad['rc']} typed={group_bad['typed']}")

    group_ok = run_case(7, 4, hw=bounded_hw(8, shared_group=True))
    ok, detail = check_run(group_ok, 7, 4)
    record("V5-d admitted shared group releases both link and group ledgers",
           ok and group_ok["admit"] == (1, 0) and group_ok["group_zero"],
           f"{detail} admit={group_ok['admit']} group0={group_ok['group_zero']}")

    independent_long = run_case(31, 4, hw=bounded_hw(32))
    shared_long = run_case(31, 4, hw=bounded_hw(32, shared_group=True))
    independent_ok, _ = check_run(independent_long, 31, 4)
    shared_ok, _ = check_run(shared_long, 31, 4)
    shared_forward_stalls = [shared_long["group_stalls"].get(i, 0)
                             for i in range(4)]
    record("V5-d link_group enforces one shared DATA cut with fair progress",
           independent_ok and shared_ok and
           shared_long["ns"] > independent_long["ns"] and
           all(stall > 0 for stall in shared_forward_stalls),
           f"independent={independent_long['ns']}ns "
           f"shared={shared_long['ns']}ns "
           f"group_stalls={shared_forward_stalls} quotas=[8,8,8,7]")


    beha_ind_hw = behavioral_hw(False)
    beha_group_hw = behavioral_hw(True)
    oracle_ind = estimate_striped(beha_ind_hw, 0, 16, 16, 31, 4)
    oracle_group = estimate_striped(beha_group_hw, 0, 16, 16, 31, 4)
    beha_ind_a = run_case(31, 4, hw=beha_ind_hw, sim=SIM_BEHA)
    beha_ind_b = run_case(31, 4, hw=beha_ind_hw, sim=SIM_BEHA)
    beha_group_a = run_case(31, 4, hw=beha_group_hw, sim=SIM_BEHA)
    beha_group_b = run_case(31, 4, hw=beha_group_hw, sim=SIM_BEHA)
    expected_ind = (1, 31, oracle_ind["bulk_service_cycles"],
                    oracle_ind["transaction_link_latency_cycles"],
                    oracle_ind["transaction_d2d_cycles"])
    expected_group = (1, 31, oracle_group["bulk_service_cycles"],
                      oracle_group["transaction_link_latency_cycles"],
                      oracle_group["transaction_d2d_cycles"])
    ind_rate = oracle_ind["effective_rate"]
    group_rate = oracle_group["effective_rate"]
    ind_service = oracle_ind["bulk_service_cycles"]
    group_service = oracle_group["bulk_service_cycles"]
    ind_ledger = beha_ind_a["beha"]
    group_ledger = beha_group_a["beha"]
    ind_ns = beha_ind_a["ns"]
    group_ns = beha_group_a["ns"]
    record("V5-e oracle: source NoC caps four independent lanes; shared group counts once",
           ind_rate == [1, 1] and ind_service == 31 and
           group_rate == [1, 4] and group_service == 124,
           f"ind={ind_rate}/S{ind_service} group={group_rate}/S{group_service}")
    record("V5-e runtime ledger equals independent multi-port oracle",
           all(r["rc"] == 0 and r["drained"] and
               r["typed"] == (4, 4, 4, 4, 4, 4)
               for r in (beha_ind_a, beha_ind_b, beha_group_a, beha_group_b)) and
           beha_ind_a["beha"] == beha_ind_b["beha"] == expected_ind and
           beha_group_a["beha"] == beha_group_b["beha"] == expected_group,
           f"ind={ind_ledger} oracle={expected_ind} "
           f"group={group_ledger} oracle={expected_group}")
    record("V5-e Behavioral completion is deterministic and shared cut is slower",
           beha_ind_a["ns"] == beha_ind_b["ns"] and
           beha_group_a["ns"] == beha_group_b["ns"] and
           beha_group_a["ns"] > beha_ind_a["ns"],
           f"ind={ind_ns}ns group={group_ns}ns")

    dynamic_hw = bounded_hw(32, policy="dynamic")
    dynamic_a = run_case(31, 4, hw=dynamic_hw)
    dynamic_b = run_case(31, 4, hw=dynamic_hw)
    dynamic_ok_a, dynamic_detail = check_run(dynamic_a, 31, 4)
    dynamic_ok_b, _ = check_run(dynamic_b, 31, 4)
    dyn_a = dynamic_a["dynamic"]
    dyn_b = dynamic_b["dynamic"]
    dynamic_drained = (dyn_a is not None and dyn_b is not None and
                       dyn_a[:3] == dyn_b[:3] == (8, 8, 0) and
                       all(v == 0 for v in dyn_a[3]) and
                       all(v == 0 for v in dyn_b[3]))
    record("V5-f dynamic pins REQUEST/DATA per flow and releases on DATA-tail/ACK",
           dynamic_ok_a and dynamic_ok_b and dynamic_drained,
           f"{dynamic_detail} dynamic={dyn_a}/{dyn_b}")
    dyn_ns_a, dyn_ns_b = dynamic_a["ns"], dynamic_b["ns"]
    record("V5-f single-flow dynamic selection is deterministic across runs",
           dynamic_a["subflows"] == dynamic_b["subflows"] and
           dynamic_a["typed"] == dynamic_b["typed"] and
           dynamic_a["ns"] == dynamic_b["ns"],
           f"ns={dyn_ns_a}/{dyn_ns_b}")

    two_wl = workload_two(31, 4)
    dynamic_two_a = run_case(31, 4, hw=dynamic_hw, wl=two_wl)
    dynamic_two_b = run_case(31, 4, hw=dynamic_hw, wl=two_wl)
    f0_ok, f0_links, f0_detail = check_flow(dynamic_two_a, 31, 4, 0, 16)
    f1_ok, f1_links, f1_detail = check_flow(dynamic_two_a, 31, 4, 1, 17)
    two_dyn_a = dynamic_two_a["dynamic"]
    two_dyn_b = dynamic_two_b["dynamic"]
    two_drained = (two_dyn_a is not None and two_dyn_b is not None and
                   two_dyn_a[:3] == two_dyn_b[:3] == (16, 16, 0) and
                   all(v == 0 for v in two_dyn_a[3]) and
                   all(v == 0 for v in two_dyn_b[3]))
    balanced_links = (sorted(f0_links) == sorted(f1_links) and
                      len(set(f0_links + f1_links)) == 4)
    record("V5-f two concurrent dynamic flows are complete, balanced and drain",
           f0_ok and f1_ok and balanced_links and two_drained and
           dynamic_two_a["rc"] == 0 and dynamic_two_a["drained"] and
           dynamic_two_a["typed"] == (8, 8, 8, 8, 62, 62),
           f"flow0={f0_detail} flow1={f1_detail} dynamic={two_dyn_a}")
    two_ns_a, two_ns_b = dynamic_two_a["ns"], dynamic_two_b["ns"]
    record("V5-f concurrent dynamic behavior is reproducible and live",
           dynamic_two_a["subflows"] == dynamic_two_b["subflows"] and
           dynamic_two_a["typed"] == dynamic_two_b["typed"] and
           dynamic_two_a["ns"] == dynamic_two_b["ns"] and
           not dynamic_two_a["watchdog"] and not dynamic_two_b["watchdog"],
           f"ns={two_ns_a}/{two_ns_b}")

    dynamic_3x1_hw = bounded_hw(32, policy="dynamic")
    dynamic_3x1_hw["die"]["x"] = 3
    multi_wl = workload(31, 4, dest=32)
    dynamic_multi_a = run_case(31, 4, hw=dynamic_3x1_hw, wl=multi_wl)
    dynamic_multi_b = run_case(31, 4, hw=dynamic_3x1_hw, wl=multi_wl)
    multi_ok_a, multi_detail = check_multihop_flow(
        dynamic_multi_a, 31, 4, 0, 32, 2)
    multi_ok_b, _ = check_multihop_flow(dynamic_multi_b, 31, 4, 0, 32, 2)
    multi_dyn_a = dynamic_multi_a["dynamic"]
    multi_dyn_b = dynamic_multi_b["dynamic"]
    multi_drained = (multi_dyn_a is not None and multi_dyn_b is not None and
                     multi_dyn_a[:3] == multi_dyn_b[:3] == (16, 16, 0) and
                     all(v == 0 for v in multi_dyn_a[3]) and
                     all(v == 0 for v in multi_dyn_b[3]))
    record("V5-f multi-hop dynamic pins are per-die and all release",
           multi_ok_a and multi_ok_b and multi_drained and
           dynamic_multi_a["repin"] == (78, 78, 0),
           "{} repin={} dynamic={}".format(
               multi_detail, dynamic_multi_a["repin"], multi_dyn_a))
    record("V5-f multi-hop dynamic selection is deterministic and live",
           dynamic_multi_a["subflows"] == dynamic_multi_b["subflows"] and
           dynamic_multi_a["typed"] == dynamic_multi_b["typed"] and
           dynamic_multi_a["ns"] == dynamic_multi_b["ns"] and
           not dynamic_multi_a["watchdog"] and not dynamic_multi_b["watchdog"],
           "ns={}/{}".format(dynamic_multi_a["ns"], dynamic_multi_b["ns"]))

    failed = sum(not ok for _, ok, _ in results)
    print(f"\nV5 D2D tests: {len(results)-failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
