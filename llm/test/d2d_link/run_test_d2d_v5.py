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
HW = os.path.join(HERE, "hardware", "core_4x4_die2x1_c2c_multi4.json")
SIM = "../llm/test/noc_congestion/sim/sim_cycle.json"
MAP = "../llm/test/noc_congestion/mapping/identity.spec"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
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


def workload(packets, stripes):
    w = load(WL)
    w["vars"]["OC"] = packets * 16
    cast = w["chips"][0]["cores"][0]["worklist"][0]["cast"][0]
    cast["stripe"] = stripes
    recv = w["chips"][0]["cores"][1]["worklist"][0]
    recv["recv_stripe"] = stripes
    return w


def temp_json(obj):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def run_case(packets, stripes, timeout=60, hw=None):
    wp = temp_json(workload(packets, stripes))
    hp = temp_json(hw) if hw is not None else None
    try:
        p = subprocess.run(
            [NPUSIM, "--workload-config", wp, "--hardware-config", hp or HW,
             "--simulation-config", SIM, "--mapping-config", MAP],
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
    finish = re.search(r"All requests finished.*?\| (\d+) ns", out)
    group_stalls = {
        int(idx): int(stall)
        for idx, stall in re.findall(
            r"\[D2D_BOUND\] idx=(\d+).*?group_stall=(\d+)", out)
    }
    return {
        "rc": rc, "out": out, "subflows": subflows, "typed": typed,
        "repin": repin, "admit": admit,
        "ns": int(finish.group(1)) if finish else None,
        "group_stalls": group_stalls,
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


    failed = sum(not ok for _, ok, _ in results)
    print(f"\nV5 D2D tests: {len(results)-failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
