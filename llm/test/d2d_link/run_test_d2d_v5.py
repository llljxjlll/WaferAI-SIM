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


def run_case(packets, stripes, timeout=60):
    wp = temp_json(workload(packets, stripes))
    try:
        p = subprocess.run(
            [NPUSIM, "--workload-config", wp, "--hardware-config", HW,
             "--simulation-config", SIM, "--mapping-config", MAP],
            cwd=BUILD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout)
        return parse(p.returncode, ANSI.sub("", p.stdout))
    except subprocess.TimeoutExpired as e:
        out = e.output if isinstance(e.output, str) else "<timeout>"
        return parse(124, ANSI.sub("", out))
    finally:
        os.remove(wp)


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
    return {
        "rc": rc, "out": out, "subflows": subflows, "typed": typed,
        "repin": repin,
        "drained": "[DRAIN] router_residual=0" in out and
                   "[DRAIN] d2d_link_residual=0" in out,
        "saf_zero": "[SAF] reserved_packets=0" in out,
        "watchdog": "[PROTO_WAIT]" in out,
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

    failed = sum(not ok for _, ok, _ in results)
    print(f"\nV5 D2D tests: {len(results)-failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
