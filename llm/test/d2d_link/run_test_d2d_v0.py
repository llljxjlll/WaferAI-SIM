#!/usr/bin/env python3
"""D2D V0 准入测试运行器（对齐 D2D_link_test.md 的 V0 测试清单）。

覆盖：
  - L0 纯函数自测（编址/端点/矩形拓扑/端口配置校验）      -> npusim --d2d-v0-selftest
  - 有效单 die + die_ports 端到端解析并运行，结果不变       -> 29109 ns
  - 非法 die_ports（2x1 缺 E 向 C2C）启动期报错             -> 非零退出
  - 单 die 回归（noc_congestion no_congestion, cycle）不变  -> 29109 ns

用法（任意目录执行，自动切到 build/）：
    python3 llm/test/d2d_link/run_test_d2d_v0.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BUILD = os.path.join(ROOT, "build")
NPUSIM = os.path.join(BUILD, "npusim")

HW_OK = "../llm/test/d2d_link/hardware/core_4x4_ports_ok.json"
HW_BAD = "../llm/test/d2d_link/hardware/core_4x4_ports_bad.json"
WL = "../llm/test/noc_congestion/workload/gemm_no_congestion.json"
SIM = "../llm/test/noc_congestion/sim/sim_cycle.json"
MAP = "../llm/test/noc_congestion/mapping/identity.spec"

NS_RE = re.compile(r"(\d+)\s*ns")
EXPECT_NS = 29109  # 单 die no_congestion cycle 基线
CORES_PER_DIE = 16  # 4x4 die；die-run 用例要求本 die 全部 16 核都执行

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def run(cmd, timeout=None):
    try:
        p = subprocess.run(cmd, cwd=BUILD, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        # 超时本身算失败（避免测试永久挂住）；用哨兵码 124 表示 hang
        return 124, (e.output or "") if isinstance(e.output, str) else "<timeout>"


def finish_ns(out):
    ns = None
    for line in out.splitlines():
        if "All requests finished" in line or "Catch test finished" in line:
            m = NS_RE.search(line)
            if m:
                ns = int(m.group(1))
    return ns


def main():
    if not os.path.exists(NPUSIM):
        sys.exit(f"npusim not found at {NPUSIM}; build it first.")

    # 1. L0 自测
    rc, out = run([NPUSIM, "--d2d-v0-selftest"])
    passed = "self-test:" in out and "FAILURES" not in out
    record("L0 pure-function self-test", rc == 0 and passed,
           out.strip().splitlines()[-1] if out.strip() else "")

    # 1b. V1 link SystemC 自测（驱动真实包：latency/序/无丢重/stall/drain/data-ctrl 独立）
    rc, out = run([NPUSIM, "--d2d-link-selftest"], timeout=60)
    lpassed = "link self-test:" in out and "FAILURES" not in out
    record("V1 link SystemC self-test (driven packets)", rc == 0 and lpassed,
           next((l.strip() for l in reversed(out.splitlines())
                 if "link self-test:" in l), ""))

    # 2. 有效 die_ports 端到端，结果不变
    rc, out = run([NPUSIM, "--workload-config", WL, "--hardware-config", HW_OK,
                   "--simulation-config", SIM, "--mapping-config", MAP])
    ns = finish_ns(out)
    record("valid die_ports parses & runs, result unchanged",
           rc == 0 and ns == EXPECT_NS, f"finished {ns} ns (expect {EXPECT_NS})")

    # 3. 非法 die_ports 启动期报错
    rc, out = run([NPUSIM, "--workload-config", WL, "--hardware-config", HW_BAD,
                   "--simulation-config", SIM, "--mapping-config", MAP])
    rejected = rc != 0 and "die_ports" in out and "requires" in out
    record("invalid die_ports rejected at startup", rejected,
           f"exit={rc}")

    # 3b. 三个非法物理参数各自单独被启动期拒绝（参数化，避免只命中第一个）
    import json
    import tempfile
    base = json.load(open(os.path.join(HERE, "hardware", "core_4x4_ports_ok.json")))
    for pname, bad in [("link_bw", 0), ("latency", -1), ("buffer_depth", 0)]:
        cfg = json.loads(json.dumps(base))
        cfg["die_ports"]["c2c"] = {"link_bw": 4, "latency": 20, "buffer_depth": 8}
        cfg["die_ports"]["c2c"][pname] = bad
        fd, path = tempfile.mkstemp(suffix=".json", prefix=f"d2d_bad_{pname}_")
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f)
        rc, out = run([NPUSIM, "--workload-config", WL, "--hardware-config", path,
                       "--simulation-config", SIM, "--mapping-config", MAP])
        os.remove(path)
        rejected = rc != 0 and "die_ports.c2c" in out and "must be" in out
        record(f"invalid c2c param {pname}={bad} rejected at startup", rejected,
               f"exit={rc}")

    # 3c. V0b-2A: 2x1/1x2/2x2 真实多 die 实例化 + die0-only workload 仍为 29109 ns
    import json as _json
    import tempfile as _tf
    base2 = _json.load(open(os.path.join(HERE, "hardware", "core_4x4_die2x1.json")))
    for dx, dy, ncore in [(2, 1, 32), (1, 2, 32), (2, 2, 64)]:
        cfg = _json.loads(_json.dumps(base2))
        cfg["die"] = {"x": dx, "y": dy}
        fd, path = _tf.mkstemp(suffix=".json", prefix=f"d2d_die{dx}x{dy}_")
        with os.fdopen(fd, "w") as f:
            _json.dump(cfg, f)
        rc, out = run([NPUSIM, "--workload-config", WL, "--hardware-config", path,
                       "--simulation-config", SIM, "--mapping-config", MAP])
        os.remove(path)
        ns = finish_ns(out)
        # 独立层级计数（dynamic_cast 遍历 SystemC 层级，非自报 TOTAL_CORES）
        inst_ok = (f"routers={ncore} workers={ncore}" in out
                   and f"expect {ncore}" in out)
        # 运行结束后 D2D 活动必须为 0（V0b-2A 无 C2C 端口）
        d2d_zero = "[D2D] in_pkts=0 out_pkts=0 busy_cycles=0 stall_cycles=0" in out
        record(f"V0b-2A {dx}x{dy} die: {ncore} modules(hier) + die0 sim-time + D2D=0",
               rc == 0 and inst_ok and ns == EXPECT_NS and d2d_zero,
               f"hier={inst_ok} sim_ns={ns} d2d0={d2d_zero} exit={rc}")

    # 3d. V0b-2C0: 跨 die destination 启动期拒绝（core0 -> core16），10s 超时保护
    import json as _j2
    import tempfile as _tf2
    xw = _j2.load(open(os.path.join(
        ROOT, "llm/test/noc_congestion/workload/gemm_no_congestion.json")))
    xw["id_space"] = "global"
    xw["chips"][0]["cores"][0]["worklist"][0]["cast"] = [{"dest": 16, "tag": 16}]
    fd, xpath = _tf2.mkstemp(suffix=".json", prefix="d2d_xdie_")
    with os.fdopen(fd, "w") as f:
        _j2.dump(xw, f)
    rc, out = run([NPUSIM, "--workload-config", xpath,
                   "--hardware-config", "../llm/test/d2d_link/hardware/core_4x4_die2x1.json",
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=10)
    entered_sim = ("All requests finished" in out or "Catch test finished" in out)
    rejected = (rc not in (0, 124) and
                "cross-die traffic requires D2D Link" in out and not entered_sim)
    record("V0b-2C0 cross-die dest rejected before sim (no hang)", rejected,
           f"exit={rc} entered_sim={entered_sim}")
    os.remove(xpath)

    # 3e. V0b-2B1: die1 等价运行（workload 平移到 die1，HOST1->die1->HOST1），期望 sim-time==基线
    xw2 = _j2.load(open(os.path.join(
        ROOT, "llm/test/noc_congestion/workload/gemm_no_congestion.json")))
    def _shift(v):
        return v + 16 if isinstance(v, int) and v >= 0 else v
    for chip in xw2.get("chips", []):
        for c in chip.get("cores", []):
            for k in ("id", "prim_copy", "send_global_mem"):
                if k in c:
                    c[k] = _shift(c[k])
            for wl in c.get("worklist", []):
                for ca in wl.get("cast", []):
                    if "dest" in ca and isinstance(ca["dest"], int):
                        ca["dest"] = _shift(ca["dest"])
    for s in xw2.get("source", []):
        if "dest" in s and isinstance(s["dest"], int):
            s["dest"] = _shift(s["dest"])
    xw2["id_space"] = "global"
    fd, dpath = _tf2.mkstemp(suffix=".json", prefix="d2d_die1_")
    with os.fdopen(fd, "w") as f:
        _j2.dump(xw2, f)
    rc, out = run([NPUSIM, "--workload-config", dpath,
                   "--hardware-config", "../llm/test/d2d_link/hardware/core_4x4_die2x1.json",
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=90)
    os.remove(dpath)
    ns = finish_ns(out)
    import re as _re
    die1_cores = set(_re.findall(r"Core (1[6-9]|2[0-9]|3[01]) ", out))
    d2d0 = "[D2D] in_pkts=0 out_pkts=0 busy_cycles=0 stall_cycles=0" in out
    record("V0b-2B1 die1-equivalent run (HOST1->die1->HOST1, all 16 cores)",
           rc == 0 and ns == EXPECT_NS and len(die1_cores) == CORES_PER_DIE and d2d0,
           f"sim_ns={ns} die1_cores={len(die1_cores)}/16 d2d0={d2d0} exit={rc}")

    # 3f. V0b-2B1 T2: dual-die 同时运行（die0+die1 各自 die 内 workload，均完成、D2D=0、互不干扰）
    import copy as _cp
    d0 = _j2.load(open(os.path.join(
        ROOT, "llm/test/noc_congestion/workload/gemm_no_congestion.json")))
    def _shift_wl(o, off):
        o = _cp.deepcopy(o)
        for chip in o.get("chips", []):
            for c in chip.get("cores", []):
                for k in ("id", "prim_copy", "send_global_mem"):
                    if k in c and isinstance(c[k], int) and c[k] >= 0:
                        c[k] += off
                for wl in c.get("worklist", []):
                    for ca in wl.get("cast", []):
                        if "dest" in ca and isinstance(ca["dest"], int) and ca["dest"] >= 0:
                            ca["dest"] += off
        for s in o.get("source", []):
            if "dest" in s and isinstance(s["dest"], int) and s["dest"] >= 0:
                s["dest"] += off
        return o
    d1 = _shift_wl(d0, 16)
    merged = _cp.deepcopy(d0)
    merged["chips"][0]["cores"] += d1["chips"][0]["cores"]
    merged["source"] = d0.get("source", []) + d1.get("source", [])
    merged["id_space"] = "global"
    fd, mpath = _tf2.mkstemp(suffix=".json", prefix="d2d_t2_")
    with os.fdopen(fd, "w") as f:
        _j2.dump(merged, f)
    rc, out = run([NPUSIM, "--workload-config", mpath,
                   "--hardware-config", "../llm/test/d2d_link/hardware/core_4x4_die2x1.json",
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=120)
    os.remove(mpath)
    ns = finish_ns(out)
    n_d0 = len(set(_re.findall(r"Core ([0-9]|1[0-5]) ", out)))
    n_d1 = len(set(_re.findall(r"Core (1[6-9]|2[0-9]|3[01]) ", out)))
    d2d0 = "[D2D] in_pkts=0 out_pkts=0 busy_cycles=0 stall_cycles=0" in out
    record("V0b-2B1 T2 dual-die simultaneous (both dies all 16, D2D=0)",
           rc == 0 and ns == EXPECT_NS and n_d0 == CORES_PER_DIE and
           n_d1 == CORES_PER_DIE and d2d0,
           f"sim_ns={ns} die0={n_d0}/16 die1={n_d1}/16 d2d0={d2d0} exit={rc}")

    # 3g. V0b-2B1: workload 在更高 die + 2D 布局运行（die3 @ 2x2，验证 lane 索引泛化）
    xw3 = _shift_wl(_j2.load(open(os.path.join(
        ROOT, "llm/test/noc_congestion/workload/gemm_no_congestion.json"))), 48)  # die3
    xw3["id_space"] = "global"
    hw = _j2.load(open(os.path.join(ROOT, "llm/test/d2d_link/hardware/core_4x4_die2x1.json")))
    hw["die"] = {"x": 2, "y": 2}
    fdw, wp = _tf2.mkstemp(suffix=".json", prefix="d2d_die3_")
    fdh, hp = _tf2.mkstemp(suffix=".json", prefix="d2d_hw22_")
    with os.fdopen(fdw, "w") as f: _j2.dump(xw3, f)
    with os.fdopen(fdh, "w") as f: _j2.dump(hw, f)
    rc, out = run([NPUSIM, "--workload-config", wp, "--hardware-config", hp,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=120)
    os.remove(wp); os.remove(hp)
    ns = finish_ns(out)
    n_d3 = len(set(_re.findall(r"Core (4[89]|5[0-9]|6[0-3]) ", out)))
    inst64 = "cores=64 routers=64 workers=64" in out
    record("V0b-2B1 die3 @ 2x2 run (higher die + 2D layout, all 16)",
           rc == 0 and ns == EXPECT_NS and n_d3 == CORES_PER_DIE and inst64,
           f"sim_ns={ns} die3_cores={n_d3}/16 inst64={inst64} exit={rc}")

    # 3h. V0b-2B1: die1 在 1x2（纵向）布局运行——使 dev-record 的 1x2 运行结果可自动复现
    xw12 = _shift_wl(_j2.load(open(os.path.join(
        ROOT, "llm/test/noc_congestion/workload/gemm_no_congestion.json"))), 16)
    xw12["id_space"] = "global"
    hw12 = _j2.load(open(os.path.join(ROOT, "llm/test/d2d_link/hardware/core_4x4_die2x1.json")))
    hw12["die"] = {"x": 1, "y": 2}
    fdw, wp12 = _tf2.mkstemp(suffix=".json", prefix="d2d_die1v_")
    fdh, hp12 = _tf2.mkstemp(suffix=".json", prefix="d2d_hw12_")
    with os.fdopen(fdw, "w") as f: _j2.dump(xw12, f)
    with os.fdopen(fdh, "w") as f: _j2.dump(hw12, f)
    rc, out = run([NPUSIM, "--workload-config", wp12, "--hardware-config", hp12,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=90)
    os.remove(wp12); os.remove(hp12)
    ns = finish_ns(out)
    n_d1v = len(set(_re.findall(r"Core (1[6-9]|2[0-9]|3[01]) ", out)))
    record("V0b-2B1 die1 @ 1x2 run (vertical layout, all 16)",
           rc == 0 and ns == EXPECT_NS and n_d1v == CORES_PER_DIE,
           f"sim_ns={ns} die1_cores={n_d1v}/16 exit={rc}")

    # 3i. V1-pre 3b-2b/3c: 非西侧 HOST 四方向端到端（W/S/E/N 各连续两次）。
    #     每方向两次都校验：mismatch=0、接收签名 DONE(source)+ACK(source,tag) 与 west 一致
    #     （比总数强——能发现「同 source 丢一条+重一条**不同 tag**」；但同 source 同 tag 的抵消仍无法
    #      发现，且未含 seq_id/事件轨迹，故只证「每 (source[,tag]) 接收计数一致」，非严格无丢包/重复）、
    #     预期 per-lane 分布（读 GRID_X/GRID_Y）、D2D=0、无绑定错误、GRID_X*GRID_Y 核完成；
    #     sim-time 只要求两次一致（差异符合各自 hop 变化，不强制 29109）。
    HL_RE = re.compile(r"mismatch=(\d+) per_lane_done=([0-9,]+)")
    SIG_RE = re.compile(r"\[HOSTSIG\] done=([0-9:,]*) ack=([0-9:,]*)")

    def parse_hl(out):
        mm = None
        for line in out.splitlines():
            m = HL_RE.search(line)
            if m:
                mm = m
        if not mm:
            return None, None
        return int(mm.group(1)), [int(x) for x in mm.group(2).split(",") if x != ""]

    def parse_sig(out):
        mm = None
        for line in out.splitlines():
            m = SIG_RE.search(line)
            if m:
                mm = m
        return (mm.group(1), mm.group(2)) if mm else (None, None)

    def sig_to_dict(sig):
        d = {}
        for tok in (sig or "").split(","):
            if tok:
                k, v = tok.split(":")
                d[int(k)] = int(v)
        return d

    HWDIR = "../llm/test/d2d_link/hardware/"

    def grid_of(hwbase):
        c = json.load(open(os.path.join(HERE, "hardware", hwbase)))
        gx = c.get("x", 4)
        return gx, c.get("y", gx)

    def west_ref(wl, whost):
        # 该网格的 west 参考签名（DONE(source)+ACK(source,tag)）+ DONE 源多重集。
        # 注：W 用 die_ports.edges.W=host 配置分支（挂载同 legacy 但非真 legacy 路径；
        #     真 legacy 无 die_ports 路径由「单 die 回归」组覆盖）。
        _, o = run([NPUSIM, "--workload-config", wl,
                    "--hardware-config", HWDIR + whost,
                    "--simulation-config", SIM, "--mapping-config", MAP])
        wd, wa = parse_sig(o)
        return wd, wa, sig_to_dict(wd)

    def ack_structure_ok(asig, ncore):
        # workload 契约：每 source 恰 2 个 ACK —— CONFIG ACK(tag 0) + WEIGHT ACK(tag=core_id)。
        # source 0 两者 tag 都为 0 → (0,0):2；source s>0 → (s,0):1 + (s,s):1。
        d = {}
        for tok in (asig or "").split(","):
            if tok:
                s, t, c = (int(x) for x in tok.split(":"))
                d[(s, t)] = c
        # 恰好覆盖所有合法 source（不多不少）——拒绝范围外/缺失 source
        if {s for s, _ in d} != set(range(ncore)):
            return False
        for s in range(ncore):
            if sum(c for (ss, _), c in d.items() if ss == s) != 2:
                return False
            if s == 0:
                if d.get((0, 0), 0) != 2:
                    return False
            elif d.get((s, 0), 0) != 1 or d.get((s, s), 0) != 1:
                return False
        return True

    # kind=row（W/E）：lane=source//GRID_X，lane 数=GRID_Y；kind=col（S/N）：lane=source%GRID_X，数=GRID_X
    # 网格从 hardware 读 GRID_X/GRID_Y（不硬编码 4）；期望核数 = GRID_X*GRID_Y。
    # ACK 用 (source,tag) 严格签名比对（CONFIG ACK tag 已修为确定 0，4×2/4×4 均严格）。
    def check_dir(tag, wl, hwbase, kind, ref_done, ref_ack, ref_done_d):
        gx, gy = grid_of(hwbase)
        ncore = gx * gy
        lane_fn, n = ((lambda s: s // gx), gy) if kind == "row" else (
            (lambda s: s % gx), gx)
        exp = [0] * n
        for src, cnt in ref_done_d.items():
            exp[lane_fn(src)] += cnt
        runs = []
        for _ in range(2):  # 每方向连续两次，两次都做以下全部校验
            rc, out = run([NPUSIM, "--workload-config", wl,
                           "--hardware-config", HWDIR + hwbase,
                           "--simulation-config", SIM, "--mapping-config", MAP],
                          timeout=90)
            mism, per_lane = parse_hl(out)
            dsig, asig = parse_sig(out)
            cores = len({int(x) for x in re.findall(r"Core (\d+) ", out)
                         if int(x) < ncore})
            runs.append(dict(
                rc=rc, ns=finish_ns(out), mism=mism, per_lane=per_lane,
                dsig=dsig, asig=asig, cores=cores,
                ack_ok=ack_structure_ok(asig, ncore),
                d2d0="[D2D] in_pkts=0 out_pkts=0 busy_cycles=0 stall_cycles=0" in out,
                bind_ok=not any(k in out.lower()
                                for k in ("unbound", "multi-writer", "multi-bind"))))
        ok = all(
            r["rc"] == 0 and r["mism"] == 0 and r["cores"] == ncore and
            r["d2d0"] and r["bind_ok"] and r["per_lane"] == exp and
            r["dsig"] == ref_done and r["asig"] == ref_ack and r["ack_ok"] and
            r["ns"] is not None
            for r in runs) and runs[0]["ns"] == runs[1]["ns"]
        sig_ok = runs[0]["dsig"] == ref_done and runs[0]["asig"] == ref_ack
        record(f"V1-pre 3c {tag} e2e (2x: mismatch=0, DONE(src)+ACK(src,tag) sig==west, "
               f"1 CONFIG@0+1 WEIGHT@id/src, per-lane={exp}, det sim-time)",
               ok, f"ns={runs[0]['ns']}/{runs[1]['ns']} per_lane={runs[0]['per_lane']} "
                   f"mism={runs[0]['mism']} sig==west={sig_ok} ack_struct={runs[0]['ack_ok']} "
                   f"cores={runs[0]['cores']}/{ncore}")

    # 方阵 4×4（16 核）：W 也连续两次
    w4, a4, wd4 = west_ref(WL, "core_4x4_ports_ok.json")
    check_dir("4x4 W-host", WL, "core_4x4_ports_ok.json", "row", w4, a4, wd4)
    check_dir("4x4 S-host", WL, "core_4x4_ports_shost.json", "col", w4, a4, wd4)
    check_dir("4x4 E-host", WL, "core_4x4_ports_ehost.json", "row", w4, a4, wd4)
    check_dir("4x4 N-host", WL, "core_4x4_ports_nhost.json", "col", w4, a4, wd4)

    # 矩形 4×2 die 内 mesh（8 核）：独立 west 参考签名、期望核数 = GRID_X*GRID_Y = 8。
    # DONE 源 {0,2,4,6} 使行/列分布明显不同：W/E per-lane=[2,2]（2 行 lane）、S/N=[2,0,2,0]（4 列 lane）。
    WL42 = "../llm/test/d2d_link/workload/gemm_4x2.json"
    w2, a2, wd2 = west_ref(WL42, "core_4x2_ports_whost.json")
    check_dir("4x2 W-host (horiz mount)", WL42, "core_4x2_ports_whost.json", "row", w2, a2, wd2)
    check_dir("4x2 S-host (vert mount)", WL42, "core_4x2_ports_shost.json", "col", w2, a2, wd2)
    check_dir("4x2 E-host (horiz mount)", WL42, "core_4x2_ports_ehost.json", "row", w2, a2, wd2)
    check_dir("4x2 N-host (vert mount)", WL42, "core_4x2_ports_nhost.json", "col", w2, a2, wd2)

    # 3j. V1-b2: D2D Link 运行时接线。2×1 单 E/W c2c 配置：识别 C2C 出口边并各插入一个
    #     D2DLinkUnit（latency FIFO）取代终结（link_sites=2、link_units=2）。无 C2C-bound 流量时
    #     idle link（对上游 avail=true、data_sent=false）对 die-local workload 无可观测影响——
    #     照跑、D2D=0、两次 sim-time 确定、无 unbound port。
    LS_RE = re.compile(r"link_sites=(\d+)")
    LU_RE = re.compile(r"link_units=(\d+)")

    def _last(rx, out):
        m = None
        for line in out.splitlines():
            mm = rx.search(line)
            if mm:
                m = mm
        return int(m.group(1)) if m else None

    C2C21 = "../llm/test/d2d_link/hardware/core_4x4_die2x1_c2c.json"
    runs_b = []
    for _ in range(2):
        rc, out = run([NPUSIM, "--workload-config", WL, "--hardware-config", C2C21,
                       "--simulation-config", SIM, "--mapping-config", MAP], timeout=90)
        runs_b.append(dict(
            rc=rc, ns=finish_ns(out), ls=_last(LS_RE, out), lu=_last(LU_RE, out),
            d0=len(set(re.findall(r"Core ([0-9]|1[0-5]) ", out))),
            d2d0="[D2D] in_pkts=0 out_pkts=0 busy_cycles=0 stall_cycles=0" in out,
            bind_ok=not any(k in out.lower()
                            for k in ("unbound", "multi-writer", "multi-bind"))))
    ok = all(r["rc"] == 0 and r["ls"] == 2 and r["lu"] == 2 and
             r["d0"] == CORES_PER_DIE and r["d2d0"] and r["bind_ok"] and
             r["ns"] is not None
             for r in runs_b) and runs_b[0]["ns"] == runs_b[1]["ns"]
    record("V1-b2 D2D link wired (2x1 c2c: link_units=2, idle link -> die-local runs, D2D=0, det)",
           ok, f"link_sites={runs_b[0]['ls']} link_units={runs_b[0]['lu']} "
               f"ns={runs_b[0]['ns']}/{runs_b[1]['ns']} "
               f"die0={runs_b[0]['d0']}/16 d2d0={runs_b[0]['d2d0']} bind_ok={runs_b[0]['bind_ok']}")

    # 3k. V1-b2 生产路径校验：ValidateV1MvpTopology 在 Monitor Link 实例化前调用——
    #     2×1 单端口但 link_bw=2（违反 V1 单 pkt/cycle 契约）必须在进入仿真前明确失败，
    #     证明校验不是只在纯函数自测里被调用。
    import json as _jbw
    import tempfile as _tbw
    cbw = _jbw.load(open(os.path.join(HERE, "hardware", "core_4x4_die2x1_c2c.json")))
    cbw["die_ports"]["c2c"]["link_bw"] = 2
    fd, bwpath = _tbw.mkstemp(suffix=".json", prefix="d2d_c2c_bw2_")
    with os.fdopen(fd, "w") as f:
        _jbw.dump(cbw, f)
    rc, out = run([NPUSIM, "--workload-config", WL, "--hardware-config", bwpath,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=30)
    os.remove(bwpath)
    entered = ("All requests finished" in out or "Catch test finished" in out)
    rejected = (rc != 0 and "link_bw must be 1" in out and not entered)
    record("V1-b2 production validation: link_bw=2 rejected before sim (2x1 c2c)",
           rejected, f"exit={rc} entered_sim={entered}")

    # 3l. V1-c3：第一条真实跨 die 协议闭环。core0(die0) 经过 REQUEST→反向 ACK→
    #     多包 DATA 向 core16(die1) 发送；core16 只有收齐完整 DATA 才会 DONE。
    #     按类型 capture/delivery 计数分别守恒，避免仅凭“仿真结束”推断穿链成功。
    D2D_RE = re.compile(
        r"\[D2D\] in_pkts=(\d+) out_pkts=(\d+) busy_cycles=(\d+) stall_cycles=(\d+)")
    D2D_TYPE_RE = re.compile(
        r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
        r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)")

    def last_groups(rx, text):
        m = None
        for line in text.splitlines():
            mm = rx.search(line)
            if mm:
                m = mm
        return tuple(int(x) for x in m.groups()) if m else None

    C3_WL = "../llm/test/d2d_link/workload/cross_die_2core.json"
    c3_runs = []
    for _ in range(2):
        rc, out = run([NPUSIM, "--workload-config", C3_WL,
                       "--hardware-config", C2C21,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=90)
        agg = last_groups(D2D_RE, out)
        typed = last_groups(D2D_TYPE_RE, out)
        mism, _ = parse_hl(out)
        done_sig, _ = parse_sig(out)
        phases = all(s in out for s in (
            "Core 0 start send primitive SEND_REQ",
            "Core 0 end recv primitive RECV_ACK",
            "Core 0 start send primitive SEND_DATA",
            "Core 16 end recv primitive RECV_DATA",
            "Core 16 start send primitive SEND_DONE"))
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        c3_runs.append(dict(
            rc=rc, ns=finish_ns(out), agg=agg, typed=typed, mism=mism,
            done=sig_to_dict(done_sig), phases=phases,
            drain=int(dr.group(1)) if dr else None,
            ls=_last(LS_RE, out), lu=_last(LU_RE, out),
            bind_ok=not any(k in out.lower()
                            for k in ("unbound", "multi-writer", "multi-bind"))))

    def c3_run_ok(r):
        if (r["rc"] != 0 or r["ns"] is None or r["agg"] is None or
                r["typed"] is None or r["mism"] != 0 or not r["bind_ok"]):
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        total_in, total_out, busy, stall = r["agg"]
        return (r["ls"] == 2 and r["lu"] == 2 and
                r["phases"] and r["done"] == {16: 1} and
                rin == rout == 1 and ain == aout == 1 and
                din == dout and din > 1 and
                total_in == total_out == rin + ain + din and
                busy == 0 and stall == 0 and
                r["drain"] == 0)  # 结束态所有 router lock/buffer 归零

    c3_ok = (all(c3_run_ok(r) for r in c3_runs) and
             c3_runs[0]["ns"] == c3_runs[1]["ns"] and
             c3_runs[0]["agg"] == c3_runs[1]["agg"] and
             c3_runs[0]["typed"] == c3_runs[1]["typed"])
    record("V1-c3 cross-die REQUEST -> ACK -> multi-packet DATA runtime e2e (2x)",
           c3_ok, f"ns={c3_runs[0]['ns']}/{c3_runs[1]['ns']} "
                  f"typed={c3_runs[0]['typed']} agg={c3_runs[0]['agg']} "
                  f"done={c3_runs[0]['done']} phases={c3_runs[0]['phases']} "
                  f"drain={c3_runs[0]['drain']}")

    # 3m. c3 开 gate 后的生产负例：3×1 die0→die2 即使逐段物理 link 都存在，V1 仍不支持
    #     多跳。必须在进入仿真前拒绝，防止 c3 的 adjacent 放行误扩成任意跨 die。
    mw = json.load(open(os.path.join(HERE, "workload", "cross_die_2core.json")))
    mw["chips"][0]["cores"][0]["worklist"][0]["cast"][0].update(
        {"dest": 32, "tag": 32})
    mw["chips"][0]["cores"][1]["id"] = 32
    mw["chips"][0]["cores"][1]["worklist"][0]["recv_tag"] = 32
    mhw = json.load(open(os.path.join(
        HERE, "hardware", "core_4x4_die2x1_c2c.json")))
    mhw["die"] = {"x": 3, "y": 1}
    fdw, mwpath = tempfile.mkstemp(suffix=".json", prefix="d2d_c3_multihop_wl_")
    fdh, mhwpath = tempfile.mkstemp(suffix=".json", prefix="d2d_c3_multihop_hw_")
    with os.fdopen(fdw, "w") as f:
        json.dump(mw, f)
    with os.fdopen(fdh, "w") as f:
        json.dump(mhw, f)
    rc, out = run([NPUSIM, "--workload-config", mwpath,
                   "--hardware-config", mhwpath,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=10)
    os.remove(mwpath)
    os.remove(mhwpath)
    entered = ("All requests finished" in out or "Catch test finished" in out)
    rejected = rc not in (0, 124) and "multi-hop cross-die not supported" in out and not entered
    record("V1-c3 production guard: multi-hop die0->die2 rejected before sim",
           rejected, f"exit={rc} entered_sim={entered}")

    # output_lock tag-only 语义的两个对照用例（共用解析器）。
    def cross_flow_run(wl):
        rc, out = run([NPUSIM, "--workload-config", wl, "--hardware-config", C2C21,
                       "--simulation-config", SIM, "--mapping-config", MAP], timeout=90)
        typed = last_groups(D2D_TYPE_RE, out)
        done_sig, _ = parse_sig(out)
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        lk = re.search(r"\[LOCK\] max_output_ref=(\d+)", out)
        return dict(rc=rc, ns=finish_ns(out), typed=typed,
                    done=sig_to_dict(done_sig),
                    drain=int(dr.group(1)) if dr else None,
                    maxref=int(lk.group(1)) if lk else None,
                    bind_ok=not any(k in out.lower()
                                    for k in ("unbound", "multi-writer", "multi-bind")))

    # 3n. distinct-tag：两条**不同 tag** 跨 die 流（core0->16 tag16, core1->17 tag17）汇入同一个
    #     E C2C 端口。各自独占锁（max_output_ref==1）、按接收槽正确串行、两核各 DONE、数据不串、归零。
    f2 = [cross_flow_run("../llm/test/d2d_link/workload/cross_die_2flow.json")
          for _ in range(2)]

    def f2_ok(r):
        if r["rc"] != 0 or r["typed"] is None or not r["bind_ok"]:
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        return (rin == rout == 2 and ain == aout == 2 and din == dout and din > 2 and
                r["done"] == {16: 1, 17: 1} and r["drain"] == 0 and r["maxref"] == 1)

    record("V1-c distinct-tag: two flows on one C2C port serialize (maxref=1, no mix, drain=0)",
           all(f2_ok(r) for r in f2) and f2[0]["ns"] == f2[1]["ns"] and
           f2[0]["typed"] == f2[1]["typed"],
           f"ns={f2[0]['ns']}/{f2[1]['ns']} typed={f2[0]['typed']} done={f2[0]['done']} "
           f"maxref={f2[0]['maxref']} drain={f2[0]['drain']}")

    # 3o. many-to-one（tag-only 核心语义）：两个源 core0/core1 用**同一个 tag16** 都发 core16
    #     （recv_cnt=2）。同 tag 共享同一把锁聚合 → **max_output_ref>=2**（若误改成 (source,tag)
    #     会退化为 1）；core16 收齐两源后 DONE 一次、数据不串、归零。
    m1 = [cross_flow_run("../llm/test/d2d_link/workload/cross_die_many2one.json")
          for _ in range(2)]

    def m1_ok(r):
        if r["rc"] != 0 or r["typed"] is None or not r["bind_ok"]:
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        return (rin == rout == 2 and ain == aout == 2 and din == dout and din > 2 and
                r["done"] == {16: 1} and r["drain"] == 0 and r["maxref"] >= 2)

    record("V1-c many-to-one: same-tag 2 sources share one lock (maxref>=2, agg to 1 recv, drain=0)",
           all(m1_ok(r) for r in m1) and m1[0]["ns"] == m1[1]["ns"] and
           m1[0]["typed"] == m1[1]["typed"],
           f"ns={m1[0]['ns']}/{m1[1]['ns']} typed={m1[0]['typed']} done={m1[0]['done']} "
           f"maxref={m1[0]['maxref']} drain={m1[0]['drain']}")

    # 3p. V1-d1：四方向相邻 die 端到端。同一对 workload（正向 die0->die1 / 反向 die1->die0）
    #     分别跑在 2×1（横向：E/W）与 1×2（纵向：N/S）两个 C2C 布局上，覆盖 die 首跳的四个方向。
    #     每方向连续两次，两次都校验：REQUEST/ACK/DATA 类型计数（REQ 正向、ACK 反向都真实穿 Link）、
    #     agg 守恒、consumer 恰好 DONE 一次（反向 ACK 已被 producer 收到 → 见 phases 的 RECV_ACK）、
    #     link_sites==link_units==2、无绑定错误、mism==0、drain==0、两次 sim-time/typed 确定一致。
    def cross_dir_run(wl, hw, producer, consumer):
        rc, out = run([NPUSIM, "--workload-config", wl, "--hardware-config", hw,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=90)
        mism, _ = parse_hl(out)
        done_sig, _ = parse_sig(out)
        # producer 侧完整握手 + 反向 ACK 落回 producer；consumer 侧收数据并 DONE。
        phases = all(s in out for s in (
            f"Core {producer} start send primitive SEND_REQ",
            f"Core {producer} end recv primitive RECV_ACK",
            f"Core {producer} start send primitive SEND_DATA",
            f"Core {consumer} end recv primitive RECV_DATA",
            f"Core {consumer} start send primitive SEND_DONE"))
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        return dict(
            rc=rc, ns=finish_ns(out), agg=last_groups(D2D_RE, out),
            typed=last_groups(D2D_TYPE_RE, out), mism=mism,
            done=sig_to_dict(done_sig), phases=phases,
            drain=int(dr.group(1)) if dr else None,
            ls=_last(LS_RE, out), lu=_last(LU_RE, out),
            bind_ok=not any(k in out.lower()
                            for k in ("unbound", "multi-writer", "multi-bind")))

    def dir_run_ok(r, consumer):
        if (r["rc"] != 0 or r["ns"] is None or r["agg"] is None or
                r["typed"] is None or r["mism"] != 0 or not r["bind_ok"]):
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        total_in, total_out, busy, stall = r["agg"]
        return (r["ls"] == 2 and r["lu"] == 2 and
                r["phases"] and r["done"] == {consumer: 1} and
                rin == rout == 1 and ain == aout == 1 and
                din == dout and din > 1 and
                total_in == total_out == rin + ain + din and
                busy == 0 and stall == 0 and r["drain"] == 0)

    C2C12 = "../llm/test/d2d_link/hardware/core_4x4_die1x2_c2c.json"
    REV_WL = "../llm/test/d2d_link/workload/cross_die_rev.json"
    # (方向名, workload, hw, producer, consumer)：2×1 横向 E/W；1×2 纵向 N/S。
    for tag, wl, hw, producer, consumer in (
            ("E (2x1 die0->die1)", C3_WL, C2C21, 0, 16),
            ("W (2x1 die1->die0)", REV_WL, C2C21, 16, 0),
            ("N (1x2 die0->die1)", C3_WL, C2C12, 0, 16),
            ("S (1x2 die1->die0)", REV_WL, C2C12, 16, 0)):
        runs = [cross_dir_run(wl, hw, producer, consumer) for _ in range(2)]
        ok = (all(dir_run_ok(r, consumer) for r in runs) and
              runs[0]["ns"] == runs[1]["ns"] and
              runs[0]["typed"] == runs[1]["typed"] and
              runs[0]["agg"] == runs[1]["agg"])
        record(f"V1-d1 {tag}: REQUEST->ACK->DATA e2e (2x, reverse-ACK@producer, drain=0)",
               ok, f"ns={runs[0]['ns']}/{runs[1]['ns']} typed={runs[0]['typed']} "
                   f"agg={runs[0]['agg']} done={runs[0]['done']} "
                   f"phases={runs[0]['phases']} drain={runs[0]['drain']}")

    # 3q. V1-d1 绑定负例：1×2 的 c2c 端口方向 N/S 必须与 die 首跳方向一致。把 N 口错标成 E
    #     （side=N 但 dir=E）会在拓扑校验阶段被 ValidateV1MvpTopology 拒绝，不进入仿真。
    import json as _jd1
    import tempfile as _tfd1
    bad12 = _jd1.load(open(os.path.join(
        HERE, "hardware", "core_4x4_die1x2_c2c.json")))
    for ov in bad12["die_ports"]["overrides"]:
        if ov["side"] == "N":
            ov["dir"] = "E"  # side N 却声称朝 E：非法
    fdb, bpath = _tfd1.mkstemp(suffix=".json", prefix="d2d_d1_baddir_")
    with os.fdopen(fdb, "w") as f:
        _jd1.dump(bad12, f)
    rc, out = run([NPUSIM, "--workload-config", C3_WL, "--hardware-config", bpath,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=10)
    os.remove(bpath)
    entered = ("All requests finished" in out or "Catch test finished" in out)
    rejected = rc not in (0, 124) and not entered
    record("V1-d1 topology guard: c2c port side!=dir rejected before sim",
           rejected, f"exit={rc} entered_sim={entered}")

    # 4. 单 die 回归
    rc, out = run([NPUSIM, "--workload-config", WL,
                   "--hardware-config",
                   "../llm/test/noc_congestion/hardware/core_4x4.json",
                   "--simulation-config", SIM, "--mapping-config", MAP])
    ns = finish_ns(out)
    record("single-die regression (noc_congestion no_congestion cycle)",
           rc == 0 and ns == EXPECT_NS, f"finished {ns} ns (expect {EXPECT_NS})")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== D2D V0: {n_pass}/{len(results)} test groups passed ====")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
