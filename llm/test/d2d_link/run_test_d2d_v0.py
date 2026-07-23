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
    D2D_DATA_RE = re.compile(
        r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+) "
        r"in_seqhash=(\d+) out_seqhash=(\d+) in_csum=(\d+) out_csum=(\d+) "
        r"out_inorder=(\d+) out_minseq=(-?\d+) out_maxseq=(-?\d+) "
        r"out_endseq=(-?\d+) out_end_count=(\d+) out_end_length=(-?\d+) "
        r"in_first_cycle=(-?\d+) in_last_cycle=(-?\d+) "
        r"out_first_cycle=(-?\d+) out_last_cycle=(-?\d+)")

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

    # 3m. V2-b 生产负例：多跳本身已放行（见 3t），但**路径上任一跳缺 C2C link** 仍必须在进入
    #     仿真前拒绝。这里 3×1 只给 E 向单端口、不给 W——die 级路径 die0→die1→die2 无双向
    #     peer link，必须报错退出且不进仿真（护栏没有随多跳放行一起被删掉）。
    mw = json.load(open(os.path.join(HERE, "workload", "cross_die_2core.json")))
    mw["chips"][0]["cores"][0]["worklist"][0]["cast"][0].update(
        {"dest": 32, "tag": 32})
    mw["chips"][0]["cores"][1]["id"] = 32
    mw["chips"][0]["cores"][1]["worklist"][0]["recv_tag"] = 32
    mhw = json.load(open(os.path.join(
        HERE, "hardware", "core_4x4_die2x1_c2c.json")))
    mhw["die"] = {"x": 3, "y": 1}
    # 去掉 W 向 c2c：E 有端口但反向缺失 → 每一跳都不成双向 peer link
    mhw["die_ports"]["overrides"] = [
        o for o in mhw["die_ports"]["overrides"] if o.get("dir") != "W"]
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
    rejected = (rc not in (0, 124) and not entered and
                "requires >=1 C2C port" in out)
    record("V2-b production guard: multi-hop over a direction lacking C2C rejected before sim",
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
        ldr = re.search(r"\[DRAIN\] d2d_link_residual=(-?\d+)", out)
        return dict(
            rc=rc, ns=finish_ns(out), agg=last_groups(D2D_RE, out),
            typed=last_groups(D2D_TYPE_RE, out), mism=mism,
            done=sig_to_dict(done_sig), phases=phases,
            drain=int(dr.group(1)) if dr else None,
            link_drain=int(ldr.group(1)) if ldr else None,
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
                busy == 0 and stall == 0 and
                r["drain"] == 0 and r["link_drain"] == 0)

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
                   f"phases={runs[0]['phases']} drain={runs[0]['drain']}/"
                   f"{runs[0]['link_drain']}")

    # 3q. V1-d1 绑定负例：1×2 的 c2c 端口方向 N/S 必须与 die 首跳方向一致。把 N 口错标成 E
    #     （side=N 但 dir=E）必须由 ParseDiePorts/ParseRoleSpec 给出精确诊断并在仿真前拒绝。
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
    rejected = (rc not in (0, 124) and
                "MVP requires C2C dir == side" in out and not entered)
    record("V1-d1 topology guard: c2c port side!=dir rejected before sim",
           rejected, f"exit={rc} entered_sim={entered}")

    # 3r. V1-d2：逐包完整性与消息大小边界。当前 4x4 测例 B=1/T=4、int8、
    #     noc_payload_per_cycle=4，因此 OC=16*N 精确产生 N 个链路 DATA 包；
    #     OC=16*N-1 仍产生 N 包，但尾包 length=96，可覆盖非整包尾部。
    def integrity_run(wl, hw):
        rc, out = run([NPUSIM, "--workload-config", wl, "--hardware-config", hw,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=90)
        mism, _ = parse_hl(out)
        done_sig, _ = parse_sig(out)
        phases_ok = all(s in out for s in (
            "Core 0 start send primitive SEND_REQ",
            "Core 0 end recv primitive RECV_ACK",
            "Core 0 start send primitive SEND_DATA",
            "Core 16 end recv primitive RECV_DATA",
            "Core 16 start send primitive SEND_DONE"))
        ldr = re.search(r"\[DRAIN\] d2d_link_residual=(-?\d+)", out)
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        return dict(
            rc=rc, ns=finish_ns(out), agg=last_groups(D2D_RE, out),
            typed=last_groups(D2D_TYPE_RE, out),
            data=last_groups(D2D_DATA_RE, out), mism=mism,
            done=sig_to_dict(done_sig), phases=phases_ok,
            drain=int(dr.group(1)) if dr else None,
            link_drain=int(ldr.group(1)) if ldr else None,
            ls=_last(LS_RE, out), lu=_last(LU_RE, out),
            bind_ok=not any(k in out.lower()
                            for k in ("unbound", "multi-writer", "multi-bind")))

    def probe_ok(d, packets, end_length):
        if d is None:
            return False
        (pin, pout, seqin, seqout, csin, csout, inorder,
         minseq, maxseq, endseq, end_count, end_len,
         in_first, in_last, out_first, out_last) = d
        return (pin == pout == packets and seqin == seqout and csin == csout and
                inorder == 1 and maxseq - minseq + 1 == packets and
                endseq == maxseq and end_count == 1 and end_len == end_length and
                in_first >= 0 and in_last >= in_first and
                out_first >= 0 and out_last >= out_first and
                in_last - in_first == out_last - out_first)

    def integrity_ok(r, packets, end_length):
        if (r["rc"] != 0 or r["ns"] is None or r["agg"] is None or
                r["typed"] is None or r["mism"] != 0 or not r["bind_ok"]):
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        total_in, total_out, busy, stall = r["agg"]
        return (r["ls"] == 2 and r["lu"] == 2 and r["phases"] and
                r["done"] == {16: 1} and
                r["drain"] == 0 and r["link_drain"] == 0 and
                rin == rout == 1 and ain == aout == 1 and
                din == dout == packets and
                total_in == total_out == packets + 2 and
                busy == 0 and stall == 0 and
                probe_ok(r["data"], packets, end_length))

    size_cases = (
        ("1-full", 16, 1, 128),
        ("2-full", 32, 2, 128),
        ("5-partial-tail", 79, 5, 96),
        ("7-before-depth", 112, 7, 128),
        ("8-at-depth", 128, 8, 128),
        ("9-after-depth", 144, 9, 128),
        ("32-long", 512, 32, 128),
    )
    d2_observed = []
    for name, oc, packets, end_length in size_cases:
        sized = _jd1.load(open(os.path.join(
            HERE, "workload", "cross_die_2core.json")))
        sized["vars"]["OC"] = oc
        fdw, wpath = _tfd1.mkstemp(suffix=".json", prefix="d2d_d2_size_")
        with os.fdopen(fdw, "w") as f:
            _jd1.dump(sized, f)
        obs = integrity_run(wpath, C2C21)
        os.remove(wpath)
        ok = integrity_ok(obs, packets, end_length)
        d2_observed.append(obs)
        record(f"V1-d2 DATA integrity {name}: packets/seq/checksum/tail/drain",
               ok, f"ns={obs['ns']} typed={obs['typed']} data={obs['data']} "
                   f"done={obs['done']} drain={obs['drain']}")

    monotonic = (all(r["ns"] is not None for r in d2_observed) and
                 all(a["ns"] < b["ns"]
                     for a, b in zip(d2_observed, d2_observed[1:])))
    record("V1-d2 message-size completion time strictly increases (1..32 packets)",
           monotonic, f"ns={[r['ns'] for r in d2_observed]}")

    # 3s. V1-d3：生产端到端 latency=0/1/7/20 扫描。完整事务有三个严格因果串联的
    #     跨链阶段（REQUEST 正向、ACK 反向、DATA 正向），CYCLE=2ns，因此：
    #       T(L)-T(0) = 3 * L * CYCLE = 6L ns。
    #     同时检查 Link 自身 DATA delivery-capture 增量恰为 L cycle，包间 span 不变。
    latency_results = {}
    base_hw = _jd1.load(open(os.path.join(
        HERE, "hardware", "core_4x4_die2x1_c2c.json")))
    for latency in (0, 1, 7, 20):
        hw = _jd1.loads(_jd1.dumps(base_hw))
        hw["die_ports"]["c2c"]["latency"] = latency
        fdh, hpath = _tfd1.mkstemp(suffix=".json", prefix="d2d_d3_latency_")
        with os.fdopen(fdh, "w") as f:
            _jd1.dump(hw, f)
        lruns = [integrity_run(C3_WL, hpath) for _ in range(2)]
        os.remove(hpath)
        ok = (all(integrity_ok(r, 4, 128) for r in lruns) and
              lruns[0]["ns"] == lruns[1]["ns"] and
              lruns[0]["typed"] == lruns[1]["typed"] and
              lruns[0]["data"] == lruns[1]["data"])
        latency_results[latency] = lruns[0]
        record(f"V1-d3 latency={latency}: deterministic e2e + integrity (2x)",
               ok, f"ns={lruns[0]['ns']}/{lruns[1]['ns']} "
                   f"typed={lruns[0]['typed']} data={lruns[0]['data']}")

    cycle_ns = 2  # llm/include/macros/macros.h:CYCLE
    causal_phases = 3  # REQUEST + ACK + DATA
    latency_law = all(r["ns"] is not None and r["data"] is not None
                      for r in latency_results.values())
    if latency_law:
        base = latency_results[0]
        base_data = base["data"]
        base_link_delay = base_data[14] - base_data[12]
        base_in_span = base_data[13] - base_data[12]
        base_out_span = base_data[15] - base_data[14]
        for latency, obs in latency_results.items():
            d = obs["data"]
            latency_law = latency_law and (
                obs["ns"] - base["ns"] ==
                causal_phases * latency * cycle_ns and
                (d[14] - d[12]) - base_link_delay == latency and
                d[13] - d[12] == base_in_span and
                d[15] - d[14] == base_out_span and
                d[:12] == base_data[:12])
    else:
        base_link_delay = None
        base_out_span = None
    latency_times = {latency: obs["ns"]
                     for latency, obs in latency_results.items()}
    record("V1-d3 latency law: link delta=L cycles, transaction delta=3*L*CYCLE, span fixed",
           latency_law, f"ns={latency_times} base_link_delay={base_link_delay} "
                        f"data_span={base_out_span}")

    # 3t. V2-b：中间 die 运行时转发（3×1 两跳 die0->die2，直线 E->E）。
    #     关键：3×1 相邻两 die 的 E 出口用**同一个模板 port id**，所以「已重新 pin」与「沿用
    #     上一跳旧值」路由结果完全相同——端到端送达**不能**证明入口重写发生。故必须断言
    #     [D2D_REPIN] 的 same>0（数值相同但确实重写过）与 total==跨 link 包数。
    REPIN_RE = re.compile(r"\[D2D_REPIN\] total=(\d+) changed=(\d+) same=(\d+)")
    HW31 = "../llm/test/d2d_link/hardware/core_4x4_die3x1_c2c.json"
    WL2HOP = "../llm/test/d2d_link/workload/cross_die_2hop.json"

    def twohop_run():
        rc, out = run([NPUSIM, "--workload-config", WL2HOP,
                       "--hardware-config", HW31,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=120)
        done_sig, ack_sig = parse_sig(out)
        mism, _ = parse_hl(out)
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        ldr = re.search(r"\[DRAIN\] d2d_link_residual=(-?\d+)", out)
        # 中间 die（die1，核 16..31）不得消费包或提前产生 ACK/DONE
        ack_srcs = {int(t.split(":")[0]) for t in (ack_sig or "").split(",") if t}
        phases = all(s in out for s in (
            "Core 0 start send primitive SEND_REQ",
            "Core 0 end recv primitive RECV_ACK",
            "Core 0 start send primitive SEND_DATA",
            "Core 32 end recv primitive RECV_DATA",
            "Core 32 start send primitive SEND_DONE"))
        return dict(rc=rc, ns=finish_ns(out), typed=last_groups(D2D_TYPE_RE, out),
                    repin=last_groups(REPIN_RE, out), done=sig_to_dict(done_sig),
                    ack_srcs=ack_srcs, mism=mism, phases=phases,
                    drain=int(dr.group(1)) if dr else None,
                    link_drain=int(ldr.group(1)) if ldr else None,
                    ls=_last(LS_RE, out), lu=_last(LU_RE, out),
                    bind_ok=not any(k in out.lower()
                                    for k in ("unbound", "multi-writer", "multi-bind")))

    def twohop_ok(r):
        if (r["rc"] != 0 or r["typed"] is None or r["repin"] is None or
                r["mism"] != 0 or not r["bind_ok"] or not r["phases"]):
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        total, changed, same = r["repin"]
        crossings = rin + ain + din  # 每个包每跨一次 link 记一次入口重写
        return (r["ls"] == 4 and r["lu"] == 4 and       # 3×1：2 对 die × 双向 = 4 条有向 link
                rin == rout == 2 and ain == aout == 2 and  # REQ/ACK 各跨 2 跳
                din == dout and din == 8 and               # 4 个 DATA 包 × 2 跳
                total == crossings and same + changed == total and
                same == changed == total // 2 and same > 0 and
                r["done"] == {32: 1} and                  # 只有最终目的核 DONE
                r["ack_srcs"] <= {0, 32} and              # 中间 die 未产生 ACK
                r["drain"] == 0 and r["link_drain"] == 0)

    th = [twohop_run() for _ in range(2)]
    record("V2-b 3x1 two-hop relay: intermediate die re-pins (same>0 proves rewrite), "
           "no premature ACK/DONE, drain=0",
           all(twohop_ok(r) for r in th) and th[0]["ns"] == th[1]["ns"] and
           th[0]["typed"] == th[1]["typed"] and th[0]["repin"] == th[1]["repin"],
           f"ns={th[0]['ns']}/{th[1]['ns']} typed={th[0]['typed']} "
           f"repin={th[0]['repin']} done={th[0]['done']} "
           f"ack_srcs={sorted(th[0]['ack_srcs'])} "
           f"drain={th[0]['drain']}/{th[0]['link_drain']}")

    # 3u. V2-b2 回归：config 驱动 HOST（每 die lane 数 != GRID_Y）下最高 die 的纯 die 内 workload。
    #     曾因权重下发用 legacy 的 `config.id / GRID_X` 索引 write_buffer 而越界段错误
    #     （2×2：core48 -> 12，而 HOST_LANES=12，合法 0..11）。现统一走 HostLaneOfCore。
    #     该用例**无任何跨 die 流量**，故 D2D 活动必须恒为 0——把 HOST lane 缺陷与多跳解耦。
    HW22 = "../llm/test/d2d_link/hardware/core_4x4_die2x2_c2c.json"
    d3wl = _jd1.load(open(os.path.join(HERE, "workload", "cross_die_2core.json")))
    d3wl["source"] = [{"dest": 48, "size": "BTP"}]
    d3wl["chips"][0]["cores"][0]["id"] = 48
    d3wl["chips"][0]["cores"][0]["worklist"][0]["cast"] = [{"dest": 49, "tag": 49}]
    d3wl["chips"][0]["cores"][1]["id"] = 49
    d3wl["chips"][0]["cores"][1]["worklist"][0]["recv_tag"] = 49
    fdw, d3path = _tfd1.mkstemp(suffix=".json", prefix="d2d_v2b2_die3_")
    with os.fdopen(fdw, "w") as f:
        _jd1.dump(d3wl, f)
    rc, out = run([NPUSIM, "--workload-config", d3path, "--hardware-config", HW22,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=120)
    os.remove(d3path)
    d3_done, _ = parse_sig(out)
    d3_dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
    d3_ldr = re.search(r"\[DRAIN\] d2d_link_residual=(-?\d+)", out)
    d3_ok = (rc == 0 and finish_ns(out) is not None and
             "[D2D] in_pkts=0 out_pkts=0 busy_cycles=0 stall_cycles=0" in out and
             sig_to_dict(d3_done) == {49: 1} and
             d3_dr is not None and int(d3_dr.group(1)) == 0 and
             d3_ldr is not None and int(d3_ldr.group(1)) == 0)
    record("V2-b2 die3-local on config-driven HOST: weight fill uses HostLaneOfCore "
           "(was q[12] overflow -> SIGSEGV), D2D=0",
           d3_ok, f"exit={rc} ns={finish_ns(out)} done={sig_to_dict(d3_done)}")

    # 3v. V2-b2：2×2 对角多跳 die0 -> die1 -> die3（正向 E 然后 N；反向 ACK W 然后 S）。
    #     这是**方向真的改变**的多跳，与 3×1 直线互补：每次入口重写都必须把 E 改成 N（反向 W->S），
    #     因此 changed==total、same==0。若运行时沿用上一跳 exit，ValidatePinnedExit 会直接抛错。
    WLDIAG = "../llm/test/d2d_link/workload/cross_die_diag.json"

    def diag_run():
        rc, out = run([NPUSIM, "--workload-config", WLDIAG, "--hardware-config", HW22,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=180)
        done_sig, ack_sig = parse_sig(out)
        mism, _ = parse_hl(out)
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        ldr = re.search(r"\[DRAIN\] d2d_link_residual=(-?\d+)", out)
        ack_srcs = {int(t.split(":")[0]) for t in (ack_sig or "").split(",") if t}
        phases = all(s in out for s in (
            "Core 0 start send primitive SEND_REQ",
            "Core 0 end recv primitive RECV_ACK",
            "Core 0 start send primitive SEND_DATA",
            "Core 48 end recv primitive RECV_DATA",
            "Core 48 start send primitive SEND_DONE"))
        return dict(rc=rc, ns=finish_ns(out), typed=last_groups(D2D_TYPE_RE, out),
                    repin=last_groups(REPIN_RE, out), done=sig_to_dict(done_sig),
                    ack_srcs=ack_srcs, mism=mism, phases=phases,
                    drain=int(dr.group(1)) if dr else None,
                    link_drain=int(ldr.group(1)) if ldr else None,
                    bind_ok=not any(k in out.lower()
                                    for k in ("unbound", "multi-writer", "multi-bind")))

    def diag_ok(r):
        if (r["rc"] != 0 or r["typed"] is None or r["repin"] is None or
                r["mism"] != 0 or not r["bind_ok"] or not r["phases"]):
            return False
        rin, rout, ain, aout, din, dout = r["typed"]
        total, changed, same = r["repin"]
        return (rin == rout == 2 and ain == aout == 2 and din == dout == 8 and
                total == rin + ain + din and
                changed == total and same == 0 and  # 每次重写都改变方向（E->N / W->S / ->-1）
                r["done"] == {48: 1} and r["ack_srcs"] <= {0, 48} and
                r["drain"] == 0 and r["link_drain"] == 0)

    dg = [diag_run() for _ in range(2)]
    record("V2-b2 2x2 diagonal two-hop (E then N, ACK W then S): every ingress re-pin "
           "changes direction (changed==total, same=0), drain=0",
           all(diag_ok(r) for r in dg) and dg[0]["ns"] == dg[1]["ns"] and
           dg[0]["typed"] == dg[1]["typed"] and dg[0]["repin"] == dg[1]["repin"],
           f"ns={dg[0]['ns']}/{dg[1]['ns']} typed={dg[0]['typed']} "
           f"repin={dg[0]['repin']} done={dg[0]['done']} "
           f"ack_srcs={sorted(dg[0]['ack_srcs'])} "
           f"drain={dg[0]['drain']}/{dg[0]['link_drain']}")

    # 3w. V2-c：多跳端到端闭环——精确到「经过哪几条有向 link、方向序列、每条各多少包、
    #     每包跳了几跳、中间 die 的 NoC 是否真有活动」。全局 [D2D_TYPE] 只有总数，无法区分
    #     路径；这里用逐条 link 归因 [D2D_LINK] + 每 die 活动 [DIE_ACT] 做精确断言。
    LINK_RE = re.compile(
        r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=(\w+) "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)")
    DIEACT_RE = re.compile(
        r"\[DIE_ACT\] router_pkts=([0-9,]+) mesh_pkts=([0-9,]+)")

    def path_run(wl, hw):
        rc, out = run([NPUSIM, "--workload-config", wl, "--hardware-config", hw,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=200)
        links = {}
        for m in LINK_RE.finditer(out):
            (idx, a, b, d, rin, rout, ain, aout, din, dout) = m.groups()
            links[(int(a), int(b))] = dict(
                dir=d, req=(int(rin), int(rout)), ack=(int(ain), int(aout)),
                data=(int(din), int(dout)))
        da = DIEACT_RE.findall(out)
        die_act = [int(x) for x in da[-1][0].split(",")] if da else []
        die_mesh = [int(x) for x in da[-1][1].split(",")] if da else []
        done_sig, ack_sig = parse_sig(out)
        dr = re.search(r"\[DRAIN\] router_residual=(-?\d+)", out)
        ldr = re.search(r"\[DRAIN\] d2d_link_residual=(-?\d+)", out)
        mism, _ = parse_hl(out)
        return dict(rc=rc, ns=finish_ns(out), links=links, die_act=die_act,
                    die_mesh=die_mesh, mism=mism,
                    bind_ok=not any(k in out.lower()
                                    for k in ("unbound", "multi-writer", "multi-bind")),
                    done=sig_to_dict(done_sig),
                    ack_srcs={int(t.split(":")[0])
                              for t in (ack_sig or "").split(",") if t},
                    drain=int(dr.group(1)) if dr else None,
                    link_drain=int(ldr.group(1)) if ldr else None,
                    repin=last_groups(REPIN_RE, out))

    def path_ok(r, fwd, ack, npkt, src, dest, exp_mesh):
        """fwd/ack: [(src_die, dst_die, dir), ...] 期望的**精确**有向 link 序列。"""
        if (r["rc"] != 0 or r["drain"] != 0 or r["link_drain"] != 0 or
                r["mism"] != 0 or not r["bind_ok"]):
            return False
        # 1) 承载 DATA 的有向 link 集合必须**恰好**等于期望正向路径，且方向、每条包数精确。
        #    active-set 用 in 或 out 判定（只看 in 会漏掉理论上的 out-only 异常）。
        data_links = {k for k, v in r["links"].items()
                      if v["data"][0] > 0 or v["data"][1] > 0}
        if data_links != {(a, b) for a, b, _ in fwd}:
            return False
        for a, b, d in fwd:
            v = r["links"][(a, b)]
            # REQUEST 与 DATA 同路；每条 link 的 in==out（link 内部不丢包）
            if (v["dir"] != d or v["data"] != (npkt, npkt) or
                    v["req"] != (1, 1) or v["ack"] != (0, 0)):
                return False
        # 1b) REQUEST 也必须**恰好**只出现在正向路径上——否则多余 link 上的 REQUEST 不会被发现
        req_links = {k for k, v in r["links"].items()
                     if v["req"][0] > 0 or v["req"][1] > 0}
        if req_links != {(a, b) for a, b, _ in fwd}:
            return False
        # 2) 承载 ACK 的有向 link 集合必须恰好等于期望反向路径（维序 X-first，可能与正向不对称）
        ack_links = {k for k, v in r["links"].items()
                     if v["ack"][0] > 0 or v["ack"][1] > 0}
        if ack_links != {(a, b) for a, b, _ in ack}:
            return False
        for a, b, d in ack:
            v = r["links"][(a, b)]
            if v["dir"] != d or v["ack"] != (1, 1) or v["data"] != (0, 0):
                return False
        # 3) 每包 hop 数 == 路径长度：DATA 总跨链次数 == 包数 × hop 数
        hops = len(fwd)
        if sum(r["links"][(a, b)]["data"][0] for a, b, _ in fwd) != npkt * hops:
            return False
        # 4) 入口重写次数 == 总跨链包数（REQ+ACK+DATA 各自跨满全程）
        if r["repin"] is None or r["repin"][0] != (1 + npkt) * hops + 1 * len(ack):
            return False
        # 5) 逐 die 片内 mesh hop 数必须**精确**等于期望（不只是 >0）。die_mesh 只统计同 die
        #    router→router 输入，是「包确实穿过该 die 的 mesh」的直接证据；die_act 含「跨 link
        #    到达那一拍」，>0 不能排除零片内 hop。期望值随几何而定：目的 die 若入口 tile 恰为
        #    目的核本身，则该 die 的片内 hop 为 0（如 3×1/1×3 的 die2），这正是要精确固化的信息。
        if r["die_mesh"] != exp_mesh:
            return False
        # 6) 中间 die 不消费、不产生 ACK/DONE：只有终点 DONE；ACK 源**恰好**是首尾两端
        #    （用 == 而非 <=，否则空集/单端点也会通过）
        return r["done"] == {dest: 1} and r["ack_srcs"] == {src, dest}

    HW31 = "../llm/test/d2d_link/hardware/core_4x4_die3x1_c2c.json"
    HW13 = "../llm/test/d2d_link/hardware/core_4x4_die1x3_c2c.json"
    v2c_cases = (
        # (名称, workload, hw, 正向 link 序列, ACK link 序列, 源核, 目的核, 逐 die 期望片内 hop)
        ("3x1 fwd die0->die2 (E,E / ACK W,W)", WL2HOP, HW31,
         [(0, 1, "E"), (1, 2, "E")], [(2, 1, "W"), (1, 0, "W")], 0, 32, [18, 18, 0]),
        ("3x1 rev die2->die0 (W,W / ACK E,E)",
         "../llm/test/d2d_link/workload/cross_die_2hop_rev.json", HW31,
         [(2, 1, "W"), (1, 0, "W")], [(0, 1, "E"), (1, 2, "E")], 32, 0, [18, 18, 0]),
        # 1×3 纵向两跳（计划要求的第三种拓扑）：正向 N,N；反向 S,S
        ("1x3 fwd die0->die2 (N,N / ACK S,S)",
         "../llm/test/d2d_link/workload/cross_die_2hop_v.json", HW13,
         [(0, 1, "N"), (1, 2, "N")], [(2, 1, "S"), (1, 0, "S")], 0, 32, [18, 18, 0]),
        ("1x3 rev die2->die0 (S,S / ACK N,N)",
         "../llm/test/d2d_link/workload/cross_die_2hop_v_rev.json", HW13,
         [(2, 1, "S"), (1, 0, "S")], [(0, 1, "N"), (1, 2, "N")], 32, 0, [18, 18, 0]),
        # 2×2 对角：die 级维序 X-first ⇒ 正向 die0-E->die1-N->die3，而 ACK 从 die3 出发也先走 X
        # ⇒ die3-W->die2-S->die0。**正反路径不对称**（构成一个矩形），这里精确固化该行为。
        ("2x2 diag die0->die3 (E,N / ACK W,S; asymmetric)", WLDIAG, HW22,
         [(0, 1, "E"), (1, 3, "N")], [(3, 2, "W"), (2, 0, "S")], 0, 48,
         [52, 15, 3, 9]),
    )
    for name, wl, hw, fwd, ack, src, dest, exp_mesh in v2c_cases:
        pr = [path_run(wl, hw) for _ in range(2)]
        ok = (all(path_ok(r, fwd, ack, 4, src, dest, exp_mesh) for r in pr) and
              pr[0]["ns"] == pr[1]["ns"] and pr[0]["links"] == pr[1]["links"] and
              pr[0]["die_act"] == pr[1]["die_act"] and
              pr[0]["die_mesh"] == pr[1]["die_mesh"])
        record(f"V2-c {name}: exact per-link counts + direction sequence + hops + "
               f"intermediate-die NoC activity",
               ok, f"ns={pr[0]['ns']}/{pr[1]['ns']} "
                   f"data_links={sorted(k for k, v in pr[0]['links'].items() if v['data'][0])} "
                   f"ack_links={sorted(k for k, v in pr[0]['links'].items() if v['ack'][0])} "
                   f"die_act={pr[0]['die_act']} die_mesh={pr[0]['die_mesh']} "
                   f"repin={pr[0]['repin']} "
                   f"done={pr[0]['done']} drain={pr[0]['drain']}/{pr[0]['link_drain']}")

    # 3x. V2-d：延迟标定与活性验收。
    #     V1-d3 在**单跳**上标定了 T(L)-T(0)=3*L*CYCLE（REQUEST/ACK/DATA 三个因果串联跨链阶段）。
    #     多跳时三个阶段各自跨 H 条 link，故推广为 **T(L)-T(0) = 3*H*L*CYCLE**。
    #     并且把「NoC 路由开销」与「每跳 D2D 固定延迟」分离：
    #       T(H,L) = T_fixed(H) + 3*H*L*CYCLE
    #     ⇒ (T2(L)-T1(L)) - (T2(0)-T1(0)) = 3*L*CYCLE
    #     即多出的那一跳里，**与 L 无关的固定开销**和**可编程 D2D latency 增量**被分离开。
    #     注意措辞：该固定开销（本配置下 54 ns）是「每多一跳的 L-independent 固定成本」，
    #     它同时包含中间 die 的 NoC/router traversal、ingress re-pin、D2D 接口固定 pipeline
    #     以及两组实验端点位置差异，**本测试并未把这些分项进一步拆开**，故不称其为纯 NoC 开销。
    CYCLE_NS = 2      # llm/include/macros/macros.h: CYCLE
    PHASES = 3        # REQUEST + ACK + DATA
    HW21 = "../llm/test/d2d_link/hardware/core_4x4_die2x1_c2c.json"

    def hw_with_latency(base_rel, latency):
        cfg = _jd1.load(open(os.path.join(
            HERE, "hardware", os.path.basename(base_rel))))
        cfg["die_ports"]["c2c"]["latency"] = latency
        fd, path = _tfd1.mkstemp(suffix=".json", prefix=f"d2d_v2d_L{latency}_")
        with os.fdopen(fd, "w") as f:
            _jd1.dump(cfg, f)
        return path

    LAT_SWEEP = (0, 1, 7, 20)
    t2, t1, obs2 = {}, {}, {}
    hang = []  # watchdog：任何一次运行超时（哨兵 124）都会被记录
    for L in LAT_SWEEP:
        p3, p2 = hw_with_latency(HW31, L), hw_with_latency(HW21, L)
        r2 = [path_run(WL2HOP, p3) for _ in range(2)]   # 2 跳
        r1 = [path_run(C3_WL, p2) for _ in range(2)]    # 1 跳
        os.remove(p3)
        os.remove(p2)
        hang += [r["rc"] for r in r2 + r1 if r["rc"] == 124]
        t2[L] = r2[0]["ns"] if r2[0]["ns"] == r2[1]["ns"] else None
        t1[L] = r1[0]["ns"] if r1[0]["ns"] == r1[1]["ns"] else None
        obs2[L] = r2[0]

    # 3x-1：两跳 latency 律 T(L)-T(0) = 3*H*L*CYCLE（H=2 → 12L ns）
    HOPS = 2
    law2 = all(v is not None for v in t2.values()) and all(
        t2[L] - t2[0] == PHASES * HOPS * L * CYCLE_NS for L in LAT_SWEEP)
    # latency 只平移固定延迟：包数/路径/重写次数/完整性在各 L 下完全不变
    inv = all(obs2[L]["links"] == obs2[0]["links"] and
              obs2[L]["repin"] == obs2[0]["repin"] and
              obs2[L]["die_mesh"] == obs2[0]["die_mesh"] and
              obs2[L]["drain"] == 0 and obs2[L]["link_drain"] == 0
              for L in LAT_SWEEP)
    record("V2-d 2-hop latency law: T(L)-T(0)=3*H*L*CYCLE (H=2 -> 12L ns), "
           "packets/path/repin latency-invariant",
           law2 and inv,
           f"T2={t2} "
           f"delta={ {L: t2[L] - t2[0] for L in LAT_SWEEP} } "
           f"expect={ {L: PHASES * HOPS * L * CYCLE_NS for L in LAT_SWEEP} } "
           f"invariant={inv}")

    # 3x-2：NoC 与 D2D 分离——多出的一跳里，NoC 开销与 L 无关，D2D 部分严格 = 3*L*CYCLE
    noc_hop = (t2[0] - t1[0]) if (t2[0] and t1[0]) else None  # L-independent 固定开销
    sep_ok = all(v is not None for v in list(t2.values()) + list(t1.values())) and all(
        (t2[L] - t1[L]) - noc_hop == PHASES * L * CYCLE_NS for L in LAT_SWEEP)
    record("V2-d hop/latency decomposition: extra hop = L-independent fixed cost "
           "+ 3*L*CYCLE programmable D2D latency",
           sep_ok,
           f"T1={t1} T2={t2} fixed_cost_per_extra_hop={noc_hop}ns "
           f"d2d_part={ {L: (t2[L] - t1[L]) - noc_hop for L in LAT_SWEEP} } "
           f"expect={ {L: PHASES * L * CYCLE_NS for L in LAT_SWEEP} }")

    # 3x-3：watchdog / 活性——V2 用无限功能 FIFO，合法多跳模式必须始终推进到完成。
    #       任何一次运行返回超时哨兵 124 都判失败（把「永久挂起」变成可见的失败而非卡住）。
    record("V2-d watchdog: no run hit the timeout sentinel (no permanent hang)",
           not hang, f"timeouts={len(hang)} runs={2 * 2 * len(LAT_SWEEP)}")

    # 3x-4：多流（仍不引入有限缓冲/背压——那属 V3）。两条 2 跳流共享同一对 link：
    #       每条 link 的计数应恰为单流的两倍，两个接收核各 DONE 一次，全部排空。
    WLMF = "../llm/test/d2d_link/workload/cross_die_2hop_multiflow.json"
    mf = [path_run(WLMF, HW31) for _ in range(2)]

    def mf_ok(r):
        if (r["rc"] != 0 or r["drain"] != 0 or r["link_drain"] != 0 or
                r["mism"] != 0 or not r["bind_ok"]):
            return False
        exp = {(0, 1): ("E", (2, 2), (0, 0), (8, 8)),
               (1, 2): ("E", (2, 2), (0, 0), (8, 8)),
               (2, 1): ("W", (0, 0), (2, 2), (0, 0)),
               (1, 0): ("W", (0, 0), (2, 2), (0, 0))}
        if set(r["links"]) != set(exp):
            return False
        for k, (d, req, ack, data) in exp.items():
            v = r["links"][k]
            if (v["dir"], v["req"], v["ack"], v["data"]) != (d, req, ack, data):
                return False
        # 2 条流 × (1 REQ + 1 ACK + 4 DATA) × 2 跳 = 24 次入口重写
        return (r["repin"] == (24, 12, 12) and r["done"] == {32: 1, 33: 1} and
                r["die_mesh"][1] > 0)

    record("V2-d multi-flow: two concurrent 2-hop flows share both links "
           "(per-link counts exactly doubled), both DONE, drain=0",
           all(mf_ok(r) for r in mf) and mf[0]["ns"] == mf[1]["ns"] and
           mf[0]["links"] == mf[1]["links"],
           f"ns={mf[0]['ns']}/{mf[1]['ns']} repin={mf[0]['repin']} "
           f"done={mf[0]['done']} die_mesh={mf[0]['die_mesh']} "
           f"drain={mf[0]['drain']}/{mf[0]['link_drain']}")

    # 3y. V2-d2：**仿真器内部**协议进展 watchdog + 已知协议依赖环的诊断。
    #     动机：Python 的 subprocess timeout（哨兵 124）只能把「永久挂起」变成测试失败，
    #     无法区分协议依赖环 / 路由丢包 / 网络残留，也拿不到等待状态。这里构造一个真实的
    #     rendezvous 依赖环（core0 等 core16 的 tag0，core16 等 core0 的 tag16，双方都先等
    #     对方），要求**仿真器自己**在 wall-clock 超时前主动诊断并非零退出。
    rc, out = run([NPUSIM, "--workload-config",
                   "../llm/test/d2d_link/workload/cross_die_rendezvous_cycle.json",
                   "--hardware-config", C2C21,
                   "--simulation-config", SIM, "--mapping-config", MAP], timeout=300)
    wd = re.search(r"\[PROTO_WAIT\] protocol_wait_cycle=(\d+) "
                   r"last_progress_cycle=(\d+) stalled_for=(\d+) "
                   r"router_residual=(-?\d+) d2d_link_residual=(-?\d+)", out)
    # 必须由仿真器主动退出（专用码 3），**不是** Python 超时哨兵 124
    wd_ok = (rc == 3 and wd is not None and
             "protocol progress watchdog fired" in out and
             int(wd.group(3)) > 0 and
             # 该环是原语层 rendezvous：网络已排空（无在途包 / 无持锁），watchdog 应如实指出
             int(wd.group(4)) == 0 and int(wd.group(5)) == 0 and
             "wait is at the primitive/rendezvous layer" in out)
    record("V2-d2 protocol watchdog diagnoses a rendezvous dependency cycle "
           "(simulator exits non-zero itself, not via test-framework timeout)",
           wd_ok,
           f"exit={rc} (3=watchdog, 124=framework timeout) "
           f"wait_cycle={wd.group(1) if wd else None} "
           f"stalled_for={wd.group(3) if wd else None} "
           f"residual=router{wd.group(4) if wd else '?'}/link{wd.group(5) if wd else '?'}")

    # 3y-2：watchdog 不得误伤合法用例——正常的两跳多流必须照常完成且不触发任何诊断。
    rc_ok, out_ok = run([NPUSIM, "--workload-config", WLMF,
                         "--hardware-config", HW31,
                         "--simulation-config", SIM, "--mapping-config", MAP],
                        timeout=300)
    record("V2-d2 watchdog does not fire on a healthy multi-hop multi-flow run",
           rc_ok == 0 and "PROTO_WAIT" not in out_ok and finish_ns(out_ok) is not None,
           f"exit={rc_ok} ns={finish_ns(out_ok)} proto_wait={'PROTO_WAIT' in out_ok}")

    # 3z. V3-a：**真实启动路径**的配置契约（self-test 之外，覆盖生产 Monitor 分支）。
    #     V3-a 只落地配置契约，生产数据路径仍是 V2 的功能性无限 FIFO。
    def run_with_c2c(c2c, timeout=90):
        cfg = _jd1.load(open(os.path.join(
            HERE, "hardware", "core_4x4_die2x1_c2c.json")))
        cfg["die_ports"]["c2c"] = c2c
        fd, path = _tfd1.mkstemp(suffix=".json", prefix="d2d_v3a_hw_")
        with os.fdopen(fd, "w") as f:
            _jd1.dump(cfg, f)
        rc, out = run([NPUSIM, "--workload-config", C3_WL,
                       "--hardware-config", path,
                       "--simulation-config", SIM, "--mapping-config", MAP],
                      timeout=timeout)
        os.remove(path)
        entered = ("All requests finished" in out or "Catch test finished" in out)
        return rc, out, entered

    # (1) 合法 functional_v2（旧 schema）照常运行，且时序与 V2 冻结值一致
    rc, out, entered = run_with_c2c({"link_bw": 1, "latency": 20,
                                     "buffer_depth": 8})
    record("V3-a startup: legacy functional_v2 config still runs (V2 timing intact)",
           rc == 0 and entered and finish_ns(out) == 398,
           f"exit={rc} ns={finish_ns(out)} (expect 398)")

    # (2) functional_v2 下出现 bounded-only 字段 → 启动期拒绝（不得接受后忽略）
    rc, out, entered = run_with_c2c(
        {"link_bw": 1, "latency": 20, "buffer_depth": 8,
         "link_rate": {"num": 1, "den": 4}}, timeout=20)
    record("V3-a startup: link_rate under functional_v2 rejected before sim",
           rc not in (0, 124) and not entered and
           "only applies to mode=bounded_saf" in out,
           f"exit={rc} entered_sim={entered}")

    # (3) 合法 bounded_saf 配置解析通过，但 runtime 尚未启用 → 必须明确拒绝，
    #     绝不能让「声称有限缓冲」的配置实际按 functional_v2 的无限 FIFO 跑完。
    rc, out, entered = run_with_c2c(
        {"mode": "bounded_saf", "safety": "whole_flow_saf",
         "port_rate": {"num": 1, "den": 2},
         "link_rate": {"num": 1, "den": 4},
         "link_latency": 20, "saf_buffer_depth": 64,
         "link_inflight_depth": 11, "rx_buffer_depth": 8,
         "ctrl_buffer_depth": 4}, timeout=20)
    record("V3-a startup: valid bounded_saf refused while its runtime is unimplemented "
           "(no silent fallback to functional_v2)",
           rc not in (0, 124) and not entered and
           "runtime is not enabled yet" in out,
           f"exit={rc} entered_sim={entered}")

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
