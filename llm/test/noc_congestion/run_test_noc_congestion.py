#!/usr/bin/env python3
"""
test_noc_congestion 运行器
==========================

在 4x4 (=16 核) 核阵列上跑同一个分布式 GEMM，用两种片上通信模式：
  * no_congestion —— 奇->偶近邻匹配（链路不相交，无拥塞）
  * congestion    —— 奇->偶远端匹配 15-i（大量共享中心链路，严重拥塞）

每种模式各在两种 NoC 模型下运行：
  * cycle —— 周期精确 router（use_beha_noc=false），会建模链路争用/拥塞
  * beha  —— 行为级 roofline NoC（use_beha_noc=true），只看负载大小，不建模争用

共 4 次运行，抓取仿真结束时刻（sc_time_stamp，单位 ns）= 总周期数，汇总成对照表。

用法（可在任意目录执行，脚本会自动切到 build/ 运行 npusim）：
    python3 test/noc_congestion/run_test_noc_congestion.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # llm/test/noc_congestion -> repo root
BUILD = os.path.join(ROOT, "build")
NPUSIM = os.path.join(BUILD, "npusim")

# 相对 build/ 的配置路径（npusim 必须在 build/ 下运行：sim 配置里引用了 ../font、../DRAMSys）
HW = "../llm/test/noc_congestion/hardware/core_4x4.json"
MAP = "../llm/test/noc_congestion/mapping/identity.spec"
SIM = {
    "cycle": "../llm/test/noc_congestion/sim/sim_cycle.json",   # 周期精确 NoC
    "beha":  "../llm/test/noc_congestion/sim/sim_beha.json",    # 行为级 NoC
}
WL = {
    "no_congestion": "../llm/test/noc_congestion/workload/gemm_no_congestion.json",
    "congestion":    "../llm/test/noc_congestion/workload/gemm_congestion.json",
}

FINISH_RE = re.compile(r"All requests finished")
NS_RE = re.compile(r"(\d+)\s*ns")

# 冻结基线（V0-exit 阻塞门）：任一进程非零 / 结束时间缺失 / 数值不符 → sys.exit(1)。
EXPECT = {
    "no_congestion": {"beha": 14781, "cycle": 29109},
    "congestion":    {"beha": 14833, "cycle": 45441},
}


def run_one(wl_path, sim_path):
    """运行一次仿真，返回 (returncode, 结束时刻 ns|None)。"""
    cmd = [NPUSIM,
           "--workload-config", wl_path,
           "--hardware-config", HW,
           "--simulation-config", sim_path,
           "--mapping-config", MAP]
    finish_ns = None
    p = subprocess.Popen(cmd, cwd=BUILD, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if FINISH_RE.search(line):
            m = NS_RE.search(line)
            if m:
                finish_ns = int(m.group(1))
    p.wait()
    if p.returncode != 0:
        print(f"   [warn] exit={p.returncode} for {wl_path} x {sim_path}")
    return p.returncode, finish_ns


def main():
    if not os.path.exists(NPUSIM):
        sys.exit(f"npusim not found at {NPUSIM}; build it first (see README).")

    # results[scenario][noc_model] = ns ; rcs[...] = 进程退出码
    results = {s: {} for s in WL}
    rcs = {s: {} for s in WL}
    for scen, wl in WL.items():
        for noc, sim in SIM.items():
            print(f"Running scenario={scen:<14} noc={noc:<5} ...", flush=True)
            rcs[scen][noc], results[scen][noc] = run_one(wl, sim)

    # ---- 汇总表 ----
    def fmt(v):
        return f"{v} ns" if v is not None else "FAILED"

    lines = []
    lines.append("NoC congestion evaluation (4x4 distributed GEMM, total cycles = sc_time_stamp)")
    lines.append("")
    header = f"| {'scenario':<16} | {'beha (roofline)':>16} | {'cycle-accurate':>16} | {'cycle-beha (=NoC)':>18} |"
    sep = "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 20 + "|"
    lines.append(header)
    lines.append(sep)
    for scen in ("no_congestion", "congestion"):
        beha = results[scen].get("beha")
        cyc = results[scen].get("cycle")
        noc = (cyc - beha) if (beha is not None and cyc is not None) else None
        lines.append(f"| {scen:<16} | {fmt(beha):>16} | {fmt(cyc):>16} | {fmt(noc):>18} |")

    lines.append("")
    # ---- 关键结论 ----
    try:
        b_no, b_co = results["no_congestion"]["beha"], results["congestion"]["beha"]
        c_no, c_co = results["no_congestion"]["cycle"], results["congestion"]["cycle"]
        beha_ratio = b_co / b_no
        cyc_ratio = c_co / c_no
        lines.append("Key findings:")
        lines.append(f"  * Behavioral (roofline) NoC : congestion/no-congestion = {beha_ratio:.2f}x "
                     f"({b_co} vs {b_no} ns)  -> essentially BLIND to congestion.")
        lines.append(f"  * Cycle-accurate router NoC : congestion/no-congestion = {cyc_ratio:.2f}x "
                     f"({c_co} vs {c_no} ns)  -> CAPTURES link contention.")
        lines.append(f"  * Isolated on-chip contention cost of the congesting pattern "
                     f"= {(c_co - b_co) - (c_no - b_no)} ns "
                     f"(extra 'cycle-beha' in congestion vs no-congestion).")
    except (TypeError, ZeroDivisionError):
        lines.append("Key findings: (some runs failed; see warnings above)")

    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(HERE, "noc_congestion_summary.txt")
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\nWritten: {out}")

    # ---- 阻塞门：进程退出码 + 结束时间存在 + 数值与冻结基线精确一致 ----
    failures = []
    for scen in WL:
        for noc in SIM:
            rc, ns, exp = rcs[scen][noc], results[scen][noc], EXPECT[scen][noc]
            if rc != 0:
                failures.append(f"{scen}/{noc}: process exit={rc}")
            elif ns is None:
                failures.append(f"{scen}/{noc}: no finish time (sim did not complete)")
            elif ns != exp:
                failures.append(f"{scen}/{noc}: {ns} ns != frozen {exp} ns")
    if failures:
        print("\n==== NoC regression FAILED (frozen-baseline mismatch) ====")
        for f_ in failures:
            print("  " + f_)
        sys.exit(1)
    print("\n==== NoC regression PASS: 4/4 == frozen baseline "
          "(no_cong 14781/29109, cong 14833/45441) ====")
    sys.exit(0)


if __name__ == "__main__":
    main()
