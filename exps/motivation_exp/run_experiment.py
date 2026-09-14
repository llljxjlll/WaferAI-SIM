#!/usr/bin/env python3
"""跨 die 流量 vs 片上流量 的 NoC 争用 motivation 实验。

用法（仓库根目录）：
    python3 exps/motivation_exp/run_experiment.py

需要先构建 build/npusim。脚本会：
  1. 为每个 sweep 点生成 5 个 workload；
  2. 每个配置连续跑两次，校验所有已解析指标完全一致（确定性）；
  3. 写出 results/results.json、figures/normalized_time.svg 和实验报告。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / "build"
NPUSIM = BUILD / "npusim"
HW = HERE / "hardware" / "mesh_3die_4x4.json"
SIM = HERE / "sim" / "cycle.json"
MAP = HERE / "mapping" / "identity.spec"
WLDIR = HERE / "workload"
RESDIR = HERE / "results"
FIGDIR = HERE / "figures"
CYCLE_NS = 2

sys.path.insert(0, str(HERE))
import gen_workloads as G  # noqa: E402

# ---- 实验参数 -------------------------------------------------------------
GEMM = (1, 64, 64, 1024)      # 片上二维切分 GEMM 的每核分片 (B, T, C, OC)
AG_BYTES = 1024               # 一维 AllGather 每条 point-to-point 消息的字节数
SRC_BYTES = 64                # host 注入每核的输入字节数（保持小，避免注入相拖长）
RATIOS = (0.25, 0.5, 1.0, 2.0)  # inter-die 流量 / 片上流量
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# 每个 sweep 点跑的场景。onchip_* 与比例无关，只跑一次。
ONCHIP = ("onchip_x", "onchip_y")
PER_RATIO = ("xdie", "overlap_x", "overlap_y")


def xdie_bytes_for(ratio: float) -> int:
    """总跨 die 字节 = ratio * 总片上 AG 字节。两条过境流平分。"""
    total_onchip = 48 * AG_BYTES               # 4 组 x 4 相 x 3 对端
    per_flow = ratio * total_onchip / len(G.XDIE_SRC_TILES)
    per_flow = int(round(per_flow / G.BYTES_PER_PACKET)) * G.BYTES_PER_PACKET
    assert per_flow > 0
    return per_flow


# ---- 运行与解析 -----------------------------------------------------------
def run(workload_path: Path) -> dict:
    proc = subprocess.run(
        [str(NPUSIM),
         "--workload-config", str(workload_path),
         "--hardware-config", str(HW),
         "--simulation-config", str(SIM),
         "--mapping-config", str(MAP)],
        cwd=str(BUILD), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=900)
    out = ANSI.sub("", proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"npusim failed on {workload_path.name}\n{out[-3000:]}")
    if "All requests finished" not in out:
        raise RuntimeError(f"{workload_path.name} did not finish\n{out[-3000:]}")
    if "PROTO_WAIT] simulation aborted" in out:
        raise RuntimeError(f"{workload_path.name} hit the protocol watchdog")
    parsed = parse(out)
    if parsed["router_residual"] != 0 or parsed["link_residual"] != 0:
        raise RuntimeError(
            f"{workload_path.name} did not drain to zero: "
            f"router={parsed['router_residual']} link={parsed['link_residual']}")
    return parsed


def parse(out: str) -> dict:
    m = re.search(r"All requests finished\.\s*\|\s*(\d+) ns", out)
    sim_ns = int(m.group(1))

    def grab(pattern, default=None):
        mm = re.search(pattern, out)
        return mm.group(1) if mm else default

    noc = re.search(r"\[NOC_ACT\] sends=([\d,]+) stalls=([\d,]+) "
                    r"d2d_source_stalls=(\d+)", out)
    die = re.search(r"\[DIE_ACT\] router_pkts=([\d,]+) mesh_pkts=([\d,]+)", out)
    d2d = re.search(r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
                    r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)", out)

    flows = {}
    fd = grab(r"\[FLOW_DONE\] ([^|]*)\|")
    if fd:
        for item in fd.strip().rstrip(".").split(","):
            item = item.strip()
            if not item:
                continue
            key, cyc = item.split("@")
            src, tag, dst = (int(v) for v in key.split(":"))
            flows[(src, tag, dst)] = int(cyc)

    links = {}
    for mm in re.finditer(r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=(\w+) "
                          r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
                          r"data_in=(\d+) data_out=(\d+)", out):
        links[int(mm.group(1))] = dict(
            src_die=int(mm.group(2)), dst_die=int(mm.group(3)),
            dir=mm.group(4), data_in=int(mm.group(9)), data_out=int(mm.group(10)))

    # host 在放行任何核之前，要给「本 workload 里声明的每一个核」发 CONFIG +
    # WEIGHT 并收齐 ACK。这是一道串行屏障，长度只取决于声明了多少核，和
    # workload 干什么无关。它必须从总时间里扣掉，否则「核多的场景」会被平白
    # 记上一笔额外时间。
    prologue_ns = int(re.search(
        r"Config helper start START data distribution\.\s*\|\s*(\d+) ns",
        out).group(1))
    return dict(
        sim_ns=sim_ns,
        prologue_ns=prologue_ns,
        work_ns=sim_ns - prologue_ns,
        cycles=sim_ns // CYCLE_NS,
        noc_sends=[int(v) for v in noc.group(1).split(",")],
        noc_stalls=[int(v) for v in noc.group(2).split(",")],
        d2d_source_stalls=int(noc.group(3)),
        router_pkts=[int(v) for v in die.group(1).split(",")],
        mesh_pkts=[int(v) for v in die.group(2).split(",")],
        d2d_type=[int(v) for v in d2d.groups()],
        d2d_links=links,
        flow_done={f"{a}:{b}:{c}": v for (a, b, c), v in sorted(flows.items())},
        router_residual=int(grab(r"\[DRAIN\] router_residual=(\d+)", "-1")),
        link_residual=int(grab(r"\[DRAIN\] d2d_link_residual=(\d+)", "-1")),
    )


def run_twice(name: str, workload: dict) -> dict:
    path = WLDIR / f"{name}.json"
    path.write_text(json.dumps(workload, indent=1, ensure_ascii=False))
    first = run(path)
    second = run(path)
    if first != second:
        diff = [k for k in first if first[k] != second[k]]
        raise RuntimeError(f"{name} is not deterministic; differing keys: {diff}")
    print(f"  {name:<24} {first['sim_ns']:>7} ns 总 = "
          f"{first['prologue_ns']:>4} 前导 + {first['work_ns']:>7} 工作")
    return first


# ---- 路径模型（静态、独立于仿真器）---------------------------------------
def directed_links(path):
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def xy_path(a, b):
    """die 内 XY 维序路由，tile 编号 = y*GRID + x。"""
    ax, ay = a % G.GRID, a // G.GRID
    bx, by = b % G.GRID, b // G.GRID
    out, x, y = [a], ax, ay
    while x != bx:
        x += 1 if bx > x else -1
        out.append(y * G.GRID + x)
    while y != by:
        y += 1 if by > y else -1
        out.append(y * G.GRID + x)
    return out


def link_load_model(direction, ag_bytes, xdie_bytes):
    """die1 上每条有向 mesh 链路承载的 DATA 字节数（静态路径模型）。"""
    on = {}
    for grp in G.groups_for(direction):
        for src in grp:
            for dst in grp:
                if src == dst:
                    continue
                for e in directed_links(xy_path(src - 16, dst - 16)):
                    on[e] = on.get(e, 0) + ag_bytes
    # 过境流：从西侧角端口 tile 进、到最近的东侧角端口 tile 出
    xd = {}
    for tile in G.XDIE_SRC_TILES:                   # 0 和 12，即第 0/3 行西端
        for e in directed_links(xy_path(tile, tile + G.GRID - 1)):
            xd[e] = xd.get(e, 0) + xdie_bytes
    shared = sorted(set(on) & set(xd))
    return on, xd, shared


def group_finish(run: dict, direction: str) -> list:
    """每个 AllGather 组里最后一条 flow 的完成 cycle（直接来自 [FLOW_DONE]）。"""
    out = []
    for grp in G.groups_for(direction):
        members = set(grp)
        cycles = [c for k, c in run["flow_done"].items()
                  if int(k.split(":")[0]) in members
                  and int(k.split(":")[2]) in members]
        out.append(max(cycles) if cycles else 0)
    return out


def transit_finish(run: dict) -> list:
    """两条跨 die 过境流到达 die2 的 cycle。"""
    return [c for k, c in sorted(run["flow_done"].items())
            if int(k.split(":")[2]) // G.CORES_PER_DIE == 2]


# ---- 图 -------------------------------------------------------------------
def _text_w(t, f):
    """粗略文本宽度：CJK 约 1 个字宽，拉丁约 0.55 个字宽。"""
    return sum(f if ord(c) > 0x2E80 else 0.55 * f for c in t)


def svg_bar_chart(rows, path: Path):
    """手写 SVG（环境里没有 matplotlib）。rows: [(ratio, [(label, norm), ...])]"""
    W, H = 900, 500
    L, R, TOP, BOT = 86, 34, 96, 78
    pw, ph = W - L - R, H - TOP - BOT
    ymax = 1.15
    gw = pw / len(rows)
    bw = gw * 0.19
    colors = ["#4C6EF5", "#F08C00", "#2F9E44"]
    labels = ["无重叠：comp-only + comm-only",
              "拥塞重叠：片上 x 方向 AG",
              "无拥塞重叠：片上 y 方向 AG"]

    def y(v):
        return TOP + ph * (1 - v / ymax)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Noto Sans CJK SC, DejaVu Sans, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         '<defs>'
         '<marker id="ar" markerWidth="8" markerHeight="8" refX="4" refY="4" '
         'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c92a2a"/></marker>'
         '<marker id="ar2" markerWidth="8" markerHeight="8" refX="4" refY="4" '
         'orient="auto"><path d="M8,0 L0,4 L8,8 z" fill="#c92a2a"/></marker>'
         '</defs>',
         f'<text x="{L}" y="30" font-size="16" fill="#111" font-weight="bold">'
         f'跨 die 过境流量与片上 GEMM + AllGather 的 NoC 链路争用</text>',
         f'<text x="{L}" y="50" font-size="12" fill="#666">'
         f'3×1 dies / 每 die 4×4 核 / 端口在四角 / 周期精确 NoC；'
         f'归一化到「无重叠」串行基线</text>']

    # 图例：标题下方横排
    lx = L
    for i, lab in enumerate(labels):
        s.append(f'<rect x="{lx:.1f}" y="63" width="12" height="12" fill="{colors[i]}"/>')
        s.append(f'<text x="{lx+17:.1f}" y="74" font-size="12" fill="#222">{lab}</text>')
        lx += 17 + _text_w(lab, 12) + 26

    for gv in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{L}" y1="{y(gv):.1f}" x2="{L+pw}" y2="{y(gv):.1f}" '
                 f'stroke="#e9ecef"/>')
        s.append(f'<text x="{L-10}" y="{y(gv)+4:.1f}" font-size="12" '
                 f'text-anchor="end" fill="#444">{gv:.2f}</text>')
    s.append(f'<line x1="{L}" y1="{TOP}" x2="{L}" y2="{TOP+ph}" stroke="#333"/>')
    s.append(f'<line x1="{L}" y1="{TOP+ph}" x2="{L+pw}" y2="{TOP+ph}" stroke="#333"/>')
    s.append(f'<text x="{L-62}" y="{TOP+ph/2}" font-size="13" fill="#222" '
             f'transform="rotate(-90 {L-62} {TOP+ph/2})" text-anchor="middle">'
             f'Norm. Time</text>')
    s.append(f'<text x="{L+pw/2}" y="{H-24}" font-size="13" fill="#222" '
             f'text-anchor="middle">inter-die / intra-die 流量比</text>')

    # hidden-time 参考线 = 无重叠基线 = 1.0
    s.append(f'<line x1="{L}" y1="{y(1.0):.1f}" x2="{L+pw}" y2="{y(1.0):.1f}" '
             f'stroke="#c92a2a" stroke-width="1.6" stroke-dasharray="7 5"/>')
    s.append(f'<text x="{L+4}" y="{y(1.0)-7:.1f}" font-size="12" '
             f'fill="#c92a2a">hidden time 基线（不重叠时的串行时间）</text>')

    for gi, (ratio, bars) in enumerate(rows):
        x0 = L + gi * gw
        if gi:
            s.append(f'<line x1="{x0:.1f}" y1="{TOP}" x2="{x0:.1f}" '
                     f'y2="{TOP+ph}" stroke="#f1f3f5"/>')
        for bi, (_, norm) in enumerate(bars):
            bx = x0 + gw * 0.15 + bi * bw
            s.append(f'<rect x="{bx:.1f}" y="{y(norm):.1f}" width="{bw:.1f}" '
                     f'height="{TOP+ph-y(norm):.1f}" fill="{colors[bi]}"/>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{y(norm)-5:.1f}" font-size="11" '
                     f'text-anchor="middle" fill="#333">{norm:.2f}</text>')
        # 第 3 根柱右侧的 hidden-time 双箭头
        ax = x0 + gw * 0.15 + 3 * bw + 9
        s.append(f'<line x1="{ax:.1f}" y1="{y(1.0)+1:.1f}" x2="{ax:.1f}" '
                 f'y2="{y(bars[2][1])-1:.1f}" stroke="#c92a2a" stroke-width="1.2" '
                 f'marker-end="url(#ar2)" marker-start="url(#ar)"/>')
        s.append(f'<text x="{x0+gw/2:.1f}" y="{TOP+ph+22:.1f}" font-size="13" '
                 f'text-anchor="middle" fill="#222">{ratio:g}x</text>')

    s.append("</svg>")
    path.write_text("\n".join(s), encoding="utf-8")


# ---- 主流程 ---------------------------------------------------------------
def main():
    if not NPUSIM.exists():
        sys.exit(f"missing {NPUSIM}; build it first")
    for d in (WLDIR, RESDIR, FIGDIR):
        d.mkdir(parents=True, exist_ok=True)

    cfg = dict(gemm=GEMM, ag_bytes=AG_BYTES, src_bytes=SRC_BYTES)
    runs = {}

    print("片上任务（与比例无关，只跑一次）：")
    for sc in ONCHIP:
        runs[sc] = run_twice(sc, G.build(sc, xdie_bytes=16, **cfg))

    for r in RATIOS:
        xb = xdie_bytes_for(r)
        print(f"ratio {r:g}x  (每条过境流 {xb} B = {xb//G.BYTES_PER_PACKET} packets)：")
        for sc in PER_RATIO:
            name = f"{sc}_r{r:g}".replace(".", "p")
            runs[name] = run_twice(name, G.build(sc, xdie_bytes=xb, **cfg))

    # 归一化。基线 = 串行执行 = T(片上) + T(跨 die) - 一次公共启动开销。
    # 启动开销取两个场景里「第一条 DATA flow 完成」之前的最小值的下界：用
    # 两次运行都稳定的 host 配置+注入相长度近似，即 xdie 场景里跨 die flow
    # 之前的固定前缀。为不夸大 hidden time，报告同时给出未做修正的裸和。
    rows, table = [], []
    w_ox, w_oy = runs["onchip_x"]["work_ns"], runs["onchip_y"]["work_ns"]
    for r in RATIOS:
        key = lambda sc: f"{sc}_r{r:g}".replace(".", "p")
        w_xd = runs[key("xdie")]["work_ns"]
        # 图里三根柱共用一条基线，取较慢的片上方向（对 x 场景更宽松）。
        base = max(w_ox, w_oy) + w_xd
        ovx = runs[key("overlap_x")]["work_ns"]
        ovy = runs[key("overlap_y")]["work_ns"]
        on_b, xd_b = G.traffic_bytes("overlap_x", AG_BYTES, xdie_bytes_for(r))
        assert abs(xd_b / on_b - r) < 1e-9, "实际流量比与目标不符"
        rows.append((r, [("no-overlap", 1.0), ("overlap-x", ovx / base),
                         ("overlap-y", ovy / base)]))
        table.append(dict(ratio=r, actual_ratio=xd_b / on_b,
                          onchip_bytes=on_b, xdie_total_bytes=xd_b,
                          xdie_bytes=xdie_bytes_for(r),
                          w_onchip_x=w_ox, w_onchip_y=w_oy, w_xdie=w_xd,
                          baseline=base, w_overlap_x=ovx, w_overlap_y=ovy,
                          # 「完美重叠」下界按各自方向取，避免把 x/y 的几何
                          # 不对称算成重叠开销
                          ideal_x=max(w_ox, w_xd), ideal_y=max(w_oy, w_xd),
                          norm_overlap_x=ovx / base, norm_overlap_y=ovy / base,
                          t_overlap_x=runs[key("overlap_x")]["sim_ns"],
                          t_overlap_y=runs[key("overlap_y")]["sim_ns"],
                          t_xdie=runs[key("xdie")]["sim_ns"]))

    svg_bar_chart(rows, FIGDIR / "normalized_time.svg")

    on_x, _, shared_x = link_load_model("x", AG_BYTES, xdie_bytes_for(1.0))
    on_y, _, shared_y = link_load_model("y", AG_BYTES, xdie_bytes_for(1.0))
    result = dict(
        params=dict(gemm=GEMM, ag_bytes=AG_BYTES, src_bytes=SRC_BYTES,
                    ratios=list(RATIOS), cycle_ns=CYCLE_NS),
        summary=table,
        runs=runs,
        path_model=dict(
            shared_links_x=[f"{a}->{b}" for a, b in shared_x],
            shared_links_y=[f"{a}->{b}" for a, b in shared_y],
            onchip_link_peak_x=max(on_x.values()),
            onchip_link_peak_y=max(on_y.values()),
            transit_link_bytes=xdie_bytes_for(1.0),
        ),
    )
    result["group_finish"] = {
        name: {d: group_finish(run, d) for d in ("x", "y")}
        for name, run in runs.items() if run["flow_done"]}
    result["transit_finish"] = {name: transit_finish(run)
                                for name, run in runs.items()}
    (RESDIR / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    write_report(result, runs)
    print(f"\nwrote {RESDIR/'results.json'}")
    print(f"wrote {FIGDIR/'normalized_time.svg'}")
    print(f"wrote {HERE/'motivation_exp_report.md'}")
    return result


def transit_delta(runs) -> str:
    """重叠场景相对「仅跨 die」场景，过境流完成 cycle 的推后量。
    x/y 两个重叠场景必须给出同一个值——y 不共享任何链路，所以这个量是启动
    偏移而不是争用；若不同则说明归因前提被破坏，直接报错。"""
    deltas = set()
    for r in RATIOS:
        k = lambda sc: f"{sc}_r{r:g}".replace(".", "p")
        base = transit_finish(runs[k("xdie")])
        for sc in ("overlap_x", "overlap_y"):
            got = transit_finish(runs[k(sc)])
            deltas.update(b - a for a, b in zip(base, got))
    if len(deltas) != 1:
        raise RuntimeError(f"transit delay is not a constant offset: {deltas}")
    return str(deltas.pop())


def over_range(t, which):
    """该场景相对「理想完美重叠 max(W_on, W_xd)」的额外开销区间。"""
    vals = [row[f"w_overlap_{which}"] / row[f"ideal_{which}"] - 1
            for row in t.values()]
    return f"{min(vals):+.1%} ~ {max(vals):+.1%}"


def write_report(result, runs):
    """把实测数字回填进实验报告（报告的结论段是脚本生成的，不手工维护）。"""
    tpl = (HERE / "report_template.md").read_text(encoding="utf-8")
    t = {row["ratio"]: row for row in result["summary"]}

    def fmt_table():
        head = ("| inter/intra 流量比 | 单条过境流 | 片上 W_on | 跨 die W_xd |"
                " 无重叠基线 | 拥塞重叠(x) | 无拥塞重叠(y) | Norm.(x) | Norm.(y) |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for r in RATIOS:
            row = t[r]
            body += (f"| {r:g}x | {row['xdie_bytes']//G.BYTES_PER_PACKET} pkt |"
                     f" {max(row['w_onchip_x'], row['w_onchip_y'])} ns |"
                     f" {row['w_xdie']} ns | {row['baseline']} ns |"
                     f" {row['w_overlap_x']} ns | {row['w_overlap_y']} ns |"
                     f" {row['norm_overlap_x']:.3f} | {row['norm_overlap_y']:.3f} |\n")
        return head + body

    def fmt_ideal():
        head = ("| 流量比 | 无拥塞(y) 理想 | 实测(y) | 相对理想 |"
                " 拥塞(x) 理想 | 实测(x) | 相对理想 |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for r in RATIOS:
            row = t[r]
            body += (f"| {r:g}x | {row['ideal_y']} ns | {row['w_overlap_y']} ns |"
                     f" {row['w_overlap_y']/row['ideal_y'] - 1:+.1%} |"
                     f" {row['ideal_x']} ns | {row['w_overlap_x']} ns |"
                     f" {row['w_overlap_x']/row['ideal_x'] - 1:+.1%} |\n")
        return head + body

    def fmt_groups():
        head = ("| 运行 | 组 0 (y=0 / x=0) | 组 1 | 组 2 | 组 3 (y=3 / x=3) |"
                " 过境流完成 |\n|---|---:|---:|---:|---:|---:|\n")
        body = ""
        rowspec = [("onchip_x", "x", "片上 x 方向 AG（无跨 die 流量）"),
                   ("onchip_y", "y", "片上 y 方向 AG（无跨 die 流量）")]
        for r in RATIOS:
            k = lambda sc: f"{sc}_r{r:g}".replace(".", "p")
            rowspec.append((k("xdie"), "x", f"仅跨 die 过境流量，{r:g}x"))
            rowspec.append((k("overlap_x"), "x", f"拥塞重叠 x，{r:g}x"))
            rowspec.append((k("overlap_y"), "y", f"无拥塞重叠 y，{r:g}x"))
        for name, d, label in rowspec:
            g4 = group_finish(runs[name], d)
            tf = transit_finish(runs[name])
            tfs = "/".join(str(c) for c in tf) if tf else "—"
            cells = " | ".join(str(c) if c else "—" for c in g4)
            body += f"| {label} | {cells} | {tfs} |\n"
        return head + body

    text = (tpl
            .replace("<<MAIN_TABLE>>", fmt_table())
            .replace("<<IDEAL_TABLE>>", fmt_ideal())
            .replace("<<GROUP_TABLE>>", fmt_groups())
            .replace("<<ONCHIP_X>>", str(runs["onchip_x"]["sim_ns"]))
            .replace("<<ONCHIP_Y>>", str(runs["onchip_y"]["sim_ns"]))
            .replace("<<SHARED_X>>", ", ".join(result["path_model"]["shared_links_x"]))
            .replace("<<SHARED_Y>>",
                     ", ".join(result["path_model"]["shared_links_y"]) or "（空集）")
            .replace("<<MESH_ONCHIP>>", str(runs["onchip_x"]["noc_sends"][1]))
            .replace("<<MESH_ONCHIP_Y>>", str(runs["onchip_y"]["noc_sends"][1]))
            .replace("<<MESH_XDIE>>",
                     str(runs["xdie_r1"]["noc_sends"][1]))
            .replace("<<TRANSIT_DELTA>>", transit_delta(runs))
            .replace("<<X_OVERHEAD_RANGE>>", over_range(t, "x"))
            .replace("<<Y_OVERHEAD_RANGE>>", over_range(t, "y"))
            .replace("<<PROLOGUE_ON>>", str(runs["onchip_x"]["prologue_ns"]))
            .replace("<<PROLOGUE_XD>>", str(runs["xdie_r1"]["prologue_ns"])))
    (HERE / "motivation_exp_report.md").write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
