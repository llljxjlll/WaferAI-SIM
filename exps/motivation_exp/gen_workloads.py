#!/usr/bin/env python3
"""生成 motivation 实验的 workload。

拓扑（见 hardware/mesh_3die_4x4.json）：3x1 dies，每 die 4x4 核。
die_ports 只在四个角上：W(0,0)/W(0,3) 入，E(3,0)/E(3,3) 出。

被观测的 die 是中间的 die1（全局核 16..31）。它上面跑「片上二维切分 GEMM
+ 一维 AllGather」。跨 die 流量是 die0 -> die2 的两条 **过境 (transit)** 流：
它们从 die1 的西侧角端口进来、横穿 die1 的第 0 行 / 第 3 行、再从东侧角端口
出去，全程不占用 die1 的任何核 —— 所以它和 die1 的片上任务天然在时间上重叠，
只在 NoC 物理链路上相遇。

三种对照场景：
  * onchip_x / onchip_y : 只有片上 GEMM+AG（comp-only 的那一半）
  * xdie                : 只有跨 die 过境流量（comm-only 的那一半）
  * overlap_x / overlap_y : 两者同时跑

x 方向 AG 的组是「行」，与过境流量共享第 0/3 行的东西向链路 -> 拥塞重叠。
y 方向 AG 的组是「列」，只用南北向链路，与过境流量链路不相交 -> 无拥塞重叠。
"""
from __future__ import annotations

import json

GRID = 4                       # 每 die 4x4 核
CORES_PER_DIE = GRID * GRID
DIE_UNDER_TEST = 1             # 中间 die
BYTES_PER_PACKET = 16          # M_D_DATA=128 bit

# die0 里的两个过境源核（局部 tile），分别落在第 0 行和第 3 行的西端
XDIE_SRC_TILES = (0, GRID * (GRID - 1))
# die2 里的两个过境汇核：进入 die2 后就是西侧端口所在 tile，0 跳
XDIE_DST_TILES = (0, GRID * (GRID - 1))


def gid(die: int, tile: int) -> int:
    return die * CORES_PER_DIE + tile


def matmul(B, T, C, OC, outdata="mm_out"):
    """Matmul_f 原语。B/T/C/OC 只能写 vars 里的变量名（模拟器按名字查表）。
    out_size = B*T*OC（元素=字节），决定后续 cast 的负载大小。"""
    return {
        "type": "Matmul_f",
        "B": B, "T": T, "C": C, "OC": OC,
        "sram_address": {"indata": "_input_label", "outdata": outdata},
        "dram_address": {"data": "matmul_data"},
    }


GEMM_PRIM = matmul("B", "T", "C", "OC")
AG_PRIM = matmul("ONE", "ONE", "ONE", "AG_OC", outdata="ag_out")
XD_PRIM = matmul("ONE", "ONE", "ONE", "XD_OC", outdata="xd_out")


def groups_for(direction: str):
    """返回 die1 上 4 个 1D AllGather 组（全局核 id）。"""
    out = []
    if direction == "x":                      # 行组：组内通信走东西向链路
        for y in range(GRID):
            out.append([gid(DIE_UNDER_TEST, y * GRID + x) for x in range(GRID)])
    elif direction == "y":                    # 列组：组内通信走南北向链路
        for x in range(GRID):
            out.append([gid(DIE_UNDER_TEST, y * GRID + x) for y in range(GRID)])
    else:
        raise ValueError(direction)
    return out


def onchip_cores(direction):
    """die1 的 16 个核：先做 GEMM 分片，再做一次 4-rank 一维 AllGather。

    AllGather 用「按 rank 分相」的方式展开：第 p 相只有 rank p 发送、其余 3 个
    rank 接收。核间传输是 REQ->ACK->DATA 的阻塞握手，任何「大家同时发」的对称
    结构都会环形等待而死锁；分相后每一相里发送方不接收、接收方不发送，无环。
    """
    cores = {}
    for grp in groups_for(direction):
        for rank, cid in enumerate(grp):
            # job 0：等 host 注入输入 + 做本核的 GEMM 分片（计算相，不发送）
            worklist = [{"recv_cnt": 1, "prims": [GEMM_PRIM], "cast": []}]
            for phase in range(len(grp)):
                if phase == rank:
                    peers = [p for p in grp if p != cid]
                    worklist.append({
                        "recv_cnt": 0,
                        "prims": [AG_PRIM],
                        "cast": [{"dest": p, "tag": p} for p in peers],
                    })
                else:
                    worklist.append({"recv_cnt": 1, "cast": []})
            # 收尾：向 host 报 DONE
            worklist.append({"recv_cnt": 0,
                             "cast": [{"dest": -1, "loopout": "true"}]})
            cores[cid] = {"id": cid, "loop": 1, "worklist": worklist}
    return cores


def xdie_cores():
    """die0 -> die2 的两条过境流；die1 不出现在这两条流的端点里。"""
    cores = {}
    for src_tile, dst_tile in zip(XDIE_SRC_TILES, XDIE_DST_TILES):
        src = gid(0, src_tile)
        dst = gid(2, dst_tile)
        cores[src] = {"id": src, "loop": 1, "worklist": [{
            "recv_cnt": 1,
            "prims": [XD_PRIM],
            "cast": [{"dest": dst, "tag": dst}],
        }]}
        cores[dst] = {"id": dst, "loop": 1, "worklist": [{
            "recv_cnt": 1,
            "cast": [{"dest": -1, "loopout": "true"}],
        }]}
    return cores


def build(scenario, gemm, ag_bytes, xdie_bytes, src_bytes=64):
    """scenario ∈ {onchip_x, onchip_y, xdie, overlap_x, overlap_y}"""
    cores = {}
    if scenario.startswith(("onchip", "overlap")):
        cores.update(onchip_cores(scenario.split("_")[1]))
    if scenario == "xdie" or scenario.startswith("overlap"):
        cores.update(xdie_cores())

    B, T, C, OC = gemm
    # 只有「job0 需要 host 输入」的核才登记 source：die1 的 16 个计算核 +
    # die0 的两个过境源核。die2 的两个汇核如果也登记 source，它们唯一的
    # recv 会被 RECV_START 吃掉，跨 die DATA 就永远没人接收。
    source = [{"dest": cid, "size": "SRC"} for cid in sorted(cores)
              if cid // CORES_PER_DIE != 2]

    return {
        "id_space": "global",
        "vars": {"B": B, "T": T, "C": C, "OC": OC, "ONE": 1,
                 "AG_OC": ag_bytes, "XD_OC": xdie_bytes,
                 "SRC": src_bytes, "matmul_data": 0},
        "pipeline": 1,
        "source": source,
        "chips": [{"chip_id": 0,
                   "cores": [cores[k] for k in sorted(cores)]}],
    }


def traffic_bytes(scenario, ag_bytes, xdie_bytes):
    """该场景注入的 DATA 字节数（不含 host 注入）。run_experiment 用它核对
    每个 sweep 点实际达到的 inter-die / intra-die 比例。"""
    on = 0
    if scenario.startswith(("onchip", "overlap")):
        on = 4 * 4 * 3 * ag_bytes      # 4 组 x 4 相 x 3 个对端
    xd = 0
    if scenario == "xdie" or scenario.startswith("overlap"):
        xd = len(XDIE_SRC_TILES) * xdie_bytes
    return on, xd


if __name__ == "__main__":
    # 自检：两个方向的 AllGather 通信量必须逐条相等，只有链路方向不同。
    for d in ("x", "y"):
        grps = groups_for(d)
        assert len(grps) == GRID and all(len(g) == GRID for g in grps)
    assert sorted(sum(groups_for("x"), [])) == sorted(sum(groups_for("y"), []))
    print("groups_for self-check OK; workload 文件由 run_experiment.py 生成")
