#!/usr/bin/env python3
"""
生成 test_noc_congestion 的两份分布式 GEMM workload（4x4 = 16 核）。

设计目标：把两场景的差异**纯粹**落在 NoC 链路争用（congestion）上，从而干净地评估
仿真器的拥塞建模。为此两份 workload 在以下方面完全一致：
  * 16 个核各做一次 Matmul_f 分片（同样的 B/T/C/OC）——算力一致；
  * 通信是一个 **二分匹配 (bipartite matching)**：8 个奇数核各发 1 条结果给 1 个偶数核，
    每个偶数核恰好收 1 条 —— 每核收发消息数一致，且**没有"多发一"的接收端串行化**。

> 为什么用二分结构而不是任意置换：核间传输是 REQ->ACK->DATA 握手，发送会阻塞等待
> 接收方回 ACK。如果每个核都"先发再收"（如反射置换 perm(i)=15-i 这种对合），会形成
> 环形等待而**死锁**。让奇数核只发、偶数核只收（偶数核先 post 接收即可回 ACK），
> 就避免了死锁。

唯一区别是奇->偶匹配决定的物理路由（XY 维序路由）是否共享链路：
  * gemm_no_congestion.json —— 近邻匹配 odd i -> even (i-1)：同行相邻、1 跳、链路不相交
                               => 基本无拥塞。
  * gemm_congestion.json    —— 远端匹配 odd i -> even (15-i)：关于网格中心点对称的最长对角流，
                               8 条流在 XY 路由下大量共享中心链路 => 严重链路争用/拥塞。

关键对照：行为级 NoC（use_beha_noc=true）是 roofline 模型，延迟只取决于负载大小、与
路由/距离/争用无关 => 两场景在行为级下应给出**几乎相同**的周期；周期精确 router
（use_beha_noc=false）会因中心链路上的 output-lock 串行化而在拥塞场景显著变慢。

所有 *_data 的 DRAM 尺寸设 0，并让 cast 负载（B*T*OC）足够大，使 NoC 主导延迟。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "workload")

# ---- GEMM 规模（两场景共用，保证算力/负载一致）----
B, T, C, OC = 1, 256, 64, 1024
BTP = B * T * C  # source 注入到每个核的输入大小（float 个数）

VARS = {
    "B": B, "T": T, "C": C, "OC": OC,
    "BTP": BTP,
    "loop": 1,
    "matmul_data": 0,
}


def match_no_congestion(i):
    """奇数核 i -> 同行相邻偶数核 i-1（1 跳、链路不相交）。"""
    return i - 1


def match_congestion(i):
    """奇数核 i -> 关于网格中心点对称的偶数核 15-i（最长对角、共享中心链路）。"""
    return 15 - i


def matmul_prim(indata, outdata):
    return {
        "type": "Matmul_f",
        "B": "B", "T": "T", "C": "C", "OC": "OC",
        "sram_address": {"indata": indata, "outdata": outdata},
        "dram_address": {"data": "matmul_data"},
    }


def build(match):
    cores = []
    for i in range(16):
        if i % 2 == 1:
            # 奇数核：源 + 计算 + 发给匹配偶数核
            cores.append({
                "id": i,
                "loop": "loop",
                "worklist": [{
                    "recv_cnt": 1,
                    "prims": [matmul_prim("_input_label", "mm_out")],
                    "cast": [{"dest": match(i)}],   # tag 默认 = dest = 接收核 id
                }],
            })
        else:
            # 偶数核：源 + 计算(work0)，再收 1 条匹配消息并结束(work1)
            cores.append({
                "id": i,
                "loop": "loop",
                "worklist": [
                    {   # work0：收 source，做本核 GEMM 分片（无对外发送）
                        "recv_cnt": 1,
                        "cast": [],
                        "prims": [matmul_prim("_input_label", "mm_out")],
                    },
                    {   # work1：收 1 条匹配消息（recv_tag 默认=核 id=i），向 host 结束
                        "recv_cnt": 1,
                        "cast": [{"dest": -1, "loopout": "true"}],
                    },
                ],
            })
    source = [{"dest": i, "size": "BTP"} for i in range(16)]
    return {"vars": VARS, "pipeline": 1, "source": source,
            "chips": [{"chip_id": 0, "cores": cores}]}


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = {
        "gemm_no_congestion": match_no_congestion,
        "gemm_congestion": match_congestion,
    }
    for name, match in targets.items():
        path = os.path.join(OUT, name + ".json")
        with open(path, "w") as f:
            json.dump(build(match), f, indent=4, ensure_ascii=False)
        flows = {i: match(i) for i in range(16) if i % 2 == 1}
        print("wrote", path, "\n   odd->even flows:", flows)


if __name__ == "__main__":
    main()
