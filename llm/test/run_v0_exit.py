#!/usr/bin/env python3
"""历史 V0–V3 兼容门：保留旧冻结契约，不是当前“全版本”聚合门。

  门 1     D2D 纯函数/Link/历史端到端（全部通过）          -> run_test_d2d_v0.py
  门 2     V3 production bounded SAF / congestion / backpressure -> run_test_d2d_v3.py
  门 3     NoC 四场景精确数值 14781/29109/14833/45441      -> run_test_noc_congestion.py

冻结契约：V0 的功能测试（自测 + 端到端，冻结时 165/23）必须**继续全部通过**、NoC 四场景
数值**精确不变**；随 V1+ 增量测试总数可以增长，但既有 V0 契约不得回归。
任一门失败（非零退出）即整体非零退出，可直接用于 CI / pre-commit 阻塞。

当前包含 V4 的 superset gate：python3 llm/test/run_v4_exit.py
本脚本保留原文件名，供旧 CI/命令兼容，不应再被描述为“全版本准入门”。
用法（任意目录）：python3 llm/test/run_v0_exit.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 门以「全部通过」判定，不硬编码计数——自测/端到端组数会随 V1+ 增长（V0 冻结时为
# 165/23）。冻结契约：既有 V0 功能测试必须继续全通过、NoC 四场景数值精确不变；
# 测试总数可增长，但 V0 契约不得回归。
GATES = [
    ("D2D self-test + e2e (all pass)",
     [sys.executable, os.path.join(HERE, "d2d_link", "run_test_d2d_v0.py")]),
    ("D2D V3 bounded SAF production (all pass)",
     [sys.executable, os.path.join(HERE, "d2d_link", "run_test_d2d_v3.py")]),
    ("NoC four-scenario exact (14781/29109, 14833/45441)",
     [sys.executable, os.path.join(HERE, "noc_congestion",
                                   "run_test_noc_congestion.py")]),
]


def main():
    results = []
    for name, cmd in GATES:
        print(f"\n########## GATE: {name} ##########", flush=True)
        rc = subprocess.run(cmd).returncode
        results.append((name, rc))

    print("\n================ D2D 历史 V0–V3 兼容门汇总 ================")
    all_ok = True
    for name, rc in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name}  (exit={rc})")
        all_ok = all_ok and rc == 0
    print("=" * 58)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
