#!/usr/bin/env python3
"""V0-exit 统一准入门（冻结基线）：依次执行三道阻塞门并汇总退出码。

  门 1+2  D2D 纯函数自测 165/165 + 端到端 23/23   -> run_test_d2d_v0.py
  门 3     NoC 四场景精确数值 14781/29109/14833/45441 -> run_test_noc_congestion.py

任一门失败（非零退出）即整体非零退出，可直接用于 CI / pre-commit 阻塞。
用法（任意目录）：python3 llm/test/run_v0_exit.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = [
    ("D2D self-test + e2e (165/165 + 23/23)",
     [sys.executable, os.path.join(HERE, "d2d_link", "run_test_d2d_v0.py")]),
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

    print("\n==================== V0-exit 准入门汇总 ====================")
    all_ok = True
    for name, rc in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name}  (exit={rc})")
        all_ok = all_ok and rc == 0
    print("=" * 58)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
