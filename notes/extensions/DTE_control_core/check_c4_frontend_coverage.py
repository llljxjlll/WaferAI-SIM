#!/usr/bin/env python3
"""Static C4 gate for the WorkerCore DTE control facade.

This intentionally checks production call sites, not DTE implementation files
or focused selftests, where direct DTEUnit/DteAsyncTracker access is expected.
Run from any directory; the repository root is derived from this file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKER_SOURCE = ROOT / "llm/src/workercore/workercore.cpp"
LOGIC_SOURCE = ROOT / "llm/src/workercore/logic.cpp"
WORKER_HEADER = ROOT / "llm/include/workercore/workercore.h"
NPUSIM_SOURCE = ROOT / "llm/unittest/npusim.cpp"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def occurrences(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def strip_cpp_comments(text: str) -> str:
    text = re.sub(r"\/\*.*?\*\/", "", text, flags=re.DOTALL)
    return re.sub(r"\/\/[^\n]*", "", text)


def main() -> int:
    failures: list[str] = []
    worker = read(WORKER_SOURCE)
    logic = read(LOGIC_SOURCE)
    header = read(WORKER_HEADER)
    npusim = read(NPUSIM_SOURCE)
    production = strip_cpp_comments(worker + "\n" + logic)

    # Construction and pointer ownership remain WorkerCore implementation
    # details, but every member operation must go through DteControlFrontend.
    # Checking all member accesses (rather than a method allow/deny list) keeps
    # newly introduced DTEUnit/DteAsyncTracker methods from silently bypassing
    # the facade.
    direct_backend_access = re.compile(r"\b(?:dte|dte_async)\s*->\s*\w+")
    for match in direct_backend_access.finditer(production):
        line = production.count("\n", 0, match.start()) + 1
        failures.append(
            f"direct DTE backend access bypasses frontend near combined line {line}: "
            f"{match.group(0)}"
        )

    if "DteTransferContext" in production:
        failures.append(
            "WorkerCore production source exposes DteTransferContext; use handle/snapshot"
        )

    required_logic_calls = {
        "Issue": r"\bdte_control->Issue\(",
        "Wait": r"\bdte_control->Wait\(",
        "Release": r"\bdte_control->Release\(",
        "Snapshot": r"\bdte_control->Snapshot\(",
        "BitWidth": r"\bdte_control->BitWidth\(",
    }
    for name, pattern in required_logic_calls.items():
        if not re.search(pattern, logic):
            failures.append(f"logic.cpp has no frontend {name} call")

    required_token_calls = {
        "IssueToken": r"\bdte_control->IssueToken\(",
        "WaitToken": r"\bdte_control->WaitToken\(",
        "PollToken": r"\bdte_control->PollToken\(",
        "Fence": r"\bdte_control->Fence\(",
        "CancelToken": r"\bdte_control->CancelToken\(",
        "HasToken": r"\bdte_control->HasToken\(",
        "OutstandingTokenCount": r"\bdte_control->OutstandingTokenCount\(",
    }
    for name, pattern in required_token_calls.items():
        if not re.search(pattern, worker):
            failures.append(f"workercore.cpp has no frontend {name} call")

    residual_contract = (
        "dte_control->OutstandingTokenCount()" in header
        and "dte_control->OutstandingTransferCount()" in header
        and "p2p_endpoint->Residual().async_tokens" in header
    )
    if not residual_contract:
        failures.append(
            "DteOutstandingCount must aggregate frontend tokens/transfers and P2P tokens"
        )
    if occurrences(npusim, r"\bDteOutstandingCount\(\)") < 3:
        failures.append(
            "npusim final/probe drain no longer consumes DteOutstandingCount broadly"
        )

    direct_binding = occurrences(
        worker, r"\bdte_async\s*->\s*BindMemoryBridge\("
    )
    if direct_binding != 0:
        failures.append("workercore binds the DTE memory bridge without frontend")
    if not re.search(r"\bdte_control\s*->\s*BindMemoryBridge\(", worker):
        failures.append("workercore has no frontend DTE memory-bridge binding")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[PASS] WorkerCore DTE business operations use DteControlFrontend")
    print(
        "[PASS] frontend physical calls: "
        + ", ".join(
            f"{name}={occurrences(logic, pattern)}"
            for name, pattern in required_logic_calls.items()
        )
    )
    print(
        "[PASS] frontend token calls: "
        + ", ".join(
            f"{name}={occurrences(worker, pattern)}"
            for name, pattern in required_token_calls.items()
        )
    )
    print("[PASS] unified DTE/P2P residual aggregation remains wired")
    print("[PASS] DTE memory bridge binding uses DteControlFrontend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
