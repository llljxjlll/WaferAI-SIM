#!/usr/bin/env python3
"""Independent oracle for the isolated R2 DCA ComputePool contracts."""

from __future__ import annotations

from dataclasses import dataclass


def ceil_div(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        raise ValueError("positive operands required")
    return value // divisor + (value % divisor != 0)


def vector_work(elements: int, inputs: int,
                vector_bits: int, dtype_bits: int) -> tuple[int, int, int, int]:
    if inputs <= 0 or vector_bits % dtype_bits:
        raise ValueError("invalid vector geometry")
    lanes = vector_bits // dtype_bits
    beats = ceil_div(elements, lanes)
    issues = beats * (inputs - 1)
    tail = elements % lanes or lanes
    return lanes, beats, issues, tail


@dataclass(frozen=True)
class Timing:
    latency: int
    ii: int


def pipeline(timings: list[Timing]) -> list[tuple[int, int]]:
    """Return (issue, completion); each issued request contributes its own II."""
    next_issue = 0
    result: list[tuple[int, int]] = []
    for timing in timings:
        if timing.latency <= 0 or timing.ii <= 0:
            raise ValueError("invalid timing")
        issue = next_issue
        result.append((issue, issue + timing.latency))
        next_issue = issue + timing.ii
    return result


def arbitration(policy: str, core: list[int], dca: list[int]) -> list[int]:
    result: list[int] = []
    rr_core = True
    while core or dca:
        if not core:
            result.append(dca.pop(0))
        elif not dca:
            result.append(core.pop(0))
        elif policy == "round_robin":
            result.append(core.pop(0) if rr_core else dca.pop(0))
            rr_core = not rr_core
        elif policy == "core_priority":
            result.append(core.pop(0))
        elif policy == "dca_priority":
            result.append(dca.pop(0))
        else:
            raise ValueError("unknown arbitration")
    return result


def signed(encoded: int, bits: int) -> int:
    mask = (1 << bits) - 1
    encoded &= mask
    sign = 1 << (bits - 1)
    return encoded - (1 << bits) if encoded & sign else encoded


def main() -> int:
    assert vector_work(65, 2, 512, 8) == (64, 2, 2, 1)
    assert vector_work(17, 2, 128, 8) == (16, 2, 2, 1)

    for dtype_bits in (8, 32, 64):
        lanes = 512 // dtype_bits
        for elements in (1, lanes, lanes + 1, 3 * lanes + 7):
            for inputs in (1, 2, 3, 5):
                _, beats, issues, tail = vector_work(
                    elements, inputs, 512, dtype_bits)
                assert issues == ceil_div(elements, lanes) * (inputs - 1)
                assert 1 <= tail <= lanes

    fast = pipeline([Timing(4, 1)] * 4)
    assert fast == [(0, 4), (1, 5), (2, 6), (3, 7)]
    assert fast[-1][1] == 4 + 4 - 1
    assert pipeline([Timing(3, 2)] * 3) == [(0, 3), (2, 5), (4, 7)]
    mixed = pipeline([Timing(7, 3), Timing(2, 1)])
    assert mixed == [(0, 7), (3, 5)]
    assert sorted(range(2), key=lambda i: mixed[i][1]) == [1, 0]

    assert arbitration("round_robin", [1, 2], [3, 4]) == [1, 3, 2, 4]
    assert arbitration("core_priority", [1, 2], [3, 4]) == [1, 2, 3, 4]
    assert arbitration("dca_priority", [1, 2], [3, 4]) == [3, 4, 1, 2]

    assert (250 + 10) & 0xff == 4
    assert max(signed(0xffffffff, 32), signed(1, 32)) == 1
    assert ((1 << 64) - 1 + 1) & ((1 << 64) - 1) == 0

    print("NoC collective R2 oracle self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
