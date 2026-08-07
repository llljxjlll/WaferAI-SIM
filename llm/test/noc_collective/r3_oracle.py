#!/usr/bin/env python3
"""Independent geometry/stage/framing oracle for collective refactor R3."""

from __future__ import annotations


def ceil_div(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        raise ValueError("positive operands required")
    return value // divisor + (value % divisor != 0)


def left_fold(input_count: int) -> list[tuple[tuple[str, int],
                                               tuple[str, int], bool]]:
    if input_count <= 0:
        raise ValueError("reduce requires an input")
    result = []
    for stage in range(input_count - 1):
        lhs = ("network", 0) if stage == 0 else ("feedback", stage - 1)
        rhs = ("network", stage + 1)
        result.append((lhs, rhs, stage == input_count - 2))
    return result


def geometry(elements: int, dtype_bits: int,
             physical_bits: int = 128,
             vector_bits: int = 512) -> tuple[int, int, int, int, int]:
    total_bits = elements * dtype_bits
    physical_flits = ceil_div(total_bits, physical_bits)
    lanes = vector_bits // dtype_bits
    vector_beats = ceil_div(elements, lanes)
    tail_lanes = elements % lanes or lanes
    tail_bits = total_bits % physical_bits or physical_bits
    return physical_flits, vector_beats, lanes, tail_lanes, tail_bits


def split_sequences(physical_flits: int,
                    slices_per_beat: int = 4) -> list[list[int]]:
    return [
        list(range(first, min(first + slices_per_beat, physical_flits)))
        for first in range(0, physical_flits, slices_per_beat)
    ]


def main() -> int:
    assert [len(left_fold(n)) for n in (1, 2, 3, 5)] == [0, 1, 2, 4]
    assert left_fold(5) == [
        (("network", 0), ("network", 1), False),
        (("feedback", 0), ("network", 2), False),
        (("feedback", 1), ("network", 3), False),
        (("feedback", 2), ("network", 4), True),
    ]
    assert len(left_fold(1)) * 9 == 0
    assert len(left_fold(5)) * 7 == 28

    assert geometry(70, 8) == (5, 2, 64, 6, 48)
    assert geometry(80, 8) == (5, 2, 64, 16, 128)
    assert geometry(17, 32) == (5, 2, 16, 1, 32)
    assert split_sequences(5) == [[0, 1, 2, 3], [4]]
    assert split_sequences(8) == [[0, 1, 2, 3], [4, 5, 6, 7]]

    # R3 stream framing is one header plus F data wires; the frozen legacy
    # two-segment encoding spends two physical wires per data flit.
    for physical_flits in (1, 5, 8, 64):
        stream_wires = 1 + physical_flits
        legacy_wires = 2 * physical_flits
        assert stream_wires <= legacy_wires + (physical_flits == 1)
        if physical_flits > 1:
            assert stream_wires < legacy_wires

    print("NoC collective R3 oracle self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
