#!/usr/bin/env python3
"""Independent NoC collective V0 contract oracle."""

from __future__ import annotations

U64_MAX = (1 << 64) - 1


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("invalid ceil_div arguments")
    return value // divisor + (value % divisor != 0)


def checked(value: int) -> int:
    if not 0 <= value <= U64_MAX:
        raise OverflowError("uint64 overflow")
    return value


def unicast(payload: int, bandwidth: int, network: int) -> int:
    return checked(ceil_div(payload, bandwidth) + network)


def broadcast(ranks: int, message: int, bandwidth: int, network: int) -> int:
    if ranks <= 0:
        raise ValueError("empty group")
    return checked(ceil_div(checked((ranks - 1) * message), bandwidth) + network)


def scatter(ranks: int, message: int, bandwidth: int, network: int) -> int:
    if ranks <= 0:
        raise ValueError("empty group")
    return checked(ceil_div(message, bandwidth) + network)


def dca(payload: int, compute: int, width: int = 128, pipeline: int = 54) -> int:
    return checked(max(compute, ceil_div(payload, width)) + pipeline)


def rank_partition(count: int, ranks: int, rank: int) -> tuple[int, int]:
    if ranks <= 0 or not 0 <= rank < ranks:
        raise ValueError("invalid rank partition")
    base, extra = divmod(count, ranks)
    return base + (rank < extra), rank * base + min(rank, extra)


def tier0_flow_count(op: str, ranks: int) -> int:
    if ranks <= 0: raise ValueError("empty group")
    if op == "p2p":
        if ranks != 2: raise ValueError("p2p requires two ranks")
        return 1
    if op in {"scatter", "gather", "broadcast", "reduce"}: return ranks - 1
    if op in {"allgather", "alltoall"}: return ranks * (ranks - 1)
    if op in {"reducescatter", "allreduce"}: return 2 * (ranks - 1)
    raise ValueError("unsupported op")

def tier0_phase_count(op: str, ranks: int) -> int:
    if op in {"p2p", "scatter", "broadcast"}: return 1
    if op in {"gather", "allgather", "alltoall"}: return ranks
    if op == "reduce": return ranks + 1
    if op in {"reducescatter", "allreduce"}: return ranks + 2
    raise ValueError("unsupported op")

def tier0_reduce_compute_cycles(count: int, ranks: int, lanes: int = 128) -> int:
    if count <= 0 or ranks <= 0 or lanes <= 0:
        raise ValueError("invalid Tier0 reduction compute shape")
    return ceil_div(count * (ranks - 1), lanes)


def main() -> int:
    assert [ceil_div(x, 128) for x in (0, 1, 128, 129)] == [0, 1, 1, 2]
    assert unicast(129, 128, 3) == 5
    assert broadcast(4, 256, 128, 3) == 9
    assert broadcast(1, 256, 128, 3) == 3
    assert scatter(4, 513, 128, 3) == 8
    assert dca(129, 1) == 56 and dca(128, 9) == 63
    assert [rank_partition(10, 3, r) for r in range(3)] == [(4, 0), (3, 4), (3, 7)]
    assert [tier0_flow_count(op, 3) for op in ("scatter", "gather", "broadcast", "allgather", "alltoall")] == [2, 2, 2, 6, 6]
    assert [tier0_phase_count(op, 3) for op in ("scatter", "gather", "broadcast", "allgather", "alltoall")] == [1, 3, 1, 3, 3]
    assert [tier0_flow_count(op, 3) for op in ("reduce", "reducescatter", "allreduce")] == [2, 4, 4]
    assert [tier0_phase_count(op, 3) for op in ("reduce", "reducescatter", "allreduce")] == [4, 5, 5]
    assert tier0_reduce_compute_cycles(10, 3) == 1
    try:
        broadcast(3, U64_MAX, 1, 0)
    except OverflowError:
        pass
    else:
        raise AssertionError("broadcast overflow was not rejected")
    print("NoC collective V0/V1/V3 oracle self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
