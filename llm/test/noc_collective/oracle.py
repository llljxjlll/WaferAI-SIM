#!/usr/bin/env python3
"""Independent NoC collective V0 contract oracle."""

from __future__ import annotations

U64_MAX = (1 << 64) - 1
COLL_PROFILES = {
    "baseline": ("unicast", "endpoint"),
    "broadcast_only": ("multicast", "endpoint"),
    "reduce_only": ("unicast", "dca_offload"),
    "reduce_broadcast": ("multicast", "dca_offload"),
}
TIER_PROFILES = {0: "baseline", 1: "broadcast_only", 2: "reduce_broadcast"}


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
    """Frozen V5/V6 legacy DCA service formula."""
    return checked(max(compute, ceil_div(payload, width)) + pipeline)


def dca_vector_work(total_elements: int, input_count: int,
                    vector_bits: int, dtype_bits: int) -> tuple[int, int, int, int, int]:
    """R0 two-input vector contract: lanes, beats, pairs, issues, tail lanes."""
    if total_elements <= 0 or input_count <= 0:
        raise ValueError("invalid reduction shape")
    if dtype_bits <= 0 or vector_bits <= 0 or vector_bits % dtype_bits:
        raise ValueError("invalid vector geometry")
    lanes = vector_bits // dtype_bits
    beats = ceil_div(total_elements, lanes)
    pairs = input_count - 1
    issues = checked(beats * pairs)
    tail = total_elements % lanes or lanes
    return lanes, beats, pairs, issues, tail


def dca_issue_cycle(ready_cycle: int, issue_index: int, initiation_interval: int) -> int:
    if ready_cycle < 0 or issue_index < 0 or initiation_interval <= 0:
        raise ValueError("invalid DCA issue timing")
    return checked(ready_cycle + checked(issue_index * initiation_interval))


def dca_completion_cycle(issue_cycle: int, latency: int) -> int:
    if issue_cycle < 0 or latency <= 0:
        raise ValueError("invalid DCA completion timing")
    return checked(issue_cycle + latency)


def collective_backends(profile: str | None = None,
                        tier: int | None = None,
                        broadcast_backend: str | None = None,
                        reduce_backend: str | None = None,
                        ) -> tuple[str, str, str]:
    selected = profile
    if tier is not None:
        if tier not in TIER_PROFILES:
            raise ValueError("invalid collective tier")
        tier_profile = TIER_PROFILES[tier]
        if selected is not None and selected != tier_profile:
            raise ValueError("tier/profile conflict")
        selected = tier_profile
    if selected is not None:
        if selected not in COLL_PROFILES:
            raise ValueError("invalid collective profile")
        expected = COLL_PROFILES[selected]
        if broadcast_backend is not None and broadcast_backend != expected[0]:
            raise ValueError("broadcast backend conflict")
        if reduce_backend is not None and reduce_backend != expected[1]:
            raise ValueError("reduce backend conflict")
        return selected, *expected

    broadcast = broadcast_backend or "unicast"
    reduce = reduce_backend or "endpoint"
    for name, backends in COLL_PROFILES.items():
        if backends == (broadcast, reduce):
            return name, broadcast, reduce
    raise ValueError("unsupported backend combination")


def collective_tree_use(broadcast_backend: str, reduce_backend: str,
                        op: str) -> tuple[bool, bool]:
    multicast = (op in {"broadcast", "allgather"} and
                 broadcast_backend == "multicast")
    reduction = op in {"reduce", "reducescatter", "allreduce"}
    reduce_tree = reduction and reduce_backend == "dca_offload"
    if op == "allreduce" and broadcast_backend == "multicast" and reduce_tree:
        multicast = True
    return multicast, reduce_tree


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
    assert dca_vector_work(129, 1, 512, 8) == (64, 3, 0, 0, 1)
    assert dca_vector_work(129, 2, 512, 8) == (64, 3, 1, 3, 1)
    assert dca_vector_work(129, 3, 512, 8) == (64, 3, 2, 6, 1)
    assert dca_vector_work(16, 3, 128, 8) == (16, 1, 2, 2, 16)
    assert dca_issue_cycle(10, 3, 1) == 13
    assert dca_completion_cycle(13, 7) == 20
    assert [collective_backends(profile=name)[1:]
            for name in COLL_PROFILES] == list(COLL_PROFILES.values())
    assert [collective_backends(tier=tier)[0]
            for tier in range(3)] == [TIER_PROFILES[tier] for tier in range(3)]
    assert collective_backends(reduce_backend="dca_offload") == (
        "reduce_only", "unicast", "dca_offload")
    assert collective_tree_use("unicast", "dca_offload", "allreduce") == (
        False, True)
    assert collective_tree_use("multicast", "dca_offload", "allreduce") == (
        True, True)
    assert collective_tree_use("multicast", "endpoint", "allgather") == (
        True, False)
    try:
        collective_backends(profile="baseline", tier=1)
    except ValueError:
        pass
    else:
        raise AssertionError("tier/profile conflict was not rejected")
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
