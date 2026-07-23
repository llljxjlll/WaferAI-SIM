#!/usr/bin/env python3
"""Independent V4 Behavioral D2D oracle.

The representative message still traverses the simulator's routers, so this oracle reports
intra-die hops but does not charge their latency again.  D2D-only time is one end-to-end bulk
service at the single-flow min-cut plus link_latency on every REQUEST/ACK/DATA link crossing.
Multi-hop forwarding is explicitly pipelined; cross-flow contention is intentionally absent.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple


OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N"}


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def parse_rate(value, name: str) -> Fraction:
    require(isinstance(value, dict) and "num" in value and "den" in value,
            f"{name} must be {{num,den}}")
    rate = Fraction(int(value["num"]), int(value["den"]))
    require(0 < rate <= 1, f"{name} must satisfy 0 < rate <= 1")
    return rate


def ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def tile_for(side: str, idx: int, gx: int, gy: int) -> int:
    require(side in OPPOSITE, f"unknown side {side}")
    limit = gy if side in ("E", "W") else gx
    require(0 <= idx < limit, f"idx {idx} out of range for side {side}")
    if side == "E":
        return idx * gx + gx - 1
    if side == "W":
        return idx * gx
    if side == "N":
        return (gy - 1) * gx + idx
    return idx


def c2c_ports(hw: dict) -> Dict[str, int]:
    gx = int(hw["x"])
    gy = int(hw.get("y", gx))
    ports: Dict[str, int] = {}
    for item in hw.get("die_ports", {}).get("overrides", []):
        if item.get("role") != "c2c":
            continue
        side, direction = item["side"], item["dir"]
        require(side == direction, "V4 MVP requires side==dir")
        require(direction not in ports, "V4 MVP permits one C2C port per direction")
        idx = item["idx"]
        require(not isinstance(idx, list) or len(idx) == 1,
                "V4 MVP permits one C2C port per direction")
        if isinstance(idx, list):
            idx = idx[0]
        ports[direction] = tile_for(side, int(idx), gx, gy)
    return ports


def first_dir(src_die: int, dst_die: int, die_x: int) -> str:
    sx, sy = src_die % die_x, src_die // die_x
    dx, dy = dst_die % die_x, dst_die // die_x
    if sx < dx:
        return "E"
    if sx > dx:
        return "W"
    if sy < dy:
        return "N"
    if sy > dy:
        return "S"
    raise ValueError("first_dir called for same die")


def next_die(die: int, direction: str, die_x: int, die_y: int) -> int:
    x, y = die % die_x, die // die_x
    if direction == "E":
        x += 1
    elif direction == "W":
        x -= 1
    elif direction == "N":
        y += 1
    elif direction == "S":
        y -= 1
    require(0 <= x < die_x and 0 <= y < die_y, "route leaves die mesh")
    return y * die_x + x


def local_manhattan(a: int, b: int, gx: int) -> int:
    ax, ay = a % gx, a // gx
    bx, by = b % gx, b // gx
    return abs(ax - bx) + abs(ay - by)


def route(hw: dict, source: int, dest: int) -> dict:
    gx = int(hw["x"])
    gy = int(hw.get("y", gx))
    die_x = int(hw.get("die", {}).get("x", 1))
    die_y = int(hw.get("die", {}).get("y", 1))
    cores_per_die = gx * gy
    total = cores_per_die * die_x * die_y
    require(0 <= source < total and 0 <= dest < total, "endpoint out of range")
    ports = c2c_ports(hw)
    cur_die, dst_die = source // cores_per_die, dest // cores_per_die
    cur_local = source % cores_per_die
    dies, directions, hops = [cur_die], [], 0
    while cur_die != dst_die:
        direction = first_dir(cur_die, dst_die, die_x)
        require(direction in ports and OPPOSITE[direction] in ports,
                f"missing bidirectional C2C peer for {direction}")
        hops += local_manhattan(cur_local, ports[direction], gx)
        cur_die = next_die(cur_die, direction, die_x, die_y)
        cur_local = ports[OPPOSITE[direction]]
        dies.append(cur_die)
        directions.append(direction)
    hops += local_manhattan(cur_local, dest % cores_per_die, gx)
    return {"dies": dies, "directions": directions,
            "d2d_hops": len(directions), "intra_die_hops": hops}


def estimate(hw: dict, source: int, dest: int, packets: int) -> dict:
    require(packets >= 1, "packets must be >= 1")
    path = route(hw, source, dest)
    if path["d2d_hops"] == 0:
        return {**path, "effective_rate": [1, 1], "per_phase_link_latency_cycles": 0,
                "transaction_link_latency_cycles": 0,
                "first_packet_service_cycles": 0, "bulk_service_cycles": 0,
                "data_first_cycles": 0, "data_last_cycles": 0,
                "transaction_d2d_cycles": 0}
    c2c = hw["die_ports"]["c2c"]
    require(c2c.get("backend") == "behavioral", "oracle requires backend=behavioral")
    port_rate = parse_rate(c2c["port_rate"], "port_rate")
    link_rate = parse_rate(c2c["link_rate"], "link_rate")
    # TODO(V5): replace this single-lane 1/1 term with the rational rate of the
    # actual shared NoC cut when multi-lane/striping is introduced; do not multiply
    # a per-lane minimum by lane count when lanes share an upstream/downstream cut.
    rate = min(Fraction(1, 1), port_rate, link_rate)
    latency = int(c2c["link_latency"])
    require(latency >= 0, "link_latency must be >= 0")
    per_phase = path["d2d_hops"] * latency
    first = ceil_fraction(1 / rate)
    bulk = ceil_fraction(Fraction(packets, 1) / rate)
    return {**path, "effective_rate": [rate.numerator, rate.denominator],
            "per_phase_link_latency_cycles": per_phase,
            "transaction_link_latency_cycles": 3 * per_phase,
            "first_packet_service_cycles": first,
            "bulk_service_cycles": bulk,
            "data_first_cycles": per_phase + first,
            "data_last_cycles": per_phase + bulk,
            "transaction_d2d_cycles": 3 * per_phase + bulk}


def fixture(die_x: int, die_y: int, directions: Tuple[str, ...]) -> dict:
    return {"x": 4, "die": {"x": die_x, "y": die_y},
            "die_ports": {"edges": {"S": {"role": "host"}},
                          "overrides": [{"side": d, "idx": 0, "role": "c2c", "dir": d}
                                        for d in directions],
                          "c2c": {"backend": "behavioral",
                                  "port_rate": {"num": 1, "den": 2},
                                  "link_rate": {"num": 1, "den": 4},
                                  "link_latency": 7}}}


def selftest() -> int:
    checks: List[Tuple[str, bool]] = []
    h31 = fixture(3, 1, ("E", "W"))
    r = route(h31, 0, 32)
    checks.append(("3x1 path E,E", r["directions"] == ["E", "E"]))
    checks.append(("3x1 intra hops", r["intra_die_hops"] == 6))
    e = estimate(h31, 0, 32, 7)
    checks.append(("min-cut 1/4", e["effective_rate"] == [1, 4]))
    checks.append(("bulk once", e["bulk_service_cycles"] == 28))
    checks.append(("2-hop transaction", e["transaction_d2d_cycles"] == 70))
    h22 = fixture(2, 2, ("E", "W", "N", "S"))
    checks.append(("2x2 X-first", route(h22, 0, 48)["directions"] == ["E", "N"]))
    checks.append(("same die zero", estimate(h22, 0, 1, 7)["transaction_d2d_cycles"] == 0))
    bad = fixture(3, 1, ("E",))
    try:
        route(bad, 0, 32)
        missing_rejected = False
    except ValueError:
        missing_rejected = True
    checks.append(("missing peer rejected", missing_rejected))
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(ok for _, ok in checks)
    print(f"V4 oracle self-test: {passed}/{len(checks)}")
    return 0 if passed == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hardware")
    ap.add_argument("--source", type=int)
    ap.add_argument("--dest", type=int)
    ap.add_argument("--packets", type=int)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    require(args.hardware and args.source is not None and args.dest is not None and
            args.packets is not None, "hardware/source/dest/packets are required")
    hw = json.loads(Path(args.hardware).read_text())
    print(json.dumps(estimate(hw, args.source, args.dest, args.packets), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
