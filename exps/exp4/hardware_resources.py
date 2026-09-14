"""Explicit analytical resource contracts for the exp4 hardware sweep."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


FREQUENCY_HZ = 500_000_000


def bandwidth_cycles(byte_count: int, capacity_GBs: float, *, frequency_hz: int = FREQUENCY_HZ) -> int:
    """Ceiling service time for decimal GB/s, expressed in simulator cycles."""

    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    if capacity_GBs <= 0:
        raise ValueError("capacity_GBs must be positive")
    return math.ceil(byte_count * frequency_hz / (capacity_GBs * 1e9))


@dataclass(frozen=True)
class ResourceSpec:
    resource_id: str
    capacity_GBs: float | None = None
    direction: str | None = None
    arbitration: str = "work_conserving_round_robin"


@dataclass(frozen=True)
class Reservation:
    action_id: str
    start_cycle: int
    finish_cycle: int
    byte_count: int = 0


@dataclass
class ResourceCalendar:
    """Deterministic non-preemptive calendar for one exclusive resource."""

    spec: ResourceSpec
    available_cycle: int = 0
    reservations: list[Reservation] = field(default_factory=list)

    def reserve(self, action_id: str, duration_cycles: int, *, ready_cycle: int = 0,
                byte_count: int = 0) -> Reservation:
        if duration_cycles < 0 or ready_cycle < 0 or byte_count < 0:
            raise ValueError("cycles and bytes must be non-negative")
        start = max(ready_cycle, self.available_cycle)
        reservation = Reservation(action_id, start, start + duration_cycles, byte_count)
        self.available_cycle = reservation.finish_cycle
        self.reservations.append(reservation)
        return reservation

    @property
    def admitted_bytes(self) -> int:
        return sum(item.byte_count for item in self.reservations)


@dataclass(frozen=True)
class SRAMSharedContract:
    """One shared read budget and one independent shared write budget per core.

    Compute, LSU, DTE and NoC receive do not receive replicated bandwidth.
    They all name the same directional resource IDs below.
    """

    core_id: int | str
    B_s_GBs: int
    bank_count: int
    bank_interleave_bytes: int = 256
    initiators: tuple[str, ...] = ("compute", "lsu", "dte", "noc_receive")

    def __post_init__(self) -> None:
        if self.B_s_GBs <= 0 or self.bank_count <= 0 or self.bank_interleave_bytes <= 0:
            raise ValueError("SRAM bandwidth, bank count and interleave must be positive")

    @property
    def read_resource_id(self) -> str:
        return f"sram.read.core[{self.core_id}]"

    @property
    def write_resource_id(self) -> str:
        return f"sram.write.core[{self.core_id}]"

    @property
    def read_spec(self) -> ResourceSpec:
        return ResourceSpec(self.read_resource_id, self.B_s_GBs, "read")

    @property
    def write_spec(self) -> ResourceSpec:
        return ResourceSpec(self.write_resource_id, self.B_s_GBs, "write")

    def resource_for(self, initiator: str, direction: str) -> str:
        if initiator not in self.initiators:
            raise ValueError(f"unknown SRAM initiator {initiator!r}")
        if direction == "read":
            return self.read_resource_id
        if direction == "write":
            return self.write_resource_id
        raise ValueError("direction must be 'read' or 'write'")

    def service_cycles(self, byte_count: int) -> int:
        return bandwidth_cycles(byte_count, self.B_s_GBs)

    def bank_for_address(self, address: int) -> int:
        if address < 0:
            raise ValueError("address must be non-negative")
        return (address // self.bank_interleave_bytes) % self.bank_count

    def new_calendars(self) -> tuple[ResourceCalendar, ResourceCalendar]:
        return ResourceCalendar(self.read_spec), ResourceCalendar(self.write_spec)

    def assert_shared_budget(self, bindings: Iterable[tuple[str, str, str]]) -> None:
        """Validate ``(initiator, direction, resource_id)`` replay bindings."""

        for initiator, direction, resource_id in bindings:
            expected = self.resource_for(initiator, direction)
            if resource_id != expected:
                raise ValueError(
                    f"{initiator}/{direction} replicates or bypasses SRAM budget: "
                    f"expected {expected!r}, got {resource_id!r}"
                )


def sram_contract_from_candidate(candidate: object, core_id: int | str) -> SRAMSharedContract:
    return SRAMSharedContract(
        core_id=core_id,
        B_s_GBs=int(getattr(candidate, "B_s_GBs")),
        bank_count=int(getattr(candidate, "sram_bank_count")),
    )


__all__ = [
    "FREQUENCY_HZ", "Reservation", "ResourceCalendar", "ResourceSpec",
    "SRAMSharedContract", "bandwidth_cycles", "sram_contract_from_candidate",
]
