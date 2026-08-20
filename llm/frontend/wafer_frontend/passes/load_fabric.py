"""N3a: lossless backend-v1 hardware JSON to :class:`PhysicalFabric`.

This module deliberately supports only the simulator subset whose units and
numbering can be represented by the current IR-1 schema.  Unsupported D2D
modes, shared link groups, per-core SRAM overrides, and implicit/theoretical
HBM bandwidth fail closed instead of being approximated.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.persistent_state import HbmAddressSpace
from ..schema.ir1 import (
    C2CPort,
    CoreSpec,
    D2DLink,
    DieSpec,
    Direction,
    MemoryInitiator,
    PhysicalFabric,
    RoutingMode,
    SramAllocator,
    SramProfile,
    SramRegionSpec,
)
from ..schema.serde import load_json_value


# Simulator evidence:
# - llm/include/macros/macros.h: M_D_DATA == 128 payload bits.
# - llm/include/defs/spec.h: HW_NOC_PAYLOAD_PER_CYCLE is packets/cycle.
# - llm/include/macros/macros.h: CYCLE == 2 ns.
# - llm/include/die/port.h: legacy link_bw is packets/cycle.
# The 256-bit physical sc_bv link carries routing metadata as well; it is not
# 32 bytes of logical payload and must not be used as the capacity unit.
SIMULATOR_PACKET_PAYLOAD_BYTES = 16
SIMULATOR_CYCLE_NS = 2
PROGRAM_ENDPOINT_LIMIT = 1 << 16

_SIDE_ORDER = ("N", "S", "W", "E")
_SIDE_DIRECTION = {
    "N": Direction.NORTH,
    "S": Direction.SOUTH,
    "W": Direction.WEST,
    "E": Direction.EAST,
}
_OPPOSITE = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.WEST: Direction.EAST,
    Direction.EAST: Direction.WEST,
}
_MAPPING_ROW = re.compile(r"^[ \t]*([+-]?\d+)[ \t]*:[ \t]*([+-]?\d+)[ \t]*$")


def _child(path: str, name: str) -> str:
    return f"{path}.{name}"


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SchemaError("expected an object", path=path)
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise SchemaError("expected an array", path=path)
    return value


def _required(node: Mapping[str, object], name: str, path: str) -> object:
    if name not in node:
        raise SchemaError("missing required field", path=_child(path, name))
    return node[name]


def _integer(value: object, path: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise SchemaError("expected int", path=path)
    if value < 0:
        raise SchemaError("must be non-negative", path=path)
    if positive and value == 0:
        raise SchemaError("must be greater than zero", path=path)
    if value >= 1 << 64:
        raise SchemaError("must fit unsigned 64-bit range", path=path)
    return value


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise SchemaError("expected a number", path=path)
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError("must be finite", path=path)
    if result < 0.0:
        raise SchemaError("must be non-negative", path=path)
    if positive and result == 0.0:
        raise SchemaError("must be greater than zero", path=path)
    return result


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise SchemaError("expected bool", path=path)
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise SchemaError("expected str", path=path)
    if not value:
        raise SchemaError("must be non-empty", path=path)
    return value


def _unknown_fields(
    node: Mapping[str, object], allowed: set[str], path: str
) -> None:
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise UnsupportedFeatureError(
            f"field {unknown[0]!r} is not representable by the N3a fabric loader",
            path=_child(path, unknown[0]),
        )


def _read_dimensions(
    hardware: Mapping[str, object], path: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    grid_x = _integer(_required(hardware, "x", path), _child(path, "x"), positive=True)
    grid_y = _integer(hardware.get("y", grid_x), _child(path, "y"), positive=True)
    die_node = _object(hardware.get("die", {}), _child(path, "die"))
    _unknown_fields(die_node, {"x", "y"}, _child(path, "die"))
    die_x = _integer(die_node.get("x", 1), f"{path}.die.x", positive=True)
    die_y = _integer(die_node.get("y", die_x), f"{path}.die.y", positive=True)
    cores_per_die = grid_x * grid_y
    die_count = die_x * die_y
    if cores_per_die * die_count + 2 * die_count > PROGRAM_ENDPOINT_LIMIT:
        raise SchemaError(
            "core + host + memory endpoints exceed backend 16-bit address space",
            path=_child(path, "die"),
        )
    return (grid_x, grid_y), (die_x, die_y)


def _read_noc_bytes_per_cycle(hardware: Mapping[str, object], path: str) -> int:
    noc_path = _child(path, "noc")
    noc = _object(_required(hardware, "noc", path), noc_path)
    packets = _integer(
        _required(noc, "noc_payload_per_cycle", noc_path),
        _child(noc_path, "noc_payload_per_cycle"),
        positive=True,
    )
    if packets > 255:
        raise SchemaError(
            "must be in [1,255] for simulator Send_prim encoding",
            path=_child(noc_path, "noc_payload_per_cycle"),
        )
    return packets * SIMULATOR_PACKET_PAYLOAD_BYTES


_INITIATORS = {
    "compute": MemoryInitiator.COMPUTE,
    "dte": MemoryInitiator.DTE,
    "lsu": MemoryInitiator.LSU,
    "noc_rx": MemoryInitiator.NOC_RX,
    "legacy": MemoryInitiator.LEGACY,
}


def _read_sram_profile(hardware: Mapping[str, object], path: str) -> SramProfile:
    memory_path = _child(path, "memory")
    memory = _object(_required(hardware, "memory", path), memory_path)
    sram_path = _child(memory_path, "sram")
    sram = _object(_required(memory, "sram", memory_path), sram_path)
    capacity = _integer(
        _required(sram, "capacity_bytes", sram_path),
        _child(sram_path, "capacity_bytes"),
        positive=True,
    )
    if "sram_size" in memory:
        legacy_capacity = _integer(
            memory["sram_size"], _child(memory_path, "sram_size"), positive=True
        )
        if legacy_capacity != capacity:
            raise SchemaError(
                "must equal memory.sram.capacity_bytes",
                path=_child(memory_path, "sram_size"),
            )
    alignment = _integer(
        _required(sram, "allocation_alignment_bytes", sram_path),
        _child(sram_path, "allocation_alignment_bytes"),
        positive=True,
    )
    bank_count = _integer(
        _required(sram, "bank_count", sram_path),
        _child(sram_path, "bank_count"),
        positive=True,
    )
    bank_interleave = _integer(
        _required(sram, "bank_interleave_bytes", sram_path),
        _child(sram_path, "bank_interleave_bytes"),
        positive=True,
    )
    capabilities: dict[str, bool] = {}
    for name in ("real_data_path", "manual_regions", "manual_memory_schedule"):
        value = _boolean(_required(sram, name, sram_path), _child(sram_path, name))
        if not value:
            raise SchemaError(
                "must be true for strict Program backend-v1",
                path=_child(sram_path, name),
            )
        capabilities[name] = value
    raw_regions = _array(_required(sram, "regions", sram_path), _child(sram_path, "regions"))
    if not raw_regions:
        raise SchemaError("must not be empty", path=_child(sram_path, "regions"))
    regions: list[SramRegionSpec] = []
    for index, raw_region in enumerate(raw_regions):
        region_path = f"{sram_path}.regions[{index}]"
        region = _object(raw_region, region_path)
        _unknown_fields(
            region,
            {"name", "base_bytes", "size_bytes", "allocator", "spillable", "access"},
            region_path,
        )
        name = _string(_required(region, "name", region_path), _child(region_path, "name"))
        allocator_text = region.get("allocator", "fixed")
        _string(allocator_text, _child(region_path, "allocator"))
        try:
            allocator = SramAllocator(allocator_text)
        except ValueError as error:
            raise SchemaError(
                "expected 'fixed' or 'block'", path=_child(region_path, "allocator")
            ) from error
        spillable = _boolean(region.get("spillable", False), _child(region_path, "spillable"))
        raw_access = region.get("access", tuple(_INITIATORS))
        accesses: list[MemoryInitiator] = []
        for access_index, item in enumerate(_array(raw_access, _child(region_path, "access"))):
            access_path = f"{region_path}.access[{access_index}]"
            text = _string(item, access_path)
            try:
                accesses.append(_INITIATORS[text])
            except KeyError as error:
                raise SchemaError("unknown SRAM initiator", path=access_path) from error
        regions.append(
            SramRegionSpec(
                id=f"sram_region_{index}",
                name=name,
                base_bytes=_integer(
                    _required(region, "base_bytes", region_path),
                    _child(region_path, "base_bytes"),
                ),
                size_bytes=_integer(
                    _required(region, "size_bytes", region_path),
                    _child(region_path, "size_bytes"),
                    positive=True,
                ),
                allocator=allocator,
                spillable=spillable,
                access=tuple(accesses),
            )
        )
    profile = SramProfile(
        id="sram_global",
        capacity_bytes=capacity,
        allocation_alignment_bytes=alignment,
        bank_count=bank_count,
        bank_interleave_bytes=bank_interleave,
        real_data_path=capabilities["real_data_path"],
        manual_regions=capabilities["manual_regions"],
        manual_memory_schedule=capabilities["manual_memory_schedule"],
        regions=tuple(regions),
    )
    profile.validate("fabric.sram_profiles[0]")
    return profile


def _validate_core_hardware(
    hardware: Mapping[str, object], cores_per_die: int, path: str
) -> None:
    cores_path = _child(path, "cores")
    raw_cores = _array(_required(hardware, "cores", path), cores_path)
    if not raw_cores:
        raise SchemaError("must not be empty", path=cores_path)
    ids: list[int] = []
    for index, raw_core in enumerate(raw_cores):
        core_path = f"{cores_path}[{index}]"
        core = _object(raw_core, core_path)
        core_id = _integer(core.get("id", index), _child(core_path, "id"))
        if core_id >= cores_per_die:
            raise SchemaError("must identify a local core in noc_grid", path=_child(core_path, "id"))
        if "sram" in core:
            raise UnsupportedFeatureError(
                "per-core SRAM overrides are not represented by the N3a loader",
                path=_child(core_path, "sram"),
            )
        ids.append(core_id)
    if ids != sorted(set(ids)):
        raise SchemaError("core ids must be unique and increasing", path=cores_path)
    if ids[0] != 0:
        raise SchemaError("first hardware core id must be zero", path=f"{cores_path}[0].id")


def _read_hbm_bytes_per_cycle(
    hardware: Mapping[str, object], die_count: int, path: str
) -> tuple[int, ...]:
    system_path = _child(path, "memory_system")
    system = _object(_required(hardware, "memory_system", path), system_path)
    topology = _string(_required(system, "topology", system_path), _child(system_path, "topology"))
    if topology != "distributed_hbm":
        raise UnsupportedFeatureError(
            "only distributed_hbm has a per-die bandwidth contract",
            path=_child(system_path, "topology"),
        )
    cache_policy = _string(system.get("cache_policy", "none"), _child(system_path, "cache_policy"))
    if cache_policy != "none":
        raise UnsupportedFeatureError(
            "distributed_hbm requires cache_policy='none'",
            path=_child(system_path, "cache_policy"),
        )
    profiles_path = _child(system_path, "profiles")
    profiles = _object(_required(system, "profiles", system_path), profiles_path)
    raw_stacks = _array(
        _required(system, "hbm_stacks", system_path), _child(system_path, "hbm_stacks")
    )
    totals = [0.0] * die_count
    seen_stacks: set[int] = set()
    for index, raw_stack in enumerate(raw_stacks):
        stack_path = f"{system_path}.hbm_stacks[{index}]"
        stack = _object(raw_stack, stack_path)
        stack_id = _integer(_required(stack, "stack_id", stack_path), _child(stack_path, "stack_id"))
        if stack_id in seen_stacks:
            raise SchemaError("duplicate stack_id", path=_child(stack_path, "stack_id"))
        seen_stacks.add(stack_id)
        die_id = _integer(
            _required(stack, "compute_die_id", stack_path),
            _child(stack_path, "compute_die_id"),
        )
        if die_id >= die_count:
            raise SchemaError("out of die_grid range", path=_child(stack_path, "compute_die_id"))
        profile = _string(_required(stack, "profile", stack_path), _child(stack_path, "profile"))
        if profile not in profiles:
            raise SchemaError("references an unknown HBM profile", path=_child(stack_path, "profile"))
        backend = _string(stack.get("backend", "dramsys"), _child(stack_path, "backend"))
        if backend != "behavioral":
            raise UnsupportedFeatureError(
                "N3a requires behavioral HBM with an explicit bandwidth cap; DRAMSys requires resolving its external memspec",
                path=_child(stack_path, "backend"),
            )
        cap = _number(
            _required(stack, "bandwidth_cap_GBps", stack_path),
            _child(stack_path, "bandwidth_cap_GBps"),
            positive=True,
        )
        efficiency = _number(
            stack.get("behavioral_efficiency", 1.0),
            _child(stack_path, "behavioral_efficiency"),
            positive=True,
        )
        if efficiency > 1.0:
            raise SchemaError("must not exceed 1", path=_child(stack_path, "behavioral_efficiency"))
        totals[die_id] += cap * efficiency * SIMULATOR_CYCLE_NS
    result: list[int] = []
    for die_id, value in enumerate(totals):
        die_path = f"{system_path}.derived_hbm_bytes_per_cycle[{die_id}]"
        if value <= 0.0:
            raise SchemaError("every compute die must have positive local HBM bandwidth", path=die_path)
        rounded = round(value)
        if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12):
            raise UnsupportedFeatureError(
                "fractional bytes/cycle cannot be represented by PhysicalFabric v1",
                path=die_path,
            )
        result.append(_integer(rounded, die_path, positive=True))
    return tuple(result)


def hbm_address_spaces_from_data(
    value: object, *, path: str = "hardware"
) -> tuple[HbmAddressSpace, ...]:
    """Construct canonical per-die HBM address spaces from strict hardware data.

    Address ownership and capacity come from ``address_policy.home_ranges`` and
    ``hbm_stacks`` respectively. Bandwidth and backend support remain fabric
    properties, but are validated here as well so an address space cannot be
    constructed for an HBM configuration that the backend cannot execute.
    """

    hardware = _object(value, path)
    _, die_grid = _read_dimensions(hardware, path)
    die_count = die_grid[0] * die_grid[1]

    # Rate belongs to DieSpec, not the content-addressed address space.
    _read_hbm_bytes_per_cycle(hardware, die_count, path)

    system_path = _child(path, "memory_system")
    system = _object(_required(hardware, "memory_system", path), system_path)
    stacks_path = _child(system_path, "hbm_stacks")
    raw_stacks = _array(_required(system, "hbm_stacks", system_path), stacks_path)
    capacity_by_die = [0] * die_count
    for index, raw_stack in enumerate(raw_stacks):
        stack_path = f"{stacks_path}[{index}]"
        stack = _object(raw_stack, stack_path)
        die_id = _integer(
            _required(stack, "compute_die_id", stack_path),
            _child(stack_path, "compute_die_id"),
        )
        if die_id >= die_count:
            raise SchemaError(
                "out of die_grid range", path=_child(stack_path, "compute_die_id")
            )
        capacity = _integer(
            _required(stack, "capacity_bytes", stack_path),
            _child(stack_path, "capacity_bytes"),
            positive=True,
        )
        capacity_by_die[die_id] += capacity
        _integer(
            capacity_by_die[die_id],
            f"{system_path}.derived_hbm_capacity_bytes[{die_id}]",
            positive=True,
        )

    policy_path = _child(system_path, "address_policy")
    policy = _object(_required(system, "address_policy", system_path), policy_path)
    _unknown_fields(
        policy,
        {
            "mode",
            "home_ranges",
            "stack_interleave_bytes",
            "channel_interleave_bytes",
        },
        policy_path,
    )
    mode = _string(_required(policy, "mode", policy_path), _child(policy_path, "mode"))
    if mode != "numa_local_interleave":
        raise UnsupportedFeatureError(
            "HBM address spaces require numa_local_interleave ownership",
            path=_child(policy_path, "mode"),
        )
    alignment = _integer(
        _required(policy, "channel_interleave_bytes", policy_path),
        _child(policy_path, "channel_interleave_bytes"),
        positive=True,
    )
    stack_interleave = _integer(
        _required(policy, "stack_interleave_bytes", policy_path),
        _child(policy_path, "stack_interleave_bytes"),
        positive=True,
    )
    if stack_interleave % alignment:
        raise SchemaError(
            "must be a multiple of channel_interleave_bytes",
            path=_child(policy_path, "stack_interleave_bytes"),
        )

    ranges_path = _child(policy_path, "home_ranges")
    raw_ranges = _array(_required(policy, "home_ranges", policy_path), ranges_path)
    if len(raw_ranges) != die_count:
        raise SchemaError(
            "must contain exactly one home range for every compute die",
            path=ranges_path,
        )

    spaces: list[HbmAddressSpace] = []
    occupied: list[tuple[int, int]] = []
    for index, raw_range in enumerate(raw_ranges):
        range_path = f"{ranges_path}[{index}]"
        home_range = _object(raw_range, range_path)
        _unknown_fields(home_range, {"die_id", "base", "size_bytes"}, range_path)
        die_id = _integer(
            _required(home_range, "die_id", range_path),
            _child(range_path, "die_id"),
        )
        if die_id != index:
            raise SchemaError(
                "home ranges must use canonical die order 0..N-1",
                path=_child(range_path, "die_id"),
            )
        base = _integer(
            _required(home_range, "base", range_path), _child(range_path, "base")
        )
        size = _integer(
            _required(home_range, "size_bytes", range_path),
            _child(range_path, "size_bytes"),
            positive=True,
        )
        if size != capacity_by_die[die_id]:
            raise SchemaError(
                "must exactly equal the capacity of HBM stacks homed on this die",
                path=_child(range_path, "size_bytes"),
            )
        space = HbmAddressSpace.create(
            die_id=die_id,
            base_address=base,
            size_bytes=size,
            alignment_bytes=alignment,
        )
        start, end = space.base_address, space.base_address + space.size_bytes
        if any(start < old_end and old_start < end for old_start, old_end in occupied):
            raise SchemaError("HBM home ranges must not overlap", path=range_path)
        occupied.append((start, end))
        spaces.append(space)
    return tuple(spaces)


@dataclass(frozen=True, slots=True)
class _TemplatePort:
    runtime_port_id: int
    side: Direction
    direction: Direction
    role: str
    noc_coord: tuple[int, int]
    bytes_per_cycle: int
    buffer_packets: int


def _edge_length(side: str, noc_grid: tuple[int, int]) -> int:
    return noc_grid[0] if side in ("N", "S") else noc_grid[1]


def _port_coord(side: str, index: int, noc_grid: tuple[int, int]) -> tuple[int, int]:
    if side == "N":
        return (index, noc_grid[1] - 1)
    if side == "S":
        return (index, 0)
    if side == "W":
        return (0, index)
    return (noc_grid[0] - 1, index)


def _read_role_spec(
    value: object, side: str, path: str
) -> tuple[str, Direction]:
    spec = _object(value, path)
    # Override records carry their selector beside the role specification;
    # edge defaults contain only the latter fields.
    _unknown_fields(spec, {"side", "idx", "role", "dir", "link_group"}, path)
    role = _string(_required(spec, "role", path), _child(path, "role"))
    if role not in ("host", "mem", "c2c"):
        raise SchemaError("expected 'host', 'mem', or 'c2c'", path=_child(path, "role"))
    direction = _SIDE_DIRECTION[side]
    if role == "c2c" and "dir" in spec:
        raw_direction = _string(spec["dir"], _child(path, "dir"))
        if raw_direction not in _SIDE_DIRECTION:
            raise SchemaError("expected N/S/W/E", path=_child(path, "dir"))
        direction = _SIDE_DIRECTION[raw_direction]
        if direction is not _SIDE_DIRECTION[side]:
            raise SchemaError("backend-v1 requires C2C dir == side", path=_child(path, "dir"))
    if role != "c2c" and "dir" in spec:
        raise SchemaError("dir only applies to role=c2c", path=_child(path, "dir"))
    if "link_group" in spec:
        raise UnsupportedFeatureError(
            "current IR-1 port contract cannot losslessly represent simulator directed-pair link_group resources",
            path=_child(path, "link_group"),
        )
    return role, direction


def _read_template_ports(
    hardware: Mapping[str, object],
    noc_grid: tuple[int, int],
    die_grid: tuple[int, int],
    path: str,
) -> tuple[_TemplatePort, ...]:
    ports_path = _child(path, "die_ports")
    if "die_ports" not in hardware:
        if die_grid[0] * die_grid[1] > 1:
            raise SchemaError("multi-die backend-v1 requires die_ports", path=ports_path)
        return ()
    ports_node = _object(hardware["die_ports"], ports_path)
    _unknown_fields(ports_node, {"edges", "overrides", "c2c"}, ports_path)
    c2c_path = _child(ports_path, "c2c")
    c2c = _object(ports_node.get("c2c", {}), c2c_path)
    legacy_fields = {"link_bw", "bw_per_cycle", "latency", "buffer_depth"}
    v3_fields = {
        "safety", "port_rate", "link_rate", "link_latency",
        "saf_buffer_depth", "link_inflight_depth", "rx_buffer_depth", "ctrl_buffer_depth",
    }
    _unknown_fields(
        c2c,
        legacy_fields | v3_fields | {"backend", "mode", "multi_port", "select_policy", "select_seed"},
        c2c_path,
    )
    backend = _string(c2c.get("backend", "cycle"), _child(c2c_path, "backend"))
    mode = _string(c2c.get("mode", "functional_v2"), _child(c2c_path, "mode"))
    if backend != "cycle" or mode != "functional_v2" or any(name in c2c for name in v3_fields):
        raise UnsupportedFeatureError(
            "N3a supports only cycle/functional_v2 legacy C2C",
            path=c2c_path,
        )
    multi_port = _boolean(
        c2c.get("multi_port", False), _child(c2c_path, "multi_port")
    )
    if multi_port or "select_policy" in c2c or "select_seed" in c2c:
        raise UnsupportedFeatureError("multi-port C2C is outside backend-v1", path=c2c_path)
    if "link_bw" in c2c and "bw_per_cycle" in c2c:
        raise SchemaError("link_bw and bw_per_cycle are ambiguous aliases", path=c2c_path)
    bandwidth_field = "link_bw" if "link_bw" in c2c else "bw_per_cycle"
    packets_per_cycle = _integer(c2c.get(bandwidth_field, 1), _child(c2c_path, bandwidth_field), positive=True)
    if packets_per_cycle != 1:
        raise UnsupportedFeatureError(
            "backend-v1 requires exactly 1 C2C packet/cycle",
            path=_child(c2c_path, bandwidth_field),
        )
    latency = _integer(c2c.get("latency", 0), _child(c2c_path, "latency"))
    buffer_packets = _integer(c2c.get("buffer_depth", 1), _child(c2c_path, "buffer_depth"), positive=True)

    edges_path = _child(ports_path, "edges")
    edges = _object(ports_node.get("edges", {}), edges_path)
    _unknown_fields(edges, set(_SIDE_ORDER), edges_path)
    overrides_path = _child(ports_path, "overrides")
    overrides_raw = _array(ports_node.get("overrides", ()), overrides_path)
    overrides: dict[tuple[str, int], tuple[object, str]] = {}
    for override_index, raw_override in enumerate(overrides_raw):
        override_path = f"{overrides_path}[{override_index}]"
        override = _object(raw_override, override_path)
        _unknown_fields(override, {"side", "idx", "role", "dir", "link_group"}, override_path)
        side = _string(_required(override, "side", override_path), _child(override_path, "side"))
        if side not in _SIDE_DIRECTION:
            raise SchemaError("expected N/S/W/E", path=_child(override_path, "side"))
        raw_indices = _required(override, "idx", override_path)
        indices = _array(raw_indices, _child(override_path, "idx")) if isinstance(raw_indices, (list, tuple)) else (raw_indices,)
        for raw_index in indices:
            index = _integer(raw_index, _child(override_path, "idx"))
            if index >= _edge_length(side, noc_grid):
                raise SchemaError("port index is outside edge", path=_child(override_path, "idx"))
            key = (side, index)
            if key in overrides:
                raise SchemaError("duplicate (side,idx) override", path=override_path)
            overrides[key] = (override, override_path)

    template: list[_TemplatePort] = []
    for side in _SIDE_ORDER:
        default = edges.get(side)
        for index in range(_edge_length(side, noc_grid)):
            override_entry = overrides.get((side, index))
            if override_entry is not None:
                spec, spec_path = override_entry
            elif default is not None:
                spec, spec_path = default, f"{edges_path}.{side}"
            else:
                continue
            role, direction = _read_role_spec(spec, side, spec_path)
            template.append(
                _TemplatePort(
                    runtime_port_id=len(template),
                    side=_SIDE_DIRECTION[side],
                    direction=direction,
                    role=role,
                    noc_coord=_port_coord(side, index, noc_grid),
                    bytes_per_cycle=packets_per_cycle * SIMULATOR_PACKET_PAYLOAD_BYTES,
                    buffer_packets=buffer_packets,
                )
            )
    c2c_ports = [port for port in template if port.role == "c2c"]
    counts = {direction: sum(port.direction is direction for port in c2c_ports) for direction in Direction}
    for direction, count in counts.items():
        if count > 1:
            raise SchemaError("backend-v1 permits one C2C port per direction", path=ports_path)
        needs_neighbor = direction in (Direction.EAST, Direction.WEST) and die_grid[0] > 1
        needs_neighbor = needs_neighbor or direction in (Direction.NORTH, Direction.SOUTH) and die_grid[1] > 1
        if needs_neighbor and count != 1:
            raise SchemaError("die-neighbor direction requires exactly one C2C port", path=ports_path)
    if die_grid[0] * die_grid[1] > 1 and not any(port.role == "host" for port in template):
        raise SchemaError("multi-die hardware requires at least one HOST port", path=ports_path)
    return tuple(template)


def _neighbor(die_id: int, direction: Direction, die_grid: tuple[int, int]) -> int | None:
    x = die_id % die_grid[0]
    y = die_id // die_grid[0]
    if direction is Direction.EAST:
        x += 1
    elif direction is Direction.WEST:
        x -= 1
    elif direction is Direction.NORTH:
        y += 1
    else:
        y -= 1
    if x < 0 or x >= die_grid[0] or y < 0 or y >= die_grid[1]:
        return None
    return y * die_grid[0] + x


def _physical_ports(die_id: int, template: tuple[_TemplatePort, ...]) -> tuple[C2CPort, ...]:
    result: list[C2CPort] = []
    for port in template:
        if port.role != "c2c":
            continue
        direction = port.direction.value
        result.append(
            C2CPort(
                id=f"c2c_port_d{die_id}_p{port.runtime_port_id}",
                runtime_port_id=port.runtime_port_id,
                side=port.side,
                direction=port.direction,
                noc_coord=port.noc_coord,
                egress_resource_id=f"port_d{die_id}_{direction}",
                bytes_per_cycle=port.bytes_per_cycle,
                buffer_packets=port.buffer_packets,
            )
        )
    return tuple(result)


def _build_links(
    dies: tuple[DieSpec, ...],
    template: tuple[_TemplatePort, ...],
    die_grid: tuple[int, int],
    latency_cycles: int,
) -> tuple[D2DLink, ...]:
    by_die_runtime = {
        die.id: {port.runtime_port_id: port for port in die.ports} for die in dies
    }
    c2c_template = tuple(port for port in template if port.role == "c2c")
    links: list[D2DLink] = []
    for die in dies:
        for port in c2c_template:
            destination = _neighbor(die.id, port.direction, die_grid)
            if destination is None:
                continue
            perpendicular = port.noc_coord[1] if port.direction in (Direction.EAST, Direction.WEST) else port.noc_coord[0]
            mirrors = [
                candidate
                for candidate in c2c_template
                if candidate.side is _OPPOSITE[port.side]
                and candidate.direction is _OPPOSITE[port.direction]
                and (candidate.noc_coord[1] if candidate.direction in (Direction.EAST, Direction.WEST) else candidate.noc_coord[0]) == perpendicular
            ]
            if len(mirrors) != 1:
                raise SchemaError(
                    "C2C port has no unique reciprocal mirror at the same edge coordinate",
                    path="hardware.die_ports",
                )
            mirror = mirrors[0]
            source_port = by_die_runtime[die.id][port.runtime_port_id]
            destination_port = by_die_runtime[destination][mirror.runtime_port_id]
            links.append(
                D2DLink(
                    id=f"link_d{die.id}_d{destination}",
                    source_die=die.id,
                    source_port_ref=source_port.id,
                    destination_die=destination,
                    destination_port_ref=destination_port.id,
                    bytes_per_cycle=port.bytes_per_cycle,
                    latency_cycles=latency_cycles,
                    resource_id=f"d2d_d{die.id}_d{destination}",
                    link_group_ref=None,
                )
            )
    return tuple(links)


def physical_fabric_from_data(value: object, *, path: str = "hardware") -> PhysicalFabric:
    """Construct a deterministic PhysicalFabric from already-decoded hardware JSON."""

    hardware = _object(value, path)
    noc_grid, die_grid = _read_dimensions(hardware, path)
    _validate_core_hardware(hardware, noc_grid[0] * noc_grid[1], path)
    noc_bytes_per_cycle = _read_noc_bytes_per_cycle(hardware, path)
    sram_profile = _read_sram_profile(hardware, path)
    hbm_per_die = _read_hbm_bytes_per_cycle(hardware, die_grid[0] * die_grid[1], path)
    template = _read_template_ports(hardware, noc_grid, die_grid, path)
    c2c_node = _object(_object(hardware.get("die_ports", {}), f"{path}.die_ports").get("c2c", {}), f"{path}.die_ports.c2c")
    latency_cycles = _integer(c2c_node.get("latency", 0), f"{path}.die_ports.c2c.latency")
    cores_per_die = noc_grid[0] * noc_grid[1]
    dies: list[DieSpec] = []
    for die_id in range(die_grid[0] * die_grid[1]):
        cores = tuple(
            CoreSpec(
                id=f"core_d{die_id}_c{local_core_id}",
                local_core_id=local_core_id,
                runtime_core_id=die_id * cores_per_die + local_core_id,
                noc_coord=(local_core_id % noc_grid[0], local_core_id // noc_grid[0]),
                sram_profile_ref=sram_profile.id,
            )
            for local_core_id in range(cores_per_die)
        )
        dies.append(
            DieSpec(
                id=die_id,
                coord=(die_id % die_grid[0], die_id // die_grid[0]),
                noc_grid=noc_grid,
                noc_bytes_per_cycle=noc_bytes_per_cycle,
                hbm_bytes_per_cycle=hbm_per_die[die_id],
                cores=cores,
                ports=_physical_ports(die_id, template),
            )
        )
    links = _build_links(tuple(dies), template, die_grid, latency_cycles)
    result = PhysicalFabric(
        routing_mode=RoutingMode.BACKEND_XY_V1,
        die_grid=die_grid,
        sram_profiles=(sram_profile,),
        dies=tuple(dies),
        links=links,
    )
    result.validate("fabric")
    return result


def validate_identity_mapping_text(
    text: str, *, total_cores: int, path: str = "mapping"
) -> tuple[tuple[int, int], ...]:
    """Validate C++ ``mapping.spec`` grammar and require an identity remap."""

    if type(text) is not str:
        raise SchemaError("expected mapping text as str", path=path)
    _integer(total_cores, f"{path}.total_cores", positive=True)
    entries: list[tuple[int, int]] = []
    seen_sources: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line == "":
            continue
        line_path = f"{path}.line[{line_number}]"
        match = _MAPPING_ROW.fullmatch(line)
        if match is None:
            raise SchemaError("must use '<source>:<destination>' with no trailing data", path=line_path)
        source = int(match.group(1))
        destination = int(match.group(2))
        if source < 0 or source >= total_cores or destination < 0 or destination >= total_cores:
            raise SchemaError("core id is outside PhysicalFabric runtime core range", path=line_path)
        if source in seen_sources:
            raise SchemaError("duplicate mapping source", path=line_path)
        seen_sources.add(source)
        if source != destination:
            raise UnsupportedFeatureError(
                "strict Program backend does not consume legacy CoreConfigRemap; only identity mapping is valid",
                path=line_path,
            )
        entries.append((source, destination))
    return tuple(entries)


def load_physical_fabric_and_hbm_address_spaces(
    hardware_path: str | Path,
    mapping_path: str | Path,
) -> tuple[PhysicalFabric, tuple[HbmAddressSpace, ...]]:
    """Load one strict hardware document into fabric and HBM address spaces."""

    hardware_source = Path(hardware_path)
    mapping_source = Path(mapping_path)
    value = load_json_value(hardware_source, path="hardware")
    fabric = physical_fabric_from_data(value, path="hardware")
    hbm_address_spaces = hbm_address_spaces_from_data(value, path="hardware")
    try:
        mapping_text = mapping_source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SchemaError(str(error), path="mapping") from error
    validate_identity_mapping_text(
        mapping_text,
        total_cores=sum(len(die.cores) for die in fabric.dies),
        path="mapping",
    )
    return fabric, hbm_address_spaces


def load_physical_fabric(
    hardware_path: str | Path,
    mapping_path: str | Path,
) -> PhysicalFabric:
    """Load strict JSON, validate identity mapping, and return PhysicalFabric."""

    fabric, _ = load_physical_fabric_and_hbm_address_spaces(
        hardware_path,
        mapping_path,
    )
    return fabric
