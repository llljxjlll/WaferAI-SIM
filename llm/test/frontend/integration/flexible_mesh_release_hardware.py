"""Canonical p5_large hardware specialization for flexible-Mesh release runs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError


_ROOT = Path(__file__).resolve().parents[4]
_P5_LARGE_HARDWARE = _ROOT / "llm/test/program/p5_large_hardware.json"
_PER_DIE_HBM_BYTES = 1 << 20
_RELEASE_SRAM_BYTES = 1 << 20
_RELEASE_SRAM_REGION = "dense_release"


def p5_large_hardware_template_json() -> str:
    """Return the checked-in p5_large template bytes as text."""

    return _P5_LARGE_HARDWARE.read_text(encoding="utf-8")


def specialize_release_hardware(
    hardware_template_json: str,
    rows: int,
    columns: int,
) -> str:
    """Deep-copy and canonically specialize p5_large for one 1..10 mesh.

    All release families deliberately share this exact representation: C2C
    ports use the p5_large port index, SRAM is one non-overlapping 1 MiB
    region, and every die owns one 1 MiB HBM address range.
    """

    if (
        type(rows) is not int
        or type(columns) is not int
        or not (1 <= rows <= 10)
        or not (1 <= columns <= 10)
    ):
        raise SchemaError("requires a 1..10 rectangle", path="mesh")
    if type(hardware_template_json) is not str or not hardware_template_json:
        raise SchemaError("hardware template JSON is empty", path="hardware_template_json")
    try:
        parsed = json.loads(hardware_template_json)
    except json.JSONDecodeError as error:
        raise SchemaError(
            "hardware template JSON is invalid", path="hardware_template_json"
        ) from error
    if type(parsed) is not dict:
        raise SchemaError(
            "hardware template must be an object", path="hardware_template_json"
        )
    hardware = deepcopy(parsed)
    memory = hardware.get("memory")
    die_ports = hardware.get("die_ports")
    memory_system = hardware.get("memory_system")
    if type(memory) is not dict or type(memory.get("sram")) is not dict:
        raise SchemaError("P5 template lacks SRAM", path="hardware.memory")
    if type(die_ports) is not dict:
        raise SchemaError("P5 template lacks die ports", path="hardware.die_ports")
    if type(memory_system) is not dict:
        raise SchemaError(
            "P5 template lacks memory system", path="hardware.memory_system"
        )
    stacks = memory_system.get("hbm_stacks")
    address_policy = memory_system.get("address_policy")
    if type(stacks) is not list or not stacks or type(stacks[0]) is not dict:
        raise SchemaError(
            "P5 template lacks an HBM stack", path="hardware.memory_system.hbm_stacks"
        )
    if type(address_policy) is not dict:
        raise SchemaError(
            "P5 template lacks address policy",
            path="hardware.memory_system.address_policy",
        )

    hardware["die"] = {"x": columns, "y": rows}
    die_ports["overrides"] = [
        {"side": side, "idx": 2, "role": "c2c", "dir": side}
        for side in ("N", "E", "S", "W")
        if (side in ("N", "S") and rows > 1)
        or (side in ("E", "W") and columns > 1)
    ]

    memory["sram_size"] = _RELEASE_SRAM_BYTES
    sram = memory["sram"]
    sram["capacity_bytes"] = _RELEASE_SRAM_BYTES
    sram["regions"] = [{
        "name": _RELEASE_SRAM_REGION,
        "base_bytes": 0,
        "size_bytes": _RELEASE_SRAM_BYTES,
        "allocator": "block",
        "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]

    rank_count = rows * columns
    base_stack = stacks[0]
    memory_system["hbm_stacks"] = [
        {
            **base_stack,
            "stack_id": rank,
            "compute_die_id": rank,
            "capacity_bytes": _PER_DIE_HBM_BYTES,
        }
        for rank in range(rank_count)
    ]
    address_policy["home_ranges"] = [
        {
            "die_id": rank,
            "base": rank * _PER_DIE_HBM_BYTES,
            "size_bytes": _PER_DIE_HBM_BYTES,
        }
        for rank in range(rank_count)
    ]
    address_policy["stack_interleave_bytes"] = _PER_DIE_HBM_BYTES
    return json.dumps(hardware, sort_keys=True, separators=(",", ":"))


def specialize_p5_large_release_hardware(rows: int, columns: int) -> str:
    return specialize_release_hardware(
        p5_large_hardware_template_json(), rows, columns,
    )


__all__ = [
    "p5_large_hardware_template_json",
    "specialize_p5_large_release_hardware",
    "specialize_release_hardware",
]
