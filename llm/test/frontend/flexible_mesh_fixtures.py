"""Shared deterministic fixtures for complete rectangular die meshes."""

from __future__ import annotations


def minimal_hardware(
    die_x: int,
    die_y: int,
    *,
    sram_bytes: int = 4096,
) -> dict[str, object]:
    """Return the smallest production-valid hardware document for one mesh."""

    if (
        type(die_x) is not int
        or type(die_y) is not int
        or die_x <= 0
        or die_y <= 0
        or type(sram_bytes) is not int
        or sram_bytes <= 0
    ):
        raise ValueError("mesh dimensions and SRAM bytes must be positive integers")
    overrides: list[dict[str, object]] = [
        {"side": "E", "idx": 0, "role": "c2c", "dir": "E"},
        {"side": "W", "idx": 0, "role": "c2c", "dir": "W"},
    ]
    if die_y > 1:
        overrides = [
            {"side": "N", "idx": 0, "role": "c2c", "dir": "N"},
            {"side": "S", "idx": 0, "role": "c2c", "dir": "S"},
        ] + overrides
    stacks = [
        {
            "stack_id": die_id,
            "compute_die_id": die_id,
            "profile": "test_hbm",
            "backend": "behavioral",
            "capacity_bytes": 1048576,
            "bandwidth_cap_GBps": 8.0,
        }
        for die_id in range(die_x * die_y)
    ]
    return {
        "x": 2,
        "y": 2,
        "die": {"x": die_x, "y": die_y},
        "noc": {"noc_payload_per_cycle": 4},
        "memory": {
            "sram_size": sram_bytes,
            "sram": {
                "capacity_bytes": sram_bytes,
                "allocation_alignment_bytes": 64,
                "bank_count": 4,
                "bank_interleave_bytes": 64,
                "real_data_path": True,
                "manual_regions": True,
                "manual_memory_schedule": True,
                "regions": [
                    {
                        "name": "sram",
                        "base_bytes": 0,
                        "size_bytes": sram_bytes,
                        "allocator": "block",
                        "spillable": False,
                        "access": [
                            "compute",
                            "dte",
                            "lsu",
                            "noc_rx",
                            "legacy",
                        ],
                    }
                ],
            },
        },
        "die_ports": {
            "edges": {"S": {"role": "host"}},
            "overrides": overrides,
            "c2c": {
                "link_bw": 1,
                "latency": 3,
                "buffer_depth": 8,
            },
        },
        "memory_system": {
            "topology": "distributed_hbm",
            "cache_policy": "none",
            "profiles": {"test_hbm": {"channels_per_stack": 1}},
            "hbm_stacks": stacks,
            "address_policy": {
                "mode": "numa_local_interleave",
                "home_ranges": [
                    {
                        "die_id": die_id,
                        "base": die_id * 1048576,
                        "size_bytes": 1048576,
                    }
                    for die_id in range(die_x * die_y)
                ],
                "stack_interleave_bytes": 1048576,
                "channel_interleave_bytes": 64,
            },
        },
        "cores": [{"id": 0}],
    }


__all__ = ["minimal_hardware"]
