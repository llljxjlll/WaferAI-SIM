from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.load_fabric import (
    SIMULATOR_CYCLE_NS,
    SIMULATOR_PACKET_PAYLOAD_BYTES,
    hbm_address_spaces_from_data,
    load_physical_fabric,
    physical_fabric_from_data,
    validate_identity_mapping_text,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    load_json_value,
    loads_json_value,
)


ROOT = Path(__file__).resolve().parents[4]


def minimal_hardware(die_x: int, die_y: int) -> dict[str, object]:
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
            "sram_size": 4096,
            "sram": {
                "capacity_bytes": 4096,
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
                        "size_bytes": 4096,
                        "allocator": "block",
                        "spillable": False,
                        "access": ["compute", "dte", "lsu", "noc_rx", "legacy"],
                    }
                ],
            },
        },
        "die_ports": {
            "edges": {"S": {"role": "host"}},
            "overrides": overrides,
            "c2c": {"link_bw": 1, "latency": 3, "buffer_depth": 8},
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


class FabricLoaderTest(unittest.TestCase):
    def test_verified_simulator_unit_constants(self) -> None:
        self.assertEqual(SIMULATOR_PACKET_PAYLOAD_BYTES, 128 // 8)
        self.assertEqual(SIMULATOR_CYCLE_NS, 2)

    def test_real_n0_2x1_fixture_has_exact_units_and_runtime_ids(self) -> None:
        fabric = load_physical_fabric(
            ROOT / "llm/test/sram/hardware_numa.json",
            ROOT / "llm/test/default/mapping.spec",
        )
        self.assertEqual(fabric.die_grid, (2, 1))
        self.assertEqual(len(fabric.dies), 2)
        self.assertEqual(fabric.dies[0].noc_bytes_per_cycle, 4 * 16)
        self.assertEqual(fabric.dies[0].hbm_bytes_per_cycle, 8 * 2)
        self.assertEqual(
            [
                (port.runtime_port_id, port.direction.value, port.noc_coord, port.bytes_per_cycle)
                for port in fabric.dies[0].ports
            ],
            [(4, "west", (0, 2), 16), (5, "east", (3, 2), 16)],
        )
        self.assertEqual(
            [
                (link.source_die, link.destination_die, link.bytes_per_cycle, link.latency_cycles)
                for link in fabric.links
            ],
            [(0, 1, 16, 3), (1, 0, 16, 3)],
        )
        self.assertEqual(fabric.dies[1].cores[0].runtime_core_id, 16)
        self.assertEqual(fabric.dies[1].cores[-1].runtime_core_id, 31)

    def test_hbm_address_spaces_match_real_2x1_and_2x2_hardware(self) -> None:
        for name, die_count in (("hardware_2x1.json", 2), ("hardware_2x2.json", 4)):
            with self.subTest(name=name):
                value = load_json_value(ROOT / "notes/frontend/examples" / name, path="hardware")
                spaces = hbm_address_spaces_from_data(value)
                fabric = physical_fabric_from_data(value)
                self.assertEqual(
                    [
                        (
                            space.die_id,
                            space.base_address,
                            space.size_bytes,
                            space.alignment_bytes,
                            fabric.dies[space.die_id].hbm_bytes_per_cycle,
                        )
                        for space in spaces
                    ],
                    [
                        (die_id, die_id * 1048576, 1048576, 64, 16)
                        for die_id in range(die_count)
                    ],
                )

    def test_hbm_address_space_construction_is_deterministic_and_input_immutable(self) -> None:
        hardware = minimal_hardware(2, 2)
        before = deepcopy(hardware)
        left = hbm_address_spaces_from_data(hardware)
        right = hbm_address_spaces_from_data(hardware)
        self.assertEqual(hardware, before)
        self.assertEqual(canonical_digest(left), canonical_digest(right))

    def test_hbm_address_spaces_reject_wrong_home_and_crossing(self) -> None:
        hardware = minimal_hardware(2, 1)
        home_ranges = hardware["memory_system"]["address_policy"]["home_ranges"]  # type: ignore[index]
        home_ranges[1]["die_id"] = 0  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "canonical die order"):
            hbm_address_spaces_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        home_ranges = hardware["memory_system"]["address_policy"]["home_ranges"]  # type: ignore[index]
        home_ranges[1]["base"] = 1048576 - 64  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "must not overlap"):
            hbm_address_spaces_from_data(hardware)

    def test_hbm_address_spaces_reject_capacity_alignment_backend_and_rate_mismatch(self) -> None:
        hardware = minimal_hardware(2, 1)
        home_range = hardware["memory_system"]["address_policy"]["home_ranges"][0]  # type: ignore[index]
        home_range["size_bytes"] = 1048576 - 64  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "exactly equal"):
            hbm_address_spaces_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        policy = hardware["memory_system"]["address_policy"]  # type: ignore[index]
        policy["channel_interleave_bytes"] = 96  # type: ignore[index]
        policy["stack_interleave_bytes"] = 96  # type: ignore[index]
        with self.assertRaisesRegex(SchemaError, "power of two"):
            hbm_address_spaces_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        stack = hardware["memory_system"]["hbm_stacks"][0]  # type: ignore[index]
        stack["backend"] = "dramsys"  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "external memspec"):
            hbm_address_spaces_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        stack = hardware["memory_system"]["hbm_stacks"][0]  # type: ignore[index]
        stack["bandwidth_cap_GBps"] = 8.25  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "fractional bytes"):
            hbm_address_spaces_from_data(hardware)

    def test_2x2_link_and_port_numbering_matches_simulator_construction(self) -> None:
        hardware = minimal_hardware(2, 2)
        fabric = physical_fabric_from_data(hardware)
        self.assertEqual(fabric.die_grid, (2, 2))
        self.assertEqual(
            [
                (port.runtime_port_id, port.direction.value, port.noc_coord)
                for port in fabric.dies[0].ports
            ],
            [
                (0, "north", (0, 1)),
                (1, "south", (0, 0)),
                (3, "west", (0, 0)),
                (4, "east", (1, 0)),
            ],
        )
        self.assertEqual(
            [(link.source_die, link.destination_die) for link in fabric.links],
            [
                (0, 2), (0, 1),
                (1, 3), (1, 0),
                (2, 0), (2, 3),
                (3, 1), (3, 2),
            ],
        )
        self.assertTrue(all(link.bytes_per_cycle == 16 for link in fabric.links))
        self.assertTrue(all(die.noc_bytes_per_cycle == 64 for die in fabric.dies))
        self.assertTrue(all(die.hbm_bytes_per_cycle == 16 for die in fabric.dies))

    def test_fabric_construction_is_digest_deterministic_and_input_immutable(self) -> None:
        hardware = minimal_hardware(2, 2)
        before = deepcopy(hardware)
        left = physical_fabric_from_data(hardware)
        right = physical_fabric_from_data(hardware)
        self.assertEqual(hardware, before)
        self.assertEqual(canonical_digest(left), canonical_digest(right))

    def test_strict_raw_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(SchemaError, "duplicate object key"):
            loads_json_value('{"x":1,"x":2}', path="hardware")
        with self.assertRaisesRegex(SchemaError, "non-finite"):
            loads_json_value('{"x":NaN}', path="hardware")

    def test_identity_mapping_accepts_blank_partial_and_full_identity(self) -> None:
        self.assertEqual(validate_identity_mapping_text("", total_cores=4), ())
        self.assertEqual(
            validate_identity_mapping_text("0:0\n 2 : 2 \n", total_cores=4),
            ((0, 0), (2, 2)),
        )

    def test_mapping_rejects_nonidentity_duplicate_out_of_range_and_bad_grammar(self) -> None:
        cases = (
            ("0:1\n", UnsupportedFeatureError, "strict Program"),
            ("0:0\n0:0\n", SchemaError, "duplicate"),
            ("4:4\n", SchemaError, "outside"),
            ("0:0 trailing\n", SchemaError, "no trailing"),
            ("   \n", SchemaError, "must use"),
        )
        for text, error_type, message in cases:
            with self.subTest(text=text), self.assertRaisesRegex(error_type, message):
                validate_identity_mapping_text(text, total_cores=4)

    def test_unsupported_d2d_modes_bandwidth_and_link_group_fail_closed(self) -> None:
        hardware = minimal_hardware(2, 1)
        c2c = hardware["die_ports"]["c2c"]  # type: ignore[index]
        c2c["link_bw"] = 2  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "1 C2C packet"):
            physical_fabric_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        c2c = hardware["die_ports"]["c2c"]  # type: ignore[index]
        c2c["backend"] = "behavioral"  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "functional_v2"):
            physical_fabric_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        override = hardware["die_ports"]["overrides"][0]  # type: ignore[index]
        override["link_group"] = 0  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "directed-pair"):
            physical_fabric_from_data(hardware)

    def test_implicit_fractional_or_missing_hbm_capacity_fails_closed(self) -> None:
        hardware = minimal_hardware(2, 1)
        stacks = hardware["memory_system"]["hbm_stacks"]  # type: ignore[index]
        stacks.pop()  # type: ignore[union-attr]
        with self.assertRaisesRegex(SchemaError, "every compute die"):
            physical_fabric_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        stack = hardware["memory_system"]["hbm_stacks"][0]  # type: ignore[index]
        stack["bandwidth_cap_GBps"] = 8.25  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "fractional bytes"):
            physical_fabric_from_data(hardware)

        hardware = minimal_hardware(2, 1)
        stack = hardware["memory_system"]["hbm_stacks"][0]  # type: ignore[index]
        stack["backend"] = "dramsys"  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "external memspec"):
            physical_fabric_from_data(hardware)

    def test_per_core_sram_override_is_not_silently_ignored(self) -> None:
        hardware = minimal_hardware(2, 1)
        hardware["cores"][0]["sram"] = {"capacity_bytes": 2048}  # type: ignore[index]
        with self.assertRaisesRegex(UnsupportedFeatureError, "per-core SRAM"):
            physical_fabric_from_data(hardware)


if __name__ == "__main__":
    unittest.main()
