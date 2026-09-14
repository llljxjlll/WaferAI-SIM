from __future__ import annotations

import json
import unittest

from physical_model import (
    STACK_CAPACITY_BYTES,
    WAFER_DIE_COUNT,
    aggregate_directed_edge_loads,
    allocate_expert_weights,
    balanced_expert_home_ranks,
    build_physical_model,
    build_remote_moe_flows,
    coordinate_to_die_id,
    default_hbm_stacks,
    die_id_to_coordinate,
    placement_groups,
    x_first_directed_edges,
)


def assignment_matrix(expert_count: int, value: int = 1) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(value for _ in range(expert_count)) for _ in range(4))


class PhysicalModelTest(unittest.TestCase):
    def test_row_major_coordinates_and_x_first_route(self) -> None:
        for die_id in range(WAFER_DIE_COUNT):
            coordinate = die_id_to_coordinate(die_id)
            self.assertEqual(coordinate_to_die_id(coordinate.x, coordinate.y), die_id)

        source = coordinate_to_die_id(1, 1)
        destination = coordinate_to_die_id(4, 4)
        route = x_first_directed_edges(source, destination)
        self.assertEqual(len(route), 6)
        self.assertEqual(
            [(edge.source_die, edge.destination_die) for edge in route],
            [(7, 8), (8, 9), (9, 10), (10, 16), (16, 22), (22, 28)],
        )
        reverse = x_first_directed_edges(destination, source)
        self.assertEqual(
            [(edge.source_die, edge.destination_die) for edge in reverse],
            [(28, 27), (27, 26), (26, 25), (25, 19), (19, 13), (13, 7)],
        )

    def test_isolated_and_loaded_placements_are_explicit(self) -> None:
        for placement in ("compact", "noncompact"):
            isolated = placement_groups(placement, "isolated_group")
            loaded = placement_groups(placement, "loaded_groups")
            self.assertEqual(len(isolated), 1)
            self.assertEqual(len(loaded), 9)
            self.assertIn(isolated[0], loaded)
            occupied = [die for group in loaded for die in group.rank_to_die]
            self.assertEqual(len(occupied), WAFER_DIE_COUNT)
            self.assertEqual(set(occupied), set(range(WAFER_DIE_COUNT)))
            self.assertEqual(len(set(group.group_id for group in loaded)), 9)

        self.assertEqual(
            placement_groups("compact", "isolated_group")[0].rank_to_die,
            (14, 15, 20, 21),
        )
        self.assertEqual(
            placement_groups("noncompact", "isolated_group")[0].rank_to_die,
            (7, 10, 25, 28),
        )

    def test_remote_assignment_and_edge_load_conservation(self) -> None:
        expert_count = 8
        homes = balanced_expert_home_ranks(expert_count)
        assignments = tuple(
            tuple(source + expert + 1 for expert in range(expert_count))
            for source in range(4)
        )
        groups = placement_groups("noncompact", "isolated_group")
        flows = build_remote_moe_flows(
            groups,
            assignments,
            homes,
            hidden_size=16,
            dtype_bytes=2,
            metadata_bytes_per_assignment=4,
        )
        expected_remote = sum(
            assignments[source][expert]
            for source in range(4)
            for expert, home in enumerate(homes)
            if source != home
        )
        self.assertEqual(sum(flow.remote_assignments for flow in flows), expected_remote)
        self.assertTrue(all(flow.source_rank != flow.destination_rank for flow in flows))
        self.assertTrue(all(flow.route for flow in flows))
        self.assertEqual(
            sum(flow.payload_bytes for flow in flows),
            expected_remote * 16 * 2,
        )

        loads = aggregate_directed_edge_loads(flows)
        self.assertEqual(
            sum(load.total_bytes for load in loads),
            sum(flow.total_bytes * len(flow.route) for flow in flows),
        )
        by_edge = {load.edge: load for load in loads}
        for edge, load in by_edge.items():
            expected = sum(flow.total_bytes for flow in flows if edge in flow.route)
            self.assertEqual(load.total_bytes, expected)

    def test_four_stack_addresses_and_capacity_audit(self) -> None:
        stacks = default_hbm_stacks()
        self.assertEqual([stack.home_die for stack in stacks], [1, 4, 31, 34])
        self.assertEqual(
            [stack.address_base for stack in stacks],
            [0, STACK_CAPACITY_BYTES, 2 * STACK_CAPACITY_BYTES, 3 * STACK_CAPACITY_BYTES],
        )
        self.assertEqual(
            [stack.address_end for stack in stacks],
            [STACK_CAPACITY_BYTES, 2 * STACK_CAPACITY_BYTES,
             3 * STACK_CAPACITY_BYTES, 4 * STACK_CAPACITY_BYTES],
        )

        homes = balanced_expert_home_ranks(256)
        allocation = allocate_expert_weights(7168, 2048, homes)
        self.assertTrue(allocation.feasible)
        self.assertEqual(allocation.total_weight_bytes, 21 * 1024**3)
        self.assertEqual(
            [audit.payload_bytes for audit in allocation.stack_audits],
            [int(5.25 * 1024**3)] * 4,
        )
        for binding in allocation.bindings:
            stack = stacks[binding.stack_id]
            self.assertGreaterEqual(binding.address, stack.address_base)
            self.assertLessEqual(binding.address + binding.size_bytes, stack.address_end)
        ranges_by_stack: dict[int, list[tuple[int, int]]] = {}
        for binding in allocation.bindings:
            ranges = ranges_by_stack.setdefault(binding.stack_id, [])
            interval = (binding.address, binding.address + binding.size_bytes)
            self.assertFalse(
                any(interval[0] < end and start < interval[1] for start, end in ranges)
            )
            ranges.append(interval)

        infeasible = allocate_expert_weights(
            131072,
            131072,
            balanced_expert_home_ranks(4),
            strict_capacity=False,
        )
        self.assertFalse(infeasible.feasible)
        self.assertTrue(all(audit.overflow_bytes > 0 for audit in infeasible.stack_audits))

    def test_loaded_contention_is_derived_from_background_flows(self) -> None:
        expert_count = 8
        homes = balanced_expert_home_ranks(expert_count)
        loaded_groups = placement_groups("noncompact", "loaded_groups")
        center = placement_groups("noncompact", "isolated_group")[0]
        active = assignment_matrix(expert_count, 1)
        zero = assignment_matrix(expert_count, 0)

        center_only_assignments = {
            group.group_id: (active if group == center else zero)
            for group in loaded_groups
        }
        center_only = build_physical_model(
            placement="noncompact",
            network_scenario="loaded_groups",
            assignments=center_only_assignments,
            expert_home_ranks=homes,
            hidden_size=4096,
            intermediate_size=14336,
        )
        isolated = build_physical_model(
            placement="noncompact",
            network_scenario="isolated_group",
            assignments=active,
            expert_home_ranks=homes,
            hidden_size=4096,
            intermediate_size=14336,
        )
        self.assertEqual(center_only.directed_edge_loads, isolated.directed_edge_loads)

        all_active = build_physical_model(
            placement="noncompact",
            network_scenario="loaded_groups",
            assignments=active,
            expert_home_ranks=homes,
            hidden_size=4096,
            intermediate_size=14336,
        )
        self.assertGreater(
            max(load.total_bytes for load in all_active.directed_edge_loads),
            max(load.total_bytes for load in center_only.directed_edge_loads),
        )
        manifest = all_active.manifest_dict()
        encoded = json.dumps(manifest, sort_keys=True)
        self.assertEqual(
            manifest["contention_model"],
            "derived_from_explicit_group_flows_and_directed_edge_incidence",
        )
        self.assertNotIn("contention_factor", encoded)
        self.assertFalse(manifest["simulator_unit_closure"])
        self.assertEqual(
            manifest["conservation"]["remote_assignments"],
            manifest["conservation"]["remote_flow_assignments"],
        )


if __name__ == "__main__":
    unittest.main()
