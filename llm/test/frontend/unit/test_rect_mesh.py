from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.policies.swizzle.rect_mesh_topology import (
    build_rect_mesh_topology,
)
from llm.frontend.wafer_frontend.schema import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware


def _rank_map(spec: RectMeshSpec) -> dict[tuple[int, int], int]:
    return {
        spec.coordinate(rank): rank
        for rank in range(spec.rank_count)
    }


class RectMeshSpecTest(unittest.TestCase):
    def test_contract_is_fixed_and_round_trips_stably(self) -> None:
        spec = RectMeshSpec(rows=3, columns=4)
        spec.validate()

        self.assertEqual(spec.rank_count, 12)
        self.assertEqual(spec.physical_shape, (4, 3))
        self.assertEqual(spec.rank(2, 3), 11)
        self.assertEqual(spec.coordinate(11), (3, 2))
        self.assertEqual(spec.directed_link_count, 34)
        self.assertEqual(spec.max_hop_count, 5)
        self.assertEqual(len(spec.ordered_rank_pairs), 132)
        self.assertTrue(spec.has_hamiltonian_cycle)
        self.assertEqual(loads_dataclass(RectMeshSpec, canonical_json(spec)), spec)
        self.assertEqual(spec.digest, RectMeshSpec(rows=3, columns=4).digest)

    def test_envelope_participants_and_placement_fail_closed(self) -> None:
        for spec in (
            RectMeshSpec(rows=0, columns=1),
            RectMeshSpec(rows=1, columns=0),
            RectMeshSpec(rows=11, columns=1),
            RectMeshSpec(rows=1, columns=11),
            RectMeshSpec(rows=True, columns=1),
            replace(RectMeshSpec(2, 2), ranks_per_die=2),
            replace(RectMeshSpec(2, 2), origin=(1, 0)),
            replace(RectMeshSpec(2, 2), timing_execution=False),
            replace(RectMeshSpec(2, 2), functional_execution=True),
        ):
            with self.subTest(spec=spec), self.assertRaises(SchemaError):
                spec.validate()

        spec = RectMeshSpec(rows=2, columns=3)
        with self.assertRaisesRegex(SchemaError, "rank count"):
            spec.validate_participant_count(5)
        with self.assertRaisesRegex(SchemaError, "hole-free"):
            spec.validate_rank_coordinates(
                ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1))
            )


class RectMeshTopologyTest(unittest.TestCase):
    def test_all_one_hundred_shapes_build_stable_physical_fabrics(self) -> None:
        for rows in range(1, 11):
            for columns in range(1, 11):
                with self.subTest(rows=rows, columns=columns):
                    spec = RectMeshSpec(rows=rows, columns=columns)
                    hardware = minimal_hardware(columns, rows)
                    fabric = physical_fabric_from_data(hardware)

                    self.assertEqual(fabric.die_grid, spec.physical_shape)
                    self.assertEqual(len(fabric.dies), spec.rank_count)
                    self.assertEqual(len(fabric.links), spec.directed_link_count)
                    self.assertEqual(
                        canonical_digest(fabric),
                        canonical_digest(physical_fabric_from_data(hardware)),
                    )

    def test_all_one_hundred_shapes_have_bounded_canonical_topology(self) -> None:
        digests: dict[tuple[int, int], str] = {}
        for rows in range(1, 11):
            for columns in range(1, 11):
                with self.subTest(rows=rows, columns=columns):
                    spec = RectMeshSpec(rows=rows, columns=columns)
                    spec.validate()
                    topology = build_rect_mesh_topology(_rank_map(spec))

                    self.assertTrue(topology.is_complete_rectangle)
                    self.assertEqual(topology.physical_shape, spec.physical_shape)
                    self.assertEqual(topology.rank_count, spec.rank_count)
                    self.assertEqual(topology.row_rank_orders, spec.row_rank_orders)
                    self.assertEqual(
                        topology.column_rank_orders, spec.column_rank_orders
                    )
                    self.assertEqual(topology.snake_rank_order, spec.snake_rank_order)
                    self.assertEqual(
                        topology.has_hamiltonian_cycle,
                        spec.has_hamiltonian_cycle,
                    )
                    cycle = topology.hamiltonian_cycle_rank_order
                    if cycle:
                        self.assertEqual(len(cycle), spec.rank_count)
                        self.assertEqual(len(set(cycle)), spec.rank_count)
                        self.assertEqual(cycle[0], 0)
                        for source, destination in zip(
                            cycle, cycle[1:] + cycle[:1]
                        ):
                            source_x, source_y = spec.coordinate(source)
                            destination_x, destination_y = spec.coordinate(destination)
                            self.assertEqual(
                                abs(source_x - destination_x)
                                + abs(source_y - destination_y),
                                1,
                            )
                    digests[(rows, columns)] = spec.digest

        self.assertEqual(len(digests), 100)
        self.assertEqual(
            digests,
            {
                (rows, columns): RectMeshSpec(rows, columns).digest
                for rows in range(1, 11)
                for columns in range(1, 11)
            },
        )

    def test_non_contiguous_selection_has_no_rectangle_or_cycle(self) -> None:
        topology = build_rect_mesh_topology({(0, 0): 0, (2, 0): 1})

        self.assertEqual(topology.physical_shape, (3, 1))
        self.assertFalse(topology.is_complete_rectangle)
        self.assertFalse(topology.has_hamiltonian_cycle)

    def test_transposed_even_rectangle_and_nonzero_origin_are_real_cycles(self) -> None:
        rank_by_coord = {
            (3 + column, 5 + row): row * 4 + column
            for row in range(3)
            for column in range(4)
        }
        topology = build_rect_mesh_topology(rank_by_coord)

        self.assertEqual(topology.physical_origin, (3, 5))
        self.assertEqual(topology.physical_shape, (4, 3))
        self.assertEqual(topology.hamiltonian_cycle_rank_order[0], 0)
        inverse = {rank: coord for coord, rank in rank_by_coord.items()}
        cycle = topology.hamiltonian_cycle_rank_order
        for source, destination in zip(cycle, cycle[1:] + cycle[:1]):
            source_x, source_y = inverse[source]
            destination_x, destination_y = inverse[destination]
            self.assertEqual(
                abs(source_x - destination_x) + abs(source_y - destination_y),
                1,
            )


if __name__ == "__main__":
    unittest.main()
