from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.group_registry import build_group_registry
from llm.frontend.wafer_frontend.policies.swizzle.topology import (
    TopologyFlow,
    build_topology_view,
    canonical_bidirectional_pairs,
    reconstruct_resource_load,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    ExplicitGroupPlacement,
    PlacementSpec,
    PlacementStrategy,
)

from test_group_registry import context, mesh_fabric, mesh_graph


def _group(
    width: int,
    height: int,
    ranks: int,
    *,
    die_ids: tuple[int, ...] | None = None,
):
    fabric = mesh_fabric(width, height)
    placement = None
    if die_ids is not None:
        placement = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("P0", "mesh_tp", die_ids),),
        )
    group = build_group_registry(
        mesh_graph(ranks),
        context(fabric, placement),
    )[0]
    return fabric, group


class SwizzleTopologyTest(unittest.TestCase):
    def test_one_by_four_is_a_real_line_without_a_fake_cycle(self) -> None:
        fabric, group = _group(4, 1, 4)
        view = build_topology_view(group, fabric)

        self.assertTrue(view.is_complete_rectangle)
        self.assertEqual(view.physical_origin, (0, 0))
        self.assertEqual(view.physical_shape, (4, 1))
        self.assertEqual(view.row_rank_orders, ((0, 1, 2, 3),))
        self.assertEqual(view.column_rank_orders, ((0,), (1,), (2,), (3,)))
        self.assertEqual(view.snake_rank_order, (0, 1, 2, 3))
        self.assertFalse(view.has_hamiltonian_cycle)
        self.assertEqual(view.cross_group_route_refs, ())
        self.assertEqual(view.route(0, 3).die_path, (0, 1, 2, 3))

    def test_two_by_two_has_canonical_rows_columns_snake_and_cycle(self) -> None:
        fabric, group = _group(2, 2, 4)
        view = build_topology_view(group, fabric)

        self.assertEqual(view.row_rank_orders, ((0, 1), (2, 3)))
        self.assertEqual(view.column_rank_orders, ((0, 2), (1, 3)))
        self.assertEqual(view.snake_rank_order, (0, 1, 3, 2))
        self.assertEqual(view.hamiltonian_cycle_rank_order, (0, 1, 3, 2))
        cycle = view.hamiltonian_cycle_rank_order
        for source, destination in zip(cycle, cycle[1:] + cycle[:1]):
            self.assertEqual(view.route(source, destination).hop_count, 1)

    def test_two_by_four_rectangle_and_resource_incidence_are_stable(self) -> None:
        fabric, group = _group(2, 4, 8)
        view = build_topology_view(group, fabric)

        self.assertTrue(view.is_complete_rectangle)
        self.assertEqual(view.physical_shape, (2, 4))
        self.assertEqual(
            view.row_rank_orders,
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        )
        self.assertEqual(view.snake_rank_order, (0, 1, 3, 2, 4, 5, 7, 6))
        self.assertTrue(view.has_hamiltonian_cycle)
        resource_ids = tuple(item.id for item in view.resources)
        self.assertEqual(resource_ids, tuple(sorted(resource_ids)))
        self.assertTrue(all(item.route_refs for item in view.resources))

    def test_non_rectangle_and_cross_group_transit_are_explicit(self) -> None:
        fabric, group = _group(2, 2, 3, die_ids=(0, 1, 2))
        view = build_topology_view(group, fabric)

        self.assertFalse(view.is_complete_rectangle)
        self.assertEqual(view.physical_shape, (2, 2))
        self.assertTrue(view.cross_group_route_refs)
        crossing = view.route(2, 1)
        self.assertEqual(crossing.die_path, (2, 3, 1))
        self.assertTrue(crossing.leaves_group)
        self.assertFalse(view.has_hamiltonian_cycle)

    def test_missing_route_fails_closed(self) -> None:
        fabric, group = _group(2, 2, 4)
        broken = replace(
            group,
            embedding=replace(
                group.embedding,
                routes=group.embedding.routes[:-1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "every ordered rank-pair"):
            build_topology_view(broken, fabric)

    def test_candidate_specific_resource_load_uses_exact_pair_routes(self) -> None:
        fabric, group = _group(2, 2, 4)
        view = build_topology_view(group, fabric)
        loads = reconstruct_resource_load(
            view,
            (
                TopologyFlow(0, 3, 64),
                TopologyFlow(3, 0, 32),
            ),
        )
        by_resource = {item.resource_id: item.logical_bytes for item in loads}

        for resource_id in view.route(0, 3).resource_ids:
            self.assertGreaterEqual(by_resource[resource_id], 64)
        for resource_id in view.route(3, 0).resource_ids:
            self.assertGreaterEqual(by_resource[resource_id], 32)
        self.assertEqual(
            tuple(item.resource_id for item in loads),
            tuple(sorted(by_resource)),
        )

    def test_bidirectional_line_pairs_never_wrap(self) -> None:
        order = (0, 1, 2, 3)
        self.assertEqual(canonical_bidirectional_pairs(order, 0), ((0, 1), (3, 2)))
        self.assertEqual(canonical_bidirectional_pairs(order, 1), ((1, 2), (2, 1)))
        self.assertNotIn((3, 0), canonical_bidirectional_pairs(order, 0))
        self.assertEqual(canonical_bidirectional_pairs(order, 3), ())


if __name__ == "__main__":
    unittest.main()
