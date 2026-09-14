from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema import (
    LogicalRankCoordinate,
    ParallelGroupKind,
    ParallelWorkloadKind,
    ParameterOwnershipKind,
    RectMeshSpec,
    build_dense_parallel_placement,
    build_moe_parallel_placement,
)
from llm.frontend.wafer_frontend.schema.parallel_placement import (
    ParallelRankPlacement,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)


class DenseParallelPlacementTest(unittest.TestCase):
    def test_fixed_logical_workload_uses_active_subset_across_mesh_shapes(self) -> None:
        shapes = ((1, 4), (2, 2), (2, 3), (3, 2), (3, 3), (10, 10))
        for rows, columns in shapes:
            with self.subTest(rows=rows, columns=columns):
                plan = build_dense_parallel_placement(
                    RectMeshSpec(rows, columns),
                    tp_degree=2,
                    dp_degree=2,
                )

                self.assertEqual(plan.logical_rank_count, 4)
                self.assertEqual(plan.active_die_ids, (0, 1, 2, 3))
                self.assertEqual(plan.routing_die_ids, tuple(range(rows * columns)))
                self.assertEqual(
                    plan.idle_die_ids,
                    tuple(range(4, rows * columns)),
                )
                self.assertEqual(
                    tuple(item.coordinate for item in plan.rank_placements),
                    (
                        LogicalRankCoordinate(dp=0, ep=0, tp=0),
                        LogicalRankCoordinate(dp=0, ep=0, tp=1),
                        LogicalRankCoordinate(dp=1, ep=0, tp=0),
                        LogicalRankCoordinate(dp=1, ep=0, tp=1),
                    ),
                )
                plan.validate()

    def test_non_contiguous_dies_core_binding_groups_and_owners_are_exact(self) -> None:
        plan = build_dense_parallel_placement(
            RectMeshSpec(3, 3),
            tp_degree=2,
            dp_degree=2,
            active_die_ids=(8, 2, 6, 0),
            local_core_ids=(3, 2, 1, 0),
        )

        self.assertEqual(plan.active_die_ids, (8, 2, 6, 0))
        self.assertEqual(plan.idle_die_ids, (1, 3, 4, 5, 7))
        self.assertEqual(
            tuple(item.local_core for item in plan.rank_placements),
            (3, 2, 1, 0),
        )
        self.assertEqual(
            tuple(group.ranks for group in plan.select_groups(ParallelGroupKind.TP)),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(
            tuple(group.ranks for group in plan.select_groups(ParallelGroupKind.DP)),
            ((0, 2), (1, 3)),
        )
        shared_groups = plan.select_groups(
            ParallelGroupKind.SHARED_PARAMETER_SYNC
        )
        self.assertEqual(
            tuple(group.ranks for group in shared_groups),
            ((0, 2), (1, 3)),
        )
        self.assertEqual(len(plan.ownership_domains), 2)
        self.assertTrue(
            all(
                owner.kind is ParameterOwnershipKind.SHARED
                for owner in plan.ownership_domains
            )
        )
        self.assertEqual(plan.ownership_domains[0].replica_ranks, (0, 2))
        self.assertEqual(plan.ownership_domains[0].primary_rank, 0)
        self.assertEqual(plan.placement_for_rank(2).die_id, 6)

    def test_round_trip_and_digest_are_stable(self) -> None:
        plan = build_dense_parallel_placement(
            RectMeshSpec(2, 3),
            tp_degree=3,
            dp_degree=1,
            active_die_ids=(5, 1, 3),
            local_core_ids=(2, 2, 2),
        )
        decoded = loads_dataclass(type(plan), canonical_json(plan))

        self.assertEqual(decoded, plan)
        self.assertEqual(decoded.digest, plan.digest)


class MoeParallelPlacementTest(unittest.TestCase):
    def test_tp_ep_dp_mapping_and_expert_replica_ownership_are_exact(self) -> None:
        plan = build_moe_parallel_placement(
            RectMeshSpec(3, 3),
            tp_degree=2,
            ep_degree=2,
            dp_degree=2,
            num_experts=4,
        )

        self.assertIs(plan.workload_kind, ParallelWorkloadKind.MOE)
        self.assertEqual(plan.logical_rank_count, 8)
        self.assertEqual(plan.idle_die_ids, (8,))
        self.assertEqual(
            tuple(group.ranks for group in plan.select_groups(ParallelGroupKind.TP)),
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        )
        self.assertEqual(
            tuple(group.ranks for group in plan.select_groups(ParallelGroupKind.EP)),
            ((0, 2), (1, 3), (4, 6), (5, 7)),
        )
        self.assertEqual(
            tuple(group.ranks for group in plan.select_groups(ParallelGroupKind.DP)),
            ((0, 4), (1, 5), (2, 6), (3, 7)),
        )
        self.assertEqual(
            tuple(
                group.ranks
                for group in plan.select_groups(
                    ParallelGroupKind.EXPERT_GRADIENT_SYNC
                )
            ),
            ((0, 4), (1, 5), (2, 6), (3, 7)),
        )

        expert_owners = tuple(
            owner
            for owner in plan.ownership_domains
            if owner.kind is ParameterOwnershipKind.EXPERT
        )
        self.assertEqual(len(expert_owners), 8)
        owner_by_key = {
            (owner.expert_id, owner.tp_shard): owner for owner in expert_owners
        }
        self.assertEqual(owner_by_key[(0, 0)].replica_ranks, (0, 4))
        self.assertEqual(owner_by_key[(1, 1)].replica_ranks, (1, 5))
        self.assertEqual(owner_by_key[(2, 0)].replica_ranks, (2, 6))
        self.assertEqual(owner_by_key[(3, 1)].replica_ranks, (3, 7))
        self.assertNotEqual(
            owner_by_key[(0, 0)].id,
            owner_by_key[(1, 0)].id,
        )

    def test_more_than_one_expert_per_ep_partition_is_supported(self) -> None:
        plan = build_moe_parallel_placement(
            RectMeshSpec(2, 3),
            tp_degree=1,
            ep_degree=3,
            dp_degree=2,
            num_experts=6,
        )
        owner_by_expert = {
            owner.expert_id: owner.replica_ranks
            for owner in plan.ownership_domains
            if owner.kind is ParameterOwnershipKind.EXPERT
        }
        self.assertEqual(
            owner_by_expert,
            {
                0: (0, 3),
                1: (0, 3),
                2: (1, 4),
                3: (1, 4),
                4: (2, 5),
                5: (2, 5),
            },
        )


class ParallelPlacementValidationTest(unittest.TestCase):
    def test_builders_reject_capacity_parallel_and_expert_errors(self) -> None:
        cases = (
            lambda: build_dense_parallel_placement(
                RectMeshSpec(1, 3), tp_degree=2, dp_degree=2
            ),
            lambda: build_dense_parallel_placement(
                RectMeshSpec(2, 2),
                tp_degree=2,
                dp_degree=2,
                pp_degree=2,
            ),
            lambda: build_dense_parallel_placement(
                RectMeshSpec(2, 2),
                tp_degree=2,
                dp_degree=2,
                active_die_ids=(0, 1, 1, 3),
            ),
            lambda: build_moe_parallel_placement(
                RectMeshSpec(2, 2),
                tp_degree=1,
                ep_degree=2,
                dp_degree=2,
                num_experts=3,
            ),
        )
        for build in cases:
            with self.subTest(build=build), self.assertRaises(SchemaError):
                build()

    def test_validation_rejects_missing_rank_group_and_bad_owner(self) -> None:
        plan = build_dense_parallel_placement(
            RectMeshSpec(2, 2), tp_degree=2, dp_degree=2
        )
        mutations = (
            replace(plan, rank_placements=plan.rank_placements[:-1]),
            replace(
                plan,
                rank_placements=(
                    plan.rank_placements[0],
                    replace(plan.rank_placements[1], die_id=0),
                    *plan.rank_placements[2:],
                ),
            ),
            replace(plan, groups=plan.groups[:-1]),
            replace(
                plan,
                ownership_domains=(
                    replace(
                        plan.ownership_domains[0],
                        synchronization_group_id="missing_group",
                    ),
                    *plan.ownership_domains[1:],
                ),
            ),
            replace(
                plan,
                rank_placements=(
                    replace(
                        plan.rank_placements[0],
                        coordinate=LogicalRankCoordinate(dp=9, ep=0, tp=0),
                    ),
                    *plan.rank_placements[1:],
                ),
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(SchemaError):
                mutation.validate()

    def test_rank_placement_rejects_negative_local_core(self) -> None:
        placement = ParallelRankPlacement(
            logical_rank=0,
            coordinate=LogicalRankCoordinate(dp=0, ep=0, tp=0),
            die_id=0,
            local_core=-1,
        )
        with self.assertRaises(SchemaError):
            placement.validate()


if __name__ == "__main__":
    unittest.main()
