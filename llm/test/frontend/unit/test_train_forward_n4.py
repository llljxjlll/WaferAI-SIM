from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from _fixtures import (
    naive_inter_die_planning_context,
    valid_hbm_address_spaces,
    valid_spec,
)

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import load_physical_fabric
from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_train_forward,
)
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.train_forward import (
    build_train_forward_ir0,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    ExperimentSpec,
    ExplicitGroupPlacement,
    PlacementSpec,
    PlacementStrategy,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.n4 import (
    TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
    TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
    FusionPartitionContext,
    TrainFusionPartitionedIR1,
    TrainInterDiePlannedIR1,
)
from llm.frontend.wafer_frontend.schema.placed_ir1 import (
    TRAIN_PLACED_IR1_SCHEMA_VERSION,
    TrainPlacedIR1,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE_4DIE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_HARDWARE_2DIE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


def _tiny_train_spec() -> ExperimentSpec:
    raw = valid_spec()
    raw["model"].update(
        V=32,
        H=16,
        I=32,
        NH=4,
        KVH=4,
        DH=4,
        rotary_dim=4,
        L=2,
        max_position_embeddings=64,
    )
    raw["workload"] = {
        "mode": "train",
        "infer": None,
        "train": {
            "global_batch": 4,
            "micro_batch": 1,
            "seq_len": 8,
            "backward": False,
            "optimizer": "none",
            "structure": {
                "micro_batch_count": 2,
                "pp_schedule": "gpipe",
                "interleave_chunks": 1,
                "recompute": "none",
            },
        },
    }
    raw["parallel"]["instances"][0].update(
        role="train", tp=2, sp=True, dp=2, replicas=1, pp=1, ep=1
    )
    return from_data(ExperimentSpec, raw, path="spec")


def _placement_context(
    spec: ExperimentSpec,
    hardware: Path = _HARDWARE_4DIE,
    *,
    placement: PlacementSpec | None = None,
) -> PlacementContext:
    fabric = load_physical_fabric(hardware, _MAPPING)
    return PlacementContext.create(
        producer_pass="n6_2_test",
        fabric=fabric,
        placement=spec.placement if placement is None else placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )


def _pipeline():
    spec = _tiny_train_spec()
    graph = build_train_forward_ir0(spec)
    placement_context = _placement_context(spec)
    placed = place_train_forward_ir0(graph, placement_context)
    partition_context = FusionPartitionContext.create(producer_pass="n6_2_test")
    partitioned = partition_train_forward(placed, partition_context)
    planning_context = naive_inter_die_planning_context("n6_2_test")
    planned = plan_train_forward(partitioned, planning_context)
    return (
        spec,
        graph,
        placement_context,
        placed,
        partition_context,
        partitioned,
        planning_context,
        planned,
    )


class TrainForwardN4Test(unittest.TestCase):
    def test_dp2_tp2_placement_partition_and_planning_goldens(self) -> None:
        _, _, _, placed, _, partitioned, planning_context, planned = _pipeline()
        self.assertEqual(placed.schema_version, TRAIN_PLACED_IR1_SCHEMA_VERSION)
        self.assertEqual(
            partitioned.schema_version,
            TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
        )
        self.assertEqual(
            planned.schema_version,
            TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
        )
        self.assertEqual(tuple(item.replica_index for item in placed.replicas), (0, 1))
        self.assertEqual(
            tuple(
                tuple(rank.die_id for rank in item.graph.groups[0].placements)
                for item in placed.replicas
            ),
            ((0, 1), (2, 3)),
        )
        placed_bytes = tuple(
            sum(
                binding.size_bytes
                for binding in item.graph.persistent_state_manifest.bindings
            )
            for item in placed.replicas
        )
        self.assertEqual(placed_bytes, (14656, 14656))
        self.assertEqual(sum(placed_bytes), 29312)
        self.assertEqual(
            tuple(len(graph.fused_op_skeletons) for graph in partitioned.replicas),
            (4, 4),
        )
        self.assertEqual(
            tuple(
                (len(item.fusion_plans), len(item.standalone_plans))
                for item in planned.replicas
            ),
            ((4, 4), (4, 4)),
        )
        self.assertEqual(
            tuple(
                sum(
                    node.workload.group_logical_payload_bytes
                    for node in item.graph.nodes
                    if node.kind is OpKind.COLLECTIVE
                )
                for item in planned.replicas
            ),
            (2048, 2048),
        )
        self.assertTrue(
            all(
                plan.group_ref == item.graph.groups[0].id
                for item in planned.replicas
                for plan in (*item.fusion_plans, *item.standalone_plans)
            )
        )
        binding_sets = tuple(
            {binding.id for binding in item.graph.persistent_state_manifest.bindings}
            for item in placed.replicas
        )
        self.assertTrue(binding_sets[0].isdisjoint(binding_sets[1]))
        node_sets = tuple(
            {node.id for node in item.graph.nodes} for item in placed.replicas
        )
        self.assertTrue(node_sets[0].isdisjoint(node_sets[1]))
        self.assertTrue(all(node.endswith("__dp0") for node in node_sets[0]))
        self.assertTrue(all(node.endswith("__dp1") for node in node_sets[1]))
        self.assertNotEqual(
            placed.replicas[0].graph.persistent_state_manifest.id,
            placed.replicas[1].graph.persistent_state_manifest.id,
        )
        planned.validate()
        planned.validate_against(partitioned, planning_context)

    def test_deterministic_strict_round_trip(self) -> None:
        first = _pipeline()
        second = _pipeline()
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[5], second[5])
        self.assertEqual(first[7], second[7])
        for carrier_type, carrier in (
            (TrainPlacedIR1, first[3]),
            (TrainFusionPartitionedIR1, first[5]),
            (TrainInterDiePlannedIR1, first[7]),
        ):
            decoded = loads_dataclass(
                carrier_type,
                canonical_json(carrier),
                path="carrier",
            )
            self.assertEqual(decoded, carrier)
            decoded.validate()

    def test_fail_closed_for_unsupported_or_cross_replica_inputs(self) -> None:
        spec = _tiny_train_spec()
        graph = build_train_forward_ir0(spec)
        instance = graph.instances[0]
        explicit = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (
                ExplicitGroupPlacement(
                    instance_id=instance.id,
                    mesh_ref=instance.meshes[0].id,
                    die_ids=(0, 1),
                ),
            ),
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "compact row-major"):
            place_train_forward_ir0(graph, _placement_context(spec, placement=explicit))
        with self.assertRaisesRegex(SchemaError, "requires 4 dies"):
            place_train_forward_ir0(
                graph,
                _placement_context(spec, _HARDWARE_2DIE),
            )

        _, _, _, placed, _, _, _, planned = _pipeline()
        with self.assertRaisesRegex(SchemaError, "canonical 0..dp-1"):
            replace(placed, replicas=tuple(reversed(placed.replicas))).validate()
        first = planned.replicas[0]
        bad_plan = replace(
            first.fusion_plans[0],
            group_ref=planned.replicas[1].graph.groups[0].id,
        )
        bad_replica = replace(
            first,
            fusion_plans=(bad_plan, *first.fusion_plans[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "another DP replica group"):
            replace(
                planned,
                replicas=(bad_replica, planned.replicas[1]),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                planned,
                schema_version="wafer_frontend.train_interdie_planned_ir1/v0",
            ).validate()


if __name__ == "__main__":
    unittest.main()
