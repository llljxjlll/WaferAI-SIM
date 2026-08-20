from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.lite_moe_graph import (
    build_lite_moe_ir0_adapter,
)
from llm.frontend.wafer_frontend.passes.lite_moe_n4 import (
    build_lite_moe_n4,
    place_lite_moe_adapter,
    validate_lite_moe_n4,
    validate_lite_moe_placement,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.lite_moe_n4 import (
    LiteMoeN4IR1,
    LiteMoePlacedIR1,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    HbmBinding,
    PersistentStateManifest,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.test.frontend.integration.lite_moe_cases import (
    build_lite_moe_source_case,
)


def _ir1(graph: IR1, **changes: object) -> IR1:
    fields = {
        "producer_pass": graph.producer_pass,
        "source_ir0_id": graph.source_ir0_id,
        "profile": graph.profile,
        "fabric": graph.fabric,
        "instances": graph.instances,
        "groups": graph.groups,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "fused_op_skeletons": graph.fused_op_skeletons,
        "cross_routes": graph.cross_routes,
        "state_accesses": graph.state_accesses,
        "persistent_state_manifest": graph.persistent_state_manifest,
        "instance_profiles": graph.instance_profiles,
        "node_profiles": graph.node_profiles,
        "pd_plan_id": graph.pd_plan_id,
    }
    fields.update(changes)
    return IR1.create(**fields)


class LiteMoeN4Test(unittest.TestCase):
    def setUp(self) -> None:
        self.source = build_lite_moe_source_case()
        self.adapter = build_lite_moe_ir0_adapter(
            self.source.spec, self.source.moe_spec, self.source.oracle
        )
        hardware = json.loads(self.source.hardware_json)
        self.placement_context = PlacementContext.create(
            producer_pass="test_lite_moe_n4",
            fabric=physical_fabric_from_data(hardware),
            placement=self.source.spec.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(hardware),
        )
        registry = production_registry()
        self.partition_context = FusionPartitionContext.create(
            producer_pass="test_lite_moe_n4"
        )
        self.planning_context = InterDiePlanningContext.create(
            producer_pass="test_lite_moe_n4",
            fused_policy=registry.instantiate(
                RegistryKind.INTER_DIE, "naive"
            ).selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE,
                "direct_all_gather",
            ).selection,
        )
        self.placed = place_lite_moe_adapter(
            self.adapter,
            self.source.spec,
            self.source.moe_spec,
            self.source.oracle,
            self.placement_context,
        )

    def test_placement_n4_exact_deterministic_and_round_trip(self) -> None:
        group = self.placed.graph.groups[0]
        self.assertEqual(group.axis.value, "ep")
        self.assertEqual(group.logical_shape, (2,))
        self.assertEqual(
            tuple((item.rank, item.die_id) for item in group.placements),
            ((0, 0), (1, 1)),
        )
        self.assertEqual(len(group.embedding.routes), 2)
        manifest = self.placed.graph.persistent_state_manifest
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(len(manifest.declarations), 12)
        self.assertEqual(len(manifest.bindings), 12)
        self.assertEqual(
            {
                die: sum(item.size_bytes for item in manifest.bindings
                         if item.die_id == die)
                for die in (0, 1)
            },
            {0: 6144, 1: 6144},
        )
        self.assertEqual(
            {
                rank: sum(1 for item in self.placed.graph.state_accesses
                          if item.rank == rank)
                for rank in (0, 1)
            },
            {0: 12, 1: 12},
        )
        n4 = build_lite_moe_n4(
            self.placed, self.partition_context, self.planning_context
        )
        self.assertEqual(len(n4.graph.nodes), 40)
        self.assertEqual(n4.graph.fused_op_skeletons, ())
        self.assertEqual(n4.fusion_plans, ())
        self.assertEqual(n4.standalone_plans, ())
        self.assertEqual(n4.p2p_bindings, self.placed.p2p_bindings)
        second_placed = place_lite_moe_adapter(
            self.adapter,
            self.source.spec,
            self.source.moe_spec,
            self.source.oracle,
            self.placement_context,
        )
        second_n4 = build_lite_moe_n4(
            second_placed, self.partition_context, self.planning_context
        )
        self.assertEqual(canonical_digest(second_placed), canonical_digest(self.placed))
        self.assertEqual(canonical_digest(second_n4), canonical_digest(n4))
        self.assertEqual(
            loads_dataclass(
                LiteMoePlacedIR1, canonical_json(self.placed), path="placed"
            ),
            self.placed,
        )
        self.assertEqual(
            loads_dataclass(LiteMoeN4IR1, canonical_json(n4), path="n4"), n4
        )

    def test_dense_placement_stays_fail_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "requires a TP mesh axis"):
            place_ir0(self.adapter.graph, self.placement_context)

    def test_home_route_state_and_provenance_tamper_fail_closed(self) -> None:
        manifest = self.placed.graph.persistent_state_manifest
        assert manifest is not None
        binding = manifest.bindings[0]
        wrong_die = 1 - binding.die_id
        wrong_address = max(
            candidate.address + candidate.size_bytes
            for candidate in manifest.bindings
            if candidate.die_id == wrong_die
        )
        wrong_binding = HbmBinding.create(
            state_ref=binding.state_ref,
            die_id=wrong_die,
            address=wrong_address,
            size_bytes=binding.size_bytes,
        )
        wrong_manifest = PersistentStateManifest.create(
            address_spaces=manifest.address_spaces,
            declarations=manifest.declarations,
            bindings=(wrong_binding, *manifest.bindings[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "home die"):
            LiteMoePlacedIR1.create(
                **(
                    self.placed._semantic_key()
                    | {
                        "graph": _ir1(
                            self.placed.graph,
                            persistent_state_manifest=wrong_manifest,
                        )
                    }
                )
            )

        group = self.placed.graph.groups[0]
        changed_group = replace(
            group,
            embedding=replace(group.embedding, routes=group.embedding.routes[:1]),
        )
        changed_graph = _ir1(self.placed.graph, groups=(changed_group,))
        changed_placed = LiteMoePlacedIR1.create(
            **(
                self.placed._semantic_key()
                | {"graph": changed_graph}
            )
        )
        with self.assertRaisesRegex(SchemaError, "placement/route"):
            validate_lite_moe_placement(
                changed_placed, self.adapter, self.placement_context
            )

        foreign = LiteMoePlacedIR1.create(
            **(
                self.placed._semantic_key()
                | {"source_adapter_id": "foreign"}
            )
        )
        with self.assertRaisesRegex(SchemaError, "provenance"):
            validate_lite_moe_placement(
                foreign, self.adapter, self.placement_context
            )

    def test_n4_binding_and_context_tamper_fail_closed(self) -> None:
        n4 = build_lite_moe_n4(
            self.placed, self.partition_context, self.planning_context
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            LiteMoeN4IR1.create(
                **(
                    n4._semantic_key()
                    | {"p2p_bindings": n4.p2p_bindings[:-1]}
                )
            )
        foreign = LiteMoeN4IR1.create(
            **(
                n4._semantic_key()
                | {"planning_context_id": "foreign"}
            )
        )
        with self.assertRaisesRegex(SchemaError, "provenance"):
            validate_lite_moe_n4(
                foreign,
                self.placed,
                self.partition_context,
                self.planning_context,
            )
        # The exact production calls used by the bridge remain independently valid.
        partitioned = partition_ir1(self.placed.graph)
        self.assertEqual(partitioned.fused_op_skeletons, ())


if __name__ == "__main__":
    unittest.main()
