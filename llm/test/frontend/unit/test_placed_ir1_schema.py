from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.schema import (
    PLACED_IR1_BUNDLE_SCHEMA_VERSION,
    PlacementContext,
    PlacementSpec,
    PlacementStrategy,
    PlacedIR1Bundle,
    PlacedProfileIR1,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import FusionImpl, IR0
from llm.frontend.wafer_frontend.schema.ir1 import (
    FusedOpSkeleton,
    IR1,
    PhysicalGroup,
    PhysicalInstance,
    PhysicalNode,
)
from llm.frontend.wafer_frontend.schema.logical import (
    ExpandedIR0Bundle,
    ExpandedProfileIR0,
    IR0Template,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec


def expanded_fixture() -> tuple[IR0Template, ExpandedIR0Bundle]:
    raw = valid_spec()
    first = raw["workload"]["infer"]["profile"]  # type: ignore[index]
    second = dict(first)  # type: ignore[arg-type]
    second.update(
        prefill_tokens=64,
        context_sum=64,
        context_max=64,
        kv_pages=4,
    )
    raw["workload"]["infer"] = {  # type: ignore[index]
        "source": "shape_dist",
        "output": "logits",
        "profile": None,
        "shape_dist": {
            "profiles": [
                {"key": first, "weight": 0.25},
                {"key": second, "weight": 0.75},
            ]
        },
    }
    spec = from_data(ExperimentSpec, raw, path="spec")
    template = build_ir0(spec)
    bundle = logical_expand(template)
    bundle.validate()
    return template, bundle


def placement_context(
    strategy: PlacementStrategy = PlacementStrategy.COMPACT,
) -> PlacementContext:
    if strategy is PlacementStrategy.COMPACT:
        placement = PlacementSpec(strategy, ())
    else:
        from llm.frontend.wafer_frontend.schema import ExplicitGroupPlacement

        placement = PlacementSpec(
            strategy,
            (ExplicitGroupPlacement("P0", "mesh_tp", (0, 1)),),
        )
    fabric = valid_ir1().fabric
    result = PlacementContext.create(
        producer_pass="load_fabric",
        fabric=fabric,
        placement=placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    result.validate()
    return result


def physical_graph(source: IR0, context: PlacementContext) -> IR1:
    return place_ir0(source, context)


def rebuild_ir1(graph: IR1, **updates: object) -> IR1:
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
    }
    fields.update(updates)
    result = IR1.create(**fields)  # type: ignore[arg-type]
    result.validate()
    return result


def placed_fixture() -> tuple[ExpandedIR0Bundle, PlacementContext, PlacedIR1Bundle]:
    _template, source = expanded_fixture()
    context = placement_context()
    entries = tuple(
        PlacedProfileIR1.create(
            source_expanded_entry_id=entry.id,
            weight=entry.weight,
            graph=physical_graph(entry.graph, context),
        )
        for entry in source.entries
    )
    result = PlacedIR1Bundle.create(
        source_expanded_bundle=source,
        placement_context=context,
        entries=entries,
    )
    result.validate_against(source, context)
    return source, context, result


def rebuild_bundle(
    source: ExpandedIR0Bundle,
    context: PlacementContext,
    entries: tuple[PlacedProfileIR1, ...],
) -> PlacedIR1Bundle:
    return PlacedIR1Bundle.create(
        source_expanded_bundle=source,
        placement_context=context,
        entries=entries,
    )


class PlacedIR1SchemaTest(unittest.TestCase):
    def test_round_trip_stable_id_and_exact_provenance_pass(self) -> None:
        source, context, bundle = placed_fixture()
        self.assertEqual(
            bundle.schema_version,
            "wafer_frontend.placed_ir1_bundle/v1alpha5",
        )
        self.assertEqual(bundle.schema_version, PLACED_IR1_BUNDLE_SCHEMA_VERSION)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                bundle,
                schema_version="wafer_frontend.placed_ir1_bundle/v1alpha2",
            ).validate()
        decoded = loads_dataclass(
            PlacedIR1Bundle,
            canonical_json(bundle),
            path="placed_ir1_bundle",
        )
        decoded.validate_against(source, context)
        self.assertEqual(decoded, bundle)
        self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.id = "changed"  # type: ignore[misc]

    def test_source_bundle_context_and_entry_impersonation_fail(self) -> None:
        source, context, bundle = placed_fixture()
        explicit_context = placement_context(PlacementStrategy.EXPLICIT)
        wrong_context_bundle = rebuild_bundle(
            source,
            explicit_context,
            bundle.entries,
        )
        with self.assertRaisesRegex(SchemaError, "placement context"):
            wrong_context_bundle.validate_against(source, context)

        wrong_entry = PlacedProfileIR1.create(
            source_expanded_entry_id="expanded_entry_impostor",
            weight=bundle.entries[0].weight,
            graph=bundle.entries[0].graph,
        )
        wrong_entry_bundle = rebuild_bundle(
            source,
            context,
            (wrong_entry, bundle.entries[1]),
        )
        with self.assertRaisesRegex(SchemaError, "corresponding expanded entry"):
            wrong_entry_bundle.validate_against(source, context)

        template, _ = expanded_fixture()
        logical = source.entries[0].graph
        changed_node = replace(logical.nodes[0], impl_ref="other_impl")
        changed_graph = IR0.create(
            producer_pass="logical_expand",
            job=logical.job,
            instances=logical.instances,
            nodes=(changed_node,) + logical.nodes[1:],
            values=logical.values,
            edges=logical.edges,
            fusion_candidates=logical.fusion_candidates,
            profile=logical.profile,
            train=logical.train,
        )
        changed_entry = ExpandedProfileIR0.create(
            source_template_id=template.id,
            weight=source.entries[0].weight,
            graph=changed_graph,
        )
        wrong_source = ExpandedIR0Bundle.create(
            source_template=template,
            entries=(changed_entry, source.entries[1]),
        )
        wrong_source.validate()
        wrong_source_bundle = rebuild_bundle(wrong_source, context, bundle.entries)
        with self.assertRaisesRegex(SchemaError, "expanded bundle"):
            wrong_source_bundle.validate_against(source, context)

    def test_missing_duplicate_and_reordered_entries_fail_self_validation(self) -> None:
        source, context, bundle = placed_fixture()
        cases = (
            (bundle.entries[:1], "exactly one"),
            ((bundle.entries[0], bundle.entries[0]), "source profile"),
            (tuple(reversed(bundle.entries)), "source profile"),
        )
        for entries, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                rebuild_bundle(source, context, entries).validate()

    def test_weight_and_profile_mismatch_fail_self_validation(self) -> None:
        source, context, bundle = placed_fixture()
        wrong_weight = PlacedProfileIR1.create(
            source_expanded_entry_id=source.entries[0].id,
            weight=bundle.entries[0].weight + 0.125,
            graph=bundle.entries[0].graph,
        )
        with self.assertRaisesRegex(SchemaError, "source profile weight"):
            rebuild_bundle(
                source,
                context,
                (wrong_weight, bundle.entries[1]),
            ).validate()

        profile_graph = rebuild_ir1(
            bundle.entries[0].graph,
            profile=bundle.entries[1].graph.profile,
        )
        wrong_profile = PlacedProfileIR1.create(
            source_expanded_entry_id=source.entries[0].id,
            weight=bundle.entries[0].weight,
            graph=profile_graph,
        )
        with self.assertRaisesRegex(SchemaError, "source profile"):
            rebuild_bundle(
                source,
                context,
                (wrong_profile, bundle.entries[1]),
            ).validate()

    def test_context_fabric_and_cross_profile_group_identity_are_exact(self) -> None:
        source, context, bundle = placed_fixture()
        changed_die = replace(
            context.fabric.dies[0],
            hbm_bytes_per_cycle=context.fabric.dies[0].hbm_bytes_per_cycle + 1,
        )
        changed_fabric = replace(
            context.fabric,
            dies=(changed_die,) + context.fabric.dies[1:],
        )
        changed_entries = tuple(
            PlacedProfileIR1.create(
                source_expanded_entry_id=source_entry.id,
                weight=source_entry.weight,
                graph=rebuild_ir1(entry.graph, fabric=changed_fabric),
            )
            for source_entry, entry in zip(source.entries, bundle.entries)
        )
        wrong_fabric = rebuild_bundle(source, context, changed_entries)
        wrong_fabric.validate()
        with self.assertRaisesRegex(SchemaError, "context fabric"):
            wrong_fabric.validate_against(source, context)

        changed_group = replace(
            bundle.entries[1].graph.groups[0],
            id="group_tp_other",
        )
        changed_instance = replace(
            bundle.entries[1].graph.instances[0],
            group_ids=(changed_group.id,),
        )
        changed_nodes = tuple(
            replace(node, execution_group_ref=changed_group.id)
            for node in bundle.entries[1].graph.nodes
        )
        changed_graph = rebuild_ir1(
            bundle.entries[1].graph,
            groups=(changed_group,),
            instances=(changed_instance,),
            nodes=changed_nodes,
        )
        changed_entry = PlacedProfileIR1.create(
            source_expanded_entry_id=source.entries[1].id,
            weight=source.entries[1].weight,
            graph=changed_graph,
        )
        with self.assertRaisesRegex(SchemaError, "identical groups"):
            rebuild_bundle(
                source,
                context,
                (bundle.entries[0], changed_entry),
            ).validate()

    def test_logical_fields_values_edges_and_candidates_are_exact(self) -> None:
        source, context, bundle = placed_fixture()
        graph = bundle.entries[0].graph
        mutations = {
            "impl_ref": rebuild_ir1(
                graph,
                nodes=(replace(graph.nodes[0], impl_ref="other_impl"),)
                + graph.nodes[1:],
            ),
            "values": rebuild_ir1(
                graph,
                values=(replace(graph.values[0], logical_layout="other_layout"),)
                + graph.values[1:],
            ),
            "edges": rebuild_ir1(
                graph,
                edges=(replace(graph.edges[0], id="other_edge"),)
                + graph.edges[1:],
            ),
            "fusion_candidates": rebuild_ir1(
                graph,
                fusion_candidates=(
                    replace(
                        graph.fusion_candidates[0],
                        id="other_candidate",
                    ),
                )
                + graph.fusion_candidates[1:],
            ),
        }
        for name, changed_graph in mutations.items():
            changed_entry = PlacedProfileIR1.create(
                source_expanded_entry_id=source.entries[0].id,
                weight=source.entries[0].weight,
                graph=changed_graph,
            )
            candidate = rebuild_bundle(
                source,
                context,
                (changed_entry, bundle.entries[1]),
            )
            candidate.validate()
            with self.subTest(name=name), self.assertRaises(SchemaError):
                candidate.validate_against(source, context)

    def test_physical_instance_ownership_and_early_skeleton_fail(self) -> None:
        source, context, bundle = placed_fixture()
        graph = bundle.entries[0].graph
        changed_instance = replace(
            graph.instances[0],
            origin_instance_id="P0_impostor",
        )
        ownership_graph = rebuild_ir1(graph, instances=(changed_instance,))
        ownership_entry = PlacedProfileIR1.create(
            source_expanded_entry_id=source.entries[0].id,
            weight=source.entries[0].weight,
            graph=ownership_graph,
        )
        ownership_bundle = rebuild_bundle(
            source,
            context,
            (ownership_entry, bundle.entries[1]),
        )
        with self.assertRaisesRegex(SchemaError, "logical instance id"):
            ownership_bundle.validate_against(source, context)

        fusion = graph.fusion_candidates[0]
        skeleton = FusedOpSkeleton(
            id="early_skeleton",
            fusion_ref=fusion.id,
            instance_id=graph.instances[0].id,
            member_node_ids=fusion.members,
            boundary_inputs=fusion.boundary_inputs,
            boundary_outputs=fusion.boundary_outputs,
            semantic_contract=fusion.semantic_contract,
            impl=FusionImpl.NAIVE,
        )
        skeleton_graph = rebuild_ir1(graph, fused_op_skeletons=(skeleton,))
        skeleton_entry = PlacedProfileIR1.create(
            source_expanded_entry_id=source.entries[0].id,
            weight=source.entries[0].weight,
            graph=skeleton_graph,
        )
        with self.assertRaisesRegex(SchemaError, "before fusion_partition"):
            skeleton_entry.validate()


if __name__ == "__main__":
    unittest.main()
