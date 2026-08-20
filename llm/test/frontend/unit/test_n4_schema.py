from __future__ import annotations

import dataclasses
import json
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.placement import place_bundle
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import FusionImpl
from llm.frontend.wafer_frontend.schema.ir1 import FusedOpSkeleton, IR1
from llm.frontend.wafer_frontend.schema.logical import ProfileEntry
from llm.frontend.wafer_frontend.schema.n4 import (
    FUSED_OP_SKELETON_SCHEMA_VERSION,
    FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
    FUSION_PARTITION_CONTEXT_SCHEMA_VERSION,
    INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
    INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION,
    FusedInterDieContract,
    FusionPartitionContext,
    FusionPartitionContract,
    FusionPartitionedIR1Bundle,
    FusionPartitionedProfileIR1,
    InterDiePlanBundle,
    InterDiePlannedProfile,
    InterDiePlanningContext,
    StandaloneInterDieContract,
)
from llm.frontend.wafer_frontend.schema.placed_ir1 import PlacedProfileIR1
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import (
    naive_inter_die_planning_context,
    valid_hbm_address_spaces,
    valid_ir1,
    valid_spec,
)
from test_action_schema import (
    bound_standalone_plan,
    valid_ir1 as action_valid_ir1,
    valid_plan,
)


def _two_profile_spec() -> ExperimentSpec:
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
            "profiles": (
                {"key": first, "weight": 0.25},
                {"key": second, "weight": 0.75},
            )
        },
    }
    return from_data(ExperimentSpec, raw, path="spec")


def _placed_bundle():
    spec = _two_profile_spec()
    expanded = logical_expand(build_ir0(spec))
    fabric = valid_ir1().fabric
    placement_context = PlacementContext.create(
        producer_pass="n4_schema_fixture",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    return place_bundle(expanded, placement_context)


def _skeleton(candidate, graph: IR1) -> FusedOpSkeleton:
    semantic_key = {
        "fusion_ref": candidate.id,
        "member_node_ids": candidate.members,
        "boundary_inputs": candidate.boundary_inputs,
        "boundary_outputs": candidate.boundary_outputs,
        "semantic_contract": candidate.semantic_contract,
        "impl": FusionImpl.NONE,
    }
    node_index = {node.id: node for node in graph.nodes}
    return FusedOpSkeleton(
        id=stable_artifact_id(
            "fused_op_skeleton",
            semantic_key,
            schema_version=FUSED_OP_SKELETON_SCHEMA_VERSION,
        ),
        fusion_ref=candidate.id,
        instance_id=node_index[candidate.members[0]].instance_id,
        member_node_ids=candidate.members,
        boundary_inputs=candidate.boundary_inputs,
        boundary_outputs=candidate.boundary_outputs,
        semantic_contract=candidate.semantic_contract,
        impl=FusionImpl.NONE,
    )


def _rebuild_ir1(graph: IR1, **updates: object) -> IR1:
    producer_pass = updates.pop("producer_pass", graph.producer_pass)
    fields = graph._semantic_key()
    fields.update(updates)
    result = IR1.create(
        producer_pass=producer_pass,  # type: ignore[arg-type]
        **fields,
    )
    result.validate()
    return result


def _partition_graph(graph: IR1) -> IR1:
    skeletons = tuple(_skeleton(candidate, graph) for candidate in graph.fusion_candidates)
    return _rebuild_ir1(
        graph,
        producer_pass="fusion_partition",
        fused_op_skeletons=skeletons,
    )


def _partition_fixture():
    placed = _placed_bundle()
    context = FusionPartitionContext.create(producer_pass="n4_schema_fixture")
    entries = tuple(
        FusionPartitionedProfileIR1.create(
            source=entry,
            context=context,
            graph=_partition_graph(entry.graph),
        )
        for entry in placed.entries
    )
    bundle = FusionPartitionedIR1Bundle.create(
        source=placed,
        context=context,
        entries=entries,
    )
    bundle.validate_against(placed, context)
    return placed, context, bundle


def _restable_partition_entry(
    entry: FusionPartitionedProfileIR1,
    **updates: object,
) -> FusionPartitionedProfileIR1:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "fusion_partitioned_profile_ir1",
            changed._semantic_key(),
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _single_profile_partition(base_graph: IR1):
    placed_graph = _rebuild_ir1(
        base_graph,
        producer_pass="placement",
        fused_op_skeletons=(),
        cross_routes=(),
    )
    placed_entry = PlacedProfileIR1.create(
        source_expanded_entry_id="expanded_entry_fixture",
        weight=1.0,
        graph=placed_graph,
    )
    placed_entry.validate()
    context = FusionPartitionContext.create(producer_pass="n4_schema_fixture")
    entry = FusionPartitionedProfileIR1.create(
        source=placed_entry,
        context=context,
        graph=_partition_graph(placed_graph),
    )
    entry.validate_against(placed_entry, context)
    source_profiles = (
        ProfileEntry.create(key=entry.graph.profile, weight=1.0),
    )
    semantic_key = {
        "source_placed_bundle_id": "placed_bundle_fixture",
        "placement_context_id": "placement_context_fixture",
        "partition_context_id": context.id,
        "source_profiles": source_profiles,
        "entries": (entry,),
    }
    bundle = FusionPartitionedIR1Bundle(
        schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        producer_pass="fusion_partition",
        id=stable_artifact_id(
            "fusion_partitioned_ir1_bundle",
            semantic_key,
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        ),
        **semantic_key,
    )
    bundle.validate()
    return context, bundle


def _rebind_fusion_plan(graph: IR1):
    template = valid_plan()
    fields = template._semantic_key()
    fields.update(
        source_ir1_id=graph.id,
        fused_op_id=graph.fused_op_skeletons[0].id,
        profile_key=graph.profile,
    )
    return type(template).create(producer_pass="inter_die_plan", **fields)


def _rebind_standalone_plan(graph: IR1, template):
    fields = template._semantic_key()
    fields.update(
        source_ir1_id=graph.id,
        op_id=graph.nodes[0].id,
        profile_key=graph.profile,
    )
    return type(template).create(producer_pass="inter_die_plan", **fields)


def _planning_fixture(*, standalone: bool = False):
    if standalone:
        base_graph, template = bound_standalone_plan()
        _partition_context, partitioned = _single_profile_partition(base_graph)
        fusion_plans = ()
        standalone_plans = (
            _rebind_standalone_plan(partitioned.entries[0].graph, template),
        )
    else:
        _partition_context, partitioned = _single_profile_partition(
            action_valid_ir1()
        )
        fusion_plans = (_rebind_fusion_plan(partitioned.entries[0].graph),)
        standalone_plans = ()
    context = naive_inter_die_planning_context("n4_schema_fixture")
    entry = InterDiePlannedProfile.create(
        source=partitioned.entries[0],
        context=context,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
    )
    bundle = InterDiePlanBundle.create(
        source=partitioned,
        context=context,
        entries=(entry,),
    )
    bundle.validate_against(partitioned, context)
    return partitioned, context, bundle


def _restable_planning_entry(
    entry: InterDiePlannedProfile,
    **updates: object,
) -> InterDiePlannedProfile:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "inter_die_planned_profile",
            changed._semantic_key(),
            schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_planning_bundle(
    bundle: InterDiePlanBundle,
    **updates: object,
) -> InterDiePlanBundle:
    changed = replace(bundle, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "inter_die_plan_bundle",
            changed._semantic_key(),
            schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
        ),
    )


class N4ContextSchemaTest(unittest.TestCase):
    def test_contexts_are_typed_frozen_stable_and_strict_round_trip(self) -> None:
        partition = FusionPartitionContext.create(producer_pass="unit")
        planning = naive_inter_die_planning_context("unit")
        partition.validate()
        planning.validate()
        self.assertEqual(
            partition.schema_version,
            FUSION_PARTITION_CONTEXT_SCHEMA_VERSION,
        )
        self.assertEqual(
            planning.schema_version,
            INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION,
        )
        self.assertEqual(partition.contract.value, "gemm_rs_all/v1")
        self.assertEqual(planning.fused_contract.value, "direct_naive/v1")
        self.assertEqual(
            planning.standalone_contract.value,
            "direct_all_gather/v1",
        )
        self.assertEqual(
            loads_dataclass(
                FusionPartitionContext,
                canonical_json(partition),
                path="partition_context",
            ),
            partition,
        )
        self.assertEqual(
            loads_dataclass(
                InterDiePlanningContext,
                canonical_json(planning),
                path="planning_context",
            ),
            planning,
        )
        self.assertEqual(
            FusionPartitionContext.create(producer_pass="other").id,
            partition.id,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            partition.id = "changed"  # type: ignore[misc]

        with self.assertRaisesRegex(SchemaError, "FusionPartitionContract"):
            replace(partition, contract="gemm_rs_all/v1").validate()  # type: ignore[arg-type]
        with self.assertRaisesRegex(SchemaError, "FusedInterDieContract"):
            replace(planning, fused_contract="direct_naive/v1").validate()  # type: ignore[arg-type]
        with self.assertRaisesRegex(SchemaError, "naive inter-die policy"):
            replace(
                planning,
                fused_policy=planning.standalone_policy,
            ).validate()
        with self.assertRaisesRegex(SchemaError, "direct_all_gather policy"):
            replace(
                planning,
                standalone_policy=planning.fused_policy,
            ).validate()
        raw = json.loads(canonical_json(planning))
        raw["standalone_contract"] = "ring/v1"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(InterDiePlanningContext, raw, path="planning_context")


class FusionPartitionedBundleSchemaTest(unittest.TestCase):
    def test_multi_profile_round_trip_stable_digest_and_exact_provenance(self) -> None:
        placed, context, bundle = _partition_fixture()
        decoded = loads_dataclass(
            FusionPartitionedIR1Bundle,
            canonical_json(bundle),
            path="fusion_partitioned_bundle",
        )
        decoded.validate_against(placed, context)
        self.assertEqual(decoded, bundle)
        self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
        self.assertEqual(len(decoded.entries), 2)
        self.assertTrue(
            all(
                skeleton.impl is FusionImpl.NONE
                for entry in decoded.entries
                for skeleton in entry.graph.fused_op_skeletons
            )
        )
        self.assertEqual(
            tuple(
                skeleton.id
                for skeleton in decoded.entries[0].graph.fused_op_skeletons
            ),
            tuple(
                skeleton.id
                for skeleton in decoded.entries[1].graph.fused_op_skeletons
            ),
        )

    def test_skeleton_coverage_order_fields_impl_and_stable_id_are_exact(self) -> None:
        _placed, _context, bundle = _partition_fixture()
        graph = bundle.entries[0].graph
        cases = (
            (
                _rebuild_ir1(graph, fused_op_skeletons=graph.fused_op_skeletons[:-1]),
                "select every candidate",
            ),
            (
                _rebuild_ir1(
                    graph,
                    fused_op_skeletons=tuple(reversed(graph.fused_op_skeletons)),
                ),
                "candidate order",
            ),
            (
                _rebuild_ir1(
                    graph,
                    fused_op_skeletons=(
                        replace(graph.fused_op_skeletons[0], id="arbitrary"),
                    )
                    + graph.fused_op_skeletons[1:],
                ),
                "unstable skeleton id",
            ),
            (
                _rebuild_ir1(
                    graph,
                    fused_op_skeletons=(
                        replace(
                            graph.fused_op_skeletons[0],
                            boundary_inputs=graph.fused_op_skeletons[0].boundary_outputs,
                        ),
                    )
                    + graph.fused_op_skeletons[1:],
                ),
                "source candidate field",
            ),
            (
                _rebuild_ir1(
                    graph,
                    fused_op_skeletons=(
                        replace(graph.fused_op_skeletons[0], impl=FusionImpl.NAIVE),
                    )
                    + graph.fused_op_skeletons[1:],
                ),
                "unplanned",
            ),
        )
        for changed_graph, message in cases:
            entry = _restable_partition_entry(bundle.entries[0], graph=changed_graph)
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                entry.validate()

    def test_source_context_profile_order_and_preserved_graph_are_exact(self) -> None:
        placed, context, bundle = _partition_fixture()
        first = bundle.entries[0]
        wrong_source = _restable_partition_entry(
            first,
            source_placed_entry_id="placed_entry_impostor",
        )
        with self.assertRaisesRegex(SchemaError, "source placed entry"):
            wrong_source.validate_against(placed.entries[0], context)

        wrong_context = _restable_partition_entry(
            first,
            partition_context_id="partition_context_impostor",
        )
        with self.assertRaisesRegex(SchemaError, "partition context"):
            wrong_context.validate_against(placed.entries[0], context)

        wrong_weight = _restable_partition_entry(
            first,
            weight=first.weight + 0.125,
        )
        with self.assertRaisesRegex(SchemaError, "profile and weight"):
            wrong_weight.validate_against(placed.entries[0], context)

        changed_node = replace(first.graph.nodes[0], impl_ref="other_impl")
        changed_graph = _rebuild_ir1(
            first.graph,
            nodes=(changed_node,) + first.graph.nodes[1:],
        )
        changed_entry = _restable_partition_entry(first, graph=changed_graph)
        with self.assertRaisesRegex(SchemaError, "only add"):
            changed_entry.validate_against(placed.entries[0], context)

        for entries in (
            bundle.entries[:1],
            (bundle.entries[0], bundle.entries[0]),
            tuple(reversed(bundle.entries)),
        ):
            candidate = FusionPartitionedIR1Bundle.create(
                source=placed,
                context=context,
                entries=entries,
            )
            with self.subTest(entries=len(entries)), self.assertRaises(SchemaError):
                candidate.validate()


class InterDiePlanBundleSchemaTest(unittest.TestCase):
    def test_fusion_and_standalone_round_trip_and_exact_binding(self) -> None:
        for standalone in (False, True):
            partitioned, context, bundle = _planning_fixture(
                standalone=standalone
            )
            decoded = loads_dataclass(
                InterDiePlanBundle,
                canonical_json(bundle),
                path="inter_die_plan_bundle",
            )
            decoded.validate_against(partitioned, context)
            self.assertEqual(decoded, bundle)
            self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
            if standalone:
                self.assertEqual(len(decoded.entries[0].standalone_plans), 1)
                self.assertFalse(decoded.entries[0].fusion_plans)
            else:
                self.assertEqual(len(decoded.entries[0].fusion_plans), 1)
                self.assertFalse(decoded.entries[0].standalone_plans)
                self.assertIs(
                    decoded.entries[0].fusion_plans[0].impl,
                    FusionImpl.NAIVE,
                )

    def test_plan_coverage_producer_source_profile_and_impl_are_exact(self) -> None:
        _partitioned, _context, bundle = _planning_fixture()
        entry = bundle.entries[0]
        plan = entry.fusion_plans[0]
        cases = (
            (
                _restable_planning_entry(entry, fusion_plans=()),
                "exactly cover skeletons",
            ),
            (
                _restable_planning_entry(
                    entry,
                    fusion_plans=(plan, plan),
                ),
                "exactly cover skeletons",
            ),
            (
                _restable_planning_entry(
                    entry,
                    fusion_plans=(replace(plan, producer_pass="other"),),
                ),
                "produced by inter_die_plan",
            ),
            (
                _restable_planning_entry(
                    entry,
                    fusion_plans=(replace(plan, source_ir1_id="impostor"),),
                ),
                "partition IR1",
            ),
            (
                _restable_planning_entry(
                    entry,
                    fusion_plans=(
                        replace(
                            plan,
                            profile_key=replace(
                                plan.profile_key,
                                prefill_tokens=plan.profile_key.prefill_tokens + 1,
                            ),
                        ),
                    ),
                ),
                "profile key",
            ),
            (
                _restable_planning_entry(
                    entry,
                    fusion_plans=(replace(plan, impl=FusionImpl.NONE),),
                ),
                "impl=naive",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                candidate.validate()

        _standalone_source, _standalone_context, standalone_bundle = (
            _planning_fixture(standalone=True)
        )
        standalone_entry = standalone_bundle.entries[0]
        standalone_plan = standalone_entry.standalone_plans[0]
        for plans, message in (
            ((), "exactly cover unfused collectives"),
            (
                (standalone_plan, standalone_plan),
                "exactly cover unfused collectives",
            ),
            (
                (replace(standalone_plan, producer_pass="other"),),
                "produced by inter_die_plan",
            ),
        ):
            candidate = _restable_planning_entry(
                standalone_entry,
                standalone_plans=plans,
            )
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                candidate.validate()

    def test_entry_and_bundle_provenance_cannot_be_impersonated(self) -> None:
        partitioned, context, bundle = _planning_fixture()
        source_entry = partitioned.entries[0]
        entry = bundle.entries[0]
        for candidate, message in (
            (
                _restable_planning_entry(
                    entry,
                    source_partitioned_entry_id="partition_entry_impostor",
                ),
                "source partitioned entry",
            ),
            (
                _restable_planning_entry(
                    entry,
                    planning_context_id="planning_context_impostor",
                ),
                "planning context",
            ),
            (
                _restable_planning_entry(entry, weight=entry.weight + 0.125),
                "profile and weight",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                candidate.validate_against(source_entry, context)

        changed_node = replace(
            entry.graph.nodes[0],
            origin_node_id="other_origin",
        )
        changed_graph = _rebuild_ir1(
            entry.graph,
            nodes=(changed_node,) + entry.graph.nodes[1:],
        )
        changed_plan = _rebind_fusion_plan(changed_graph)
        changed_entry = _restable_planning_entry(
            entry,
            source_ir1_id=changed_graph.id,
            graph=changed_graph,
            fusion_plans=(changed_plan,),
        )
        changed_entry.validate()
        with self.assertRaisesRegex(SchemaError, "preserve partition IR1"):
            changed_entry.validate_against(source_entry, context)

        for candidate, message in (
            (
                _restable_planning_bundle(
                    bundle,
                    source_partitioned_bundle_id="partition_bundle_impostor",
                ),
                "source partition bundle",
            ),
            (
                _restable_planning_bundle(
                    bundle,
                    placement_context_id="placement_context_impostor",
                ),
                "upstream contexts",
            ),
            (
                _restable_planning_bundle(
                    bundle,
                    planning_context_id="planning_context_impostor",
                ),
                "planning context",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError, message
            ):
                candidate.validate_against(partitioned, context)

        for entries in ((), (entry, entry)):
            candidate = InterDiePlanBundle.create(
                source=partitioned,
                context=context,
                entries=entries,
            )
            with self.subTest(entries=len(entries)), self.assertRaisesRegex(
                SchemaError, "one immutable entry per profile"
            ):
                candidate.validate()


if __name__ == "__main__":
    unittest.main()
