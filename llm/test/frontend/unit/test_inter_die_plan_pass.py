from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_bundle
from llm.frontend.wafer_frontend.passes.inter_die_plan import (
    plan_bundle,
    plan_profile,
)
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.placement import place_bundle
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.policies.swizzle_topo import SwizzlePlanner
from llm.frontend.wafer_frontend.schema.action import (
    FusionPlan,
    StandaloneCollectivePlan,
)
from llm.frontend.wafer_frontend.schema.common import ProfileKey, stable_artifact_id
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    FusionImpl,
    OpKind,
    ReduceOp,
)
from llm.frontend.wafer_frontend.schema.ir1 import FusedOpSkeleton, IR1, PhysicalNode
from llm.frontend.wafer_frontend.schema.n4 import (
    FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
    FusionPartitionContext,
    FusionPartitionedProfileIR1,
    InterDiePlanningContext,
    FusedInterDieContract,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.swizzle_plan import (
    SwizzleDeploymentReason,
    SwizzleFusionPlan,
)

from _fixtures import (
    naive_inter_die_planning_context,
    valid_hbm_address_spaces,
    valid_ir1,
    valid_spec,
)


def _spec(*, tp: int = 2, layers: int = 1, multiple_profiles: bool = False) -> ExperimentSpec:
    raw = valid_spec()
    raw["model"]["L"] = layers  # type: ignore[index]
    instance = raw["parallel"]["instances"][0]  # type: ignore[index]
    instance["tp"] = tp
    instance["sp"] = tp > 1
    if multiple_profiles:
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


def _partitioned_bundle(
    *,
    tp: int = 2,
    layers: int = 1,
    multiple_profiles: bool = False,
):
    spec = _spec(tp=tp, layers=layers, multiple_profiles=multiple_profiles)
    expanded = logical_expand(build_ir0(spec))
    fabric = valid_ir1().fabric
    placement = PlacementContext.create(
        producer_pass="unit",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = place_bundle(expanded, placement)
    partition_context = FusionPartitionContext.create(producer_pass="unit")
    return partition_bundle(placed, partition_context)


def _unfused_collective_ids(graph: IR1) -> tuple[str, ...]:
    fused_members = {
        node_id
        for skeleton in graph.fused_op_skeletons
        for node_id in skeleton.member_node_ids
    }
    return tuple(
        node.id
        for node in graph.nodes
        if node.kind is OpKind.COLLECTIVE and node.id not in fused_members
    )


class _RecordingFusedPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, ProfileKey]] = []
        self.delegate = NaiveInterDiePolicy()

    def plan(
        self,
        ir1: IR1,
        fused_op: FusedOpSkeleton,
        profile: ProfileKey,
    ) -> FusionPlan:
        self.calls.append((ir1.id, fused_op.id, profile))
        return self.delegate.plan(ir1, fused_op, profile)


class _RecordingStandalonePolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, ProfileKey]] = []
        self.delegate = DirectAllGatherPolicy()

    def plan(
        self,
        ir1: IR1,
        collective_op: PhysicalNode,
        profile: ProfileKey,
    ) -> StandaloneCollectivePlan:
        self.calls.append((ir1.id, collective_op.id, profile))
        return self.delegate.plan(ir1, collective_op, profile)


def _replace_entry_graph(
    source: FusionPartitionedProfileIR1,
    graph: IR1,
) -> FusionPartitionedProfileIR1:
    changed = replace(source, graph=graph)
    return replace(
        changed,
        id=stable_artifact_id(
            "fusion_partitioned_profile_ir1",
            changed._semantic_key(),
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        ),
    )


class InterDiePlanPassTest(unittest.TestCase):
    def test_tp2_l1_exact_order_impl_determinism_and_provenance(self) -> None:
        source = _partitioned_bundle()
        context = naive_inter_die_planning_context("unit")
        source_digest = canonical_digest(source)

        profile_result = plan_profile(source.entries[0], context)
        profile_result.validate_against(source.entries[0], context)
        result = plan_bundle(source, context)
        result.validate_against(source, context)
        self.assertEqual(result.entries[0], profile_result)
        entry = result.entries[0]
        self.assertEqual(len(entry.fusion_plans), 2)
        self.assertEqual(len(entry.standalone_plans), 2)
        self.assertEqual(
            tuple(plan.fused_op_id for plan in entry.fusion_plans),
            tuple(skeleton.id for skeleton in entry.graph.fused_op_skeletons),
        )
        self.assertEqual(
            tuple(plan.op_id for plan in entry.standalone_plans),
            _unfused_collective_ids(entry.graph),
        )
        self.assertTrue(
            all(plan.impl is FusionImpl.NAIVE for plan in entry.fusion_plans)
        )
        self.assertTrue(
            all(
                next(node for node in entry.graph.nodes if node.id == plan.op_id)
                .workload.collective
                is CollectiveKind.ALL_GATHER
                for plan in entry.standalone_plans
            )
        )
        self.assertEqual(entry.source_partitioned_entry_id, source.entries[0].id)
        self.assertEqual(entry.source_ir1_id, source.entries[0].graph.id)
        self.assertEqual(entry.planning_context_id, context.id)
        self.assertEqual(result.source_partitioned_bundle_id, source.id)
        self.assertEqual(result.placement_context_id, source.placement_context_id)
        self.assertEqual(result.partition_context_id, source.partition_context_id)
        self.assertEqual(result.planning_context_id, context.id)
        self.assertEqual(result.source_profiles, source.source_profiles)
        self.assertEqual(plan_bundle(source, context), result)
        self.assertEqual(canonical_digest(source), source_digest)

    def test_tp1_emits_no_inter_die_plans(self) -> None:
        source = _partitioned_bundle(tp=1)
        context = naive_inter_die_planning_context("unit")
        result = plan_bundle(source, context)
        result.validate_against(source, context)
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].fusion_plans, ())
        self.assertEqual(result.entries[0].standalone_plans, ())

    def test_l2_and_multiple_profiles_preserve_exact_counts_and_order(self) -> None:
        context = naive_inter_die_planning_context("unit")
        for multiple_profiles, expected_entries in ((False, 1), (True, 2)):
            with self.subTest(multiple_profiles=multiple_profiles):
                source = _partitioned_bundle(
                    layers=2,
                    multiple_profiles=multiple_profiles,
                )
                result = plan_bundle(source, context)
                result.validate_against(source, context)
                self.assertEqual(len(result.entries), expected_entries)
                self.assertEqual(
                    tuple(entry.profile_id for entry in result.entries),
                    tuple(entry.profile_id for entry in source.entries),
                )
                for source_entry, entry in zip(source.entries, result.entries):
                    self.assertEqual(len(entry.fusion_plans), 4)
                    self.assertEqual(len(entry.standalone_plans), 4)
                    self.assertEqual(
                        tuple(plan.fused_op_id for plan in entry.fusion_plans),
                        tuple(
                            skeleton.id
                            for skeleton in source_entry.graph.fused_op_skeletons
                        ),
                    )
                    self.assertEqual(
                        tuple(plan.op_id for plan in entry.standalone_plans),
                        _unfused_collective_ids(source_entry.graph),
                    )

    def test_protocol_policies_are_explicitly_injectable_and_called_in_order(self) -> None:
        source = _partitioned_bundle(multiple_profiles=True)
        context = naive_inter_die_planning_context("unit")
        fused = _RecordingFusedPolicy()
        standalone = _RecordingStandalonePolicy()
        result = plan_bundle(source, context, fused, standalone)
        result.validate_against(source, context)

        self.assertEqual(
            fused.calls,
            [
                (entry.graph.id, skeleton.id, entry.graph.profile)
                for entry in source.entries
                for skeleton in entry.graph.fused_op_skeletons
            ],
        )
        self.assertEqual(
            standalone.calls,
            [
                (entry.graph.id, node_id, entry.graph.profile)
                for entry in source.entries
                for node_id in _unfused_collective_ids(entry.graph)
            ],
        )

    def test_known_naive_and_swizzle_policies_cannot_cross_contracts(self) -> None:
        source = _partitioned_bundle()
        registry = production_registry()
        swizzle_resolved = registry.instantiate(
            RegistryKind.INTER_DIE,
            "swizzle_topo",
        )
        self.assertIsInstance(swizzle_resolved.implementation, SwizzlePlanner)
        naive_context = naive_inter_die_planning_context("unit")
        swizzle_context = InterDiePlanningContext.create(
            producer_pass="unit",
            fused_policy=swizzle_resolved.selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE,
                "direct_all_gather",
            ).selection,
            fused_contract=FusedInterDieContract.SWIZZLE_TOPO_V1,
        )

        for context, policy in (
            (naive_context, swizzle_resolved.implementation),
            (swizzle_context, NaiveInterDiePolicy()),
        ):
            with self.subTest(contract=context.fused_contract.value):
                with self.assertRaisesRegex(
                    SchemaError,
                    "fused policy implementation disagrees",
                ):
                    plan_bundle(source, context, policy)

    def test_forced_swizzle_plan_closes_real_ir1_without_rewriting_economics(self) -> None:
        entry = _partitioned_bundle().entries[0]
        planner = production_registry().create(
            RegistryKind.INTER_DIE,
            "swizzle_topo",
        )
        self.assertIsInstance(planner, SwizzlePlanner)
        graph = entry.graph
        plan = planner.plan_forced(
            graph,
            graph.fused_op_skeletons[0],
            graph.profile,
        )

        self.assertIs(
            plan.deployment_selection.reason,
            SwizzleDeploymentReason.FORCED_BY_POLICY,
        )
        self.assertNotEqual(
            plan.decision.selected_candidate_ref,
            plan.candidate.id,
        )
        plan.validate_against(graph)
        restored = loads_dataclass(
            SwizzleFusionPlan,
            canonical_json(plan),
            path="swizzle_plan",
        )
        self.assertEqual(restored, plan)
        restored.validate_against(graph)

    def test_unfused_reduce_scatter_is_rejected_before_policy_dispatch(self) -> None:
        source = _partitioned_bundle()
        entry = source.entries[0]
        graph = entry.graph
        fused_members = {
            node_id
            for skeleton in graph.fused_op_skeletons
            for node_id in skeleton.member_node_ids
        }
        node_index = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.kind is OpKind.COLLECTIVE and node.id not in fused_members
        )
        all_gather = graph.nodes[node_index]
        workload = replace(
            all_gather.workload,
            collective=CollectiveKind.REDUCE_SCATTER,
            reduce_op=ReduceOp.SUM,
            reduction_mesh_axes=all_gather.workload.mesh_axes,
            scatter_tensor_axis=0,
            gather_tensor_axis=None,
            rank_input_bytes=all_gather.workload.logical_tensor_bytes,
            rank_output_bytes=(
                all_gather.workload.logical_tensor_bytes
                // all_gather.workload.participant_count
            ),
        )
        nodes = (
            graph.nodes[:node_index]
            + (replace(all_gather, workload=workload),)
            + graph.nodes[node_index + 1 :]
        )
        fields = graph._semantic_key()
        fields["nodes"] = nodes
        changed_graph = IR1.create(producer_pass=graph.producer_pass, **fields)
        changed_entry = _replace_entry_graph(entry, changed_graph)
        changed_entry.validate()
        fused = _RecordingFusedPolicy()
        standalone = _RecordingStandalonePolicy()

        with self.assertRaisesRegex(SchemaError, "only AllGather"):
            plan_profile(
                changed_entry,
                naive_inter_die_planning_context("unit"),
                fused,
                standalone,
            )
        self.assertEqual(fused.calls, [])
        self.assertEqual(standalone.calls, [])


if __name__ == "__main__":
    unittest.main()
