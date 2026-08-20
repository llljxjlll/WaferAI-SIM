from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_ir1
from llm.frontend.wafer_frontend.passes.placement import (
    place_stage4_ir0,
)
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_separated_ir0,
)
from llm.frontend.wafer_frontend.passes.stage4_pd import build_stage4_pd_plan
from llm.frontend.wafer_frontend.schema.action import (
    FUSION_PLAN_SCHEMA_VERSION,
    STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
    FusionPlan,
    StandaloneCollectivePlan,
)
from llm.frontend.wafer_frontend.schema.ir0 import NodeProfileBinding
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.placement import PlacementContext

from _fixtures import (
    naive_inter_die_planning_context,
    valid_hbm_address_spaces,
    valid_ir1,
)
from test_stage4_pd import _profile, _spec


def _three_die_fabric():
    base = valid_ir1().fabric
    die0, die1 = base.dies
    east1 = replace(
        die0.ports[0],
        id="east_1",
        runtime_port_id=1,
        egress_resource_id="port_1_east",
    )
    middle = replace(die1, ports=(*die1.ports, east1))
    west2 = replace(
        die1.ports[0],
        id="west_2",
        egress_resource_id="port_2_west",
    )
    die2 = replace(
        die1,
        id=2,
        coord=(2, 0),
        cores=tuple(
            replace(
                core,
                id=f"core_2_{core.local_core_id}",
                runtime_core_id=32 + core.local_core_id,
            )
            for core in die1.cores
        ),
        ports=(west2,),
    )
    link12 = replace(
        base.links[0],
        id="link_1_2",
        source_die=1,
        source_port_ref="east_1",
        destination_die=2,
        destination_port_ref="west_2",
        resource_id="d2d_1_2",
        link_group_ref="cut_1_2",
    )
    link21 = replace(
        base.links[1],
        id="link_2_1",
        source_die=2,
        source_port_ref="west_2",
        destination_die=1,
        destination_port_ref="east_1",
        resource_id="d2d_2_1",
        link_group_ref="cut_2_1",
    )
    fabric = replace(
        base,
        die_grid=(3, 1),
        dies=(die0, middle, die2),
        links=(*base.links, link12, link21),
    )
    fabric.validate("fabric")
    return fabric


def _stage4_case(prefill_tp: int, decode_tp: int):
    spec = _spec(prefill_tp, decode_tp)
    prefill_profile = _profile(prefill=True)
    decode_profile = _profile(prefill=False)
    if decode_tp > 1:
        request = replace(
            decode_profile.requests[0],
            decode_tokens=decode_tp,
            context_tokens=8 + decode_tp,
        )
        decode_profile = type(decode_profile).create(
            key=replace(
                decode_profile.key,
                decode_tokens=decode_tp,
                context_sum=8 + decode_tp,
                context_max=8 + decode_tp,
            ),
            requests=(request,),
        )
    spec = replace(
        spec,
        parallel=replace(
            spec.parallel,
            instances=tuple(
                replace(instance, sp=instance.tp > 1)
                for instance in spec.parallel.instances
            ),
        ),
        workload=replace(
            spec.workload,
            infer=replace(
                spec.workload.infer,
                pd_static=replace(
                    spec.workload.infer.pd_static,
                    prefill_profile=prefill_profile.key,
                    decode_profile=decode_profile.key,
                ),
            ),
        ),
    )
    spec.validate("spec")
    plan = build_stage4_pd_plan(
        spec,
        prefill_profile=prefill_profile,
        decode_profile=decode_profile,
    )
    graph = build_stage4_separated_ir0(spec, plan)
    fabric = _three_die_fabric()
    context = PlacementContext.create(
        producer_pass="stage4_inter_die_plan_test",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    return graph, context, plan


def _partitioned(prefill_tp: int, decode_tp: int) -> IR1:
    graph, context, plan = _stage4_case(prefill_tp, decode_tp)
    return partition_ir1(place_stage4_ir0(graph, context, plan))


class Stage4InterDiePlanTest(unittest.TestCase):
    def test_heterogeneous_tp_owner_profiles_are_exact(self) -> None:
        self.assertEqual(
            FUSION_PLAN_SCHEMA_VERSION,
            "wafer_frontend.fusion_plan/v1alpha10",
        )
        self.assertEqual(
            STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
            "wafer_frontend.standalone_collective_plan/v1alpha9",
        )
        planning_context = naive_inter_die_planning_context("stage4")
        for prefill_tp, decode_tp, owner in ((2, 1, "P0"), (1, 2, "D0")):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                graph = _partitioned(prefill_tp, decode_tp)
                profile_by_instance = {
                    binding.instance_ref: binding.profile
                    for binding in graph.instance_profiles
                }
                fusion_plans, standalone_plans = plan_ir1(
                    graph,
                    planning_context,
                )
                self.assertEqual((len(fusion_plans), len(standalone_plans)), (4, 4))
                groups = {group.id: group for group in graph.groups}
                nodes = {node.id: node for node in graph.nodes}
                skeletons = {
                    skeleton.id: skeleton for skeleton in graph.fused_op_skeletons
                }
                for fusion_plan in fusion_plans:
                    self.assertEqual(skeletons[fusion_plan.fused_op_id].instance_id, owner)
                    self.assertEqual(groups[fusion_plan.group_ref].instance_id, owner)
                    self.assertEqual(fusion_plan.profile_key, profile_by_instance[owner])
                    fusion_plan.validate_against(graph)
                for standalone_plan in standalone_plans:
                    self.assertEqual(nodes[standalone_plan.op_id].instance_id, owner)
                    self.assertEqual(groups[standalone_plan.group_ref].instance_id, owner)
                    self.assertEqual(standalone_plan.profile_key, profile_by_instance[owner])
                    standalone_plan.validate_against(graph)
                if owner == "P0":
                    self.assertNotEqual(profile_by_instance[owner], graph.profile)
                else:
                    self.assertEqual(profile_by_instance[owner], graph.profile)

    def test_owner_profile_and_cross_instance_skeleton_fail_closed(self) -> None:
        graph = _partitioned(2, 1)
        planning_context = naive_inter_die_planning_context("stage4")
        fusion_plans, standalone_plans = plan_ir1(graph, planning_context)
        plan = fusion_plans[0]
        other_profile = next(
            binding.profile
            for binding in graph.instance_profiles
            if binding.instance_ref != "P0"
        )
        semantic_key = plan._semantic_key()
        semantic_key["profile_key"] = other_profile
        wrong_profile = FusionPlan.create(
            producer_pass=plan.producer_pass,
            **semantic_key,
        )
        with self.assertRaisesRegex(SchemaError, "owner instance"):
            wrong_profile.validate_against(graph)

        standalone = standalone_plans[0]
        semantic_key = standalone._semantic_key()
        semantic_key["profile_key"] = other_profile
        wrong_standalone_profile = StandaloneCollectivePlan.create(
            producer_pass=standalone.producer_pass,
            **semantic_key,
        )
        with self.assertRaisesRegex(SchemaError, "owner instance"):
            wrong_standalone_profile.validate_against(graph)

        semantic_key = graph._semantic_key()
        extra_binding = replace(
            next(
                binding
                for binding in graph.instance_profiles
                if binding.instance_ref == "P0"
            ),
            profile=other_profile,
        )
        semantic_key["instance_profiles"] = tuple(
            sorted(
                (*graph.instance_profiles, extra_binding),
                key=lambda binding: (
                    binding.instance_ref,
                    binding.profile.stable_id(),
                ),
            )
        )
        profile_by_instance = {
            binding.instance_ref: binding.profile
            for binding in graph.instance_profiles
        }
        semantic_key["node_profiles"] = tuple(
            NodeProfileBinding(
                node_ref=node.id,
                profile=profile_by_instance[node.instance_id],
            )
            for node in graph.nodes
        )
        ambiguous_profile = IR1.create(
            producer_pass=graph.producer_pass,
            **semantic_key,
        )
        ambiguous_profile.validate("ambiguous_profile")
        with self.assertRaisesRegex(SchemaError, "exactly one profile"):
            plan_ir1(ambiguous_profile, planning_context)

        skeleton = graph.fused_op_skeletons[0]
        semantic_key = graph._semantic_key()
        semantic_key["fused_op_skeletons"] = (
            replace(skeleton, instance_id="D0"),
            *graph.fused_op_skeletons[1:],
        )
        cross_instance = IR1.create(
            producer_pass=graph.producer_pass,
            **semantic_key,
        )
        with self.assertRaisesRegex(SchemaError, "members do not belong"):
            plan_ir1(cross_instance, planning_context)


if __name__ == "__main__":
    unittest.main()
