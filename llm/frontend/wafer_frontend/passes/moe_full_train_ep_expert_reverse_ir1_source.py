"""Official EP2 physical IR1 for per-expert dExpert and FP32 WGRAD source.

This carries true HBM owners, not executable rank-selective IR2 or native
multi-expert reverse. A later transport/lowering pass must consume both.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.ir0 import IR0, OpKind
from ..schema.ir1 import IR1
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_combine_backward_ir0 import (
    append_moe_full_train_combine_backward_ir0,
)
from .moe_full_train_expert_backward_ir0 import (
    append_moe_full_train_expert_backward_ir0,
)
from .moe_full_train_ep_ir1_source import (
    MoeEpSharedReverseIr1Candidate,
    build_moe_ep_shared_reverse_ir1_candidate,
)
from .moe_full_train_ep_placement import MoeFullTrainEpPlacement
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase
from .placement import _physical_instance, _physical_node


@dataclass(frozen=True, slots=True)
class MoeEp2ExpertReverseIr1Candidate:
    shared: MoeEpSharedReverseIr1Candidate
    source_ir0: IR0
    physical_ir1: IR1

    def validate_source_against(
        self, phase: FullMoeForwardIr0Phase, *, original_dense,
        sequence: MoeCompileSequence, placement: MoeFullTrainEpPlacement,
        context, dense_manifest,
    ) -> None:
        self.shared.validate_source_against(
            phase, original_dense=original_dense, sequence=sequence,
            placement=placement, context=context,
            dense_manifest=dense_manifest,
        )
        if phase.graph.instances[0].parallel.ep != 2:
            raise SchemaError("expert reverse IR1 requires EP2 source",
                              path="phase")
        expected = append_moe_full_train_expert_backward_ir0(
            append_moe_full_train_combine_backward_ir0(
                self.shared.source_ir0,
            ),
        )
        group = placement.physical_group
        ir1 = self.physical_ir1
        if (self.source_ir0 != expected
                or ir1.source_ir0_id != expected.id
                or ir1.profile != expected.profile
                or ir1.fusion_candidates != expected.fusion_candidates
                or ir1.instance_profiles != expected.instance_profiles
                or ir1.node_profiles != expected.node_profiles
                or ir1.pd_plan_id != expected.pd_plan_id
                or ir1.fabric != context.fabric
                or ir1.groups != (group,)
                or ir1.persistent_state_manifest !=
                   placement.persistent_state_manifest
                or ir1.nodes != tuple(_physical_node(node, group.id)
                                      for node in expected.nodes)
                or ir1.instances != (_physical_instance(
                    expected.instances[0], expected, (group,)),)
                or ir1.values != expected.values
                or ir1.edges != expected.edges
                or ir1.state_accesses != expected.state_accesses):
            raise SchemaError(
                "EP2 expert reverse IR1 differs from signed source or HBM owners",
                path="moe_ep2_expert_reverse_ir1_candidate.source",
            )
        node_index = {node.id: node for node in expected.nodes}
        states = {state.id: state for state in expected.persistent_states}
        bindings = {binding.state_ref: binding
                    for binding in ir1.persistent_state_manifest.bindings}
        rank_dies = {place.rank: place.die_id
                     for place in group.placements}
        if set(rank_dies) != {0, 1}:
            raise SchemaError("EP2 owner ranks are incomplete", path="ir1.groups")
        for expert in (0, 1):
            forward = node_index[f"{expected.instances[0].id}.layer1.moe.expert{expert}"]
            reverse = node_index[f"backward::{forward.id}"]
            if (reverse.kind is not OpKind.MOE_EXPERT_BACKWARD
                    or reverse.workload.expert != expert
                    or reverse.inputs[:4] != forward.inputs
                    or any(len(matches := tuple(
                        state for state in states.values()
                        if state.identity.tensor_ref == tensor_ref
                    )) != 1
                    or matches[0].identity.ep_owner_rank != expert
                    or bindings[matches[0].id].die_id != rank_dies[expert]
                    for tensor_ref in forward.inputs[1:])):
                raise SchemaError("expert reverse weight tape has wrong physical HBM owner",
                                  path=f"source.expert{expert}")
        ir1.validate("moe_ep2_expert_reverse_ir1_candidate.official")


def build_moe_ep2_expert_reverse_ir1_candidate(
    phase: FullMoeForwardIr0Phase, *, original_dense,
    sequence: MoeCompileSequence, placement: MoeFullTrainEpPlacement,
    context, dense_manifest,
) -> MoeEp2ExpertReverseIr1Candidate:
    shared = build_moe_ep_shared_reverse_ir1_candidate(
        phase, original_dense=original_dense, sequence=sequence,
        placement=placement, context=context,
        dense_manifest=dense_manifest,
    )
    source = append_moe_full_train_expert_backward_ir0(
        append_moe_full_train_combine_backward_ir0(shared.source_ir0),
    )
    group = placement.physical_group
    physical = IR1.create(
        producer_pass="moe_ep2_expert_reverse_ir1_candidate",
        source_ir0_id=source.id, profile=source.profile,
        fabric=context.fabric,
        instances=(_physical_instance(source.instances[0], source, (group,)),),
        groups=(group,),
        nodes=tuple(_physical_node(node, group.id) for node in source.nodes),
        values=source.values, edges=source.edges,
        fusion_candidates=source.fusion_candidates,
        state_accesses=source.state_accesses,
        persistent_state_manifest=placement.persistent_state_manifest,
        instance_profiles=source.instance_profiles,
        node_profiles=source.node_profiles, pd_plan_id=source.pd_plan_id,
    )
    result = MoeEp2ExpertReverseIr1Candidate(shared, source, physical)
    result.validate_source_against(
        phase, original_dense=original_dense, sequence=sequence,
        placement=placement, context=context, dense_manifest=dense_manifest,
    )
    return result


__all__ = ["MoeEp2ExpertReverseIr1Candidate",
           "build_moe_ep2_expert_reverse_ir1_candidate"]
