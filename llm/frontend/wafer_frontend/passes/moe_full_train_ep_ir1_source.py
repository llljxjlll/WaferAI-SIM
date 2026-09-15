"""Strict EP2 source→physical IR1 candidate with explicit expert owner proof.

An actual IR1 is constructed with official schema types.  Existing IR1.validate
still rejects the MoE PhysicalNode workload and then mistakes TP shard0 for an
EP rank on expert1.  Callers must not submit this candidate to a finalizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.ir1 import IR1
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_ep_placement import MoeFullTrainEpPlacement
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase
from .placement import _physical_instance,_physical_node


@dataclass(frozen=True,slots=True)
class MoeEpPlacedStateOwnerProof:
    declaration_ref: str
    e2e_state_version0_ref: str
    e2e_tensor_view_ref: str
    e2e_ep_rank: int
    tp_shard: int
    physical_die: int
    physical_hbm_binding_ref: str
    source_parameter_state_abi_ref: str
    source_layer: int
    source_aggregate_slice_offset: int


@dataclass(frozen=True,slots=True)
class MoeEpPlacedIr1SourceCandidate:
    """Physical candidate/provenance; success requires public IR1.validate."""

    source_ir0_ref: str
    source_moe_sequence_ref: str
    physical_placement_case_ref: str
    physical_ir1: IR1
    owner_proofs: tuple[MoeEpPlacedStateOwnerProof,...]

    def validate_source_against(
        self,phase: FullMoeForwardIr0Phase,
        original_dense,sequence: MoeCompileSequence,
        placement: MoeFullTrainEpPlacement,context,dense_manifest,
    ) -> None:
        """Exact source/die/parameter bijection, no IR1 runtime acceptance."""
        placement.validate(phase,original_dense,dense_manifest,sequence,context)
        if (self.source_ir0_ref!=phase.graph.id
                or self.source_moe_sequence_ref!=sequence.id
                or self.physical_placement_case_ref!=placement.physical_case_id
                or self.physical_ir1.source_ir0_id!=phase.graph.id
                or self.physical_ir1.fabric!=context.fabric
                or self.physical_ir1.groups!=(placement.physical_group,)
                or self.physical_ir1.persistent_state_manifest!=
                   placement.persistent_state_manifest):
            raise SchemaError("physical IR1 source/case/EP group or true Dense+MoE HBM backing drifted",
                              path="moe_ep_ir1_candidate.source")
        group=placement.physical_group
        source={node.id:node for node in phase.graph.nodes}
        actual={node.origin_node_id:node for node in self.physical_ir1.nodes}
        if (set(source)!=set(actual) or len(source)!=32
                or any(node.execution_group_ref!=group.id
                       or node.id!=ref or node.mesh_ref!=source[ref].mesh_ref
                       or node.kind is not source[ref].kind
                       or node.workload!=source[ref].workload
                       or node.inputs!=source[ref].inputs
                       or node.outputs!=source[ref].outputs
                       for ref,node in actual.items())
                or self.physical_ir1.values!=phase.graph.values
                or self.physical_ir1.edges!=phase.graph.edges
                or self.physical_ir1.state_accesses!=phase.graph.state_accesses):
            raise SchemaError("every original shared/MoE forward node and DATA state access needs one physical candidate",
                              path="moe_ep_ir1_candidate.nodes")
        manifest=placement.persistent_state_manifest
        bindings={binding.state_ref:binding for binding in manifest.bindings}
        states={state.id:state for state in phase.graph.persistent_states}
        homes={home.declaration_ref:home for home in placement.hbm_layout.ep}
        owners={owner.source_state_decl_ref:owner
                for owner in phase.ep_state_owners}
        proofs={proof.declaration_ref:proof for proof in self.owner_proofs}
        if (len(proofs)!=16 or set(proofs)!=set(owners)):
            raise SchemaError("all 16 router/expert E2E EP owner proofs must be unique",
                              path="moe_ep_ir1_candidate.owner_proofs")
        ranks={place.rank:place.die_id for place in group.placements}
        for ref,proof in proofs.items():
            owner=owners[ref]
            home=homes[ref]
            state=states[ref]
            binding=bindings[ref]
            if (proof.e2e_state_version0_ref!=owner.source_e2e_state_version0_ref
                    or proof.e2e_tensor_view_ref!=owner.source_e2e_parameter_view_ref
                    or proof.e2e_ep_rank!=owner.ep_owner
                    or proof.tp_shard!=owner.tp_shard
                    or proof.tp_shard!=state.identity.shard_index
                    or proof.physical_die!=ranks[proof.e2e_ep_rank]
                    or proof.physical_die!=binding.die_id
                    or proof.physical_hbm_binding_ref!=binding.id
                    or proof.source_parameter_state_abi_ref!=home.original_abi_ref
                    or proof.source_layer!=home.source_layer
                    or proof.source_aggregate_slice_offset!=home.slice_offset
                    or binding.address!=home.physical_address
                    or binding.size_bytes!=home.tensor_size):
                raise SchemaError("actual source E2E EP tensor/aggregate StateABI disagrees with IR1 Die owner",
                                  path=f"moe_ep_ir1_candidate.owner[{ref}]")

    def validate_official_ir1(self) -> None:
        """The mandatory public IR1 validator currently rejects this phase."""
        self.physical_ir1.validate("moe_ep_ir1_candidate.official")


def build_moe_ep_placed_ir1_candidate(
    phase: FullMoeForwardIr0Phase,*,original_dense,
    sequence: MoeCompileSequence,
    placement: MoeFullTrainEpPlacement,context,dense_manifest,
) -> MoeEpPlacedIr1SourceCandidate:
    placement.validate(phase,original_dense,dense_manifest,sequence,context)
    group=placement.physical_group
    graph=phase.graph
    physical=tuple(_physical_node(node,group.id) for node in graph.nodes)
    instance=_physical_instance(graph.instances[0],graph,(group,))
    candidate=IR1.create(
        producer_pass="moe_ep_train_forward_placement_candidate",
        source_ir0_id=graph.id,profile=graph.profile,
        fabric=context.fabric,instances=(instance,),groups=(group,),
        nodes=physical,values=graph.values,edges=graph.edges,
        fusion_candidates=graph.fusion_candidates,
        state_accesses=graph.state_accesses,
        persistent_state_manifest=placement.persistent_state_manifest,
        instance_profiles=graph.instance_profiles,
        node_profiles=graph.node_profiles,pd_plan_id=graph.pd_plan_id,
    )
    homes={home.declaration_ref:home for home in placement.hbm_layout.ep}
    bindings={item.state_ref:item for item in
              placement.persistent_state_manifest.bindings}
    proofs=tuple(MoeEpPlacedStateOwnerProof(
        owner.source_state_decl_ref,owner.source_e2e_state_version0_ref,
        owner.source_e2e_parameter_view_ref,owner.ep_owner,owner.tp_shard,
        bindings[owner.source_state_decl_ref].die_id,
        bindings[owner.source_state_decl_ref].id,
        homes[owner.source_state_decl_ref].original_abi_ref,
        homes[owner.source_state_decl_ref].source_layer,
        homes[owner.source_state_decl_ref].slice_offset,
    ) for owner in phase.ep_state_owners)
    result=MoeEpPlacedIr1SourceCandidate(
        graph.id,sequence.id,placement.physical_case_id,candidate,proofs,
    )
    result.validate_source_against(
        phase,original_dense,sequence,placement,context,dense_manifest,
    )
    return result


__all__=["MoeEpPlacedStateOwnerProof","MoeEpPlacedIr1SourceCandidate",
           "build_moe_ep_placed_ir1_candidate"]
