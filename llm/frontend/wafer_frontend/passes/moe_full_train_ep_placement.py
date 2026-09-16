"""Source-bound 2D TP1×EP1/EP2 group and disjoint Dense/MoE HBM home proposal.

The existing TP-only TrainPlacedIR1 builder cannot represent EP ownership;
this typed result is a physical placement input, not an executable IR1.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.moe_full_training_namespace import _SRAM_REGION_SHIFT
from ..schema.artifact_manifest import LinkedProgramManifest,ProgramSymbolKind
from ..schema.common import MeshAxisName
from ..schema.ir1 import (
    CanonicalBandwidthProfile,FlowWeight,GroupEmbedding,PhysicalGroup,
    RankPlacement,ResourceCapacity,ResourceWork,
)
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.persistent_state import HbmBinding,PersistentStateManifest
from ..schema.placement import PlacementContext,TrafficTemplate
from ..schema.common import stable_artifact_id
from .group_registry import _capacity_catalog,_link_index,_route
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase
from .moe_full_train_hbm_layout import (
    MoeFullTrainHbmLayout,build_moe_full_train_hbm_layout,
)


@dataclass(frozen=True,slots=True)
class MoeFullTrainEpPlacement:
    physical_case_id: str
    source_workload_case_ref: str
    source_ir0_ref: str
    source_context_ref: str
    physical_group: PhysicalGroup
    hbm_layout: MoeFullTrainHbmLayout
    persistent_state_manifest: PersistentStateManifest

    def validate(self, phase: FullMoeForwardIr0Phase,
                 original_dense, dense_manifest: LinkedProgramManifest,
                 sequence: MoeCompileSequence,
                 context: PlacementContext) -> None:
        phase.validate_against(original_dense,sequence)
        context.validate("moe_full_train_ep.context")
        _require_source_memory_and_sram(phase,sequence,context)
        self.physical_group.validate("moe_full_train_ep.physical_group")
        self.hbm_layout.validate(phase,dense_manifest,sequence)
        self.persistent_state_manifest.validate("moe_full_train_ep.state_manifest")
        expected_case=_physical_case_id(phase,sequence,context)
        if (self.physical_case_id != expected_case
                or self.source_workload_case_ref !=
                   sequence.materialization.request.case_id
                or self.source_ir0_ref != phase.graph.id
                or self.source_context_ref != context.id
                or self.hbm_layout.source_spaces != context.hbm_address_spaces):
            raise SchemaError("MoE EP2 group/HBM proposal lost exact signed source context",
                              path="moe_full_train_ep.source")
        expected = _physical_ep_group(phase,context)
        if self.physical_group != expected:
            raise SchemaError("EP ranks/routes/capacities disagree with real 2D source/fabric",
                              path="moe_full_train_ep.group")
        if self.persistent_state_manifest != _physical_ep_state_manifest(
                phase,context,self.hbm_layout):
            raise SchemaError("every Dense+EP state binding must equal true rebase home",
                              path="moe_full_train_ep.state_manifest")
        rank_home = {place.rank:place.die_id
                     for place in self.physical_group.placements}
        if any(home.die_id != rank_home[owner.ep_owner]
               for owner in phase.ep_state_owners
               for home in self.hbm_layout.ep
               if home.declaration_ref == owner.source_state_decl_ref):
            raise SchemaError("expert/router rank does not match real HBM Die owner",
                              path="moe_full_train_ep.owners")


def _physical_case_id(phase: FullMoeForwardIr0Phase,
                      sequence: MoeCompileSequence,
                      context: PlacementContext) -> str:
    """Hardware-bound public case, distinct from hardware-free request.case_id."""
    return stable_artifact_id(
        "moe_full_train_physical_case",
        {"request":sequence.materialization.request.digest,
         "memory_plan":sequence.materialization.memory_plan.id,
         "dense_moe_ir0":phase.graph.id,"hardware_placement":context.id},
        schema_version="wafer_frontend.moe_full_train_physical_case/v1alpha1",
    )


def _physical_ep_state_manifest(
    phase: FullMoeForwardIr0Phase,context: PlacementContext,
    layout: MoeFullTrainHbmLayout,
) -> PersistentStateManifest:
    bindings=tuple(HbmBinding.create(
        state_ref=home.declaration_ref,die_id=home.die_id,
        address=home.physical_address,size_bytes=home.tensor_size,
    ) for home in (*layout.shared,*layout.ep,*layout.routes))
    return PersistentStateManifest.create(
        address_spaces=context.hbm_address_spaces,
        declarations=phase.graph.persistent_states,bindings=bindings,
    )


def _require_source_memory_and_sram(
    phase: FullMoeForwardIr0Phase, sequence: MoeCompileSequence,
    context: PlacementContext,
) -> None:
    """Compare actual NpuSim region/high HBM address with P2 preflight limits."""
    memory=sequence.materialization.memory_plan
    capacities={entry.location_ref:entry for entry in memory.capacities
                if entry.tier.value=="hbm"}
    spaces={space.die_id:space for space in context.hbm_address_spaces}
    expected_dies=set(range(sequence.materialization.request.mesh.rank_count))
    if (set(capacities)!={f"die:{die}" for die in expected_dies}
            or set(spaces)!=expected_dies
            or any(capacities[f"die:{die}"].base_address!=0
                   or capacities[f"die:{die}"].capacity_bytes!=space.size_bytes
                   for die,space in spaces.items())):
        raise SchemaError("MoE production preflight HBM capacities do not match actual die address spaces",
                          path="moe_full_train_ep.memory_plan")
    profiles={profile.id:profile for profile in context.fabric.sram_profiles}
    minimum=min(profiles[core.sram_profile_ref].capacity_bytes
                for die in context.fabric.dies for core in die.cores)
    physical_peak=max(definition.value+_SRAM_REGION_SHIFT+
                      definition.size_bytes
                      for unit in sequence.units if unit.step==phase.step
                      for definition in
                          unit.linked_manifest.program_symbol_definitions
                      if definition.symbol.kind is ProgramSymbolKind.SRAM_REGION)
    if physical_peak>minimum:
        raise SchemaError("full shared+MoE physical SRAM region peak exceeds actual core SRAM capacity",
                          path="moe_full_train_ep.sram_peak")


def _physical_ep_group(phase: FullMoeForwardIr0Phase,
                       context: PlacementContext) -> PhysicalGroup:
    instance = phase.graph.instances[0]
    mesh = instance.meshes[0]
    ep_degree=instance.parallel.ep
    if (len(mesh.axes)!=2
            or tuple((axis.name,axis.size) for axis in mesh.axes)
                != ((MeshAxisName.TP,1),(MeshAxisName.EP,ep_degree))
            or instance.parallel.tp != 1 or ep_degree not in (1,2)
            or instance.parallel.dp != 1
            or context.placement.strategy.value != "compact"
            or not set(range(ep_degree)).issubset(
                {die.id for die in context.fabric.dies})):
        raise SchemaError("requires original TP1×EP1/EP2×DP1 source topology",
                          path="moe_full_train_ep.mesh")
    gid=f"group__{instance.id}__{mesh.id}__ep{ep_degree}"
    dies=tuple(range(ep_degree))
    placements=tuple(RankPlacement(rank,die,(0,rank))
                     for rank,die in enumerate(dies))
    if ep_degree==1:
        return PhysicalGroup(gid,instance.id,mesh.id,MeshAxisName.EP,(1,1),
                             placements,GroupEmbedding((),(),()))
    links=_link_index(context.fabric)
    catalog=_capacity_catalog(context.fabric)
    routes=tuple(_route(group_id=gid,source_rank=src,
                        destination_rank=dst,rank_to_die=dies,
                        fabric=context.fabric,links=links)
                 for src,dst in ((0,1),(1,0)))
    resources=list(dict.fromkeys(resource for route in routes
                                 for resource in route.resource_ids))
    capacities=tuple(ResourceCapacity(resource,catalog[resource])
                     for resource in resources)
    flow=tuple(FlowWeight(route.source_rank,route.destination_rank,1.0)
               for route in routes)
    work={resource:sum(resource in route.resource_ids for route in routes)
          for resource in resources}
    physical_work=tuple(ResourceWork(resource,float(work[resource]))
                        for resource in resources)
    bottleneck=min(capacities,key=lambda capacity:
                   capacity.bytes_per_cycle/work[capacity.id])
    profile=CanonicalBandwidthProfile(
        f"canonical__{gid}__direct_a2a_unit_chunk_v1",
        TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1.value,
        flow,physical_work,bottleneck.id,
        bottleneck.bytes_per_cycle/work[bottleneck.id],
    )
    return PhysicalGroup(gid,instance.id,mesh.id,MeshAxisName.EP,(1,2),
                         placements,GroupEmbedding(routes,capacities,(profile,)))


def build_moe_full_train_ep_placement(
    phase: FullMoeForwardIr0Phase, *, original_dense,
    dense_manifest: LinkedProgramManifest,
    sequence: MoeCompileSequence,
    context: PlacementContext,
) -> MoeFullTrainEpPlacement:
    phase.validate_against(original_dense,sequence)
    context.validate("moe_full_train_ep.context")
    _require_source_memory_and_sram(phase,sequence,context)
    group=_physical_ep_group(phase,context)
    layout=build_moe_full_train_hbm_layout(
        phase, original_dense=original_dense,
        dense_manifest=dense_manifest,sequence=sequence,
        spaces=context.hbm_address_spaces,
    )
    result=MoeFullTrainEpPlacement(
        _physical_case_id(phase,sequence,context),
        sequence.materialization.request.case_id,phase.graph.id,context.id,
        group,layout,_physical_ep_state_manifest(phase,context,layout),
    )
    result.validate(phase,original_dense,dense_manifest,sequence,context)
    return result


__all__=["MoeFullTrainEpPlacement","build_moe_full_train_ep_placement"]
