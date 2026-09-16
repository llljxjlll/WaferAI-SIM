"""Real E2E/P2-source-backed 0x25 FP32 WGRAD tiles for both MoE layers.

The 12 named projection tiles describe exact physical operands and outputs;
this source materialization alone does not generate a gradient SRAM producer,
backward dataflow or optimizer-linked two-step executable TRAIN program.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.flexible_moe import MoeRectActionKind
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_ep_placement import MoeFullTrainEpPlacement
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase


_PROJECTIONS=("gate","up","down")


@dataclass(frozen=True,slots=True)
class MoeNamedExpertWgradTile:
    layer: int
    expert: int
    projection: str
    owner_ep_rank: int
    source_route_trace_ref: str
    source_route_trace_digest: str
    source_e2e_backward_op_ref: str
    source_p2_expert_wgrad_action_ref: str
    source_aggregate_state_abi_ref: str
    source_aggregate_slice_offset: int
    source_e2e_parameter_state_version0_ref: str
    native_workload: GemmWeightWgradWorkload

    @property
    def logical_flops(self) -> int:
        return self.native_workload.fma_ops*2

    @property
    def gradient_bytes(self) -> int:
        return self.native_workload.gradient_bytes


@dataclass(frozen=True,slots=True)
class MoeFullTrainNamedWgradTiles:
    source_moe_sequence_ref: str
    source_ir0_ref: str
    source_physical_case_ref: str
    tiles: tuple[MoeNamedExpertWgradTile,...]
    router_local_gradient_flops: tuple[tuple[int,int],...]

    def validate_against(
        self,phase: FullMoeForwardIr0Phase,sequence: MoeCompileSequence,
        placement: MoeFullTrainEpPlacement,*,original_dense,
        dense_manifest,context,
    ) -> None:
        placement.validate(phase,original_dense,dense_manifest,sequence,context)
        expected_tiles,expected_router=_derive(phase,sequence,placement)
        if (self.source_moe_sequence_ref!=sequence.id
                or self.source_ir0_ref!=phase.graph.id
                or self.source_physical_case_ref!=placement.physical_case_id
                or self.tiles!=expected_tiles
                or self.router_local_gradient_flops!=expected_router):
            raise SchemaError("every MoE expert projection WGRAD/source rank/FP32 output must match original E2E/P2 and physical ABI",
                              path="moe_full_train_named_wgrad_tiles")


def _derive(
    phase: FullMoeForwardIr0Phase,sequence: MoeCompileSequence,
    placement: MoeFullTrainEpPlacement,
) -> tuple[tuple[MoeNamedExpertWgradTile,...],tuple[tuple[int,int],...]]:
    request=sequence.materialization.request
    model=request.model
    if (phase.step!=0 or model.num_layers!=2
            or model.num_experts not in (1,2)
            or model.hidden_size!=4 or model.intermediate_size!=8
            or request.parallel.tp!=1
            or request.parallel.ep!=model.num_experts):
        raise SchemaError("only true two-layer TP1×EP1/EP2 V16/H4/I8 step0 source is enabled",
                          path="moe_full_train_named_wgrad_tiles.source")
    owners={owner.source_state_decl_ref:owner for owner in phase.ep_state_owners}
    states={state.id:state for state in phase.graph.persistent_states}
    homes={home.declaration_ref:home for home in placement.hbm_layout.ep}
    tiles=[]
    router=[]
    for layer in range(model.num_layers):
        unit=next(unit for unit in sequence.units
                  if (unit.step,unit.layer)==(0,layer))
        gate={action.rank:action for action in unit.plan.actions
              if action.kind is MoeRectActionKind.GATE_WGRAD}
        if (set(gate)!=set(range(model.num_experts))
                or gate[0].flops!=2*model.hidden_size*model.num_experts*unit.spec.trace.token_count
                or any(gate[rank].flops!=0 for rank in range(1,model.num_experts))):
            raise SchemaError("router rank1 zero local work and rank0 gate gradient must follow true P2",
                              path=f"moe_full_train_named_wgrad_tiles.layer{layer}.router")
        router.append(tuple(gate[rank].flops for rank in range(model.num_experts)))
        wgrad={action.rank:action for action in unit.plan.actions
               if action.kind is MoeRectActionKind.EXPERT_WGRAD}
        if set(wgrad)!=set(range(model.num_experts)):
            raise SchemaError("source expert FP32 WGRAD action per EP rank missing",
                              path=f"moe_full_train_named_wgrad_tiles.layer{layer}.expert")
        trace=next(trace for trace in
                   sequence.materialization.logical_graph.route_traces
                   if (trace.step,trace.layer)==(0,layer))
        for expert in range(model.num_experts):
            action=wgrad[expert]
            total=0
            k=trace.expert_token_counts[expert]
            if k==0:
                raise SchemaError("zero routed tokens need explicit zero-work expert backward phase",
                                  path=f"moe_full_train_named_wgrad_tiles.layer{layer}.expert{expert}")
            aggregate=next(group for group in unit.parameter_bindings
                           if group.expert==expert)
            if tuple(aggregate.parameter_refs)!=tuple(
                    f"layer.{layer}.expert.{expert}.{name}.weight"
                    for name in _PROJECTIONS):
                raise SchemaError("source expert aggregate is not exact gate/up/down full weights",
                                  path=f"moe_full_train_named_wgrad_tiles.layer{layer}.expert{expert}")
            for index,name in enumerate(_PROJECTIONS):
                ref=f"{phase.graph.instances[0].id}.layer{layer}.moe.expert{expert}.{name}.weight"
                state=next((state for state in phase.graph.persistent_states
                            if state.identity.tensor_ref==ref),None)
                if state is None:
                    raise SchemaError("named FP32 WGRAD has no expert parameter StateDecl",
                                      path=ref)
                owner=owners[state.id]
                home=homes[state.id]
                m,n=((model.intermediate_size,model.hidden_size)
                     if name=="down" else
                     (model.hidden_size,model.intermediate_size))
                native=GemmWeightWgradWorkload(
                    m,n,k,
                    source_forward_op_ref=unit.operation_binding.
                        expert_forward_operation_refs[expert],
                    source_parameter_state_ref=state.id,
                )
                native.validate()
                if (owner.ep_owner!=expert or home.die_id!=expert
                        or state.shape!=(m,n)
                        or state.tensor_bytes!=m*n*2
                        or home.original_leaf_size!=192
                        or home.slice_offset!=index*64
                        or native.gradient_bytes!=m*n*4):
                    raise SchemaError("one true FP32 expert WGRAD tensor or 192B aggregate slice was lost",
                                      path=f"moe_full_train_named_wgrad_tiles.{ref}")
                tile=MoeNamedExpertWgradTile(
                    layer,expert,name,owner.ep_owner,trace.id,
                    unit.route_trace_digest,
                    unit.operation_binding.expert_backward_operation_refs[expert],
                    action.id,home.original_abi_ref,home.slice_offset,
                    owner.source_e2e_state_version0_ref,native,
                )
                tiles.append(tile)
                total+=tile.logical_flops
            if (total!=action.flops
                    or total!=6*k*model.hidden_size*model.intermediate_size
                    or sum(tile.gradient_bytes for tile in tiles[-3:])!=384):
                raise SchemaError("three real projection WGRAD records fail P2 exact FLOPs/FP32 bytes",
                                  path=f"moe_full_train_named_wgrad_tiles.layer{layer}.expert{expert}")
    return tuple(tiles),tuple(router)


def build_moe_full_train_named_wgrad_tiles(
    phase: FullMoeForwardIr0Phase,sequence: MoeCompileSequence,
    placement: MoeFullTrainEpPlacement,*,original_dense,
    dense_manifest,context,
) -> MoeFullTrainNamedWgradTiles:
    placement.validate(phase,original_dense,dense_manifest,sequence,context)
    tiles,router=_derive(phase,sequence,placement)
    result=MoeFullTrainNamedWgradTiles(
        sequence.id,phase.graph.id,placement.physical_case_id,tiles,router,
    )
    result.validate_against(
        phase,sequence,placement,original_dense=original_dense,
        dense_manifest=dense_manifest,context=context,
    )
    return result


__all__=["MoeNamedExpertWgradTile","MoeFullTrainNamedWgradTiles",
           "build_moe_full_train_named_wgrad_tiles"]
