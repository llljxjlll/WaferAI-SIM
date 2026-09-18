"""Strict TP1/EP2 full MoE forward source validation with real StateABI owners."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType,MeshAxisName
from ..schema.ir0 import (
    AttentionWorkload, JobKind, OpKind,OpPhase,StateAccess,
    StateAccessMode,
)
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase
from .validate_ir0 import (
    DenseIR0Validator,_rank_shape,_validate_dependency_dag,
)


class MoeFullTrainForwardValidator:
    """Validate only full forward; a complete TRAIN backward needs another gate."""

    @staticmethod
    def validate(phase: FullMoeForwardIr0Phase, *,
                 original_dense, sequence: MoeCompileSequence) -> None:
        if type(phase) is not FullMoeForwardIr0Phase:
            raise SchemaError("full MoE source requires typed source EP owner phase",
                              path="moe_full_forward_validator.phase")
        phase.validate_against(original_dense,sequence)
        graph = phase.graph
        path = "moe_full_forward_validator"
        if graph.job is not JobKind.TRAIN or phase.step not in (0, 1):
            raise SchemaError("only bounded step0/step1 forward TRAIN source is supported",
                              path=path)
        _validate_dependency_dag(graph,path)
        axes = {mesh.id:{axis.name:axis.size for axis in mesh.axes}
                for instance in graph.instances for mesh in instance.meshes}
        values = {value.id:value for value in graph.values}
        local = {value.id:_rank_shape(value,axes[value.sharding.mesh_ref],
                                     f"{path}.values[{i}]")
                 for i,value in enumerate(graph.values)}
        source_kinds = {
            OpKind.MOE_ROUTER, OpKind.MOE_ROUTE_FREEZE,
            OpKind.MOE_DISPATCH, OpKind.MOE_EXPERT_FORWARD,
            OpKind.MOE_COMBINE,
        }
        for node in graph.nodes:
            if node.kind in source_kinds:
                continue
            at = f"{path}.nodes[{node.id}]"
            if node.phase is not OpPhase.FWD or node.stage != 0:
                raise SchemaError("one full forward source cannot carry fabricated reverse actions",
                                  path=at)
            if node.kind is OpKind.GEMM:
                DenseIR0Validator._validate_gemm(
                    node,values,local,axes[node.mesh_ref],at,
                )
            elif node.kind is OpKind.NORM:
                DenseIR0Validator._validate_norm(
                    node,values,local,at,profile=None,
                )
            elif node.kind is OpKind.ELEMENTWISE:
                DenseIR0Validator._validate_elementwise(
                    node,values,local,at,
                )
            elif node.kind is OpKind.ATTENTION:
                DenseIR0Validator._validate_attention(
                    node,values,local,axes[node.mesh_ref],graph.profile,at,
                )
            elif node.kind is OpKind.COLLECTIVE:
                DenseIR0Validator._validate_collective(
                    node,values,local,axes[node.mesh_ref],at,
                )
            elif node.kind is OpKind.EMBEDDING:
                DenseIR0Validator._validate_embedding(
                    node,values,local,axes[node.mesh_ref],graph.profile,at,
                )
            elif node.kind is OpKind.ROPE:
                DenseIR0Validator._validate_rope(
                    node,values,local,axes[node.mesh_ref],graph.profile,at,
                )
            elif node.kind is OpKind.CE_FORWARD:
                DenseIR0Validator._validate_cross_entropy(
                    node,values,local,graph.profile,at,
                )
            else:
                raise UnsupportedFeatureError("non-forward or unsupported shared spine operator",
                                              path=at)
        source_states = {state.id:state for state in original_dense.persistent_states}
        current = {state.id:state for state in graph.persistent_states}
        mapped = dict(phase.shared_source_state_refs)
        if (len(mapped) != 11
                or set(mapped) != set(source_states) -
                   set(phase.removed_dense_state_refs)):
            raise SchemaError("shared trainable StateDecl must cover original source 11 exactly",
                              path=f"{path}.shared_states")
        for old_id,new_id in mapped.items():
            old = source_states[old_id]
            state = current[new_id]
            if (old.identity.tensor_ref != state.identity.tensor_ref
                    or old.identity.shard_index != state.identity.shard_index
                    or old.identity.mesh_ref != state.identity.mesh_ref
                    or old.shape != state.shape or old.dtype is not state.dtype
                    or old.tensor_bytes != state.tensor_bytes
                    or state.identity.kind is not StateKind.TRAINABLE_PARAMETER
                    or state.lifetime is not PersistentStateLifetime.PERSISTENT
                    or state.access is not PersistentStateAccess.READ_WRITE):
                raise SchemaError("shared Dense StateDecl was renamed, shrunk or not trainable",
                                  path=f"{path}.shared_states[{old_id}]")
        expected_accesses = {
            StateAccess.create(
                node_ref=access.node_ref,state_ref=mapped[access.state_ref],
                mode=access.mode,rank=access.rank,
                read_offset=access.read_offset,read_shape=access.read_shape,
                write_offset=access.write_offset,write_shape=access.write_shape,
            ) for access in original_dense.state_accesses
            if access.state_ref in mapped
        }
        for owner in phase.ep_state_owners:
            state = current[owner.source_state_decl_ref]
            weight = values[state.identity.tensor_ref]
            if (len(weight.consumers) != 1
                    or state.identity.kind is not StateKind.TRAINABLE_PARAMETER
                    or state.access is not PersistentStateAccess.READ_WRITE):
                raise SchemaError("expert/router weight lacks one true forward producer home",
                                  path=f"{path}.owner[{state.id}]")
            expected_accesses.add(StateAccess.create(
                node_ref=weight.consumers[0], state_ref=state.id,
                mode=StateAccessMode.READ,rank=owner.ep_owner,
            ))
        for layer, state_ref in enumerate(phase.route_state_refs):
            expected_accesses.add(StateAccess.create(
                node_ref=f"{graph.instances[0].id}.layer{layer}.moe.route_freeze",
                state_ref=state_ref, mode=StateAccessMode.READ, rank=0,
            ))
        if set(graph.state_accesses) != expected_accesses:
            raise SchemaError("source shared/EP forward parameter READ StateAccess mapping drifted",
                              path=f"{path}.state_accesses")


__all__ = ["MoeFullTrainForwardValidator"]
