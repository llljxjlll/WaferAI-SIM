"""Bind genuine E2E expert parameter versions to named IR0/0x25 WGRAD tiles.

This is source lineage, not backward physical execution.  No activation tape,
upstream producer, FP32 gradient SRAM allocation or optimizer timeline is
invented here; those must be supplied by a later executable trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import OpKind, OpPhase, StateAccessMode
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_ep_placement import MoeFullTrainEpPlacement
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase
from .moe_full_train_named_wgrad_tiles import (
    MoeFullTrainNamedWgradTiles,
)


@dataclass(frozen=True, slots=True)
class MoeExpertParameterGradientSource:
    layer: int
    expert: int
    projection: str
    source_ir0_forward_node_ref: str
    source_e2e_forward_op_ref: str
    source_e2e_backward_op_ref: str
    source_e2e_gradient_op_ref: str
    source_e2e_sync_op_ref: str
    source_e2e_sgd_op_ref: str
    source_e2e_store_op_ref: str
    source_e2e_gradient_v1_op_ref: str
    source_e2e_sync_v1_op_ref: str
    source_e2e_sgd_v1_op_ref: str
    source_e2e_store_v1_op_ref: str
    source_parameter_state_decl_ref: str
    owner_ep_rank: int
    owner_physical_die: int
    source_parameter_v0_ref: str
    source_parameter_v1_ref: str
    source_parameter_v2_ref: str
    source_raw_gradient_v0_ref: str
    source_synced_gradient_v0_ref: str
    source_raw_gradient_v1_ref: str
    source_synced_gradient_v1_ref: str
    native_wgrad: GemmWeightWgradWorkload


@dataclass(frozen=True, slots=True)
class MoeFullTrainGradientSourceBridge:
    source_ir0_ref: str
    source_moe_sequence_ref: str
    source_physical_case_ref: str
    paths: tuple[MoeExpertParameterGradientSource, ...]

    def validate_against(
        self, phase: FullMoeForwardIr0Phase,
        tiles: MoeFullTrainNamedWgradTiles,
        sequence: MoeCompileSequence,
        placement: MoeFullTrainEpPlacement, *, original_dense,
        dense_manifest, context,
    ) -> None:
        tiles.validate_against(
            phase, sequence, placement, original_dense=original_dense,
            dense_manifest=dense_manifest, context=context,
        )
        if (self.source_ir0_ref != phase.graph.id
                or self.source_moe_sequence_ref != sequence.id
                or self.source_physical_case_ref != placement.physical_case_id
                or self.paths != _derive(phase, tiles, sequence, placement)):
            raise SchemaError(
                "each EP projection needs real IR0 forward/source E2E operation and two-step gradient-to-SGD version lineage",
                path="moe_full_train_gradient_source_bridge",
            )


def _derive(
    phase: FullMoeForwardIr0Phase,
    tiles: MoeFullTrainNamedWgradTiles,
    sequence: MoeCompileSequence,
    placement: MoeFullTrainEpPlacement,
) -> tuple[MoeExpertParameterGradientSource, ...]:
    if (phase.step != 0
            or len(tiles.tiles) != 6*sequence.materialization.request.model.num_experts):
        raise SchemaError("requires the true complete two-layer step0 source",
                          path="moe_full_train_gradient_source_bridge.source")
    nodes = {node.id: node for node in phase.graph.nodes}
    declarations = {item.id: item for item in phase.graph.persistent_states}
    bindings = {item.state_ref: item for item in
                placement.persistent_state_manifest.bindings}
    graph_states = {item.id: item for item in
                    sequence.materialization.logical_graph.state_versions}
    expected = []
    for tile in tiles.tiles:
        layer, expert, projection = (tile.layer, tile.expert, tile.projection)
        forward_node_ref = (
            f"{phase.graph.instances[0].id}.layer{layer}.moe.expert{expert}")
        node = nodes.get(forward_node_ref)
        declaration = declarations.get(tile.native_workload.
                                       source_parameter_state_ref)
        home = bindings.get(tile.native_workload.source_parameter_state_ref)
        if (node is None or node.kind is not OpKind.MOE_EXPERT_FORWARD
                or node.phase is not OpPhase.FWD or declaration is None
                or home is None or declaration.identity.ep_owner_rank != expert
                or home.die_id != expert
                or declaration.identity.tensor_ref not in node.inputs
                or declaration.shape != (tile.native_workload.m,
                                         tile.native_workload.n)
                or declaration.dtype is not DType.FP16
                or not any(access.node_ref == node.id
                           and access.state_ref == declaration.id
                           and access.mode is StateAccessMode.READ
                           for access in phase.graph.state_accesses)):
            raise SchemaError("real IR0 forward node does not read the EP expert weight",
                              path=f"moe_gradient_bridge.layer{layer}.expert{expert}.{projection}")
        two = tuple(next((unit for unit in sequence.units
                          if (unit.step, unit.layer) == (step, layer)), None)
                    for step in (0, 1))
        if None in two:
            raise SchemaError("both real E2E training steps are required",
                              path=f"moe_gradient_bridge.layer{layer}")
        bindings_by_step = tuple(next((item for item in unit.parameter_bindings
                                       if item.expert == expert), None)
                                 for unit in two)
        if None in bindings_by_step:
            raise SchemaError("expert owner parameter group absent from a step",
                              path=f"moe_gradient_bridge.layer{layer}.expert{expert}")
        first, second = bindings_by_step
        name = f"layer.{layer}.expert.{expert}.{projection}.weight"
        if (name not in first.parameter_refs
                or first.parameter_refs != second.parameter_refs
                or tile.native_workload.source_forward_op_ref !=
                   two[0].operation_binding.expert_forward_operation_refs[expert]
                or tile.source_e2e_backward_op_ref !=
                   two[0].operation_binding.expert_backward_operation_refs[expert]):
            raise SchemaError("source E2E forward/backward/parameter group changed",
                              path=f"moe_gradient_bridge.{name}")
        position = first.parameter_refs.index(name)
        input0, output0 = (first.input_parameter_state_refs[position],
                           first.output_parameter_state_refs[position])
        input1, output1 = (second.input_parameter_state_refs[position],
                           second.output_parameter_state_refs[position])
        states = tuple(graph_states.get(ref) for ref in
                       (input0, output0, input1, output1))
        if (None in states or output0 != input1
                or tuple((state.logical_name, state.version) for state in states)
                   != ((name, 0), (name, 1), (name, 1), (name, 2))
                or tile.source_e2e_parameter_state_version0_ref != input0):
            raise SchemaError("expert source versions must be v0→v1→v2 across SGD",
                              path=f"moe_gradient_bridge.{name}")
        native = replace(tile.native_workload,
                         source_forward_op_ref=node.id)
        native.validate()
        if (native.source_parameter_state_ref != declaration.id
                or native.gradient_bytes != declaration.tensor_bytes * 2):
            raise SchemaError("named 0x25 FP32 parameter gradient has wrong source extent",
                              path=f"moe_gradient_bridge.{name}")
        expected.append(MoeExpertParameterGradientSource(
            layer, expert, projection, node.id,
            tile.native_workload.source_forward_op_ref,
            tile.source_e2e_backward_op_ref,
            first.gradient_operation_refs[position],
            first.sync_operation_refs[position],
            first.sgd_operation_refs[position],
            first.store_operation_refs[position],
            second.gradient_operation_refs[position],
            second.sync_operation_refs[position],
            second.sgd_operation_refs[position],
            second.store_operation_refs[position],
            declaration.id, expert, home.die_id,
            input0, output0, output1,
            first.raw_gradient_state_refs[position],
            first.synced_gradient_state_refs[position],
            second.raw_gradient_state_refs[position],
            second.synced_gradient_state_refs[position], native,
        ))
    return tuple(expected)


def build_moe_full_train_gradient_source_bridge(
    phase: FullMoeForwardIr0Phase,
    tiles: MoeFullTrainNamedWgradTiles,
    sequence: MoeCompileSequence,
    placement: MoeFullTrainEpPlacement, *, original_dense,
    dense_manifest, context,
) -> MoeFullTrainGradientSourceBridge:
    tiles.validate_against(phase, sequence, placement,
                           original_dense=original_dense,
                           dense_manifest=dense_manifest, context=context)
    result = MoeFullTrainGradientSourceBridge(
        phase.graph.id, sequence.id, placement.physical_case_id,
        _derive(phase, tiles, sequence, placement),
    )
    result.validate_against(phase, tiles, sequence, placement,
                            original_dense=original_dense,
                            dense_manifest=dense_manifest, context=context)
    return result


__all__ = ["MoeExpertParameterGradientSource",
           "MoeFullTrainGradientSourceBridge",
           "build_moe_full_train_gradient_source_bridge"]
