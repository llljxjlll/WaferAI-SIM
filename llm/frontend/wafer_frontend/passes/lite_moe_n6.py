"""Build the exact pre-fragment S3-Lite lowering intent."""

from __future__ import annotations

import math

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    BufferABI,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir0 import OpKind, SwiGluWorkload
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.lite_moe_execution import (
    LiteMoeGlobalDag,
    LiteMoeProjection,
    LiteMoeScheduled,
    LiteMoeTaskKind,
)
from ..schema.lite_moe_n4 import LiteMoeN4IR1
from ..schema.lite_moe_n6 import (
    LITE_MOE_N6_INTENT_SCHEMA_VERSION,
    LiteMoeBufferOperand,
    LiteMoeComputeUnit,
    LiteMoeDteUnit,
    LiteMoeN6Intent,
    LiteMoeStateLoadUnit,
)
from .lite_moe_execution import (
    validate_lite_moe_global,
    validate_lite_moe_projection,
    validate_lite_moe_schedule,
)


def _metadata(source: LiteMoeN4IR1) -> dict[str, tuple[tuple[int, ...], DType, str]]:
    result = {
        item.id: (item.shape, item.dtype, item.logical_layout)
        for item in source.graph.values
    }
    manifest = source.graph.persistent_state_manifest
    assert manifest is not None
    declarations = {item.id: item for item in manifest.declarations}
    from ..schema.persistent_state import canonical_state_staging_value_id
    for access in source.graph.state_accesses:
        declaration = declarations[access.state_ref]
        result[canonical_state_staging_value_id(access.id)] = (
            declaration.shape,
            declaration.dtype,
            declaration.layout,
        )
    for node in source.graph.nodes:
        if node.kind is OpKind.ELEMENTWISE and type(node.workload) is SwiGluWorkload:
            root = f"moe.value.{node.id.rsplit('.', 1)[0]}.gate_up"
            result[root] = (
                node.workload.rank_input_shape,
                node.workload.dtype,
                "MI_packed_gate_up",
            )
    return result


def _buffer_abis(
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> tuple[BufferABI, ...]:
    metadata = _metadata(source)
    die_index = {item.id: item for item in source.graph.fabric.dies}
    profile_index = {item.id: item for item in source.graph.fabric.sram_profiles}
    result: list[BufferABI] = []
    for binding in schedule.buffers:
        die = die_index[binding.die_id]
        core = next(item for item in die.cores if item.id == binding.core_ref)
        profile = profile_index[core.sram_profile_ref]
        region = next(item for item in profile.regions if item.name == "comm")
        shape, dtype, layout = metadata[binding.value_ref]
        banks = tuple(sorted({
            ((binding.address + offset) // profile.bank_interleave_bytes)
            % profile.bank_count
            for offset in range(0, binding.size_bytes, profile.bank_interleave_bytes)
        }))
        semantic = {
            "schedule_id": schedule.id,
            "binding_id": binding.id,
            "value_id": binding.value_ref,
            "logical_core": LogicalCoreRef(binding.die_id, core.local_core_id),
            "tensor_slice": TensorSlice(binding.value_ref, (0,) * len(shape), shape),
            "region_ref": region.id,
            "region_offset_bytes": binding.address - region.base_bytes,
            "size_bytes": binding.size_bytes,
            "alignment_bytes": profile.allocation_alignment_bytes,
            "banks": banks,
            "storage_id": f"moe.storage.{binding.id}",
            "alias_of": None,
            "lifetime_start": binding.first_ordinal,
            "lifetime_end_exclusive": binding.last_ordinal + 1,
            "dtype": dtype,
            "layout": layout,
            "ownership": BufferOwnership.OWNED,
        }
        abi = BufferABI(
            stable_artifact_id(
                "buffer_abi",
                semantic,
                schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        abi.validate("lite_moe_n6.buffer_abi")
        result.append(abi)
    return tuple(result)


def _components(
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    projection: LiteMoeProjection,
    source: LiteMoeN4IR1,
) -> tuple[
    tuple[BufferABI, ...],
    tuple[LiteMoeStateLoadUnit, ...],
    tuple[LiteMoeComputeUnit, ...],
    tuple[LiteMoeDteUnit, ...],
]:
    abis = _buffer_abis(schedule, source)
    abi_by_local_value = {
        (abi.logical_core.die_id, abi.value_id): abi for abi in abis
    }
    task_index = {task.id: task for die in projection.dies for task in die.tasks}
    action_index = {action.task_ref: action for action in global_dag.actions}
    state_loads: list[LiteMoeStateLoadUnit] = []
    computes: list[LiteMoeComputeUnit] = []
    for action in global_dag.actions:
        task = task_index[action.task_ref]
        if task.kind is LiteMoeTaskKind.DMA_IN:
            assert task.state_ref is not None and task.hbm_binding_ref is not None
            state_loads.append(LiteMoeStateLoadUnit(
                action.id,
                task.state_ref,
                task.hbm_binding_ref,
                abi_by_local_value[(task.die_id, task.write_values[0])].id,
                task.bytes,
            ))
        elif task.kind in (LiteMoeTaskKind.GEMM, LiteMoeTaskKind.SWIGLU):
            input_operands = tuple(
                LiteMoeBufferOperand(
                    abi_by_local_value[(task.die_id, value_ref)].id,
                    0,
                    abi_by_local_value[(task.die_id, value_ref)].size_bytes,
                )
                for value_ref in task.read_values
            )
            output_abi = abi_by_local_value[(task.die_id, task.write_values[0])]
            output = LiteMoeBufferOperand(
                output_abi.id,
                task.packed_output.offset_bytes if task.packed_output is not None else 0,
                task.packed_output.size_bytes if task.packed_output is not None else output_abi.size_bytes,
            )
            assert task.workload is not None
            computes.append(LiteMoeComputeUnit(
                action.id, task.node_ref, task.kind, task.workload,
                input_operands, output,
            ))
    dte_units: list[LiteMoeDteUnit] = []
    for flow in projection.flows:
        send = task_index[flow.send_task_ref]
        recv = task_index[flow.recv_task_ref]
        dte_units.append(LiteMoeDteUnit(
            flow.id,
            flow.p2p_binding_ref,
            flow.pair_route_ref,
            flow.source_die_id,
            flow.destination_die_id,
            action_index[flow.send_task_ref].id,
            action_index[flow.recv_task_ref].id,
            action_index[flow.wait_task_ref].id,
            abi_by_local_value[(send.die_id, send.read_values[0])].id,
            abi_by_local_value[(recv.die_id, recv.write_values[0])].id,
            stable_artifact_id(
                "s3_lite_moe_dte_channel", flow.id,
                schema_version=LITE_MOE_N6_INTENT_SCHEMA_VERSION,
            ),
            stable_artifact_id(
                "s3_lite_moe_dte_token", flow.id,
                schema_version=LITE_MOE_N6_INTENT_SCHEMA_VERSION,
            ),
            flow.bytes,
        ))
    return abis, tuple(state_loads), tuple(computes), tuple(dte_units)


def build_lite_moe_n6_intent(
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    projection: LiteMoeProjection,
    source: LiteMoeN4IR1,
) -> LiteMoeN6Intent:
    validate_lite_moe_global(global_dag, schedule, projection, source)
    abis, state_loads, computes, dte_units = _components(
        global_dag, schedule, projection, source
    )
    result = LiteMoeN6Intent.create(
        source_global_id=global_dag.id,
        source_schedule_id=schedule.id,
        source_projection_id=projection.id,
        source_n4_id=source.id,
        buffer_abis=abis,
        state_loads=state_loads,
        compute_units=computes,
        dte_units=dte_units,
    )
    validate_lite_moe_n6_intent(
        result, global_dag, schedule, projection, source
    )
    return result


def validate_lite_moe_n6_intent(
    result: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    projection: LiteMoeProjection,
    source: LiteMoeN4IR1,
) -> None:
    result.validate()
    validate_lite_moe_global(global_dag, schedule, projection, source)
    expected = _components(global_dag, schedule, projection, source)
    if (
        result.source_global_id != global_dag.id
        or result.source_schedule_id != schedule.id
        or result.source_projection_id != projection.id
        or result.source_n4_id != source.id
        or (
            result.buffer_abis,
            result.state_loads,
            result.compute_units,
            result.dte_units,
        )
        != expected
    ):
        raise SchemaError(
            "N6 intent is not an exact lowering quotient",
            path="lite_moe_n6_intent",
        )
    abi_index = {item.id: item for item in result.buffer_abis}
    if any(
        operand.offset_bytes + operand.size_bytes
        > abi_index[operand.buffer_abi_ref].size_bytes
        for unit in result.compute_units
        for operand in (*unit.inputs, unit.output)
    ):
        raise SchemaError(
            "compute operand view exceeds BufferABI",
            path="lite_moe_n6_intent.compute_units",
        )
    manifest = source.graph.persistent_state_manifest
    assert manifest is not None
    hbm_index = {item.id: item for item in manifest.bindings}
    if any(
        unit.hbm_binding_ref not in hbm_index
        or hbm_index[unit.hbm_binding_ref].state_ref != unit.state_ref
        or hbm_index[unit.hbm_binding_ref].size_bytes != unit.bytes
        for unit in result.state_loads
    ):
        raise SchemaError(
            "state load HBM closure mismatch",
            path="lite_moe_n6_intent.state_loads",
        )


__all__ = ["build_lite_moe_n6_intent", "validate_lite_moe_n6_intent"]
