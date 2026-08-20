"""Build the exact pre-fragment intent for four-die S3-Lite MoE inference."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import COMMAND_FRAGMENT_SCHEMA_VERSION, BufferABI
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir0 import SwiGluWorkload
from ..schema.ir1 import SramAllocator
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.lite_moe_dp4_execution import (
    LiteMoeDp4ExecutionCase,
    LiteMoeDp4TaskKind,
)
from ..schema.lite_moe_dp4_n6 import (
    LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION,
    LiteMoeDp4ComputeUnit,
    LiteMoeDp4DteUnit,
    LiteMoeDp4InferN6Intent,
)
from ..schema.lite_moe_n6 import LiteMoeBufferOperand, LiteMoeStateLoadUnit
from ..schema.persistent_state import canonical_state_staging_value_id
from .lite_moe_dp4_execution import validate_lite_moe_dp4_execution_case


def _metadata(source: LiteMoeDp4ExecutionCase) -> dict[str, tuple[tuple[int, ...], DType, str]]:
    result = {
        item.id: (item.shape, item.dtype, item.logical_layout)
        for item in source.n4.graph.values
    }
    manifest = source.n4.graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError("DP4 N6 requires persistent state", path="source.n4.graph")
    declarations = {item.id: item for item in manifest.declarations}
    for access in source.n4.graph.state_accesses:
        declaration = declarations[access.state_ref]
        result[canonical_state_staging_value_id(access.id)] = (
            declaration.shape,
            declaration.dtype,
            declaration.layout,
        )
    for projected_die in source.projection.dies:
        for task in projected_die.tasks:
            if (
                task.kind is not LiteMoeDp4TaskKind.SWIGLU
                or type(task.workload) is not SwiGluWorkload
                or len(task.read_values) != 1
            ):
                continue
            result[task.read_values[0]] = (
                task.workload.rank_input_shape,
                task.workload.dtype,
                "MI_packed_gate_up",
            )
    return result


def _buffer_abis(source: LiteMoeDp4ExecutionCase) -> tuple[BufferABI, ...]:
    metadata = _metadata(source)
    graph = source.n4.graph
    die_index = {item.id: item for item in graph.fabric.dies}
    profile_index = {item.id: item for item in graph.fabric.sram_profiles}
    result: list[BufferABI] = []
    for index, binding in enumerate(source.schedule.buffers):
        die = die_index[binding.die_id]
        core = next(item for item in die.cores if item.id == binding.core_ref)
        profile = profile_index[core.sram_profile_ref]
        region = next(item for item in profile.regions if item.name == "comm")
        if region.allocator is not SramAllocator.BLOCK:
            raise SchemaError("DP4 buffers require BLOCK comm SRAM", path=f"schedule.buffers[{index}]")
        try:
            shape, dtype, layout = metadata[binding.value_ref]
        except KeyError as error:
            raise SchemaError(
                "scheduled buffer lacks typed IR1 metadata",
                path=f"schedule.buffers[{index}].value_ref",
            ) from error
        banks = tuple(sorted({
            ((binding.address + offset) // profile.bank_interleave_bytes)
            % profile.bank_count
            for offset in range(0, binding.size_bytes, profile.bank_interleave_bytes)
        }))
        semantic = {
            "schedule_id": source.schedule.id,
            "binding_id": binding.id,
            "value_id": binding.value_ref,
            "logical_core": LogicalCoreRef(binding.die_id, core.local_core_id),
            "tensor_slice": TensorSlice(binding.value_ref, (0,) * len(shape), shape),
            "region_ref": region.id,
            "region_offset_bytes": binding.address - region.base_bytes,
            "size_bytes": binding.size_bytes,
            "alignment_bytes": profile.allocation_alignment_bytes,
            "banks": banks,
            "storage_id": f"moe.dp4.storage.{binding.id}",
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
        abi.validate(f"lite_moe_dp4_n6.buffer_abis[{index}]")
        result.append(abi)
    return tuple(result)


def _components(source: LiteMoeDp4ExecutionCase):
    abis = _buffer_abis(source)
    abi_by_local_value = {
        (abi.logical_core.die_id, abi.value_id): abi for abi in abis
    }
    task_index = {
        task.id: task for projected_die in source.projection.dies for task in projected_die.tasks
    }
    action_index = {action.task_ref: action for action in source.global_dag.actions}
    state_loads: list[LiteMoeStateLoadUnit] = []
    computes: list[LiteMoeDp4ComputeUnit] = []
    for action in source.global_dag.actions:
        task = task_index[action.task_ref]
        if task.kind is LiteMoeDp4TaskKind.DMA_IN:
            assert task.state_ref is not None and task.hbm_binding_ref is not None
            state_loads.append(LiteMoeStateLoadUnit(
                action.id,
                task.state_ref,
                task.hbm_binding_ref,
                abi_by_local_value[(task.die_id, task.write_values[0])].id,
                task.bytes,
            ))
        elif task.kind in (LiteMoeDp4TaskKind.GEMM, LiteMoeDp4TaskKind.SWIGLU):
            inputs = tuple(
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
            computes.append(LiteMoeDp4ComputeUnit(
                action.id,
                task.node_ref,
                task.kind,
                task.workload,
                inputs,
                output,
            ))
    dte_units: list[LiteMoeDp4DteUnit] = []
    for flow in source.projection.flows:
        send = task_index[flow.send_task_ref]
        recv = task_index[flow.recv_task_ref]
        dte_units.append(LiteMoeDp4DteUnit(
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
                "s3_lite_moe_dp4_dte_channel",
                flow.id,
                schema_version=LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION,
            ),
            stable_artifact_id(
                "s3_lite_moe_dp4_dte_token",
                flow.id,
                schema_version=LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION,
            ),
            flow.bytes,
        ))
    return abis, tuple(state_loads), tuple(computes), tuple(dte_units)


def build_lite_moe_dp4_infer_n6_intent(
    source: LiteMoeDp4ExecutionCase,
) -> LiteMoeDp4InferN6Intent:
    validate_lite_moe_dp4_execution_case(source)
    abis, state_loads, computes, dte_units = _components(source)
    result = LiteMoeDp4InferN6Intent.create(
        source_case_id=source.id,
        source_global_id=source.global_dag.id,
        source_schedule_id=source.schedule.id,
        source_projection_id=source.projection.id,
        source_n4_id=source.n4.id,
        buffer_abis=abis,
        state_loads=state_loads,
        compute_units=computes,
        dte_units=dte_units,
    )
    validate_lite_moe_dp4_infer_n6_intent(result, source)
    return result


def validate_lite_moe_dp4_infer_n6_intent(
    result: LiteMoeDp4InferN6Intent,
    source: LiteMoeDp4ExecutionCase,
) -> None:
    result.validate()
    validate_lite_moe_dp4_execution_case(source)
    if (
        result.source_case_id != source.id
        or result.source_global_id != source.global_dag.id
        or result.source_schedule_id != source.schedule.id
        or result.source_projection_id != source.projection.id
        or result.source_n4_id != source.n4.id
        or (
            result.buffer_abis,
            result.state_loads,
            result.compute_units,
            result.dte_units,
        ) != _components(source)
    ):
        raise SchemaError("intent is not the exact DP4 infer quotient", path="intent")


__all__ = [
    "build_lite_moe_dp4_infer_n6_intent",
    "validate_lite_moe_dp4_infer_n6_intent",
]
