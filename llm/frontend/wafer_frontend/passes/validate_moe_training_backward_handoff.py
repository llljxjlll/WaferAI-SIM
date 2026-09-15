"""Independently check the physical backward transport inside a P3 MoE block.

This covers the grad/dX SRAM handoff and all three expert DGRAD projections.
It cannot certify complete training: the shared loss/backward spine and two
step parameter state still require separate physical oracles and runtime.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION, LinkedProgramManifest,
    RecordOpcode, SemanticOperandId,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan, MoeRectActionKind, MoeRectFlowStage, MoeRectStateRole,
)
from ..schema.global_action import LogicalCoreRef


def validate_moe_training_backward_handoff(
    plan: FlexibleMoeExecutablePlan,
    manifest: LinkedProgramManifest,
) -> None:
    """Prove GRAD RECV→DGRAD→dX SEND→COMBINE against physical ABI refs."""
    if manifest.source_global_dag_id != plan.id:
        raise SchemaError("MoE training backward plan source drifted", path="manifest.source_global_dag_id")
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    buffers = {abi.id: abi for fragment in manifest.fragments for abi in fragment.buffer_abi}
    bindings = {
        (item.fragment_id, item.logical_core, item.fragment_record_index, item.operand_id): item
        for item in manifest.address_operand_bindings
    }
    records = {}
    for stream in manifest.core_streams:
        for ref in stream.records:
            fragment = fragments[ref.fragment_id]
            local = next(item for item in fragment.core_streams
                         if item.logical_core == stream.logical_core)
            record = local.records[ref.fragment_record_index]
            records.setdefault((ref.source_global_action_id, stream.logical_core, record.opcode), []).append((ref, record))

    def endpoint(action_id, rank, opcode, operand_id):
        core = LogicalCoreRef(rank, 0)
        refs = records.get((action_id, core, opcode), ())
        if len(refs) != 1:
            raise SchemaError("MoE training backward lacks one physical record", path=f"action[{action_id}].{opcode.name}")
        ref, record = refs[0]
        binding = bindings.get((ref.fragment_id, core, ref.fragment_record_index, operand_id))
        if binding is None or len(binding.buffer_abi_ids) != 1:
            raise SchemaError("MoE training backward has no exact SRAM endpoint", path=f"action[{action_id}].{operand_id.name}")
        local = next(item for item in fragments[ref.fragment_id].core_streams
                     if item.logical_core == core)
        relocation = next((item for item in local.address_relocations
                           if item.record_index == ref.fragment_record_index
                           and item.operand_id == operand_id), None)
        if relocation is None or len(binding.tensor_slices) != 1:
            raise SchemaError("MoE training backward has no exact physical view", path=action_id)
        return buffers[binding.buffer_abi_ids[0]], record, relocation.addend, binding.tensor_slices[0]

    def expect(abi, rank, suffix):
        if abi.value_id != f"flexible_moe.value.rank{rank}.{suffix}":
            raise SchemaError("MoE training backward SRAM endpoint disagrees with P2 flow", path=f"rank[{rank}].{suffix}")

    actions = plan.actions
    for flow in plan.flows:
        if flow.stage not in (MoeRectFlowStage.BACKWARD_GRADIENT, MoeRectFlowStage.BACKWARD_DX):
            continue
        send = next(item for item in actions
                    if item.kind is MoeRectActionKind.SEND and item.flow_ref == flow.id)
        recv = next(item for item in actions
                    if item.kind is MoeRectActionKind.RECV and item.flow_ref == flow.id)
        source, _, _, _ = endpoint(send.id, flow.source_rank, RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS)
        target, _, _, _ = endpoint(recv.id, flow.destination_rank, RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS)
        suffix = "backward_gradient" if flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT else "output"
        expect(source, flow.source_rank, suffix)
        expect(target, flow.destination_rank, suffix)
        if min(source.size_bytes, target.size_bytes) < flow.logical_bytes:
            raise SchemaError("MoE training backward physical SRAM payload is shorter than P2", path=flow.id)

    states = {item.id: item for item in plan.state_bindings}
    def literal(record, name):
        return next(item.literal_value for item in record.operands if item.name == name)

    def dgrad_child(action, stage):
        return stable_artifact_id(
            "flexible_moe_expert_dgrad_stage",
            {"plan": plan.id, "dgrad": action.id, "stage": stage},
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
        )

    for action in actions:
        if action.kind is MoeRectActionKind.EXPERT_DGRAD:
            if not action.assignment_refs:
                for stage, opcode in ((None, RecordOpcode.MATMUL),
                                      ("swiglu_backward", RecordOpcode.SWIGLU_BACKWARD_TIMING),
                                      ("gate", RecordOpcode.MATMUL),
                                      ("up", RecordOpcode.MATMUL),
                                      ("sum_dx", RecordOpcode.LOCAL_REDUCE)):
                    child = action.id if stage is None else dgrad_child(action, stage)
                    if records.get((child, LogicalCoreRef(action.rank, 0), opcode)):
                        raise SchemaError("zero-work expert gradient emitted phantom physical work", path=child)
                continue
            rank, m = action.rank, len(action.assignment_refs)
            down_in, down_record, _, down_in_view = endpoint(
                action.id, rank, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS)
            down_weight, _, down_weight_offset, down_weight_view = endpoint(
                action.id, rank, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS)
            down_out, _, _, down_out_view = endpoint(
                action.id, rank, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            expect(down_in, rank, "backward_gradient")
            params = tuple(literal(down_record, "parameters"))
            if len(params) != 4 or params[:2] != (1, m):
                raise SchemaError("expert down DGRAD GEMM shape differs from P2", path=action.id)
            _, _, h, i = params
            matrix_bytes, input_bytes, activated_bytes = 2 * h * i, 2 * m * h, 2 * m * i
            param_ref = next((ref for ref in action.state_refs
                              if states[ref].role is MoeRectStateRole.EXPERT_PARAMETER), None)
            if (not h or not i or action.flops != 6 * m * h * i
                    or param_ref is None or states[param_ref].size_bytes != 3 * matrix_bytes
                    or down_weight.value_id != f"flexible_moe.value.rank{rank}.state.{param_ref}"
                    or down_weight.dtype is not DType.FP16
                    or down_weight_offset != 2 * matrix_bytes
                    or down_weight_view.shape != (h * i,)
                    or down_in_view.shape[0] < m * h
                    or down_out.value_id != f"flexible_moe.value.rank{rank}.dgrad_activated"
                    or down_out_view.shape != (m * i,)
                    or down_in.size_bytes < input_bytes or down_out.size_bytes < activated_bytes):
                raise SchemaError("expert down DGRAD SRAM/FLOPs disagree with P2", path=action.id)
            backward_id = dgrad_child(action, "swiglu_backward")
            sw_in, sw_record, _, sw_in_view = endpoint(
                backward_id, rank, RecordOpcode.SWIGLU_BACKWARD_TIMING,
                SemanticOperandId.COMPUTE_INPUT_ADDRESS)
            sw_data, _, _, sw_data_view = endpoint(
                backward_id, rank, RecordOpcode.SWIGLU_BACKWARD_TIMING,
                SemanticOperandId.COMPUTE_DATA_ADDRESS)
            sw_out, _, _, sw_out_view = endpoint(
                backward_id, rank, RecordOpcode.SWIGLU_BACKWARD_TIMING,
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            if (sw_in.value_id != f"flexible_moe.value.rank{rank}.expert_gate_up"
                    or sw_in_view.shape != (2 * m * i,)
                    or sw_data.id != down_out.id or sw_data_view.shape != (m * i,)
                    or sw_out.value_id != f"flexible_moe.value.rank{rank}.dgrad_gate_up"
                    or sw_out_view.shape != (2 * m * i,)
                    or literal(sw_record, "parameters") != (m * i,)
                    or any(abi.dtype is not DType.FP16 for abi in (sw_in, sw_data, sw_out))):
                raise SchemaError("expert native backward SwiGLU dependency differs from P2", path=action.id)
            dx_halves = []
            for projection, offset in (("gate", 0), ("up", 1)):
                projection_id = dgrad_child(action, projection)
                proj_in, proj_record, proj_in_offset, proj_in_view = endpoint(
                    projection_id, rank, RecordOpcode.MATMUL,
                    SemanticOperandId.COMPUTE_INPUT_ADDRESS)
                proj_weight, _, proj_weight_offset, proj_weight_view = endpoint(
                    projection_id, rank, RecordOpcode.MATMUL,
                    SemanticOperandId.COMPUTE_DATA_ADDRESS)
                proj_out, _, proj_out_offset, proj_out_view = endpoint(
                    projection_id, rank, RecordOpcode.MATMUL,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
                if (tuple(literal(proj_record, "parameters")) != (1, m, i, h)
                        or proj_in.id != sw_out.id
                        or proj_in_offset != offset * activated_bytes
                        or proj_in_view.shape != (m * i,)
                        or proj_weight.id != down_weight.id
                        or proj_weight_offset != offset * matrix_bytes
                        or proj_weight_view.shape != (h * i,)
                        or proj_out.value_id != f"flexible_moe.value.rank{rank}.dgrad_dx_parts"
                        or proj_out_offset != offset * input_bytes
                        or proj_out_view.shape != (m * h,)):
                    raise SchemaError("expert gate/up DGRAD SRAM/FLOPs disagree with P2", path=projection_id)
                dx_halves.append(proj_out.id)
            sum_id = dgrad_child(action, "sum_dx")
            sum_in, sum_record, sum_in_offset, sum_in_view = endpoint(
                sum_id, rank, RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS)
            sum_out, _, sum_out_offset, sum_out_view = endpoint(
                sum_id, rank, RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS)
            if (len(set(dx_halves)) != 1 or sum_in.id != dx_halves[0]
                    or sum_in_offset or sum_in_view.shape != (2 * m * h,)
                    or sum_out_offset or sum_out_view.shape != (m * h,)
                    or sum_out.value_id != f"flexible_moe.value.rank{rank}.output"
                    or sum_in.dtype is not DType.FP16 or sum_out.dtype is not DType.FP16
                    or any(literal(sum_record, key) != value for key, value in (
                        ("input_dtype", 0), ("output_dtype", 0), ("input_count", 2),
                        ("element_count", m * h), ("input_stride_bytes", input_bytes)))):
                raise SchemaError("expert dX sum does not bridge the two P2 DGRAD halves", path=sum_id)
        if action.kind is MoeRectActionKind.COMBINE_BACKWARD and action.assignment_refs:
            returned, _, _, _ = endpoint(action.id, action.rank, RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS)
            expect(returned, action.rank, "output")


__all__ = ["validate_moe_training_backward_handoff"]
