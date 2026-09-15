"""Independently check the physical backward transport inside a P3 MoE block.

This covers the grad/dX SRAM handoff only.  It cannot certify complete
training: FP32 gradient production, the shared loss/backward spine, and two
step parameter state still require separate physical oracles and runtime.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest, RecordOpcode, SemanticOperandId
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan, MoeRectActionKind, MoeRectFlowStage,
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
            records.setdefault((ref.source_global_action_id, stream.logical_core, record.opcode), []).append(ref)

    def endpoint(action_id, rank, opcode, operand_id):
        core = LogicalCoreRef(rank, 0)
        refs = records.get((action_id, core, opcode), ())
        if len(refs) != 1:
            raise SchemaError("MoE training backward lacks one physical record", path=f"action[{action_id}].{opcode.name}")
        ref = refs[0]
        binding = bindings.get((ref.fragment_id, core, ref.fragment_record_index, operand_id))
        if binding is None or len(binding.buffer_abi_ids) != 1:
            raise SchemaError("MoE training backward has no exact SRAM endpoint", path=f"action[{action_id}].{operand_id.name}")
        return buffers[binding.buffer_abi_ids[0]]

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
        source = endpoint(send.id, flow.source_rank, RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS)
        target = endpoint(recv.id, flow.destination_rank, RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS)
        suffix = "backward_gradient" if flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT else "output"
        expect(source, flow.source_rank, suffix)
        expect(target, flow.destination_rank, suffix)
        if min(source.size_bytes, target.size_bytes) < flow.logical_bytes:
            raise SchemaError("MoE training backward physical SRAM payload is shorter than P2", path=flow.id)

    for action in actions:
        if action.kind is MoeRectActionKind.EXPERT_DGRAD:
            if not action.assignment_refs:
                if records.get((action.id, LogicalCoreRef(action.rank, 0), RecordOpcode.MATMUL)):
                    raise SchemaError("zero-work expert gradient emitted phantom MATMUL", path=action.id)
                continue
            inbound = endpoint(action.id, action.rank, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS)
            outbound = endpoint(action.id, action.rank, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            expect(inbound, action.rank, "backward_gradient")
            expect(outbound, action.rank, "output")
        if action.kind is MoeRectActionKind.COMBINE_BACKWARD and action.assignment_refs:
            returned = endpoint(action.id, action.rank, RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS)
            expect(returned, action.rank, "output")


__all__ = ["validate_moe_training_backward_handoff"]
