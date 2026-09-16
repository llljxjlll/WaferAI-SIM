"""Independent P2-to-physical oracle for local MoE FP32 WGRAD production.

This checks staged three-projection expert and gate WGRAD up to the SGD
operand. Gate tree synchronization and shared Dense loss/backward are separate
contracts; this check does not certify complete training or numerical logits.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION, LinkedProgramManifest,
    RecordOpcode, SemanticOperandId,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan, FlexibleMoeSpec, MoeRectActionKind,
    MoeRectFlowStage, MoeRectStateRole,
)
from ..schema.global_action import LogicalCoreRef


def validate_moe_training_fp32_gradient_producers(
    plan: FlexibleMoeExecutablePlan, spec: FlexibleMoeSpec,
    manifest: LinkedProgramManifest,
) -> None:
    if manifest.source_global_dag_id != plan.id:
        raise SchemaError("gradient physical plan source drifted", path="manifest.source_global_dag_id")
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    abis = {abi.id: abi for fragment in manifest.fragments for abi in fragment.buffer_abi}
    bindings = {(item.fragment_id, item.logical_core, item.fragment_record_index, item.operand_id): item
                for item in manifest.address_operand_bindings}
    refs = {}
    for stream in manifest.core_streams:
        for entry in stream.records:
            local = next(local for local in fragments[entry.fragment_id].core_streams
                         if local.logical_core == stream.logical_core)
            record = local.records[entry.fragment_record_index]
            refs.setdefault((entry.source_global_action_id, stream.logical_core, record.opcode), []).append(
                (entry, record))

    def endpoint(action_id, rank, opcode, operand):
        core = LogicalCoreRef(rank, 0)
        records = refs.get((action_id, core, opcode), ())
        if len(records) != 1:
            raise SchemaError("FP32 gradient lacks one physical record", path=f"action[{action_id}].{opcode.name}")
        entry, record = records[0]
        binding = bindings.get((entry.fragment_id, core, entry.fragment_record_index, operand))
        if binding is None or len(binding.buffer_abi_ids) != 1:
            raise SchemaError("FP32 gradient lacks exact SRAM BufferABI", path=f"action[{action_id}].{operand.name}")
        abi = abis[binding.buffer_abi_ids[0]]
        local = next(local for local in fragments[entry.fragment_id].core_streams
                     if local.logical_core == core)
        relocation = next((reloc for reloc in local.address_relocations
                           if reloc.record_index == entry.fragment_record_index and reloc.operand_id == operand), None)
        if relocation is None or len(binding.tensor_slices) != 1:
            raise SchemaError("FP32 gradient physical relocation/view is missing", path=action_id)
        return abi, record, relocation.addend, binding.tensor_slices[0]

    states = {item.id: item for item in plan.state_bindings}
    actions = plan.actions
    for flow in plan.flows:
        if flow.stage is not MoeRectFlowStage.GATE_ALL_REDUCE:
            continue
        source_rank, target_rank = flow.source_rank, flow.destination_rank
        send = next(item for item in actions
                    if item.kind is MoeRectActionKind.SEND and item.flow_ref == flow.id)
        recv = next(item for item in actions
                    if item.kind is MoeRectActionKind.RECV and item.flow_ref == flow.id)
        source_grad = next(item.id for item in plan.state_bindings
                           if item.owner_rank == source_rank and item.role is MoeRectStateRole.GATE_GRADIENT)
        target_grad = next(item.id for item in plan.state_bindings
                           if item.owner_rank == target_rank and item.role is MoeRectStateRole.GATE_GRADIENT)
        source, outbound, source_addend, _ = endpoint(
            send.id, source_rank, RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS)
        target, inbound, target_addend, target_view = endpoint(
            recv.id, target_rank, RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS)
        is_reduce = flow.assignment_refs[0].startswith("gate_gradient.reduce.")
        expected_target = (f"flexible_moe.value.rank{target_rank}.gate_reduce_inputs"
                           if is_reduce else f"flexible_moe.value.rank{target_rank}.state.{target_grad}")
        expected_addend = (
            (1 + (2 * target_rank + 1, 2 * target_rank + 2).index(source_rank))
            * spec.hidden_size * spec.expert_count * 4 if is_reduce else 0
        )
        send_bytes = next(item.literal_value for item in outbound.operands if item.name == "length_bytes")
        recv_bytes = next(item.literal_value for item in inbound.operands if item.name == "length_bytes")
        if (source.value_id != f"flexible_moe.value.rank{source_rank}.state.{source_grad}"
                or target.value_id != expected_target
                or source.dtype is not DType.FP32 or target.dtype is not DType.FP32
                or source_addend != 0 or target_addend != expected_addend
                or target_view.shape != (spec.hidden_size * spec.expert_count,)
                or send_bytes != flow.logical_bytes or recv_bytes != flow.logical_bytes
                or flow.logical_bytes != 4 * spec.hidden_size * spec.expert_count):
            raise SchemaError("gate FP32 tree DTE endpoints disagree with P2", path=flow.id)
    for action in plan.actions:
        rank, m, h, i, e = (
            action.rank, len(action.assignment_refs), spec.hidden_size,
            spec.intermediate_size, spec.expert_count,
        )
        core = LogicalCoreRef(rank, 0)
        if action.kind is MoeRectActionKind.EXPERT_WGRAD:
            if action.flops != m * 6 * h * i:
                raise SchemaError("expert WGRAD physical work disagrees with P2", path=action.id)
            if m == 0:
                if refs.get((action.id, core, RecordOpcode.MATMUL)):
                    raise SchemaError("empty expert WGRAD emits phantom MATMUL", path=action.id)
                continue
            grad_ref = next(ref for ref in action.state_refs
                            if states[ref].role is MoeRectStateRole.EXPERT_GRADIENT)
            matrix_fp16, matrix_fp32 = 2 * h * i, 4 * h * i
            if states[grad_ref].size_bytes != 3 * matrix_fp32:
                raise SchemaError("P2 expert FP32 gradient tensor has incorrect size", path=grad_ref)
            projection_ids = (
                action.id,
                *(stable_artifact_id(
                    "flexible_moe_expert_wgrad_stage",
                    {"plan": plan.id, "wgrad": action.id, "stage": stage},
                    schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
                ) for stage in ("up", "down")),
            )
            cast_ids = tuple(stable_artifact_id(
                "flexible_moe_expert_wgrad_stage",
                {"plan": plan.id, "wgrad": action.id, "stage": f"cast_{stage}"},
                schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            ) for stage in ("gate", "up", "down"))
            native = bool(refs.get((action.id, core,
                                    RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING)))
            if native:
                for index, native_id in enumerate(projection_ids):
                    activation, record, activation_addend, activation_view = endpoint(
                        native_id, rank, RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                        SemanticOperandId.COMPUTE_INPUT_ADDRESS)
                    upstream, _, upstream_addend, upstream_view = endpoint(
                        native_id, rank, RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                        SemanticOperandId.COMPUTE_DATA_ADDRESS)
                    gradient, _, gradient_addend, gradient_view = endpoint(
                        native_id, rank, RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
                    literals = {item.name: item.literal_value for item in record.operands
                                if item.kind.name == "LITERAL"}
                    expected_activation = (
                        f"flexible_moe.value.rank{rank}.expert_activated" if index == 2
                        else f"flexible_moe.value.rank{rank}.activation")
                    expected_upstream = (
                        f"flexible_moe.value.rank{rank}.backward_gradient" if index == 2
                        else f"flexible_moe.value.rank{rank}.dgrad_gate_up")
                    expected_mn = (i, h) if index == 2 else (h, i)
                    expected_input_elements = m * (i if index == 2 else h)
                    expected_upstream_elements = m * (h if index == 2 else i)
                    if (activation.value_id != expected_activation
                            or upstream.value_id != expected_upstream
                            or gradient.value_id != f"flexible_moe.value.rank{rank}.state.{grad_ref}"
                            or (activation.dtype, upstream.dtype, gradient.dtype)
                               != (DType.FP16, DType.FP16, DType.FP32)
                            or (activation_addend, upstream_addend, gradient_addend)
                               != (0, 2*m*i if index == 1 else 0, index*matrix_fp32)
                            or activation_view.shape != (expected_input_elements,)
                            or upstream_view.shape != (expected_upstream_elements,)
                            or gradient_view.shape != (h*i,)
                            or tuple(literals.get(name) for name in ("m", "n", "k"))
                               != (*expected_mn, m)
                            or tuple(literals.get(name) for name in (
                                "activation_datatype", "upstream_datatype",
                                "gradient_datatype")) != (1, 1, 3)):
                        raise SchemaError("native 0x25 expert FP32 gradient tape, dimensions or output StateABI differs from P2", path=action.id)
                continue
            for index, (gemm_id, cast_id) in enumerate(zip(projection_ids, cast_ids)):
                stage, gemm, output_addend, stage_view = endpoint(
                    gemm_id, rank, RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
                cast_source, cast, source_addend, _ = endpoint(
                    cast_id, rank, RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS)
                gradient, _, gradient_addend, gradient_view = endpoint(
                    cast_id, rank, RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS)
                params = tuple(next(item.literal_value for item in gemm.operands if item.name == "parameters"))
                expected_params = (1, h, m, i) if index < 2 else (1, i, m, h)
                lits = {item.name: item.literal_value for item in cast.operands if item.kind.name == "LITERAL"}
                if (params != expected_params or 2 * params[0] * params[1] * params[2] * params[3] != m * 2 * h * i
                        or stage.id != cast_source.id
                        or stage.value_id != f"flexible_moe.value.rank{rank}.expert_wgrad_stage"
                        or gradient.value_id != f"flexible_moe.value.rank{rank}.state.{grad_ref}"
                        or gradient.dtype is not DType.FP32 or gradient.size_bytes != 3 * matrix_fp32
                        or (output_addend, source_addend, gradient_addend) !=
                           (index * matrix_fp16, index * matrix_fp16, index * matrix_fp32)
                        or stage_view.shape != (h * i,) or gradient_view.shape != (h * i,)
                        or any(lits[name] != value for name, value in (
                            ("input_dtype", 0), ("accumulator_dtype", 1), ("output_dtype", 1),
                            ("input_count", 1), ("element_count", h * i),
                            ("input_stride_bytes", matrix_fp16),
                        ))):
                    raise SchemaError("expert three-projection FP32 gradient handoff differs from P2", path=action.id)
        elif action.kind is MoeRectActionKind.GATE_WGRAD:
            if action.flops != m * 2 * h * e:
                raise SchemaError("gate WGRAD physical work disagrees with P2", path=action.id)
            if m == 0:
                if refs.get((action.id, core, RecordOpcode.MATMUL)):
                    raise SchemaError("empty gate WGRAD emits phantom MATMUL", path=action.id)
                continue
            grad_ref = next(ref for ref in action.state_refs
                            if states[ref].role is MoeRectStateRole.GATE_GRADIENT)
            cast_id = stable_artifact_id(
                "flexible_moe_gate_wgrad_cast_action",
                {"plan": plan.id, "wgrad": action.id},
                schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            )
            stage, gemm, _, _ = endpoint(action.id, rank, RecordOpcode.MATMUL,
                                         SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            cast_source, cast, _, _ = endpoint(cast_id, rank, RecordOpcode.LOCAL_REDUCE,
                                               SemanticOperandId.SOURCE_ADDRESS)
            gradient, _, _, view = endpoint(cast_id, rank, RecordOpcode.LOCAL_REDUCE,
                                            SemanticOperandId.DESTINATION_ADDRESS)
            params = tuple(next(item.literal_value for item in gemm.operands if item.name == "parameters"))
            lits = {item.name: item.literal_value for item in cast.operands if item.kind.name == "LITERAL"}
            has_children = any(child < spec.mesh.rank_count
                               for child in (2 * rank + 1, 2 * rank + 2))
            expected_destination = (
                f"flexible_moe.value.rank{rank}.gate_reduce_inputs" if has_children
                else f"flexible_moe.value.rank{rank}.state.{grad_ref}"
            )
            if (params != (1, h, m, e) or stage.id != cast_source.id
                    or gradient.value_id != expected_destination
                    or gradient.dtype is not DType.FP32 or gradient.size_bytes < 4 * h * e
                    or view.shape != (h * e,) or lits.get("element_count") != h * e
                    or lits.get("output_dtype") != 1):
                raise SchemaError("gate FP32 gradient handoff differs from P2", path=action.id)
        elif action.kind is MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE:
            grad_ref = next(ref for ref in action.state_refs
                            if states[ref].role is MoeRectStateRole.GATE_GRADIENT)
            children = tuple(child for child in (2 * rank + 1, 2 * rank + 2)
                             if child < spec.mesh.rank_count)
            source, reducer, _, _ = endpoint(action.id, rank, RecordOpcode.LOCAL_REDUCE,
                                              SemanticOperandId.SOURCE_ADDRESS)
            dest, _, _, dest_view = endpoint(action.id, rank, RecordOpcode.LOCAL_REDUCE,
                                             SemanticOperandId.DESTINATION_ADDRESS)
            lits = {item.name: item.literal_value for item in reducer.operands if item.kind.name == "LITERAL"}
            expected_source = (
                f"flexible_moe.value.rank{rank}.gate_reduce_inputs" if children
                else f"flexible_moe.value.rank{rank}.state.{grad_ref}"
            )
            if (source.value_id != expected_source or source.dtype is not DType.FP32
                    or source.size_bytes < (len(children) + 1) * h * e * 4
                    or dest.value_id != f"flexible_moe.value.rank{rank}.state.{grad_ref}"
                    or dest.dtype is not DType.FP32 or dest_view.shape != (h * e,)
                    or any(lits.get(name) != value for name, value in (
                        ("input_dtype", 1), ("output_dtype", 1),
                        ("input_count", 1 + len(children)),
                        ("element_count", h * e), ("input_stride_bytes", 4 * h * e),
                    ))):
                raise SchemaError("gate FP32 local reduce did not write P2 StateABI", path=action.id)
        elif action.kind is MoeRectActionKind.EXPERT_SGD:
            grad_ref = next(ref for ref in action.state_refs
                            if states[ref].role is MoeRectStateRole.EXPERT_GRADIENT)
            grad, update, _, _ = endpoint(action.id, rank, RecordOpcode.SGD_UPDATE,
                                           SemanticOperandId.COMPUTE_DATA_ADDRESS)
            if (grad.value_id != f"flexible_moe.value.rank{rank}.state.{grad_ref}"
                    or grad.dtype is not DType.FP32
                    or next(item.literal_value for item in update.operands
                            if item.name == "element_count") * 4 != grad.size_bytes):
                raise SchemaError("expert SGD does not read complete FP32 gradient", path=action.id)
        elif action.kind is MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE:
            source, reducer, _, _ = endpoint(action.id, rank, RecordOpcode.LOCAL_REDUCE,
                                              SemanticOperandId.SOURCE_ADDRESS)
            dest, _, _, _ = endpoint(action.id, rank, RecordOpcode.LOCAL_REDUCE,
                                     SemanticOperandId.DESTINATION_ADDRESS)
            grad_ref = next(ref for ref in action.state_refs
                            if states[ref].role is MoeRectStateRole.GATE_GRADIENT)
            gradient_value = f"flexible_moe.value.rank{rank}.state.{grad_ref}"
            lits = {item.name: item.literal_value for item in reducer.operands if item.kind.name == "LITERAL"}
            if (source.value_id != gradient_value or dest.value_id != gradient_value
                    or source.dtype is not DType.FP32 or dest.dtype is not DType.FP32
                    or lits.get("input_count") != 1 or lits.get("element_count") != h * e):
                raise SchemaError("gate sync completion does not retain FP32 StateABI", path=action.id)
        elif action.kind is MoeRectActionKind.GATE_SGD:
            grad_ref = next(ref for ref in action.state_refs
                            if states[ref].role is MoeRectStateRole.GATE_GRADIENT)
            grad, update, _, _ = endpoint(action.id, rank, RecordOpcode.SGD_UPDATE,
                                           SemanticOperandId.COMPUTE_DATA_ADDRESS)
            if (grad.value_id != f"flexible_moe.value.rank{rank}.state.{grad_ref}"
                    or grad.dtype is not DType.FP32 or grad.size_bytes != 4 * h * e
                    or next(item.literal_value for item in update.operands
                            if item.name == "element_count") != h * e):
                raise SchemaError("gate SGD does not consume synced FP32 gradient", path=action.id)


__all__ = ["validate_moe_training_fp32_gradient_producers"]
