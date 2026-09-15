"""Exact per-source-state gradient producer→sync→SGD→STORE physical gate.

The independent source oracle names every TP shard/DP owner/step and its
forward/backward producers.  No StateABI or operation label may be inferred
by substring or converted from an AdamW E2E carrier count.  This gate fails
closed while the Dense native reverse source/physical chain is incomplete.
"""

from __future__ import annotations

from collections import defaultdict
from math import prod
from typing import Mapping

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    BufferABI, LinkedProgramManifest, RecordOpcode, SemanticOperandId,
    StateABI,
)
from ..schema.common import DType
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import (
    DenseFullTrainRequirements, DenseRequiredGradientPath,
    DenseGradientDPReduction, DenseGradientLossObjective,
)
from ..schema.full_training_physical_dag import FullTrainingPhysicalDAG
from ..schema.ir0 import OpKind, ResidualWorkload, SwiGluWorkload
from ..schema.persistent_state import PersistentStateAccess, StateKind


def require_exact_dense_parameter_state_inventory(
    manifest: LinkedProgramManifest,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
) -> Mapping[tuple[str, int], StateABI]:
    """Prove real per-shard source StateDecl IDs, owners and FP16 extent."""
    requirements.validate_against(plan)
    manifest.validate("full_dense_gradient_parameter_source")
    templates = {template.state_ref: template for template
                 in plan.parameter_templates}
    if set(templates) != {path.parameter_state_ref for path in
                         requirements.paths}:
        raise SchemaError("source parameter templates and gradient oracle lack a bijection",
                          path="dense_parameter_source")
    expected = {(path.parameter_state_ref, path.rank)
                for path in requirements.paths}
    declarations = {decl.id: decl for decl in
                    plan.forward_graph.persistent_states}
    actual = {}
    for fragment in manifest.fragments:
        for abi in fragment.state_abi:
            if abi.state_ref not in templates:
                continue
            key = (abi.state_ref, abi.die_id)
            previous = actual.setdefault(key, abi)
            if previous != abi:
                raise SchemaError("source StateDecl shard changes physical HBM ABI across phases",
                                  path=f"dense_parameter_source[{key}]")
    if set(actual) != expected:
        raise SchemaError("source StateDecl TP/DP owner↔StateABI exact bijection absent",
                          path="dense_parameter_source")
    for key, abi in actual.items():
        template = templates[key[0]]
        if (key[1] not in template.owner_ranks
                or key[1] % plan.spec.tp_degree != template.tp_shard_index
                or abi.kind is not StateKind.TRAINABLE_PARAMETER
                or abi.access is not PersistentStateAccess.READ_WRITE
                or abi.dtype is not DType.FP16
                or abi.shape != declarations[key[0]].shape
                or abi.size_bytes != template.weight_bytes):
            raise SchemaError("source parameter lacks owned physical TRAINABLE_PARAMETER home",
                              path=f"dense_parameter_source[{key}]")
    return actual


def require_source_gemm_wgrad_geometry(
    plan: FlexibleDenseTrainPlan,
    path: DenseRequiredGradientPath,
    *,
    m: int, n: int, k: int,
) -> None:
    """WGRAD (M,N,K) outputs source shard M×N, K is forward rank rows."""
    states = {decl.id: decl for decl in plan.forward_graph.persistent_states}
    nodes = {node.id: node for node in plan.forward_graph.nodes}
    state = states[path.parameter_state_ref]
    forward = nodes[path.forward_op_refs[0]]
    if (forward.kind is not OpKind.GEMM or len(state.shape) != 2
            or type(m) is not int or type(n) is not int or type(k) is not int
            or min(m, n, k) < 1 or m * n != prod(state.shape)
            or (m, n) not in (state.shape, state.shape[::-1])
            or k != forward.workload.rank_shape[0]
            or forward.workload.rank_shape[1] *
                forward.workload.rank_shape[2] != prod(state.shape)):
        raise SchemaError("native WGRAD M×N and K differ from source weight shard and rows",
                          path=f"gradient_path[{path.parameter_state_ref}].geometry")


def require_full_dense_physical_gradient_paths(
    manifest: LinkedProgramManifest,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
    dag: FullTrainingPhysicalDAG,
    *,
    required_backward_opcodes: Mapping[str, RecordOpcode],
    required_wgrad_opcodes: Mapping[str, RecordOpcode],
) -> None:
    """Require every named native reverse producer and its actual SGD chain.

    `required_backward_opcodes` is an independent producer-owned versioned
    primitive contract for *all* required reverse operations.  A legacy
    MATMUL reused for attention/norm/backbone may not satisfy those contracts.
    Gradient BufferABI closures, exact source parameter state IDs, record
    dimensions, SGD input and state versions must agree on each physical die.
    """
    homes = require_exact_dense_parameter_state_inventory(
        manifest, plan, requirements,
    )
    if (requirements.loss_objective is not
            DenseGradientLossObjective.PER_ROW_CE_SUM
            or requirements.loss_gradient_seed_per_row != 1.0
            or requirements.dp_reduction is not
            DenseGradientDPReduction.FP32_RANK_MAJOR_SUM
            or requirements.optimizer_gradient_normalization):
        raise SchemaError("loss seed, physical DP SUM and unscaled SGD math contract differ",
                          path="full_dense_gradient_math_contract")
    if set(required_backward_opcodes) != set(
            requirements.required_backbone_backward_refs):
        raise SchemaError("all named backbone reverse operations need native physical opcode contract",
                          path="required_backward_opcodes")
    if set(required_wgrad_opcodes) != {
            path.named_wgrad_op_ref for path in requirements.paths}:
        raise SchemaError("all named parameter derivatives need independent native opcode contract",
                          path="required_wgrad_opcodes")
    source_nodes = {node.id: node for node in plan.forward_graph.nodes}
    reverse_native = {
        OpKind.GEMM: RecordOpcode.MATMUL,
        OpKind.NORM: getattr(RecordOpcode, "RMSNORM_BACKWARD_TIMING", None),
        OpKind.ATTENTION: getattr(RecordOpcode, "ATTENTION_BACKWARD_TIMING", None),
        OpKind.ROPE: getattr(RecordOpcode, "ROPE_BACKWARD_TIMING", None),
        OpKind.EMBEDDING: getattr(RecordOpcode, "EMBEDDING_WGRAD_TIMING", None),
    }
    for reverse_ref in requirements.required_backbone_backward_refs:
        if not reverse_ref.startswith("backward::") or (
            reverse_ref[len("backward::"):] not in source_nodes
        ):
            raise SchemaError("reverse source identity is not one validated forward node",
                              path=f"required_backward_opcodes[{reverse_ref}]")
        source = source_nodes[reverse_ref[len("backward::"):]]
        if source.kind is OpKind.ELEMENTWISE:
            expected_opcode = (
                RecordOpcode.SWIGLU_BACKWARD_TIMING
                if type(source.workload) is SwiGluWorkload else
                getattr(RecordOpcode, "RESIDUAL_BACKWARD_TIMING", None)
                if type(source.workload) is ResidualWorkload else None
            )
        else:
            expected_opcode = reverse_native.get(source.kind)
        if (expected_opcode is None
                or required_backward_opcodes[reverse_ref] is not expected_opcode):
            raise SchemaError("reverse native opcode lacks real source derivative family",
                              path=f"required_backward_opcodes[{reverse_ref}]")
    for template in plan.parameter_templates:
        families = {source_nodes[ref].kind for ref in template.forward_consumer_refs}
        opcode = required_wgrad_opcodes[template.wgrad_ref]
        allowed = {
            OpKind.GEMM: RecordOpcode.MATMUL,
            OpKind.NORM: getattr(RecordOpcode, "NORM_GAMMA_WGRAD_TIMING", None),
            OpKind.EMBEDDING: getattr(RecordOpcode, "EMBEDDING_WGRAD_TIMING", None),
        }
        if (len(families) != 1 or next(iter(families)) not in allowed
                or opcode is not allowed[next(iter(families))]):
            raise SchemaError("WGRAD opcode must implement the actual source derivative family",
                              path=f"required_wgrad_opcodes[{template.state_ref}]")
    dag.validate_against(
        manifest.fragments, manifest.core_streams,
        required_operation_ids=tuple(sorted({
            *(path.named_wgrad_op_ref for path in requirements.paths),
            *(path.named_sync_op_ref for path in requirements.paths),
            *(path.named_optimizer_op_ref for path in requirements.paths),
            *(path.named_store_op_ref for path in requirements.paths),
            *requirements.required_backbone_backward_refs,
        })),
    )
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    buffers = {abi.id: abi for fragment in manifest.fragments
               for abi in fragment.buffer_abi}
    states = {abi.id: abi for fragment in manifest.fragments
              for abi in fragment.state_abi}
    closures = {
        (entry.fragment_id, entry.logical_core, entry.fragment_record_index,
         entry.operand_id): entry
        for entry in manifest.address_operand_bindings
    }
    state_uses = {
        (entry.fragment_id, entry.logical_core, entry.fragment_record_index,
         entry.operand_id): entry
        for entry in manifest.state_operand_bindings
    }
    by_action = {item.id: item for item in dag.actions}
    named = defaultdict(list)
    for action in dag.actions:
        named[(action.step, action.logical_core.die_id,
               action.operation_ref)].append(action)

    def one(step: int, rank: int, ref: str, opcode: RecordOpcode):
        matches = [(action, fragment_id, index)
                   for action in named[(step, rank, ref)]
                   for fragment_id, index, physical_opcode
                   in action.executable_records if physical_opcode is opcode]
        if len(matches) != 1:
            raise SchemaError("one exact named native physical record is required",
                              path=f"gradient_path.step{step}.rank{rank}.{ref}.{opcode.name}")
        action, fragment_id, index = matches[0]
        fragment = fragments[fragment_id]
        local = next(stream for stream in fragment.core_streams
                     if stream.logical_core == action.logical_core)
        return action, local.records[index], fragment_id, index

    def buffer(action, fragment_id, index, operand: SemanticOperandId) -> BufferABI:
        closure = closures.get((fragment_id, action.logical_core, index, operand))
        if closure is None or len(closure.buffer_abi_ids) != 1:
            raise SchemaError("native producer/consumer lacks one exact physical BufferABI",
                              path=f"gradient_path[{action.operation_ref}].{operand.name}")
        return buffers[closure.buffer_abi_ids[0]]

    def same_physical_value(a: BufferABI, b: BufferABI) -> bool:
        return (a.id == b.id and a.storage_id == b.storage_id
                and a.binding_id == b.binding_id
                and a.value_id == b.value_id
                and a.logical_core == b.logical_core)

    def depends_on(later, earlier) -> bool:
        pending = list(later.depends_on)
        found = set()
        while pending:
            current = pending.pop()
            if current == earlier.id:
                return True
            if current in found:
                continue
            found.add(current)
            pending.extend(by_action[current].depends_on)
        return False

    for path in requirements.paths:
        step, rank = path.step, path.rank
        for reverse_ref in path.backward_producer_refs:
            opcode = required_backward_opcodes.get(reverse_ref)
            if opcode is None:
                raise SchemaError("parameter reverse source lacks independent native opcode",
                                  path=f"gradient_path.step{step}.rank{rank}.{reverse_ref}")
            one(step, rank, reverse_ref, opcode)
        derivative_opcode = required_wgrad_opcodes[path.named_wgrad_op_ref]
        wgrad, native, fid, idx = one(step, rank, path.named_wgrad_op_ref,
                                       derivative_opcode)
        produced = buffer(wgrad, fid, idx,
                          SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
        literals = {item.name: item.literal_value for item in native.operands
                    if item.literal_value is not None}
        if (produced.dtype is not DType.FP32
                or produced.size_bytes != path.gradient_bytes
                or produced.logical_core.die_id != rank
                or literals.get("output_datatype") != 3):
            raise SchemaError("WGRAD FP32 physical output/dimensions differ from source parameter",
                              path=f"gradient_path[{path.parameter_state_ref}].wgrad")
        if derivative_opcode is RecordOpcode.MATMUL and (
            not all(type(literals.get(field)) is int and literals[field] > 0
                    for field in ("m", "k", "n"))
        ):
            raise SchemaError("native GEMM derivative shape differs from exact source parameter",
                              path=f"gradient_path[{path.parameter_state_ref}].wgrad")
        if derivative_opcode is RecordOpcode.MATMUL:
            require_source_gemm_wgrad_geometry(
                plan, path, m=literals["m"], n=literals["n"],
                k=literals["k"],
            )
        if len(path.dp_group_ranks) > 1:
            waves = tuple(wave for wave in plan.gradient_waves
                          if wave.state_ref == path.parameter_state_ref)
            if not waves:
                raise SchemaError("source parameter has no DP gradient tree",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_sync")
            wave_payloads = {}
            for wave in waves:
                for transfer in wave.transfers:
                    source, destination = transfer.source_rank, transfer.destination_rank
                    edge_matches = [
                        (by_action[src], by_action[dst])
                        for src, dst in dag.transport_edges
                        if by_action[src].step == step and by_action[dst].step == step
                        and by_action[src].logical_core.die_id == source
                        and by_action[dst].logical_core.die_id == destination
                        and by_action[src].operation_ref == path.named_sync_op_ref
                        and by_action[dst].operation_ref == path.named_sync_op_ref
                    ]
                    if len(edge_matches) != 1:
                        raise SchemaError("source P2 DP wave lacks one physical SEND→RECV edge",
                                          path=f"gradient_path[{path.parameter_state_ref}].wave{wave.index}")
                    sending, receiving = edge_matches[0]
                    sends = [(fid, idx) for fid, idx, opcode
                             in sending.executable_records
                             if opcode is RecordOpcode.DTE_SEND]
                    receives = [(fid, idx) for fid, idx, opcode
                                in receiving.executable_records
                                if opcode is RecordOpcode.DTE_RECV]
                    if len(sends) != 1 or len(receives) != 1:
                        raise SchemaError("DP physical wave requires one SEND and one RECV",
                                          path=f"gradient_path[{path.parameter_state_ref}].wave{wave.index}")
                    sfid, sidx = sends[0]
                    rfid, ridx = receives[0]
                    sender = next(stream for stream in fragments[sfid].core_streams
                                  if stream.logical_core == sending.logical_core).records[sidx]
                    receiver = next(stream for stream in fragments[rfid].core_streams
                                    if stream.logical_core == receiving.logical_core).records[ridx]
                    send_buffer = buffer(sending, sfid, sidx,
                                         SemanticOperandId.SOURCE_ADDRESS)
                    recv_buffer = buffer(receiving, rfid, ridx,
                                         SemanticOperandId.DESTINATION_ADDRESS)
                    if any(
                        next((operand.literal_value for operand in record.operands
                              if operand.name == "length_bytes"), None) !=
                            transfer.logical_bytes
                        for record in (sender, receiver)
                    ) or any(abi.dtype is not DType.FP32
                             or abi.size_bytes != transfer.logical_bytes
                             for abi in (send_buffer, recv_buffer)):
                        raise SchemaError("DP P2 wave lacks exact physical FP32 gradient payload",
                                          path=f"gradient_path[{path.parameter_state_ref}].wave{wave.index}")
                    wave_payloads[wave.index] = (
                        sending, receiving, send_buffer, recv_buffer,
                    )
            if len(path.dp_group_ranks) != 2:
                raise SchemaError("multi-rank DP tree needs complete rank-major native source reduction",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_sync")
            root = path.dp_group_ranks[0]
            reduction, reduced, rid, rindex = one(
                step, root, path.named_sync_op_ref, RecordOpcode.LOCAL_REDUCE,
            )
            reduction_source = buffer(reduction, rid, rindex,
                                      SemanticOperandId.SOURCE_ADDRESS)
            reduction_result = buffer(reduction, rid, rindex,
                                      SemanticOperandId.DESTINATION_ADDRESS)
            rlits = {item.name: item.literal_value for item in reduced.operands
                     if item.literal_value is not None}
            if (rlits.get("input_dtype") != 1 or rlits.get("accumulator_dtype") != 1
                    or rlits.get("output_dtype") != 1 or rlits.get("reduce_op") != 1
                    or rlits.get("input_count") != 2
                    or rlits.get("element_count") != path.gradient_bytes // 4
                    or rlits.get("input_stride_bytes") != path.gradient_bytes
                    or reduction_source.dtype is not DType.FP32
                    or reduction_source.size_bytes != 2 * path.gradient_bytes
                    or reduction_result.dtype is not DType.FP32
                    or reduction_result.size_bytes != path.gradient_bytes):
                raise SchemaError("DP root native rank-major FP32 SUM contract/bytes differ",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_reduce")
            incoming = [by_action[dst] for src, dst in dag.transport_edges
                        if by_action[src].step == step
                        and by_action[src].operation_ref == path.named_sync_op_ref
                        and by_action[src].logical_core.die_id != root
                        and by_action[dst].logical_core.die_id == root
                        and by_action[dst].operation_ref == path.named_sync_op_ref]
            if len(incoming) != 1 or not depends_on(reduction, incoming[0]):
                raise SchemaError("DP FP32 SUM cannot run before all real RECV payloads",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_reduce")
            first, second = (wave_payloads[wave.index] for wave in waves)
            reduced_recv = first[3]
            broadcast_source = second[2]
            if (reduced_recv.logical_core.die_id != root
                    or reduced_recv.region_ref != reduction_source.region_ref
                    or reduced_recv.region_offset_bytes !=
                        reduction_source.region_offset_bytes + path.gradient_bytes
                    or broadcast_source.logical_core.die_id != root
                    or not same_physical_value(broadcast_source, reduction_result)):
                raise SchemaError("DP transport fails true rank-major input or reduced-output broadcast binding",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            root_copy_actions = named[(step, root, path.named_sync_op_ref)]
            root_copies = [
                (action, fid, idx, opcode) for action in root_copy_actions
                for fid, idx, opcode in action.executable_records
                if opcode is RecordOpcode.DTE_ISSUE
            ]
            if len(root_copies) != 1:
                raise SchemaError("DP root local rank-major input needs one real WGRAD copy",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            local_action, local_fid, local_idx, _ = root_copies[0]
            copy_source = buffer(local_action, local_fid, local_idx,
                                 SemanticOperandId.SOURCE_ADDRESS)
            copy_dest = buffer(local_action, local_fid, local_idx,
                               SemanticOperandId.DESTINATION_ADDRESS)
            root_wgrad_action, _, root_fid, root_idx = one(
                step, root, path.named_wgrad_op_ref, derivative_opcode,
            )
            root_gradient = buffer(root_wgrad_action, root_fid, root_idx,
                                   SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            if (not same_physical_value(copy_source, root_gradient)
                    or copy_dest.region_ref != reduction_source.region_ref
                    or copy_dest.region_offset_bytes !=
                        reduction_source.region_offset_bytes
                    or copy_dest.size_bytes != path.gradient_bytes
                    or not depends_on(local_action, root_wgrad_action)
                    or not depends_on(reduction, local_action)):
                raise SchemaError("DP root local gradient bytes do not populate first rank-major slice",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            if rank != root and not same_physical_value(produced, first[2]):
                raise SchemaError("DP child SEND does not consume this source parameter WGRAD",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            if rank == root:
                sync_result, sync = reduction_result, reduction
            else:
                matches = [by_action[dst] for src, dst in dag.transport_edges
                           if by_action[src].step == step
                           and by_action[src].logical_core.die_id == root
                           and by_action[dst].logical_core.die_id == rank
                           and by_action[src].operation_ref == path.named_sync_op_ref
                           and by_action[dst].operation_ref == path.named_sync_op_ref]
                if len(matches) != 1 or not depends_on(
                    by_action[next(src for src, dst in dag.transport_edges
                                   if dst == matches[0].id)], reduction,
                ):
                    raise SchemaError("DP child cannot consume an unreduced broadcast",
                                      path=f"gradient_path[{path.parameter_state_ref}].dp_broadcast")
                sync = matches[0]
                recv, recv_fid, recv_idx = next(
                    (record, fid, idx) for fid, idx, opcode
                    in sync.executable_records if opcode is RecordOpcode.DTE_RECV
                    for record in (next(stream for stream in
                        fragments[fid].core_streams if stream.logical_core ==
                        sync.logical_core).records[idx],)
                )
                sync_result = buffer(sync, recv_fid, recv_idx,
                                     SemanticOperandId.DESTINATION_ADDRESS)
                if (sync_result.dtype is not DType.FP32 or
                        sync_result.size_bytes != path.gradient_bytes):
                    raise SchemaError("DP broadcast destination lacks same FP32 extent",
                                      path=f"gradient_path[{path.parameter_state_ref}].dp_broadcast")
        else:
            sync, copy, sync_fid, sync_idx = one(
                step, rank, path.named_sync_op_ref, RecordOpcode.DTE_ISSUE,
            )
            wait, _, _, _ = one(
                step, rank, path.named_sync_op_ref, RecordOpcode.DTE_WAIT,
            )
            if sync.id != wait.id or not depends_on(sync, wgrad):
                raise SchemaError("local gradient sync lacks same-action blocking producer dependency",
                                  path=f"gradient_path[{path.parameter_state_ref}].local_sync")
            sync_source = buffer(sync, sync_fid, sync_idx, SemanticOperandId.SOURCE_ADDRESS)
            sync_result = buffer(sync, sync_fid, sync_idx,
                                 SemanticOperandId.DESTINATION_ADDRESS)
            size = next((item.literal_value for item in copy.operands
                         if item.name == "size_bytes"), None)
            if (not same_physical_value(produced, sync_source)
                    or sync_result.dtype is not DType.FP32
                    or sync_result.size_bytes != path.gradient_bytes
                    or size != path.gradient_bytes):
                raise SchemaError("local gradient sync does not copy source FP32 bytes",
                                  path=f"gradient_path[{path.parameter_state_ref}].local_sync")
        update, sgd, update_fid, update_idx = one(
            step, rank, path.named_optimizer_op_ref, RecordOpcode.SGD_UPDATE,
        )
        consumed = buffer(update, update_fid, update_idx,
                          SemanticOperandId.COMPUTE_DATA_ADDRESS)
        updated_weight = buffer(update, update_fid, update_idx,
                                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
        sgd_literals = {item.name: item.literal_value for item in sgd.operands
                        if item.literal_value is not None}
        if (not same_physical_value(sync_result, consumed)
                or sgd_literals.get("gradient_datatype") != 3
                or sgd_literals.get("element_count") != path.weight_bytes // 2
                or updated_weight.dtype is not DType.FP16
                or updated_weight.size_bytes != path.weight_bytes
                or not depends_on(update, sync)):
            raise SchemaError("SGD did not consume synchronized physical gradient",
                              path=f"gradient_path[{path.parameter_state_ref}].sgd")
        store, record, store_fid, store_idx = one(
            step, rank, path.named_store_op_ref, RecordOpcode.LSU_STORE,
        )
        closure = state_uses.get((store_fid, store.logical_core,
                                  store_idx, SemanticOperandId.HBM_ADDRESS))
        store_source = buffer(store, store_fid, store_idx,
                              SemanticOperandId.SOURCE_ADDRESS)
        literals = {item.name: item.literal_value for item in record.operands
                    if item.literal_value is not None}
        if (closure is None or states[closure.state_abi_id] !=
            homes[(path.parameter_state_ref, rank)]
                or literals.get("size_bytes") != path.weight_bytes
                or not same_physical_value(updated_weight, store_source)
                or not depends_on(store, update)):
            raise SchemaError("SGD must physically store same source StateDecl parameter",
                              path=f"gradient_path[{path.parameter_state_ref}].store")
        next_loads = [
            (by_action[target], fid, idx)
            for source, target in dag.state_version_edges if source == store.id
            and by_action[target].step == step + 1
            and by_action[target].logical_core.die_id == rank
            for fid, idx, opcode in by_action[target].executable_records
            if opcode is RecordOpcode.LSU_LOAD
            and (fid, by_action[target].logical_core, idx,
                 SemanticOperandId.HBM_ADDRESS) in state_uses
            and states[state_uses[(fid, by_action[target].logical_core,
                                   idx, SemanticOperandId.HBM_ADDRESS)]
                       .state_abi_id] == homes[(path.parameter_state_ref, rank)]
        ]
        if step < requirements.steps - 1 and len(next_loads) != 1:
            raise SchemaError("updated parameter needs exact next-step HBM LOAD version",
                              path=f"gradient_path[{path.parameter_state_ref}].version")


__all__ = ["require_exact_dense_parameter_state_inventory",
           "require_source_gemm_wgrad_geometry",
           "require_full_dense_physical_gradient_paths"]
