"""Rank-indexed production materialization for flexible Dense backward."""

from __future__ import annotations

import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, BufferABI, CommandFragment,
    CoreFragmentStream, CoreRuntimeBinding, EmptyCoreAckPolicy,
    FragmentInterface, FragmentKind, LinkedCoreStream, LinkedProgramManifest,
    LinkedRecordRef, ManifestInputDigest, ManifestInputKind,
    ProgramControlEnvelope, ProgramFailurePolicy, ProgramSymbol,
    ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, RuntimeOperandField, RuntimeRelocation, RuntimeSymbol,
    RuntimeSymbolDefinition, RuntimeSymbolKind, SemanticOperandId, StateABI,
    StateOperandBinding, LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_dense_backward import FlexibleDenseBackwardLinkedProgram
from ..schema.flexible_dense_train import (
    FlexibleDenseTrainActionKind, FlexibleDenseTrainGradientSyncRole,
)
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import TensorSlice
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from ..schema.serde import canonical_digest
from .flexible_dense_backward_projection import build_flexible_dense_backward_lineage
from .flexible_dense_backward import (
    _align, _buffer, _hbm_symbol, _id, _label_symbol, _matmul,
    _region_symbol, _symbol,
)


def _runtime(kind: str, semantic: object, symbol_kind: RuntimeSymbolKind,
             source_ref: str) -> RuntimeSymbol:
    return RuntimeSymbol(
        stable_artifact_id(
            f"flexible_dense_backward_{kind}", semantic,
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
        ),
        symbol_kind,
        source_ref,
    )


def _gradient_carrier_bytes(logical_bytes: int, dp_degree: int) -> int:
    if type(logical_bytes) is not int or logical_bytes <= 0:
        raise SchemaError("logical gradient bytes must be positive", path="logical_bytes")
    if type(dp_degree) is not int or dp_degree <= 0:
        raise SchemaError("DP degree must be positive", path="dp_degree")
    return max(4096, logical_bytes) if dp_degree > 1 else logical_bytes


def materialize_flexible_dense_backward_multi(forward, fabric, spaces):
    forward.validate("forward")
    plan = forward.plan
    plan.validate("plan")
    fabric.validate("fabric")
    ranks = plan.spec.mesh.rank_count
    if ranks < 2:
        raise SchemaError("multi-rank materializer requires R>=2", path="plan.spec.mesh")
    if fabric.die_grid != plan.spec.mesh.physical_shape or len(fabric.dies) != ranks:
        raise SchemaError("fabric must exactly match Mesh", path="fabric")
    if tuple(space.die_id for space in spaces) != tuple(range(ranks)):
        raise SchemaError("HBM spaces must exactly cover ranks", path="hbm_address_spaces")
    for index, space in enumerate(spaces):
        space.validate(f"hbm_address_spaces[{index}]")

    ir, projection, schedule, global_dag = build_flexible_dense_backward_lineage(plan)
    state_decl = {item.id: item for item in plan.forward_graph.persistent_states}
    templates_by_state = {item.state_ref: item for item in plan.parameter_templates}
    cores = []
    core_specs = {}
    regions = {}
    for rank, die in enumerate(fabric.dies):
        if die.id != rank or not die.cores:
            raise SchemaError("each rank requires an ordered die/core", path="fabric.dies")
        spec = die.cores[0]
        core = LogicalCoreRef(rank, spec.local_core_id)
        profile = next(item for item in fabric.sram_profiles if item.id == spec.sram_profile_ref)
        if not profile.regions:
            raise SchemaError("rank requires an SRAM region", path="fabric.sram_profiles")
        cores.append(core)
        core_specs[core] = spec
        regions[core] = profile.regions[0]
    cores_tuple = tuple(cores)
    core_by_rank = dict(enumerate(cores_tuple))

    actions_by_rank = {
        rank: tuple(
            item for item in plan.rank_actions
            if item.rank == rank and item.kind is not FlexibleDenseTrainActionKind.FORWARD
        ) for rank in range(ranks)
    }
    local_templates = {
        rank: tuple(item for item in plan.parameter_templates if rank in item.owner_ranks)
        for rank in range(ranks)
    }
    buffers = {}
    state_abis = {}
    hbm_symbols = {}
    for rank, core in enumerate(cores_tuple):
        region = regions[core]
        offset = _align(region.base_bytes, 64)
        for template in local_templates[rank]:
            for role, dtype, size in (
                ("weight", DType.FP16, template.weight_bytes),
                ("gradient", DType.FP32,
                 _gradient_carrier_bytes(
                     template.gradient_bytes, plan.spec.dp_degree,
                 )),
            ):
                carrier_size = max(size, 256)
                offset = _align(offset, 64)
                buffers[(rank, template.state_ref, role)] = _buffer(
                    plan=plan, core=core, region_ref=region.id,
                    offset=offset - region.base_bytes, template=template,
                    role=role, dtype=dtype, size_bytes=carrier_size,
                    lifetime_end=len(actions_by_rank[rank]) + 1,
                )
                offset += carrier_size
        if offset > region.base_bytes + region.size_bytes:
            raise SchemaError("rank backward buffers exceed SRAM", path=f"fabric.rank{rank}")
        hbm_offset = _align(spaces[rank].base_address, spaces[rank].alignment_bytes)
        for template in local_templates[rank]:
            hbm_offset = _align(hbm_offset, spaces[rank].alignment_bytes)
            declaration = state_decl[template.state_ref]
            binding = _id("hbm_binding", {
                "state": template.state_ref, "die": rank, "address": hbm_offset,
            })
            abi = StateABI.create(
                state_ref=template.state_ref, hbm_binding_ref=binding,
                kind=StateKind.TRAINABLE_PARAMETER,
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=PersistentStateAccess.READ_WRITE,
                shape=declaration.shape, dtype=declaration.dtype,
                layout=declaration.layout, die_id=rank, address=hbm_offset,
                size_bytes=template.weight_bytes,
                alignment_bytes=spaces[rank].alignment_bytes,
            )
            state_abis[(rank, template.state_ref)] = abi
            hbm_symbols[(rank, template.state_ref)] = _hbm_symbol(binding)
            hbm_offset += template.weight_bytes
        if hbm_offset > spaces[rank].base_address + spaces[rank].size_bytes:
            raise SchemaError("rank trainable states exceed HBM", path=f"hbm_address_spaces[{rank}]")

    absolute = {key: _symbol(abi) for key, abi in buffers.items()}
    labels = {key: _label_symbol(abi) for key, abi in buffers.items()}
    region_symbol_by_ref = {}
    region_symbols = {}
    for core in cores_tuple:
        region_ref = regions[core].id
        region_symbol_by_ref.setdefault(
            region_ref, _region_symbol(region_ref, cores_tuple[0])
        )
        region_symbols[core] = region_symbol_by_ref[region_ref]
    records = {core: [] for core in cores_tuple}
    address_relocs = {core: [] for core in cores_tuple}
    runtime_relocs = {core: [] for core in cores_tuple}
    address_uses = []
    state_uses = []

    def add_address(core, record_index, operand, symbol, abi, addend=0, tensor_slice=None,
                    symbol_kind=ProgramSymbolKind.ABSOLUTE_ADDRESS):
        address_relocs[core].append(AddressRelocation(
            record_index, operand, symbol_kind, symbol.id, addend,
        ))
        address_uses.append((core, record_index, operand, abi,
                             abi.tensor_slice if tensor_slice is None else tensor_slice))

    def add_runtime(core, record_index, field, symbol):
        runtime_relocs[core].append(RuntimeRelocation(record_index, field, symbol.id))

    def append_bind(core, action_id, inputs, output):
        index = len(records[core])
        operands = [RecordOperand.literal("input_count", len(inputs))]
        for slot in range(16):
            operand = SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + slot)
            if slot < len(inputs):
                symbol, _abi = inputs[slot]
                operands.append(RecordOperand.address(f"input_label_{slot}", operand, symbol.id))
            else:
                operands.append(RecordOperand.literal(f"input_label_{slot}", 0))
        operands.append(RecordOperand.address(
            "output_label", SemanticOperandId.SRAM_BIND_OUTPUT, output[0].id,
        ))
        records[core].append(RelocatableRecord(action_id, RecordOpcode.SRAM_BIND, tuple(operands)))
        for slot, (symbol, abi) in enumerate(inputs):
            add_address(core, index,
                        SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + slot),
                        symbol, abi, symbol_kind=ProgramSymbolKind.SRAM_LABEL)
        add_address(core, index, SemanticOperandId.SRAM_BIND_OUTPUT,
                    output[0], output[1], symbol_kind=ProgramSymbolKind.SRAM_LABEL)

    peer_symbols = {}
    runtime_symbols = {}
    runtime_definitions = []
    def peer(rank):
        if rank not in peer_symbols:
            symbol = _runtime("runtime_core", {"plan": plan.id, "rank": rank},
                              RuntimeSymbolKind.RUNTIME_CORE, f"rank.{rank}")
            peer_symbols[rank] = symbol
            runtime_symbols[symbol.id] = symbol
            runtime_definitions.append(RuntimeSymbolDefinition(
                symbol, (core_by_rank[rank],), None, None,
            ))
        return peer_symbols[rank]

    sync_actions = tuple(
        action for action in plan.rank_actions
        if action.kind is FlexibleDenseTrainActionKind.GRADIENT_SYNC
    )
    outgoing = {}
    incoming = {}
    if plan.spec.dp_degree > 1:
        send_action_by_flow = {
            (action.rank, action.send_peer_rank, action.state_ref): action
            for action in sync_actions if action.send_peer_rank is not None
        }
        recv_action_by_flow = {
            (action.receive_peer_rank, action.rank, action.state_ref): action
            for action in sync_actions if action.receive_peer_rank is not None
        }
        if set(send_action_by_flow) != set(recv_action_by_flow):
            raise SchemaError(
                "gradient sync send/receive flows must close exactly",
                path="plan.rank_actions",
            )
        for (source, destination, state_ref), send_action in send_action_by_flow.items():
            recv_action = recv_action_by_flow[(source, destination, state_ref)]
            flow = f"{state_ref}:{source}->{destination}"
            endpoints = tuple(sorted(
                (core_by_rank[source], core_by_rank[destination]),
                key=lambda item: (item.die_id, item.local_core_id),
            ))
            fsm = _runtime("dte_fsm", {"plan": plan.id, "flow": flow},
                           RuntimeSymbolKind.DTE_FSM, flow)
            token = _runtime("dte_token", {"plan": plan.id, "flow": flow},
                             RuntimeSymbolKind.DTE_TOKEN, recv_action.id)
            runtime_symbols[fsm.id] = fsm
            runtime_symbols[token.id] = token
            runtime_definitions.append(RuntimeSymbolDefinition(
                fsm, endpoints, send_action.id, recv_action.id,
            ))
            runtime_definitions.append(RuntimeSymbolDefinition(
                token, (core_by_rank[destination],), recv_action.id, recv_action.id,
            ))
            outgoing[send_action.id] = (fsm, peer(destination), token)
            incoming[recv_action.id] = (fsm, peer(source), token)

    for rank, core in enumerate(cores_tuple):
        first_template = local_templates[rank][0]
        for action in actions_by_rank[rank]:
            template = None if action.state_ref is None else templates_by_state[action.state_ref]
            if action.kind is FlexibleDenseTrainActionKind.PARAMETER_LOAD:
                for role in ("weight", "gradient"):
                    abi = buffers[(rank, template.state_ref, role)]
                    label = labels[(rank, template.state_ref, role)]
                    index = len(records[core])
                    records[core].append(RelocatableRecord(action.id, RecordOpcode.SRAM_ALLOC_AT, (
                        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbols[core].id),
                        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.id),
                        RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
                        RecordOperand.literal("size_bytes", abi.size_bytes),
                        RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
                        RecordOperand.literal("lifetime", 0), RecordOperand.literal("spillable", False),
                    )))
                    add_address(core, index, SemanticOperandId.REGION_NAME, region_symbols[core], abi,
                                symbol_kind=ProgramSymbolKind.SRAM_REGION)
                    add_address(core, index, SemanticOperandId.LABEL_SYMBOL, label, abi,
                                symbol_kind=ProgramSymbolKind.SRAM_LABEL)
                weight = buffers[(rank, template.state_ref, "weight")]
                index = len(records[core])
                records[core].append(RelocatableRecord(action.id, RecordOpcode.LSU_LOAD, (
                    RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS,
                                          hbm_symbols[(rank, template.state_ref)].id),
                    RecordOperand.literal("size_bytes", template.weight_bytes),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS,
                                          absolute[(rank, template.state_ref, "weight")].id),
                )))
                address_relocs[core].append(AddressRelocation(index, SemanticOperandId.HBM_ADDRESS,
                    ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm_symbols[(rank, template.state_ref)].id, 0))
                state_uses.append((core, index, state_abis[(rank, template.state_ref)]))
                add_address(core, index, SemanticOperandId.DESTINATION_ADDRESS,
                            absolute[(rank, template.state_ref, "weight")], weight)
            elif action.kind in (FlexibleDenseTrainActionKind.BACKWARD,
                                  FlexibleDenseTrainActionKind.WEIGHT_GRADIENT):
                selected = first_template if template is None else template
                weight = buffers[(rank, selected.state_ref, "weight")]
                gradient = buffers[(rank, selected.state_ref, "gradient")]
                append_bind(core, action.id,
                    ((labels[(rank, selected.state_ref, "weight")], weight),),
                    (labels[(rank, selected.state_ref, "gradient")], gradient))
                index = len(records[core])
                records[core].append(_matmul(action.id,
                    absolute[(rank, selected.state_ref, "weight")],
                    absolute[(rank, selected.state_ref, "gradient")]))
                add_address(core, index, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                            absolute[(rank, selected.state_ref, "weight")], weight)
                add_address(core, index, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                            absolute[(rank, selected.state_ref, "weight")], weight)
                add_address(core, index, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                            absolute[(rank, selected.state_ref, "gradient")], gradient)
            elif action.kind is FlexibleDenseTrainActionKind.GRADIENT_SYNC:
                gradient = buffers[(rank, template.state_ref, "gradient")]
                symbol = absolute[(rank, template.state_ref, "gradient")]
                is_send = action.send_peer_rank is not None
                if is_send:
                    send_fsm, send_peer, _ = outgoing[action.id]
                else:
                    recv_fsm, recv_peer, token = incoming[action.id]
                # The currently executed LOCAL_REDUCE wire supports one exact
                # FP32 pair (2 x 512 elements).  The synthetic timing carrier
                # pads every logical gradient into that fixed scratch shape;
                # SGD still consumes only the template's logical prefix.
                transport_bytes = 2048
                # This is an explicitly paired P2P byte transfer, not a
                # collective DTE issue.  The strict ABI therefore requires
                # the whole collective key (including epoch) to be zero.
                epoch = 0
                if is_send:
                    index = len(records[core])
                    records[core].append(RelocatableRecord(action.id, RecordOpcode.DTE_SEND, (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0),
                        RecordOperand.literal("completion", 1), RecordOperand.literal("datatype", 0),
                        RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, send_fsm.id),
                        RecordOperand.literal("token", 0), RecordOperand.literal("length_bytes", transport_bytes),
                        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, symbol.id),
                        RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, send_peer.id),
                        RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
                        RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
                        RecordOperand.literal("epoch", epoch),
                    )))
                    add_address(core, index, SemanticOperandId.SOURCE_ADDRESS, symbol, gradient)
                    add_runtime(core, index, RuntimeOperandField.DTE_FSM, send_fsm)
                    add_runtime(core, index, RuntimeOperandField.PEER_CORE, send_peer)
                    continue
                index = len(records[core])
                is_reduce = (
                    action.gradient_sync_role
                    is FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE
                )
                recv_addend = transport_bytes if is_reduce else 0
                recv_slice = TensorSlice(
                    gradient.value_id,
                    (recv_addend // 4,),
                    (transport_bytes // 4,),
                )
                records[core].append(RelocatableRecord(action.id, RecordOpcode.DTE_RECV, (
                    RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0),
                    RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                    RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, recv_fsm.id),
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
                    RecordOperand.literal("length_bytes", transport_bytes),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, symbol.id),
                    RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, recv_peer.id),
                    RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
                    RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
                    RecordOperand.literal("epoch", epoch),
                )))
                add_address(core, index, SemanticOperandId.DESTINATION_ADDRESS,
                            symbol, gradient, recv_addend, recv_slice)
                add_runtime(core, index, RuntimeOperandField.DTE_FSM, recv_fsm)
                add_runtime(core, index, RuntimeOperandField.DTE_TOKEN, token)
                add_runtime(core, index, RuntimeOperandField.PEER_CORE, recv_peer)
                index = len(records[core])
                records[core].append(RelocatableRecord(action.id, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
                )))
                add_runtime(core, index, RuntimeOperandField.DTE_TOKEN, token)
                if not is_reduce:
                    continue
                index = len(records[core])
                records[core].append(RelocatableRecord(action.id, RecordOpcode.LOCAL_REDUCE, (
                    RecordOperand.literal("input_dtype", 1), RecordOperand.literal("accumulator_dtype", 1),
                    RecordOperand.literal("output_dtype", 1), RecordOperand.literal("reduce_op", 1),
                    RecordOperand.literal("rounding", 0), RecordOperand.literal("order", 0),
                    RecordOperand.literal("input_count", 2),
                    RecordOperand.literal("element_count", 512),
                    RecordOperand.literal("input_stride_bytes", transport_bytes),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, symbol.id),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, symbol.id),
                )))
                add_address(core, index, SemanticOperandId.SOURCE_ADDRESS,
                            symbol, gradient, 0, gradient.tensor_slice)
                add_address(core, index, SemanticOperandId.DESTINATION_ADDRESS,
                            symbol, gradient, 0, TensorSlice(gradient.value_id, (0,),
                                                           (transport_bytes // 4,)))
            elif action.kind is FlexibleDenseTrainActionKind.SGD_UPDATE:
                weight = buffers[(rank, template.state_ref, "weight")]
                gradient = buffers[(rank, template.state_ref, "gradient")]
                append_bind(core, action.id, (
                    (labels[(rank, template.state_ref, "weight")], weight),
                    (labels[(rank, template.state_ref, "gradient")], gradient),
                ), (labels[(rank, template.state_ref, "weight")], weight))
                index = len(records[core])
                weight_symbol = absolute[(rank, template.state_ref, "weight")]
                gradient_symbol = absolute[(rank, template.state_ref, "gradient")]
                records[core].append(RelocatableRecord(action.id, RecordOpcode.SGD_UPDATE, (
                    RecordOperand.literal("weight_datatype", 1), RecordOperand.literal("gradient_datatype", 3),
                    RecordOperand.literal("output_datatype", 1), RecordOperand.literal("rounding", 0),
                    RecordOperand.address("weight_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, weight_symbol.id),
                    RecordOperand.address("gradient_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, gradient_symbol.id),
                    RecordOperand.address("updated_weight_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, weight_symbol.id),
                    RecordOperand.literal("element_count", template.weight_bytes // 2),
                    RecordOperand.literal("learning_rate_f64_bits", struct.unpack("<Q", struct.pack("<d", plan.spec.learning_rate))[0]),
                    RecordOperand.literal("momentum_f64_bits", 0),
                )))
                add_address(core, index, SemanticOperandId.COMPUTE_INPUT_ADDRESS, weight_symbol, weight)
                add_address(core, index, SemanticOperandId.COMPUTE_DATA_ADDRESS, gradient_symbol, gradient)
                add_address(core, index, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, weight_symbol, weight)
            elif action.kind is FlexibleDenseTrainActionKind.PARAMETER_STORE:
                weight = buffers[(rank, template.state_ref, "weight")]
                symbol = absolute[(rank, template.state_ref, "weight")]
                index = len(records[core])
                records[core].append(RelocatableRecord(action.id, RecordOpcode.LSU_STORE, (
                    RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS,
                                          hbm_symbols[(rank, template.state_ref)].id),
                    RecordOperand.literal("size_bytes", template.weight_bytes),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, symbol.id),
                )))
                address_relocs[core].append(AddressRelocation(index, SemanticOperandId.HBM_ADDRESS,
                    ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm_symbols[(rank, template.state_ref)].id, 0))
                state_uses.append((core, index, state_abis[(rank, template.state_ref)]))
                add_address(core, index, SemanticOperandId.SOURCE_ADDRESS, symbol, weight)
                for role in ("weight", "gradient"):
                    abi = buffers[(rank, template.state_ref, role)]
                    label = labels[(rank, template.state_ref, role)]
                    free_index = len(records[core])
                    records[core].append(RelocatableRecord(action.id, RecordOpcode.SRAM_FREE, (
                        RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.id),
                    )))
                    add_address(core, free_index, SemanticOperandId.SYMBOL, label, abi,
                                symbol_kind=ProgramSymbolKind.SRAM_LABEL)
            else:
                raise SchemaError("unexpected non-forward action", path="plan.rank_actions")

    program_symbols = tuple(sorted({item.id: item for item in (
        *absolute.values(), *labels.values(), *hbm_symbols.values(),
        *region_symbols.values(),
    )}.values(), key=lambda item: item.id))
    fragment = CommandFragment.create(
        producer_pass="flexible_dense_backward_lowering",
        source_global_dag_id=global_dag.id, kind=FragmentKind.STATE_IO,
        claimed_action_ids=tuple(sorted(action.id for values in actions_by_rank.values() for action in values)),
        core_streams=tuple(CoreFragmentStream(
            core, tuple(records[core]),
            tuple(sorted(runtime_relocs[core], key=lambda item: (
                item.record_index, tuple(RuntimeOperandField).index(item.field),
            ))),
            tuple(sorted(address_relocs[core], key=lambda item: (item.record_index, int(item.operand_id)))),
        ) for core in cores_tuple),
        runtime_symbols=tuple(sorted(runtime_symbols.values(), key=lambda item: item.id)),
        program_symbols=program_symbols,
        buffer_abi=tuple(sorted(buffers.values(), key=lambda item: item.id)),
        state_abi=tuple(sorted(state_abis.values(), key=lambda item: item.id)),
    )
    definitions = []
    buffer_by_symbol = {
        absolute[key].id: abi for key, abi in buffers.items()
    }
    label_by_symbol = {
        labels[key].id: abi for key, abi in buffers.items()
    }
    state_by_symbol = {
        hbm_symbols[key].id: abi for key, abi in state_abis.items()
    }
    region_core_by_symbol = {
        value.id: core for core, value in region_symbols.items()
    }
    for ordinal, symbol in enumerate(program_symbols):
        buffer = buffer_by_symbol.get(symbol.id)
        label = label_by_symbol.get(symbol.id)
        state = state_by_symbol.get(symbol.id)
        region_core = region_core_by_symbol.get(symbol.id)
        if region_core is not None:
            region = regions[region_core]
            owners = tuple(core for core in cores_tuple if region_symbols[core].id == symbol.id)
            value, size, name = region.base_bytes, region.size_bytes, region.name
        elif label is not None:
            value, size, name, owners = 0, 0, f"fd_bwd_label_{ordinal:05d}", (label.logical_core,)
        elif buffer is not None:
            region = regions[buffer.logical_core]
            value, size, name, owners = region.base_bytes + buffer.region_offset_bytes, buffer.size_bytes, f"fd_bwd_abs_{ordinal:05d}", (buffer.logical_core,)
        else:
            assert state is not None
            value, size, name, owners = state.address, state.size_bytes, f"fd_bwd_hbm_{ordinal:05d}", (core_by_rank[state.die_id],)
        definitions.append(ProgramSymbolDefinition(symbol, name, value, size, owners))
    address_bindings = tuple(sorted((AddressOperandBinding(
        fragment.id, core, index, operand, (abi.id,), (tensor_slice,)
    ) for core, index, operand, abi, tensor_slice in address_uses),
        key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id,
                          item.fragment_id, item.fragment_record_index, int(item.operand_id))))
    state_bindings = tuple(sorted((StateOperandBinding(
        fragment.id, core, index, SemanticOperandId.HBM_ADDRESS, abi.id
    ) for core, index, abi in state_uses),
        key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id,
                          item.fragment_id, item.fragment_record_index, int(item.operand_id))))
    inputs = (
        ManifestInputDigest(ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_IR, ir.id, ir.schema_version, ir.digest),
        ManifestInputDigest(ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_PROJECTION, projection.id, projection.schema_version, projection.digest),
        ManifestInputDigest(ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_SCHEDULE, schedule.id, schedule.schema_version, schedule.digest),
        ManifestInputDigest(ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_GLOBAL, global_dag.id, global_dag.schema_version, global_dag.digest),
        ManifestInputDigest(ManifestInputKind.COMMAND_FRAGMENT, fragment.id, fragment.schema_version, canonical_digest(fragment)),
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="flexible_dense_backward_linker", capabilities=0,
        source_ir1_id=ir.id, source_projection_id=projection.id,
        source_schedule_set_id=schedule.id, source_global_dag_id=global_dag.id,
        input_digests=tuple(sorted(inputs, key=lambda item: (item.kind.value, item.artifact_id))),
        fragments=(fragment,), fragment_interfaces=(FragmentInterface(
            fragment.id, (), tuple(sorted(runtime_symbols)), (),
            tuple(symbol.id for symbol in program_symbols), (), (),
        ),),
        core_bindings=tuple(CoreRuntimeBinding(
            core, core_specs[core].id, core_specs[core].runtime_core_id,
            core_specs[core].sram_profile_ref,
        ) for core in cores_tuple),
        core_streams=tuple(LinkedCoreStream(
            core, core_specs[core].runtime_core_id,
            tuple(LinkedRecordRef(fragment.id, index, record.source_global_action_id)
                  for index, record in enumerate(records[core])),
        ) for core in cores_tuple),
        runtime_symbol_definitions=tuple(sorted(runtime_definitions, key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(definitions, key=lambda item: item.symbol.id)),
        address_operand_bindings=address_bindings,
        state_operand_bindings=state_bindings, core_groups=(),
        envelope=ProgramControlEnvelope(
            cores_tuple, (), cores_tuple, cores_tuple, cores_tuple,
            EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    return FlexibleDenseBackwardLinkedProgram.create(
        plan=plan, forward_lineage=forward.linked_forward,
        backward_ir=ir, backward_projection=projection,
        backward_schedule=schedule, backward_global_dag=global_dag,
        fabric=fabric, hbm_address_spaces=spaces, manifest=manifest,
        record_count=sum(len(items) for items in records.values()),
        runtime_verified=False,
    )


__all__ = ["materialize_flexible_dense_backward_multi"]
