"""Build the explicit timing-only ProgramArtifact host IO contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    BufferABI,
    CommandFragment,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RegionManifest,
    SemanticOperandId,
    StateABI,
)
from ..schema.common import DType
from ..schema.global_action import ActionBufferUse, GlobalAction, LogicalCoreRef
from ..schema.ir0 import (
    CrossEntropyBackwardWorkload,
    CrossEntropyForwardWorkload,
    OpKind,
)
from ..schema.ir2 import (
    BufferAccess,
    BufferOwnership,
    BufferUseRole,
    StateUseAccess,
)
from ..schema.n6 import LinkedProgramProfile, Stage4LinkedProgram
from ..schema.lite_train_n6 import S2LiteTrainLinkedProgram
from ..schema.lite_train_rooted_ar_n6 import (
    RootedArExecutableKind,
    S2LiteRootedArLinkedProgram,
)
from ..schema.lite_train_dp4_n6 import (
    Dp4TreeExecutableKind,
    S2LiteDp4TreeArLinkedProgram,
)
from ..schema.lite_moe_execution import LiteMoeBufferAccess
from ..schema.lite_moe_dp4_execution import LiteMoeDp4BufferAccess
from ..schema.lite_moe_dp4_n6 import (
    LiteMoeDp4BackwardLinkedProgram,
    LiteMoeDp4InferLinkedProgram,
    LiteMoeDp4TrainForwardLinkedProgram,
)
from ..schema.lite_moe_n6 import LiteMoeLinkedProgram
from ..schema.lite_moe_backward_n6 import LiteMoeBackwardLinkedProgram
from ..schema.persistent_state import PersistentStateAccess, StateKind
from ..schema.program_io import (
    ProgramBlob,
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramIoTargetKind,
    ProgramOutputCapture,
    ProgramOutputComparison,
    ProgramOutputProbe,
    ProgramSramInitialization,
    ProgramSramTarget,
)
from ..schema.train_n6 import TrainLinkedProgram
from ..schema.swizzle_standard import SwizzleStandardLinkedProgram
from ..schema.swizzle_ir2 import admits_wang_4rank_packed_layout
from ..schema.swizzle_unfused_standard import (
    UnfusedComparisonStandardLinkedProgram,
)
from .swizzle_program_io import (
    swizzle_semantic_uses,
    swizzle_terminal_abi_ids,
    swizzle_terminal_value_ids,
)
from .unfused_comparison_program_io import (
    unfused_comparison_semantic_uses,
    unfused_comparison_terminal_abi_ids,
    unfused_comparison_terminal_value_ids,
)


_READ_ROLES = {
    BufferUseRole.COMP_INPUT,
    BufferUseRole.SEND_SOURCE,
    BufferUseRole.REDUCE_INPUT,
    BufferUseRole.LOCAL_COPY_SOURCE,
    BufferUseRole.DMA_SOURCE,
}
_WRITE_ROLES = {
    BufferUseRole.COMP_OUTPUT,
    BufferUseRole.RECV_DESTINATION,
    BufferUseRole.REDUCE_OUTPUT,
    BufferUseRole.LOCAL_COPY_DESTINATION,
    BufferUseRole.DMA_DESTINATION,
}


@dataclass(frozen=True, slots=True)
class _SemanticUse:
    action_tuple_index: int
    use_tuple_index: int
    action: GlobalAction
    use: ActionBufferUse
    replica_index: int

    @property
    def order_key(self) -> tuple[int, int, int]:
        assert self.action.core_order_index is not None
        return (
            self.action.core_order_index,
            self.action_tuple_index,
            self.use_tuple_index,
        )


@dataclass(frozen=True, slots=True)
class _LiteSemanticUse:
    action_tuple_index: int
    use_tuple_index: int
    action: object
    access: BufferAccess
    role: BufferUseRole
    operand_index: int
    order_index: int
    replica_index: int = 0

    @property
    def use(self) -> "_LiteSemanticUse":
        return self

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (self.order_index, self.action_tuple_index, self.use_tuple_index)


@dataclass(frozen=True, slots=True)
class _ResolvedAbi:
    abi: BufferABI
    runtime_core_id: int
    symbol_index: int
    symbol_definition: ProgramSymbolDefinition
    uses: tuple[_SemanticUse, ...]

@dataclass(frozen=True, slots=True)
class _ResolvedStateAbi:
    abi: StateABI
    symbol_index: int
    symbol_definition: ProgramSymbolDefinition
    uses: tuple[tuple[int, StateUseAccess], ...]

    @property
    def first_access(self) -> StateUseAccess:
        return self.uses[0][1]


def _leaf(fragment: CommandFragment | RegionManifest) -> CommandFragment:
    return fragment.fragment if type(fragment) is RegionManifest else fragment


LinkedProgramSource = (
    LinkedProgramProfile
    | Stage4LinkedProgram
    | TrainLinkedProgram
    | S2LiteTrainLinkedProgram
    | S2LiteRootedArLinkedProgram
    | S2LiteDp4TreeArLinkedProgram
    | LiteMoeLinkedProgram
    | LiteMoeBackwardLinkedProgram
    | LiteMoeDp4InferLinkedProgram
    | LiteMoeDp4TrainForwardLinkedProgram
    | LiteMoeDp4BackwardLinkedProgram
    | SwizzleStandardLinkedProgram
    | UnfusedComparisonStandardLinkedProgram
)


_LINKED_PROGRAM_SOURCE_TYPES = (
    LinkedProgramProfile,
    Stage4LinkedProgram,
    TrainLinkedProgram,
    S2LiteTrainLinkedProgram,
    S2LiteRootedArLinkedProgram,
    S2LiteDp4TreeArLinkedProgram,
    LiteMoeLinkedProgram,
    LiteMoeBackwardLinkedProgram,
    LiteMoeDp4InferLinkedProgram,
    LiteMoeDp4TrainForwardLinkedProgram,
    LiteMoeDp4BackwardLinkedProgram,
    SwizzleStandardLinkedProgram,
    UnfusedComparisonStandardLinkedProgram,
)


def _lowering_contexts(
    source: LinkedProgramSource,
) -> tuple[tuple[int, object], ...]:
    if type(source) is TrainLinkedProgram:
        return tuple(
            (replica.replica_index, replica.lowering_context)
            for replica in source.source.replicas
        )
    if type(source) is S2LiteTrainLinkedProgram:
        return ((0, source.source.lowering_context),)
    if type(source) in (
        S2LiteRootedArLinkedProgram,
        S2LiteDp4TreeArLinkedProgram,
    ):
        return tuple(enumerate(source.source.intent.lowering_contexts))
    if type(source) is LiteMoeLinkedProgram:
        raise SchemaError(
            "S3-Lite uses its dedicated execution carrier",
            path="source",
        )
    if type(source) is LiteMoeBackwardLinkedProgram:
        raise SchemaError(
            "S3-Lite backward uses its dedicated execution carrier",
            path="source",
        )
    if type(source) in (
        LiteMoeDp4InferLinkedProgram,
        LiteMoeDp4TrainForwardLinkedProgram,
        LiteMoeDp4BackwardLinkedProgram,
    ):
        raise SchemaError(
            "S3-Lite DP4 uses its dedicated execution carrier",
            path="source",
        )
    return ((0, source.lowering_context),)


_LITE_TRAIN_SOURCE_TYPES = (
    S2LiteTrainLinkedProgram,
    S2LiteRootedArLinkedProgram,
    S2LiteDp4TreeArLinkedProgram,
)
_TRAIN_SOURCE_TYPES = (TrainLinkedProgram, *_LITE_TRAIN_SOURCE_TYPES)


def _unique_abis(source: LinkedProgramSource) -> dict[str, BufferABI]:
    by_id: dict[str, BufferABI] = {}
    by_binding: dict[tuple[str, str], BufferABI] = {}
    for linked in source.manifest.fragments:
        for abi in _leaf(linked).buffer_abi:
            previous = by_id.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError(
                    "conflicting shared BufferABI definition",
                    path="source.manifest.fragments",
                )
            key = (abi.schedule_id, abi.binding_id)
            previous = by_binding.setdefault(key, abi)
            if previous != abi:
                raise SchemaError(
                    "one schedule binding resolves to conflicting BufferABIs",
                    path="source.manifest.fragments",
                )
    if not by_id:
        raise SchemaError(
            "timing ProgramIo requires BufferABI entries",
            path="source.manifest.fragments",
        )
    return by_id


def _semantic_uses(
    source: LinkedProgramSource,
    abis: dict[str, BufferABI],
) -> dict[str, tuple[_SemanticUse, ...]]:
    if type(source) is SwizzleStandardLinkedProgram:
        return swizzle_semantic_uses(source, abis)  # type: ignore[return-value]
    if type(source) is UnfusedComparisonStandardLinkedProgram:
        return unfused_comparison_semantic_uses(source, abis)  # type: ignore[return-value]
    if type(source) in (
        LiteMoeLinkedProgram,
        LiteMoeDp4InferLinkedProgram,
        LiteMoeDp4TrainForwardLinkedProgram,
    ):
        by_binding = {abi.binding_id: abi for abi in abis.values()}
        if type(source) is LiteMoeDp4InferLinkedProgram:
            execution = source.source.source
        elif type(source) is LiteMoeDp4TrainForwardLinkedProgram:
            execution = source.source.source.forward
        else:
            execution = source.source
        placements = {
            placement.task_ref: placement
            for placement in execution.schedule.placements
        }
        collected: dict[str, list[_LiteSemanticUse]] = {
            abi_id: [] for abi_id in abis
        }
        for action_index, action in enumerate(execution.global_dag.actions):
            placement = placements.get(action.task_ref)
            if placement is None:
                raise SchemaError(
                    "S3-Lite action has no schedule placement",
                    path="source.source.global_dag.actions",
                )
            for use_index, use in enumerate(action.buffer_uses):
                abi = by_binding.get(use.binding_ref)
                if abi is None:
                    raise SchemaError(
                        "S3-Lite buffer use has no manifest BufferABI",
                        path="source.source.global_dag.actions",
                    )
                access = (
                    BufferAccess.READ
                    if use.access in (
                        LiteMoeBufferAccess.READ,
                        LiteMoeDp4BufferAccess.READ,
                    )
                    else BufferAccess.WRITE
                )
                collected[abi.id].append(_LiteSemanticUse(
                    action_index,
                    use_index,
                    action,
                    access,
                    BufferUseRole.COMP_INPUT if access is BufferAccess.READ else BufferUseRole.COMP_OUTPUT,
                    0,
                    placement.ordinal,
                ))
        if type(source) is LiteMoeDp4TrainForwardLinkedProgram:
            action_order: dict[str, tuple[LogicalCoreRef, int]] = {}
            for stream in source.manifest.core_streams:
                for order_index, record_ref in enumerate(stream.records):
                    previous = action_order.get(record_ref.source_global_action_id)
                    if previous is None or order_index < previous[1]:
                        action_order[record_ref.source_global_action_id] = (
                            stream.logical_core,
                            order_index,
                        )
            base = len(execution.global_dag.actions)
            for copy_index, copy in enumerate(source.source.source.tape_copies):
                order = action_order.get(copy.id)
                source_abi = by_binding.get(copy.source_buffer_ref)
                destination_abi = by_binding.get(copy.destination_buffer_ref)
                if (
                    order is None
                    or source_abi is None
                    or destination_abi is None
                    or source_abi.logical_core != order[0]
                    or destination_abi.logical_core != order[0]
                    or source_abi.size_bytes != copy.bytes
                    or destination_abi.size_bytes != copy.bytes
                ):
                    raise SchemaError(
                        "DP4 tape copy lacks exact linked BufferABI/core closure",
                        path="source.source.source.tape_copies",
                    )
                collected[source_abi.id].append(_LiteSemanticUse(
                    base + copy_index,
                    0,
                    copy,
                    BufferAccess.READ,
                    BufferUseRole.LOCAL_COPY_SOURCE,
                    0,
                    order[1],
                ))
                collected[destination_abi.id].append(_LiteSemanticUse(
                    base + copy_index,
                    1,
                    copy,
                    BufferAccess.WRITE,
                    BufferUseRole.LOCAL_COPY_DESTINATION,
                    0,
                    order[1],
                ))
        result = {}
        for abi_id, uses in collected.items():
            if not uses:
                raise SchemaError(
                    "S3-Lite BufferABI has no semantic use",
                    path="source.manifest.fragments",
                )
            ordered = tuple(sorted(uses, key=lambda item: item.order_key))
            ownership = abis[abi_id].ownership
            valid_first = (
                ordered[0].access is BufferAccess.READ
                if ownership is BufferOwnership.BORROWED
                else ordered[0].access is BufferAccess.WRITE
                or all(item.access is BufferAccess.READ for item in ordered)
            )
            if not valid_first:
                raise SchemaError(
                    "S3-Lite BufferABI first-use ownership mismatch",
                    path="source.source.global_dag.actions",
                )
            result[abi_id] = ordered
        return result  # type: ignore[return-value]
    if type(source) in (
        LiteMoeBackwardLinkedProgram,
        LiteMoeDp4BackwardLinkedProgram,
    ):
        fragments = {
            _leaf(fragment).id: _leaf(fragment)
            for fragment in source.manifest.fragments
        }
        record_order: dict[tuple[str, int, LogicalCoreRef], int] = {}
        for stream in source.manifest.core_streams:
            for order_index, record_ref in enumerate(stream.records):
                record_order[
                    (record_ref.fragment_id, record_ref.fragment_record_index, stream.logical_core)
                ] = order_index
        collected: dict[str, list[_LiteSemanticUse]] = {
            abi_id: [] for abi_id in abis
        }
        semantic = {
            (RecordOpcode.LSU_LOAD, SemanticOperandId.DESTINATION_ADDRESS):
                (BufferAccess.WRITE, BufferUseRole.DMA_DESTINATION, 0),
            (RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS):
                (BufferAccess.READ, BufferUseRole.SEND_SOURCE, 0),
            (RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS):
                (BufferAccess.WRITE, BufferUseRole.RECV_DESTINATION, 0),
            (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS):
                (BufferAccess.READ, BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS):
                (BufferAccess.READ, BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS):
                (BufferAccess.WRITE, BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS):
                (BufferAccess.READ, BufferUseRole.REDUCE_INPUT, 0),
            (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS):
                (BufferAccess.WRITE, BufferUseRole.REDUCE_OUTPUT, 0),
            (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_INPUT_ADDRESS):
                (BufferAccess.READ, BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_DATA_ADDRESS):
                (BufferAccess.READ, BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS):
                (BufferAccess.WRITE, BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.LSU_STORE, SemanticOperandId.SOURCE_ADDRESS):
                (BufferAccess.READ, BufferUseRole.DMA_SOURCE, 0),
        }
        overlay = (
            source.source.source
            if type(source) is LiteMoeDp4BackwardLinkedProgram
            else source.source.overlay
        )
        sgd_ids = {item.id for item in overlay.sgd_stores}
        use_index = 0
        for binding in source.manifest.address_operand_bindings:
            fragment = fragments[binding.fragment_id]
            stream = next(
                item for item in fragment.core_streams
                if item.logical_core == binding.logical_core
            )
            record = stream.records[binding.fragment_record_index]
            rule = semantic.get((record.opcode, binding.operand_id))
            if (
                rule is None
                and record.opcode is RecordOpcode.SRAM_BIND
                and record.source_global_action_id in sgd_ids
                and binding.operand_id is SemanticOperandId.SRAM_BIND_OUTPUT
            ):
                rule = (BufferAccess.WRITE, BufferUseRole.COMP_OUTPUT, 0)
            if rule is None:
                continue
            access, role, operand_index = rule
            order_index = record_order.get(
                (binding.fragment_id, binding.fragment_record_index, binding.logical_core)
            )
            if order_index is None:
                raise SchemaError(
                    "backward address binding lacks one linked-core position",
                    path="source.manifest.address_operand_bindings",
                )
            for abi_id in binding.buffer_abi_ids:
                if abi_id not in collected:
                    raise SchemaError(
                        "backward address binding references unknown BufferABI",
                        path="source.manifest.address_operand_bindings",
                    )
                collected[abi_id].append(_LiteSemanticUse(
                    use_index,
                    0,
                    record,
                    access,
                    role,
                    operand_index,
                    order_index,
                ))
                use_index += 1
        by_storage: dict[str, list[BufferABI]] = {}
        for abi in abis.values():
            by_storage.setdefault(abi.storage_id, []).append(abi)
        result: dict[str, tuple[_LiteSemanticUse, ...]] = {}
        for abi_id, uses in collected.items():
            abi = abis[abi_id]
            if not uses and abi.ownership is BufferOwnership.OWNED:
                uses = [
                    use
                    for sibling in by_storage[abi.storage_id]
                    if sibling.ownership is BufferOwnership.ALIASED
                    for use in collected[sibling.id]
                ]
            if not uses:
                raise SchemaError(
                    "S3-Lite backward BufferABI has no exact record semantic use",
                    path="source.manifest.fragments",
                )
            ordered = tuple(sorted(uses, key=lambda item: item.order_key))
            expected_first = (
                BufferAccess.READ
                if abi.ownership is BufferOwnership.BORROWED
                else BufferAccess.WRITE
            )
            if ordered[0].access is not expected_first:
                raise SchemaError(
                    "S3-Lite backward BufferABI first-use ownership mismatch",
                    path="source.manifest.fragments",
                )
            result[abi_id] = ordered
        return result  # type: ignore[return-value]
    if type(source) in (
        S2LiteRootedArLinkedProgram,
        S2LiteDp4TreeArLinkedProgram,
    ):
        by_binding = {
            (abi.schedule_id, abi.binding_id): abi
            for abi in abis.values()
        }
        collected: dict[str, list[_LiteSemanticUse]] = {
            abi_id: [] for abi_id in abis
        }
        action_order: dict[str, tuple[LogicalCoreRef, int]] = {}
        for stream in source.manifest.core_streams:
            for record_index, record_ref in enumerate(stream.records):
                previous = action_order.get(record_ref.source_global_action_id)
                if previous is not None and previous[0] != stream.logical_core:
                    raise SchemaError(
                        "rooted-AR action appears on multiple core streams",
                        path="source.manifest.core_streams",
                    )
                if previous is None or record_index < previous[1]:
                    action_order[record_ref.source_global_action_id] = (
                        stream.logical_core,
                        record_index,
                    )

        flattened_action_index = 0
        for replica_index, context in _lowering_contexts(source):
            for action_index, action in enumerate(context.global_dag.actions):
                order = action_order.get(action.id)
                if order is None or order[0] != action.logical_core:
                    raise SchemaError(
                        "rooted-AR local action lacks its exact linked-core position",
                        path=(
                            "source.source.intent.lowering_contexts"
                            f"[{replica_index}].global_dag.actions[{action_index}]"
                        ),
                    )
                for use_index, use in enumerate(action.buffer_uses):
                    abi = by_binding.get(
                        (action.source.schedule_id, use.binding_id)
                    )
                    if abi is None or abi.logical_core != action.logical_core:
                        raise SchemaError(
                            "rooted-AR local use has no exact manifest BufferABI",
                            path="source.manifest.fragments",
                        )
                    expected_access = (
                        BufferAccess.READ
                        if use.role in _READ_ROLES
                        else BufferAccess.WRITE
                        if use.role in _WRITE_ROLES
                        else None
                    )
                    if expected_access is None or use.access is not expected_access:
                        raise SchemaError(
                            "rooted-AR local buffer role/access is inconsistent",
                            path="source.source.intent.lowering_contexts",
                        )
                    collected[abi.id].append(
                        _LiteSemanticUse(
                            flattened_action_index,
                            use_index,
                            action,
                            use.access,
                            use.role,
                            use.operand_index,
                            order[1],
                            replica_index,
                        )
                    )
                flattened_action_index += 1

        replica_by_core = {
            abi.logical_core: replica_index
            for replica_index, abi in enumerate(
                source.source.intent.gradient_buffer_abis
            )
        }
        unit_base = flattened_action_index
        for unit_index, unit in enumerate(source.source.intent.units):
            order = action_order.get(unit.id)
            if order is None or order[0] != unit.logical_core:
                raise SchemaError(
                    "rooted-AR executable unit lacks its exact linked-core position",
                    path="source.manifest.core_streams",
                )
            replica_index = replica_by_core.get(unit.logical_core)
            if replica_index is None:
                raise SchemaError(
                    "rooted-AR unit core does not identify one replica",
                    path="source.source.intent.units",
                )
            if type(source) is S2LiteRootedArLinkedProgram:
                input_role = {
                    RootedArExecutableKind.LOCAL_COPY:
                        BufferUseRole.LOCAL_COPY_SOURCE,
                    RootedArExecutableKind.UPLOAD_SEND:
                        BufferUseRole.SEND_SOURCE,
                    RootedArExecutableKind.ROOT_REDUCE:
                        BufferUseRole.REDUCE_INPUT,
                    RootedArExecutableKind.DOWNLOAD_SEND:
                        BufferUseRole.SEND_SOURCE,
                }.get(unit.kind)
                output_role = {
                    RootedArExecutableKind.LOCAL_COPY:
                        BufferUseRole.LOCAL_COPY_DESTINATION,
                    RootedArExecutableKind.UPLOAD_RECV:
                        BufferUseRole.RECV_DESTINATION,
                    RootedArExecutableKind.ROOT_REDUCE:
                        BufferUseRole.REDUCE_OUTPUT,
                    RootedArExecutableKind.DOWNLOAD_RECV:
                        BufferUseRole.RECV_DESTINATION,
                }.get(unit.kind)
            else:
                input_role = {
                    Dp4TreeExecutableKind.LOCAL_COPY:
                        BufferUseRole.LOCAL_COPY_SOURCE,
                    Dp4TreeExecutableKind.FLOW_SEND:
                        BufferUseRole.SEND_SOURCE,
                    Dp4TreeExecutableKind.LOCAL_REDUCE:
                        BufferUseRole.REDUCE_INPUT,
                }.get(unit.kind)
                output_role = {
                    Dp4TreeExecutableKind.LOCAL_COPY:
                        BufferUseRole.LOCAL_COPY_DESTINATION,
                    Dp4TreeExecutableKind.FLOW_RECV:
                        BufferUseRole.RECV_DESTINATION,
                    Dp4TreeExecutableKind.LOCAL_REDUCE:
                        BufferUseRole.REDUCE_OUTPUT,
                }.get(unit.kind)
            if unit.input_buffer_abi_refs and input_role is None:
                raise SchemaError(
                    "rooted-AR unit kind cannot consume buffers",
                    path="source.source.intent.units",
                )
            if unit.output_buffer_abi_ref is not None and output_role is None:
                raise SchemaError(
                    "rooted-AR unit kind cannot produce a buffer",
                    path="source.source.intent.units",
                )
            for use_index, abi_id in enumerate(unit.input_buffer_abi_refs):
                abi = abis.get(abi_id)
                if abi is None or abi.logical_core != unit.logical_core:
                    raise SchemaError(
                        "rooted-AR unit input lacks a core-local BufferABI",
                        path="source.source.intent.units",
                    )
                assert input_role is not None
                collected[abi.id].append(
                    _LiteSemanticUse(
                        unit_base + unit_index,
                        use_index,
                        unit,
                        BufferAccess.READ,
                        input_role,
                        use_index,
                        order[1],
                        replica_index,
                    )
                )
            if unit.output_buffer_abi_ref is not None:
                abi = abis.get(unit.output_buffer_abi_ref)
                if abi is None or abi.logical_core != unit.logical_core:
                    raise SchemaError(
                        "rooted-AR unit output lacks a core-local BufferABI",
                        path="source.source.intent.units",
                    )
                assert output_role is not None
                collected[abi.id].append(
                    _LiteSemanticUse(
                        unit_base + unit_index,
                        len(unit.input_buffer_abi_refs),
                        unit,
                        BufferAccess.WRITE,
                        output_role,
                        0,
                        order[1],
                        replica_index,
                    )
                )

        result: dict[str, tuple[_LiteSemanticUse, ...]] = {}
        for abi_id, uses in collected.items():
            if not uses:
                raise SchemaError(
                    "rooted-AR BufferABI has no typed semantic use",
                    path="source.manifest.fragments",
                )
            ordered = tuple(sorted(uses, key=lambda item: item.order_key))
            ownership = abis[abi_id].ownership
            valid_first = (
                ordered[0].access is BufferAccess.READ
                if ownership is BufferOwnership.BORROWED
                else ordered[0].access is BufferAccess.WRITE
            )
            if not valid_first:
                raise SchemaError(
                    "rooted-AR BufferABI first use violates ownership",
                    path="source.manifest.core_streams",
                )
            result[abi_id] = ordered
        return result  # type: ignore[return-value]
    by_binding = {
        (abi.schedule_id, abi.binding_id): abi for abi in abis.values()
    }
    collected: dict[str, list[_SemanticUse]] = {
        abi_id: [] for abi_id in abis
    }
    flattened_action_index = 0
    for replica_index, context in _lowering_contexts(source):
        for action_index, action in enumerate(context.global_dag.actions):
            action_path = (
                "source.lowering_context.global_dag.actions"
                f"[{action_index}]"
            )
            for use_index, use in enumerate(action.buffer_uses):
                abi = by_binding.get(
                    (action.source.schedule_id, use.binding_id)
                )
                if abi is None:
                    raise SchemaError(
                        "GlobalAction buffer use has no manifest BufferABI",
                        path=f"{action_path}.buffer_uses[{use_index}]",
                    )
                if action.logical_core != abi.logical_core:
                    raise SchemaError(
                        "GlobalAction use and BufferABI logical core disagree",
                        path=f"{action_path}.buffer_uses[{use_index}]",
                    )
                expected_access = (
                    BufferAccess.READ
                    if use.role in _READ_ROLES
                    else BufferAccess.WRITE
                    if use.role in _WRITE_ROLES
                    else None
                )
                if (
                    expected_access is None
                    or use.access is not expected_access
                ):
                    raise SchemaError(
                        "buffer role and semantic access conflict",
                        path=f"{action_path}.buffer_uses[{use_index}]",
                    )
                if action.core_order_index is None:
                    raise SchemaError(
                        "executable buffer use lacks core order",
                        path=f"{action_path}.core_order_index",
                    )
                collected[abi.id].append(
                    _SemanticUse(
                        flattened_action_index,
                        use_index,
                        action,
                        use,
                        replica_index,
                    )
                )
            flattened_action_index += 1

    result: dict[str, tuple[_SemanticUse, ...]] = {}
    for abi_id, uses in collected.items():
        if not uses:
            raise SchemaError(
                "manifest BufferABI has no GlobalAction semantic use",
                path="source.manifest.fragments",
            )
        ordered = tuple(sorted(uses, key=lambda item: item.order_key))
        first_order = ordered[0].order_key[0]
        first = tuple(
            item for item in ordered if item.order_key[0] == first_order
        )
        ownership = abis[abi_id].ownership
        expected_first = (
            BufferAccess.READ
            if ownership is BufferOwnership.BORROWED
            else BufferAccess.WRITE
            if ownership is BufferOwnership.OWNED
            else BufferAccess.WRITE
            if (
                ownership is BufferOwnership.ALIASED
                and type(source) in _LITE_TRAIN_SOURCE_TYPES
            )
            else None
        )
        if expected_first is None or any(
            item.use.access is not expected_first for item in first
        ):
            raise SchemaError(
                "BORROWED first use must be READ and OWNED first use must be WRITE",
                path="source.lowering_context.global_dag.actions",
            )
        result[abi_id] = ordered
    return result


def _resolved_abis(source: LinkedProgramSource) -> tuple[_ResolvedAbi, ...]:
    abis = _unique_abis(source)
    uses = _semantic_uses(source, abis)
    runtime_by_core = {
        binding.logical_core: binding.runtime_core_id
        for binding in source.manifest.core_bindings
    }
    definitions: dict[
        tuple[str, LogicalCoreRef], list[tuple[int, ProgramSymbolDefinition]]
    ] = {}
    storage_ids = {abi.storage_id for abi in abis.values()}
    for index, definition in enumerate(
        source.manifest.program_symbol_definitions
    ):
        if (
            definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
            and definition.symbol.source_ref in storage_ids
        ):
            for core in definition.logical_cores:
                definitions.setdefault(
                    (definition.symbol.source_ref, core), []
                ).append((index, definition))

    result: list[_ResolvedAbi] = []
    for abi in sorted(abis.values(), key=lambda item: item.id):
        runtime_core_id = runtime_by_core.get(abi.logical_core)
        matches = definitions.get((abi.storage_id, abi.logical_core), ())
        if runtime_core_id is None or len(matches) != 1:
            raise SchemaError(
                "BufferABI requires one runtime core and one storage-backed SRAM_LABEL definition",
                path="source.manifest.program_symbol_definitions",
            )
        symbol_index, definition = matches[0]
        result.append(
            _ResolvedAbi(
                abi,
                runtime_core_id,
                symbol_index,
                definition,
                uses[abi.id],
            )
        )
    return tuple(result)


def _train_ce_nodes(
    source: (
        TrainLinkedProgram
        | S2LiteTrainLinkedProgram
        | S2LiteRootedArLinkedProgram
        | S2LiteDp4TreeArLinkedProgram
    ),
) -> tuple[tuple[int, CrossEntropyForwardWorkload, str, str], ...]:
    result: list[
        tuple[int, CrossEntropyForwardWorkload, str, str]
    ] = []
    replicas = _lowering_contexts(source)
    for replica_index, context in replicas:
        nodes = tuple(
            node
            for node in context.ir1.nodes
            if node.kind is OpKind.CE_FORWARD
        )
        if len(nodes) != 1:
            raise SchemaError(
                "Train replica requires exactly one CE_FORWARD terminal",
                path="source.source.replicas",
            )
        node = nodes[0]
        if (
            type(node.workload) is not CrossEntropyForwardWorkload
            or len(node.inputs) != 2
            or len(node.outputs) != 1
        ):
            raise SchemaError(
                "Train CE_FORWARD terminal contract is not exact",
                path="source.source.replicas",
            )
        result.append(
            (
                replica_index,
                node.workload,
                node.inputs[1],
                node.outputs[0],
            )
        )
    return tuple(result)


def _train_label_seed_overrides(
    source: LinkedProgramSource,
    resolved: tuple[_ResolvedAbi, ...],
) -> dict[str, bytes]:
    if type(source) not in _TRAIN_SOURCE_TYPES:
        return {}

    result: dict[str, bytes] = {}
    label_abi_ids: set[str] = set()
    for replica_index, workload, label_value_id, _loss_value_id in (
        _train_ce_nodes(source)
    ):
        logical_rows = workload.logical_label_shape[0]
        rank_rows = workload.rank_label_shape[0]
        vocabulary = workload.logical_logits_shape[1]
        if logical_rows % rank_rows:
            raise SchemaError(
                "Train CE rank rows must exactly partition logical rows",
                path="source.source.replicas",
            )
        labels = tuple(
            item
            for item in resolved
            if item.abi.value_id == label_value_id
            and {use.replica_index for use in item.uses}
            == {replica_index}
        )
        if len(labels) != logical_rows // rank_rows:
            raise SchemaError(
                "Train labels require one BufferABI per TP row shard",
                path="source.manifest.fragments",
            )
        if tuple(
            sorted(item.abi.tensor_slice.offset[0] for item in labels)
        ) != tuple(range(0, logical_rows, rank_rows)):
            raise SchemaError(
                "Train label BufferABIs must exactly partition logical rows",
                path="source.manifest.fragments",
            )
        for item in labels:
            abi = item.abi
            if type(source) in _LITE_TRAIN_SOURCE_TYPES:
                valid_uses = (
                    len(item.uses) == 2
                    and {
                        (
                            use.action.op_kind,
                            use.use.role,
                            use.use.operand_index,
                            use.use.access,
                            type(use.action.compute.workload)
                            if use.action.compute is not None
                            else None,
                        )
                        for use in item.uses
                    }
                    == {
                        (
                            OpKind.CE_FORWARD,
                            BufferUseRole.COMP_INPUT,
                            1,
                            BufferAccess.READ,
                            CrossEntropyForwardWorkload,
                        ),
                        (
                            OpKind.CE_BACKWARD,
                            BufferUseRole.COMP_INPUT,
                            1,
                            BufferAccess.READ,
                            CrossEntropyBackwardWorkload,
                        ),
                    }
                )
            else:
                valid_uses = all(
                    use.action.op_kind is OpKind.CE_FORWARD
                    and use.use.role is BufferUseRole.COMP_INPUT
                    and use.use.operand_index == 1
                    and use.use.access is BufferAccess.READ
                    and use.action.compute is not None
                    and type(use.action.compute.workload)
                    is CrossEntropyForwardWorkload
                    for use in item.uses
                )
            if (
                abi.ownership is not BufferOwnership.BORROWED
                or abi.dtype is not DType.INT32
                or abi.tensor_slice.shape != (rank_rows,)
                or len(abi.tensor_slice.offset) != 1
                or abi.size_bytes != rank_rows * 4
                or not valid_uses
            ):
                raise SchemaError(
                    "Train label BufferABI/use contract is not exact",
                    path="source.manifest.fragments",
                )
            row_offset = abi.tensor_slice.offset[0]
            payload = b"".join(
                (
                    (
                        replica_index * logical_rows
                        + row_offset
                        + row
                    )
                    % vocabulary
                ).to_bytes(4, "little")
                for row in range(rank_rows)
            )
            result[abi.id] = payload
            label_abi_ids.add(abi.id)

    if len(result) != len(label_abi_ids) or not result:
        raise SchemaError(
            "Train label BufferABIs must be unique and nonempty",
            path="source.manifest.fragments",
        )
    return result


def _lite_loss_gradient_seed_overrides(
    source: LinkedProgramSource,
    resolved: tuple[_ResolvedAbi, ...],
) -> dict[str, bytes]:
    if type(source) not in _LITE_TRAIN_SOURCE_TYPES:
        return {}
    result: dict[str, bytes] = {}
    for replica_index, context in _lowering_contexts(source):
        nodes = tuple(
            node
            for node in context.ir1.nodes
            if node.kind is OpKind.CE_BACKWARD
        )
        if len(nodes) != 1:
            raise SchemaError(
                "S2-Lite requires exactly one CE_BACKWARD node per replica",
                path="source.source.lowering_context.ir1.nodes",
            )
        node = nodes[0]
        if (
            type(node.workload) is not CrossEntropyBackwardWorkload
            or len(node.inputs) != 3
            or len(node.outputs) != 1
        ):
            raise SchemaError(
                "S2-Lite CE_BACKWARD contract is not exact",
                path="source.source.lowering_context.ir1.nodes",
            )
        workload = node.workload
        loss_gradient_value_id = node.inputs[2]
        items = tuple(
            item
            for item in resolved
            if item.abi.value_id == loss_gradient_value_id
            and {use.replica_index for use in item.uses} == {replica_index}
        )
        rank_rows = workload.rank_loss_gradient_shape[0]
        logical_rows = workload.logical_loss_gradient_shape[0]
        if (
            logical_rows % rank_rows
            or len(items) != logical_rows // rank_rows
            or tuple(sorted(item.abi.tensor_slice.offset[0] for item in items))
            != tuple(range(0, logical_rows, rank_rows))
        ):
            raise SchemaError(
                "S2-Lite loss-gradient BufferABIs must partition each replica's logical rows",
                path="source.manifest.fragments",
            )
        for item in items:
            abi = item.abi
            if (
                abi.id in result
                or abi.ownership is not BufferOwnership.BORROWED
                or abi.dtype is not DType.FP32
                or abi.tensor_slice.shape != (rank_rows,)
                or len(abi.tensor_slice.offset) != 1
                or abi.size_bytes != rank_rows * 4
                or len(item.uses) != 1
                or item.uses[0].action.op_kind is not OpKind.CE_BACKWARD
                or item.uses[0].use.role is not BufferUseRole.COMP_INPUT
                or item.uses[0].use.operand_index != 2
                or item.uses[0].use.access is not BufferAccess.READ
            ):
                raise SchemaError(
                    "S2-Lite loss gradient requires exact replica-local BORROWED FP32 CE_BACKWARD input2",
                    path="source.manifest.fragments",
                )
            result[abi.id] = bytes(abi.size_bytes)
    return result


def _terminal_value_ids(source: LinkedProgramSource) -> set[str]:
    if type(source) is SwizzleStandardLinkedProgram:
        return swizzle_terminal_value_ids(source)
    if type(source) is UnfusedComparisonStandardLinkedProgram:
        return unfused_comparison_terminal_value_ids(source)
    if type(source) is S2LiteDp4TreeArLinkedProgram:
        return set()
    if type(source) is LiteMoeBackwardLinkedProgram:
        return set()
    if type(source) is LiteMoeDp4BackwardLinkedProgram:
        return set()
    if type(source) is LiteMoeDp4InferLinkedProgram:
        terminals = set(source.source.source.global_dag.combined_output_refs)
        if len(terminals) != 8:
            raise SchemaError(
                "S3-Lite DP4 infer requires eight combined token outputs",
                path="source.source.source.global_dag.combined_output_refs",
            )
        return terminals
    if type(source) is LiteMoeDp4TrainForwardLinkedProgram:
        forward = source.source.source.forward
        combined = set(forward.global_dag.combined_output_refs)
        tapes = {item.value_ref for item in source.source.source.tape_buffers}
        if len(combined) != 8 or len(tapes) != 8 or combined.intersection(tapes):
            raise SchemaError(
                "S3-Lite DP4 TF requires eight combined and eight distinct tape terminals",
                path="source.source.source",
            )
        return combined.union(tapes)
    if type(source) is LiteMoeLinkedProgram:
        resolved = _resolved_abis(source)
        terminals = {
            item.abi.value_id
            for item in resolved
            if item.abi.ownership is BufferOwnership.OWNED
            and item.uses[-1].use.access is BufferAccess.WRITE
        }
        if not terminals:
            raise SchemaError(
                "S3-Lite requires at least one final combined token output",
                path="source.source.global_dag.actions",
            )
        return terminals
    actual = {
        value.id
        for _replica_index, context in _lowering_contexts(source)
        for value in context.ir1.values
        if not value.consumers
    }
    if type(source) not in _LITE_TRAIN_SOURCE_TYPES:
        return actual
    expected: set[str] = set()
    losses: set[str] = set()
    for _replica_index, context in _lowering_contexts(source):
        ce_nodes = tuple(
            node
            for node in context.ir1.nodes
            if node.kind is OpKind.CE_FORWARD
        )
        sgd_nodes = tuple(
            node
            for node in context.ir1.nodes
            if node.kind is OpKind.OPTIMIZER_UPDATE
        )
        if len(ce_nodes) != 1 or len(sgd_nodes) != 1:
            raise SchemaError(
                "S2-Lite requires one CE_FORWARD and one OPTIMIZER_UPDATE per replica",
                path="source.source.lowering_context.ir1.nodes",
            )
        loss_id = ce_nodes[0].outputs[0]
        losses.add(loss_id)
        expected.update((loss_id, sgd_nodes[0].outputs[0]))
    if actual != expected:
        raise SchemaError(
            "S2-Lite IR1 terminals must be exactly each replica's loss and updated weight",
            path="source.source.lowering_context.ir1.values",
        )
    return losses


def _validate_train_terminal_abis(
    source: LinkedProgramSource,
    terminal_values: set[str],
    resolved_terminal: tuple[_ResolvedAbi, ...],
) -> None:
    if type(source) not in _TRAIN_SOURCE_TYPES:
        return
    ce_nodes = _train_ce_nodes(source)
    expected_values = {item[3] for item in ce_nodes}
    if terminal_values != expected_values:
        raise SchemaError(
            "Train terminals must be exactly the CE loss values",
            path="source.source.replicas",
        )
    covered_abi_ids: set[str] = set()
    for replica_index, workload, _label_value_id, loss_value_id in ce_nodes:
        logical_rows = workload.logical_loss_shape[0]
        rank_rows = workload.rank_loss_shape[0]
        losses = tuple(
            item
            for item in resolved_terminal
            if item.abi.value_id == loss_value_id
            and {use.replica_index for use in item.uses}
            == {replica_index}
        )
        if (
            logical_rows % rank_rows
            or len(losses) != logical_rows // rank_rows
            or tuple(
                sorted(item.abi.tensor_slice.offset[0] for item in losses)
            )
            != tuple(range(0, logical_rows, rank_rows))
        ):
            raise SchemaError(
                "Train loss probes must exactly cover every replica TP shard",
                path="source.manifest.fragments",
            )
        for item in losses:
            if (
                item.abi.id in covered_abi_ids
                or item.abi.ownership is not BufferOwnership.OWNED
                or item.abi.dtype is not DType.FP32
                or item.abi.tensor_slice.shape != (rank_rows,)
                or len(item.abi.tensor_slice.offset) != 1
                or item.abi.size_bytes != rank_rows * 4
                or any(
                    use.action.op_kind is not OpKind.CE_FORWARD
                    or use.use.role is not BufferUseRole.COMP_OUTPUT
                    or use.use.operand_index != 0
                    or use.use.access is not BufferAccess.WRITE
                    for use in item.uses
                )
            ):
                raise SchemaError(
                    "Train loss probes require exact OWNED FP32 CE outputs",
                    path="source.manifest.fragments",
                )
            covered_abi_ids.add(item.abi.id)
    if covered_abi_ids != {item.abi.id for item in resolved_terminal}:
        raise SchemaError(
            "Train loss BufferABIs cannot cross or escape DP replicas",
            path="source.manifest.fragments",
        )


def _validate_lite_train_state_update(
    source: LinkedProgramSource,
    resolved: tuple[_ResolvedAbi, ...],
    resolved_state: tuple[_ResolvedStateAbi, ...],
) -> None:
    if type(source) not in _LITE_TRAIN_SOURCE_TYPES:
        return
    covered_state_abis: set[str] = set()
    covered_alias_abis: set[str] = set()
    flattened_offset = 0
    for replica_index, context in _lowering_contexts(source):
        actions = context.global_dag.actions
        replica_start = flattened_offset
        replica_end = replica_start + len(actions)
        sgd_actions = tuple(
            (replica_start + index, action)
            for index, action in enumerate(actions)
            if action.op_kind is OpKind.OPTIMIZER_UPDATE
        )
        if len(sgd_actions) != 1:
            raise SchemaError(
                "S2-Lite requires one exact SGD GlobalAction per replica",
                path="source.source.lowering_context.global_dag.actions",
            )
        sgd_index, sgd_action = sgd_actions[0]
        active_dies = {action.logical_core.die_id for action in actions}
        trainable = tuple(
            item
            for item in resolved_state
            if item.abi.kind is StateKind.TRAINABLE_PARAMETER
            and item.abi.die_id in active_dies
            and item.uses
            and all(
                replica_start <= index < replica_end
                for index, _access in item.uses
            )
        )
        if (
            len(trainable) != 1
            or trainable[0].abi.id in covered_state_abis
            or trainable[0].abi.access
            is not PersistentStateAccess.READ_WRITE
            or tuple(access for _index, access in trainable[0].uses)
            != (
                StateUseAccess.READ,
                StateUseAccess.READ,
                StateUseAccess.WRITE,
            )
            or any(
                index >= sgd_index
                for index, access in trainable[0].uses
                if access is StateUseAccess.READ
            )
            or any(
                index <= sgd_index
                for index, access in trainable[0].uses
                if access is StateUseAccess.WRITE
            )
        ):
            raise SchemaError(
                "each S2-Lite replica requires two trainable-state loads before SGD and one store after",
                path="source.manifest.fragments",
            )
        covered_state_abis.add(trainable[0].abi.id)

        sgd_nodes = tuple(
            node
            for node in context.ir1.nodes
            if node.kind is OpKind.OPTIMIZER_UPDATE
        )
        if len(sgd_nodes) != 1:
            raise SchemaError(
                "S2-Lite requires one exact SGD IR1 node per replica",
                path="source.source.lowering_context.ir1.nodes",
            )
        aliases = tuple(
            item
            for item in resolved
            if item.abi.value_id == sgd_nodes[0].outputs[0]
            and {use.replica_index for use in item.uses} == {replica_index}
        )
        if len(aliases) != 1 or aliases[0].abi.id in covered_alias_abis:
            raise SchemaError(
                "each S2-Lite replica requires one alias BufferABI",
                path="source.manifest.fragments",
            )
        alias = aliases[0]
        roots = tuple(
            item
            for item in resolved
            if item.abi.schedule_id == alias.abi.schedule_id
            and item.abi.binding_id == alias.abi.alias_of
        )
        if (
            alias.abi.ownership is not BufferOwnership.ALIASED
            or len(roots) != 1
            or roots[0].abi.storage_id != alias.abi.storage_id
            or len(alias.uses) != 1
            or alias.uses[0].replica_index != replica_index
            or alias.uses[0].action != sgd_action
            or alias.uses[0].use.role is not BufferUseRole.COMP_OUTPUT
            or alias.uses[0].use.operand_index != 0
            or alias.uses[0].use.access is not BufferAccess.WRITE
        ):
            raise SchemaError(
                "each S2-Lite updated weight must exactly alias its replica-local trainable staging root",
                path="source.manifest.fragments",
            )
        covered_alias_abis.add(alias.abi.id)
        flattened_offset = replica_end

    if covered_state_abis != {
        item.abi.id
        for item in resolved_state
        if item.abi.kind is StateKind.TRAINABLE_PARAMETER
    }:
        raise SchemaError(
            "S2-Lite trainable StateABIs must be replica-disjoint and complete",
            path="source.manifest.fragments",
        )


def _state_access_is_permitted(
    declared: PersistentStateAccess,
    actual: set[StateUseAccess],
) -> bool:
    """Return whether nonempty runtime directions fit declaration rights."""

    if not actual or declared is PersistentStateAccess.RESERVED:
        return False
    allowed = (
        {StateUseAccess.READ}
        if declared is PersistentStateAccess.READ_ONLY
        else {StateUseAccess.READ, StateUseAccess.WRITE}
    )
    return actual.issubset(allowed)


def _resolved_state_abis(
    source: LinkedProgramSource,
) -> tuple[_ResolvedStateAbi, ...]:
    if type(source) in (SwizzleStandardLinkedProgram, UnfusedComparisonStandardLinkedProgram):
        if source.manifest.state_operand_bindings or any(
            _leaf(linked).state_abi for linked in source.manifest.fragments
        ):
            raise SchemaError(
                "Swizzle/UNFUSED standard V1 forbids persistent StateABI",
                path="source.manifest.fragments",
            )
        return ()
    by_id: dict[str, StateABI] = {}
    by_binding: dict[str, StateABI] = {}
    for linked in source.manifest.fragments:
        for abi in _leaf(linked).state_abi:
            previous = by_id.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError(
                    "conflicting shared StateABI definition",
                    path="source.manifest.fragments",
                )
            previous = by_binding.setdefault(abi.hbm_binding_ref, abi)
            if previous != abi:
                raise SchemaError(
                    "one HBM binding resolves to conflicting StateABIs",
                    path="source.manifest.fragments",
                )

    uses: dict[str, list[tuple[int, StateUseAccess]]] = {
        abi_id: [] for abi_id in by_id
    }
    if type(source) in (
        LiteMoeLinkedProgram,
        LiteMoeDp4InferLinkedProgram,
        LiteMoeDp4TrainForwardLinkedProgram,
    ):
        if type(source) is LiteMoeDp4InferLinkedProgram:
            execution = source.source.source
            intent = source.source.intent
        elif type(source) is LiteMoeDp4TrainForwardLinkedProgram:
            execution = source.source.source.forward
            intent = source.source.forward.intent
        else:
            execution = source.source
            intent = source.source.intent
        action_order = {
            action.id: index
            for index, action in enumerate(execution.global_dag.actions)
        }
        for unit in intent.state_loads:
            abi = by_binding.get(unit.hbm_binding_ref)
            if abi is None:
                raise SchemaError(
                    "S3-Lite state load has no manifest StateABI",
                    path="source.source.intent.state_loads",
                )
            uses[abi.id].append((action_order[unit.action_ref], StateUseAccess.READ))
    elif type(source) in (
        LiteMoeBackwardLinkedProgram,
        LiteMoeDp4BackwardLinkedProgram,
    ):
        fragments = {
            _leaf(fragment).id: _leaf(fragment)
            for fragment in source.manifest.fragments
        }
        witnessed: dict[str, set[RecordOpcode]] = {
            abi_id: set() for abi_id in by_id
        }
        for binding in source.manifest.state_operand_bindings:
            fragment = fragments[binding.fragment_id]
            stream = next(
                item for item in fragment.core_streams
                if item.logical_core == binding.logical_core
            )
            witnessed[binding.state_abi_id].add(
                stream.records[binding.fragment_record_index].opcode
            )
        overlay = (
            source.source.source
            if type(source) is LiteMoeDp4BackwardLinkedProgram
            else source.source.overlay
        )
        for state in overlay.trainable_down_states:
            abi = by_binding.get(state.binding.id)
            if abi is None or witnessed.get(abi.id) != {
                RecordOpcode.LSU_LOAD,
                RecordOpcode.LSU_STORE,
            }:
                raise SchemaError(
                    "MoE backward trainable state requires one load and one store witness",
                    path="source.manifest.state_operand_bindings",
                )
            uses[abi.id].extend((
                (state.expert_index, StateUseAccess.READ),
                (100 + state.expert_index, StateUseAccess.WRITE),
            ))
    else:
        flattened_action_index = 0
        for _replica_index, context in _lowering_contexts(source):
            for action_index, action in enumerate(context.global_dag.actions):
                for use_index, use in enumerate(action.state_uses):
                    abi = by_binding.get(use.hbm_binding_ref)
                    if abi is None:
                        raise SchemaError(
                            "GlobalAction state use has no manifest StateABI",
                            path=(
                                "source.lowering_context.global_dag.actions"
                                f"[{action_index}].state_uses[{use_index}]"
                            ),
                        )
                    uses[abi.id].append(
                        (flattened_action_index, use.access)
                    )
                flattened_action_index += 1

    definitions: dict[str, list[tuple[int, ProgramSymbolDefinition]]] = {}
    for index, definition in enumerate(
        source.manifest.program_symbol_definitions
    ):
        if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
            definitions.setdefault(definition.symbol.source_ref, []).append(
                (index, definition)
            )

    result: list[_ResolvedStateAbi] = []
    for abi in sorted(by_id.values(), key=lambda item: item.id):
        abi_uses = tuple(uses[abi.id])
        matches = definitions.get(abi.hbm_binding_ref, ())
        if not abi_uses or len(matches) != 1:
            raise SchemaError(
                "StateABI requires GlobalAction use and one final HBM symbol",
                path="source.manifest.fragments",
            )
        access_set = {access for _index, access in abi_uses}
        if not _state_access_is_permitted(abi.access, access_set):
            raise SchemaError(
                "StateABI access permissions reject GlobalAction state directions",
                path="source.lowering_context.global_dag.actions",
            )
        symbol_index, definition = matches[0]
        result.append(
            _ResolvedStateAbi(
                abi=abi,
                symbol_index=symbol_index,
                symbol_definition=definition,
                uses=abi_uses,
            )
        )
    if type(source) not in (
        LiteMoeLinkedProgram,
        LiteMoeBackwardLinkedProgram,
        LiteMoeDp4InferLinkedProgram,
        LiteMoeDp4TrainForwardLinkedProgram,
        LiteMoeDp4BackwardLinkedProgram,
    ) and (
        any(
            action.state_uses
            for _replica_index, context in _lowering_contexts(source)
            for action in context.global_dag.actions
        )
        and not result
    ):
        raise SchemaError(
            "GlobalAction state uses require manifested StateABI entries",
            path="source.manifest.fragments",
        )
    return tuple(result)


def _timing_state_pattern(
    ordinal: int,
    resolved: _ResolvedStateAbi,
) -> bytes:
    """Return a stable, nonzero whole-state pattern for one canonical state."""

    material = b"wafer_frontend.timing_state_seed/v1\0" + b"\0".join(
        (
            ordinal.to_bytes(8, "big"),
            resolved.abi.state_ref.encode("utf-8"),
            resolved.abi.id.encode("utf-8"),
        )
    )
    payload = bytearray()
    counter = 0
    while len(payload) < resolved.abi.size_bytes:
        block = hashlib.sha256(
            material + counter.to_bytes(8, "big")
        ).digest()
        payload.extend(byte if byte != 0 else 0xA5 for byte in block)
        counter += 1
    return bytes(payload[: resolved.abi.size_bytes])


def _state_logical_signature(resolved: _ResolvedStateAbi) -> tuple[object, ...]:
    abi = resolved.abi
    return (
        abi.kind,
        abi.lifetime,
        abi.access,
        abi.shape,
        abi.dtype,
        abi.layout,
        abi.size_bytes,
        abi.alignment_bytes,
    )


def _state_abi_groups(
    resolved_state: tuple[_ResolvedStateAbi, ...],
) -> dict[str, tuple[_ResolvedStateAbi, ...]]:
    collected: dict[str, list[_ResolvedStateAbi]] = {}
    for item in sorted(
        resolved_state,
        key=lambda candidate: (candidate.abi.state_ref, candidate.abi.id),
    ):
        collected.setdefault(item.abi.state_ref, []).append(item)

    result: dict[str, tuple[_ResolvedStateAbi, ...]] = {}
    physical_ids: set[str] = set()
    physical_bindings: set[str] = set()
    for state_ref in sorted(collected):
        group = tuple(collected[state_ref])
        if len({_state_logical_signature(item) for item in group}) != 1:
            raise SchemaError(
                "one logical state ref has conflicting physical StateABI semantics",
                path="source.manifest.fragments",
            )
        if len(
            {
                (
                    item.first_access,
                    any(
                        access is StateUseAccess.WRITE
                        for _index, access in item.uses
                    ),
                )
                for item in group
            }
        ) != 1:
            raise SchemaError(
                "one logical state ref has asymmetric physical access traces",
                path="source.lowering_context.global_dag.actions",
            )
        local_ids = {item.abi.id for item in group}
        local_bindings = {item.abi.hbm_binding_ref for item in group}
        if (
            len(local_ids) != len(group)
            or len(local_bindings) != len(group)
            or physical_ids.intersection(local_ids)
            or physical_bindings.intersection(local_bindings)
        ):
            raise SchemaError(
                "physical StateABI ids and HBM bindings must be unique",
                path="source.manifest.fragments",
            )
        physical_ids.update(local_ids)
        physical_bindings.update(local_bindings)
        result[state_ref] = group
    return result


def _deterministic_timing_state_overrides(
    resolved_state: tuple[_ResolvedStateAbi, ...],
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Derive exact timing seeds/probes without weakening ProgramIo closure."""

    groups = _state_abi_groups(resolved_state)

    seeds: dict[str, bytes] = {}
    expected: dict[str, bytes] = {}
    payload_owners: dict[bytes, str] = {}
    for ordinal, (state_ref, group) in enumerate(groups.items(), start=1):
        item = group[0]
        if item.first_access is not StateUseAccess.READ:
            continue
        payload = _timing_state_pattern(ordinal, item)
        previous_owner = payload_owners.setdefault(payload, item.abi.state_ref)
        if previous_owner != item.abi.state_ref:
            raise SchemaError(
                "deterministic timing state patterns must be distinct",
                path="source.manifest.fragments",
            )
        seeds[state_ref] = payload
        if (
            item.abi.access is PersistentStateAccess.READ_WRITE
            and item.abi.kind is not StateKind.TRAINABLE_PARAMETER
            and any(
                access is StateUseAccess.WRITE
                for _action_index, access in item.uses
            )
        ):
            expected[state_ref] = payload
    return seeds, expected


def build_deterministic_timing_state_overrides(
    source: LinkedProgramSource,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Build explicit production timing seeds and exact stored-state probes."""

    if type(source) not in _LINKED_PROGRAM_SOURCE_TYPES:
        raise SchemaError(
            "source must be a supported linked program carrier",
            path="source",
        )
    if type(source) in (SwizzleStandardLinkedProgram, UnfusedComparisonStandardLinkedProgram):
        source.validate_against("source")
    else:
        source.validate("source")
    return _deterministic_timing_state_overrides(
        _resolved_state_abis(source)
    )


def _override_bytes(
    values: Mapping[str, bytes] | None,
    path: str,
) -> dict[str, bytes]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SchemaError("must be a mapping", path=path)
    result: dict[str, bytes] = {}
    for state_ref, payload in values.items():
        if type(state_ref) is not str or not state_ref:
            raise SchemaError("keys must be nonempty state refs", path=path)
        if type(payload) is not bytes:
            raise SchemaError(
                "values must be immutable bytes",
                path=f"{path}[{state_ref!r}]",
            )
        result[state_ref] = payload
    return result


def _sram_override_bytes(
    values: Mapping[str, bytes] | None,
    abis: Mapping[str, BufferABI],
    required_ownership: BufferOwnership,
    path: str,
) -> dict[str, bytes]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SchemaError("must be a mapping", path=path)
    result: dict[str, bytes] = {}
    for buffer_abi_id, payload in values.items():
        if type(buffer_abi_id) is not str or not buffer_abi_id:
            raise SchemaError(
                "keys must be nonempty BufferABI ids",
                path=path,
            )
        abi = abis.get(buffer_abi_id)
        if abi is None:
            raise SchemaError(
                "must reference a manifested BufferABI",
                path=f"{path}[{buffer_abi_id!r}]",
            )
        if abi.ownership is not required_ownership:
            raise SchemaError(
                f"must reference only {required_ownership.value} BufferABI entries",
                path=f"{path}[{buffer_abi_id!r}]",
            )
        if type(payload) is not bytes:
            raise SchemaError(
                "values must be immutable bytes",
                path=f"{path}[{buffer_abi_id!r}]",
            )
        if len(payload) != abi.size_bytes:
            raise SchemaError(
                "payload must cover exactly one whole BufferABI root",
                path=f"{path}[{buffer_abi_id!r}]",
            )
        result[buffer_abi_id] = payload
    return result


def _hbm_target(resolved: _ResolvedStateAbi) -> ProgramHbmTarget:
    return ProgramHbmTarget(
        kind=ProgramIoTargetKind.HBM,
        program_symbol_ref=resolved.symbol_definition.symbol.id,
        finalized_symbol_index=resolved.symbol_index,
        expected_symbol_name=resolved.symbol_definition.name,
        state_abi_id=resolved.abi.id,
        state_ref=resolved.abi.state_ref,
        hbm_binding_ref=resolved.abi.hbm_binding_ref,
    )


def _initialization(
    resolved: _ResolvedAbi,
    blob: ProgramBlob,
) -> ProgramSramInitialization:
    abi = resolved.abi
    if abi.ownership is BufferOwnership.BORROWED:
        purpose = (
            ProgramIoPurpose.WEIGHT
            if all(
                item.use.role is BufferUseRole.COMP_INPUT
                and item.use.operand_index == 1
                for item in resolved.uses
            )
            and abi.layout != "s3_lite_moe_upstream_gradient"
            else ProgramIoPurpose.ACTIVATION
        )
    elif abi.ownership is BufferOwnership.OWNED:
        purpose = ProgramIoPurpose.TIMING_PARTIAL
    else:
        raise SchemaError(
            "timing ProgramIo rejects aliased BufferABI",
            path="source.manifest.fragments",
        )
    return ProgramSramInitialization.create(
        target=ProgramSramTarget(
            kind=ProgramIoTargetKind.SRAM,
            runtime_core_id=resolved.runtime_core_id,
            program_symbol_ref=resolved.symbol_definition.symbol.id,
            finalized_symbol_index=resolved.symbol_index,
            expected_symbol_name=resolved.symbol_definition.name,
            buffer_abi_id=abi.id,
            storage_id=abi.storage_id,
            value_id=abi.value_id,
            tensor_slice=abi.tensor_slice,
            dtype=abi.dtype,
            layout=abi.layout,
        ),
        offset_bytes=0,
        length_bytes=abi.size_bytes,
        blob_ref=blob.id,
        purpose=purpose,
    )


def _probe(resolved: _ResolvedAbi, blob: ProgramBlob) -> ProgramOutputProbe:
    abi = resolved.abi
    return ProgramOutputProbe.create(
        target=ProgramSramTarget(
            kind=ProgramIoTargetKind.SRAM,
            runtime_core_id=resolved.runtime_core_id,
            program_symbol_ref=resolved.symbol_definition.symbol.id,
            finalized_symbol_index=resolved.symbol_index,
            expected_symbol_name=resolved.symbol_definition.name,
            buffer_abi_id=abi.id,
            storage_id=abi.storage_id,
            value_id=abi.value_id,
            tensor_slice=abi.tensor_slice,
            dtype=abi.dtype,
            layout=abi.layout,
        ),
        offset_bytes=0,
        length_bytes=abi.size_bytes,
        blob_ref=blob.id,
        comparison=ProgramOutputComparison.EXACT_BYTES,
        capture=ProgramOutputCapture.AFTER_PROGRAM,
    )


def _state_initialization(
    resolved: _ResolvedStateAbi,
    blob: ProgramBlob,
) -> ProgramSramInitialization:
    return ProgramSramInitialization.create(
        target=_hbm_target(resolved),
        offset_bytes=0,
        length_bytes=resolved.abi.size_bytes,
        blob_ref=blob.id,
        purpose=ProgramIoPurpose.STATE,
    )


def _state_probe(
    resolved: _ResolvedStateAbi,
    blob: ProgramBlob,
) -> ProgramOutputProbe:
    return ProgramOutputProbe.create(
        target=_hbm_target(resolved),
        offset_bytes=0,
        length_bytes=resolved.abi.size_bytes,
        blob_ref=blob.id,
        comparison=ProgramOutputComparison.EXACT_BYTES,
        capture=ProgramOutputCapture.AFTER_PROGRAM,
    )


def _validate_dp4_updated_weight_probes(
    source: LinkedProgramSource,
    contract: ProgramIoContract,
    resolved_state: tuple[_ResolvedStateAbi, ...],
) -> None:
    if type(source) is not S2LiteDp4TreeArLinkedProgram:
        return
    updated = tuple(
        item
        for item in resolved_state
        if item.abi.kind is StateKind.TRAINABLE_PARAMETER
        and item.abi.access is PersistentStateAccess.READ_WRITE
        and any(access is StateUseAccess.WRITE for _index, access in item.uses)
    )
    if (
        len(updated) != 4
        or tuple(sorted(item.abi.die_id for item in updated)) != (0, 1, 2, 3)
        or any(item.abi.size_bytes != 1024 for item in updated)
        or len({item.abi.state_ref for item in updated}) != 1
    ):
        raise SchemaError(
            "DP4 requires four replica-local 1024B updated LM-head StateABIs",
            path="source.manifest.fragments",
        )
    expected_ids = {item.abi.id for item in updated}
    probes = tuple(
        probe
        for probe in contract.output_probes
        if type(probe.target) is ProgramHbmTarget
    )
    if (
        len(contract.output_probes) != 4
        or len(probes) != 4
        or {probe.target.state_abi_id for probe in probes} != expected_ids
        or sum(probe.length_bytes for probe in probes) != 4096
    ):
        raise SchemaError(
            "DP4 output probes must exactly cover four updated LM-head HBM states",
            path="program_io_contract.output_probes",
        )
    initializations = {
        item.target.state_abi_id: item
        for item in contract.initializations
        if type(item.target) is ProgramHbmTarget
    }
    for index, probe in enumerate(probes):
        initialization = initializations.get(probe.target.state_abi_id)
        if initialization is None or probe.blob_ref != initialization.blob_ref:
            raise SchemaError(
                "DP4 updated-weight probe must reuse its exact deterministic state seed",
                path=f"program_io_contract.output_probes[{index}]",
            )


def _validate_lite_moe_dp4_backward_program_io(
    source: LinkedProgramSource,
    contract: ProgramIoContract,
    resolved: tuple[_ResolvedAbi, ...],
    resolved_state: tuple[_ResolvedStateAbi, ...],
) -> None:
    if type(source) is not LiteMoeDp4BackwardLinkedProgram:
        return
    overlay = source.source.source
    tape_by_value = {
        item.value_ref: item
        for item in overlay.train_forward.tape_buffers
    }
    saved = tuple(
        item for item in resolved
        if item.abi.layout == "s3_lite_moe_saved_activation"
    )
    upstream = tuple(
        item for item in resolved
        if item.abi.layout == "s3_lite_moe_upstream_gradient"
    )
    borrowed = tuple(
        item for item in resolved
        if item.abi.ownership is BufferOwnership.BORROWED
    )
    if (
        len(saved) != 8
        or len(upstream) != 8
        or {item.abi.id for item in borrowed}
        != {item.abi.id for item in (*saved, *upstream)}
        or any(
            item.abi.dtype is not DType.FP16
            or item.abi.size_bytes != 64
            or item.abi.tensor_slice.shape != (1, 32)
            or item.abi.value_id not in tape_by_value
            or item.abi.logical_core.die_id
            != tape_by_value[item.abi.value_id].die_id
            for item in saved
        )
    ):
        raise SchemaError(
            "DP4 backward requires exact eight saved tape activation inputs",
            path="source.manifest.fragments",
        )
    remote = {
        item.token_index: item
        for item in overlay.remote_gradients
    }
    expected_upstream = {
        (
            (
                remote[item.token_index].source_gradient_ref
                if item.token_index in remote
                else item.upstream_gradient_ref
            ),
            (
                remote[item.token_index].source_die_id
                if item.token_index in remote
                else item.home_die_id
            ),
        )
        for item in overlay.token_wgrads
    }
    if (
        {
            (item.abi.value_id, item.abi.logical_core.die_id)
            for item in upstream
        }
        != expected_upstream
        or any(
            item.abi.dtype is not DType.FP16
            or item.abi.size_bytes != 32
            or item.abi.tensor_slice.shape != (1, 16)
            for item in upstream
        )
    ):
        raise SchemaError(
            "DP4 backward requires exact eight token-source upstream gradients",
            path="source.manifest.fragments",
        )
    trainable = tuple(
        item for item in resolved_state
        if item.abi.kind is StateKind.TRAINABLE_PARAMETER
    )
    if (
        len(trainable) != 4
        or tuple(sorted(item.abi.die_id for item in trainable)) != (0, 1, 2, 3)
        or any(
            item.abi.access is not PersistentStateAccess.READ_WRITE
            or item.abi.size_bytes != 1024
            or {access for _index, access in item.uses}
            != {StateUseAccess.READ, StateUseAccess.WRITE}
            for item in trainable
        )
        or {item.abi.state_ref for item in trainable}
        != {
            item.declaration.id
            for item in overlay.trainable_down_states
        }
    ):
        raise SchemaError(
            "DP4 backward requires four exact trainable down-weight states",
            path="source.manifest.fragments",
        )
    sram_initializations = tuple(
        item for item in contract.initializations
        if type(item.target) is ProgramSramTarget
    )
    hbm_initializations = tuple(
        item for item in contract.initializations
        if type(item.target) is ProgramHbmTarget
    )
    probes = tuple(
        item for item in contract.output_probes
        if type(item.target) is ProgramHbmTarget
    )
    input_ids = {item.abi.id for item in borrowed}
    initialized_inputs = {
        item.target.buffer_abi_id
        for item in sram_initializations
        if item.target.buffer_abi_id in input_ids
        and item.purpose is ProgramIoPurpose.ACTIVATION
    }
    state_ids = {item.abi.id for item in trainable}
    if (
        len(contract.blobs) != 8
        or len(contract.initializations) != 34
        or len(sram_initializations) != 30
        or initialized_inputs != input_ids
        or len(hbm_initializations) != 4
        or {item.target.state_abi_id for item in hbm_initializations} != state_ids
        or len(contract.output_probes) != 4
        or len(probes) != 4
        or {item.target.state_abi_id for item in probes} != state_ids
        or sum(item.length_bytes for item in probes) != 4096
    ):
        raise SchemaError(
            "DP4 backward ProgramIo input/state/probe closure changed",
            path="program_io_contract",
        )
    initialization_by_state = {
        item.target.state_abi_id: item
        for item in hbm_initializations
    }
    if any(
        probe.blob_ref
        != initialization_by_state[probe.target.state_abi_id].blob_ref
        for probe in probes
    ):
        raise SchemaError(
            "DP4 backward updated-weight probes must reuse exact state seeds",
            path="program_io_contract.output_probes",
        )


def build_timing_program_io(
    source: LinkedProgramSource,
    program_artifact_sha256: str,
    *,
    sram_seed_overrides: Mapping[str, bytes] | None = None,
    sram_expected_overrides: Mapping[str, bytes] | None = None,
    state_seed_overrides: Mapping[str, bytes] | None = None,
    state_expected_overrides: Mapping[str, bytes] | None = None,
) -> ProgramIoContract:
    """Build timing IO with explicit whole-state HBM payloads.

    SRAM defaults to zero-filled timing payloads. Callers may opt specific
    whole-root BORROWED inputs and OWNED outputs into exact bytes. Every state
    whose first global action is an HBM load requires an exact seed override;
    expected state is opt-in and READ_WRITE-only.
    """

    if type(source) not in _LINKED_PROGRAM_SOURCE_TYPES:
        raise SchemaError(
            "source must be a supported linked program carrier",
            path="source",
        )
    if type(source) in (SwizzleStandardLinkedProgram, UnfusedComparisonStandardLinkedProgram):
        source.validate_against("source")
    else:
        source.validate("source")
    resolved = _resolved_abis(source)
    resolved_state = _resolved_state_abis(source)
    _validate_lite_train_state_update(source, resolved, resolved_state)
    buffer_by_id = {item.abi.id: item.abi for item in resolved}
    sram_seeds = _sram_override_bytes(
        sram_seed_overrides,
        buffer_by_id,
        BufferOwnership.BORROWED,
        "sram_seed_overrides",
    )
    sram_expected = _sram_override_bytes(
        sram_expected_overrides,
        buffer_by_id,
        BufferOwnership.OWNED,
        "sram_expected_overrides",
    )
    if (
        type(source) in _TRAIN_SOURCE_TYPES
        or type(source) in (
            LiteMoeBackwardLinkedProgram,
            LiteMoeDp4BackwardLinkedProgram,
        )
    ) and sram_expected:
        raise SchemaError(
            "Train/MoE-backward timing ProgramIo does not accept numeric SRAM expectations",
            path="sram_expected_overrides",
        )
    automatic_label_seeds = _train_label_seed_overrides(source, resolved)
    automatic_loss_gradient_seeds = _lite_loss_gradient_seed_overrides(
        source,
        resolved,
    )
    for automatic_name, automatic in (
        ("label", automatic_label_seeds),
        ("loss-gradient", automatic_loss_gradient_seeds),
    ):
        overlap = set(sram_seeds).intersection(automatic)
        if any(
            sram_seeds[abi_id] != automatic[abi_id]
            for abi_id in overlap
        ):
            raise SchemaError(
                f"Train {automatic_name} override conflicts with the automatic payload",
                path="sram_seed_overrides",
            )
        sram_seeds.update(automatic)
    seeds = _override_bytes(state_seed_overrides, "state_seed_overrides")
    expected = _override_bytes(
        state_expected_overrides,
        "state_expected_overrides",
    )
    state_by_ref = _state_abi_groups(resolved_state)

    required_seed_refs = {
        item.abi.state_ref
        for item in resolved_state
        if item.first_access is StateUseAccess.READ
    }
    if set(seeds) != required_seed_refs:
        raise SchemaError(
            "state_seed_overrides must exactly cover every load-before-store state",
            path="state_seed_overrides",
        )
    writable_refs = {
        item.abi.state_ref
        for item in resolved_state
        if item.abi.access is PersistentStateAccess.READ_WRITE
        and any(access is StateUseAccess.WRITE for _index, access in item.uses)
    }
    if not set(expected).issubset(writable_refs):
        raise SchemaError(
            "state_expected_overrides may reference only stored READ_WRITE state",
            path="state_expected_overrides",
        )
    if (
        type(source) in _LITE_TRAIN_SOURCE_TYPES
        or type(source) in (
            LiteMoeBackwardLinkedProgram,
            LiteMoeDp4BackwardLinkedProgram,
        )
    ) and expected:
        raise SchemaError(
            "Lite timing does not accept caller-provided updated-weight expectations",
            path="state_expected_overrides",
        )
    for path, payloads in (
        ("state_seed_overrides", seeds),
        ("state_expected_overrides", expected),
    ):
        for state_ref, payload in payloads.items():
            abi = state_by_ref[state_ref][0].abi
            if len(payload) != abi.size_bytes:
                raise SchemaError(
                    "payload must cover exactly one whole state",
                    path=f"{path}[{state_ref!r}]",
                )

    terminal_abi_ids: set[str] | None
    packed_swizzle = (
        type(source) is SwizzleStandardLinkedProgram
        and admits_wang_4rank_packed_layout(source.projection)
    )
    if packed_swizzle or type(source) is UnfusedComparisonStandardLinkedProgram:
        terminal_abi_ids = (
            swizzle_terminal_abi_ids(source)
            if packed_swizzle
            else unfused_comparison_terminal_abi_ids(source)
        )
        resolved_terminal = tuple(
            item for item in resolved if item.abi.id in terminal_abi_ids
        )
        terminal_values = {item.abi.value_id for item in resolved_terminal}
    else:
        terminal_abi_ids = None
        terminal_values = _terminal_value_ids(source)
        resolved_terminal = tuple(
            item for item in resolved if item.abi.value_id in terminal_values
        )
    hbm_state_terminals = type(source) in (
        LiteMoeBackwardLinkedProgram,
        LiteMoeDp4BackwardLinkedProgram,
        S2LiteDp4TreeArLinkedProgram,
    )
    if not terminal_values and not hbm_state_terminals:
        raise SchemaError(
            "timing ProgramIo requires at least one IR1 terminal value",
            path="source.lowering_context.ir1.values",
        )
    if (
        (
            terminal_abi_ids is not None
            and {item.abi.id for item in resolved_terminal} != terminal_abi_ids
        )
        or
        {item.abi.value_id for item in resolved_terminal} != terminal_values
        or any(
            item.abi.ownership is not BufferOwnership.OWNED
            for item in resolved_terminal
        )
    ):
        raise SchemaError(
            "every IR1 terminal value must map only to one or more OWNED BufferABIs",
            path="source.lowering_context.ir1.values",
        )
    if type(source) is not S2LiteDp4TreeArLinkedProgram:
        _validate_train_terminal_abis(
            source,
            terminal_values,
            resolved_terminal,
        )

    reused_owned_roots = set()
    if packed_swizzle or (
        type(source) is UnfusedComparisonStandardLinkedProgram
        and len(source.manifest.core_streams) == 4
    ):
        roots = tuple(
            item
            for item in resolved
            if item.abi.ownership is not BufferOwnership.ALIASED
        )
        reused_owned_roots = {
            item.abi.id
            for item in roots
            if item.abi.ownership is BufferOwnership.OWNED
            and any(
                other is not item
                and other.runtime_core_id == item.runtime_core_id
                and other.abi.region_ref == item.abi.region_ref
                and item.abi.region_offset_bytes
                < other.abi.region_offset_bytes + other.abi.size_bytes
                and other.abi.region_offset_bytes
                < item.abi.region_offset_bytes + item.abi.size_bytes
                and other.abi.lifetime_end_exclusive
                <= item.abi.lifetime_start
                for other in roots
            )
        }

    initialization_blobs_by_abi = {
        item.abi.id: ProgramBlob.create(
            sram_seeds.get(item.abi.id, bytes(item.abi.size_bytes))
        )
        for item in resolved
        if item.abi.ownership is not BufferOwnership.ALIASED
        and item.abi.id not in reused_owned_roots
    }
    probe_payloads_by_abi = {
        item.abi.id: bytes(item.abi.size_bytes)
        for item in resolved_terminal
    }
    for buffer_abi_id, payload in sram_expected.items():
        previous = probe_payloads_by_abi.get(buffer_abi_id)
        if previous is not None and previous != payload:
            raise SchemaError(
                "conflicting expected payloads for one SRAM target",
                path=f"sram_expected_overrides[{buffer_abi_id!r}]",
            )
        probe_payloads_by_abi.setdefault(buffer_abi_id, payload)
    probe_blobs_by_abi = {
        buffer_abi_id: ProgramBlob.create(payload)
        for buffer_abi_id, payload in probe_payloads_by_abi.items()
    }
    blobs = {
        blob.id: blob for blob in initialization_blobs_by_abi.values()
    }
    blobs.update((blob.id, blob) for blob in probe_blobs_by_abi.values())
    seed_blobs = {
        state_ref: ProgramBlob.create(payload)
        for state_ref, payload in seeds.items()
    }
    expected_blobs = {
        state_ref: ProgramBlob.create(payload)
        for state_ref, payload in expected.items()
    }
    if hbm_state_terminals:
        expected_blobs.update(
            (
                state_ref,
                ProgramBlob.create(seeds[state_ref]),
            )
            for state_ref in sorted(writable_refs)
        )
    blobs.update((blob.id, blob) for blob in seed_blobs.values())
    blobs.update((blob.id, blob) for blob in expected_blobs.values())

    sram_initializations = tuple(
        _initialization(item, initialization_blobs_by_abi[item.abi.id])
        for item in resolved
        if item.abi.ownership is not BufferOwnership.ALIASED
        and item.abi.id not in reused_owned_roots
    )
    state_initializations = tuple(
        _state_initialization(item, seed_blobs[state_ref])
        for state_ref in sorted(seeds)
        for item in state_by_ref[state_ref]
    )
    sram_probes = tuple(
        _probe(item, probe_blobs_by_abi[item.abi.id])
        for item in resolved
        if item.abi.id in probe_payloads_by_abi
    )
    state_probes = tuple(
        _state_probe(item, expected_blobs[state_ref])
        for state_ref in sorted(expected_blobs)
        for item in state_by_ref[state_ref]
    )
    contract = ProgramIoContract.create(
        producer_pass="program_io",
        mode=ProgramIoMode.TIMING,
        source_manifest=source.manifest,
        program_artifact_sha256=program_artifact_sha256,
        blobs=tuple(blobs.values()),
        initializations=(*sram_initializations, *state_initializations),
        output_probes=(*sram_probes, *state_probes),
    )
    contract.validate_against(source.manifest)
    _validate_dp4_updated_weight_probes(source, contract, resolved_state)
    _validate_lite_moe_dp4_backward_program_io(
        source,
        contract,
        resolved,
        resolved_state,
    )
    return contract
