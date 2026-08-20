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
    RegionManifest,
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
from ..schema.lite_moe_execution import LiteMoeBufferAccess
from ..schema.lite_moe_n6 import LiteMoeLinkedProgram
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
    | LiteMoeLinkedProgram
)


_LINKED_PROGRAM_SOURCE_TYPES = (
    LinkedProgramProfile,
    Stage4LinkedProgram,
    TrainLinkedProgram,
    S2LiteTrainLinkedProgram,
    LiteMoeLinkedProgram,
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
    if type(source) is LiteMoeLinkedProgram:
        raise SchemaError(
            "S3-Lite uses its dedicated execution carrier",
            path="source",
        )
    return ((0, source.lowering_context),)


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
    if type(source) is LiteMoeLinkedProgram:
        by_binding = {abi.binding_id: abi for abi in abis.values()}
        placements = {
            placement.task_ref: placement
            for placement in source.source.schedule.placements
        }
        collected: dict[str, list[_LiteSemanticUse]] = {
            abi_id: [] for abi_id in abis
        }
        for action_index, action in enumerate(source.source.global_dag.actions):
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
                    if use.access is LiteMoeBufferAccess.READ
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
                and type(source) is S2LiteTrainLinkedProgram
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
    source: TrainLinkedProgram | S2LiteTrainLinkedProgram,
) -> tuple[tuple[int, CrossEntropyForwardWorkload, str, str], ...]:
    result: list[
        tuple[int, CrossEntropyForwardWorkload, str, str]
    ] = []
    replicas = (
        tuple(
            (replica.replica_index, replica.lowering_context)
            for replica in source.source.replicas
        )
        if type(source) is TrainLinkedProgram
        else ((0, source.source.lowering_context),)
    )
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
    if type(source) not in (TrainLinkedProgram, S2LiteTrainLinkedProgram):
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
            if type(source) is S2LiteTrainLinkedProgram:
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
    if type(source) is not S2LiteTrainLinkedProgram:
        return {}
    context = source.source.lowering_context
    nodes = tuple(
        node for node in context.ir1.nodes if node.kind is OpKind.CE_BACKWARD
    )
    if len(nodes) != 1:
        raise SchemaError(
            "S2-Lite requires exactly one CE_BACKWARD node",
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
        item for item in resolved if item.abi.value_id == loss_gradient_value_id
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
            "S2-Lite loss-gradient BufferABIs must partition logical rows",
            path="source.manifest.fragments",
        )
    result: dict[str, bytes] = {}
    for item in items:
        abi = item.abi
        if (
            abi.ownership is not BufferOwnership.BORROWED
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
                "S2-Lite loss gradient requires exact BORROWED FP32 CE_BACKWARD input2",
                path="source.manifest.fragments",
            )
        result[abi.id] = bytes(abi.size_bytes)
    return result


def _terminal_value_ids(source: LinkedProgramSource) -> set[str]:
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
    if type(source) is not S2LiteTrainLinkedProgram:
        return actual
    context = source.source.lowering_context
    ce_nodes = tuple(
        node for node in context.ir1.nodes if node.kind is OpKind.CE_FORWARD
    )
    sgd_nodes = tuple(
        node
        for node in context.ir1.nodes
        if node.kind is OpKind.OPTIMIZER_UPDATE
    )
    if len(ce_nodes) != 1 or len(sgd_nodes) != 1:
        raise SchemaError(
            "S2-Lite requires one CE_FORWARD and one OPTIMIZER_UPDATE",
            path="source.source.lowering_context.ir1.nodes",
        )
    expected = {ce_nodes[0].outputs[0], sgd_nodes[0].outputs[0]}
    if actual != expected:
        raise SchemaError(
            "S2-Lite IR1 terminals must be exactly loss and updated weight",
            path="source.source.lowering_context.ir1.values",
        )
    return {ce_nodes[0].outputs[0]}


def _validate_train_terminal_abis(
    source: LinkedProgramSource,
    terminal_values: set[str],
    resolved_terminal: tuple[_ResolvedAbi, ...],
) -> None:
    if type(source) not in (TrainLinkedProgram, S2LiteTrainLinkedProgram):
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
    if type(source) is not S2LiteTrainLinkedProgram:
        return
    context = source.source.lowering_context
    sgd_actions = tuple(
        (index, action)
        for index, action in enumerate(context.global_dag.actions)
        if action.op_kind is OpKind.OPTIMIZER_UPDATE
    )
    if len(sgd_actions) != 1:
        raise SchemaError(
            "S2-Lite requires one exact SGD GlobalAction",
            path="source.source.lowering_context.global_dag.actions",
        )
    sgd_index, sgd_action = sgd_actions[0]
    trainable = tuple(
        item
        for item in resolved_state
        if item.abi.kind is StateKind.TRAINABLE_PARAMETER
    )
    if (
        len(trainable) != 1
        or trainable[0].abi.access is not PersistentStateAccess.READ_WRITE
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
            "S2-Lite trainable state requires two loads before SGD and one store after",
            path="source.manifest.fragments",
        )
    sgd_node = next(
        node
        for node in context.ir1.nodes
        if node.kind is OpKind.OPTIMIZER_UPDATE
    )
    aliases = tuple(
        item
        for item in resolved
        if item.abi.value_id == sgd_node.outputs[0]
    )
    if len(aliases) != 1:
        raise SchemaError(
            "S2-Lite updated weight requires one alias BufferABI",
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
        or alias.uses[0].action != sgd_action
        or alias.uses[0].use.role is not BufferUseRole.COMP_OUTPUT
        or alias.uses[0].use.operand_index != 0
        or alias.uses[0].use.access is not BufferAccess.WRITE
    ):
        raise SchemaError(
            "S2-Lite updated weight must exactly alias the trainable staging root",
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
    if type(source) is LiteMoeLinkedProgram:
        action_order = {
            action.id: index
            for index, action in enumerate(source.source.global_dag.actions)
        }
        for unit in source.source.intent.state_loads:
            abi = by_binding.get(unit.hbm_binding_ref)
            if abi is None:
                raise SchemaError(
                    "S3-Lite state load has no manifest StateABI",
                    path="source.source.intent.state_loads",
                )
            uses[abi.id].append((action_order[unit.action_ref], StateUseAccess.READ))
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
    if type(source) is not LiteMoeLinkedProgram and (
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
    if type(source) in (TrainLinkedProgram, S2LiteTrainLinkedProgram) and sram_expected:
        raise SchemaError(
            "Train timing ProgramIo does not accept numeric loss expectations",
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
    if type(source) is S2LiteTrainLinkedProgram and expected:
        raise SchemaError(
            "S2-Lite timing does not claim numeric updated-weight expectations",
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

    terminal_values = _terminal_value_ids(source)
    if not terminal_values:
        raise SchemaError(
            "timing ProgramIo requires at least one IR1 terminal value",
            path="source.lowering_context.ir1.values",
        )
    resolved_terminal = tuple(
        item for item in resolved if item.abi.value_id in terminal_values
    )
    if (
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
    _validate_train_terminal_abis(
        source,
        terminal_values,
        resolved_terminal,
    )

    initialization_blobs_by_abi = {
        item.abi.id: ProgramBlob.create(
            sram_seeds.get(item.abi.id, bytes(item.abi.size_bytes))
        )
        for item in resolved
        if item.abi.ownership is not BufferOwnership.ALIASED
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
    blobs.update((blob.id, blob) for blob in seed_blobs.values())
    blobs.update((blob.id, blob) for blob in expected_blobs.values())

    sram_initializations = tuple(
        _initialization(item, initialization_blobs_by_abi[item.abi.id])
        for item in resolved
        if item.abi.ownership is not BufferOwnership.ALIASED
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
        for state_ref in sorted(expected)
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
    return contract
