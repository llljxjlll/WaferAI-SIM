"""Deterministic Python-side manifest linker for the executable N6 subset."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    AddressOperandBinding,
    BufferABI,
    CommandFragment,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    EventCredit,
    FragmentInterface,
    FragmentKind,
    LinkedFragment,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    LogicalStartEvent,
    ManifestInputDigest,
    ManifestInputKind,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RelocatableRecord,
    RegionManifest,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    RuntimeOperandField,
    SemanticOperandId,
    StateABI,
    StateOperandBinding,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import GlobalAction, LogicalCoreRef
from ..schema.ir2 import BufferUseRole, SemanticTaskKind
from ..schema.serde import canonical_digest
from .context import LoweringContext

if TYPE_CHECKING:
    from ..schema.train_n6 import TrainLoweredProgram


def _input_digests(
    context: LoweringContext,
    fragments: tuple[LinkedFragment, ...],
) -> tuple[ManifestInputDigest, ...]:
    artifacts = [
        (ManifestInputKind.IR1, context.ir1),
        *((ManifestInputKind.FUSION_PLAN, plan) for plan in context.fusion_plans),
        *((ManifestInputKind.STANDALONE_PLAN, plan) for plan in context.standalone_plans),
        (ManifestInputKind.IR2_PROJECTION, context.projection),
        (ManifestInputKind.SCHEDULE_SET, context.schedule_set),
        (ManifestInputKind.GLOBAL_ACTION_DAG, context.global_dag),
    ]
    for linked in fragments:
        if isinstance(linked, RegionManifest):
            artifacts.append((ManifestInputKind.REGION_MANIFEST, linked))
            artifacts.append((ManifestInputKind.COMMAND_FRAGMENT, linked.fragment))
        else:
            artifacts.append((ManifestInputKind.COMMAND_FRAGMENT, linked))
    return tuple(
        sorted(
            (
                ManifestInputDigest(
                    kind,
                    artifact.id,
                    artifact.schema_version,
                    canonical_digest(artifact),
                )
                for kind, artifact in artifacts
            ),
            key=lambda digest: (digest.kind.value, digest.artifact_id),
        )
    )


def _train_input_digests(
    source: "TrainLoweredProgram",
    fragments: tuple[LinkedFragment, ...],
) -> tuple[ManifestInputDigest, ...]:
    artifacts: list[tuple[ManifestInputKind, object]] = [
        (ManifestInputKind.TRAIN_LOWERED_PROGRAM, source),
    ]
    for replica in source.replicas:
        context = replica.lowering_context
        artifacts.extend(
            (
                (ManifestInputKind.IR1, context.ir1),
                *((ManifestInputKind.FUSION_PLAN, plan) for plan in context.fusion_plans),
                *((ManifestInputKind.STANDALONE_PLAN, plan) for plan in context.standalone_plans),
                (ManifestInputKind.IR2_PROJECTION, context.projection),
                (ManifestInputKind.SCHEDULE_SET, context.schedule_set),
                (ManifestInputKind.GLOBAL_ACTION_DAG, context.global_dag),
            )
        )
    for linked in fragments:
        if isinstance(linked, RegionManifest):
            artifacts.append((ManifestInputKind.REGION_MANIFEST, linked))
            artifacts.append((ManifestInputKind.COMMAND_FRAGMENT, linked.fragment))
        else:
            artifacts.append((ManifestInputKind.COMMAND_FRAGMENT, linked))

    by_key: dict[tuple[str, str], ManifestInputDigest] = {}
    for kind, artifact in artifacts:
        digest = ManifestInputDigest(
            kind,
            artifact.id,
            artifact.schema_version,
            canonical_digest(artifact),
        )
        key = (kind.value, artifact.id)
        previous = by_key.setdefault(key, digest)
        if previous != digest:
            raise SchemaError(
                "conflicting Train lowering inputs share one artifact id",
                path="source.replicas",
            )
    return tuple(by_key[key] for key in sorted(by_key))


def _action_use_abi(
    action: GlobalAction,
    abi_by_schedule_binding: dict[tuple[str, str], BufferABI],
    role: BufferUseRole,
    operand_index: int,
) -> tuple[BufferABI, ...]:
    matches = list(
        use
        for use in action.buffer_uses
        if use.role is role
        and (operand_index < 0 or use.operand_index == operand_index)
    )
    if role is BufferUseRole.REDUCE_INPUT and action.reduction is not None:
        by_rank = {use.contribution_rank: use for use in matches}
        matches = [by_rank[rank] for rank in action.reduction.input_ranks]
    if not matches:
        raise SchemaError(
            "relocated operand has no matching action buffer use",
            path="fragments",
        )
    result = []
    for use in matches:
        abi = abi_by_schedule_binding.get(
            (action.source.schedule_id, use.binding_id)
        )
        if abi is None:
            raise SchemaError(
                "relocated operand has no matching BufferABI",
                path="fragments",
            )
        result.append(abi)
    return tuple(result)


def _action_operand_slices(
    action: GlobalAction,
    role: BufferUseRole,
    operand_index: int,
):
    matches = list(
        use
        for use in action.buffer_uses
        if use.role is role
        and (operand_index < 0 or use.operand_index == operand_index)
    )
    if role is BufferUseRole.REDUCE_INPUT and action.reduction is not None:
        by_rank = {use.contribution_rank: use for use in matches}
        matches = [by_rank[rank] for rank in action.reduction.input_ranks]
    return tuple(use.tensor_slice for use in matches)


def _operand_role(
    opcode: RecordOpcode,
    operand_id: SemanticOperandId,
) -> tuple[BufferUseRole, int]:
    if opcode is RecordOpcode.SRAM_BIND:
        if operand_id is SemanticOperandId.SRAM_BIND_OUTPUT:
            return (BufferUseRole.COMP_OUTPUT, 0)
        input_index = int(operand_id) - int(
            SemanticOperandId.SRAM_BIND_INPUT_0
        )
        if 0 <= input_index < 16:
            return (BufferUseRole.COMP_INPUT, input_index)
    mapping = {
        (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.ATTENTION, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.ATTENTION, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.SWIGLU, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.SWIGLU, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.ROPE_QK_EXACT, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.ROPE_QK_EXACT, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.ATTENTION_EXACT, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.ATTENTION_EXACT, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_AUX_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            2,
        ),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_DATA_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            1,
        ),
        (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.GREEDY_SAMPLE, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (
            BufferUseRole.COMP_INPUT,
            0,
        ),
        (RecordOpcode.GREEDY_SAMPLE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (
            BufferUseRole.COMP_OUTPUT,
            0,
        ),
        (RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS): (
            BufferUseRole.SEND_SOURCE,
            0,
        ),
        (RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS): (
            BufferUseRole.RECV_DESTINATION,
            0,
        ),
        (RecordOpcode.LOCAL_NOC_SEND, SemanticOperandId.SOURCE_ADDRESS): (
            BufferUseRole.SEND_SOURCE,
            0,
        ),
        (RecordOpcode.LOCAL_NOC_RECV, SemanticOperandId.DESTINATION_ADDRESS): (
            BufferUseRole.RECV_DESTINATION,
            0,
        ),
        (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS): (
            BufferUseRole.REDUCE_INPUT,
            -1,
        ),
        (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS): (
            BufferUseRole.REDUCE_OUTPUT,
            0,
        ),
        (RecordOpcode.DTE_ISSUE, SemanticOperandId.SOURCE_ADDRESS): (
            BufferUseRole.LOCAL_COPY_SOURCE,
            0,
        ),
        (RecordOpcode.DTE_ISSUE, SemanticOperandId.DESTINATION_ADDRESS): (
            BufferUseRole.LOCAL_COPY_DESTINATION,
            0,
        ),
        (RecordOpcode.LSU_LOAD, SemanticOperandId.DESTINATION_ADDRESS): (
            BufferUseRole.DMA_DESTINATION,
            0,
        ),
        (RecordOpcode.LSU_STORE, SemanticOperandId.SOURCE_ADDRESS): (
            BufferUseRole.DMA_SOURCE,
            0,
        ),
    }
    result = mapping.get((opcode, operand_id))
    if result is None:
        raise SchemaError(
            "ordinary-only linker received an unsupported relocated operand",
            path="fragments",
        )
    return result


def _lifecycle_operand_abis(
    action: GlobalAction,
    record: RelocatableRecord,
    operand_id: SemanticOperandId,
    symbol_source_ref: str,
    abi_by_schedule_binding: dict[tuple[str, str], BufferABI],
) -> tuple[BufferABI, ...]:
    candidates = tuple(
        abi_by_schedule_binding[(action.source.schedule_id, use.binding_id)]
        for use in action.buffer_uses
        if (action.source.schedule_id, use.binding_id)
        in abi_by_schedule_binding
    )
    if record.opcode is RecordOpcode.SRAM_ALLOC_AT:
        candidates = tuple(
            abi
            for abi in candidates
            if abi.lifetime_start == action.core_order_index
            and (
                (
                    operand_id is SemanticOperandId.REGION_NAME
                    and abi.region_ref == symbol_source_ref
                    and (
                        abi.region_offset_bytes,
                        abi.size_bytes,
                        abi.alignment_bytes,
                    )
                    == tuple(
                        operand.literal_value for operand in record.operands[2:5]
                    )
                )
                or (
                    operand_id is SemanticOperandId.LABEL_SYMBOL
                    and abi.storage_id == symbol_source_ref
                )
            )
        )
    elif record.opcode is RecordOpcode.SRAM_FREE:
        candidates = tuple(
            abi
            for abi in candidates
            if abi.lifetime_end_exclusive == action.core_order_index + 1
            and operand_id is SemanticOperandId.SYMBOL
            and abi.storage_id == symbol_source_ref
        )
    else:
        raise SchemaError("not a lifecycle record", path="fragments")
    unique = {abi.id: abi for abi in candidates}
    if len(unique) != 1:
        raise SchemaError(
            "lifecycle operand must identify one exact first/last-use BufferABI",
            path="fragments",
        )
    return tuple(unique.values())


def _program_name(symbol_id: str, kind: ProgramSymbolKind) -> str:
    prefix = {
        ProgramSymbolKind.ABSOLUTE_ADDRESS: "abs",
        ProgramSymbolKind.SRAM_REGION: "region",
        ProgramSymbolKind.SRAM_LABEL: "label",
    }[kind]
    return f"frontend_{prefix}_{symbol_id}"


def _transport_peer(
    action: GlobalAction,
    actions: dict[str, GlobalAction],
) -> GlobalAction:
    opposite = (
        SemanticTaskKind.RECV
        if action.task_kind is SemanticTaskKind.SEND
        else SemanticTaskKind.SEND
    )
    candidates = tuple(
        candidate
        for candidate in actions.values()
        if candidate.task_kind is opposite
        and candidate.flow_id == action.flow_id
        and candidate.source_rank == action.source_rank
        and candidate.destination_rank == action.destination_rank
    )
    if len(candidates) != 1:
        raise SchemaError(
            "transport runtime symbol requires one exact SEND/RECV peer",
            path="fragments",
        )
    return candidates[0]


def _runtime_definitions(
    actions: dict[str, GlobalAction],
    fragments: tuple[CommandFragment, ...],
) -> tuple[RuntimeSymbolDefinition, ...]:
    declarations: dict[str, RuntimeSymbol] = {}
    uses: dict[
        str,
        list[
            tuple[
                GlobalAction,
                RuntimeOperandField,
                RecordOpcode,
                LogicalCoreRef,
                str | None,
            ]
        ],
    ] = defaultdict(list)
    event_records: dict[str, list[tuple[RecordOpcode, GlobalAction]]] = defaultdict(list)
    for fragment in fragments:
        for symbol in fragment.runtime_symbols:
            previous = declarations.setdefault(symbol.id, symbol)
            if previous != symbol:
                raise SchemaError(
                    "conflicting runtime symbol declarations",
                    path="fragments",
                )
        for stream in fragment.core_streams:
            by_record: dict[int, dict[RuntimeOperandField, str]] = defaultdict(dict)
            for relocation in stream.runtime_relocations:
                by_record[relocation.record_index][relocation.field] = (
                    relocation.symbol_ref
                )
            for record_index, fields in by_record.items():
                record = stream.records[record_index]
                action = actions[record.source_global_action_id]
                event_ref = fields.get(RuntimeOperandField.EVENT_TAG)
                if event_ref is not None:
                    event_records[event_ref].append((record.opcode, action))
                for field, symbol_ref in fields.items():
                    uses[symbol_ref].append(
                        (action, field, record.opcode, stream.logical_core, event_ref)
                    )

    event_endpoints: dict[str, tuple[GlobalAction, GlobalAction]] = {}
    for symbol_ref, records in event_records.items():
        sources = tuple(action for opcode, action in records if opcode is RecordOpcode.EVENT_SET)
        destinations = tuple(action for opcode, action in records if opcode is RecordOpcode.EVENT_WAIT)
        if len(sources) != 1 or len(destinations) != 1:
            raise SchemaError(
                "event runtime symbol requires one SET and one WAIT",
                path="fragments",
            )
        event_endpoints[symbol_ref] = (sources[0], destinations[0])

    definitions: list[RuntimeSymbolDefinition] = []
    for symbol_id in sorted(declarations):
        symbol = declarations[symbol_id]
        occurrences = uses.get(symbol_id, [])
        if not occurrences:
            raise SchemaError(
                "runtime symbol is not used by a relocation",
                path="fragments",
            )
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            endpoint_actions = {occurrence[0].id: occurrence[0] for occurrence in occurrences}
            sends = tuple(
                action for action in endpoint_actions.values()
                if action.task_kind is SemanticTaskKind.SEND
            )
            recvs = tuple(
                action for action in endpoint_actions.values()
                if action.task_kind is SemanticTaskKind.RECV
            )
            if len(sends) != 1 or len(recvs) != 1 or _transport_peer(sends[0], actions) != recvs[0]:
                raise SchemaError(
                    "DTE FSM must be shared by one exact SEND/RECV pair",
                    path="fragments",
                )
            source, destination = sends[0], recvs[0]
            cores = tuple(
                sorted(
                    (source.logical_core, destination.logical_core),
                    key=lambda core: (core.die_id, core.local_core_id),
                )
            )
            definition = RuntimeSymbolDefinition(
                symbol, cores, source.id, destination.id
            )
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            endpoint_actions = {occurrence[0].id: occurrence[0] for occurrence in occurrences}
            recvs = tuple(
                action for action in endpoint_actions.values()
                if action.task_kind is SemanticTaskKind.RECV
            )
            waits = tuple(
                action for action in endpoint_actions.values()
                if action.task_kind is SemanticTaskKind.WAIT
            )
            copies = tuple(
                action for action in endpoint_actions.values()
                if action.task_kind is SemanticTaskKind.LOCAL_COPY
            )
            if len(recvs) == 1 and len(waits) == 1 and len(endpoint_actions) == 2:
                source, destination = recvs[0], waits[0]
            elif len(copies) == 1 and len(endpoint_actions) == 1:
                source, destination = copies[0], None
            else:
                raise SchemaError(
                    "DTE token must bind one RECV/WAIT pair or one LOCAL_COPY",
                    path="fragments",
                )
            definition = RuntimeSymbolDefinition(
                symbol,
                (source.logical_core,),
                source.id,
                destination.id if destination is not None else None,
            )
        elif symbol.kind is RuntimeSymbolKind.EVENT_TAG:
            endpoints = event_endpoints.get(symbol_id)
            if endpoints is None:
                raise SchemaError(
                    "EVENT_TAG lacks one exact SET/WAIT pair",
                    path="fragments",
                )
            source, destination = endpoints
            definition = RuntimeSymbolDefinition(
                symbol,
                tuple(
                    sorted(
                        (source.logical_core, destination.logical_core),
                        key=lambda core: (core.die_id, core.local_core_id),
                    )
                ),
                source.id,
                destination.id,
            )
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            expected_cores: set[LogicalCoreRef] = set()
            for action, field, _opcode, _core, event_ref in occurrences:
                if field is RuntimeOperandField.PEER_CORE:
                    expected_cores.add(_transport_peer(action, actions).logical_core)
                elif field in (
                    RuntimeOperandField.SOURCE_CORE,
                    RuntimeOperandField.DESTINATION_CORE,
                ):
                    endpoints = event_endpoints.get(event_ref or "")
                    if endpoints is None:
                        raise SchemaError(
                            "event core symbol lacks exact event endpoints",
                            path="fragments",
                        )
                    expected_cores.add(
                        endpoints[
                            0
                            if field is RuntimeOperandField.SOURCE_CORE
                            else 1
                        ].logical_core
                    )
                else:
                    raise SchemaError(
                        "runtime-core symbol has an unsupported field",
                        path="fragments",
                    )
            if len(expected_cores) != 1:
                raise SchemaError(
                    "runtime-core symbol must identify one exact logical core",
                    path="fragments",
                )
            definition = RuntimeSymbolDefinition(
                symbol, (next(iter(expected_cores)),), None, None
            )
        else:
            raise SchemaError(
                "GROUP runtime symbols are unsupported in the naive executable subset",
                path="fragments",
            )
        definition.validate("runtime_symbol_definition")
        definitions.append(definition)
    return tuple(definitions)


class NaiveManifestLinker:
    """Link canonical coarse, fused ISA-region, and standalone fragments."""

    def link_lite_moe(self, source):
        """Link the exact isolated S3-Lite MoE lowering."""

        from .lite_moe_linker import link_lite_moe_manifest

        return link_lite_moe_manifest(source)

    def link(
        self,
        context: LoweringContext,
        fragments: tuple[LinkedFragment, ...],
    ) -> LinkedProgramManifest:
        if type(context) is not LoweringContext:
            raise SchemaError("must be a LoweringContext", path="context")
        if type(fragments) is not tuple or not fragments:
            raise SchemaError(
                "must contain lowering fragments",
                path="fragments",
            )
        context.validate()
        if any(
            type(fragment) not in (CommandFragment, RegionManifest)
            for fragment in fragments
        ):
            raise SchemaError(
                "linker accepts CommandFragment or RegionManifest values",
                path="fragments",
            )
        ordered_fragments = tuple(
            sorted(
                fragments,
                key=lambda linked: linked.id,
            )
        )
        if len({fragment.id for fragment in ordered_fragments}) != len(ordered_fragments):
            raise SchemaError("duplicate fragment", path="fragments")
        leaf_fragments = tuple(
            sorted(
                (
                    linked.fragment
                    if isinstance(linked, RegionManifest)
                    else linked
                    for linked in ordered_fragments
                ),
                key=lambda fragment: fragment.id,
            )
        )
        if len({fragment.id for fragment in leaf_fragments}) != len(leaf_fragments):
            raise SchemaError("duplicate leaf fragment", path="fragments")

        actions = {
            action.id: action
            for action in context.global_dag.actions
            if action.task_kind is not SemanticTaskKind.TRANSIT
        }
        claimed: dict[str, CommandFragment] = {}
        for index, linked in enumerate(ordered_fragments):
            linked.validate_against(
                context.global_dag,
                f"fragments[{index}]",
            )
            fragment = (
                linked.fragment if isinstance(linked, RegionManifest) else linked
            )
            for action_id in fragment.claimed_action_ids:
                if action_id in claimed:
                    raise SchemaError(
                        "one action is claimed by multiple fragments",
                        path="fragments",
                    )
                claimed[action_id] = fragment
        if set(claimed) != set(actions):
            raise SchemaError(
                "fragments must cover every executable action exactly once",
                path="fragments",
            )

        abi_by_id: dict[str, BufferABI] = {}
        abi_by_schedule_binding: dict[tuple[str, str], BufferABI] = {}
        state_abi_by_id: dict[str, StateABI] = {}
        state_abi_by_hbm_binding: dict[str, StateABI] = {}
        for fragment in leaf_fragments:
            for abi in fragment.buffer_abi:
                previous = abi_by_id.setdefault(abi.id, abi)
                if previous != abi:
                    raise SchemaError("conflicting BufferABI", path="fragments")
                key = (abi.schedule_id, abi.binding_id)
                previous = abi_by_schedule_binding.setdefault(key, abi)
                if previous != abi:
                    raise SchemaError(
                        "one schedule binding has conflicting BufferABI values",
                        path="fragments",
                    )
            for abi in fragment.state_abi:
                previous = state_abi_by_id.setdefault(abi.id, abi)
                if previous != abi:
                    raise SchemaError("conflicting StateABI", path="fragments")
                previous = state_abi_by_hbm_binding.setdefault(
                    abi.hbm_binding_ref,
                    abi,
                )
                if previous != abi:
                    raise SchemaError(
                        "one HBM binding has conflicting StateABI values",
                        path="fragments",
                    )

        address_bindings: list[AddressOperandBinding] = []
        state_bindings: list[StateOperandBinding] = []
        symbol_abis: dict[str, list[BufferABI]] = defaultdict(list)
        symbol_state_abis: dict[str, list[StateABI]] = defaultdict(list)
        symbol_closures: dict[str, list[tuple[BufferABI, ...]]] = defaultdict(list)
        symbol_cores: dict[str, set[LogicalCoreRef]] = defaultdict(set)
        declared_symbols = {}
        for fragment in leaf_fragments:
            for symbol in fragment.program_symbols:
                previous = declared_symbols.setdefault(symbol.id, symbol)
                if previous != symbol:
                    raise SchemaError(
                        "conflicting program symbol declarations",
                        path="fragments",
                    )
            for stream in fragment.core_streams:
                for relocation in stream.address_relocations:
                    record = stream.records[relocation.record_index]
                    action = actions[record.source_global_action_id]
                    if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                        symbol = declared_symbols[relocation.symbol_ref]
                        abi = state_abi_by_hbm_binding.get(symbol.source_ref)
                        if (
                            abi is None
                            or abi not in fragment.state_abi
                            or symbol.kind
                            is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                            or len(action.state_uses) != 1
                            or action.state_uses[0].hbm_binding_ref
                            != abi.hbm_binding_ref
                        ):
                            raise SchemaError(
                                "HBM relocation requires one exact leaf StateABI endpoint",
                                path="fragments",
                            )
                        state_bindings.append(
                            StateOperandBinding(
                                fragment.id,
                                stream.logical_core,
                                relocation.record_index,
                                relocation.operand_id,
                                abi.id,
                            )
                        )
                        symbol_state_abis[relocation.symbol_ref].append(abi)
                        symbol_cores[relocation.symbol_ref].add(
                            stream.logical_core
                        )
                        continue
                    if record.opcode in (
                        RecordOpcode.SRAM_ALLOC_AT,
                        RecordOpcode.SRAM_FREE,
                    ):
                        symbol = declared_symbols[relocation.symbol_ref]
                        abis = _lifecycle_operand_abis(
                            action,
                            record,
                            relocation.operand_id,
                            symbol.source_ref,
                            abi_by_schedule_binding,
                        )
                        tensor_slices = tuple(abi.tensor_slice for abi in abis)
                    else:
                        role, operand_index = _operand_role(
                            record.opcode,
                            relocation.operand_id,
                        )
                        abis = _action_use_abi(
                            action,
                            abi_by_schedule_binding,
                            role,
                            operand_index,
                        )
                        tensor_slices = _action_operand_slices(
                            action,
                            role,
                            operand_index,
                        )
                    address_bindings.append(
                        AddressOperandBinding(
                            fragment.id,
                            stream.logical_core,
                            relocation.record_index,
                            relocation.operand_id,
                            tuple(abi.id for abi in abis),
                            tensor_slices,
                        )
                    )
                    symbol_abis[relocation.symbol_ref].extend(abis)
                    symbol_closures[relocation.symbol_ref].append(abis)
                    symbol_cores[relocation.symbol_ref].add(stream.logical_core)

        definitions = []
        region_specs = {
            (die.id, core.local_core_id): (
                core,
                next(
                    profile
                    for profile in context.ir1.fabric.sram_profiles
                    if profile.id == core.sram_profile_ref
                ),
            )
            for die in context.ir1.fabric.dies
            for core in die.cores
        }
        for symbol_id in sorted(declared_symbols):
            symbol = declared_symbols[symbol_id]
            abis = symbol_abis.get(symbol_id, [])
            state_abis = symbol_state_abis.get(symbol_id, [])
            if abis and state_abis:
                raise SchemaError(
                    "program symbol cannot mix BufferABI and StateABI closures",
                    path="fragments",
                )
            if not abis and not state_abis:
                raise SchemaError(
                    "program symbol is not used by a relocation",
                    path="fragments",
                )
            if state_abis:
                unique_state_abis = {abi.id: abi for abi in state_abis}
                if (
                    len(unique_state_abis) != 1
                    or symbol.kind
                    is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                ):
                    raise SchemaError(
                        "HBM absolute symbol requires one exact shared StateABI",
                        path="fragments",
                    )
                state_abi = next(iter(unique_state_abis.values()))
                if symbol.source_ref != state_abi.hbm_binding_ref:
                    raise SchemaError(
                        "HBM absolute symbol source must equal its HBM binding",
                        path="fragments",
                    )
                value, size_bytes = state_abi.address, state_abi.size_bytes
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                value, size_bytes = 0, 0
            elif symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
                closure_spans = []
                for closure in symbol_closures[symbol_id]:
                    starts = []
                    ends = []
                    for abi in closure:
                        _core, profile = region_specs[
                            (
                                abi.logical_core.die_id,
                                abi.logical_core.local_core_id,
                            )
                        ]
                        region = next(
                            item
                            for item in profile.regions
                            if item.id == abi.region_ref
                        )
                        start = region.base_bytes + abi.region_offset_bytes
                        starts.append(start)
                        ends.append(start + abi.size_bytes)
                    closure_start = min(starts)
                    closure_spans.append(
                        (closure_start, max(ends) - closure_start)
                    )
                if len(set(closure_spans)) != 1:
                    raise SchemaError(
                        "shared absolute symbol resolves to different spans",
                        path="fragments",
                    )
                value, size_bytes = closure_spans[0]
            else:
                regions = []
                for abi in abis:
                    _core, profile = region_specs[
                        (abi.logical_core.die_id, abi.logical_core.local_core_id)
                    ]
                    regions.append(
                        next(
                            item
                            for item in profile.regions
                            if item.id == abi.region_ref
                        )
                    )
                if not regions or any(
                    (item.name, item.base_bytes, item.size_bytes)
                    != (
                        regions[0].name,
                        regions[0].base_bytes,
                        regions[0].size_bytes,
                    )
                    for item in regions
                ):
                    raise SchemaError(
                        "shared SRAM region symbol resolves inconsistently",
                        path="fragments",
                    )
                value, size_bytes = regions[0].base_bytes, regions[0].size_bytes
            definitions.append(
                ProgramSymbolDefinition(
                    symbol,
                    (
                        regions[0].name
                        if symbol.kind is ProgramSymbolKind.SRAM_REGION
                        else _program_name(symbol.id, symbol.kind)
                    ),
                    value,
                    size_bytes,
                    tuple(
                        sorted(
                            symbol_cores[symbol_id],
                            key=lambda core: (core.die_id, core.local_core_id),
                        )
                    ),
                )
            )

        program_symbol_fragments: dict[str, list[str]] = defaultdict(list)
        runtime_symbol_fragments: dict[str, list[str]] = defaultdict(list)
        for fragment in leaf_fragments:
            for symbol in fragment.program_symbols:
                program_symbol_fragments[symbol.id].append(fragment.id)
            for symbol in fragment.runtime_symbols:
                runtime_symbol_fragments[symbol.id].append(fragment.id)
        interfaces = []
        for fragment in leaf_fragments:
            local_program = tuple(symbol.id for symbol in fragment.program_symbols)
            program_exports = tuple(
                symbol_id
                for symbol_id in local_program
                if fragment.id == min(program_symbol_fragments[symbol_id])
            )
            program_imports = tuple(
                symbol_id
                for symbol_id in local_program
                if symbol_id not in program_exports
            )
            local_runtime = tuple(symbol.id for symbol in fragment.runtime_symbols)
            runtime_exports = tuple(
                symbol_id
                for symbol_id in local_runtime
                if fragment.id == min(runtime_symbol_fragments[symbol_id])
            )
            runtime_imports = tuple(
                symbol_id
                for symbol_id in local_runtime
                if symbol_id not in runtime_exports
            )
            entry_counts: dict[str, int] = defaultdict(int)
            exit_counts: dict[str, int] = defaultdict(int)
            for stream in fragment.core_streams:
                for record in stream.records:
                    operands = {operand.name: operand for operand in record.operands}
                    if record.opcode is RecordOpcode.EVENT_SET:
                        tag = operands["tag"].symbol_ref
                        assert tag is not None
                        exit_counts[tag] += 1
                    elif record.opcode is RecordOpcode.EVENT_WAIT:
                        tag = operands["tag"].symbol_ref
                        count = operands["count"].literal_value
                        assert tag is not None and type(count) is int
                        entry_counts[tag] += count
            interfaces.append(
                FragmentInterface(
                    fragment.id,
                    tuple(sorted(runtime_imports)),
                    tuple(sorted(runtime_exports)),
                    tuple(sorted(program_imports)),
                    tuple(sorted(program_exports)),
                    tuple(
                        EventCredit(symbol_ref, entry_counts[symbol_ref])
                        for symbol_ref in sorted(entry_counts)
                    ),
                    tuple(
                        EventCredit(symbol_ref, exit_counts[symbol_ref])
                        for symbol_ref in sorted(exit_counts)
                    ),
                )
            )

        actions_by_core: dict[LogicalCoreRef, list[GlobalAction]] = defaultdict(list)
        for action in actions.values():
            if action.logical_core is None:
                raise SchemaError("executable action lacks a core", path="context")
            actions_by_core[action.logical_core].append(action)
        active_cores = tuple(
            sorted(actions_by_core, key=lambda core: (core.die_id, core.local_core_id))
        )
        core_bindings = []
        core_streams = []
        for logical_core in active_cores:
            die = next(
                item for item in context.ir1.fabric.dies if item.id == logical_core.die_id
            )
            core = next(
                item for item in die.cores if item.local_core_id == logical_core.local_core_id
            )
            core_bindings.append(
                CoreRuntimeBinding(
                    logical_core,
                    core.id,
                    core.runtime_core_id,
                    core.sram_profile_ref,
                )
            )
            refs = []
            for action in sorted(
                actions_by_core[logical_core],
                key=lambda item: item.core_order_index,
            ):
                fragment = claimed[action.id]
                stream = next(
                    item
                    for item in fragment.core_streams
                    if item.logical_core == logical_core
                )
                refs.extend(
                    LinkedRecordRef(fragment.id, index, action.id)
                    for index, record in enumerate(stream.records)
                    if record.source_global_action_id == action.id
                )
            core_streams.append(
                LinkedCoreStream(logical_core, core.runtime_core_id, tuple(refs))
            )

        runtime_definitions = list(_runtime_definitions(actions, leaf_fragments))
        starts = []
        first_action_by_core = {
            logical_core: min(
                actions_by_core[logical_core],
                key=lambda action: action.core_order_index,
            )
            for logical_core in active_cores
        }
        for logical_core in active_cores:
            first_action = first_action_by_core[logical_core]
            symbol = RuntimeSymbol(
                stable_artifact_id(
                    "start_tag",
                    {
                        "source_global_dag_id": context.global_dag.id,
                        "logical_core": logical_core,
                        "first_action_id": first_action.id,
                    },
                    schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
                ),
                RuntimeSymbolKind.START_TAG,
                first_action.id,
            )
            runtime_definitions.append(
                RuntimeSymbolDefinition(symbol, (logical_core,), None, None)
            )
            starts.append(LogicalStartEvent(logical_core, symbol.id, 1))

        envelope = ProgramControlEnvelope(
            active_cores,
            tuple(starts),
            active_cores,
            active_cores,
            active_cores,
            EmptyCoreAckPolicy.INCLUDE_EMPTY,
            ProgramFailurePolicy.ABORT_ALL,
        )
        manifest = LinkedProgramManifest.create(
            producer_pass="manifest_linker",
            capabilities=0,
            source_ir1_id=context.ir1.id,
            source_projection_id=context.projection.id,
            source_schedule_set_id=context.schedule_set.id,
            source_global_dag_id=context.global_dag.id,
            input_digests=_input_digests(context, ordered_fragments),
            fragments=ordered_fragments,
            fragment_interfaces=tuple(interfaces),
            core_bindings=tuple(core_bindings),
            core_streams=tuple(core_streams),
            runtime_symbol_definitions=tuple(
                sorted(runtime_definitions, key=lambda definition: definition.symbol.id)
            ),
            program_symbol_definitions=tuple(
                sorted(definitions, key=lambda definition: definition.symbol.id)
            ),
            address_operand_bindings=tuple(
                sorted(
                    address_bindings,
                    key=lambda binding: (
                        binding.logical_core.die_id,
                        binding.logical_core.local_core_id,
                        binding.fragment_id,
                        binding.fragment_record_index,
                        int(binding.operand_id),
                    ),
                )
            ),
            state_operand_bindings=tuple(
                sorted(
                    state_bindings,
                    key=lambda binding: (
                        binding.logical_core.die_id,
                        binding.logical_core.local_core_id,
                        binding.fragment_id,
                        binding.fragment_record_index,
                        int(binding.operand_id),
                    ),
                )
            ),
            core_groups=(),
            envelope=envelope,
        )
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        return manifest

    def link_train(
        self,
        source: "TrainLoweredProgram",
    ) -> LinkedProgramManifest:
        """Link all DP replicas into one strict ProgramArtifact manifest."""

        from ..schema.train_n6 import TrainLoweredProgram

        if type(source) is not TrainLoweredProgram:
            raise SchemaError(
                "must be a TrainLoweredProgram",
                path="source",
            )
        source.validate("source")
        replica_manifests = tuple(
            self.link(replica.lowering_context, replica.fragments)
            for replica in source.replicas
        )

        namespace_specs = (
            (
                "logical cores",
                lambda manifest: {
                    binding.logical_core for binding in manifest.core_bindings
                },
            ),
            (
                "runtime core ids",
                lambda manifest: {
                    binding.runtime_core_id for binding in manifest.core_bindings
                },
            ),
            (
                "runtime symbols",
                lambda manifest: {
                    definition.symbol.id
                    for definition in manifest.runtime_symbol_definitions
                },
            ),
        )
        for label, values in namespace_specs:
            seen: set[object] = set()
            for replica_index, manifest in enumerate(replica_manifests):
                local = values(manifest)
                if seen.intersection(local):
                    raise SchemaError(
                        f"Train replicas must have disjoint {label}",
                        path=f"source.replicas[{replica_index}]",
                    )
                seen.update(local)

        def flatten(attribute: str) -> tuple[object, ...]:
            return tuple(
                value
                for manifest in replica_manifests
                for value in getattr(manifest, attribute)
            )

        fragments = tuple(
            sorted(flatten("fragments"), key=lambda item: item.id)
        )
        leaf_fragments = tuple(
            linked.fragment if isinstance(linked, RegionManifest) else linked
            for linked in fragments
        )
        program_symbol_fragments: dict[str, list[str]] = defaultdict(list)
        runtime_symbol_fragments: dict[str, list[str]] = defaultdict(list)
        for fragment in leaf_fragments:
            for symbol in fragment.program_symbols:
                program_symbol_fragments[symbol.id].append(fragment.id)
            for symbol in fragment.runtime_symbols:
                runtime_symbol_fragments[symbol.id].append(fragment.id)
        interfaces = []
        for fragment in leaf_fragments:
            local_program = tuple(symbol.id for symbol in fragment.program_symbols)
            local_runtime = tuple(symbol.id for symbol in fragment.runtime_symbols)
            entry_counts: dict[str, int] = defaultdict(int)
            exit_counts: dict[str, int] = defaultdict(int)
            for stream in fragment.core_streams:
                for record in stream.records:
                    operands = {
                        operand.name: operand for operand in record.operands
                    }
                    if record.opcode is RecordOpcode.EVENT_SET:
                        tag = operands["tag"].symbol_ref
                        assert tag is not None
                        exit_counts[tag] += 1
                    elif record.opcode is RecordOpcode.EVENT_WAIT:
                        tag = operands["tag"].symbol_ref
                        count = operands["count"].literal_value
                        assert tag is not None and type(count) is int
                        entry_counts[tag] += count
            interfaces.append(
                FragmentInterface(
                    fragment.id,
                    tuple(
                        symbol_id
                        for symbol_id in local_runtime
                        if fragment.id
                        != min(runtime_symbol_fragments[symbol_id])
                    ),
                    tuple(
                        symbol_id
                        for symbol_id in local_runtime
                        if fragment.id
                        == min(runtime_symbol_fragments[symbol_id])
                    ),
                    tuple(
                        symbol_id
                        for symbol_id in local_program
                        if fragment.id
                        != min(program_symbol_fragments[symbol_id])
                    ),
                    tuple(
                        symbol_id
                        for symbol_id in local_program
                        if fragment.id
                        == min(program_symbol_fragments[symbol_id])
                    ),
                    tuple(
                        EventCredit(symbol_ref, entry_counts[symbol_ref])
                        for symbol_ref in sorted(entry_counts)
                    ),
                    tuple(
                        EventCredit(symbol_ref, exit_counts[symbol_ref])
                        for symbol_ref in sorted(exit_counts)
                    ),
                )
            )
        interfaces = tuple(sorted(interfaces, key=lambda item: item.fragment_id))
        core_bindings = tuple(
            sorted(
                flatten("core_bindings"),
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                ),
            )
        )
        core_streams = tuple(
            sorted(
                flatten("core_streams"),
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                ),
            )
        )
        runtime_definitions = tuple(
            sorted(
                flatten("runtime_symbol_definitions"),
                key=lambda item: item.symbol.id,
            )
        )
        program_definition_by_id: dict[str, ProgramSymbolDefinition] = {}
        program_name_to_id: dict[str, str] = {}
        for definition in flatten("program_symbol_definitions"):
            previous_name_id = program_name_to_id.setdefault(
                definition.name,
                definition.symbol.id,
            )
            if previous_name_id != definition.symbol.id:
                raise SchemaError(
                    "Train program symbol names must identify one symbol",
                    path="source.replicas",
                )
            previous = program_definition_by_id.get(definition.symbol.id)
            if previous is None:
                program_definition_by_id[definition.symbol.id] = definition
                continue
            if (
                previous.symbol != definition.symbol
                or previous.name != definition.name
                or previous.value != definition.value
                or previous.size_bytes != definition.size_bytes
            ):
                raise SchemaError(
                    "conflicting Train program symbol definitions",
                    path="source.replicas",
                )
            program_definition_by_id[definition.symbol.id] = (
                ProgramSymbolDefinition(
                    previous.symbol,
                    previous.name,
                    previous.value,
                    previous.size_bytes,
                    tuple(
                        sorted(
                            set(previous.logical_cores).union(
                                definition.logical_cores
                            ),
                            key=lambda core: (
                                core.die_id,
                                core.local_core_id,
                            ),
                        )
                    ),
                )
            )
        program_definitions = tuple(
            program_definition_by_id[symbol_id]
            for symbol_id in sorted(program_definition_by_id)
        )
        address_bindings = tuple(
            sorted(
                flatten("address_operand_bindings"),
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                    item.fragment_id,
                    item.fragment_record_index,
                    int(item.operand_id),
                ),
            )
        )
        state_bindings = tuple(
            sorted(
                flatten("state_operand_bindings"),
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                    item.fragment_id,
                    item.fragment_record_index,
                    int(item.operand_id),
                ),
            )
        )
        core_groups = tuple(
            sorted(flatten("core_groups"), key=lambda item: item.symbol_ref)
        )
        active_cores = tuple(binding.logical_core for binding in core_bindings)
        start_events = tuple(
            sorted(
                (
                    event
                    for manifest in replica_manifests
                    for event in manifest.envelope.start_events
                ),
                key=lambda item: (
                    item.target_core.die_id,
                    item.target_core.local_core_id,
                    item.tag_symbol_ref,
                ),
            )
        )
        envelope = ProgramControlEnvelope(
            active_cores,
            start_events,
            active_cores,
            active_cores,
            active_cores,
            EmptyCoreAckPolicy.INCLUDE_EMPTY,
            ProgramFailurePolicy.ABORT_ALL,
        )
        manifest = LinkedProgramManifest.create(
            producer_pass="train_manifest_linker",
            capabilities=0,
            source_ir1_id=source.source_planned_carrier_id,
            source_projection_id=source.source_projected_carrier_id,
            source_schedule_set_id=source.source_scheduled_carrier_id,
            source_global_dag_id=source.source_global_action_carrier_id,
            input_digests=_train_input_digests(source, fragments),
            fragments=fragments,
            fragment_interfaces=interfaces,
            core_bindings=core_bindings,
            core_streams=core_streams,
            runtime_symbol_definitions=runtime_definitions,
            program_symbol_definitions=program_definitions,
            address_operand_bindings=address_bindings,
            state_operand_bindings=state_bindings,
            core_groups=core_groups,
            envelope=envelope,
        )
        manifest.validate("train_linked_program.manifest")
        return manifest
