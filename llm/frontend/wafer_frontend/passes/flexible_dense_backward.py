"""Materialize flexible Dense backward actions into a real symbolic manifest."""

from __future__ import annotations

import struct

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    FragmentInterface,
    FragmentKind,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    ManifestInputDigest,
    ManifestInputKind,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbol,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    SemanticOperandId,
    StateABI,
    StateOperandBinding,
)
from ..schema.common import DType, stable_artifact_id
from ..schema._validation_session import builder_validation_session
from ..schema.flexible_dense_backward import (
    FlexibleDenseBackwardLinkedProgram,
)
from .flexible_dense_backward_projection import (
    build_flexible_dense_backward_lineage,
)
from ..schema.flexible_dense_train import (
    DenseTrainParameterTemplate,
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainForwardCarrier,
    FlexibleDenseTrainPlan,
)
from ..schema.global_action import LogicalCoreRef
from ..schema.ir1 import PhysicalFabric
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.persistent_state import (
    HbmAddressSpace,
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)
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
from ..schema.serde import canonical_digest


_SCHEMA = "wafer_frontend.flexible_dense_backward_lowering/v1alpha1"


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _id(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"flexible_dense_backward_{kind}", semantic, schema_version=_SCHEMA
    )


def _buffer(
    *,
    plan: FlexibleDenseTrainPlan,
    core: LogicalCoreRef,
    region_ref: str,
    offset: int,
    template: DenseTrainParameterTemplate,
    role: str,
    dtype: DType,
    size_bytes: int,
    lifetime_end: int,
) -> BufferABI:
    elements = size_bytes // (2 if dtype is DType.FP16 else 4)
    value_id = _id(
        "value",
        {
            "plan": plan.id,
            "rank": core.die_id,
            "state": template.state_ref,
            "role": role,
        },
    )
    binding_id = _id("binding", {"value": value_id})
    semantic = {
        "schedule_id": plan.id,
        "binding_id": binding_id,
        "value_id": value_id,
        "logical_core": core,
        "tensor_slice": TensorSlice(value_id, (0,), (elements,)),
        "region_ref": region_ref,
        "region_offset_bytes": offset,
        "size_bytes": size_bytes,
        "alignment_bytes": 64,
        "banks": (),
        "storage_id": _id("storage", {"binding": binding_id}),
        "alias_of": None,
        "lifetime_start": 0,
        "lifetime_end_exclusive": lifetime_end,
        "dtype": dtype,
        "layout": f"flexible_dense_{role}_flat/v1",
        "ownership": BufferOwnership.OWNED,
    }
    return BufferABI(id=_id("buffer_abi", semantic), **semantic)


def _symbol(abi: BufferABI) -> ProgramSymbol:
    return ProgramSymbol(
        _id("sram_symbol", {"binding": abi.binding_id}),
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        abi.binding_id,
    )


def _hbm_symbol(binding_ref: str) -> ProgramSymbol:
    return ProgramSymbol(
        _id("hbm_symbol", {"binding": binding_ref}),
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        binding_ref,
    )


def _label_symbol(abi: BufferABI) -> ProgramSymbol:
    return ProgramSymbol(
        _id("label_symbol", {"storage": abi.storage_id}),
        ProgramSymbolKind.SRAM_LABEL,
        abi.storage_id,
    )


def _region_symbol(region_ref: str, core: LogicalCoreRef) -> ProgramSymbol:
    return ProgramSymbol(
        _id("region_symbol", {"region": region_ref, "core": core}),
        ProgramSymbolKind.SRAM_REGION,
        region_ref,
    )


def _matmul(
    action_id: str,
    source: ProgramSymbol,
    output: ProgramSymbol,
) -> RelocatableRecord:
    return RelocatableRecord(
        action_id,
        RecordOpcode.MATMUL,
        (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address(
                "input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                source.id,
            ),
            RecordOperand.address(
                "data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS,
                source.id,
            ),
            RecordOperand.address(
                "output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output.id,
            ),
            RecordOperand.literal(
                "parameters", (1, 8, 8, 8)
            ),
        ),
    )


def _materialize_flexible_dense_backward(
    forward: FlexibleDenseTrainForwardCarrier,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> FlexibleDenseBackwardLinkedProgram:
    """Build the real DP1 command manifest; DP transport follows in R4.1."""

    forward.validate("forward")
    plan = forward.plan
    plan.validate("plan")
    if plan.spec.mesh.rank_count > 1:
        from .flexible_dense_backward_multi import (
            materialize_flexible_dense_backward_multi,
        )
        return materialize_flexible_dense_backward_multi(
            forward, fabric, hbm_address_spaces
        )
    backward_ir, backward_projection, backward_schedule, backward_global = (
        build_flexible_dense_backward_lineage(plan)
    )
    fabric.validate("fabric")
    if plan.spec.mesh.rank_count != 1:
        raise UnsupportedFeatureError(
            "first executable slice supports only a 1x1 Mesh",
            path="plan.spec.mesh",
        )
    if fabric.die_grid != (1, 1) or len(fabric.dies) != 1:
        raise SchemaError("fabric must exactly match 1x1 Mesh", path="fabric")
    if len(hbm_address_spaces) != 1 or hbm_address_spaces[0].die_id != 0:
        raise SchemaError("requires one die0 HBM address space", path="hbm_address_spaces")
    space = hbm_address_spaces[0]
    space.validate("hbm_address_spaces[0]")
    die = fabric.dies[0]
    core_spec = die.cores[0]
    core = LogicalCoreRef(0, core_spec.local_core_id)
    profile = next(
        item for item in fabric.sram_profiles
        if item.id == core_spec.sram_profile_ref
    )
    region = profile.regions[0]
    actions = tuple(
        item for item in plan.rank_actions
        if item.kind is not FlexibleDenseTrainActionKind.FORWARD
    )
    state_decl = {item.id: item for item in plan.forward_graph.persistent_states}
    templates = tuple(
        item for item in plan.parameter_templates if 0 in item.owner_ranks
    )
    template_by_state = {item.state_ref: item for item in templates}

    sram_offset = _align(region.base_bytes, 64)
    buffers: dict[tuple[str, str], BufferABI] = {}
    for template in templates:
        for role, dtype, size in (
            ("weight", DType.FP16, template.weight_bytes),
            ("gradient", DType.FP32, template.gradient_bytes),
        ):
            carrier_size = max(size, 256)
            sram_offset = _align(sram_offset, 64)
            buffers[(template.state_ref, role)] = _buffer(
                plan=plan,
                core=core,
                region_ref=region.id,
                offset=sram_offset - region.base_bytes,
                template=template,
                role=role,
                dtype=dtype,
                size_bytes=carrier_size,
                lifetime_end=len(actions) + 1,
            )
            sram_offset += carrier_size
    if sram_offset > region.base_bytes + region.size_bytes:
        raise SchemaError("backward buffers exceed SRAM region", path="fabric")

    hbm_offset = _align(space.base_address, space.alignment_bytes)
    state_abis: dict[str, StateABI] = {}
    hbm_symbols: dict[str, ProgramSymbol] = {}
    for template in templates:
        declaration = state_decl[template.state_ref]
        hbm_offset = _align(hbm_offset, space.alignment_bytes)
        binding_ref = _id(
            "hbm_binding",
            {"state": template.state_ref, "die": 0, "address": hbm_offset},
        )
        abi = StateABI.create(
            state_ref=template.state_ref,
            hbm_binding_ref=binding_ref,
            kind=StateKind.TRAINABLE_PARAMETER,
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
            shape=declaration.shape,
            dtype=declaration.dtype,
            layout=declaration.layout,
            die_id=0,
            address=hbm_offset,
            size_bytes=template.weight_bytes,
            alignment_bytes=space.alignment_bytes,
        )
        state_abis[template.state_ref] = abi
        hbm_symbols[template.state_ref] = _hbm_symbol(binding_ref)
        hbm_offset += template.weight_bytes
    if hbm_offset > space.base_address + space.size_bytes:
        raise SchemaError("trainable states exceed HBM space", path="hbm_address_spaces")

    sram_symbols = {key: _symbol(abi) for key, abi in buffers.items()}
    label_symbols = {key: _label_symbol(abi) for key, abi in buffers.items()}
    region_symbol = _region_symbol(region.id, core)
    records: list[RelocatableRecord] = []
    relocations: list[AddressRelocation] = []
    address_closures: list[tuple[int, SemanticOperandId, BufferABI]] = []
    state_closures: list[tuple[int, StateABI]] = []

    def relocate(index: int, operand: SemanticOperandId,
                 symbol: ProgramSymbol, abi: BufferABI,
                 symbol_kind: ProgramSymbolKind =
                 ProgramSymbolKind.ABSOLUTE_ADDRESS) -> None:
        relocations.append(AddressRelocation(
            index, operand, symbol_kind, symbol.id, 0
        ))
        address_closures.append((index, operand, abi))

    def append_bind(
        action_id: str,
        inputs: tuple[tuple[ProgramSymbol, BufferABI], ...],
        output: tuple[ProgramSymbol, BufferABI],
    ) -> None:
        bind_index = len(records)
        operands = [RecordOperand.literal("input_count", len(inputs))]
        for slot in range(16):
            if slot < len(inputs):
                symbol, _ = inputs[slot]
                operands.append(RecordOperand.address(
                    f"input_label_{slot}",
                    SemanticOperandId(
                        int(SemanticOperandId.SRAM_BIND_INPUT_0) + slot
                    ),
                    symbol.id,
                ))
            else:
                operands.append(RecordOperand.literal(
                    f"input_label_{slot}", 0
                ))
        operands.append(RecordOperand.address(
            "output_label",
            SemanticOperandId.SRAM_BIND_OUTPUT,
            output[0].id,
        ))
        records.append(RelocatableRecord(
            action_id, RecordOpcode.SRAM_BIND, tuple(operands)
        ))
        for slot, (symbol, abi) in enumerate(inputs):
            relocate(
                bind_index,
                SemanticOperandId(
                    int(SemanticOperandId.SRAM_BIND_INPUT_0) + slot
                ),
                symbol,
                abi,
                ProgramSymbolKind.SRAM_LABEL,
            )
        relocate(
            bind_index,
            SemanticOperandId.SRAM_BIND_OUTPUT,
            output[0],
            output[1],
            ProgramSymbolKind.SRAM_LABEL,
        )

    first = templates[0]
    for action in actions:
        index = len(records)
        template = (
            None if action.state_ref is None
            else template_by_state[action.state_ref]
        )
        if action.kind is FlexibleDenseTrainActionKind.PARAMETER_LOAD:
            assert template is not None
            weight = buffers[(template.state_ref, "weight")]
            gradient = buffers[(template.state_ref, "gradient")]
            for role, abi in (("weight", weight), ("gradient", gradient)):
                alloc_index = len(records)
                label = label_symbols[(template.state_ref, role)]
                records.append(RelocatableRecord(
                    action.id,
                    RecordOpcode.SRAM_ALLOC_AT,
                    (
                        RecordOperand.address(
                            "region_name", SemanticOperandId.REGION_NAME,
                            region_symbol.id,
                        ),
                        RecordOperand.address(
                            "label_symbol", SemanticOperandId.LABEL_SYMBOL,
                            label.id,
                        ),
                        RecordOperand.literal(
                            "region_offset_bytes", abi.region_offset_bytes
                        ),
                        RecordOperand.literal("size_bytes", abi.size_bytes),
                        RecordOperand.literal(
                            "alignment_bytes", abi.alignment_bytes
                        ),
                        RecordOperand.literal("lifetime", 0),
                        RecordOperand.literal("spillable", False),
                    ),
                ))
                relocate(
                    alloc_index, SemanticOperandId.REGION_NAME,
                    region_symbol, abi, ProgramSymbolKind.SRAM_REGION,
                )
                relocate(
                    alloc_index, SemanticOperandId.LABEL_SYMBOL,
                    label, abi, ProgramSymbolKind.SRAM_LABEL,
                )
            index = len(records)
            weight_symbol = sram_symbols[(template.state_ref, "weight")]
            hbm_symbol = hbm_symbols[template.state_ref]
            records.append(RelocatableRecord(
                action.id,
                RecordOpcode.LSU_LOAD,
                (
                    RecordOperand.address(
                        "hbm_address", SemanticOperandId.HBM_ADDRESS,
                        hbm_symbol.id,
                    ),
                    RecordOperand.literal("size_bytes", template.weight_bytes),
                    RecordOperand.address(
                        "destination_address",
                        SemanticOperandId.DESTINATION_ADDRESS,
                        weight_symbol.id,
                    ),
                ),
            ))
            relocations.append(AddressRelocation(
                index, SemanticOperandId.HBM_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm_symbol.id, 0,
            ))
            state_closures.append((index, state_abis[template.state_ref]))
            relocate(index, SemanticOperandId.DESTINATION_ADDRESS,
                     weight_symbol, weight)
        elif action.kind in (
            FlexibleDenseTrainActionKind.BACKWARD,
            FlexibleDenseTrainActionKind.WEIGHT_GRADIENT,
        ):
            selected = first if template is None else template
            weight = buffers[(selected.state_ref, "weight")]
            gradient = buffers[(selected.state_ref, "gradient")]
            weight_symbol = sram_symbols[(selected.state_ref, "weight")]
            gradient_symbol = sram_symbols[(selected.state_ref, "gradient")]
            append_bind(
                action.id,
                ((label_symbols[(selected.state_ref, "weight")], weight),),
                (label_symbols[(selected.state_ref, "gradient")], gradient),
            )
            index = len(records)
            records.append(_matmul(
                action.id, weight_symbol, gradient_symbol,
            ))
            relocate(index, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                     weight_symbol, weight)
            relocate(index, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                     weight_symbol, weight)
            relocate(index, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                     gradient_symbol, gradient)
        elif action.kind is FlexibleDenseTrainActionKind.SGD_UPDATE:
            assert template is not None
            weight = buffers[(template.state_ref, "weight")]
            gradient = buffers[(template.state_ref, "gradient")]
            weight_symbol = sram_symbols[(template.state_ref, "weight")]
            gradient_symbol = sram_symbols[(template.state_ref, "gradient")]
            append_bind(
                action.id,
                (
                    (label_symbols[(template.state_ref, "weight")], weight),
                    (
                        label_symbols[(template.state_ref, "gradient")],
                        gradient,
                    ),
                ),
                (label_symbols[(template.state_ref, "weight")], weight),
            )
            index = len(records)
            records.append(RelocatableRecord(
                action.id,
                RecordOpcode.SGD_UPDATE,
                (
                    RecordOperand.literal("weight_datatype", 1),
                    RecordOperand.literal("gradient_datatype", 3),
                    RecordOperand.literal("output_datatype", 1),
                    RecordOperand.literal("rounding", 0),
                    RecordOperand.address(
                        "weight_address",
                        SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                        weight_symbol.id,
                    ),
                    RecordOperand.address(
                        "gradient_address",
                        SemanticOperandId.COMPUTE_DATA_ADDRESS,
                        gradient_symbol.id,
                    ),
                    RecordOperand.address(
                        "updated_weight_address",
                        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                        weight_symbol.id,
                    ),
                    RecordOperand.literal(
                        "element_count", template.weight_bytes // 2
                    ),
                    RecordOperand.literal(
                        "learning_rate_f64_bits",
                        struct.unpack(
                            "<Q", struct.pack("<d", plan.spec.learning_rate)
                        )[0],
                    ),
                    RecordOperand.literal("momentum_f64_bits", 0),
                ),
            ))
            relocate(index, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                     weight_symbol, weight)
            relocate(index, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                     gradient_symbol, gradient)
            relocate(index, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                     weight_symbol, weight)
        elif action.kind is FlexibleDenseTrainActionKind.PARAMETER_STORE:
            assert template is not None
            weight = buffers[(template.state_ref, "weight")]
            weight_symbol = sram_symbols[(template.state_ref, "weight")]
            hbm_symbol = hbm_symbols[template.state_ref]
            records.append(RelocatableRecord(
                action.id,
                RecordOpcode.LSU_STORE,
                (
                    RecordOperand.address(
                        "hbm_address", SemanticOperandId.HBM_ADDRESS,
                        hbm_symbol.id,
                    ),
                    RecordOperand.literal("size_bytes", template.weight_bytes),
                    RecordOperand.address(
                        "source_address", SemanticOperandId.SOURCE_ADDRESS,
                        weight_symbol.id,
                    ),
                ),
            ))
            relocations.append(AddressRelocation(
                index, SemanticOperandId.HBM_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm_symbol.id, 0,
            ))
            state_closures.append((index, state_abis[template.state_ref]))
            relocate(index, SemanticOperandId.SOURCE_ADDRESS,
                     weight_symbol, weight)
            for role in ("weight", "gradient"):
                abi = buffers[(template.state_ref, role)]
                label = label_symbols[(template.state_ref, role)]
                free_index = len(records)
                records.append(RelocatableRecord(
                    action.id,
                    RecordOpcode.SRAM_FREE,
                    (RecordOperand.address(
                        "symbol", SemanticOperandId.SYMBOL, label.id
                    ),),
                ))
                relocate(
                    free_index,
                    SemanticOperandId.SYMBOL,
                    label,
                    abi,
                    ProgramSymbolKind.SRAM_LABEL,
                )
        else:
            raise SchemaError(
                "DP1 backward contains an unexpected action",
                path="plan.rank_actions",
            )

    program_symbols = tuple(sorted(
        (
            *sram_symbols.values(),
            *label_symbols.values(),
            *hbm_symbols.values(),
            region_symbol,
        ),
        key=lambda item: item.id,
    ))
    lineage_manifest = forward.linked_forward.manifest
    global_lineage_ids = tuple(
        digest.artifact_id
        for digest in lineage_manifest.input_digests
        if digest.kind is ManifestInputKind.GLOBAL_ACTION_DAG
    )
    if len(global_lineage_ids) != 1:
        raise SchemaError(
            "DP1 requires one exact Train global-action lineage",
            path="forward.linked_forward.manifest.input_digests",
        )
    fragment = CommandFragment.create(
        producer_pass="flexible_dense_backward_lowering",
        source_global_dag_id=backward_global.id,
        kind=FragmentKind.STATE_IO,
        claimed_action_ids=tuple(sorted(item.id for item in actions)),
        core_streams=(CoreFragmentStream(
            core,
            tuple(records),
            (),
            tuple(sorted(
                relocations,
                key=lambda item: (item.record_index, int(item.operand_id)),
            )),
        ),),
        runtime_symbols=(),
        program_symbols=program_symbols,
        buffer_abi=tuple(sorted(buffers.values(), key=lambda item: item.id)),
        state_abi=tuple(sorted(state_abis.values(), key=lambda item: item.id)),
    )
    fragment.validate("flexible_dense_backward.fragment")

    definitions = []
    for ordinal, symbol in enumerate(program_symbols):
        abi = next(
            (item for key, item in buffers.items()
             if sram_symbols[key].id == symbol.id),
            None,
        )
        state = next(
            (item for key, item in state_abis.items()
             if hbm_symbols[key].id == symbol.id),
            None,
        )
        label_abi = next(
            (item for key, item in buffers.items()
             if label_symbols[key].id == symbol.id),
            None,
        )
        if symbol.id == region_symbol.id:
            value = region.base_bytes
            size = region.size_bytes
            name = region.name
        elif label_abi is not None:
            value = 0
            size = 0
            name = f"fd_bwd_label_{ordinal:04d}"
        elif abi is not None:
            value = region.base_bytes + abi.region_offset_bytes
            size = abi.size_bytes
            name = f"fd_bwd_abs_{ordinal:04d}"
        else:
            assert state is not None
            value = state.address
            size = state.size_bytes
            name = f"fd_bwd_hbm_{ordinal:04d}"
        definitions.append(ProgramSymbolDefinition(
            symbol,
            name,
            value,
            size,
            (core,),
        ))
    address_bindings = tuple(sorted(
        (
            AddressOperandBinding(
                fragment.id, core, index, operand,
                (abi.id,), (abi.tensor_slice,),
            )
            for index, operand, abi in address_closures
        ),
        key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ),
    ))
    state_bindings = tuple(sorted(
        (
            StateOperandBinding(
                fragment.id, core, index,
                SemanticOperandId.HBM_ADDRESS, abi.id,
            )
            for index, abi in state_closures
        ),
        key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ),
    ))
    core_binding = CoreRuntimeBinding(
        core, core_spec.id, core_spec.runtime_core_id, core_spec.sram_profile_ref
    )
    linked_stream = LinkedCoreStream(
        core,
        core_spec.runtime_core_id,
        tuple(
            LinkedRecordRef(fragment.id, index, record.source_global_action_id)
            for index, record in enumerate(records)
        ),
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="flexible_dense_backward_linker",
        capabilities=0,
        source_ir1_id=backward_ir.id,
        source_projection_id=backward_projection.id,
        source_schedule_set_id=backward_schedule.id,
        source_global_dag_id=backward_global.id,
        input_digests=tuple(sorted(
            (
                ManifestInputDigest(
                    ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_IR,
                    backward_ir.id, backward_ir.schema_version,
                    backward_ir.digest,
                ),
                ManifestInputDigest(
                    ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_PROJECTION,
                    backward_projection.id, backward_projection.schema_version,
                    backward_projection.digest,
                ),
                ManifestInputDigest(
                    ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_SCHEDULE,
                    backward_schedule.id, backward_schedule.schema_version,
                    backward_schedule.digest,
                ),
                ManifestInputDigest(
                    ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_GLOBAL,
                    backward_global.id, backward_global.schema_version,
                    backward_global.digest,
                ),
                ManifestInputDigest(
                    ManifestInputKind.COMMAND_FRAGMENT,
                    fragment.id,
                    fragment.schema_version,
                    canonical_digest(fragment),
                ),
            ),
            key=lambda item: (item.kind.value, item.artifact_id),
        )),
        fragments=(fragment,),
        fragment_interfaces=(FragmentInterface(
            fragment.id, (), (), (),
            tuple(item.id for item in program_symbols), (), (),
        ),),
        core_bindings=(core_binding,),
        core_streams=(linked_stream,),
        runtime_symbol_definitions=(),
        program_symbol_definitions=tuple(sorted(
            definitions, key=lambda item: item.symbol.id
        )),
        address_operand_bindings=address_bindings,
        state_operand_bindings=state_bindings,
        core_groups=(),
        envelope=ProgramControlEnvelope(
            (core,), (), (core,), (core,), (core,),
            EmptyCoreAckPolicy.INCLUDE_EMPTY,
            ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    manifest.validate("flexible_dense_backward.manifest")
    return FlexibleDenseBackwardLinkedProgram.create(
        plan=plan,
        forward_lineage=forward.linked_forward,
        backward_ir=backward_ir,
        backward_projection=backward_projection,
        backward_schedule=backward_schedule,
        backward_global_dag=backward_global,
        fabric=fabric,
        hbm_address_spaces=hbm_address_spaces,
        manifest=manifest,
        record_count=len(records),
        runtime_verified=False,
    )

def materialize_flexible_dense_backward(
    forward: FlexibleDenseTrainForwardCarrier,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> FlexibleDenseBackwardLinkedProgram:
    """Materialize one strict backward program in a private validation session."""

    with builder_validation_session():
        return _materialize_flexible_dense_backward(
            forward, fabric, hbm_address_spaces,
        )



def _build_flexible_dense_backward_program_io(
    source: FlexibleDenseBackwardLinkedProgram,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    """Build whole-state timing seeds/probes against the real linked manifest."""

    source.validate("source")
    manifest = source.manifest
    definitions = {
        item.symbol.source_ref: (index, item)
        for index, item in enumerate(manifest.program_symbol_definitions)
    }
    state_abis = tuple(
        abi for fragment in manifest.fragments for abi in fragment.state_abi
    )
    gradient_scratch = tuple(
        abi
        for fragment in manifest.fragments
        for abi in fragment.buffer_abi
        if abi.layout == "flexible_dense_gradient_flat/v1"
        and source.plan.spec.dp_degree > 1
    )
    blobs_by_size = {
        size: ProgramBlob.create(bytes(size))
        for size in {
            item.size_bytes for item in (*state_abis, *gradient_scratch)
        }
    }
    initializations = []
    probes = []
    runtime_core_by_logical = {
        item.logical_core: item.runtime_core_id
        for item in manifest.core_bindings
    }
    for abi in gradient_scratch:
        symbol_index, definition = definitions[abi.storage_id]
        initializations.append(ProgramSramInitialization.create(
            target=ProgramSramTarget(
                kind=ProgramIoTargetKind.SRAM,
                runtime_core_id=runtime_core_by_logical[abi.logical_core],
                program_symbol_ref=definition.symbol.id,
                finalized_symbol_index=symbol_index,
                expected_symbol_name=definition.name,
                buffer_abi_id=abi.id,
                storage_id=abi.storage_id,
                value_id=abi.value_id,
                tensor_slice=abi.tensor_slice,
                dtype=abi.dtype,
                layout=abi.layout,
            ),
            offset_bytes=0,
            length_bytes=abi.size_bytes,
            blob_ref=blobs_by_size[abi.size_bytes].id,
            purpose=ProgramIoPurpose.TIMING_PARTIAL,
        ))
    for abi in state_abis:
        symbol_index, definition = definitions[abi.hbm_binding_ref]
        target = ProgramHbmTarget(
            kind=ProgramIoTargetKind.HBM,
            program_symbol_ref=definition.symbol.id,
            finalized_symbol_index=symbol_index,
            expected_symbol_name=definition.name,
            state_abi_id=abi.id,
            state_ref=abi.state_ref,
            hbm_binding_ref=abi.hbm_binding_ref,
        )
        blob = blobs_by_size[abi.size_bytes]
        initializations.append(ProgramSramInitialization.create(
            target=target,
            offset_bytes=0,
            length_bytes=abi.size_bytes,
            blob_ref=blob.id,
            purpose=ProgramIoPurpose.STATE,
        ))
        probes.append(ProgramOutputProbe.create(
            target=target,
            offset_bytes=0,
            length_bytes=abi.size_bytes,
            blob_ref=blob.id,
            comparison=ProgramOutputComparison.EXACT_BYTES,
            capture=ProgramOutputCapture.AFTER_PROGRAM,
        ))
    result = ProgramIoContract.create(
        producer_pass="flexible_dense_backward_program_io",
        mode=ProgramIoMode.TIMING,
        source_manifest=manifest,
        program_artifact_sha256=program_artifact_sha256,
        blobs=tuple(blobs_by_size.values()),
        initializations=tuple(initializations),
        output_probes=tuple(probes),
    )
    result.validate_against(manifest)
    return result


def build_flexible_dense_backward_program_io(
    source: FlexibleDenseBackwardLinkedProgram,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    """Build strict ProgramIO in a private validation session."""

    with builder_validation_session():
        return _build_flexible_dense_backward_program_io(
            source, program_artifact_sha256,
        )


__all__ = [
    "build_flexible_dense_backward_program_io",
    "materialize_flexible_dense_backward",
]
