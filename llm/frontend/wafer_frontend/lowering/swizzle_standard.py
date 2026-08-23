"""Lower a fully witnessed Swizzle projection to the standard command ABI."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    EventCredit,
    FragmentKind,
    FragmentInterface,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    LogicalStartEvent,
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
    RuntimeOperandField,
    RuntimeRelocation,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from ..schema.common import DType, MeshAxisName, stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.swizzle import SwizzleActionKind, SwizzleAlgorithm
from ..schema.swizzle_abi import SwizzleCoreAddressABI
from ..schema.swizzle_ir2 import (
    SwizzleIr2Projection,
    admits_wang_4rank_packed_layout,
)
from ..schema.swizzle_lowering import SwizzleLoweredProgram
from ..schema.swizzle_operand_abi import (
    SwizzleDteDirection,
    SwizzleOperandABI,
)
from ..schema.swizzle_plan import SwizzleFusionPlan
from ..schema.swizzle_plan import SwizzleValueUse
from ..schema.swizzle_standard import SwizzleStandardLinkedProgram
from ..schema.serde import canonical_digest


_PRODUCER = "swizzle_standard_lowering"
_SCHEMA = "wafer_frontend.swizzle_standard_lowering/v1alpha1"
_TERMINAL_ROOT_LAYOUT = "swizzle_standard_terminal_root/v1"
_TERMINAL_SUBVIEW_LAYOUT = "swizzle_standard_terminal_subview/v1"
_STORAGE_ROOT_LAYOUT = "swizzle_standard_storage_root/v1"
_STORAGE_SUBVIEW_LAYOUT = "swizzle_standard_storage_subview/v1"


def _id(kind: str, semantic: object) -> str:
    return stable_artifact_id(f"swizzle_standard_{kind}", semantic, schema_version=_SCHEMA)


def _dtype_code(dtype: DType) -> int:
    if dtype is DType.FP16:
        return 0
    if dtype is DType.FP32:
        return 1
    raise SchemaError("standard Swizzle supports FP16/FP32 reduction dtypes", path="operand_abi")


def _region(ir1: IR1, binding) -> object:
    core = next(
        core
        for die in ir1.fabric.dies
        if die.id == binding.logical_core.die_id
        for core in die.cores
        if core.local_core_id == binding.logical_core.local_core_id
    )
    profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
    return next(item for item in profile.regions if item.id == binding.region_ref)


def _derive_legacy_ar_buffers(
    ir1: IR1,
    projection: SwizzleIr2Projection,
    core_abi: SwizzleCoreAddressABI,
    operand_abi: SwizzleOperandABI,
) -> tuple[BufferABI, ...]:
    """Preserve the proven S0 AR physical-placement ABI until AR is packed."""

    orders = {item.task_ref: item.core_order for item in core_abi.task_bindings}
    views_by_key: dict[tuple[str, int], list[object]] = defaultdict(list)
    for view in operand_abi.operands:
        views_by_key[(view.value_ref, view.slot)].append(view)
    address_by_key = {
        (item.value_ref, item.slot): item for item in core_abi.value_bindings
    }
    projected_values = {
        item.id: item for dag in projection.rank_dags for item in dag.values
    }
    boundary_outputs = {
        ref
        for ownership in projection.output_ownership
        for ref in ownership.boundary_output_refs
    }
    keys_by_storage: dict[
        tuple[object, object], list[tuple[str, int]]
    ] = defaultdict(list)
    for key, binding in address_by_key.items():
        if key in views_by_key:
            symbolic_ref = projected_values[key[0]].symbolic_ref
            logical_terminal = (
                projection.pattern.value == "gemm_ar"
                and any(
                    symbolic_ref == boundary_ref
                    or symbolic_ref.startswith(f"{boundary_ref}::")
                    for boundary_ref in boundary_outputs
                )
            )
            keys_by_storage[
                (
                    binding.logical_core,
                    ("terminal", key)
                    if logical_terminal else
                    ("physical", binding.address, binding.size_bytes),
                )
            ].append(key)
    reduce_accumulators = {
        (views[1].value_ref, views[1].slot)
        for contract in operand_abi.reduce_contracts
        for views in [sorted(
            (
                view for view in operand_abi.operands
                if view.task_ref == contract.task_ref
            ),
            key=lambda item: item.ordinal,
        )]
    }
    result = []
    for storage_key, group_keys in keys_by_storage.items():
        root_key = next(
            (key for key in group_keys if key in reduce_accumulators),
            min(group_keys),
        )
        all_orders = [
            orders[view.task_ref]
            for key in group_keys for view in views_by_key[key]
        ]
        lifetime = (min(all_orders), max(all_orders) + 1)
        root_binding_id = _id(
            "buffer_binding", {"core_abi": core_abi.id, "key": root_key}
        )
        storage_id = _id(
            "storage", {"core_abi": core_abi.id, "storage": storage_key}
        )
        for key in sorted(group_keys):
            address = address_by_key[key]
            views = views_by_key[key]
            first = views[0]
            if any(
                (
                    view.shape, view.layout, view.dtype,
                    view.byte_offset, view.byte_extent,
                )
                != (
                    first.shape, first.layout, first.dtype,
                    first.byte_offset, first.byte_extent,
                )
                for view in views[1:]
            ):
                raise SchemaError(
                    "one value-slot has inconsistent typed views",
                    path="operand_abi.operands",
                )
            binding_id = _id(
                "buffer_binding", {"core_abi": core_abi.id, "key": key}
            )
            alias = key != root_key
            first_use = min(
                views, key=lambda view: (orders[view.task_ref], view.ordinal)
            ).use
            region = _region(ir1, address)
            semantic = {
                "schedule_id": core_abi.id,
                "binding_id": binding_id,
                "value_id": key[0],
                "logical_core": address.logical_core,
                "tensor_slice": TensorSlice(
                    key[0], (0,) * len(first.shape), first.shape
                ),
                "region_ref": address.region_ref,
                "region_offset_bytes": address.address - region.base_bytes,
                "size_bytes": address.size_bytes,
                "alignment_bytes": address.alignment_bytes,
                "banks": (),
                "storage_id": storage_id,
                "alias_of": root_binding_id if alias else None,
                "lifetime_start": (
                    min(orders[view.task_ref] for view in views)
                    if alias else lifetime[0]
                ),
                "lifetime_end_exclusive": (
                    max(orders[view.task_ref] for view in views) + 1
                    if alias else lifetime[1]
                ),
                "dtype": first.dtype,
                "layout": first.layout,
                "ownership": (
                    BufferOwnership.ALIASED
                    if alias else BufferOwnership.BORROWED
                    if first_use is SwizzleValueUse.READ
                    else BufferOwnership.OWNED
                ),
            }
            result.append(BufferABI(
                id=_id("buffer_abi", semantic), **semantic
            ))
    return tuple(sorted(result, key=lambda item: item.id))


def _derive_buffers(
    ir1: IR1,
    projection: SwizzleIr2Projection,
    core_abi: SwizzleCoreAddressABI,
    operand_abi: SwizzleOperandABI,
) -> tuple[BufferABI, ...]:
    if not admits_wang_4rank_packed_layout(projection):
        return _derive_legacy_ar_buffers(
            ir1, projection, core_abi, operand_abi
        )
    orders = {item.task_ref: item.core_order for item in core_abi.task_bindings}
    projected_values = {
        value.id: value for dag in projection.rank_dags for value in dag.values
    }
    views_by_key: dict[tuple[str, int], list[object]] = defaultdict(list)
    for view in operand_abi.operands:
        views_by_key[(view.value_ref, view.slot)].append(view)
    address_by_key = {
        (item.value_ref, item.slot): item for item in core_abi.value_bindings
    }
    keys_by_storage: dict[tuple[object, str, str], list[tuple[str, int]]] = defaultdict(list)
    for key, binding in address_by_key.items():
        if key in views_by_key:
            keys_by_storage[
                (binding.logical_core, binding.region_ref, binding.storage_ref)
            ].append(key)

    terminal_by_storage: dict[tuple[object, str, str], str] = {}
    for ownership in projection.output_ownership:
        for boundary_ref in ownership.boundary_output_refs:
            keys = tuple(
                key
                for key in views_by_key
                if address_by_key[key].rank == ownership.rank
                and (
                    projected_values[key[0]].symbolic_ref == boundary_ref
                    or projected_values[key[0]].symbolic_ref.startswith(
                        f"{boundary_ref}::"
                    )
                )
                and not projected_values[key[0]].consumer_task_refs
            )
            if not keys:
                if projection.pattern.value == "gemm_ar":
                    continue
                raise SchemaError(
                    "terminal output lacks exact typed chunk bindings",
                    path="projection.output_ownership",
                )
            storage_keys = {
                (
                    address_by_key[key].logical_core,
                    address_by_key[key].region_ref,
                    address_by_key[key].storage_ref,
                )
                for key in keys
            }
            if len(storage_keys) != 1:
                raise SchemaError(
                    "one rank-local terminal output requires one storage_ref",
                    path="core_abi.value_bindings",
                )
            storage_key = next(iter(storage_keys))
            previous = terminal_by_storage.setdefault(storage_key, boundary_ref)
            if previous != boundary_ref:
                raise SchemaError(
                    "terminal storage cannot combine boundary outputs",
                    path="core_abi.value_bindings",
                )

    ir1_values = {item.id: item for item in ir1.values}
    result = []
    for storage_key, group_keys in keys_by_storage.items():
        group_keys = sorted(group_keys)
        bindings = [address_by_key[key] for key in group_keys]
        bases = {item.address - item.storage_offset_bytes for item in bindings}
        if len(bases) != 1:
            raise SchemaError(
                "storage_ref does not resolve to one physical root base",
                path="core_abi.value_bindings",
            )
        storage_id = _id(
            "storage", {"core_abi": core_abi.id, "storage": storage_key}
        )
        terminal_ref = terminal_by_storage.get(storage_key)
        storage_subviews = False
        if terminal_ref is None:
            placements = {
                (item.address, item.size_bytes, item.alignment_bytes)
                for item in bindings
            }
            root_lifetime = (
                min(item.lifetime_start for item in bindings),
                max(item.lifetime_end_exclusive for item in bindings),
            )
            if len(placements) == 1:
                root_key = min(
                    group_keys,
                    key=lambda key: (
                        address_by_key[key].lifetime_start,
                        address_by_key[key].lifetime_end_exclusive,
                        key,
                    ),
                )
                root_binding_id = _id(
                    "buffer_binding", {"core_abi": core_abi.id, "key": root_key}
                )
            else:
                storage_subviews = True
                root_key = None
                root_binding_id = _id(
                    "storage_root_binding",
                    {"core_abi": core_abi.id, "storage": storage_key},
                )
                dtypes = {
                    view.dtype
                    for key in group_keys for view in views_by_key[key]
                }
                if len(dtypes) != 1:
                    raise SchemaError(
                        "one fused physical storage root requires one dtype",
                        path="operand_abi.operands",
                    )
                dtype = next(iter(dtypes))
                element_bytes = {DType.FP16: 2, DType.FP32: 4}.get(dtype)
                span = max(
                    item.storage_offset_bytes + item.size_bytes
                    for item in bindings
                )
                if element_bytes is None or span % element_bytes:
                    raise SchemaError(
                        "fused physical storage span requires a dense typed extent",
                        path="core_abi.value_bindings",
                    )
                first_use = min(
                    (
                        (orders[view.task_ref], view.ordinal, view.use)
                        for key in group_keys for view in views_by_key[key]
                    ),
                    key=lambda item: item[:2],
                )[2]
                root_value_id = _id(
                    "storage_root_value",
                    {"core_abi": core_abi.id, "storage": storage_key},
                )
                root_semantic = {
                    "schedule_id": core_abi.id,
                    "binding_id": root_binding_id,
                    "value_id": root_value_id,
                    "logical_core": bindings[0].logical_core,
                    "tensor_slice": TensorSlice(
                        root_value_id, (0,), (span // element_bytes,)
                    ),
                    "region_ref": bindings[0].region_ref,
                    "region_offset_bytes": next(iter(bases))
                    - _region(ir1, bindings[0]).base_bytes,
                    "size_bytes": span,
                    "alignment_bytes": bindings[0].alignment_bytes,
                    "banks": (),
                    "storage_id": storage_id,
                    "alias_of": None,
                    "lifetime_start": root_lifetime[0],
                    "lifetime_end_exclusive": root_lifetime[1],
                    "dtype": dtype,
                    "layout": _STORAGE_ROOT_LAYOUT,
                    "ownership": (
                        BufferOwnership.BORROWED
                        if first_use is SwizzleValueUse.READ
                        else BufferOwnership.OWNED
                    ),
                }
                result.append(BufferABI(
                    id=_id("buffer_abi", root_semantic), **root_semantic
                ))
        else:
            root_key = None
            root_binding_id = _id(
                "terminal_root_binding",
                {
                    "core_abi": core_abi.id,
                    "storage": storage_key,
                    "boundary_ref": terminal_ref,
                },
            )
            root_lifetime = (
                min(item.lifetime_start for item in bindings),
                max(item.lifetime_end_exclusive for item in bindings),
            )
            offsets = tuple(sorted(
                (item.storage_offset_bytes, item.size_bytes) for item in bindings
            ))
            cursor = 0
            for offset, size in offsets:
                if offset != cursor:
                    raise SchemaError(
                        "terminal chunks must exactly pack one contiguous storage root",
                        path="core_abi.value_bindings",
                    )
                cursor += size
            output = ir1_values[terminal_ref]
            shape = list(output.shape)
            tensor_offset = [0] * len(shape)
            tp_axes = tuple(
                index
                for index, axis in enumerate(output.sharding.dim_map)
                if axis is MeshAxisName.TP
            )
            if len(tp_axes) > 1:
                raise SchemaError(
                    "terminal root supports at most one TP-sharded dimension",
                    path=f"ir1.values[{terminal_ref!r}].sharding",
                )
            rank = bindings[0].rank
            if tp_axes:
                axis = tp_axes[0]
                degree = len(projection.rank_dags)
                if shape[axis] % degree:
                    raise SchemaError(
                        "terminal tensor is not exactly TP-divisible",
                        path=f"ir1.values[{terminal_ref!r}].shape",
                    )
                shape[axis] //= degree
                tensor_offset[axis] = rank * shape[axis]
            root_semantic = {
                "schedule_id": core_abi.id,
                "binding_id": root_binding_id,
                "value_id": terminal_ref,
                "logical_core": bindings[0].logical_core,
                "tensor_slice": TensorSlice(
                    terminal_ref, tuple(tensor_offset), tuple(shape)
                ),
                "region_ref": bindings[0].region_ref,
                "region_offset_bytes": next(iter(bases))
                - _region(ir1, bindings[0]).base_bytes,
                "size_bytes": cursor,
                "alignment_bytes": bindings[0].alignment_bytes,
                "banks": (),
                "storage_id": storage_id,
                "alias_of": None,
                "lifetime_start": root_lifetime[0],
                "lifetime_end_exclusive": root_lifetime[1],
                "dtype": output.dtype,
                "layout": _TERMINAL_ROOT_LAYOUT,
                "ownership": BufferOwnership.OWNED,
            }
            result.append(BufferABI(id=_id("buffer_abi", root_semantic), **root_semantic))

        for key in sorted(group_keys):
            address = address_by_key[key]
            views = views_by_key[key]
            if not views:
                raise SchemaError("address binding lacks a typed operand view", path="core_abi.value_bindings")
            first = views[0]
            if any(
                (view.shape, view.layout, view.dtype, view.byte_offset, view.byte_extent)
                != (first.shape, first.layout, first.dtype, first.byte_offset, first.byte_extent)
                for view in views[1:]
            ):
                raise SchemaError("one value-slot has inconsistent typed views", path="operand_abi.operands")
            binding_id = _id("buffer_binding", {"core_abi": core_abi.id, "key": key})
            alias = terminal_ref is not None or storage_subviews or key != root_key
            first_use = min(
                views,
                key=lambda view: (orders[view.task_ref], view.ordinal),
            ).use
            region = _region(ir1, address)
            if terminal_ref is not None:
                root = next(
                    item
                    for item in result
                    if item.binding_id == root_binding_id
                )
                chunk_axes = tuple(
                    index
                    for index, (chunk, whole)
                    in enumerate(zip(first.shape, root.tensor_slice.shape, strict=True))
                    if chunk != whole
                )
                if not chunk_axes and len(group_keys) == 1:
                    tensor_slice = TensorSlice(
                        terminal_ref,
                        root.tensor_slice.offset,
                        first.shape,
                    )
                    value_id = terminal_ref
                    layout = _TERMINAL_SUBVIEW_LAYOUT
                elif len(chunk_axes) != 1:
                    raise SchemaError(
                        "terminal chunk must partition exactly one tensor dimension",
                        path="operand_abi.operands",
                    )
                else:
                    chunk_axis = chunk_axes[0]
                    if address.storage_offset_bytes % address.size_bytes:
                        raise SchemaError(
                            "terminal storage offset must be chunk aligned",
                            path="core_abi.value_bindings",
                        )
                    chunk_index = address.storage_offset_bytes // address.size_bytes
                    tensor_offset = list(root.tensor_slice.offset)
                    tensor_offset[chunk_axis] += chunk_index * first.shape[chunk_axis]
                    tensor_slice = TensorSlice(
                        terminal_ref, tuple(tensor_offset), first.shape
                    )
                    value_id = terminal_ref
                    layout = _TERMINAL_SUBVIEW_LAYOUT
            elif storage_subviews:
                tensor_slice = TensorSlice(
                    key[0], (0,) * len(first.shape), first.shape
                )
                value_id = key[0]
                layout = _STORAGE_SUBVIEW_LAYOUT
            else:
                tensor_slice = TensorSlice(
                    key[0], (0,) * len(first.shape), first.shape
                )
                value_id = key[0]
                layout = first.layout
            semantic = {
                "schedule_id": core_abi.id,
                "binding_id": binding_id,
                "value_id": value_id,
                "logical_core": address.logical_core,
                "tensor_slice": tensor_slice,
                "region_ref": address.region_ref,
                "region_offset_bytes": address.address - region.base_bytes,
                "size_bytes": address.size_bytes,
                "alignment_bytes": address.alignment_bytes,
                "banks": (),
                "storage_id": storage_id,
                "alias_of": root_binding_id if alias else None,
                "lifetime_start": root_lifetime[0] if not alias else address.lifetime_start,
                "lifetime_end_exclusive": root_lifetime[1] if not alias else address.lifetime_end_exclusive,
                "dtype": first.dtype,
                "layout": layout,
                "ownership": (
                    BufferOwnership.ALIASED
                    if alias
                    else BufferOwnership.BORROWED
                    if first_use is SwizzleValueUse.READ
                    else BufferOwnership.OWNED
                ),
            }
            result.append(BufferABI(id=_id("buffer_abi", semantic), **semantic))
    return tuple(sorted(result, key=lambda item: item.id))


def _relocations(records: list[RelocatableRecord]) -> tuple[tuple[RuntimeRelocation, ...], tuple[AddressRelocation, ...]]:
    runtime = []
    address = []
    for record_index, record in enumerate(records):
        for operand in record.operands:
            if operand.runtime_field is not None:
                runtime.append(RuntimeRelocation(record_index, operand.runtime_field, operand.symbol_ref))
            elif operand.operand_id is not None:
                symbol_kind = {
                    SemanticOperandId.REGION_NAME: ProgramSymbolKind.SRAM_REGION,
                    SemanticOperandId.LABEL_SYMBOL: ProgramSymbolKind.SRAM_LABEL,
                    SemanticOperandId.SYMBOL: ProgramSymbolKind.SRAM_LABEL,
                    **{
                        SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index): ProgramSymbolKind.SRAM_LABEL
                        for index in range(16)
                    },
                    SemanticOperandId.SRAM_BIND_OUTPUT: ProgramSymbolKind.SRAM_LABEL,
                }.get(operand.operand_id, ProgramSymbolKind.ABSOLUTE_ADDRESS)
                address.append(AddressRelocation(record_index, operand.operand_id, symbol_kind, operand.symbol_ref, 0))
    runtime.sort(key=lambda item: (item.record_index, list(RuntimeOperandField).index(item.field)))
    address.sort(key=lambda item: (item.record_index, int(item.operand_id)))
    return tuple(runtime), tuple(address)


def lower_swizzle_standard_fragment(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    lowered: SwizzleLoweredProgram,
    core_abi: SwizzleCoreAddressABI,
    operand_abi: SwizzleOperandABI,
) -> CommandFragment:
    """Build one standard fragment, consuming only explicitly witnessed ABI facts."""

    lowered.validate_against(plan, projection)
    operand_abi.validate_against(ir1, plan, projection, core_abi)
    buffers = _derive_buffers(ir1, projection, core_abi, operand_abi)
    # Binding IDs are the authoritative key; rebuild value-slot mapping without shape guessing.
    buffer_by_key = {
        key: next(item for item in buffers if item.binding_id == _id("buffer_binding", {"core_abi": core_abi.id, "key": key}))
        for key in {(view.value_ref, view.slot) for view in operand_abi.operands}
    }
    task_bindings = {item.task_ref: item for item in core_abi.task_bindings}
    runtime_bindings = {item.task_ref: item for item in core_abi.runtime_bindings}
    views_by_task = defaultdict(list)
    for view in operand_abi.operands:
        views_by_task[view.task_ref].append(view)
    matmuls = {item.task_ref: item for item in operand_abi.matmul_contracts}
    dtes = {item.task_ref: item for item in operand_abi.dte_contracts}
    reduces = {item.task_ref: item for item in operand_abi.reduce_contracts}
    events_by_owner = defaultdict(list)
    for event in core_abi.barrier_events:
        events_by_owner[event.owner_task_ref].append(event)

    program_symbols: dict[str, ProgramSymbol] = {}
    runtime_symbols: dict[str, RuntimeSymbol] = {}

    def program(kind: ProgramSymbolKind, source_ref: str) -> str:
        symbol_id = _id("program_symbol", {"kind": kind, "source_ref": source_ref})
        program_symbols.setdefault(symbol_id, ProgramSymbol(symbol_id, kind, source_ref))
        return symbol_id

    def runtime(kind: RuntimeSymbolKind, symbol_id: str, source_ref: str) -> str:
        runtime_symbols.setdefault(symbol_id, RuntimeSymbol(symbol_id, kind, source_ref))
        return symbol_id

    abs_symbol = {key: program(ProgramSymbolKind.ABSOLUTE_ADDRESS, abi.binding_id) for key, abi in buffer_by_key.items()}
    label_symbol = {abi.storage_id: program(ProgramSymbolKind.SRAM_LABEL, abi.storage_id) for abi in buffers}
    region_symbol = {abi.region_ref: program(ProgramSymbolKind.SRAM_REGION, abi.region_ref) for abi in buffers}
    roots = {abi.storage_id: abi for abi in buffers if abi.alias_of is None}
    tasks = {task.id: task for dag in projection.rank_dags for task in dag.tasks}
    records_by_core: dict[object, list[RelocatableRecord]] = defaultdict(list)

    for dag in projection.rank_dags:
        for task in sorted(dag.tasks, key=lambda item: task_bindings[item.id].core_order):
            owner = task.id
            core = task_bindings[owner].logical_core
            records = records_by_core[core]
            views = sorted(views_by_task[owner], key=lambda item: item.ordinal)
            used_roots = {
                buffer_by_key[(view.value_ref, view.slot)].storage_id: roots[buffer_by_key[(view.value_ref, view.slot)].storage_id]
                for view in views
            }
            for abi in sorted(used_roots.values(), key=lambda item: (item.region_ref, item.region_offset_bytes, item.id)):
                if abi.lifetime_start == task_bindings[owner].core_order:
                    region = _region(
                        ir1,
                        next(
                            item for item in core_abi.value_bindings
                            if item.logical_core == abi.logical_core
                            and item.region_ref == abi.region_ref
                        ),
                    )
                    records.append(RelocatableRecord(owner, RecordOpcode.SRAM_ALLOC_AT, (
                        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbol[abi.region_ref]),
                        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label_symbol[abi.storage_id]),
                        RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
                        RecordOperand.literal("size_bytes", abi.size_bytes),
                        RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
                        RecordOperand.literal("lifetime", 0),
                        RecordOperand.literal("spillable", region.spillable),
                    )))

            if task.kind is SwizzleActionKind.COMP:
                contract = matmuls[owner]
                if plan.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS:
                    if task.chunk_index == 0:
                        if len(views) != 3:
                            raise SchemaError(
                                "first MeshSlice MATMUL requires lhs/rhs/output views",
                                path="operand_abi.operands",
                            )
                        input0, weight, output = views
                        if output.use is not SwizzleValueUse.WRITE:
                            raise SchemaError(
                                "first MeshSlice output must be a pure write",
                                path="operand_abi.operands",
                            )
                    elif len(views) != 4:
                        raise SchemaError(
                            "MeshSlice MATMUL requires lhs/rhs/accumulator/output views",
                            path="operand_abi.operands",
                        )
                    else:
                        input0, weight, accumulator, output = views
                        if (
                            accumulator.use is not SwizzleValueUse.READ
                            or output.use is not SwizzleValueUse.WRITE
                            or (accumulator.value_ref, accumulator.slot) !=
                                (output.value_ref, output.slot)
                        ):
                            raise SchemaError(
                                "MeshSlice output must exactly alias its accumulator",
                                path="operand_abi.operands",
                            )
                else:
                    if len(views) != 3:
                        raise SchemaError(
                            "MATMUL requires exactly two inputs and one output",
                            path="operand_abi.operands",
                        )
                    input0, weight, output = views
                input_abi = buffer_by_key[(input0.value_ref, input0.slot)]
                output_abi = buffer_by_key[(output.value_ref, output.slot)]
                bind_operands = [RecordOperand.literal("input_count", 1)]
                bind_operands.append(RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0, label_symbol[input_abi.storage_id]))
                bind_operands.extend(RecordOperand.literal(f"input_label_{index}", 0) for index in range(1, 16))
                bind_operands.append(RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT, label_symbol[output_abi.storage_id]))
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_BIND, tuple(bind_operands)))
                records.append(RelocatableRecord(owner, RecordOpcode.MATMUL, (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, abs_symbol[(input0.value_ref, input0.slot)]),
                    RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, abs_symbol[(weight.value_ref, weight.slot)]),
                    RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, abs_symbol[(output.value_ref, output.slot)]),
                    RecordOperand.literal("parameters", (1, contract.m, contract.k, contract.n)),
                )))
            elif task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                contract = dtes[owner]
                binding = runtime_bindings[owner]
                assert binding.fsm_symbol_ref and binding.peer_symbol_ref
                runtime(RuntimeSymbolKind.DTE_FSM, binding.fsm_symbol_ref, binding.flow_ref)
                runtime(RuntimeSymbolKind.RUNTIME_CORE, binding.peer_symbol_ref, str(binding.peer_core))
                common_tail = (
                    RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, binding.peer_symbol_ref),
                    RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
                    RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0), RecordOperand.literal("epoch", 0),
                )
                view = views[0]
                if task.kind is SwizzleActionKind.SEND:
                    operands = (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0), RecordOperand.literal("completion", 1),
                        RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, binding.fsm_symbol_ref), RecordOperand.literal("token", 0),
                        RecordOperand.literal("length_bytes", contract.logical_bytes),
                        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, abs_symbol[(view.value_ref, view.slot)]), *common_tail,
                    )
                    opcode = RecordOpcode.DTE_SEND
                else:
                    assert binding.token_symbol_ref
                    runtime(RuntimeSymbolKind.DTE_TOKEN, binding.token_symbol_ref, owner)
                    operands = (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0), RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, binding.fsm_symbol_ref),
                        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref), RecordOperand.literal("length_bytes", contract.logical_bytes),
                        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_symbol[(view.value_ref, view.slot)]), *common_tail,
                    )
                    opcode = RecordOpcode.DTE_RECV
                records.append(RelocatableRecord(owner, opcode, operands))
            elif task.kind is SwizzleActionKind.WAIT:
                binding = runtime_bindings[owner]
                assert binding.token_symbol_ref
                runtime(RuntimeSymbolKind.DTE_TOKEN, binding.token_symbol_ref, next(iter(task.deps)))
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref),
                )))
            elif task.kind is SwizzleActionKind.LOCAL_COPY:
                contract = dtes[owner]
                binding = runtime_bindings[owner]
                assert binding.token_symbol_ref and contract.direction is SwizzleDteDirection.LOCAL_COPY
                runtime(RuntimeSymbolKind.DTE_TOKEN, binding.token_symbol_ref, owner)
                source, destination = views
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_ISSUE, (
                    RecordOperand.literal("direction", 0), RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref),
                    RecordOperand.literal("payload_bits", contract.payload_bits), RecordOperand.literal("size_bytes", contract.logical_bytes), RecordOperand.literal("hbm_address", 0),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, abs_symbol[(source.value_ref, source.slot)]),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_symbol[(destination.value_ref, destination.slot)]),
                )))
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref),
                )))
            elif task.kind is SwizzleActionKind.REDUCE:
                contract = reduces[owner]
                source, accumulator, _output = views
                span_ref = _id("reduce_span", {"task": owner, "inputs": ((source.value_ref, source.slot), (accumulator.value_ref, accumulator.slot))})
                source_symbol = program(ProgramSymbolKind.ABSOLUTE_ADDRESS, span_ref)
                records.append(RelocatableRecord(owner, RecordOpcode.LOCAL_REDUCE, (
                    RecordOperand.literal("input_dtype", _dtype_code(contract.input_dtype)),
                    RecordOperand.literal("accumulator_dtype", _dtype_code(contract.accumulation_dtype)),
                    RecordOperand.literal("output_dtype", _dtype_code(contract.output_dtype)),
                    RecordOperand.literal("reduce_op", 1), RecordOperand.literal("rounding", 0), RecordOperand.literal("order", 0),
                    RecordOperand.literal("input_count", contract.input_count), RecordOperand.literal("element_count", contract.element_count),
                    RecordOperand.literal("input_stride_bytes", contract.input_stride_bytes),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source_symbol),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_symbol[(accumulator.value_ref, accumulator.slot)]),
                )))
            elif task.kind is SwizzleActionKind.BARRIER:
                for event in sorted(events_by_owner[owner], key=lambda item: (item.opcode.value, item.event_symbol_ref)):
                    runtime(RuntimeSymbolKind.RUNTIME_CORE, event.source_core_symbol_ref, str(event.source_core))
                    runtime(RuntimeSymbolKind.RUNTIME_CORE, event.destination_core_symbol_ref, str(event.destination_core))
                    runtime(RuntimeSymbolKind.EVENT_TAG, event.event_symbol_ref, event.barrier_ref)
                    operands = (
                        RecordOperand.runtime("source_core", RuntimeOperandField.SOURCE_CORE, event.source_core_symbol_ref),
                        RecordOperand.runtime("destination_core", RuntimeOperandField.DESTINATION_CORE, event.destination_core_symbol_ref),
                        RecordOperand.runtime("tag", RuntimeOperandField.EVENT_TAG, event.event_symbol_ref),
                    )
                    if event.opcode is RecordOpcode.EVENT_WAIT:
                        operands = (*operands, RecordOperand.literal("count", 1))
                    records.append(RelocatableRecord(owner, event.opcode, operands))
            else:
                raise SchemaError("unsupported Swizzle task kind", path=f"projection.task[{owner}].kind")

            ending_roots = tuple(
                abi
                for abi in roots.values()
                if abi.logical_core == core
                and abi.lifetime_end_exclusive
                == task_bindings[owner].core_order + 1
            )
            for abi in reversed(sorted(ending_roots, key=lambda item: (item.region_ref, item.region_offset_bytes, item.id))):
                if abi.lifetime_end_exclusive == task_bindings[owner].core_order + 1:
                    records.append(RelocatableRecord(owner, RecordOpcode.SRAM_FREE, (
                        RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label_symbol[abi.storage_id]),
                    )))

    streams = []
    for core, records in sorted(records_by_core.items(), key=lambda item: (item[0].die_id, item[0].local_core_id)):
        runtime_relocations, address_relocations = _relocations(records)
        streams.append(CoreFragmentStream(core, tuple(records), runtime_relocations, address_relocations))
    used_program_symbols = {
        relocation.symbol_ref
        for stream in streams for relocation in stream.address_relocations
    }
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=projection.id,
        kind=FragmentKind.SWIZZLE,
        claimed_action_ids=tuple(sorted(tasks)),
        core_streams=tuple(streams),
        runtime_symbols=tuple(runtime_symbols[key] for key in sorted(runtime_symbols)),
        program_symbols=tuple(
            program_symbols[key]
            for key in sorted(used_program_symbols)
        ),
        buffer_abi=buffers,
        state_abi=(),
    )
    fragment.validate()
    return fragment


def link_swizzle_standard_manifest(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    lowered: SwizzleLoweredProgram,
    core_abi: SwizzleCoreAddressABI,
    operand_abi: SwizzleOperandABI,
    fragment: CommandFragment,
) -> LinkedProgramManifest:
    """Link the dedicated fragment through the unchanged standard ABI closure."""

    expected_fragment = lower_swizzle_standard_fragment(
        ir1, plan, projection, lowered, core_abi, operand_abi
    )
    if fragment != expected_fragment:
        raise SchemaError("fragment is not the exact Swizzle standard lowering", path="fragment")
    task_by_id = {task.id: task for dag in projection.rank_dags for task in dag.tasks}
    task_core = {item.task_ref: item.logical_core for item in core_abi.task_bindings}
    core_binding_by_ref = {item.logical_core: item for item in core_abi.task_bindings}
    buffer_by_id = {item.id: item for item in fragment.buffer_abi}
    buffer_by_binding = {item.binding_id: item for item in fragment.buffer_abi}
    buffer_by_storage = defaultdict(list)
    for abi in fragment.buffer_abi:
        buffer_by_storage[abi.storage_id].append(abi)

    symbol_uses: dict[str, set[object]] = defaultdict(set)
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            symbol_uses[relocation.symbol_ref].add(stream.logical_core)

    program_definitions = []
    address_bindings = []
    symbol_by_id = {symbol.id: symbol for symbol in fragment.program_symbols}
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            record = stream.records[relocation.record_index]
            symbol = symbol_by_id[relocation.symbol_ref]
            if symbol.kind is ProgramSymbolKind.SRAM_REGION:
                label_operand = next(
                    operand
                    for operand in record.operands
                    if operand.operand_id is SemanticOperandId.LABEL_SYMBOL
                )
                label = symbol_by_id[label_operand.symbol_ref]
                candidates = [
                    abi
                    for abi in buffer_by_storage[label.source_ref]
                    if abi.alias_of is None
                ]
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                candidates = [abi for abi in buffer_by_storage[symbol.source_ref] if abi.alias_of is None]
            elif symbol.source_ref in buffer_by_binding:
                candidates = [buffer_by_binding[symbol.source_ref]]
            elif record.opcode is RecordOpcode.LOCAL_REDUCE and relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS:
                views = sorted(
                    (view for view in operand_abi.operands if view.task_ref == record.source_global_action_id),
                    key=lambda item: item.ordinal,
                )
                candidates = [
                    next(
                        abi for abi in fragment.buffer_abi
                        if abi.binding_id == _id("buffer_binding", {"core_abi": core_abi.id, "key": (view.value_ref, view.slot)})
                    )
                    for view in views[:2]
                ]
            else:
                raise SchemaError("program symbol has no exact typed storage witness", path="fragment.program_symbols")
            candidates = sorted(candidates, key=lambda item: (item.region_offset_bytes, item.id))
            key = (fragment.id, stream.logical_core, relocation.record_index, relocation.operand_id)
            if not any(
                (item.fragment_id, item.logical_core, item.fragment_record_index, item.operand_id) == key
                for item in address_bindings
            ):
                address_bindings.append(AddressOperandBinding(
                    fragment.id,
                    stream.logical_core,
                    relocation.record_index,
                    relocation.operand_id,
                    tuple(item.id for item in candidates),
                    tuple(item.tensor_slice for item in candidates),
                ))

    for symbol in fragment.program_symbols:
        cores = tuple(sorted(symbol_uses[symbol.id], key=lambda core: (core.die_id, core.local_core_id)))
        if symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            name, value, size = f"swz_label_{symbol.id[-16:]}", 0, 0
        elif symbol.kind is ProgramSymbolKind.SRAM_REGION:
            abi = next(item for item in fragment.buffer_abi if item.region_ref == symbol.source_ref and item.logical_core in cores)
            binding = next(item for item in core_abi.value_bindings if item.logical_core == abi.logical_core and item.region_ref == abi.region_ref)
            region = _region(ir1, binding)
            name, value, size = region.name, region.base_bytes, region.size_bytes
        elif symbol.source_ref in buffer_by_binding:
            abi = buffer_by_binding[symbol.source_ref]
            binding = next(item for item in core_abi.value_bindings if (item.value_ref, item.slot) == (
                next(key for key in {(view.value_ref, view.slot) for view in operand_abi.operands} if _id("buffer_binding", {"core_abi": core_abi.id, "key": key}) == abi.binding_id)
            ))
            name, value, size = f"swz_abs_{symbol.id[-16:]}", binding.address, binding.size_bytes
        else:
            # The only non-BufferABI absolute is the ordered LOCAL_REDUCE input span.
            matching = [
                binding for binding in address_bindings
                if symbol_by_id[next(
                    relocation.symbol_ref
                    for stream in fragment.core_streams
                    if stream.logical_core == binding.logical_core
                    for relocation in stream.address_relocations
                    if relocation.record_index == binding.fragment_record_index and relocation.operand_id == binding.operand_id
                )] == symbol
            ]
            if len(matching) != 1:
                raise SchemaError("reduce span symbol must have one exact operand witness", path="fragment.program_symbols")
            abis = [buffer_by_id[ref] for ref in matching[0].buffer_abi_ids]
            starts = []
            for abi in abis:
                binding = next(item for item in core_abi.value_bindings if _id("buffer_binding", {"core_abi": core_abi.id, "key": (item.value_ref, item.slot)}) == abi.binding_id)
                starts.append((binding.address, binding.size_bytes))
            value = starts[0][0]
            size = sum(item[1] for item in starts)
            name = f"swz_reduce_{symbol.id[-16:]}"
        program_definitions.append(ProgramSymbolDefinition(symbol, name, value, size, cores))

    runtime_defs = []
    runtime_by_id = {item.id: item for item in fragment.runtime_symbols}
    flow_by_task = {
        ref: flow
        for flow in projection.flows
        for ref in (flow.send_task_ref, flow.recv_task_ref)
    }
    wait_by_recv = {
        dep: task.id
        for task in task_by_id.values()
        if task.kind is SwizzleActionKind.WAIT
        for dep in task.deps
        if task_by_id[dep].kind is SwizzleActionKind.RECV
    }
    local_copy_tokens = {
        binding.token_symbol_ref: binding.task_ref
        for binding in core_abi.runtime_bindings
        if task_by_id[binding.task_ref].kind is SwizzleActionKind.LOCAL_COPY
    }
    for symbol in fragment.runtime_symbols:
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            flow = next(flow for flow in projection.flows if any(
                binding.fsm_symbol_ref == symbol.id and binding.flow_ref == flow.id
                for binding in core_abi.runtime_bindings
            ))
            cores = tuple(sorted((task_core[flow.send_task_ref], task_core[flow.recv_task_ref]), key=lambda core: (core.die_id, core.local_core_id)))
            source, destination = flow.send_task_ref, flow.recv_task_ref
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            recv = next((binding.task_ref for binding in core_abi.runtime_bindings if binding.token_symbol_ref == symbol.id and task_by_id[binding.task_ref].kind is SwizzleActionKind.RECV), None)
            owner = recv if recv is not None else local_copy_tokens[symbol.id]
            cores = (task_core[owner],)
            source, destination = owner, wait_by_recv.get(owner, owner)
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            event = next((item for item in core_abi.barrier_events if symbol.id in (item.source_core_symbol_ref, item.destination_core_symbol_ref)), None)
            if event is not None:
                represented = event.source_core if symbol.id == event.source_core_symbol_ref else event.destination_core
            else:
                binding = next(item for item in core_abi.runtime_bindings if item.peer_symbol_ref == symbol.id)
                represented = binding.peer_core
            cores, source, destination = (represented,), None, None
        elif symbol.kind is RuntimeSymbolKind.EVENT_TAG:
            event = next(item for item in core_abi.barrier_events if item.event_symbol_ref == symbol.id)
            cores = tuple(sorted((event.source_core, event.destination_core), key=lambda core: (core.die_id, core.local_core_id)))
            source, destination = event.source_task_ref, event.destination_task_ref
        else:
            raise SchemaError("unexpected runtime symbol kind", path="fragment.runtime_symbols")
        runtime_defs.append(RuntimeSymbolDefinition(symbol, cores, source, destination))

    active_cores = tuple(stream.logical_core for stream in fragment.core_streams)
    starts = []
    for core in active_cores:
        first = min((item for item in core_abi.task_bindings if item.logical_core == core), key=lambda item: item.core_order)
        symbol = RuntimeSymbol(_id("start_tag", {"projection": projection.id, "core": core, "first": first.task_ref}), RuntimeSymbolKind.START_TAG, first.task_ref)
        runtime_defs.append(RuntimeSymbolDefinition(symbol, (core,), None, None))
        starts.append(LogicalStartEvent(core, symbol.id, 1))
    terminal_task_refs = {ref for ownership in projection.output_ownership for ref in ownership.terminal_task_refs}
    terminal_cores = tuple(sorted({task_core[ref] for ref in terminal_task_refs}, key=lambda core: (core.die_id, core.local_core_id)))
    core_bindings = []
    linked_streams = []
    for stream in fragment.core_streams:
        task_binding = core_binding_by_ref[stream.logical_core]
        die = next(item for item in ir1.fabric.dies if item.id == stream.logical_core.die_id)
        core = next(item for item in die.cores if item.local_core_id == stream.logical_core.local_core_id)
        core_bindings.append(CoreRuntimeBinding(stream.logical_core, core.id, task_binding.runtime_core_id, core.sram_profile_ref))
        linked_streams.append(LinkedCoreStream(stream.logical_core, task_binding.runtime_core_id, tuple(
            LinkedRecordRef(fragment.id, index, record.source_global_action_id)
            for index, record in enumerate(stream.records)
        )))
    entry_counts = defaultdict(int)
    exit_counts = defaultdict(int)
    for stream in fragment.core_streams:
        for record in stream.records:
            operands = {item.name: item for item in record.operands}
            if record.opcode is RecordOpcode.EVENT_WAIT:
                entry_counts[operands["tag"].symbol_ref] += operands["count"].literal_value
            elif record.opcode is RecordOpcode.EVENT_SET:
                exit_counts[operands["tag"].symbol_ref] += 1
    interface = FragmentInterface(
        fragment.id, (), tuple(sorted(runtime_by_id)), (), tuple(sorted(symbol_by_id)),
        tuple(EventCredit(key, entry_counts[key]) for key in sorted(entry_counts)),
        tuple(EventCredit(key, exit_counts[key]) for key in sorted(exit_counts)),
    )
    artifacts = (
        (ManifestInputKind.IR1, ir1),
        (ManifestInputKind.SWIZZLE_DECISION, plan.decision),
        (ManifestInputKind.SWIZZLE_CANDIDATE, plan.candidate),
        (ManifestInputKind.SWIZZLE_FUSION_PLAN, plan),
        (ManifestInputKind.SWIZZLE_PROJECTION, projection),
        (ManifestInputKind.SWIZZLE_LOWERED_PROGRAM, lowered),
        (ManifestInputKind.SWIZZLE_CORE_ADDRESS_ABI, core_abi),
        (ManifestInputKind.SWIZZLE_OPERAND_ABI, operand_abi),
        (ManifestInputKind.COMMAND_FRAGMENT, fragment),
    )
    digests = tuple(sorted(
        (ManifestInputDigest(kind, artifact.id, artifact.schema_version, canonical_digest(artifact)) for kind, artifact in artifacts),
        key=lambda item: (item.kind.value, item.artifact_id),
    ))
    manifest = LinkedProgramManifest.create(
        producer_pass="swizzle_standard_linker",
        capabilities=0,
        source_ir1_id=ir1.id,
        source_projection_id=projection.id,
        source_schedule_set_id=core_abi.id,
        source_global_dag_id=projection.id,
        input_digests=digests,
        fragments=(fragment,),
        fragment_interfaces=(interface,),
        core_bindings=tuple(core_bindings),
        core_streams=tuple(linked_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_defs, key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program_definitions, key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id,
            item.fragment_record_index, int(item.operand_id),
        ))),
        state_operand_bindings=(),
        core_groups=(),
        envelope=ProgramControlEnvelope(
            active_cores, tuple(starts), terminal_cores, active_cores, terminal_cores,
            EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    manifest.validate()
    return manifest


def link_swizzle_standard_program(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    lowered: SwizzleLoweredProgram,
    core_abi: SwizzleCoreAddressABI,
    operand_abi: SwizzleOperandABI,
) -> SwizzleStandardLinkedProgram:
    fragment = lower_swizzle_standard_fragment(
        ir1, plan, projection, lowered, core_abi, operand_abi
    )
    manifest = link_swizzle_standard_manifest(
        ir1, plan, projection, lowered, core_abi, operand_abi, fragment
    )
    result = SwizzleStandardLinkedProgram.create(
        ir1=ir1,
        plan=plan,
        projection=projection,
        lowered=lowered,
        core_abi=core_abi,
        operand_abi=operand_abi,
        fragment=fragment,
        manifest=manifest,
    )
    result.validate_against()
    return result


__all__ = [
    "link_swizzle_standard_manifest",
    "link_swizzle_standard_program",
    "lower_swizzle_standard_fragment",
]
