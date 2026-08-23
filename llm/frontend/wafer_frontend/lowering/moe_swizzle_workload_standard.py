"""Canonical single-leaf lowering for a complete MoE Swizzle workload."""

from __future__ import annotations

from collections import defaultdict
from math import prod

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    BufferABI, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbol, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, RuntimeOperandField, RuntimeSymbol,
    RuntimeSymbolKind, SemanticOperandId, StateABI,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe import MoeHardwareFacts
from ..schema.swizzle_moe_abi import MoeSwizzleCoreAddressABI
from ..schema.swizzle_moe_execution import MoeScaleExecution
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_operand_abi import MoeSwizzleOperandABI
from ..schema.swizzle_moe_placement import build_moe_swizzle_workload_placement
from ..schema.swizzle_moe_state import MoeSwizzleWorkloadStateABI
from ..schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection
from ..schema.swizzle_moe_workload_abi import MoeSwizzleWorkloadABI
from ..schema.swizzle_moe_workload_bridge import (
    MoeSwizzleWorkloadPhysicalUse, MoeSwizzleWorkloadValueBridge,
)
from ..passes.build_moe_swizzle_workload_value_bridge import (
    validate_moe_swizzle_workload_value_bridge_against,
)
from .moe_swizzle_standard import _id, _relocations


_SCHEMA = "wafer_frontend.moe_swizzle_workload_standard_lowering/v1alpha1"
_PRODUCER = "moe_swizzle_standard_lowering"


def _wid(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"moe_swizzle_workload_standard_{kind}", semantic,
        schema_version=_SCHEMA,
    )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _root_addresses(
    roots: tuple[object, ...], hardware_facts: MoeHardwareFacts,
) -> dict[tuple[int, str, int], tuple[int, str]]:
    facts = {
        item.runtime_core_id: item
        for die in hardware_facts.ordered_cores_by_die for item in die
    }
    assigned: dict[int, list[tuple[int, int, object]]] = defaultdict(list)
    result = {}
    for root in sorted(
        roots,
        key=lambda item: (
            item.runtime_core_id, item.lifetime_start,
            item.lifetime_end_exclusive, item.family, item.slot,
        ),
    ):
        fact = facts[root.runtime_core_id]
        candidates = {0}
        for offset, extent, prior in assigned[root.runtime_core_id]:
            if (
                root.lifetime_start < prior.lifetime_end_exclusive
                and prior.lifetime_start < root.lifetime_end_exclusive
            ):
                candidates.add(_align(offset + extent, fact.allocation_alignment_bytes))
        chosen = None
        for offset in sorted(candidates):
            if offset + root.extent_bytes > fact.region_size_bytes:
                continue
            conflict = False
            for prior_offset, prior_extent, prior in assigned[root.runtime_core_id]:
                live = (
                    root.lifetime_start < prior.lifetime_end_exclusive
                    and prior.lifetime_start < root.lifetime_end_exclusive
                )
                overlap = (
                    offset < prior_offset + prior_extent
                    and prior_offset < offset + root.extent_bytes
                )
                if live and overlap:
                    conflict = True
                    break
            if not conflict:
                chosen = offset
                break
        if chosen is None:
            raise SchemaError(
                "whole workload roots exceed exact per-core comm SRAM",
                path="moe_workload_standard.workload_abi.roots",
            )
        key = (root.runtime_core_id, root.family, root.slot)
        result[key] = (fact.region_base_bytes + chosen, fact.region_name)
        assigned[root.runtime_core_id].append((chosen, root.extent_bytes, root))
    return result


def lower_moe_swizzle_workload_fragment(
    ir1: IR1,
    execution: MoeScaleExecution,
    workload: MoeSwizzleWorkloadProjection,
    projection: MoeSwizzleIr2Projection,
    core_abi: MoeSwizzleCoreAddressABI,
    operand_abi: MoeSwizzleOperandABI,
    state_abi: MoeSwizzleWorkloadStateABI,
    value_bridge: MoeSwizzleWorkloadValueBridge,
    workload_abi: MoeSwizzleWorkloadABI,
    hardware_facts: MoeHardwareFacts,
) -> CommandFragment:
    """Emit one MOE_SWIZZLE leaf claiming every linked overlay action."""

    ir1.validate("moe_workload_standard.ir1")
    workload.validate("moe_workload_standard.workload")
    projection.validate("moe_workload_standard.projection")
    core_abi.validate_against(
        ir1, projection, "moe_workload_standard.core_abi",
        workload_projection=workload,
    )
    operand_abi.validate_against(projection, "moe_workload_standard.operand_abi")
    state_abi.validate("moe_workload_standard.state_abi")
    validate_moe_swizzle_workload_value_bridge_against(
        value_bridge, execution, workload, projection,
    )
    workload_abi.validate("moe_workload_standard.workload_abi")
    hardware_facts.validate("moe_workload_standard.hardware_facts")
    if (
        workload.replacement_projection_id != projection.id
        or workload.source_execution_id != execution.id
        or workload.state_abi_id != state_abi.id
        or workload_abi.source_workload_projection_id != workload.id
        or workload_abi.source_replacement_projection_id != projection.id
        or workload_abi.source_state_abi_id != state_abi.id
        or workload_abi.source_value_bridge_id != value_bridge.id
    ):
        raise SchemaError("whole lowering lineage is not exact", path="moe_workload_standard")

    tasks = {item.id: item for item in projection.tasks}
    values = {item.id: item for item in projection.values}
    actions = {item.id: item for item in workload.actions}
    placement = {
        item.action_ref: item
        for item in build_moe_swizzle_workload_placement(
            ir1, workload, projection, hardware_facts,
        )
    }
    task_core = {item.task_ref: item.logical_core for item in core_abi.task_bindings}
    runtime_bindings = {item.task_ref: item for item in core_abi.runtime_bindings}
    views = defaultdict(list)
    for item in operand_abi.operands:
        views[item.task_ref].append(item)
    matmuls = {item.task_ref: item for item in operand_abi.matmuls}
    dtes = {item.task_ref: item for item in operand_abi.dtes}
    swiglus = {item.task_ref: item for item in operand_abi.swiglus}

    addresses = _root_addresses(workload_abi.roots, hardware_facts)
    program_symbols: dict[str, ProgramSymbol] = {}
    runtime_symbols: dict[str, RuntimeSymbol] = {}
    def program(kind: ProgramSymbolKind, source_ref: str) -> str:
        ref = _wid("program_symbol", {"kind": kind, "source_ref": source_ref})
        program_symbols.setdefault(ref, ProgramSymbol(ref, kind, source_ref))
        return ref
    def runtime(kind: RuntimeSymbolKind, ref: str, source_ref: str) -> str:
        runtime_symbols.setdefault(ref, RuntimeSymbol(ref, kind, source_ref))
        return ref

    roots_by_key = {
        (item.runtime_core_id, item.family, item.slot): item
        for item in workload_abi.roots
    }
    buffers = []
    root_binding = {}
    root_buffer = {}
    label = {}
    region_symbol = {}
    absolute = {}
    for key, root in sorted(roots_by_key.items()):
        address, region_name = addresses[key]
        fact = next(
            item for die in hardware_facts.ordered_cores_by_die for item in die
            if item.runtime_core_id == root.runtime_core_id
        )
        die = ir1.fabric.dies[root.logical_core.die_id]
        core = next(item for item in die.cores if item.local_core_id == root.logical_core.local_core_id)
        profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
        region = next(item for item in profile.regions if item.name == region_name)
        storage = _wid("storage", {"abi": workload_abi.id, "key": key})
        binding = _wid("binding", {"storage": storage})
        semantic = {
            "schedule_id": workload_abi.id, "binding_id": binding,
            "value_id": storage, "logical_core": root.logical_core,
            "tensor_slice": TensorSlice(storage, (0,), (root.extent_bytes // 2,)),
            "region_ref": region.id,
            "region_offset_bytes": address - region.base_bytes,
            "size_bytes": root.extent_bytes, "alignment_bytes": 64,
            "banks": (), "storage_id": storage, "alias_of": None,
            "lifetime_start": root.lifetime_start,
            "lifetime_end_exclusive": root.lifetime_end_exclusive,
            "dtype": DType.FP16,
            "layout": f"moe_swizzle_{root.family}_root/v1",
            "ownership": root.ownership,
        }
        abi = BufferABI(id=_wid("buffer", semantic), **semantic)
        buffers.append(abi)
        root_binding[key] = binding
        root_buffer[key] = abi
        label[key] = program(ProgramSymbolKind.SRAM_LABEL, storage)
        region_symbol.setdefault(region.id, program(ProgramSymbolKind.SRAM_REGION, region.id))

    runtime_to_core = {
        item.runtime_core_id: item.logical_core
        for die in hardware_facts.ordered_cores_by_die for item in die
    }
    bridge_by_physical = defaultdict(list)
    bridge_by_semantic = {}
    for binding in value_bridge.bindings:
        bridge_by_semantic[binding.semantic_value_ref] = binding
        for physical in binding.physical_slices:
            bridge_by_physical[(physical.physical_task_ref, physical.physical_value_ref)].append(
                (binding, physical)
            )

    terminal_sizes = {item.value_ref: item.bytes for item in workload.terminals}
    terminals_by_ref = {item.value_ref: item for item in workload.terminals}

    alias_by_key = {}
    def root_for(action_ref: str, value_ref: str) -> tuple[int, str, int]:
        runtime_core_id = placement[action_ref].runtime_core_id
        direct = tuple(
            key for key, root in roots_by_key.items()
            if key[0] == runtime_core_id and value_ref in root.value_refs
        )
        if len(direct) == 1:
            return direct[0]
        semantic_refs = {
            physical.physical_value_ref
            for binding in value_bridge.bindings
            if binding.semantic_value_ref == value_ref
            for physical in binding.physical_slices
            if placement[physical.physical_task_ref].runtime_core_id == runtime_core_id
        }
        semantic_refs.update(
            physical.physical_value_ref
            for binding, physical in bridge_by_physical.get((action_ref, value_ref), ())
        )
        semantic_refs.update(
            binding.semantic_value_ref
            for binding, physical in bridge_by_physical.get((action_ref, value_ref), ())
        )
        bridged = tuple(
            key for key, root in roots_by_key.items()
            if key[0] == runtime_core_id and semantic_refs.intersection(root.value_refs)
        )
        candidates = direct or bridged
        if len(candidates) != 1:
            raise SchemaError(
                f"physical operand does not select one exact root: action={action_ref!r}, "
                f"value={value_ref!r}, runtime_core={runtime_core_id}, "
                f"direct={direct!r}, bridged={bridged!r}, "
                f"available={tuple(key for key in roots_by_key if key[0] == runtime_core_id)!r}",
                path="moe_workload_standard.value_bridge",
            )
        return candidates[0]

    def add_alias(
        action_ref: str, value_ref: str, shape: tuple[int, ...], dtype: DType,
        size_bytes: int, byte_offset: int,
    ) -> str:
        key = root_for(action_ref, value_ref)
        root = roots_by_key[key]
        direct_placements = tuple(
            item for item in root.value_placements if item.value_ref == value_ref
        )
        if len(direct_placements) == 1:
            byte_offset = direct_placements[0].root_offset_bytes
        elif value_ref in bridge_by_semantic:
            offsets = set()
            for physical in bridge_by_semantic[value_ref].physical_slices:
                if placement[physical.physical_task_ref].runtime_core_id != key[0]:
                    continue
                physical_placement = next((
                    item for item in root.value_placements
                    if item.value_ref == physical.physical_value_ref
                ), None)
                if physical_placement is not None:
                    offsets.add(
                        physical_placement.root_offset_bytes
                        + physical.physical_byte_offset
                    )
            if len(offsets) != 1:
                raise SchemaError(
                    "semantic operand does not select one typed root-local addend: "
                    f"action={action_ref!r}, value={value_ref!r}, key={key!r}, "
                    f"offsets={tuple(sorted(offsets))!r}",
                    path="moe_workload_standard.value_bridge",
                )
            byte_offset = next(iter(offsets))
        elif bridge_by_physical.get((action_ref, value_ref)):
            offsets = set()
            for binding, physical in bridge_by_physical[(action_ref, value_ref)]:
                semantic_placement = next((
                    item for item in root.value_placements
                    if item.value_ref == binding.semantic_value_ref
                ), None)
                terminal = terminals_by_ref.get(binding.semantic_value_ref)
                if semantic_placement is None or terminal is None:
                    continue
                element_offset = sum(
                    origin * prod(terminal.shape[index + 1:])
                    for index, origin in enumerate(physical.semantic_origin)
                )
                dtype_bytes = 2 if terminal.dtype is DType.FP16 else 4
                offsets.add(
                    semantic_placement.root_offset_bytes
                    + element_offset * dtype_bytes
                )
            if len(offsets) != 1:
                raise SchemaError(
                    "physical terminal fragment does not select one typed semantic addend: "
                    f"action={action_ref!r}, value={value_ref!r}, key={key!r}, offsets={tuple(sorted(offsets))!r}",
                    path="moe_workload_standard.value_bridge",
                )
            byte_offset = next(iter(offsets))
        else:
            raise SchemaError(
                "operand lacks an authoritative root-local value placement",
                path="moe_workload_standard.workload_abi",
            )
        if byte_offset + size_bytes > root.extent_bytes:
            raise SchemaError(
                "operand subview escapes its exact physical root: "
                f"action={action_ref!r}, value={value_ref!r}, key={key!r}, "
                f"offset={byte_offset}, size={size_bytes}, extent={root.extent_bytes}",
                path="moe_workload_standard.buffer_abi",
            )
        alias_key = (key, value_ref, byte_offset, size_bytes, shape, dtype)
        prior = alias_by_key.get(alias_key)
        if prior is not None:
            absolute[(action_ref, value_ref)] = prior
            return prior
        parent = root_buffer[key]
        binding_ref = _wid("value_binding", {
            "workload_abi": workload_abi.id, "root": key,
            "value": value_ref, "offset": byte_offset, "size": size_bytes,
            "shape": shape, "dtype": dtype,
        })
        semantic = {
            "schedule_id": workload_abi.id, "binding_id": binding_ref,
            "value_id": value_ref, "logical_core": root.logical_core,
            "tensor_slice": TensorSlice(value_ref, (0,) * len(shape), shape),
            "region_ref": parent.region_ref,
            "region_offset_bytes": parent.region_offset_bytes + byte_offset,
            "size_bytes": size_bytes, "alignment_bytes": parent.alignment_bytes,
            "banks": (), "storage_id": parent.storage_id,
            "alias_of": parent.binding_id,
            "lifetime_start": root.lifetime_start,
            "lifetime_end_exclusive": root.lifetime_end_exclusive,
            "dtype": dtype, "layout": f"moe_swizzle_{root.family}_subview/v1",
            "ownership": BufferOwnership.ALIASED,
        }
        abi = BufferABI(id=_wid("buffer", semantic), **semantic)
        buffers.append(abi)
        symbol = program(ProgramSymbolKind.ABSOLUTE_ADDRESS, binding_ref)
        alias_by_key[alias_key] = symbol
        absolute[(action_ref, value_ref)] = symbol
        return symbol

    for task in projection.tasks:
        for ref in task.read_value_refs + task.write_value_refs:
            value = values[ref]
            add_alias(task.id, ref, value.shape, value.dtype, value.size_bytes, value.byte_offset)
    for action in workload.actions:
        if not action.preserved:
            continue
        for ref in action.read_value_refs + action.write_value_refs:
            if ref in values:
                value = values[ref]
                add_alias(action.id, ref, value.shape, value.dtype, value.size_bytes, value.byte_offset)
                continue
            size = terminal_sizes.get(ref, action.bytes)
            if not size:
                binding = bridge_by_semantic.get(ref)
                if binding is not None:
                    size = sum(item.size_bytes for item in binding.physical_slices if item.use is MoeSwizzleWorkloadPhysicalUse.IR2_WRITE)
            if not size or size % 2:
                raise SchemaError("preserved operand lacks an exact FP16 extent", path="moe_workload_standard.workload")
            add_alias(action.id, ref, (size // 2,), action.dtype, size, 0)
    # ProgramIo probes retain semantic token granularity even when terminal
    # storage is packet-packed.  Materialize one typed alias per semantic
    # terminal; this adds no physical root and derives its root-local offset
    # exclusively through the ValueBridge/workload-ABI placement above.
    for terminal in workload.terminals:
        semantic_binding = bridge_by_semantic.get(terminal.value_ref)
        if (
            terminal.kind.value == "combined"
            and (
                semantic_binding is None
                or len(semantic_binding.physical_slices) != 1
            )
        ):
            # Split-N/T2 terminal storage is packet-contiguous rather than
            # token-contiguous.  The standard ProgramIo contract has one
            # contiguous range per probe, so this mode remains fail-closed
            # until a typed gather or multi-range probe carrier exists.
            continue
        add_alias(
            terminal.producer_action_ref,
            terminal.value_ref,
            terminal.shape,
            terminal.dtype,
            terminal.bytes,
            0,
        )

    state_tensors = {item.state_ref: item for item in state_abi.tensors}
    fragment_states = tuple(sorted((StateABI.create(
        state_ref=item.state_ref, hbm_binding_ref=item.hbm_binding_ref,
        kind=StateKind.PARAMETER,
        lifetime=PersistentStateLifetime.PERSISTENT,
        access=PersistentStateAccess.READ_ONLY,
        shape=item.shape, dtype=item.dtype, layout=item.layout,
        die_id=item.home_die_id, address=item.hbm_address,
        size_bytes=item.size_bytes, alignment_bytes=item.alignment_bytes,
    ) for item in state_abi.tensors), key=lambda item: item.id))
    state_actions = {item.execution_action_ref: item for item in state_abi.action_bindings}

    first_for_root = {}
    last_for_root = {}
    action_order = {item.id: index for index, item in enumerate(workload.actions)}
    for key, root in roots_by_key.items():
        refs = tuple(
            ref for ref in root.action_refs
            if ref in action_order
            and placement[ref].runtime_core_id == root.runtime_core_id
        )
        if not refs:
            raise SchemaError("whole root lacks same-core workload lifecycle owner", path="moe_workload_standard.workload_abi")
        first_for_root[key] = min(refs, key=action_order.__getitem__)
        last_for_root[key] = max(refs, key=action_order.__getitem__)

    records_by_core = defaultdict(list)
    for action in workload.actions:
        owner = action.id
        core = placement[owner].logical_core
        if action.replacement_task_ref is not None and task_core[action.replacement_task_ref] != core:
            raise SchemaError("whole/replacement task owner drifted", path="moe_workload_standard.placement")
        records = records_by_core[core]
        action_roots = tuple(
            (key, root) for key, root in roots_by_key.items() if owner in root.action_refs
        )
        for key, root in sorted(action_roots):
            if root.allocate and first_for_root[key] == owner:
                address, _ = addresses[key]
                fact = next(item for die in hardware_facts.ordered_cores_by_die for item in die if item.runtime_core_id == root.runtime_core_id)
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_ALLOC_AT, (
                    RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbol[next(item.region_ref for item in buffers if item.binding_id == root_binding[key])]),
                    RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label[key]),
                    RecordOperand.literal("region_offset_bytes", address - fact.region_base_bytes),
                    RecordOperand.literal("size_bytes", root.extent_bytes),
                    RecordOperand.literal("alignment_bytes", 64),
                    RecordOperand.literal(
                        "lifetime",
                        2 if root.family.startswith("terminal_") else 0,
                    ),
                    RecordOperand.literal("spillable", False),
                )))

        if action.preserved:
            if action.kind == "preserved.dma_in":
                binding = state_actions[action.source_action_refs[0]]
                tensor = state_tensors[binding.state_ref]
                hbm = program(ProgramSymbolKind.ABSOLUTE_ADDRESS, tensor.hbm_binding_ref)
                records.append(RelocatableRecord(owner, RecordOpcode.LSU_LOAD, (
                    RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm),
                    RecordOperand.literal("size_bytes", binding.bytes),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute[(owner, binding.staging_value_ref)]),
                )))
            elif action.kind == "preserved.tape_copy":
                if len(action.read_value_refs) != 1 or len(action.write_value_refs) != 1:
                    raise SchemaError("tape copy arity drifted", path="moe_workload_standard.workload")
                token = _wid("tape_token", {"workload": workload.id, "action": owner})
                runtime(RuntimeSymbolKind.DTE_TOKEN, token, owner)
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_ISSUE, (
                    RecordOperand.literal("direction", 0),
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
                    RecordOperand.literal("payload_bits", action.bytes * 8),
                    RecordOperand.literal("size_bytes", action.bytes),
                    RecordOperand.literal("hbm_address", 0),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute[(owner, action.read_value_refs[0])]),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute[(owner, action.write_value_refs[0])]),
                )))
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
                )))
            else:
                raise SchemaError("unsupported preserved whole action", path=f"moe_workload_standard.actions.{owner}")
        else:
            task = tasks[action.replacement_task_ref]
            task_views = sorted(views[task.id], key=lambda item: item.ordinal)
            if task.kind is SwizzleActionKind.COMP:
                contract = matmuls[task.id]
                lhs, rhs, output = task_views
                lhs_key = root_for(owner, lhs.value_ref)
                rhs_key = root_for(owner, rhs.value_ref)
                output_key = root_for(owner, output.value_ref)
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_BIND, (
                    RecordOperand.literal("input_count", 2),
                    RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0, label[lhs_key]),
                    RecordOperand.address("input_label_1", SemanticOperandId.SRAM_BIND_INPUT_1, label[rhs_key]),
                    *(RecordOperand.literal(f"input_label_{index}", 0) for index in range(2, 16)),
                    RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT, label[output_key]),
                )))
                records.append(RelocatableRecord(owner, RecordOpcode.MATMUL, (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute[(owner, lhs.value_ref)]),
                    RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute[(owner, rhs.value_ref)]),
                    RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute[(owner, output.value_ref)]),
                    RecordOperand.literal("parameters", (1, contract.m, contract.k, contract.n)),
                )))
            elif task.kind is SwizzleActionKind.SWIGLU:
                contract = swiglus[task.id]
                input_view, output_view = task_views
                input_key = root_for(owner, input_view.value_ref)
                output_key = root_for(owner, output_view.value_ref)
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_BIND, (
                    RecordOperand.literal("input_count", 1),
                    RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0, label[input_key]),
                    *(RecordOperand.literal(f"input_label_{index}", 0) for index in range(1, 16)),
                    RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT, label[output_key]),
                )))
                records.append(RelocatableRecord(owner, RecordOpcode.SWIGLU, (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute[(owner, input_view.value_ref)]),
                    RecordOperand.literal("data_address", 0),
                    RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute[(owner, output_view.value_ref)]),
                    RecordOperand.literal("parameters", (contract.element_count,)),
                )))
            elif task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                contract = dtes[task.id]
                rb = runtime_bindings[task.id]
                runtime(RuntimeSymbolKind.DTE_FSM, rb.fsm_symbol_ref, rb.flow_ref)
                peer_ref = _id("peer_core", {"task": task.id, "peer": rb.peer_core})
                runtime(RuntimeSymbolKind.RUNTIME_CORE, peer_ref, str(rb.peer_core))
                common = (
                    RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer_ref),
                    RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
                    RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
                    RecordOperand.literal("epoch", 0),
                )
                value = task_views[0].value_ref
                if task.kind is SwizzleActionKind.SEND:
                    opcode = RecordOpcode.DTE_SEND
                    operands = (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0),
                        RecordOperand.literal("completion", 1), RecordOperand.literal("datatype", 0),
                        RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, rb.fsm_symbol_ref),
                        RecordOperand.literal("token", 0),
                        RecordOperand.literal("length_bytes", contract.logical_bytes),
                        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute[(owner, value)]),
                        *common,
                    )
                else:
                    runtime(RuntimeSymbolKind.DTE_TOKEN, rb.token_symbol_ref, task.id)
                    opcode = RecordOpcode.DTE_RECV
                    operands = (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0),
                        RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, rb.fsm_symbol_ref),
                        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, rb.token_symbol_ref),
                        RecordOperand.literal("length_bytes", contract.logical_bytes),
                        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute[(owner, value)]),
                        *common,
                    )
                records.append(RelocatableRecord(owner, opcode, operands))
            elif task.kind is SwizzleActionKind.WAIT:
                rb = runtime_bindings[task.id]
                runtime(RuntimeSymbolKind.DTE_TOKEN, rb.token_symbol_ref, task.deps[0])
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, rb.token_symbol_ref),
                )))
            else:
                raise SchemaError("unsupported replacement whole action", path=f"moe_workload_standard.actions.{owner}")

        for key, root in sorted(action_roots, reverse=True):
            if root.free and last_for_root[key] == owner:
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_FREE, (
                    RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label[key]),
                )))

    streams = []
    for core, records in sorted(records_by_core.items(), key=lambda item: (item[0].die_id, item[0].local_core_id)):
        runtime_relocations, address_relocations = _relocations(records)
        streams.append(CoreFragmentStream(core, tuple(records), runtime_relocations, address_relocations))
    used_program = {item.symbol_ref for stream in streams for item in stream.address_relocations}
    result = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=workload.id,
        kind=FragmentKind.MOE_SWIZZLE,
        claimed_action_ids=tuple(sorted(actions)),
        core_streams=tuple(streams),
        runtime_symbols=tuple(runtime_symbols[key] for key in sorted(runtime_symbols)),
        program_symbols=tuple(program_symbols[key] for key in sorted(used_program)),
        buffer_abi=tuple(sorted(buffers, key=lambda item: item.id)),
        state_abi=fragment_states,
    )
    result.validate()
    return result


__all__ = ["lower_moe_swizzle_workload_fragment"]
