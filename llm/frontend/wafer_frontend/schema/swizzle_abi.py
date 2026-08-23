"""Explicit core, address and runtime ABI required by W9 finalization."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import PlanBarrierEventPhase, RecordOpcode
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir1 import IR1
from .swizzle import SwizzleActionKind
from .swizzle_ir2 import SwizzleIr2Projection, SwizzleIr2ValueOriginKind
from .swizzle_lowering import validate_swizzle_plan_projection
from .swizzle_plan import SwizzleFusionPlan


SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_core_address_abi/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class SwizzleRankCoreBinding:
    rank: int
    logical_core: LogicalCoreRef
    runtime_core_id: int

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError("must fit uint16", path=f"{path}.runtime_core_id")


@dataclass(frozen=True, slots=True)
class SwizzleTaskCoreBinding:
    task_ref: str
    rank: int
    logical_core: LogicalCoreRef
    core_order: int
    runtime_core_id: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_uint64(self.rank, f"{path}.rank")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.core_order, f"{path}.core_order")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")


@dataclass(frozen=True, slots=True)
class SwizzleValueAddressBinding:
    value_ref: str
    slot: int
    rank: int
    logical_core: LogicalCoreRef
    region_ref: str
    address: int
    size_bytes: int
    alignment_bytes: int
    storage_ref: str
    storage_offset_bytes: int
    lifetime_start: int
    lifetime_end_exclusive: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_uint64(self.slot, f"{path}.slot")
        validate_uint64(self.rank, f"{path}.rank")
        self.logical_core.validate(f"{path}.logical_core")
        validate_nonempty(self.region_ref, f"{path}.region_ref")
        validate_nonempty(self.storage_ref, f"{path}.storage_ref")
        for name in (
            "address", "size_bytes", "alignment_bytes",
            "storage_offset_bytes", "lifetime_start",
            "lifetime_end_exclusive",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.size_bytes == 0 or self.alignment_bytes == 0 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise SchemaError("requires positive size and power-of-two alignment", path=path)
        if self.address % self.alignment_bytes:
            raise SchemaError("address violates alignment", path=f"{path}.address")
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError(
                "value lifetime must be a non-empty half-open interval",
                path=f"{path}.lifetime_end_exclusive",
            )


@dataclass(frozen=True, slots=True)
class SwizzleBufferSliceBinding:
    value_ref: str
    slot: int
    offset_bytes: int
    size_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_uint64(self.slot, f"{path}.slot")
        validate_uint64(self.offset_bytes, f"{path}.offset_bytes")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.size_bytes")


@dataclass(frozen=True, slots=True)
class SwizzleBufferAddressBinding:
    rank: int
    buffer_ref: str
    logical_core: LogicalCoreRef
    region_name: str
    base_address: int
    span_bytes: int
    alignment_bytes: int
    slices: tuple[SwizzleBufferSliceBinding, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        self.logical_core.validate(f"{path}.logical_core")
        validate_nonempty(self.region_name, f"{path}.region_name")
        for name in ("base_address", "span_bytes", "alignment_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.span_bytes == 0 or self.alignment_bytes == 0 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise SchemaError("requires positive span and power-of-two alignment", path=path)
        if self.base_address % self.alignment_bytes:
            raise SchemaError("base violates alignment", path=f"{path}.base_address")
        refs = tuple((item.value_ref, item.slot) for item in self.slices)
        if refs != tuple(sorted(set(refs))):
            raise SchemaError("slices must be unique and canonical", path=f"{path}.slices")
        for index, item in enumerate(self.slices):
            item.validate(f"{path}.slices[{index}]")
            if item.offset_bytes > self.span_bytes or item.size_bytes > self.span_bytes - item.offset_bytes:
                raise SchemaError("slice exceeds region", path=f"{path}.slices[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleTaskRuntimeBinding:
    task_ref: str
    flow_ref: str | None
    token_symbol_ref: str | None
    fsm_symbol_ref: str | None
    peer_symbol_ref: str | None
    peer_core: LogicalCoreRef | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        if self.flow_ref is not None:
            validate_nonempty(self.flow_ref, f"{path}.flow_ref")
        if self.token_symbol_ref is not None:
            validate_nonempty(self.token_symbol_ref, f"{path}.token_symbol_ref")
        for name in ("fsm_symbol_ref", "peer_symbol_ref"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")
        if self.peer_core is not None:
            self.peer_core.validate(f"{path}.peer_core")
        if (self.fsm_symbol_ref is None) != (self.peer_symbol_ref is None) or (
            self.peer_symbol_ref is None
        ) != (self.peer_core is None):
            raise SchemaError("FSM/peer symbol/core must be all present or absent", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleBarrierEventBinding:
    barrier_ref: str
    owner_task_ref: str
    source_task_ref: str
    destination_task_ref: str
    phase: PlanBarrierEventPhase
    opcode: RecordOpcode
    event_symbol_ref: str
    source_core_symbol_ref: str
    destination_core_symbol_ref: str
    source_core: LogicalCoreRef
    destination_core: LogicalCoreRef

    def validate(self, path: str) -> None:
        for name in (
            "barrier_ref", "owner_task_ref", "source_task_ref",
            "destination_task_ref", "event_symbol_ref",
            "source_core_symbol_ref", "destination_core_symbol_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.phase) is not PlanBarrierEventPhase:
            raise SchemaError("must be a PlanBarrierEventPhase", path=f"{path}.phase")
        if self.opcode not in (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT):
            raise SchemaError("barrier event requires EVENT_SET/WAIT", path=f"{path}.opcode")
        self.source_core.validate(f"{path}.source_core")
        self.destination_core.validate(f"{path}.destination_core")


@dataclass(frozen=True, slots=True)
class SwizzleCoreAddressABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_plan_ref: str
    source_projection_ref: str
    task_bindings: tuple[SwizzleTaskCoreBinding, ...]
    value_bindings: tuple[SwizzleValueAddressBinding, ...]
    buffer_bindings: tuple[SwizzleBufferAddressBinding, ...]
    runtime_bindings: tuple[SwizzleTaskRuntimeBinding, ...]
    barrier_events: tuple[SwizzleBarrierEventBinding, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic: object) -> "SwizzleCoreAddressABI":
        result = cls(
            schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("swizzle_core_address_abi", semantic, schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "source_plan_ref", "source_projection_ref",
            "task_bindings", "value_bindings", "buffer_bindings",
            "runtime_bindings", "barrier_events",
        )}

    def validate(self, path: str = "swizzle_core_address_abi") -> None:
        if self.schema_version != SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("producer_pass", "source_ir1_id", "source_plan_ref", "source_projection_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for field_name in ("task_bindings", "value_bindings", "buffer_bindings", "runtime_bindings", "barrier_events"):
            values = getattr(self, field_name)
            for index, value in enumerate(values):
                value.validate(f"{path}.{field_name}[{index}]")
        for values, key, field_name in (
            (self.task_bindings, lambda item: item.task_ref, "task_bindings"),
            (self.value_bindings, lambda item: (item.value_ref, item.slot), "value_bindings"),
            (self.buffer_bindings, lambda item: (item.rank, item.buffer_ref), "buffer_bindings"),
            (self.runtime_bindings, lambda item: item.task_ref, "runtime_bindings"),
            (self.barrier_events, lambda item: (item.owner_task_ref, item.opcode.value, item.event_symbol_ref), "barrier_events"),
        ):
            keys = tuple(key(item) for item in values)
            if len(keys) != len(set(keys)):
                raise SchemaError("contains duplicate bindings", path=f"{path}.{field_name}")
        expected = stable_artifact_id("swizzle_core_address_abi", self._semantic_key(), schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(self, ir1: IR1, plan: SwizzleFusionPlan, projection: SwizzleIr2Projection, path: str = "swizzle_core_address_abi") -> None:
        self.validate(path)
        ir1.validate("ir1")
        validate_swizzle_plan_projection(plan, projection, f"{path}.inputs")
        if (self.source_ir1_id, self.source_plan_ref, self.source_projection_ref) != (ir1.id, plan.id, projection.id):
            raise SchemaError("ABI provenance is not exact", path=path)
        tasks = {task.id: task for dag in projection.rank_dags for task in dag.tasks}
        task_bindings = {item.task_ref: item for item in self.task_bindings}
        if set(task_bindings) != set(tasks):
            raise SchemaError("task bindings must cover every task exactly once", path=f"{path}.task_bindings")
        cores = {(die.id, core.local_core_id): core for die in ir1.fabric.dies for core in die.cores}
        profiles = {profile.id: profile for profile in ir1.fabric.sram_profiles}
        by_core: dict[LogicalCoreRef, list[SwizzleTaskCoreBinding]] = {}
        for ref, task in tasks.items():
            binding = task_bindings[ref]
            core = cores.get((binding.logical_core.die_id, binding.logical_core.local_core_id))
            if core is None or core.runtime_core_id != binding.runtime_core_id or task.rank != binding.rank or task.die_id != binding.logical_core.die_id:
                raise SchemaError("task core binding disagrees with fabric/projection", path=f"{path}.task_bindings")
            by_core.setdefault(binding.logical_core, []).append(binding)
        expected_ranks = set(range(len(projection.rank_dags)))
        rank_placements = {rank: set() for rank in expected_ranks}
        for binding in self.task_bindings:
            rank_placements[binding.rank].add(
                (binding.logical_core, binding.runtime_core_id)
            )
        if set(rank_placements) != expected_ranks or any(
            len(placements) != 1 for placements in rank_placements.values()
        ):
            raise SchemaError(
                "each rank must bind exactly one logical/runtime core",
                path=f"{path}.task_bindings",
            )
        for bindings in by_core.values():
            if tuple(item.core_order for item in bindings) != tuple(range(len(bindings))):
                raise SchemaError("core order must be dense projection order", path=f"{path}.task_bindings")

        values = {value.id: value for dag in projection.rank_dags for value in dag.values}
        rank_core = {task.rank: task_bindings[task.id].logical_core for task in tasks.values()}
        buffers = {(buffer.rank, buffer.buffer_ref): buffer for dag in projection.rank_dags for buffer in dag.buffers}
        expected_value_keys = {
            (value.id, slot)
            for value in values.values()
            for slot in range(
                buffers[(value.rank, value.buffer_ref)].slot_count
                if value.buffer_ref is not None
                else 1
            )
        }
        value_bindings = {(item.value_ref, item.slot): item for item in self.value_bindings}
        if set(value_bindings) != expected_value_keys:
            raise SchemaError("value bindings must cover every value slot", path=f"{path}.value_bindings")
        from .swizzle_operand_abi import build_swizzle_operand_abi

        operand_abi = build_swizzle_operand_abi(ir1, plan, projection)
        views_by_key = {}
        for view in operand_abi.operands:
            key = (view.value_ref, view.slot)
            signature = (
                view.shape, view.layout, view.dtype,
                view.byte_offset, view.byte_extent,
            )
            prior = views_by_key.setdefault(key, signature)
            if prior != signature:
                raise SchemaError(
                    "one value-slot must have one exact typed view",
                    path=f"{path}.value_bindings",
                )
        task_orders = {
            item.task_ref: item.core_order for item in self.task_bindings
        }
        terminal_keys = {
            (value.id, 0)
            for ownership in projection.output_ownership
            for value in projection.rank_dags[ownership.rank].values
            if not value.consumer_task_refs
            and len(value.producer_task_refs) == 1
            and any(
                value.symbolic_ref == boundary_ref
                or value.symbolic_ref.startswith(f"{boundary_ref}::")
                for boundary_ref in ownership.boundary_output_refs
            )
        }
        all_views_by_key = {}
        for view in operand_abi.operands:
            all_views_by_key.setdefault((view.value_ref, view.slot), []).append(view)
        buffer_index = {
            (item.rank, item.buffer_ref): item
            for dag in projection.rank_dags for item in dag.buffers
        }
        for (ref, slot), binding in value_bindings.items():
            value = values[ref]
            if binding.rank != value.rank or binding.logical_core != rank_core[value.rank]:
                raise SchemaError("value address has wrong rank/core", path=f"{path}.value_bindings")
            if value.buffer_ref is None and slot != 0:
                raise SchemaError("unbuffered value only admits slot zero", path=f"{path}.value_bindings")
            core = cores[(binding.logical_core.die_id, binding.logical_core.local_core_id)]
            region = next(
                (
                    item
                    for item in profiles[core.sram_profile_ref].regions
                    if item.id == binding.region_ref
                ),
                None,
            )
            if (
                region is None
                or binding.address < region.base_bytes
                or binding.address + binding.size_bytes
                > region.base_bytes + region.size_bytes
            ):
                raise SchemaError("value address is outside its exact SRAM region", path=f"{path}.value_bindings")
            views = all_views_by_key.get((ref, slot), ())
            if views:
                lifetime_start = min(task_orders[item.task_ref] for item in views)
                lifetime_end = max(task_orders[item.task_ref] for item in views) + 1
            else:
                buffer = buffer_index[(value.rank, value.buffer_ref)]
                orders = tuple(task_orders[item] for item in buffer.lifetime_task_refs)
                lifetime_start, lifetime_end = min(orders), max(orders) + 1
            if (ref, slot) in terminal_keys:
                lifetime_end = len(projection.rank_dags[value.rank].tasks)
            if (
                binding.lifetime_start,
                binding.lifetime_end_exclusive,
            ) != (lifetime_start, lifetime_end):
                raise SchemaError(
                    "value binding lifetime disagrees with exact task uses",
                    path=f"{path}.value_bindings",
                )
        storage_bases = {}
        for binding in self.value_bindings:
            key = (
                binding.rank, binding.logical_core,
                binding.region_ref, binding.storage_ref,
            )
            base = binding.address - binding.storage_offset_bytes
            if base < 0 or storage_bases.setdefault(key, base) != base:
                raise SchemaError(
                    "storage_ref/slice disagrees with one exact physical root base",
                    path=f"{path}.value_bindings",
                )

        buffer_bindings = {(item.rank, item.buffer_ref): item for item in self.buffer_bindings}
        if set(buffer_bindings) != set(buffers):
            raise SchemaError("buffer bindings must cover every buffer", path=f"{path}.buffer_bindings")
        for key, buffer in buffers.items():
            binding = buffer_bindings[key]
            expected_slices = tuple(
                (value_ref, slot)
                for value_ref in sorted(buffer.value_refs)
                for slot in range(buffer.slot_count)
            )
            if binding.logical_core != rank_core[buffer.rank] or tuple((item.value_ref, item.slot) for item in binding.slices) != expected_slices:
                raise SchemaError("buffer region/span/slices are not exact", path=f"{path}.buffer_bindings")
            core = cores[(binding.logical_core.die_id, binding.logical_core.local_core_id)]
            region = next(
                (
                    item
                    for item in profiles[core.sram_profile_ref].regions
                    if item.id == binding.region_name
                ),
                None,
            )
            if (
                region is None
                or binding.base_address < region.base_bytes
                or binding.base_address + binding.span_bytes
                > region.base_bytes + region.size_bytes
            ):
                raise SchemaError("buffer is outside its exact SRAM region", path=f"{path}.buffer_bindings")
            for value_index, value_ref in enumerate(sorted(buffer.value_refs)):
                for slot in range(buffer.slot_count):
                    item = next(
                        slice_binding
                        for slice_binding in binding.slices
                        if (slice_binding.value_ref, slice_binding.slot)
                        == (value_ref, slot)
                    )
                    value_address = value_bindings[(item.value_ref, item.slot)]
                    if item.size_bytes != buffer.size_bytes or value_address.region_ref != binding.region_name or value_address.address != binding.base_address + item.offset_bytes or value_address.size_bytes != buffer.size_bytes:
                        raise SchemaError("buffer slice disagrees with value address", path=f"{path}.buffer_bindings")
            expected_span = max(
                item.offset_bytes + item.size_bytes for item in binding.slices
            )
            if binding.span_bytes != expected_span:
                raise SchemaError(
                    "buffer region/span/slices are not exact",
                    path=f"{path}.buffer_bindings",
                )
        regions_by_core = {}
        for key, binding in buffer_bindings.items():
            regions_by_core.setdefault(binding.logical_core, []).append(
                (binding.base_address, binding.base_address + binding.span_bytes, key)
            )
        for regions in regions_by_core.values():
            ordered = sorted(regions)
            for left, right in zip(ordered, ordered[1:]):
                if left[1] > right[0]:
                    raise SchemaError("buffer regions on one core must not overlap", path=f"{path}.buffer_bindings")
        reduction_aliases = {
            frozenset((task.write_value_refs[0], task.read_value_refs[1]))
            for task in tasks.values()
            if task.kind is SwizzleActionKind.REDUCE
            and len(task.read_value_refs) == 2
            and len(task.write_value_refs) == 1
        }
        reduction_buffer_aliases = {}
        for task in tasks.values():
            if (
                task.kind is not SwizzleActionKind.REDUCE
                or len(task.read_value_refs) != 2
                or len(task.write_value_refs) != 1
            ):
                continue
            accumulator_ref = task.read_value_refs[1]
            accumulator_value = values[accumulator_ref]
            slot_by_buffer = {use.buffer_ref: use.slot for use in task.buffer_uses}
            if accumulator_value.buffer_ref is not None:
                staged_buffer_slot = (
                    accumulator_value.buffer_ref,
                    slot_by_buffer[accumulator_value.buffer_ref],
                )
                reduction_buffer_aliases[task.write_value_refs[0]] = (
                    staged_buffer_slot
                )
                reduction_buffer_aliases.setdefault(
                    task.read_value_refs[0], staged_buffer_slot
                )
        ranges_by_core = {}
        for (ref, slot), binding in value_bindings.items():
            ranges_by_core.setdefault(binding.logical_core, []).append(
                (binding, ref, slot)
            )
        for ranges in ranges_by_core.values():
            for index, (left, left_ref, left_slot) in enumerate(ranges):
                for right, right_ref, right_slot in ranges[index + 1 :]:
                    overlaps = (
                        left.address < right.address + right.size_bytes
                        and right.address < left.address + left.size_bytes
                    )
                    if not overlaps:
                        continue
                    exact_update_alias = (
                        (left.address, left.size_bytes)
                        == (right.address, right.size_bytes)
                        and frozenset((left_ref, right_ref)) in reduction_aliases
                    )
                    left_value = values[left_ref]
                    right_value = values[right_ref]
                    left_buffer_slot = (
                        (left_value.buffer_ref, left_slot)
                        if left_value.buffer_ref is not None
                        else reduction_buffer_aliases.get(left_ref)
                    )
                    right_buffer_slot = (
                        (right_value.buffer_ref, right_slot)
                        if right_value.buffer_ref is not None
                        else reduction_buffer_aliases.get(right_ref)
                    )
                    exact_span = (
                        left.address, left.size_bytes,
                    ) == (right.address, right.size_bytes)
                    same_storage = (
                        left.storage_ref == right.storage_ref
                        and left.storage_offset_bytes
                        == right.storage_offset_bytes
                    )
                    lifetime_disjoint = (
                        left.lifetime_end_exclusive <= right.lifetime_start
                        or right.lifetime_end_exclusive <= left.lifetime_start
                    )
                    exact_buffer_reuse = (
                        exact_span
                        and same_storage
                        and lifetime_disjoint
                        and left_buffer_slot is not None
                        and left_buffer_slot == right_buffer_slot
                        and views_by_key.get((left_ref, left_slot))
                        == views_by_key.get((right_ref, right_slot))
                        and views_by_key.get((left_ref, left_slot)) is not None
                    )
                    exact_invariant_alias = (
                        exact_span
                        and same_storage
                        and not left_value.producer_task_refs
                        and not right_value.producer_task_refs
                        and left_value.symbolic_ref == left_value.origin_ref
                        and right_value.symbolic_ref == right_value.origin_ref
                        and left_value.origin_ref == right_value.origin_ref
                        and views_by_key.get((left_ref, left_slot))
                        == views_by_key.get((right_ref, right_slot))
                        and views_by_key.get((left_ref, left_slot)) is not None
                    )
                    exact_boundary_chunk_reuse = (
                        exact_span
                        and same_storage
                        and lifetime_disjoint
                        and not left_value.producer_task_refs
                        and not right_value.producer_task_refs
                        and left_value.origin_kind
                        is SwizzleIr2ValueOriginKind.BOUNDARY_INPUT
                        and right_value.origin_kind
                        is SwizzleIr2ValueOriginKind.BOUNDARY_INPUT
                        and left_value.origin_ref == right_value.origin_ref
                        and left_value.symbolic_ref.startswith(
                            f"{left_value.origin_ref}::"
                        )
                        and right_value.symbolic_ref.startswith(
                            f"{right_value.origin_ref}::"
                        )
                        and views_by_key.get((left_ref, left_slot))
                        == views_by_key.get((right_ref, right_slot))
                        and views_by_key.get((left_ref, left_slot)) is not None
                    )
                    if not (
                        exact_update_alias
                        or exact_buffer_reuse
                        or exact_invariant_alias
                        or exact_boundary_chunk_reuse
                    ):
                        raise SchemaError(
                            "distinct value-slot ranges must not alias without exact reuse contract",
                            path=f"{path}.value_bindings",
                        )
        for task in tasks.values():
            if task.kind is not SwizzleActionKind.REDUCE:
                continue
            if len(task.read_value_refs) != 2 or len(task.write_value_refs) != 1:
                raise SchemaError("REDUCE requires two inputs and one output", path=f"{path}.value_bindings")
            slot_by_buffer = {use.buffer_ref: use.slot for use in task.buffer_uses}
            keys = []
            for value_ref in task.read_value_refs + task.write_value_refs:
                value = values[value_ref]
                keys.append(
                    (value_ref, slot_by_buffer.get(value.buffer_ref, 0))
                )
            first, accumulator, output = (value_bindings[key] for key in keys)
            if (
                first.size_bytes != accumulator.size_bytes
                or first.address + first.size_bytes != accumulator.address
                or (output.address, output.size_bytes)
                != (accumulator.address, accumulator.size_bytes)
            ):
                raise SchemaError(
                    "REDUCE requires ordered contiguous inputs and output alias of input1",
                    path=f"{path}.value_bindings",
                )
        for task in tasks.values():
            for use in task.buffer_uses:
                for value_ref in task.read_value_refs + task.write_value_refs:
                    value = values[value_ref]
                    if value.buffer_ref == use.buffer_ref and (value_ref, use.slot) not in value_bindings:
                        raise SchemaError("task buffer slot lacks an address binding", path=f"{path}.value_bindings")

        runtime_tasks = {task.id: task for task in tasks.values() if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT, SwizzleActionKind.LOCAL_COPY)}
        runtime = {item.task_ref: item for item in self.runtime_bindings}
        if set(runtime) != set(runtime_tasks):
            raise SchemaError("runtime bindings must exactly cover DTE/wait/copy tasks", path=f"{path}.runtime_bindings")
        flow_by_task = {}
        for flow in projection.flows:
            flow_by_task[flow.send_task_ref] = (flow, "send")
            flow_by_task[flow.recv_task_ref] = (flow, "recv")
        for ref, task in runtime_tasks.items():
            binding = runtime[ref]
            transport = task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
            if transport != (binding.fsm_symbol_ref is not None):
                raise SchemaError("transport FSM/peer closure is not exact", path=f"{path}.runtime_bindings")
            if transport:
                flow, role = flow_by_task[ref]
                peer_task_ref = flow.recv_task_ref if role == "send" else flow.send_task_ref
                peer_binding = runtime[peer_task_ref]
                expected_token = (
                    stable_artifact_id(
                        "swizzle_dte_token",
                        {"projection": projection.id, "flow": flow.id, "role": role},
                        schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
                    )
                    if role == "recv"
                    else None
                )
                expected_fsm = stable_artifact_id(
                    "swizzle_dte_fsm",
                    {"projection": projection.id, "flow": flow.id},
                    schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
                )
                expected_peer = stable_artifact_id(
                    "swizzle_peer_core",
                    {"projection": projection.id, "flow": flow.id, "role": role},
                    schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
                )
                if (
                    binding.flow_ref != flow.id
                    or binding.token_symbol_ref != expected_token
                    or binding.fsm_symbol_ref != expected_fsm
                    or binding.fsm_symbol_ref != peer_binding.fsm_symbol_ref
                    or binding.peer_symbol_ref != expected_peer
                    or binding.peer_core != rank_core[task.peer_rank]
                    or peer_binding.peer_core != rank_core[task.rank]
                ):
                    raise SchemaError("transport runtime binding disagrees with exact flow endpoints", path=f"{path}.runtime_bindings")
            elif binding.flow_ref is not None:
                raise SchemaError("non-transport runtime binding cannot reference a flow", path=f"{path}.runtime_bindings")
            if task.kind is SwizzleActionKind.WAIT:
                recv_deps = [dep for dep in task.deps if tasks[dep].kind is SwizzleActionKind.RECV]
                if (
                    binding.token_symbol_ref is None
                    or len(recv_deps) != 1
                    or binding.token_symbol_ref != runtime[recv_deps[0]].token_symbol_ref
                ):
                    raise SchemaError("WAIT must share the depended RECV token", path=f"{path}.runtime_bindings")
            elif task.kind is SwizzleActionKind.LOCAL_COPY and binding.token_symbol_ref is None:
                raise SchemaError("LOCAL_COPY requires an exact runtime token", path=f"{path}.runtime_bindings")

        barrier_actions = {action.sync.barrier.id: {} for program in plan.rank_programs for action in program.actions if action.sync.barrier is not None}
        source_to_task = {task.source_action_ref: task.id for task in tasks.values()}
        for program in plan.rank_programs:
            for action in program.actions:
                if action.sync.barrier is not None:
                    barrier_actions[action.sync.barrier.id][program.rank] = source_to_task[action.source_action.id]
        expected_events = set()
        for barrier_ref, by_rank in barrier_actions.items():
            leader_rank = min(by_rank)
            leader = by_rank[leader_rank]
            for peer_rank, peer in by_rank.items():
                if peer_rank == leader_rank:
                    continue
                expected_events.update((
                    (barrier_ref, peer, peer, leader, PlanBarrierEventPhase.ARRIVE, RecordOpcode.EVENT_SET),
                    (barrier_ref, leader, peer, leader, PlanBarrierEventPhase.ARRIVE, RecordOpcode.EVENT_WAIT),
                    (barrier_ref, leader, leader, peer, PlanBarrierEventPhase.RELEASE, RecordOpcode.EVENT_SET),
                    (barrier_ref, peer, leader, peer, PlanBarrierEventPhase.RELEASE, RecordOpcode.EVENT_WAIT),
                ))
        actual_events = {(item.barrier_ref, item.owner_task_ref, item.source_task_ref, item.destination_task_ref, item.phase, item.opcode) for item in self.barrier_events}
        if actual_events != expected_events:
            raise SchemaError("barrier event bindings are not exact", path=f"{path}.barrier_events")
        for item in self.barrier_events:
            if item.source_core != task_bindings[item.source_task_ref].logical_core or item.destination_core != task_bindings[item.destination_task_ref].logical_core:
                raise SchemaError("barrier event cores are not exact", path=f"{path}.barrier_events")
            semantic = {
                "projection": projection.id,
                "barrier": item.barrier_ref,
                "source": item.source_task_ref,
                "destination": item.destination_task_ref,
                "phase": item.phase,
            }
            expected_source = stable_artifact_id(
                "swizzle_barrier_source_core",
                semantic,
                schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
            )
            expected_destination = stable_artifact_id(
                "swizzle_barrier_destination_core",
                semantic,
                schema_version=SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
            )
            if (
                item.source_core_symbol_ref != expected_source
                or item.destination_core_symbol_ref != expected_destination
            ):
                raise SchemaError(
                    "barrier runtime-core symbols are not exact",
                    path=f"{path}.barrier_events",
                )
        if self.producer_pass == "swizzle_core_address_abi_allocator":
            from ..lowering.swizzle_abi import (
                _allocate_swizzle_address_bindings,
                build_swizzle_core_address_abi,
            )

            rank_cores, exact_values, exact_buffers = (
                _allocate_swizzle_address_bindings(ir1, plan, projection)
            )
            expected = build_swizzle_core_address_abi(
                ir1,
                plan,
                projection,
                rank_cores=rank_cores,
                value_bindings=exact_values,
                buffer_bindings=exact_buffers,
            )
            if self._semantic_key() != expected._semantic_key():
                raise SchemaError(
                    "ABI is not the exact deterministic physical allocation",
                    path=path,
                )


__all__ = [name for name in globals() if name.startswith("Swizzle") or name.startswith("SWIZZLE_")]
