"""Physical multi-core ABI for the dedicated MoE Swizzle pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir1 import IR1
from .swizzle import SwizzleActionKind
from .swizzle_moe_ir2 import MoeSwizzleIr2Projection


MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_core_address_abi/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class MoeSwizzleTaskCoreBinding:
    task_ref: str
    rank: int
    logical_core: LogicalCoreRef
    runtime_core_id: int
    core_order: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_uint64(self.rank, f"{path}.rank")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.core_order, f"{path}.core_order")


@dataclass(frozen=True, slots=True)
class MoeSwizzleStorageRootBinding:
    rank: int
    logical_core: LogicalCoreRef
    region_ref: str
    buffer_ref: str
    slot: int
    storage_ref: str
    address: int
    span_bytes: int

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        self.logical_core.validate(f"{path}.logical_core")
        for name in ("region_ref", "buffer_ref", "storage_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.slot, f"{path}.slot")
        validate_uint64(self.address, f"{path}.address")
        validate_uint64(self.span_bytes, f"{path}.span_bytes")
        if self.slot not in (0, 1) or self.span_bytes == 0 or self.address % 64:
            raise SchemaError("storage root requires slot 0/1, positive span and 64B alignment", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleValueAddressBinding:
    value_ref: str
    rank: int
    logical_core: LogicalCoreRef
    region_ref: str
    slot: int
    storage_ref: str
    address: int
    size_bytes: int
    lifetime_start: int
    lifetime_end_exclusive: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_uint64(self.rank, f"{path}.rank")
        self.logical_core.validate(f"{path}.logical_core")
        validate_nonempty(self.region_ref, f"{path}.region_ref")
        validate_nonempty(self.storage_ref, f"{path}.storage_ref")
        for name in ("slot", "address", "size_bytes", "lifetime_start", "lifetime_end_exclusive"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.slot not in (0, 1)
            or self.size_bytes == 0
            or self.address % 2
            or self.size_bytes % 2
        ):
            raise SchemaError("value binding requires slot 0/1, positive FP16-aligned range", path=path)
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError("value lifetime must be nonempty", path=f"{path}.lifetime_end_exclusive")


@dataclass(frozen=True, slots=True)
class MoeSwizzleTaskRuntimeBinding:
    task_ref: str
    flow_ref: str | None
    token_symbol_ref: str | None
    fsm_symbol_ref: str | None
    peer_core: LogicalCoreRef | None
    copy_source_core: LogicalCoreRef | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        for name in ("flow_ref", "token_symbol_ref", "fsm_symbol_ref"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")
        if self.peer_core is not None:
            self.peer_core.validate(f"{path}.peer_core")
        if self.copy_source_core is not None:
            self.copy_source_core.validate(f"{path}.copy_source_core")


@dataclass(frozen=True, slots=True)
class MoeSwizzleCoreAddressABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_projection_id: str
    task_bindings: tuple[MoeSwizzleTaskCoreBinding, ...]
    storage_roots: tuple[MoeSwizzleStorageRootBinding, ...]
    value_bindings: tuple[MoeSwizzleValueAddressBinding, ...]
    runtime_bindings: tuple[MoeSwizzleTaskRuntimeBinding, ...]
    source_workload_projection_id: str | None = None

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleCoreAddressABI":
        result = cls(
            schema_version=MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
            producer_pass="moe_swizzle_core_address_abi_allocator",
            id=stable_artifact_id(
                "moe_swizzle_core_address_abi",
                semantic,
                schema_version=MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id", "source_projection_id", "task_bindings",
                "storage_roots", "value_bindings", "runtime_bindings",
                "source_workload_projection_id",
            )
        }

    def validate(self, path: str = "moe_swizzle_core_address_abi") -> None:
        if self.schema_version != MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION or self.producer_pass != "moe_swizzle_core_address_abi_allocator":
            raise SchemaError("unsupported MoE Swizzle ABI schema/producer", path=path)
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        validate_nonempty(self.source_projection_id, f"{path}.source_projection_id")
        if self.source_workload_projection_id is not None:
            validate_nonempty(
                self.source_workload_projection_id,
                f"{path}.source_workload_projection_id",
            )
        for field_name in ("task_bindings", "storage_roots", "value_bindings", "runtime_bindings"):
            for index, item in enumerate(getattr(self, field_name)):
                item.validate(f"{path}.{field_name}[{index}]")
        for values, key, field_name in (
            (self.task_bindings, lambda item: item.task_ref, "task_bindings"),
            (self.storage_roots, lambda item: (item.rank, item.logical_core, item.buffer_ref, item.slot), "storage_roots"),
            (self.value_bindings, lambda item: (item.value_ref, item.logical_core, item.slot), "value_bindings"),
            (self.runtime_bindings, lambda item: item.task_ref, "runtime_bindings"),
        ):
            keys = [key(item) for item in values]
            if len(keys) != len(set(keys)):
                raise SchemaError("contains duplicate bindings", path=f"{path}.{field_name}")
        expected = stable_artifact_id("moe_swizzle_core_address_abi", self._semantic(), schema_version=MOE_SWIZZLE_CORE_ADDRESS_ABI_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable ABI id", path=f"{path}.id")

    def validate_against(
        self,
        ir1: IR1,
        projection: MoeSwizzleIr2Projection,
        path: str = "moe_swizzle_core_address_abi",
        *,
        workload_projection: object | None = None,
    ) -> None:
        self.validate(path)
        ir1.validate("ir1")
        projection.validate("projection")
        if (self.source_ir1_id, self.source_projection_id) != (ir1.id, projection.id):
            raise SchemaError("ABI provenance is not exact", path=path)
        if workload_projection is None:
            if self.source_workload_projection_id is not None:
                raise SchemaError(
                    "whole-workload ABI requires its exact workload projection",
                    path=f"{path}.source_workload_projection_id",
                )
        else:
            workload_projection.validate(f"{path}.workload_projection")
            if (
                self.source_workload_projection_id != workload_projection.id
                or workload_projection.replacement_projection_id != projection.id
            ):
                raise SchemaError(
                    "whole-workload ABI provenance is not exact",
                    path=f"{path}.source_workload_projection_id",
                )
        tasks = {item.id: item for item in projection.tasks}
        values = {item.id: item for item in projection.values}
        bindings = {item.task_ref: item for item in self.task_bindings}
        if set(bindings) != set(tasks):
            raise SchemaError("task bindings must cover projection exactly", path=f"{path}.task_bindings")
        cores = {(die.id, core.local_core_id): core for die in ir1.fabric.dies for core in die.cores}
        profiles = {item.id: item for item in ir1.fabric.sram_profiles}
        by_core = {}
        for ref, binding in bindings.items():
            task = tasks[ref]
            core = cores.get((binding.logical_core.die_id, binding.logical_core.local_core_id))
            if core is None or core.runtime_core_id != binding.runtime_core_id or task.rank != binding.rank or task.die_id != binding.logical_core.die_id:
                raise SchemaError("task binding does not name a real matching core", path=f"{path}.task_bindings")
            by_core.setdefault(binding.logical_core, []).append(binding)
        for core_bindings in by_core.values():
            if sorted(item.core_order for item in core_bindings) != list(range(len(core_bindings))):
                raise SchemaError("per-core order must be dense", path=f"{path}.task_bindings")
        roots = {item.storage_ref: item for item in self.storage_roots}
        for root in roots.values():
            core = cores[(root.logical_core.die_id, root.logical_core.local_core_id)]
            region = next((item for item in profiles[core.sram_profile_ref].regions if item.id == root.region_ref), None)
            if region is None or root.address < region.base_bytes or root.address + root.span_bytes > region.base_bytes + region.size_bytes:
                raise SchemaError("storage root exceeds exact SRAM region", path=f"{path}.storage_roots")
        intervals = {}
        for root in roots.values():
            ranges = intervals.setdefault((root.logical_core, root.region_ref), [])
            if any(root.address < end and start < root.address + root.span_bytes for start, end in ranges):
                raise SchemaError("storage roots overlap on one core", path=f"{path}.storage_roots")
            ranges.append((root.address, root.address + root.span_bytes))
        value_bindings = {}
        for binding in self.value_bindings:
            root = roots.get(binding.storage_ref)
            if root is None or (binding.rank, binding.logical_core, binding.region_ref, binding.slot) != (root.rank, root.logical_core, root.region_ref, root.slot) or binding.address < root.address or binding.address + binding.size_bytes > root.address + root.span_bytes:
                raise SchemaError("value binding is not contained in its exact root", path=f"{path}.value_bindings")
            value_bindings.setdefault(binding.value_ref, []).append(binding)
        if set(value_bindings) != set(values):
            raise SchemaError("value bindings must cover every value", path=f"{path}.value_bindings")
        for ref, value in values.items():
            owners = value_bindings[ref]
            if not value.replicated and len(owners) != 1:
                raise SchemaError("non-replicated value must have one physical owner", path=f"{path}.value_bindings")
            if any(item.rank != value.rank or item.size_bytes != value.size_bytes for item in owners):
                raise SchemaError("value rank/extent changed in ABI", path=f"{path}.value_bindings")
        runtime = {item.task_ref: item for item in self.runtime_bindings}
        runtime_tasks = {
            item.id for item in projection.tasks
            if item.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT, SwizzleActionKind.LOCAL_COPY)
        }
        if set(runtime) != runtime_tasks:
            raise SchemaError("runtime bindings must exactly cover transport/copy/wait", path=f"{path}.runtime_bindings")
        for flow in projection.flows:
            send, recv, wait = (runtime[ref] for ref in (flow.send_task_ref, flow.recv_task_ref, flow.wait_task_ref))
            send_core, recv_core = bindings[flow.send_task_ref].logical_core, bindings[flow.recv_task_ref].logical_core
            if (
                send.flow_ref != flow.id or recv.flow_ref != flow.id or wait.flow_ref != flow.id
                or send.peer_core != recv_core or recv.peer_core != send_core
                or send.token_symbol_ref is not None or recv.token_symbol_ref is None
                or wait.token_symbol_ref != recv.token_symbol_ref
                or send.fsm_symbol_ref is None or send.fsm_symbol_ref != recv.fsm_symbol_ref
            ):
                raise SchemaError("flow runtime FSM/token/core closure mismatch", path=f"{path}.runtime_bindings")
        # Every ordinary local value edge stays on one core.  LOCAL_COPY is the
        # sole explicit bridge and binds its source core separately.
        def value_producer_core(value: object) -> LogicalCoreRef | None:
            if value.producer_task_ref is not None:
                return bindings[value.producer_task_ref].logical_core
            if len(value.alias_source_refs) == 1:
                return value_producer_core(values[value.alias_source_refs[0]])
            return None

        for value in values.values():
            producer_core = value_producer_core(value)
            for consumer_ref in value.consumer_task_refs:
                consumer = tasks[consumer_ref]
                consumer_core = bindings[consumer_ref].logical_core
                if producer_core is not None and producer_core != consumer_core and consumer.kind is not SwizzleActionKind.LOCAL_COPY:
                    raise SchemaError("cross-core value edge requires explicit LOCAL_COPY", path=f"{path}.task_bindings")
                if consumer.kind is SwizzleActionKind.LOCAL_COPY and runtime[consumer_ref].copy_source_core != producer_core:
                    raise SchemaError("LOCAL_COPY source core is not exact", path=f"{path}.runtime_bindings")
        # Binary reduce requires contiguous inputs and an in-place accumulator.
        for task in tasks.values():
            if task.kind is not SwizzleActionKind.REDUCE:
                continue
            core = bindings[task.id].logical_core
            slot_by_buffer = {item.buffer_ref: item.slot for item in task.buffer_uses}
            selected = []
            for value_ref in task.read_value_refs + task.write_value_refs:
                value = values[value_ref]
                slot = slot_by_buffer.get(value.buffer_ref, 0)
                selected.append(next(item for item in value_bindings[value_ref] if item.logical_core == core and item.slot == slot))
            first, accumulator, output = selected
            if first.size_bytes != accumulator.size_bytes or first.address + first.size_bytes != accumulator.address or (output.address, output.size_bytes) != (accumulator.address, accumulator.size_bytes):
                raise SchemaError("binary REDUCE requires contiguous inputs and in-place accumulator", path=f"{path}.value_bindings")


__all__ = [name for name in globals() if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE")]
