"""Exact physical ABI for the executable UNFUSED comparison branch."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import PlanBarrierEventPhase, RecordOpcode
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir1 import IR1
from .swizzle import SwizzleActionKind
from .swizzle_abi import (
    SwizzleBarrierEventBinding,
    SwizzleTaskCoreBinding,
    SwizzleTaskRuntimeBinding,
)
from .swizzle_unfused import UnfusedComparisonPlan, UnfusedComparisonProjection


UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION = (
    "wafer_frontend.unfused_comparison_core_abi/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonStorageBinding:
    rank: int
    storage_ref: str
    logical_core: LogicalCoreRef
    region_ref: str
    base_address: int
    size_bytes: int
    alignment_bytes: int
    lifetime_start: int
    lifetime_end_exclusive: int

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.storage_ref, f"{path}.storage_ref")
        self.logical_core.validate(f"{path}.logical_core")
        validate_nonempty(self.region_ref, f"{path}.region_ref")
        for name in (
            "base_address", "size_bytes", "alignment_bytes",
            "lifetime_start", "lifetime_end_exclusive",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.size_bytes == 0
            or self.alignment_bytes == 0
            or self.alignment_bytes & (self.alignment_bytes - 1)
            or self.base_address % self.alignment_bytes
        ):
            raise SchemaError(
                "storage requires positive aligned physical span",
                path=path,
            )
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError(
                "storage lifetime must be a non-empty half-open interval",
                path=f"{path}.lifetime_end_exclusive",
            )


@dataclass(frozen=True, slots=True)
class UnfusedComparisonCoreABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_plan_ref: str
    source_projection_ref: str
    task_bindings: tuple[SwizzleTaskCoreBinding, ...]
    storage_bindings: tuple[UnfusedComparisonStorageBinding, ...]
    runtime_bindings: tuple[SwizzleTaskRuntimeBinding, ...]
    barrier_events: tuple[SwizzleBarrierEventBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonCoreABI":
        result = cls(
            schema_version=UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION,
            producer_pass="unfused_comparison_abi_allocator",
            id=stable_artifact_id(
                "unfused_comparison_core_abi",
                semantic,
                schema_version=UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id", "source_plan_ref", "source_projection_ref",
                "task_bindings", "storage_bindings", "runtime_bindings",
                "barrier_events",
            )
        }

    def validate(self, path: str = "unfused_comparison_core_abi") -> None:
        if self.schema_version != UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "unfused_comparison_abi_allocator":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        for name in ("source_ir1_id", "source_plan_ref", "source_projection_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for field_name in (
            "task_bindings", "storage_bindings", "runtime_bindings", "barrier_events",
        ):
            for index, item in enumerate(getattr(self, field_name)):
                item.validate(f"{path}.{field_name}[{index}]")
        for values, key, field_name in (
            (self.task_bindings, lambda item: item.task_ref, "task_bindings"),
            (self.storage_bindings, lambda item: (item.rank, item.storage_ref), "storage_bindings"),
            (self.runtime_bindings, lambda item: item.task_ref, "runtime_bindings"),
        ):
            keys = tuple(key(item) for item in values)
            if len(keys) != len(set(keys)):
                raise SchemaError("contains duplicate bindings", path=f"{path}.{field_name}")
        expected = stable_artifact_id(
            "unfused_comparison_core_abi",
            self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self,
        ir1: IR1,
        plan: UnfusedComparisonPlan,
        projection: UnfusedComparisonProjection,
        path: str = "unfused_comparison_core_abi",
    ) -> None:
        self.validate(path)
        projection.validate_against(ir1, plan, f"{path}.projection")
        if (
            self.source_ir1_id,
            self.source_plan_ref,
            self.source_projection_ref,
        ) != (ir1.id, plan.id, projection.id):
            raise SchemaError("ABI provenance is not exact", path=path)
        actions = {
            action.id: action
            for program in plan.rank_programs
            for action in program.actions
        }
        task_bindings = {item.task_ref: item for item in self.task_bindings}
        if set(task_bindings) != set(actions):
            raise SchemaError("task bindings must exactly cover actions", path=f"{path}.task_bindings")
        cores = {
            (die.id, core.local_core_id): core
            for die in ir1.fabric.dies
            for core in die.cores
        }
        for rank_projection in projection.ranks:
            rank_bindings = [task_bindings[ref] for ref in rank_projection.task_refs]
            if tuple(item.core_order for item in rank_bindings) != tuple(range(len(rank_bindings))):
                raise SchemaError("task order must be dense rank order", path=f"{path}.task_bindings")
            placements = {
                (item.logical_core, item.runtime_core_id) for item in rank_bindings
            }
            if len(placements) != 1:
                raise SchemaError("each rank requires one exact core", path=f"{path}.task_bindings")
            for item in rank_bindings:
                core = cores.get((item.logical_core.die_id, item.logical_core.local_core_id))
                if (
                    item.rank != rank_projection.rank
                    or item.logical_core.die_id != rank_projection.die_id
                    or core is None
                    or core.runtime_core_id != item.runtime_core_id
                ):
                    raise SchemaError("task binding disagrees with fabric/rank", path=f"{path}.task_bindings")
        order_by_task = {
            item.task_ref: item.core_order for item in self.task_bindings
        }
        expected_storage = {}
        for operand in projection.operands:
            key = (actions[operand.task_ref].rank, operand.storage_ref)
            order = order_by_task[operand.task_ref]
            previous = expected_storage.setdefault(
                key,
                [operand.storage_bytes, order, order + 1],
            )
            if previous[0] != operand.storage_bytes:
                raise SchemaError(
                    "one storage ref has inconsistent spans",
                    path=f"{path}.projection.operands",
                )
            previous[1] = min(previous[1], order)
            previous[2] = max(previous[2], order + 1)
        if any(
            expected_storage[
                (actions[item.task_ref].rank, item.storage_ref)
            ][0] != item.storage_bytes
            for item in projection.operands
        ):
            raise SchemaError("one storage ref has inconsistent spans", path=f"{path}.projection.operands")
        storage = {(item.rank, item.storage_ref): item for item in self.storage_bindings}
        if set(storage) != set(expected_storage):
            raise SchemaError("storage bindings must exactly cover typed storage", path=f"{path}.storage_bindings")
        profiles = {item.id: item for item in ir1.fabric.sram_profiles}
        ranges: dict[
            LogicalCoreRef,
            list[UnfusedComparisonStorageBinding],
        ] = {}
        for key, (size, lifetime_start, lifetime_end) in expected_storage.items():
            item = storage[key]
            if (
                item.size_bytes,
                item.lifetime_start,
                item.lifetime_end_exclusive,
            ) != (size, lifetime_start, lifetime_end):
                raise SchemaError(
                    "storage span/lifetime is not exact",
                    path=f"{path}.storage_bindings",
                )
            core = cores[(item.logical_core.die_id, item.logical_core.local_core_id)]
            region = next((r for r in profiles[core.sram_profile_ref].regions if r.id == item.region_ref), None)
            if region is None or item.base_address < region.base_bytes or item.base_address + item.size_bytes > region.base_bytes + region.size_bytes:
                raise SchemaError("storage is outside exact SRAM region", path=f"{path}.storage_bindings")
            ranges.setdefault(item.logical_core, []).append(item)
        for items in ranges.values():
            for index, left in enumerate(items):
                for right in items[index + 1 :]:
                    physical_overlap = (
                        left.region_ref == right.region_ref
                        and left.base_address < right.base_address + right.size_bytes
                        and right.base_address < left.base_address + left.size_bytes
                    )
                    lifetime_overlap = (
                        left.lifetime_start < right.lifetime_end_exclusive
                        and right.lifetime_start < left.lifetime_end_exclusive
                    )
                    if physical_overlap and lifetime_overlap:
                        raise SchemaError(
                            "physical storage spans overlap during live intervals",
                            path=f"{path}.storage_bindings",
                        )
        operands_by_task = {}
        for operand in projection.operands:
            operands_by_task.setdefault(operand.task_ref, []).append(operand)
        for task_ref, action in actions.items():
            if action.kind is not SwizzleActionKind.REDUCE:
                continue
            views = sorted(
                operands_by_task[task_ref], key=lambda item: item.ordinal
            )
            source, accumulator, _output = views
            if source.storage_ref == accumulator.storage_ref:
                continue
            source_binding = storage[(action.rank, source.storage_ref)]
            accumulator_binding = storage[(action.rank, accumulator.storage_ref)]
            if (
                source_binding.logical_core != accumulator_binding.logical_core
                or source_binding.region_ref != accumulator_binding.region_ref
                or source_binding.size_bytes != accumulator_binding.size_bytes
                or source_binding.base_address + source_binding.size_bytes
                != accumulator_binding.base_address
            ):
                raise SchemaError(
                    "split REDUCE inputs must form one exact contiguous physical span",
                    path=f"{path}.storage_bindings",
                )
        from ..lowering.swizzle_unfused import allocate_unfused_comparison_core_abi

        expected = allocate_unfused_comparison_core_abi(ir1, plan, projection)
        if self != expected:
            raise SchemaError("ABI is not the exact deterministic allocation", path=path)


__all__ = [name for name in globals() if name.startswith("Unfused") or name.startswith("UNFUSED_")]
