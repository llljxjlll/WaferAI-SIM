"""Typed whole-workload projection over preserved and fused MoE overlay actions."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .swizzle_moe_execution import MoeScaleExecutionTerminalKind


MOE_SWIZZLE_WORKLOAD_PROJECTION_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_workload_projection/v1alpha1"


def _refs(values: tuple[str, ...], path: str, *, nonempty: bool = False) -> None:
    if type(values) is not tuple or (nonempty and not values) or len(values) != len(set(values)):
        raise SchemaError("refs must be an immutable unique tuple", path=path)
    for index, value in enumerate(values):
        validate_nonempty(value, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadAction:
    id: str
    rank: int
    die_id: int
    kind: str
    preserved: bool
    source_action_refs: tuple[str, ...]
    replacement_task_ref: str | None
    deps: tuple[str, ...]
    read_value_refs: tuple[str, ...]
    write_value_refs: tuple[str, ...]
    token_index: int | None
    expert_index: int | None
    role: str
    bytes: int
    flops: int
    dtype: DType
    state_action_binding_ref: str | None

    def validate(self, path: str) -> None:
        for name in ("id", "kind", "role"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("rank", "die_id", "bytes", "flops"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for name in ("token_index", "expert_index"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        if type(self.preserved) is not bool or type(self.dtype) is not DType:
            raise SchemaError("preserved/dtype is not typed", path=path)
        for name in ("source_action_refs", "deps", "read_value_refs", "write_value_refs"):
            _refs(getattr(self, name), f"{path}.{name}")
        if self.preserved:
            if len(self.source_action_refs) != 1 or self.replacement_task_ref is not None:
                raise SchemaError("preserved action requires one source and no replacement task", path=path)
            if self.kind not in (
                "preserved.dma_in", "preserved.swiglu", "preserved.tape_copy",
            ):
                raise SchemaError("unsupported preserved workload action", path=f"{path}.kind")
        else:
            if self.replacement_task_ref != self.id:
                raise SchemaError("replacement action must identify its exact IR2 task", path=f"{path}.replacement_task_ref")
        if self.state_action_binding_ref is not None:
            validate_nonempty(self.state_action_binding_ref, f"{path}.state_action_binding_ref")
        dma = self.kind == "preserved.dma_in"
        if dma != (self.state_action_binding_ref is not None):
            raise SchemaError("only preserved DMA_IN carries a state binding", path=path)
        if dma and (self.read_value_refs or len(self.write_value_refs) != 1 or self.bytes == 0 or self.flops):
            raise SchemaError("preserved DMA_IN contract is not exact", path=path)
        if self.kind == "preserved.swiglu" and (
            len(self.read_value_refs) != 2 or len(self.write_value_refs) != 1 or self.bytes or self.flops
        ):
            raise SchemaError("preserved SwiGLU contract is not exact", path=path)
        if self.kind == "preserved.tape_copy" and (
            len(self.read_value_refs) != 1 or len(self.write_value_refs) != 1 or self.bytes == 0 or self.flops
        ):
            raise SchemaError("preserved tape-copy contract is not exact", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadTerminal:
    kind: MoeScaleExecutionTerminalKind
    value_ref: str
    producer_action_ref: str
    token_index: int
    expert_index: int
    die_id: int
    shape: tuple[int, ...]
    bytes: int
    dtype: DType

    def validate(self, path: str) -> None:
        if type(self.kind) is not MoeScaleExecutionTerminalKind or type(self.dtype) is not DType:
            raise SchemaError("terminal kind/dtype is not typed", path=path)
        for name in ("value_ref", "producer_action_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("token_index", "expert_index", "die_id", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if not self.shape or any(type(item) is not int or item <= 0 for item in self.shape):
            raise SchemaError("terminal shape is invalid", path=f"{path}.shape")
        if self.bytes != 2 * prod(self.shape) or self.dtype is not DType.FP16:
            raise SchemaError("terminal bytes/dtype disagree with shape", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadStorageSlotAssignment:
    storage_unit_ref: str
    runtime_core_id: int
    family: str
    slot: int
    writer_action_ref: str
    consumer_action_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        for name in ("storage_unit_ref", "family", "writer_action_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.slot, f"{path}.slot")
        if self.family != "swiglu_output" or self.slot not in (0, 1):
            raise SchemaError("storage slot family/index is invalid", path=path)
        _refs(self.consumer_action_refs, f"{path}.consumer_action_refs", nonempty=True)


@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadProjection:
    schema_version: str
    producer_pass: str
    id: str
    source_execution_id: str
    source_overlay_id: str
    replacement_projection_id: str
    state_abi_id: str
    actions: tuple[MoeSwizzleWorkloadAction, ...]
    terminals: tuple[MoeSwizzleWorkloadTerminal, ...]
    endpoint_lane_edges: tuple[tuple[str, str], ...] = ()
    storage_reuse_edges: tuple[tuple[str, str], ...] = ()
    storage_slot_assignments: tuple[MoeSwizzleWorkloadStorageSlotAssignment, ...] = ()

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleWorkloadProjection":
        semantic.setdefault("endpoint_lane_edges", ())
        semantic.setdefault("storage_reuse_edges", ())
        semantic.setdefault("storage_slot_assignments", ())
        result = cls(
            MOE_SWIZZLE_WORKLOAD_PROJECTION_SCHEMA_VERSION,
            "project_moe_swizzle_whole_workload",
            stable_artifact_id(
                "moe_swizzle_workload_projection",
                semantic,
                schema_version=MOE_SWIZZLE_WORKLOAD_PROJECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_execution_id", "source_overlay_id",
                "replacement_projection_id", "state_abi_id", "actions", "terminals",
                "endpoint_lane_edges", "storage_reuse_edges",
                "storage_slot_assignments",
            )
        }

    def validate(self, path: str = "moe_swizzle_workload_projection") -> None:
        if self.schema_version != MOE_SWIZZLE_WORKLOAD_PROJECTION_SCHEMA_VERSION or self.producer_pass != "project_moe_swizzle_whole_workload":
            raise SchemaError("unsupported whole-workload projection schema/producer", path=path)
        for name in ("source_execution_id", "source_overlay_id", "replacement_projection_id", "state_abi_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.actions) is not tuple or not self.actions:
            raise SchemaError("whole-workload actions must be nonempty", path=f"{path}.actions")
        actions = {}
        for index, action in enumerate(self.actions):
            if type(action) is not MoeSwizzleWorkloadAction:
                raise SchemaError("requires exact MoeSwizzleWorkloadAction", path=f"{path}.actions[{index}]")
            action.validate(f"{path}.actions[{index}]")
            if action.id in actions:
                raise SchemaError("whole-workload action id is duplicated", path=f"{path}.actions")
            actions[action.id] = action
        if any(ref not in actions for action in actions.values() for ref in action.deps):
            raise SchemaError("whole-workload dependency is unknown", path=f"{path}.actions")
        remaining = {ref: set(item.deps) for ref, item in actions.items()}
        ready = sorted(ref for ref, deps in remaining.items() if not deps)
        visited = []
        while ready:
            ref = ready.pop(0)
            visited.append(ref)
            for other in sorted(remaining):
                if ref in remaining[other]:
                    remaining[other].remove(ref)
                    if not remaining[other] and other not in visited and other not in ready:
                        ready.append(other)
                        ready.sort()
        if len(visited) != len(actions):
            raise SchemaError("whole-workload action DAG contains a cycle", path=f"{path}.actions")
        if (
            type(self.endpoint_lane_edges) is not tuple
            or self.endpoint_lane_edges != tuple(sorted(set(self.endpoint_lane_edges)))
        ):
            raise SchemaError(
                "endpoint lane edges must be canonical/unique",
                path=f"{path}.endpoint_lane_edges",
            )
        for index, edge in enumerate(self.endpoint_lane_edges):
            if type(edge) is not tuple or len(edge) != 2:
                raise SchemaError(
                    "endpoint lane edge must be a predecessor/successor pair",
                    path=f"{path}.endpoint_lane_edges[{index}]",
                )
            predecessor, successor = edge
            if (
                predecessor not in actions or successor not in actions
                or predecessor not in actions[successor].deps
            ):
                raise SchemaError(
                    "endpoint lane edge is not materialized in the workload DAG",
                    path=f"{path}.endpoint_lane_edges[{index}]",
                )
        if (
            type(self.storage_reuse_edges) is not tuple
            or self.storage_reuse_edges
            != tuple(sorted(set(self.storage_reuse_edges)))
            or set(self.storage_reuse_edges).intersection(self.endpoint_lane_edges)
        ):
            raise SchemaError(
                "storage reuse edges must be canonical/unique and non-endpoint",
                path=f"{path}.storage_reuse_edges",
            )
        for index, edge in enumerate(self.storage_reuse_edges):
            if type(edge) is not tuple or len(edge) != 2:
                raise SchemaError(
                    "storage reuse edge must be a predecessor/successor pair",
                    path=f"{path}.storage_reuse_edges[{index}]",
                )
            predecessor, successor = edge
            if (
                predecessor not in actions or successor not in actions
                or predecessor not in actions[successor].deps
            ):
                raise SchemaError(
                    "storage reuse edge is not materialized in the workload DAG",
                    path=f"{path}.storage_reuse_edges[{index}]",
                )
        if (
            type(self.storage_slot_assignments) is not tuple
            or self.storage_slot_assignments != tuple(sorted(
                self.storage_slot_assignments,
                key=lambda item: (item.runtime_core_id, item.family, item.slot, item.storage_unit_ref),
            ))
        ):
            raise SchemaError(
                "storage slot assignments must be canonical",
                path=f"{path}.storage_slot_assignments",
            )
        units = set()
        for index, assignment in enumerate(self.storage_slot_assignments):
            if type(assignment) is not MoeSwizzleWorkloadStorageSlotAssignment:
                raise SchemaError(
                    "requires exact storage slot assignment",
                    path=f"{path}.storage_slot_assignments[{index}]",
                )
            assignment.validate(f"{path}.storage_slot_assignments[{index}]")
            if (
                assignment.storage_unit_ref in units
                or assignment.writer_action_ref not in actions
                or any(ref not in actions for ref in assignment.consumer_action_refs)
            ):
                raise SchemaError(
                    "storage slot assignment action/unit closure is not exact",
                    path=f"{path}.storage_slot_assignments[{index}]",
                )
            units.add(assignment.storage_unit_ref)
        if type(self.terminals) is not tuple or not self.terminals:
            raise SchemaError("whole-workload terminals must be nonempty", path=f"{path}.terminals")
        terminal_refs = set()
        for index, terminal in enumerate(self.terminals):
            if type(terminal) is not MoeSwizzleWorkloadTerminal:
                raise SchemaError("requires exact MoeSwizzleWorkloadTerminal", path=f"{path}.terminals[{index}]")
            terminal.validate(f"{path}.terminals[{index}]")
            if terminal.value_ref in terminal_refs or terminal.producer_action_ref not in actions:
                raise SchemaError("terminal identity/producer closure is not exact", path=f"{path}.terminals")
            terminal_refs.add(terminal.value_ref)
        expected = stable_artifact_id(
            "moe_swizzle_workload_projection",
            self._semantic(),
            schema_version=MOE_SWIZZLE_WORKLOAD_PROJECTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable whole-workload projection id", path=f"{path}.id")


__all__ = [name for name in globals() if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE")]
