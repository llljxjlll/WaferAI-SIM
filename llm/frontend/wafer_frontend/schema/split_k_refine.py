"""Exact provenance for the first split-K intra-die graph refinement."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import GemmWorkload
from .ir1 import IR1, MemoryInitiator
from .intra_die_v2_search import (
    IntraDieV2CandidateKind,
    IntraDieV2SearchDecision,
)
from .ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
)


SPLIT_K_TASK_REWRITE_SCHEMA_VERSION = (
    "wafer_frontend.split_k_task_rewrite/v1alpha5"
)
SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION = (
    "wafer_frontend.split_k_refined_projection/v1alpha3"
)


def split_k_part_task_id(source_task_id: str, part_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}"


def split_k_partial_value_id(
    source_task_id: str, output_value_id: str, part_index: int
) -> str:
    return (
        f"{output_value_id}::refine.{source_task_id}.split_k.part.{part_index}"
    )


def split_k_reduce_task_id(source_task_id: str) -> str:
    return f"{source_task_id}.split_k.reduce"


def split_k_reduce_step_task_id(
    source_task_id: str, step_index: int, split_k_parts: int
) -> str:
    """Canonical streaming-reduce task id; the last step is the commit."""
    if step_index == split_k_parts - 1:
        return split_k_reduce_task_id(source_task_id)
    return f"{source_task_id}.split_k.reduce.step.{step_index}"


def split_k_accumulator_value_id(
    source_task_id: str, output_value_id: str, step_index: int
) -> str:
    return (
        f"{output_value_id}::refine.{source_task_id}."
        f"split_k.accumulator.step.{step_index}"
    )


def split_k_stage_task_id(
    source_task_id: str, part_index: int, operand_index: int
) -> str:
    return (
        f"{source_task_id}.split_k.part.{part_index}."
        f"input.{operand_index}.stage"
    )


def split_k_pack_value_id(
    source_task_id: str, source_value_id: str, part_index: int, operand_index: int
) -> str:
    return (
        f"{source_value_id}::refine.{source_task_id}.split_k.part.{part_index}."
        f"input.{operand_index}.packed"
    )


def split_k_pack_row_task_id(
    source_task_id: str, part_index: int, operand_index: int, row_index: int
) -> str:
    return (
        f"{source_task_id}.split_k.part.{part_index}.input.{operand_index}."
        f"pack.row.{row_index}"
    )


def split_k_staged_value_id(
    source_task_id: str,
    source_value_id: str,
    part_index: int,
    operand_index: int,
    compute_group_count: int = 2,
) -> str:
    slot = (part_index // compute_group_count) % 2
    return (
        f"{source_value_id}::refine.{source_task_id}.input.{operand_index}."
        f"slot.{slot}.version.{part_index}"
    )


def split_k_ready_event_id(
    source_task_id: str, part_index: int, operand_index: int,
    compute_group_count: int = 2,
) -> str:
    slot = (part_index // compute_group_count) % 2
    return (
        f"event.{source_task_id}.split_k.input.{operand_index}."
        f"slot.{slot}.version.{part_index}.ready"
    )


def split_k_compute_event_id(source_task_id: str, part_index: int) -> str:
    return f"event.{source_task_id}.split_k.part.{part_index}.done"


def split_k_slot_alias(source_task_id: str, operand_index: int, slot: int) -> str:
    return f"refine.{source_task_id}.input.{operand_index}.slot.{slot}"


def split_k_local_flow_id(source_task_id: str, part_index: int) -> str:
    return f"local.split_k.{source_task_id}.part.{part_index}.to_reduce"


def split_k_local_send_task_id(source_task_id: str, part_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}.local_send"


def split_k_local_recv_task_id(source_task_id: str, part_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}.local_recv"


def split_k_local_wait_task_id(source_task_id: str, part_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}.local_wait"


def split_k_input_flow_id(source_task_id: str, part_index: int, operand_index: int) -> str:
    return f"local.split_k.{source_task_id}.part.{part_index}.input.{operand_index}"


def split_k_input_send_task_id(source_task_id: str, part_index: int, operand_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}.input.{operand_index}.local_send"


def split_k_input_recv_task_id(source_task_id: str, part_index: int, operand_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}.input.{operand_index}.local_recv"


def split_k_input_wait_task_id(source_task_id: str, part_index: int, operand_index: int) -> str:
    return f"{source_task_id}.split_k.part.{part_index}.input.{operand_index}.local_wait"


def split_k_direct_dma_task_id(
    source_dma_task_id: str, split_source_task_id: str,
    part_index: int, operand_index: int,
) -> str:
    return (
        f"{source_dma_task_id}.split_k.direct_dma.{split_source_task_id}."
        f"part.{part_index}.input.{operand_index}"
    )


def split_k_direct_dma_region_id(task_id: str) -> str:
    return f"{task_id}.region"



@dataclass(frozen=True, slots=True)
class SplitKInputHandoff:
    part_index: int
    operand_index: int
    source_task_id: str
    send_dependency_task_id: str
    destination_task_id: str
    value_id: str
    destination_value_id: str
    flow_id: str
    send_task_id: str
    recv_task_id: str
    wait_task_id: str
    source_compute_group: int
    destination_compute_group: int

    def validate(
        self, split_source_task_id: str, compute_group_count: int, path: str
    ) -> None:
        validate_uint64(self.part_index, f"{path}.part_index")
        validate_uint64(self.operand_index, f"{path}.operand_index")
        for name in (
            "source_task_id", "send_dependency_task_id", "destination_task_id", "value_id", "destination_value_id", "flow_id",
            "send_task_id", "recv_task_id", "wait_task_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        expected = (
            split_k_input_flow_id(split_source_task_id, self.part_index, self.operand_index),
            split_k_input_send_task_id(split_source_task_id, self.part_index, self.operand_index),
            split_k_input_recv_task_id(split_source_task_id, self.part_index, self.operand_index),
            split_k_input_wait_task_id(split_source_task_id, self.part_index, self.operand_index),
            0, self.part_index % compute_group_count,
        )
        if (
            self.flow_id, self.send_task_id, self.recv_task_id, self.wait_task_id,
            self.source_compute_group, self.destination_compute_group,
        ) != expected or self.part_index % compute_group_count == 0:
            raise SchemaError(
                "input handoff must be canonical CG0 -> remote part group",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class SplitKLocalHandoff:
    """Exact odd-part CG1 -> reducer CG0 local-NoC cut."""

    part_index: int
    part_task_id: str
    partial_value_id: str
    reduce_task_id: str
    flow_id: str
    send_task_id: str
    recv_task_id: str
    wait_task_id: str
    source_compute_group: int
    destination_compute_group: int

    def validate(
        self, source_task_id: str, output_value_id: str,
        split_k_parts: int, compute_group_count: int,
        enable_tree_reduce: bool, path: str,
    ) -> None:
        validate_uint64(self.part_index, f"{path}.part_index")
        for name in (
            "part_task_id", "partial_value_id", "reduce_task_id", "flow_id",
            "send_task_id", "recv_task_id", "wait_task_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.source_compute_group, f"{path}.source_compute_group")
        validate_uint64(self.destination_compute_group, f"{path}.destination_compute_group")
        transport_identity = (
            self.flow_id, self.send_task_id, self.recv_task_id, self.wait_task_id
        )
        expected_transport = (
            split_k_local_flow_id(source_task_id, self.part_index),
            split_k_local_send_task_id(source_task_id, self.part_index),
            split_k_local_recv_task_id(source_task_id, self.part_index),
            split_k_local_wait_task_id(source_task_id, self.part_index),
        )
        if transport_identity != expected_transport:
            raise SchemaError("local handoff transport identity is not canonical", path=path)
        if enable_tree_reduce:
            if (
                not 1 <= self.part_index < split_k_parts
                or not 0 <= self.source_compute_group < compute_group_count
                or not 0 <= self.destination_compute_group < compute_group_count
                or self.source_compute_group == self.destination_compute_group
            ):
                raise SchemaError("tree handoff groups are invalid", path=path)
            return
        expected = (
            split_k_part_task_id(source_task_id, self.part_index),
            split_k_partial_value_id(source_task_id, output_value_id, self.part_index),
            split_k_reduce_step_task_id(source_task_id, self.part_index, split_k_parts),
            self.part_index % compute_group_count,
            0,
        )
        actual = (
            self.part_task_id, self.partial_value_id, self.reduce_task_id,
            self.source_compute_group, self.destination_compute_group,
        )
        if actual != expected or self.part_index % compute_group_count == 0:
            raise SchemaError(
                "local handoff must be the canonical remote group -> CG0 cut",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class DoubleBufferVersion:
    part_index: int
    temporal_chunk_index: int
    slot: int
    stage_task_ids: tuple[str, ...]
    staged_value_ids: tuple[str, ...]
    ready_event_ids: tuple[str, ...]
    direct_receive_operand_indices: tuple[int, ...]
    compute_event_id: str

    def validate(
        self,
        source_task_id: str,
        source_input_ids: tuple[str, ...],
        compute_group_count: int,
        path: str,
    ) -> None:
        validate_uint64(self.part_index, f"{path}.part_index")
        validate_uint64(self.temporal_chunk_index, f"{path}.temporal_chunk_index")
        validate_uint64(self.slot, f"{path}.slot")
        expected_chunk = self.part_index // compute_group_count
        if self.temporal_chunk_index != expected_chunk:
            raise SchemaError(
                "temporal chunk must be the canonical compute-group wave",
                path=f"{path}.temporal_chunk_index",
            )
        if self.slot != expected_chunk % 2:
            raise SchemaError("slot must alternate by temporal chunk", path=f"{path}.slot")
        operand_count = len(source_input_ids)
        if not operand_count or any(
            len(items) != operand_count
            for items in (
                self.stage_task_ids,
                self.staged_value_ids,
                self.ready_event_ids,
            )
        ):
            raise SchemaError(
                "double-buffer bindings must exactly cover compute inputs",
                path=path,
            )
        if self.direct_receive_operand_indices != tuple(
            sorted(set(self.direct_receive_operand_indices))
        ) or any(index >= operand_count for index in self.direct_receive_operand_indices):
            raise SchemaError(
                "direct receive operands must be unique canonical input indices",
                path=f"{path}.direct_receive_operand_indices",
            )
        if self.direct_receive_operand_indices and (
            self.part_index % compute_group_count == 0
        ):
            raise SchemaError(
                "direct receive is legal only for a remote compute group",
                path=f"{path}.direct_receive_operand_indices",
            )
        direct = set(self.direct_receive_operand_indices)
        expected_stage_tasks = tuple(
            split_k_input_wait_task_id(source_task_id, self.part_index, index)
            if index in direct
            else split_k_stage_task_id(source_task_id, self.part_index, index)
            for index in range(operand_count)
        )
        expected_values = tuple(
            split_k_staged_value_id(
                source_task_id,
                value_id,
                self.part_index,
                index,
                compute_group_count,
            )
            for index, value_id in enumerate(source_input_ids)
        )
        expected_events = tuple(
            split_k_input_wait_task_id(source_task_id, self.part_index, index)
            if index in direct
            else split_k_ready_event_id(
                source_task_id, self.part_index, index, compute_group_count
            )
            for index in range(operand_count)
        )
        if (
            self.stage_task_ids,
            self.staged_value_ids,
            self.ready_event_ids,
            self.compute_event_id,
        ) != (
            expected_stage_tasks,
            expected_values,
            expected_events,
            split_k_compute_event_id(source_task_id, self.part_index),
        ):
            raise SchemaError(
                "double-buffer value/task/event identities are not canonical",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class SplitKTaskRewrite:
    schema_version: str
    id: str
    source_dag_id: str
    source_task_id: str
    source_output_value_id: str
    part_task_ids: tuple[str, ...]
    partial_value_ids: tuple[str, ...]
    reduce_task_id: str | None
    reduction_task_ids: tuple[str, ...]
    reduction_accumulator_value_ids: tuple[str, ...]
    semantic_commit_task_id: str | None
    temporal_chunks: int
    double_buffer_versions: tuple[DoubleBufferVersion, ...]
    compute_group_count: int
    part_compute_groups: tuple[int, ...]
    reduce_compute_group: int | None
    local_handoffs: tuple[SplitKLocalHandoff, ...]
    input_handoffs: tuple[SplitKInputHandoff, ...]
    enable_tree_reduce: bool = False
    enable_direct_dma: bool = False
    direct_dma_task_ids: tuple[str, ...] = ()

    @classmethod
    def create(cls, **semantic: object) -> "SplitKTaskRewrite":
        return cls(
            schema_version=SPLIT_K_TASK_REWRITE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "split_k_task_rewrite",
                semantic,
                schema_version=SPLIT_K_TASK_REWRITE_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_dag_id",
                "source_task_id",
                "source_output_value_id",
                "part_task_ids",
                "partial_value_ids",
                "reduce_task_id",
                "reduction_task_ids",
                "reduction_accumulator_value_ids",
                "semantic_commit_task_id",
                "temporal_chunks",
                "double_buffer_versions",
                "compute_group_count",
                "part_compute_groups",
                "reduce_compute_group",
                "local_handoffs",
                "input_handoffs",
                "enable_tree_reduce",
                "enable_direct_dma",
                "direct_dma_task_ids",
            )
        }

    def validate(
        self,
        *,
        split_k_parts: int,
        enable_reduce: bool,
        enable_double_buffer: bool,
        enable_streaming_reduce: bool = True,
        enable_tree_reduce: bool = False,
        enable_direct_dma: bool = False,
        source_input_ids: tuple[str, ...],
        path: str = "split_k_task_rewrite",
    ) -> None:
        if self.schema_version != SPLIT_K_TASK_REWRITE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if (self.enable_tree_reduce, self.enable_direct_dma) != (
            enable_tree_reduce, enable_direct_dma
        ):
            raise SchemaError("rewrite optimization flags disagree", path=path)
        for name in (
            "source_dag_id",
            "source_task_id",
            "source_output_value_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        expected_parts = tuple(
            split_k_part_task_id(self.source_task_id, index)
            for index in range(split_k_parts)
        )
        expected_values = tuple(
            split_k_partial_value_id(
                self.source_task_id,
                self.source_output_value_id,
                index,
            )
            for index in range(split_k_parts)
        )
        if (self.part_task_ids, self.partial_value_ids) != (
            expected_parts,
            expected_values,
        ):
            raise SchemaError(
                "split task/value identities are not canonical", path=path
            )
        if enable_reduce:
            if (
                self.reduce_task_id
                != split_k_reduce_task_id(self.source_task_id)
                or self.semantic_commit_task_id is not None
            ):
                raise SchemaError(
                    "explicit reduce requires only its canonical reduce task",
                    path=path,
                )
        elif (
            self.reduce_task_id is not None
            or self.semantic_commit_task_id != self.source_task_id
        ):
            raise SchemaError(
                "reduce-disabled split-K must retain the source semantic commit",
                path=path,
            )
        expected_reduction_tasks = (
            tuple(
                split_k_reduce_step_task_id(self.source_task_id, index, split_k_parts)
                for index in range(1, split_k_parts)
            ) if enable_reduce else ()
        )
        expected_accumulators = (
            tuple(
                split_k_accumulator_value_id(
                    self.source_task_id, self.source_output_value_id, index
                )
                for index in range(1, split_k_parts - 1)
            ) if enable_reduce else ()
        )
        if (self.reduction_task_ids, self.reduction_accumulator_value_ids) != (
            expected_reduction_tasks, expected_accumulators
        ):
            raise SchemaError(
                "streaming reduce task/accumulator identities are not canonical",
                path=f"{path}.reduction_task_ids",
            )
        if enable_double_buffer:
            if tuple(item.part_index for item in self.double_buffer_versions) != tuple(
                range(split_k_parts)
            ):
                raise SchemaError(
                    "double-buffer versions must follow part order", path=path
                )
            for index, version in enumerate(self.double_buffer_versions):
                version.validate(
                    self.source_task_id,
                    source_input_ids,
                    self.compute_group_count,
                    f"{path}.double_buffer_versions[{index}]",
                )
        elif self.double_buffer_versions:
            raise SchemaError(
                "double-buffer versions require enable_double_buffer",
                path=f"{path}.double_buffer_versions",
            )
        validate_uint64(self.compute_group_count, f"{path}.compute_group_count")
        if not 1 <= self.compute_group_count <= split_k_parts:
            raise SchemaError(
                "compute_group_count must be in [1, split_k_parts]",
                path=f"{path}.compute_group_count",
            )
        expected_temporal_chunks = (split_k_parts + self.compute_group_count - 1) // self.compute_group_count
        if self.temporal_chunks != expected_temporal_chunks:
            raise SchemaError(
                "temporal_chunks must exactly count compute-group waves",
                path=f"{path}.temporal_chunks",

            )
        expected_groups = (
            tuple(index % self.compute_group_count for index in range(split_k_parts))
            if enable_reduce
            else (0,) * split_k_parts
        )
        if self.part_compute_groups != expected_groups:
            raise SchemaError(
                "part compute groups must be canonical parity groups",
                path=f"{path}.part_compute_groups",
            )
        if self.reduce_compute_group != (0 if enable_reduce else None):
            raise SchemaError(
                "reduce compute group must be CG0 exactly when reduce is enabled",
                path=f"{path}.reduce_compute_group",
            )
        expected_handoff_parts = (
            tuple(range(1, split_k_parts))
            if enable_reduce and enable_tree_reduce
            else tuple(
                index for index, group in enumerate(expected_groups) if group != 0
            ) if enable_reduce else ()
        )
        if tuple(item.part_index for item in self.local_handoffs) != expected_handoff_parts:
            raise SchemaError(
                "local handoffs must exactly cover parts outside reducer CG0",
                path=f"{path}.local_handoffs",
            )
        for index, handoff in enumerate(self.local_handoffs):
            if type(handoff) is not SplitKLocalHandoff:
                raise SchemaError(
                    "must be SplitKLocalHandoff",
                    path=f"{path}.local_handoffs[{index}]",
                )
            handoff.validate(
                self.source_task_id,
                self.source_output_value_id,
                split_k_parts,
                self.compute_group_count,
                enable_tree_reduce,
                f"{path}.local_handoffs[{index}]",
            )
        if self.compute_group_count == 1 and self.input_handoffs:
            raise SchemaError(
                "input handoffs require two compute groups",
                path=f"{path}.input_handoffs",
            )
        keys = tuple((item.part_index, item.operand_index) for item in self.input_handoffs)
        if keys != tuple(sorted(set(keys))):
            raise SchemaError(
                "input handoffs must use unique canonical part/operand order",
                path=f"{path}.input_handoffs",
            )
        for index, handoff in enumerate(self.input_handoffs):
            if type(handoff) is not SplitKInputHandoff:
                raise SchemaError(
                    "must be SplitKInputHandoff",
                    path=f"{path}.input_handoffs[{index}]",
                )
            handoff.validate(
                self.source_task_id,
                self.compute_group_count,
                f"{path}.input_handoffs[{index}]",
            )
        if self.enable_direct_dma:
            if (
                not self.direct_dma_task_ids
                or len(set(self.direct_dma_task_ids)) != len(self.direct_dma_task_ids)
            ):
                raise SchemaError(
                    "direct DMA task ids must be non-empty and unique",
                    path=f"{path}.direct_dma_task_ids",
                )
            for index, task_id in enumerate(self.direct_dma_task_ids):
                validate_nonempty(task_id, f"{path}.direct_dma_task_ids[{index}]")
        elif self.direct_dma_task_ids:
            raise SchemaError(
                "direct DMA ids require enable_direct_dma",
                path=f"{path}.direct_dma_task_ids",
            )
        expected_id = stable_artifact_id(
            "split_k_task_rewrite",
            self._semantic_key(),
            schema_version=SPLIT_K_TASK_REWRITE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


def _expected_compute_group_count(
    ir1: IR1, die_id: int, enable_double_buffer: bool, split_k_parts: int
) -> int:
    die = next(item for item in ir1.fabric.dies if item.id == die_id)
    profiles = {item.id: item for item in ir1.fabric.sram_profiles}
    common = {MemoryInitiator.COMPUTE}
    if enable_double_buffer:
        common.add(MemoryInitiator.LSU)
    source_required = common | {MemoryInitiator.DTE}
    destination_required = common | {MemoryInitiator.NOC_RX}
    def supports(core: object, required: set[MemoryInitiator]) -> bool:
        return any(
            required.issubset(region.access)
            for region in profiles[core.sram_profile_ref].regions
        )
    available = max(
        (
            1 + sum(
                destination.runtime_core_id != source.runtime_core_id
                and supports(destination, destination_required)
                for destination in die.cores
            )
            for source in die.cores
            if supports(source, source_required)
        ),
        default=1,
    )
    return min(split_k_parts, available)


@dataclass(frozen=True, slots=True)
class SplitKRefinedProjection:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    projection: IR2ProjectionResult
    rewrites: tuple[SplitKTaskRewrite, ...]
    search_decision: IntraDieV2SearchDecision

    @classmethod
    def create(
        cls,
        *,
        source_projection_id: str,
        projection: IR2ProjectionResult,
        rewrites: tuple[SplitKTaskRewrite, ...],
        search_decision: IntraDieV2SearchDecision,
    ) -> "SplitKRefinedProjection":
        semantic = {
            "source_projection_id": source_projection_id,
            "projection": projection,
            "rewrites": rewrites,
            "search_decision": search_decision,
        }
        return cls(
            schema_version=SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION,
            producer_pass="intra_die_refine",
            id=stable_artifact_id(
                "split_k_refined_projection",
                semantic,
                schema_version=SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_projection_id": self.source_projection_id,
            "projection": self.projection,
            "rewrites": self.rewrites,
            "search_decision": self.search_decision,
        }

    def validate_against(
        self,
        source: IR2ProjectionResult,
        ir1: IR1,
        *,
        split_k_parts: int,
        enable_reduce: bool,
        enable_double_buffer: bool,
        enable_streaming_reduce: bool | None = None,
        enable_tree_reduce: bool | None = None,
        enable_direct_dma: bool | None = None,
        path: str = "split_k_refined_projection",
    ) -> None:
        if self.schema_version != SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "intra_die_refine":
            raise SchemaError("must be 'intra_die_refine'", path=f"{path}.producer_pass")
        source.validate("source_projection")
        if type(ir1) is not IR1 or ir1.id != source.source_ir1_id:
            raise SchemaError(
                "must carry the exact source IR-1 hardware", path="ir1"
            )
        ir1.validate("ir1")
        self.projection.validate(f"{path}.projection")
        if self.source_projection_id != source.id:
            raise SchemaError(
                "references a different source projection",
                path=f"{path}.source_projection_id",
            )
        if type(self.search_decision) is not IntraDieV2SearchDecision:
            raise SchemaError(
                "must carry an IntraDieV2SearchDecision",
                path=f"{path}.search_decision",
            )
        self.search_decision.validate(f"{path}.search_decision")
        selected = next(
            (
                candidate
                for candidate in self.search_decision.candidates
                if candidate.id == self.search_decision.selected_candidate_ref
            ),
            None,
        )
        if enable_streaming_reduce is None:
            enable_streaming_reduce = (
                selected.enable_streaming_reduce
                if selected is not None
                else enable_reduce
            )
        if enable_tree_reduce is None:
            enable_tree_reduce = (
                selected.enable_tree_reduce if selected is not None else False
            )
        if enable_direct_dma is None:
            enable_direct_dma = (
                selected.enable_direct_dma if selected is not None else False
            )
        identity_selected = (
            selected is not None
            and selected.kind is IntraDieV2CandidateKind.IDENTITY
        )
        split_selected = (
            selected is not None
            and selected.kind is IntraDieV2CandidateKind.SPLIT_K_FALLBACK
        )
        if (
            self.search_decision.source_projection_id != source.id
            or self.search_decision.source_ir1_id != source.source_ir1_id
            or selected is None
            or (
                identity_selected
                and (
                    split_k_parts,
                    enable_reduce,
                    enable_double_buffer,
                    enable_streaming_reduce,
                    enable_tree_reduce,
                    enable_direct_dma,
                )
                != (1, False, False, False, False, False)
            )
            or (split_selected and (
                selected.split_k_parts != split_k_parts
                or selected.enable_reduce != enable_reduce
                or selected.enable_double_buffer != enable_double_buffer
                or selected.enable_streaming_reduce
                != enable_streaming_reduce
                or selected.enable_tree_reduce != enable_tree_reduce
                or selected.enable_direct_dma != enable_direct_dma
            ))
            or not (identity_selected or split_selected)
        ):
            raise SchemaError(
                "selected v2 candidate does not exactly describe this refinement",
                path=f"{path}.search_decision",
            )
        if (
            self.projection.source_ir1_id,
            self.projection.fusion_plan_ids,
            self.projection.standalone_collective_plan_ids,
            self.projection.source_state_manifest_id,
            self.projection.state_transfers,
        ) != (
            source.source_ir1_id,
            source.fusion_plan_ids,
            source.standalone_collective_plan_ids,
            source.source_state_manifest_id,
            source.state_transfers,
        ):
            raise SchemaError(
                "refinement must preserve projection-level provenance",
                path=f"{path}.projection",
            )
        if tuple(dag.die_id for dag in self.projection.dags) != tuple(
            dag.die_id for dag in source.dags
        ):
            raise SchemaError(
                "refined DAGs must follow source die order",
                path=f"{path}.projection.dags",
            )
        if identity_selected:
            if self.projection != source or self.rewrites:
                raise SchemaError(
                    "identity selection must preserve the exact projection and carry no rewrites",
                    path=path,
                )
            expected_id = stable_artifact_id(
                "split_k_refined_projection", self._semantic_key(),
                schema_version=SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION,
            )
            if self.id != expected_id:
                raise SchemaError(
                    f"unstable artifact id; expected {expected_id!r}",
                    path=f"{path}.id",
                )
            return
        source_dags = {dag.id: dag for dag in source.dags}
        refined_by_die = {dag.die_id: dag for dag in self.projection.dags}
        seen: set[tuple[str, str]] = set()
        for index, rewrite in enumerate(self.rewrites):
            source_dag = source_dags.get(rewrite.source_dag_id)
            if source_dag is None:
                raise SchemaError(
                    "rewrite references a missing source DAG",
                    path=f"{path}.rewrites[{index}].source_dag_id",
                )
            source_task = next(
                (
                    task
                    for task in source_dag.tasks
                    if task.id == rewrite.source_task_id
                ),
                None,
            )
            if (
                source_task is None
                or source_task.kind is not SemanticTaskKind.COMP
                or source_task.compute is None
                or type(source_task.compute.workload) is not GemmWorkload
                or not isinstance(source_task.origin_ref, OrdinaryNodeOrigin)
            ):
                raise SchemaError(
                    "rewrite source must be an ordinary GEMM COMP",
                    path=f"{path}.rewrites[{index}].source_task_id",
                )
            key = (source_dag.id, source_task.id)
            if key in seen:
                raise SchemaError("duplicate source rewrite", path=f"{path}.rewrites[{index}]")
            seen.add(key)
            rewrite.validate(
                split_k_parts=split_k_parts,
                enable_reduce=enable_reduce,
                enable_double_buffer=enable_double_buffer,
                enable_streaming_reduce=enable_streaming_reduce,
                enable_tree_reduce=enable_tree_reduce,
                enable_direct_dma=enable_direct_dma,
                source_input_ids=source_task.read_values,
                path=f"{path}.rewrites[{index}]",
            )
            expected_cg_count = _expected_compute_group_count(
                ir1, source_dag.die_id, enable_double_buffer, split_k_parts
            )
            if not 1 <= rewrite.compute_group_count <= expected_cg_count:
                raise SchemaError(
                    "compute_group_count exceeds compatible source hardware",
                    path=f"{path}.rewrites[{index}].compute_group_count",
                )
            refined_dag = refined_by_die[source_dag.die_id]
            tasks = {task.id: task for task in refined_dag.tasks}
            values = {value.id: value for value in refined_dag.values}
            part_tasks = tuple(tasks.get(task_id) for task_id in rewrite.part_task_ids)
            if any(task is None or task.kind is not SemanticTaskKind.COMP for task in part_tasks):
                raise SchemaError(
                    "refined DAG is missing a split COMP task",
                    path=f"{path}.rewrites[{index}].part_task_ids",
                )
            for part_index, (task, value_id) in enumerate(
                zip(part_tasks, rewrite.partial_value_ids, strict=True)
            ):
                assert task is not None
                if (
                    task.write_values != (value_id,)
                    or value_id not in values
                    or values[value_id].producer_tasks != (task.id,)
                ):
                    raise SchemaError(
                        "split task/partial value closure is incomplete",
                        path=f"{path}.rewrites[{index}].part_task_ids[{part_index}]",
                    )
            expected_input_keys = tuple(
                (part_index, operand_index, producer_task_id)
                for part_index in range(split_k_parts)
                if enable_reduce
                and part_index % rewrite.compute_group_count != 0
                for operand_index, value_id in enumerate(source_task.read_values)
                if not (
                    enable_direct_dma
                    and any(
                        value.id == value_id
                        for value in source_dag.state_staging_values
                    )
                )
                for producer_task_id in tuple(
                    next(
                        (value.producer_tasks for value in (*source_dag.values, *source_dag.state_staging_values)
                         if value.id == value_id),
                        (),
                    )
                )
                if producer_task_id in source_task.deps
            )
            actual_input_keys = tuple(
                (item.part_index, item.operand_index, item.source_task_id)
                for item in rewrite.input_handoffs
            )
            if actual_input_keys != expected_input_keys:
                raise SchemaError(
                    "input handoffs must exactly cover produced odd-CG operands",
                    path=f"{path}.rewrites[{index}].input_handoffs",
                )
            for handoff_index, handoff in enumerate(rewrite.input_handoffs):
                is_state_input = any(
                    value.id == handoff.value_id
                    for value in source_dag.state_staging_values
                )
                expected_destination_value = (
                    split_k_staged_value_id(
                        rewrite.source_task_id, handoff.value_id, handoff.part_index,
                        handoff.operand_index, rewrite.compute_group_count,
                    )
                    if enable_double_buffer or is_state_input else handoff.value_id
                )
                if handoff.destination_value_id != expected_destination_value:
                    raise SchemaError(
                        "input handoff destination does not name the exact stage value",
                        path=f"{path}.rewrites[{index}].input_handoffs[{handoff_index}]",
                    )
                send = tasks.get(handoff.send_task_id)
                recv = tasks.get(handoff.recv_task_id)
                wait = tasks.get(handoff.wait_task_id)
                destination = tasks.get(handoff.destination_task_id)
                local_tasks = (send, recv, wait)
                if (
                    any(task is None for task in local_tasks)
                    or tuple(task.kind for task in local_tasks if task is not None)
                    != (SemanticTaskKind.LOCAL_SEND, SemanticTaskKind.LOCAL_RECV, SemanticTaskKind.LOCAL_WAIT)
                    or any(task is not None and task.flow_id != handoff.flow_id for task in local_tasks)
                    or send.deps != (handoff.send_dependency_task_id,)
                    or (
                        handoff.send_dependency_task_id != handoff.source_task_id
                        and handoff.send_dependency_task_id not in tasks
                    )
                    or recv.deps != (handoff.send_task_id,)
                    or wait.deps != (handoff.recv_task_id,)
                    or destination is None
                    or handoff.wait_task_id not in destination.deps
                    or send.tensor_slice is None
                    or recv.tensor_slice is None
                    or recv.tensor_slice.offset != send.tensor_slice.offset
                    or recv.tensor_slice.shape != send.tensor_slice.shape
                    or send.tensor_slice.value_id != handoff.value_id
                    or recv.tensor_slice.value_id != handoff.destination_value_id
                    or send.bytes != recv.bytes
                    or send.dtype != recv.dtype
                    or (
                        tasks.get(handoff.source_task_id) is not None
                        and tasks[handoff.source_task_id].dma is not None
                        and (
                            handoff.send_task_id not in tasks[handoff.source_task_id].dma.access_task_refs
                            or handoff.destination_task_id in tasks[handoff.source_task_id].dma.access_task_refs
                        )
                    )
                ):
                    raise SchemaError(
                        "split-K input local handoff closure is incomplete",
                        path=f"{path}.rewrites[{index}].input_handoffs[{handoff_index}]",
                    )
            for dma_index, task_id in enumerate(rewrite.direct_dma_task_ids):
                dma = tasks.get(task_id)
                access_refs = dma.dma.access_task_refs if dma is not None and dma.dma is not None else ()
                if (
                    dma is None
                    or dma.kind is not SemanticTaskKind.DMA_IN
                    or len(access_refs) != 1
                    or access_refs[0] not in rewrite.part_task_ids
                    or task_id not in tasks[access_refs[0]].deps
                    or dma.tensor_slice is None
                    or dma.tensor_slice.value_id != dma.dma.local_value_ref
                ):
                    raise SchemaError(
                        "target-core direct DMA closure is incomplete",
                        path=f"{path}.rewrites[{index}].direct_dma_task_ids[{dma_index}]",
                    )
            if enable_reduce and enable_tree_reduce:
                for step_index, (task_id, handoff) in enumerate(
                    zip(
                        rewrite.reduction_task_ids,
                        rewrite.local_handoffs,
                        strict=True,
                    ),
                    start=1,
                ):
                    reduce = tasks.get(task_id)
                    send = tasks.get(handoff.send_task_id)
                    recv = tasks.get(handoff.recv_task_id)
                    wait = tasks.get(handoff.wait_task_id)
                    is_commit = step_index == split_k_parts - 1
                    output_value_id = (
                        rewrite.source_output_value_id if is_commit
                        else rewrite.reduction_accumulator_value_ids[step_index - 1]
                    )
                    if (
                        handoff.reduce_task_id != task_id
                        or send is None
                        or recv is None
                        or wait is None
                        or (send.kind, recv.kind, wait.kind) != (
                            SemanticTaskKind.LOCAL_SEND,
                            SemanticTaskKind.LOCAL_RECV,
                            SemanticTaskKind.LOCAL_WAIT,
                        )
                        or send.deps != (handoff.part_task_id,)
                        or recv.deps != (handoff.send_task_id,)
                        or wait.deps != (handoff.recv_task_id,)
                        or send.tensor_slice is None
                        or recv.tensor_slice != send.tensor_slice
                        or send.tensor_slice.value_id != handoff.partial_value_id
                        or reduce is None
                        or reduce.kind is not SemanticTaskKind.REDUCE
                        or len(reduce.read_values) != 2
                        or handoff.partial_value_id not in reduce.read_values
                        or reduce.write_values != (output_value_id,)
                        or len(reduce.deps) != 2
                        or handoff.wait_task_id not in reduce.deps
                        or reduce.reduction is None
                        or reduce.reduction.input_ranks != (0, 1)
                    ):
                        raise SchemaError(
                            "tree split-K reduce closure is incomplete",
                            path=f"{path}.rewrites[{index}].reduction_task_ids[{step_index - 1}]",
                        )
                    if not is_commit:
                        accumulator = values.get(output_value_id)
                        if (
                            accumulator is None
                            or accumulator.producer_tasks != (task_id,)
                            or not accumulator.consumer_tasks
                        ):
                            raise SchemaError(
                                "tree reduce accumulator lifetime is not closed",
                                path=f"{path}.rewrites[{index}].reduction_accumulator_value_ids",
                            )
            elif enable_reduce:
                handoff_by_part = {
                    item.part_index: item for item in rewrite.local_handoffs
                }
                ready_by_part = tuple(
                    handoff_by_part[part_index].wait_task_id
                    if part_index in handoff_by_part else part_task_id
                    for part_index, part_task_id in enumerate(rewrite.part_task_ids)
                )
                previous_value = rewrite.partial_value_ids[0]
                previous_task_id: str | None = None
                for step_index, task_id in enumerate(
                    rewrite.reduction_task_ids, start=1
                ):
                    reduce = tasks.get(task_id)
                    is_commit = step_index == split_k_parts - 1
                    output_value_id = (
                        rewrite.source_output_value_id if is_commit
                        else rewrite.reduction_accumulator_value_ids[step_index - 1]
                    )
                    if enable_streaming_reduce:
                        expected_deps = tuple(dict.fromkeys(
                            (
                                (ready_by_part[0],)
                                if step_index == 1
                                else (previous_task_id,)
                            )
                            + (ready_by_part[step_index],)
                        ))
                    else:
                        expected_deps = tuple(dict.fromkeys(
                            tuple(ready_by_part)
                            if step_index == 1
                            else (previous_task_id,)
                        ))
                    if (
                        reduce is None
                        or reduce.kind is not SemanticTaskKind.REDUCE
                        or reduce.read_values != (
                            previous_value, rewrite.partial_value_ids[step_index]
                        )
                        or reduce.write_values != (output_value_id,)
                        or reduce.deps != expected_deps
                        or reduce.reduction is None
                        or reduce.reduction.input_ranks != (0, 1)
                    ):
                        raise SchemaError(
                            "streaming split-K reduce chain closure is incomplete",
                            path=f"{path}.rewrites[{index}].reduction_task_ids[{step_index - 1}]",
                        )
                    if not is_commit:
                        accumulator = values.get(output_value_id)
                        next_task_id = rewrite.reduction_task_ids[step_index]
                        if accumulator is None or (
                            accumulator.producer_tasks, accumulator.consumer_tasks
                        ) != ((task_id,), (next_task_id,)):
                            raise SchemaError(
                                "streaming reduce accumulator lifetime is not closed",
                                path=f"{path}.rewrites[{index}].reduction_accumulator_value_ids",
                            )
                    previous_value = output_value_id
                    previous_task_id = task_id
                for handoff_index, handoff in enumerate(rewrite.local_handoffs):
                    send = tasks.get(handoff.send_task_id)
                    recv = tasks.get(handoff.recv_task_id)
                    wait = tasks.get(handoff.wait_task_id)
                    local_tasks = (send, recv, wait)
                    if (
                        any(task is None for task in local_tasks)
                        or tuple(task.kind for task in local_tasks if task is not None)
                        != (
                            SemanticTaskKind.LOCAL_SEND,
                            SemanticTaskKind.LOCAL_RECV,
                            SemanticTaskKind.LOCAL_WAIT,
                        )
                        or any(
                            task is not None and task.flow_id != handoff.flow_id
                            for task in local_tasks
                        )
                        or send.deps != (handoff.part_task_id,)
                        or recv.deps != (handoff.send_task_id,)
                        or wait.deps != (handoff.recv_task_id,)
                        or send.tensor_slice is None
                        or recv.tensor_slice != send.tensor_slice
                        or send.tensor_slice.value_id != handoff.partial_value_id
                        or send.bytes != recv.bytes
                        or send.dtype != recv.dtype
                        or send.region_id != recv.region_id
                        or recv.region_id != wait.region_id
                    ):
                        raise SchemaError(
                            "split-K local handoff closure is incomplete",
                            path=f"{path}.rewrites[{index}].local_handoffs[{handoff_index}]",
                        )
            else:
                commit = tasks.get(rewrite.source_task_id)
                if (
                    commit is None
                    or commit.kind is not SemanticTaskKind.COMP
                    or not set(rewrite.part_task_ids).issubset(commit.deps)
                ):
                    raise SchemaError(
                        "reduce-disabled semantic commit must follow every split part",
                        path=f"{path}.rewrites[{index}].semantic_commit_task_id",
                    )
            if enable_double_buffer:
                for version in rewrite.double_buffer_versions:
                    part = tasks[rewrite.part_task_ids[version.part_index]]
                    if part.read_values != version.staged_value_ids or not set(
                        version.stage_task_ids
                    ).issubset(part.deps):
                        raise SchemaError(
                            "split compute does not consume its double-buffer version",
                            path=f"{path}.rewrites[{index}].double_buffer_versions",
                        )
                    expected_direct = tuple(
                        item.operand_index for item in rewrite.input_handoffs
                        if item.part_index == version.part_index
                    )
                    if version.direct_receive_operand_indices != expected_direct:
                        raise SchemaError(
                            "direct receive operands must exactly match input handoffs",
                            path=f"{path}.rewrites[{index}].double_buffer_versions",
                        )
                    direct = set(version.direct_receive_operand_indices)
                    for operand_index, stage_id in enumerate(version.stage_task_ids):
                        stage = tasks.get(stage_id)
                        staged_id = version.staged_value_ids[operand_index]
                        direct_ok = (
                            operand_index in direct
                            and stage is not None
                            and stage.kind is SemanticTaskKind.LOCAL_WAIT
                            and version.ready_event_ids[operand_index] == stage.id
                            and values.get(staged_id) is not None
                            and values[staged_id].producer_tasks == ()
                        )
                        staged_ok = (
                            operand_index not in direct
                            and stage is not None
                            and stage.kind is SemanticTaskKind.LOCAL_COPY
                            and stage.write_values == (staged_id,)
                            and staged_id in values
                            and stage.sync is not None
                            and stage.sync.completion_event
                            == version.ready_event_ids[operand_index]
                        )
                        if not (direct_ok or staged_ok):
                            raise SchemaError(
                                "double-buffer fill/value/readiness closure is incomplete",
                                path=f"{path}.rewrites[{index}].double_buffer_versions",
                            )
        expected_rewrites = {
            (dag.id, task.id)
            for dag in source.dags
            for task in dag.tasks
            if (
                task.kind is SemanticTaskKind.COMP
                and isinstance(task.origin_ref, OrdinaryNodeOrigin)
                and task.compute is not None
                and type(task.compute.workload) is GemmWorkload
                and len(task.read_values) == 2
                and len(task.write_values) == 1
                and len(task.compute.inputs) == 2
                and len(task.compute.outputs) == 1
                and any(
                    value.id == task.write_values[0]
                    and not value.consumer_tasks
                    for value in dag.values
                )
            )
        }
        if seen != expected_rewrites:
            raise SchemaError(
                "rewrites must exactly cover every eligible ordinary GEMM",
                path=f"{path}.rewrites",
            )
        expected_id = stable_artifact_id(
            "split_k_refined_projection",
            self._semantic_key(),
            schema_version=SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


__all__ = [
    "SPLIT_K_REFINED_PROJECTION_SCHEMA_VERSION",
    "SPLIT_K_TASK_REWRITE_SCHEMA_VERSION",
    "DoubleBufferVersion",
    "SplitKInputHandoff",
    "SplitKLocalHandoff",
    "SplitKRefinedProjection",
    "SplitKTaskRewrite",
    "split_k_compute_event_id",
    "split_k_accumulator_value_id",
    "split_k_local_flow_id",
    "split_k_local_send_task_id",
    "split_k_local_recv_task_id",
    "split_k_local_wait_task_id",
    "split_k_part_task_id",
    "split_k_partial_value_id",
    "split_k_ready_event_id",
    "split_k_reduce_task_id",
    "split_k_reduce_step_task_id",
    "split_k_slot_alias",
    "split_k_stage_task_id",
    "split_k_staged_value_id",
]
