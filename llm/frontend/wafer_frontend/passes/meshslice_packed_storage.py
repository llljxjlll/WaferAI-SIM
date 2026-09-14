"""Exact packed storage aliases for rectangular MeshSlice operands.

The frozen 2x2 ABI is intentionally excluded. For every other communicating
rectangle this pass turns a value-slot with peer subviews into one physical
root, one compute full-view, and one alias per distinct dense DTE chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.swizzle import SwizzleActionKind, SwizzleAlgorithm
from ..schema.swizzle_abi import SwizzleCoreAddressABI, SwizzleValueAddressBinding
from ..schema.swizzle_ir2 import SwizzleIr2Projection
from ..schema.swizzle_operand_abi import SwizzleOperandABI, SwizzleTaskOperandView
from ..schema.swizzle_plan import SwizzleFusionPlan


MESHSLICE_PACKED_STORAGE_SCHEMA_VERSION = (
    "wafer_frontend.meshslice_packed_storage/v1alpha1"
)


class MeshSlicePackedAliasKind(str, Enum):
    FULL_VIEW = "full_view"
    CHUNK = "chunk"


@dataclass(frozen=True, slots=True)
class MeshSlicePackedAlias:
    binding_id: str
    kind: MeshSlicePackedAliasKind
    value_ref: str
    slot: int
    shape: tuple[int, ...]
    layout: str
    dtype: DType
    byte_offset: int
    byte_extent: int
    task_uses: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class MeshSlicePackedRoot:
    binding_id: str
    storage_id: str
    root_value_id: str
    value_ref: str
    slot: int
    address_binding: SwizzleValueAddressBinding
    dtype: DType
    aliases: tuple[MeshSlicePackedAlias, ...]

    @property
    def full_view(self) -> MeshSlicePackedAlias:
        return next(
            alias
            for alias in self.aliases
            if alias.kind is MeshSlicePackedAliasKind.FULL_VIEW
        )

    @property
    def chunks(self) -> tuple[MeshSlicePackedAlias, ...]:
        return tuple(
            alias
            for alias in self.aliases
            if alias.kind is MeshSlicePackedAliasKind.CHUNK
        )


@dataclass(frozen=True, slots=True)
class MeshSlicePackedStoragePlan:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_ref: str
    source_projection_ref: str
    source_core_abi_ref: str
    source_operand_abi_ref: str
    rows: int
    columns: int
    roots: tuple[MeshSlicePackedRoot, ...]

    def alias_for(
        self, task_ref: str, ordinal: int
    ) -> MeshSlicePackedAlias | None:
        matches = tuple(
            alias
            for root in self.roots
            for alias in root.aliases
            if (task_ref, ordinal) in alias.task_uses
        )
        if len(matches) > 1:
            raise SchemaError(
                "task operand maps to multiple packed aliases",
                path="meshslice_packed_storage.roots",
            )
        return matches[0] if matches else None

    def root_for(
        self, value_ref: str, slot: int
    ) -> MeshSlicePackedRoot | None:
        return next(
            (
                root
                for root in self.roots
                if (root.value_ref, root.slot) == (value_ref, slot)
            ),
            None,
        )


def _dtype_bytes(dtype: DType) -> int:
    if dtype is DType.FP16:
        return 2
    if dtype in (DType.FP32, DType.INT32):
        return 4
    raise SchemaError("unsupported packed dtype", path="operand_abi.operands")


def _artifact_id(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"meshslice_packed_{kind}",
        semantic,
        schema_version=MESHSLICE_PACKED_STORAGE_SCHEMA_VERSION,
    )


def build_meshslice_packed_storage(
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    core_abi: SwizzleCoreAddressABI,
    operand_abi: SwizzleOperandABI,
) -> MeshSlicePackedStoragePlan | None:
    """Derive the exact alias graph, or ``None`` for local/frozen/non-MeshSlice."""

    rows = len(plan.candidate.topology_witness.row_orders)
    columns = len(plan.candidate.topology_witness.column_orders)
    if plan.algorithm is not SwizzleAlgorithm.MESHSLICE_2D_OS:
        return None
    if (rows, columns) in ((1, 1), (2, 2)):
        return None
    tasks = {
        task.id: task for dag in projection.rank_dags for task in dag.tasks
    }
    addresses = {
        (item.value_ref, item.slot): item for item in core_abi.value_bindings
    }
    views_by_key: dict[tuple[str, int], list[SwizzleTaskOperandView]] = {}
    for view in operand_abi.operands:
        views_by_key.setdefault((view.value_ref, view.slot), []).append(view)

    roots = []
    for key, views in sorted(views_by_key.items()):
        dte_views = tuple(
            view
            for view in views
            if tasks[view.task_ref].kind
            in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
        )
        if not dte_views:
            continue
        address = addresses[key]
        dtype_set = {view.dtype for view in views}
        layout_set = {view.layout for view in views}
        if len(dtype_set) != 1 or len(layout_set) != 1:
            raise SchemaError(
                "packed aliases require one dtype/layout",
                path="operand_abi.operands",
            )
        dtype = next(iter(dtype_set))
        full_candidates = tuple(
            view
            for view in views
            if view.byte_offset == 0 and view.byte_extent == address.size_bytes
        )
        if not full_candidates:
            raise SchemaError(
                "packed value lacks an exact full compute view",
                path="operand_abi.operands",
            )
        full = max(full_candidates, key=lambda item: (len(item.shape), item.shape))
        if prod(full.shape) * _dtype_bytes(dtype) != address.size_bytes:
            raise SchemaError(
                "full view does not cover physical root",
                path="operand_abi.operands",
            )

        spans: dict[
            tuple[int, int, tuple[int, ...]], list[tuple[str, int]]
        ] = {}
        for view in dte_views:
            if view.byte_offset + view.byte_extent > address.size_bytes:
                raise SchemaError(
                    "DTE chunk exceeds packed root", path="operand_abi.operands"
                )
            spans.setdefault(
                (view.byte_offset, view.byte_extent, view.shape), []
            ).append((view.task_ref, view.ordinal))
        ordered_spans = sorted(spans)
        cursor = 0
        for offset, extent, _shape in ordered_spans:
            if offset != cursor:
                raise SchemaError(
                    "DTE chunks must exactly partition packed root",
                    path="operand_abi.operands",
                )
            cursor += extent
        if cursor != address.size_bytes:
            raise SchemaError(
                "DTE chunks do not cover packed root",
                path="operand_abi.operands",
            )

        root_semantic = {"core_abi": core_abi.id, "key": key}
        aliases = [
            MeshSlicePackedAlias(
                _artifact_id("full_binding", root_semantic),
                MeshSlicePackedAliasKind.FULL_VIEW,
                key[0], key[1], full.shape, full.layout, dtype, 0,
                address.size_bytes,
                tuple(sorted(
                    (view.task_ref, view.ordinal)
                    for view in full_candidates
                    if tasks[view.task_ref].kind is SwizzleActionKind.COMP
                )),
            )
        ]
        for offset, extent, shape in ordered_spans:
            aliases.append(MeshSlicePackedAlias(
                _artifact_id("chunk_binding", {
                    **root_semantic, "offset": offset,
                    "extent": extent, "shape": shape,
                }),
                MeshSlicePackedAliasKind.CHUNK,
                key[0], key[1], shape, full.layout, dtype, offset, extent,
                tuple(sorted(spans[(offset, extent, shape)])),
            ))
        roots.append(MeshSlicePackedRoot(
            _artifact_id("root_binding", root_semantic),
            _artifact_id("storage", root_semantic),
            _artifact_id("root_value", root_semantic),
            key[0], key[1], address, dtype, tuple(aliases),
        ))

    if not roots:
        raise SchemaError(
            "communicating MeshSlice has no packed operands",
            path="operand_abi.operands",
        )
    semantic = {
        "source_plan_ref": plan.id,
        "source_projection_ref": projection.id,
        "source_core_abi_ref": core_abi.id,
        "source_operand_abi_ref": operand_abi.id,
        "rows": rows,
        "columns": columns,
        "roots": tuple(roots),
    }
    return MeshSlicePackedStoragePlan(
        MESHSLICE_PACKED_STORAGE_SCHEMA_VERSION,
        "meshslice_packed_storage",
        _artifact_id("plan", semantic),
        **semantic,
    )


__all__ = [
    "MESHSLICE_PACKED_STORAGE_SCHEMA_VERSION",
    "MeshSlicePackedAlias",
    "MeshSlicePackedAliasKind",
    "MeshSlicePackedRoot",
    "MeshSlicePackedStoragePlan",
    "build_meshslice_packed_storage",
]
