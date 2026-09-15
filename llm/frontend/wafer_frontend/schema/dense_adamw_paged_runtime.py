"""Source-signed two-step Dense AdamW on-demand external DMA runtime contract."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id
from .serde import canonical_digest


_SCHEMA = "wafer_frontend.dense_adamw_paged_runtime/v1alpha1"


@dataclass(frozen=True, slots=True)
class DenseAdamwPagedState:
    state_ref: str
    state_abi_id: str
    source_allocation_ref: str
    external_address: int
    hbm_address: int
    size_bytes: int
    kind: str


@dataclass(frozen=True, slots=True)
class DenseAdamwPagedEvent:
    step_index: int
    linked_record_index: int
    kind: str
    state_ref: str
    external_address: int
    hbm_address: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DenseAdamwPagedRuntime:
    schema_version: str
    producer_pass: str
    id: str
    source_dma_program_relative_path: str
    source_dma_program_id: str
    source_dma_program_digest: str
    request_digest: str
    logical_graph_digest: str
    source_memory_plan_digest: str
    blocking_offload_plan_digest: str
    linked_manifest_ids: tuple[str, str]
    linked_manifest_digests: tuple[str, str]
    hbm_capacity_bytes: int
    workspace_end_bytes: int
    state_bytes: int
    state_spans: tuple[DenseAdamwPagedState, ...]
    events: tuple[DenseAdamwPagedEvent, ...]

    @classmethod
    def create(cls, **semantic: object) -> "DenseAdamwPagedRuntime":
        result = cls(
            _SCHEMA, "dense_adamw_paged_runtime",
            stable_artifact_id(
                "dense_adamw_paged_runtime", semantic, schema_version=_SCHEMA,
            ), **semantic,
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(self, path: str = "dense_adamw_paged_runtime") -> None:
        if (
            self.schema_version != _SCHEMA
            or self.producer_pass != "dense_adamw_paged_runtime"
            or self.source_dma_program_relative_path != "artifacts/external_dma_program.json"
            or len(set(self.linked_manifest_ids)) != 2
            or len(set(self.linked_manifest_digests)) != 2
            or self.hbm_capacity_bytes <= self.workspace_end_bytes
            or self.state_bytes != 32100
            or len(self.state_spans) != 83
            or len({item.state_ref for item in self.state_spans}) != 83
            or len({item.state_abi_id for item in self.state_spans}) != 83
            or len(self.events) != 332
        ):
            raise SchemaError("strict two-step paged AdamW runtime shape drifted", path=path)
        if (
            tuple(item.state_ref for item in self.state_spans)
            != tuple(sorted(item.state_ref for item in self.state_spans))
            or sum(item.size_bytes for item in self.state_spans) != self.state_bytes
        ):
            raise SchemaError("83 true StateABI spans must be canonical/tight", path=path)
        span_by_ref = {item.state_ref: item for item in self.state_spans}
        for index, item in enumerate(self.state_spans):
            if (
                item.size_bytes <= 0
                or item.hbm_address < self.workspace_end_bytes
                or item.hbm_address + item.size_bytes > self.hbm_capacity_bytes
                or item.external_address < 0
                or item.kind not in (
                    "trainable_parameter", "optimizer_master",
                    "optimizer_moment1", "optimizer_moment2", "optimizer_step",
                )
            ):
                raise SchemaError("paged StateABI span exceeds workspace/bounds", path=f"{path}.state_spans[{index}]")
        for step in (0, 1):
            events = tuple(item for item in self.events if item.step_index == step)
            loads = tuple(item for item in events if item.kind == "restore_before_lsu_load")
            stores = tuple(item for item in events if item.kind == "writeback_after_lsu_store")
            if (
                len(events) != 166
                or len(loads) != 83 or len(stores) != 83
                or {item.state_ref for item in loads} != set(span_by_ref)
                or {item.state_ref for item in stores} != set(span_by_ref)
                or tuple(item.linked_record_index for item in events)
                != tuple(sorted(item.linked_record_index for item in events))
                or any(
                    (item.external_address, item.hbm_address, item.size_bytes)
                    != (span_by_ref[item.state_ref].external_address,
                        span_by_ref[item.state_ref].hbm_address,
                        span_by_ref[item.state_ref].size_bytes)
                    for item in events
                )
            ):
                raise SchemaError("every step must exactly gate 83 LSU restore/writeback", path=path)
        if self.events[:166] != tuple(
            DenseAdamwPagedEvent(
                0, item.linked_record_index, item.kind, item.state_ref,
                item.external_address, item.hbm_address, item.size_bytes,
            ) for item in self.events[166:]
        ):
            raise SchemaError("step0/step1 physical pager LSU sequence changed", path=path)
        key = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in ("schema_version", "producer_pass", "id")
        }
        if self.id != stable_artifact_id(
            "dense_adamw_paged_runtime", key, schema_version=_SCHEMA,
        ):
            raise SchemaError("paged runtime stable source identity drifted", path=path)


__all__ = ["DenseAdamwPagedState", "DenseAdamwPagedEvent", "DenseAdamwPagedRuntime"]
