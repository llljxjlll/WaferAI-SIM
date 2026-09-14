"""Typed binding of P3 workload actions to P2 residency and P4 DMA actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .serde import canonical_digest


EXTERNAL_DMA_ACTION_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_bound_action/v1alpha1"
)
EXTERNAL_DMA_ACTION_GRAPH_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_action_graph/v1alpha1"
)
EXTERNAL_DMA_RUNTIME_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_runtime_binding/v1alpha1"
)


class ExternalDmaBoundActionKind(str, Enum):
    DMA_TRANSFER = "dma_transfer"
    LOGICAL_OPERATION = "logical_operation"


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class ExternalDmaBoundAction:
    id: str
    sequence: int
    kind: ExternalDmaBoundActionKind
    source_ref: str
    depends_on: tuple[str, ...]
    state_version_refs: tuple[str, ...]
    residency_refs: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        kind: ExternalDmaBoundActionKind,
        source_ref: str,
        depends_on: tuple[str, ...],
        state_version_refs: tuple[str, ...],
        residency_refs: tuple[str, ...],
    ) -> "ExternalDmaBoundAction":
        key = {
            "sequence": sequence,
            "kind": kind,
            "source_ref": source_ref,
            "depends_on": tuple(sorted(depends_on)),
            "state_version_refs": tuple(sorted(state_version_refs)),
            "residency_refs": tuple(sorted(residency_refs)),
        }
        result = cls(
            id=stable_artifact_id(
                "external_dma_bound_action",
                key,
                schema_version=EXTERNAL_DMA_ACTION_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "id"}

    def validate(self, path: str = "external_dma_bound_action") -> None:
        validate_uint64(self.sequence, f"{path}.sequence")
        if type(self.kind) is not ExternalDmaBoundActionKind:
            raise SchemaError("must be an ExternalDmaBoundActionKind", path=f"{path}.kind")
        validate_nonempty(self.source_ref, f"{path}.source_ref")
        for name in ("depends_on", "state_version_refs", "residency_refs"):
            values = getattr(self, name)
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise SchemaError("must be a sorted unique tuple", path=f"{path}.{name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        expected = stable_artifact_id(
            "external_dma_bound_action", self._key(),
            schema_version=EXTERNAL_DMA_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable action id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class ExternalDmaActionGraph:
    schema_version: str
    producer_pass: str
    id: str
    manifest_digest: str
    case_digest: str
    request_digest: str
    logical_graph_digest: str
    source_memory_plan_digest: str
    blocking_offload_plan_digest: str
    external_dma_program_digest: str
    actions: tuple[ExternalDmaBoundAction, ...]

    @classmethod
    def create(cls, **key: object) -> "ExternalDmaActionGraph":
        result = cls(
            schema_version=EXTERNAL_DMA_ACTION_GRAPH_SCHEMA_VERSION,
            producer_pass="external_dma_action_graph_builder",
            id=stable_artifact_id(
                "external_dma_action_graph", key,
                schema_version=EXTERNAL_DMA_ACTION_GRAPH_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "external_dma_action_graph") -> None:
        if self.schema_version != EXTERNAL_DMA_ACTION_GRAPH_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "external_dma_action_graph_builder":
            raise SchemaError("unexpected producer pass", path=f"{path}.producer_pass")
        for name in (
            "manifest_digest", "case_digest", "request_digest",
            "logical_graph_digest", "source_memory_plan_digest",
            "blocking_offload_plan_digest", "external_dma_program_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        if type(self.actions) is not tuple or not self.actions:
            raise SchemaError("must contain actions", path=f"{path}.actions")
        ids: set[str] = set()
        completed: set[str] = set()
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.sequence != index:
                raise SchemaError("must use contiguous sequence", path=f"{path}.actions[{index}].sequence")
            if action.id in ids:
                raise SchemaError("contains duplicate action id", path=f"{path}.actions")
            if not set(action.depends_on).issubset(completed):
                raise SchemaError("dependency must precede action", path=f"{path}.actions[{index}].depends_on")
            ids.add(action.id)
            completed.add(action.id)
        expected = stable_artifact_id(
            "external_dma_action_graph", self._key(),
            schema_version=EXTERNAL_DMA_ACTION_GRAPH_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable action graph id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class ExternalDmaRuntimeBinding:
    schema_version: str
    id: str
    action_graph_digest: str
    program_relative_path: str
    case_digest: str
    request_digest: str
    logical_graph_digest: str
    source_memory_plan_digest: str
    blocking_offload_plan_digest: str

    @classmethod
    def create(cls, **key: str) -> "ExternalDmaRuntimeBinding":
        result = cls(
            schema_version=EXTERNAL_DMA_RUNTIME_BINDING_SCHEMA_VERSION,
            id=stable_artifact_id(
                "external_dma_runtime_binding", key,
                schema_version=EXTERNAL_DMA_RUNTIME_BINDING_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name not in ("schema_version", "id")}

    def validate(self, path: str = "external_dma_runtime_binding") -> None:
        if self.schema_version != EXTERNAL_DMA_RUNTIME_BINDING_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in (
            "action_graph_digest", "case_digest", "request_digest",
            "logical_graph_digest", "source_memory_plan_digest",
            "blocking_offload_plan_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        validate_nonempty(self.program_relative_path, f"{path}.program_relative_path")
        if self.program_relative_path != "artifacts/external_dma_program.json":
            raise SchemaError("unsupported program path", path=f"{path}.program_relative_path")
        expected = stable_artifact_id(
            "external_dma_runtime_binding", self._key(),
            schema_version=EXTERNAL_DMA_RUNTIME_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runtime binding id", path=f"{path}.id")


__all__ = [
    "EXTERNAL_DMA_ACTION_GRAPH_SCHEMA_VERSION",
    "EXTERNAL_DMA_RUNTIME_BINDING_SCHEMA_VERSION",
    "ExternalDmaActionGraph",
    "ExternalDmaBoundAction",
    "ExternalDmaBoundActionKind",
    "ExternalDmaRuntimeBinding",
]
