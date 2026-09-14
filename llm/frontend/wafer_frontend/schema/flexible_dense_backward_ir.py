"""Versioned exact lineage carriers for flexible Dense backward training."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .flexible_dense_train import FlexibleDenseTrainActionKind
from .serde import canonical_digest


FLEXIBLE_DENSE_BACKWARD_IR_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_backward_ir/v1alpha1"
)
FLEXIBLE_DENSE_BACKWARD_PROJECTION_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_backward_projection/v1alpha1"
)
FLEXIBLE_DENSE_BACKWARD_SCHEDULE_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_backward_schedule/v1alpha1"
)
FLEXIBLE_DENSE_BACKWARD_GLOBAL_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_backward_global/v1alpha1"
)


def _stable(kind: str, schema_version: str, semantic: object) -> str:
    return stable_artifact_id(kind, semantic, schema_version=schema_version)


@dataclass(frozen=True, slots=True)
class FlexibleDenseBackwardActionLineage:
    """One exact non-forward action and its forward/tape/state witnesses."""

    action_id: str
    rank: int
    index: int
    kind: FlexibleDenseTrainActionKind
    state_ref: str | None
    op_ref: str | None
    depends_on: tuple[str, ...]
    forward_consumer_refs: tuple[str, ...]
    tape_origin_ref: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.action_id, f"{path}.action_id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.index, f"{path}.index")
        if type(self.kind) is not FlexibleDenseTrainActionKind:
            raise SchemaError("invalid action kind", path=f"{path}.kind")
        for field in ("state_ref", "op_ref", "tape_origin_ref"):
            value = getattr(self, field)
            if value is not None:
                validate_nonempty(value, f"{path}.{field}")
        for field in ("depends_on", "forward_consumer_refs"):
            values = getattr(self, field)
            if type(values) is not tuple or len(set(values)) != len(values):
                raise SchemaError("must be a unique tuple", path=f"{path}.{field}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{field}[{index}]")
        needs_tape = self.kind is FlexibleDenseTrainActionKind.BACKWARD
        if needs_tape != (self.tape_origin_ref is not None):
            raise SchemaError("tape origin is exact for BACKWARD only", path=path)
        state_lineage = self.kind in (
            FlexibleDenseTrainActionKind.PARAMETER_LOAD,
            FlexibleDenseTrainActionKind.WEIGHT_GRADIENT,
            FlexibleDenseTrainActionKind.GRADIENT_SYNC,
            FlexibleDenseTrainActionKind.SGD_UPDATE,
            FlexibleDenseTrainActionKind.PARAMETER_STORE,
        )
        if state_lineage != bool(self.forward_consumer_refs):
            raise SchemaError(
                "parameter actions require exact forward consumers",
                path=f"{path}.forward_consumer_refs",
            )


@dataclass(frozen=True, slots=True)
class FlexibleDenseBackwardIR:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_id: str
    source_plan_digest: str
    mesh_digest: str
    rank_count: int
    actions: tuple[FlexibleDenseBackwardActionLineage, ...]

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleDenseBackwardIR":
        result = cls(
            FLEXIBLE_DENSE_BACKWARD_IR_SCHEMA_VERSION,
            "flexible_dense_backward_ir",
            _stable(
                "flexible_dense_backward_ir",
                FLEXIBLE_DENSE_BACKWARD_IR_SCHEMA_VERSION,
                semantic,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(self, path: str = "flexible_dense_backward_ir") -> None:
        if (
            self.schema_version != FLEXIBLE_DENSE_BACKWARD_IR_SCHEMA_VERSION
            or self.producer_pass != "flexible_dense_backward_ir"
        ):
            raise SchemaError("unsupported backward IR", path=path)
        for field in ("source_plan_id", "source_plan_digest", "mesh_digest"):
            validate_nonempty(getattr(self, field), f"{path}.{field}")
        validate_uint64(self.rank_count, f"{path}.rank_count")
        if self.rank_count == 0 or not self.actions:
            raise SchemaError("requires ranks and actions", path=path)
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
        ids = tuple(action.action_id for action in self.actions)
        if len(set(ids)) != len(ids):
            raise SchemaError("action ids must be unique", path=f"{path}.actions")
        by_rank: dict[int, list[FlexibleDenseBackwardActionLineage]] = {
            rank: [] for rank in range(self.rank_count)
        }
        for action in self.actions:
            if action.rank not in by_rank:
                raise SchemaError("action rank escapes Mesh", path=f"{path}.actions")
            by_rank[action.rank].append(action)
        for rank, actions in by_rank.items():
            if tuple(item.index for item in actions) != tuple(
                range(len(actions))
            ):
                raise SchemaError(
                    "per-rank action indices must be contiguous",
                    path=f"{path}.actions.rank{rank}",
                )
        semantic = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }
        if self.id != _stable(
            "flexible_dense_backward_ir",
            FLEXIBLE_DENSE_BACKWARD_IR_SCHEMA_VERSION,
            semantic,
        ):
            raise SchemaError("unstable backward IR id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleDenseBackwardProjection:
    schema_version: str
    producer_pass: str
    id: str
    source_ir_id: str
    action_ids: tuple[str, ...]
    state_owner_pairs: tuple[tuple[str, int], ...]

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleDenseBackwardProjection":
        result = cls(
            FLEXIBLE_DENSE_BACKWARD_PROJECTION_SCHEMA_VERSION,
            "flexible_dense_backward_projection",
            _stable(
                "flexible_dense_backward_projection",
                FLEXIBLE_DENSE_BACKWARD_PROJECTION_SCHEMA_VERSION,
                semantic,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(self, path: str = "flexible_dense_backward_projection") -> None:
        if (
            self.schema_version
            != FLEXIBLE_DENSE_BACKWARD_PROJECTION_SCHEMA_VERSION
            or self.producer_pass != "flexible_dense_backward_projection"
        ):
            raise SchemaError("unsupported backward projection", path=path)
        validate_nonempty(self.source_ir_id, f"{path}.source_ir_id")
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise SchemaError("action ids must be non-empty and unique", path=path)
        for index, action_id in enumerate(self.action_ids):
            validate_nonempty(action_id, f"{path}.action_ids[{index}]")
        if (
            not self.state_owner_pairs
            or self.state_owner_pairs != tuple(sorted(set(self.state_owner_pairs)))
        ):
            raise SchemaError("state owners must be canonical", path=path)
        semantic = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }
        if self.id != _stable(
            "flexible_dense_backward_projection",
            FLEXIBLE_DENSE_BACKWARD_PROJECTION_SCHEMA_VERSION,
            semantic,
        ):
            raise SchemaError("unstable projection id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleDenseBackwardSchedule:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    rank_action_ids: tuple[tuple[str, ...], ...]

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleDenseBackwardSchedule":
        result = cls(
            FLEXIBLE_DENSE_BACKWARD_SCHEDULE_SCHEMA_VERSION,
            "flexible_dense_backward_schedule",
            _stable(
                "flexible_dense_backward_schedule",
                FLEXIBLE_DENSE_BACKWARD_SCHEDULE_SCHEMA_VERSION,
                semantic,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(self, path: str = "flexible_dense_backward_schedule") -> None:
        if (
            self.schema_version != FLEXIBLE_DENSE_BACKWARD_SCHEDULE_SCHEMA_VERSION
            or self.producer_pass != "flexible_dense_backward_schedule"
        ):
            raise SchemaError("unsupported backward schedule", path=path)
        validate_nonempty(self.source_projection_id, f"{path}.source_projection_id")
        if not self.rank_action_ids or any(not row for row in self.rank_action_ids):
            raise SchemaError("every rank requires a non-empty stream", path=path)
        flat = tuple(item for row in self.rank_action_ids for item in row)
        if len(set(flat)) != len(flat):
            raise SchemaError("scheduled action ids must be unique", path=path)
        semantic = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }
        if self.id != _stable(
            "flexible_dense_backward_schedule",
            FLEXIBLE_DENSE_BACKWARD_SCHEDULE_SCHEMA_VERSION,
            semantic,
        ):
            raise SchemaError("unstable schedule id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleDenseBackwardGlobalDAG:
    schema_version: str
    producer_pass: str
    id: str
    source_schedule_id: str
    action_ids: tuple[str, ...]
    dependency_edges: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleDenseBackwardGlobalDAG":
        result = cls(
            FLEXIBLE_DENSE_BACKWARD_GLOBAL_SCHEMA_VERSION,
            "flexible_dense_backward_global",
            _stable(
                "flexible_dense_backward_global",
                FLEXIBLE_DENSE_BACKWARD_GLOBAL_SCHEMA_VERSION,
                semantic,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(self, path: str = "flexible_dense_backward_global") -> None:
        if (
            self.schema_version != FLEXIBLE_DENSE_BACKWARD_GLOBAL_SCHEMA_VERSION
            or self.producer_pass != "flexible_dense_backward_global"
        ):
            raise SchemaError("unsupported backward global DAG", path=path)
        validate_nonempty(self.source_schedule_id, f"{path}.source_schedule_id")
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise SchemaError("global actions must be non-empty and unique", path=path)
        action_set = set(self.action_ids)
        if self.dependency_edges != tuple(sorted(set(self.dependency_edges))):
            raise SchemaError("dependency edges must be canonical", path=path)
        for source, destination in self.dependency_edges:
            if source not in action_set or destination not in action_set:
                raise SchemaError("dependency escapes action set", path=path)
        semantic = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }
        if self.id != _stable(
            "flexible_dense_backward_global",
            FLEXIBLE_DENSE_BACKWARD_GLOBAL_SCHEMA_VERSION,
            semantic,
        ):
            raise SchemaError("unstable global DAG id", path=f"{path}.id")


__all__ = [
    "FLEXIBLE_DENSE_BACKWARD_GLOBAL_SCHEMA_VERSION",
    "FLEXIBLE_DENSE_BACKWARD_IR_SCHEMA_VERSION",
    "FLEXIBLE_DENSE_BACKWARD_PROJECTION_SCHEMA_VERSION",
    "FLEXIBLE_DENSE_BACKWARD_SCHEDULE_SCHEMA_VERSION",
    "FlexibleDenseBackwardActionLineage",
    "FlexibleDenseBackwardGlobalDAG",
    "FlexibleDenseBackwardIR",
    "FlexibleDenseBackwardProjection",
    "FlexibleDenseBackwardSchedule",
]
