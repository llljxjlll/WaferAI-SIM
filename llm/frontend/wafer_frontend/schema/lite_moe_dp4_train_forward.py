"""Typed training-forward tape overlay for fixed EP4 S3-Lite MoE."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty
from .ir2 import BufferOwnership
from .lite_moe_dp4 import S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID
from .lite_moe_dp4_execution import LiteMoeDp4ExecutionCase


LITE_MOE_DP4_TRAIN_FORWARD_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_train_forward/v1alpha1"
)
LITE_MOE_DP4_TAPE_BUFFER_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_tape_buffer/v1alpha1"
)
LITE_MOE_DP4_TAPE_COPY_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_tape_copy/v1alpha1"
)


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _stable(instance: object, prefix: str, version: str, path: str) -> None:
    expected = stable_artifact_id(prefix, _semantic(instance), schema_version=version)
    if getattr(instance, "id") != expected:
        raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TapeBuffer:
    schema_version: str
    id: str
    token_index: int
    expert_index: int
    slot_index: int
    die_id: int
    core_ref: str
    value_ref: str
    address: int
    size_bytes: int
    alignment_bytes: int
    ownership: BufferOwnership
    terminal: bool
    alias_of: str | None
    ordinal: int

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4TapeBuffer":
        result = cls(
            LITE_MOE_DP4_TAPE_BUFFER_SCHEMA_VERSION,
            stable_artifact_id(
                "s3_lite_moe_dp4_tape_buffer",
                semantic,
                schema_version=LITE_MOE_DP4_TAPE_BUFFER_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_tape_buffer") -> None:
        if self.schema_version != LITE_MOE_DP4_TAPE_BUFFER_SCHEMA_VERSION:
            raise SchemaError("unsupported tape buffer schema", path=path)
        validate_nonempty(self.core_ref, f"{path}.core_ref")
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        if (
            self.token_index not in range(8)
            or self.expert_index != self.token_index // 2
            or self.slot_index != self.token_index % 2
            or self.die_id != self.expert_index
            or self.address % 64
            or self.size_bytes != 64
            or self.alignment_bytes != 64
            or self.ownership is not BufferOwnership.OWNED
            or self.terminal is not True
            or self.alias_of is not None
            or self.ordinal < 20
        ):
            raise SchemaError("requires exact independent OWNED terminal 64B tape", path=path)
        _stable(
            self,
            "s3_lite_moe_dp4_tape_buffer",
            LITE_MOE_DP4_TAPE_BUFFER_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TapeCopy:
    schema_version: str
    id: str
    token_index: int
    expert_index: int
    slot_index: int
    die_id: int
    core_ref: str
    source_value_ref: str
    source_task_ref: str
    source_action_ref: str
    source_buffer_ref: str
    down_task_ref: str
    down_action_ref: str
    destination_buffer_ref: str
    bytes: int
    dtype: DType
    deps: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4TapeCopy":
        result = cls(
            LITE_MOE_DP4_TAPE_COPY_SCHEMA_VERSION,
            stable_artifact_id(
                "s3_lite_moe_dp4_tape_copy",
                semantic,
                schema_version=LITE_MOE_DP4_TAPE_COPY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_tape_copy") -> None:
        if self.schema_version != LITE_MOE_DP4_TAPE_COPY_SCHEMA_VERSION:
            raise SchemaError("unsupported tape-copy schema", path=path)
        for name in (
            "core_ref",
            "source_value_ref",
            "source_task_ref",
            "source_action_ref",
            "source_buffer_ref",
            "down_task_ref",
            "down_action_ref",
            "destination_buffer_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.token_index not in range(8)
            or self.expert_index != self.token_index // 2
            or self.slot_index != self.token_index % 2
            or self.die_id != self.expert_index
            or self.bytes != 64
            or self.dtype is not DType.FP16
            or self.deps != (self.source_action_ref,)
            or self.source_task_ref == self.down_task_ref
            or self.source_buffer_ref == self.destination_buffer_ref
        ):
            raise SchemaError("tape copy does not preserve exact producer/fork semantics", path=path)
        _stable(
            self,
            "s3_lite_moe_dp4_tape_copy",
            LITE_MOE_DP4_TAPE_COPY_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TrainForward:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    source_topology_id: str
    source_oracle_id: str
    forward: LiteMoeDp4ExecutionCase
    tape_buffers: tuple[LiteMoeDp4TapeBuffer, ...]
    tape_copies: tuple[LiteMoeDp4TapeCopy, ...]
    total_tape_bytes: int

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4TrainForward":
        result = cls(
            LITE_MOE_DP4_TRAIN_FORWARD_SCHEMA_VERSION,
            "lite_moe_dp4_train_forward",
            stable_artifact_id(
                "s3_lite_moe_dp4_train_forward",
                semantic,
                schema_version=LITE_MOE_DP4_TRAIN_FORWARD_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_train_forward") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_TRAIN_FORWARD_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_train_forward"
            or self.case_id != S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID
        ):
            raise SchemaError("unsupported training-forward schema/producer/case", path=path)
        self.forward.validate(f"{path}.forward")
        topology = self.forward.adapter.topology
        oracle = self.forward.adapter.oracle
        if (
            self.source_topology_id != topology.id
            or self.source_oracle_id != oracle.id
            or len(self.tape_buffers) != 8
            or len(self.tape_copies) != 8
            or self.total_tape_bytes != 512
        ):
            raise SchemaError("training-forward source/count provenance changed", path=path)
        for index, item in enumerate(self.tape_buffers):
            item.validate(f"{path}.tape_buffers[{index}]")
        for index, item in enumerate(self.tape_copies):
            item.validate(f"{path}.tape_copies[{index}]")
        if (
            tuple(item.token_index for item in self.tape_buffers) != tuple(range(8))
            or tuple(item.token_index for item in self.tape_copies) != tuple(range(8))
            or len({item.id for item in self.tape_buffers}) != 8
            or len({item.id for item in self.tape_copies}) != 8
            or {item.destination_buffer_ref for item in self.tape_copies}
            != {item.id for item in self.tape_buffers}
        ):
            raise SchemaError("tape catalog/copy coverage is not exact T8", path=path)
        _stable(
            self,
            "s3_lite_moe_dp4_train_forward",
            LITE_MOE_DP4_TRAIN_FORWARD_SCHEMA_VERSION,
            path,
        )


__all__ = [
    "LITE_MOE_DP4_TAPE_BUFFER_SCHEMA_VERSION",
    "LITE_MOE_DP4_TAPE_COPY_SCHEMA_VERSION",
    "LITE_MOE_DP4_TRAIN_FORWARD_SCHEMA_VERSION",
    "LiteMoeDp4TapeBuffer",
    "LiteMoeDp4TapeCopy",
    "LiteMoeDp4TrainForward",
]
