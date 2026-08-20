"""Typed fixed-EP4 down-projection WGRAD backward overlay."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty
from .ir0 import ReduceOp
from .ir2 import BufferOwnership
from .lite_moe_dp4 import S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID
from .lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from .persistent_state import (
    HbmBinding,
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateLifetime,
    StateKind,
)


LITE_MOE_DP4_BACKWARD_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_backward/v1alpha1"
)


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _unit_id(kind: str, semantic: dict[str, object]) -> str:
    return stable_artifact_id(
        f"s3_lite_moe_dp4_{kind}",
        semantic,
        schema_version=LITE_MOE_DP4_BACKWARD_SCHEMA_VERSION,
    )


def _stable(instance: object, kind: str, path: str) -> None:
    if getattr(instance, "id") != _unit_id(kind, _semantic(instance)):
        raise SchemaError("unstable DP4 backward unit id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4RemoteGrad:
    id: str
    token_index: int
    expert_index: int
    slot_index: int
    forward_combine_flow_ref: str
    reverse_pair_route_ref: str
    source_die_id: int
    destination_die_id: int
    source_gradient_ref: str
    received_gradient_ref: str
    send_ref: str
    recv_ref: str
    wait_ref: str
    bytes: int
    dtype: DType

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4RemoteGrad":
        result = cls(_unit_id("remote_grad", semantic), **semantic)
        result.validate("lite_moe_dp4_remote_grad")
        return result

    def validate(self, path: str) -> None:
        for name in (
            "forward_combine_flow_ref",
            "reverse_pair_route_ref",
            "source_gradient_ref",
            "received_gradient_ref",
            "send_ref",
            "recv_ref",
            "wait_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.token_index not in (1, 2, 3, 4, 5, 6)
            or self.expert_index != self.token_index // 2
            or self.slot_index != self.token_index % 2
            or self.source_die_id != self.token_index % 4
            or self.destination_die_id != self.expert_index
            or self.source_die_id == self.destination_die_id
            or self.bytes != 32
            or self.dtype is not DType.FP16
            or len({self.send_ref, self.recv_ref, self.wait_ref}) != 3
        ):
            raise SchemaError("invalid exact reverse-combine gradient flow", path=path)
        _stable(self, "remote_grad", path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TrainableDownState:
    expert_index: int
    home_die_id: int
    source_parameter_state_ref: str
    declaration: PersistentStateDecl
    binding: HbmBinding

    def validate(self, path: str) -> None:
        validate_nonempty(
            self.source_parameter_state_ref, f"{path}.source_parameter_state_ref"
        )
        self.declaration.validate(f"{path}.declaration")
        self.binding.validate(f"{path}.binding")
        if (
            self.expert_index not in range(4)
            or self.home_die_id != self.expert_index
            or self.declaration.identity.kind is not StateKind.TRAINABLE_PARAMETER
            or self.declaration.lifetime is not PersistentStateLifetime.PERSISTENT
            or self.declaration.access is not PersistentStateAccess.READ_WRITE
            or self.declaration.tensor_bytes != 1024
            or self.binding.state_ref != self.declaration.id
            or self.binding.die_id != self.home_die_id
            or self.binding.size_bytes != 1024
            or self.source_parameter_state_ref == self.declaration.id
        ):
            raise SchemaError("invalid expert-local trainable down state", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TokenWgrad:
    id: str
    token_index: int
    expert_index: int
    slot_index: int
    home_die_id: int
    down_node_ref: str
    tape_buffer_ref: str
    tape_value_ref: str
    upstream_gradient_ref: str
    trainable_state_ref: str
    root_buffer_ref: str
    contribution_ref: str
    offset_bytes: int
    size_bytes: int
    dtype: DType
    deps: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4TokenWgrad":
        result = cls(_unit_id("token_wgrad", semantic), **semantic)
        result.validate("lite_moe_dp4_token_wgrad")
        return result

    def validate(self, path: str) -> None:
        for name in (
            "down_node_ref",
            "tape_buffer_ref",
            "tape_value_ref",
            "upstream_gradient_ref",
            "trainable_state_ref",
            "root_buffer_ref",
            "contribution_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        remote = self.token_index in (1, 2, 3, 4, 5, 6)
        if (
            self.token_index not in range(8)
            or self.expert_index != self.token_index // 2
            or self.slot_index != self.token_index % 2
            or self.home_die_id != self.expert_index
            or self.offset_bytes != self.slot_index * 2048
            or self.size_bytes != 2048
            or self.dtype is not DType.FP32
            or len(self.deps) != int(remote)
        ):
            raise SchemaError("invalid exact FP32 token WGRAD", path=path)
        _stable(self, "token_wgrad", path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4ExpertReduce:
    id: str
    expert_index: int
    home_die_id: int
    root_buffer_ref: str
    contribution_refs: tuple[str, str]
    input_offsets: tuple[int, int]
    input_dtype: DType
    accumulator_dtype: DType
    output_dtype: DType
    reduce_op: ReduceOp
    input_count: int
    element_count: int
    input_stride_bytes: int
    source_span_bytes: int
    destination_span_bytes: int
    output_alias_ref: str
    output_ownership: BufferOwnership
    deps: tuple[str, str]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4ExpertReduce":
        result = cls(_unit_id("expert_reduce", semantic), **semantic)
        result.validate("lite_moe_dp4_expert_reduce")
        return result

    def validate(self, path: str) -> None:
        if (
            self.expert_index not in range(4)
            or self.home_die_id != self.expert_index
            or len(set(self.contribution_refs)) != 2
            or self.input_offsets != (0, 2048)
            or (self.input_dtype, self.accumulator_dtype, self.output_dtype)
            != (DType.FP32, DType.FP32, DType.FP32)
            or self.reduce_op is not ReduceOp.SUM
            or (
                self.input_count,
                self.element_count,
                self.input_stride_bytes,
                self.source_span_bytes,
                self.destination_span_bytes,
            )
            != (2, 512, 2048, 4096, 2048)
            or self.output_alias_ref != self.contribution_refs[0]
            or self.output_ownership is not BufferOwnership.ALIASED
            or len(set(self.deps)) != 2
        ):
            raise SchemaError("invalid exact two-input FP32 SUM reduction", path=path)
        _stable(self, "expert_reduce", path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4ExpertSgdStore:
    id: str
    expert_index: int
    home_die_id: int
    down_weight_state_ref: str
    down_weight_hbm_binding_ref: str
    reduce_ref: str
    gradient_alias_ref: str
    updated_weight_ref: str
    hbm_store_ref: str
    weight_read_bytes: int
    gradient_read_bytes: int
    state_store_bytes: int
    learning_rate: float
    momentum: float
    deps: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4ExpertSgdStore":
        result = cls(_unit_id("expert_sgd_store", semantic), **semantic)
        result.validate("lite_moe_dp4_expert_sgd_store")
        return result

    def validate(self, path: str) -> None:
        for name in (
            "down_weight_state_ref",
            "down_weight_hbm_binding_ref",
            "reduce_ref",
            "gradient_alias_ref",
            "updated_weight_ref",
            "hbm_store_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.expert_index not in range(4)
            or self.home_die_id != self.expert_index
            or (self.weight_read_bytes, self.gradient_read_bytes, self.state_store_bytes)
            != (1024, 2048, 1024)
            or type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate != 0.001
            or self.momentum != 0.0
            or self.deps != (self.reduce_ref,)
        ):
            raise SchemaError("invalid exact SGD/update/store dependency", path=path)
        _stable(self, "expert_sgd_store", path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Backward:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    source_topology_id: str
    source_oracle_id: str
    train_forward: LiteMoeDp4TrainForward
    remote_gradients: tuple[LiteMoeDp4RemoteGrad, ...]
    trainable_down_states: tuple[LiteMoeDp4TrainableDownState, ...]
    token_wgrads: tuple[LiteMoeDp4TokenWgrad, ...]
    expert_reduces: tuple[LiteMoeDp4ExpertReduce, ...]
    sgd_stores: tuple[LiteMoeDp4ExpertSgdStore, ...]
    updated_weight_refs: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4Backward":
        result = cls(
            LITE_MOE_DP4_BACKWARD_SCHEMA_VERSION,
            "lite_moe_dp4_backward",
            _unit_id("backward", semantic),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_backward") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_BACKWARD_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_backward"
            or self.case_id != S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID
        ):
            raise SchemaError("unsupported DP4 backward schema/producer/case", path=path)
        self.train_forward.validate(f"{path}.train_forward")
        if (
            self.source_topology_id != self.train_forward.source_topology_id
            or self.source_oracle_id != self.train_forward.source_oracle_id
            or tuple(
                map(
                    len,
                    (
                        self.remote_gradients,
                        self.trainable_down_states,
                        self.token_wgrads,
                        self.expert_reduces,
                        self.sgd_stores,
                    ),
                )
            )
            != (6, 4, 8, 4, 4)
        ):
            raise SchemaError("DP4 backward count/provenance quotient changed", path=path)
        for name, items in (
            ("remote_gradients", self.remote_gradients),
            ("trainable_down_states", self.trainable_down_states),
            ("token_wgrads", self.token_wgrads),
            ("expert_reduces", self.expert_reduces),
            ("sgd_stores", self.sgd_stores),
        ):
            for index, item in enumerate(items):
                item.validate(f"{path}.{name}[{index}]")
        if (
            tuple(item.token_index for item in self.remote_gradients)
            != (1, 2, 3, 4, 5, 6)
            or tuple(item.token_index for item in self.token_wgrads) != tuple(range(8))
            or tuple(item.expert_index for item in self.trainable_down_states)
            != tuple(range(4))
            or tuple(item.expert_index for item in self.expert_reduces)
            != tuple(range(4))
            or tuple(item.expert_index for item in self.sgd_stores)
            != tuple(range(4))
            or self.updated_weight_refs
            != tuple(item.updated_weight_ref for item in self.sgd_stores)
            or len(set(self.updated_weight_refs)) != 4
        ):
            raise SchemaError("DP4 backward canonical ordering/output changed", path=path)
        tapes = {item.token_index: item for item in self.train_forward.tape_buffers}
        remote = {item.token_index: item for item in self.remote_gradients}
        states = {item.expert_index: item for item in self.trainable_down_states}
        for wgrad in self.token_wgrads:
            tape = tapes[wgrad.token_index]
            transport = remote.get(wgrad.token_index)
            if (
                wgrad.tape_buffer_ref != tape.id
                or wgrad.tape_value_ref != tape.value_ref
                or wgrad.home_die_id != tape.die_id
                or wgrad.trainable_state_ref != states[wgrad.expert_index].declaration.id
                or wgrad.deps != (() if transport is None else (transport.id,))
            ):
                raise SchemaError("WGRAD does not cross-validate exact TF tape/DTE", path=path)
        for expert, reduce in enumerate(self.expert_reduces):
            contributions = tuple(
                item for item in self.token_wgrads if item.expert_index == expert
            )
            store = self.sgd_stores[expert]
            state = self.trainable_down_states[expert]
            if (
                len(contributions) != 2
                or reduce.root_buffer_ref != contributions[0].root_buffer_ref
                or reduce.contribution_refs
                != tuple(item.contribution_ref for item in contributions)
                or reduce.deps != tuple(item.id for item in contributions)
                or store.reduce_ref != reduce.id
                or store.gradient_alias_ref != reduce.output_alias_ref
                or store.down_weight_state_ref != state.declaration.id
                or store.down_weight_hbm_binding_ref != state.binding.id
            ):
                raise SchemaError("reduce/SGD/state lineage is not exact", path=path)
        _stable(self, "backward", path)


__all__ = [
    "LITE_MOE_DP4_BACKWARD_SCHEMA_VERSION",
    "LiteMoeDp4Backward",
    "LiteMoeDp4ExpertReduce",
    "LiteMoeDp4ExpertSgdStore",
    "LiteMoeDp4RemoteGrad",
    "LiteMoeDp4TokenWgrad",
    "LiteMoeDp4TrainableDownState",
]
