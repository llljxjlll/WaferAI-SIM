"""Typed S3-Lite MoE down-projection backward overlay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .lite_moe import LiteMoeStaticTrace
from .persistent_state import HbmBinding, PersistentStateDecl, StateKind


LITE_MOE_BACKWARD_CONTRACT_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_backward_contract/v1alpha1"
)
LITE_MOE_BACKWARD_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_backward_oracle/v1alpha1"
)
LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_backward_overlay/v1alpha1"
)
S3_LITE_MOE_BACKWARD_CASE_ID = "case.s3_lite.static_moe_down_wgrad"


class LiteMoeBackwardCoverage(str, Enum):
    DOWN_PROJECTION_WGRAD_ONLY = "down_projection_wgrad_only"
    FULL_EXPERT = "full_expert"


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardContract:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    trace: LiteMoeStaticTrace
    coverage: LiteMoeBackwardCoverage
    die_count: int
    ep_degree: int
    expert_count: int
    top_k: int
    token_count: int
    expert_histogram: tuple[int, int, int, int]
    learning_rate: float
    momentum: float

    @classmethod
    def create(cls, *, trace: LiteMoeStaticTrace) -> "LiteMoeBackwardContract":
        semantic = {
            "case_id": S3_LITE_MOE_BACKWARD_CASE_ID,
            "trace": trace,
            "coverage": LiteMoeBackwardCoverage.DOWN_PROJECTION_WGRAD_ONLY,
            "die_count": 2,
            "ep_degree": 2,
            "expert_count": 4,
            "top_k": 1,
            "token_count": 8,
            "expert_histogram": (2, 2, 2, 2),
            "learning_rate": 0.001,
            "momentum": 0.0,
        }
        result = cls(
            LITE_MOE_BACKWARD_CONTRACT_SCHEMA_VERSION,
            "lite_moe_backward_contract",
            stable_artifact_id(
                "s3_lite_moe_backward_contract",
                semantic,
                schema_version=LITE_MOE_BACKWARD_CONTRACT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_moe_backward_contract") -> None:
        if (
            self.schema_version != LITE_MOE_BACKWARD_CONTRACT_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_backward_contract"
        ):
            raise SchemaError("unsupported contract schema/producer", path=path)
        if self.case_id != S3_LITE_MOE_BACKWARD_CASE_ID:
            raise SchemaError("wrong backward case id", path=f"{path}.case_id")
        if type(self.trace) is not LiteMoeStaticTrace:
            raise SchemaError("must be a LiteMoeStaticTrace", path=f"{path}.trace")
        self.trace.validate(f"{path}.trace")
        if self.coverage is not LiteMoeBackwardCoverage.DOWN_PROJECTION_WGRAD_ONLY:
            raise SchemaError("FULL_EXPERT backward is unsupported", path=f"{path}.coverage")
        exact = {
            "die_count": 2,
            "ep_degree": 2,
            "expert_count": 4,
            "top_k": 1,
            "token_count": 8,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if value != expected:
                raise SchemaError(f"must equal {expected}", path=f"{path}.{name}")
        if (
            self.expert_histogram != (2, 2, 2, 2)
            or self.trace.token_count != self.token_count
            or self.trace.expert_histogram != self.expert_histogram
        ):
            raise SchemaError("requires the exact balanced T8 trace", path=f"{path}.trace")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or type(self.momentum) is not float
            or self.momentum != 0.0
        ):
            raise SchemaError("requires finite positive SGD lr and zero momentum", path=path)
        expected_id = stable_artifact_id(
            "s3_lite_moe_backward_contract",
            self._semantic(),
            schema_version=LITE_MOE_BACKWARD_CONTRACT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable contract id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardOracle:
    schema_version: str
    producer_pass: str
    id: str
    contract_id: str
    remote_grad_count: int
    remote_grad_bytes_each: int
    remote_grad_bytes_total: int
    token_wgrad_count: int
    token_wgrad_bytes_each: int
    token_wgrad_bytes_total: int
    expert_reduce_count: int
    reduce_input_bytes_each: int
    reduce_output_bytes_each: int
    sgd_store_count: int
    weight_read_bytes_each: int
    gradient_read_bytes_each: int
    state_store_bytes_each: int

    @classmethod
    def create(cls, *, contract: LiteMoeBackwardContract) -> "LiteMoeBackwardOracle":
        contract.validate("contract")
        semantic = {
            "contract_id": contract.id,
            "remote_grad_count": 4,
            "remote_grad_bytes_each": 32,
            "remote_grad_bytes_total": 128,
            "token_wgrad_count": 8,
            "token_wgrad_bytes_each": 2048,
            "token_wgrad_bytes_total": 16384,
            "expert_reduce_count": 4,
            "reduce_input_bytes_each": 4096,
            "reduce_output_bytes_each": 2048,
            "sgd_store_count": 4,
            "weight_read_bytes_each": 1024,
            "gradient_read_bytes_each": 2048,
            "state_store_bytes_each": 1024,
        }
        result = cls(
            LITE_MOE_BACKWARD_ORACLE_SCHEMA_VERSION,
            "lite_moe_backward_oracle",
            stable_artifact_id(
                "s3_lite_moe_backward_oracle",
                semantic,
                schema_version=LITE_MOE_BACKWARD_ORACLE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(contract)
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate_against(self, contract: LiteMoeBackwardContract, path: str = "lite_moe_backward_oracle") -> None:
        contract.validate("contract")
        if (
            self.schema_version != LITE_MOE_BACKWARD_ORACLE_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_backward_oracle"
            or self.contract_id != contract.id
        ):
            raise SchemaError("oracle identity/provenance mismatch", path=path)
        expected = {
            "remote_grad_count": 4,
            "remote_grad_bytes_each": 32,
            "remote_grad_bytes_total": 4 * 32,
            "token_wgrad_count": 8,
            "token_wgrad_bytes_each": 2048,
            "token_wgrad_bytes_total": 8 * 2048,
            "expert_reduce_count": 4,
            "reduce_input_bytes_each": 4096,
            "reduce_output_bytes_each": 2048,
            "sgd_store_count": 4,
            "weight_read_bytes_each": 1024,
            "gradient_read_bytes_each": 2048,
            "state_store_bytes_each": 1024,
        }
        for name, value in expected.items():
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) != value:
                raise SchemaError("oracle numeric mismatch", path=f"{path}.{name}")
        expected_id = stable_artifact_id(
            "s3_lite_moe_backward_oracle",
            self._semantic(),
            schema_version=LITE_MOE_BACKWARD_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable oracle id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeRemoteGradDte:
    id: str
    token_index: int
    expert_index: int
    forward_flow_ref: str
    pair_route_ref: str
    source_die_id: int
    destination_die_id: int
    bytes: int

    def validate(self, path: str) -> None:
        for name in ("token_index", "expert_index", "source_die_id", "destination_die_id", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for name in ("forward_flow_ref", "pair_route_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.token_index not in (1, 3, 4, 6) or self.expert_index != self.token_index // 2:
            raise SchemaError("must be one of four remote balanced assignments", path=path)
        if self.destination_die_id != self.expert_index // 2 or self.source_die_id == self.destination_die_id or self.bytes != 32:
            raise SchemaError("invalid remote grad DTE geometry", path=path)
        expected = stable_artifact_id("s3_lite_moe_remote_grad_dte", self._semantic(), schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable remote grad id", path=f"{path}.id")

    def _semantic(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "id"}


@dataclass(frozen=True, slots=True)
class LiteMoeTrainableDownState:
    expert_index: int
    home_die_id: int
    source_parameter_state_ref: str
    declaration: PersistentStateDecl
    binding: HbmBinding

    def validate(self, path: str) -> None:
        if (
            self.expert_index not in range(4)
            or self.home_die_id != self.expert_index // 2
        ):
            raise SchemaError("invalid trainable expert/home", path=path)
        validate_nonempty(
            self.source_parameter_state_ref,
            f"{path}.source_parameter_state_ref",
        )
        self.declaration.validate(f"{path}.declaration")
        self.binding.validate(f"{path}.binding")
        if (
            self.declaration.identity.kind is not StateKind.TRAINABLE_PARAMETER
            or self.declaration.tensor_bytes != 1024
            or self.binding.state_ref != self.declaration.id
            or self.binding.die_id != self.home_die_id
            or self.binding.size_bytes != self.declaration.tensor_bytes
            or self.source_parameter_state_ref == self.declaration.id
        ):
            raise SchemaError(
                "invalid derived trainable down-weight state", path=path
            )


@dataclass(frozen=True, slots=True)
class LiteMoeTokenWgrad:
    id: str
    token_index: int
    expert_index: int
    home_die_id: int
    down_node_ref: str
    saved_activation_ref: str
    upstream_grad_ref: str
    down_weight_state_ref: str
    root_buffer_ref: str
    contribution_ref: str
    offset_bytes: int
    size_bytes: int
    dtype: DType
    deps: tuple[str, ...]

    def validate(self, path: str) -> None:
        for name in ("token_index", "expert_index", "home_die_id", "offset_bytes", "size_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for name in ("down_node_ref", "saved_activation_ref", "upstream_grad_ref", "down_weight_state_ref", "root_buffer_ref", "contribution_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.expert_index != self.token_index // 2
            or self.home_die_id != self.expert_index // 2
            or self.offset_bytes != (self.token_index % 2) * 2048
            or self.size_bytes != 2048
            or self.dtype is not DType.FP32
            or len(self.deps) > 1
        ):
            raise SchemaError("invalid token WGRAD geometry", path=path)
        expected = stable_artifact_id("s3_lite_moe_token_wgrad", self._semantic(), schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable WGRAD id", path=f"{path}.id")

    def _semantic(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "id"}


@dataclass(frozen=True, slots=True)
class LiteMoeExpertReduce:
    id: str
    expert_index: int
    home_die_id: int
    root_buffer_ref: str
    contribution_refs: tuple[str, str]
    input_offsets: tuple[int, int]
    input_span_bytes: int
    output_alias_ref: str
    output_offset_bytes: int
    output_size_bytes: int
    deps: tuple[str, str]

    def validate(self, path: str) -> None:
        if (
            self.expert_index not in range(4)
            or self.home_die_id != self.expert_index // 2
            or len(set(self.contribution_refs)) != 2
            or self.input_offsets != (0, 2048)
            or self.input_span_bytes != 4096
            or self.output_alias_ref != self.contribution_refs[0]
            or self.output_offset_bytes != 0
            or self.output_size_bytes != 2048
            or len(set(self.deps)) != 2
        ):
            raise SchemaError("reduce must consume contiguous 4096B and alias root slice0", path=path)
        expected = stable_artifact_id("s3_lite_moe_expert_reduce", self._semantic(), schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable reduce id", path=f"{path}.id")

    def _semantic(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "id"}


@dataclass(frozen=True, slots=True)
class LiteMoeExpertSgdStore:
    id: str
    expert_index: int
    home_die_id: int
    down_weight_state_ref: str
    down_weight_hbm_binding_ref: str
    reduce_ref: str
    gradient_alias_ref: str
    weight_read_bytes: int
    gradient_read_bytes: int
    state_store_bytes: int
    learning_rate: float
    momentum: float
    deps: tuple[str, ...]

    def validate(self, path: str) -> None:
        for name in (
            "down_weight_state_ref",
            "down_weight_hbm_binding_ref",
            "reduce_ref",
            "gradient_alias_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.expert_index not in range(4)
            or self.home_die_id != self.expert_index // 2
            or (self.weight_read_bytes, self.gradient_read_bytes, self.state_store_bytes) != (1024, 2048, 1024)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or self.momentum != 0.0
            or self.deps != (self.reduce_ref,)
        ):
            raise SchemaError("invalid exact expert SGD/store", path=path)
        expected = stable_artifact_id("s3_lite_moe_expert_sgd_store", self._semantic(), schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable SGD/store id", path=f"{path}.id")

    def _semantic(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "id"}


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardOverlay:
    schema_version: str
    producer_pass: str
    id: str
    source_n4_id: str
    source_projection_id: str
    source_schedule_id: str
    source_global_id: str
    source_n6_intent_id: str
    contract: LiteMoeBackwardContract
    oracle: LiteMoeBackwardOracle
    trainable_down_states: tuple[LiteMoeTrainableDownState, ...]
    remote_grad_dtes: tuple[LiteMoeRemoteGradDte, ...]
    token_wgrads: tuple[LiteMoeTokenWgrad, ...]
    expert_reduces: tuple[LiteMoeExpertReduce, ...]
    sgd_stores: tuple[LiteMoeExpertSgdStore, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeBackwardOverlay":
        result = cls(
            LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION,
            "lite_moe_backward_overlay",
            stable_artifact_id("s3_lite_moe_backward_overlay", semantic, schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name not in ("schema_version", "producer_pass", "id")}

    def validate(self, path: str = "lite_moe_backward_overlay") -> None:
        if self.schema_version != LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION or self.producer_pass != "lite_moe_backward_overlay":
            raise SchemaError("unsupported overlay schema/producer", path=path)
        for name in ("source_n4_id", "source_projection_id", "source_schedule_id", "source_global_id", "source_n6_intent_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        self.contract.validate(f"{path}.contract")
        self.oracle.validate_against(self.contract, f"{path}.oracle")
        expected_counts = (4, 4, 8, 4, 4)
        if tuple(map(len, (self.trainable_down_states, self.remote_grad_dtes, self.token_wgrads, self.expert_reduces, self.sgd_stores))) != expected_counts:
            raise SchemaError("overlay count quotient changed", path=path)
        for name, values in (("trainable_down_states", self.trainable_down_states), ("remote_grad_dtes", self.remote_grad_dtes), ("token_wgrads", self.token_wgrads), ("expert_reduces", self.expert_reduces), ("sgd_stores", self.sgd_stores)):
            for index, value in enumerate(values):
                value.validate(f"{path}.{name}[{index}]")
        state_ids = tuple(item.declaration.id for item in self.trainable_down_states)
        binding_ids = tuple(item.binding.id for item in self.trainable_down_states)
        source_ids = tuple(
            item.source_parameter_state_ref for item in self.trainable_down_states
        )
        if (
            len(set(state_ids)) != 4
            or len(set(binding_ids)) != 4
            or len(set(source_ids)) != 4
        ):
            raise SchemaError(
                "trainable down-state mapping must be a four-way bijection",
                path=f"{path}.trainable_down_states",
            )
        for left_index, left in enumerate(self.trainable_down_states):
            left_end = left.binding.address + left.binding.size_bytes
            for right in self.trainable_down_states[left_index + 1 :]:
                right_end = right.binding.address + right.binding.size_bytes
                if (
                    left.binding.die_id == right.binding.die_id
                    and left.binding.address < right_end
                    and right.binding.address < left_end
                ):
                    raise SchemaError(
                        "trainable HBM bindings overlap on one die",
                        path=f"{path}.trainable_down_states",
                    )
        if tuple(item.token_index for item in self.token_wgrads) != tuple(range(8)) or tuple(item.expert_index for item in self.trainable_down_states) != tuple(range(4)) or tuple(item.expert_index for item in self.expert_reduces) != tuple(range(4)) or tuple(item.expert_index for item in self.sgd_stores) != tuple(range(4)):
            raise SchemaError("overlay ordering must be canonical", path=path)
        if tuple(item.token_index for item in self.remote_grad_dtes) != (1, 3, 4, 6):
            raise SchemaError(
                "remote DTE ordering must be the exact balanced remote-token tuple",
                path=f"{path}.remote_grad_dtes",
            )
        remote_by_token = {item.token_index: item for item in self.remote_grad_dtes}
        for wgrad in self.token_wgrads:
            remote = remote_by_token.get(wgrad.token_index)
            if remote is None:
                if wgrad.deps:
                    raise SchemaError(
                        "local WGRAD must have no transport dependency",
                        path=f"{path}.token_wgrads[{wgrad.token_index}].deps",
                    )
            elif (
                wgrad.deps != (remote.id,)
                or remote.expert_index != wgrad.expert_index
                or remote.destination_die_id != wgrad.home_die_id
            ):
                raise SchemaError(
                    "remote WGRAD must depend on its exact DTE witness",
                    path=f"{path}.token_wgrads[{wgrad.token_index}].deps",
                )
        wgrad_by_expert = {expert: tuple(item for item in self.token_wgrads if item.expert_index == expert) for expert in range(4)}
        for expert, reduce in enumerate(self.expert_reduces):
            values = wgrad_by_expert[expert]
            if len(values) != 2 or reduce.root_buffer_ref != values[0].root_buffer_ref or reduce.contribution_refs != tuple(item.contribution_ref for item in values) or reduce.deps != tuple(item.id for item in values):
                raise SchemaError("reduce does not close exact expert WGRAD contributions", path=f"{path}.expert_reduces[{expert}]")
            store = self.sgd_stores[expert]
            trainable = self.trainable_down_states[expert]
            if store.reduce_ref != reduce.id or store.gradient_alias_ref != reduce.output_alias_ref or store.down_weight_state_ref != trainable.declaration.id or store.down_weight_hbm_binding_ref != trainable.binding.id or values[0].down_weight_state_ref != trainable.declaration.id or any(item.down_weight_state_ref != store.down_weight_state_ref for item in values):
                raise SchemaError("SGD/store does not close exact reduce/state lineage", path=f"{path}.sgd_stores[{expert}]")
        expected_id = stable_artifact_id("s3_lite_moe_backward_overlay", self._semantic(), schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError("unstable overlay id", path=f"{path}.id")


__all__ = [
    "LITE_MOE_BACKWARD_CONTRACT_SCHEMA_VERSION",
    "LITE_MOE_BACKWARD_ORACLE_SCHEMA_VERSION",
    "LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION",
    "S3_LITE_MOE_BACKWARD_CASE_ID",
    "LiteMoeBackwardCoverage",
    "LiteMoeBackwardContract",
    "LiteMoeBackwardOracle",
    "LiteMoeRemoteGradDte",
    "LiteMoeTrainableDownState",
    "LiteMoeTokenWgrad",
    "LiteMoeExpertReduce",
    "LiteMoeExpertSgdStore",
    "LiteMoeBackwardOverlay",
]
