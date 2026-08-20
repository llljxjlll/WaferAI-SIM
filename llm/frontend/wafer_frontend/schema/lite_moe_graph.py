"""Isolated IR0 carrier for the S3-Lite static-route MoE preview."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import IR0
from .lite_moe import S3_LITE_STATIC_MOE_CASE_ID, LiteMoeTransferRole


LITE_MOE_P2P_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_p2p_binding/v1alpha1"
)
LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_ir0_adapter/v1alpha1"
)

_ROLE_RANK = {
    LiteMoeTransferRole.MOE_DISPATCH: 0,
    LiteMoeTransferRole.MOE_COMBINE: 1,
}


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a canonical SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeP2PBinding:
    schema_version: str
    id: str
    node_ref: str
    role: LiteMoeTransferRole
    token_index: int
    expert_index: int
    source_die_id: int
    destination_die_id: int

    @classmethod
    def create(
        cls,
        *,
        node_ref: str,
        role: LiteMoeTransferRole,
        token_index: int,
        expert_index: int,
        source_die_id: int,
        destination_die_id: int,
    ) -> "LiteMoeP2PBinding":
        semantic_key = {
            "node_ref": node_ref,
            "role": role,
            "token_index": token_index,
            "expert_index": expert_index,
            "source_die_id": source_die_id,
            "destination_die_id": destination_die_id,
        }
        result = cls(
            schema_version=LITE_MOE_P2P_BINDING_SCHEMA_VERSION,
            id=stable_artifact_id(
                "s3_lite_static_moe_p2p_binding",
                semantic_key,
                schema_version=LITE_MOE_P2P_BINDING_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "lite_moe_p2p_binding") -> None:
        if self.schema_version != LITE_MOE_P2P_BINDING_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if type(self.role) is not LiteMoeTransferRole:
            raise SchemaError(
                "must be a LiteMoeTransferRole", path=f"{path}.role"
            )
        for field_name in (
            "token_index",
            "expert_index",
            "source_die_id",
            "destination_die_id",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.token_index >= 8 or self.expert_index >= 4:
            raise SchemaError(
                "token/expert lies outside the fixed S3-Lite case", path=path
            )
        token_home = self.token_index % 2
        expert_home = self.expert_index // 2
        expected_endpoints = (
            (token_home, expert_home)
            if self.role is LiteMoeTransferRole.MOE_DISPATCH
            else (expert_home, token_home)
        )
        if token_home == expert_home:
            raise SchemaError(
                "local tokens must not create P2P bindings", path=path
            )
        if (self.source_die_id, self.destination_die_id) != expected_endpoints:
            raise SchemaError(
                "endpoints must follow token-home/expert-home direction",
                path=path,
            )
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_p2p_binding",
            self._semantic_key(),
            schema_version=LITE_MOE_P2P_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeIR0Adapter:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    source_experiment_digest: str
    source_moe_spec_id: str
    source_moe_spec_digest: str
    source_oracle_id: str
    source_oracle_digest: str
    graph: IR0
    p2p_bindings: tuple[LiteMoeP2PBinding, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeIR0Adapter":
        result = cls(
            schema_version=LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
            producer_pass="lite_moe_graph",
            id=stable_artifact_id(
                "s3_lite_static_moe_ir0_adapter",
                semantic_key,
                schema_version=LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_moe_ir0_adapter") -> None:
        if self.schema_version != LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "lite_moe_graph":
            raise SchemaError(
                "must be 'lite_moe_graph'", path=f"{path}.producer_pass"
            )
        if self.case_id != S3_LITE_STATIC_MOE_CASE_ID:
            raise SchemaError(
                f"must be {S3_LITE_STATIC_MOE_CASE_ID!r}",
                path=f"{path}.case_id",
            )
        for field_name in (
            "source_experiment_digest",
            "source_moe_spec_digest",
            "source_oracle_digest",
        ):
            _digest(getattr(self, field_name), f"{path}.{field_name}")
        validate_nonempty(self.source_moe_spec_id, f"{path}.source_moe_spec_id")
        validate_nonempty(self.source_oracle_id, f"{path}.source_oracle_id")
        if type(self.graph) is not IR0:
            raise SchemaError("must be an IR0", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if type(self.p2p_bindings) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.p2p_bindings"
            )
        keys: list[tuple[int, int]] = []
        node_refs: set[str] = set()
        binding_ids: set[str] = set()
        for index, binding in enumerate(self.p2p_bindings):
            binding_path = f"{path}.p2p_bindings[{index}]"
            if type(binding) is not LiteMoeP2PBinding:
                raise SchemaError("must be a LiteMoeP2PBinding", path=binding_path)
            binding.validate(binding_path)
            if binding.node_ref in node_refs or binding.id in binding_ids:
                raise SchemaError(
                    "contains a duplicate node or binding identity",
                    path=binding_path,
                )
            node_refs.add(binding.node_ref)
            binding_ids.add(binding.id)
            keys.append((binding.token_index, _ROLE_RANK[binding.role]))
        if tuple(keys) != tuple(sorted(keys)):
            raise SchemaError(
                "must use canonical token/role order",
                path=f"{path}.p2p_bindings",
            )
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_ir0_adapter",
            self._semantic_key(),
            schema_version=LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


__all__ = [
    "LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION",
    "LITE_MOE_P2P_BINDING_SCHEMA_VERSION",
    "LiteMoeIR0Adapter",
    "LiteMoeP2PBinding",
]
