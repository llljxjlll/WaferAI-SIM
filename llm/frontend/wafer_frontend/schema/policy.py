"""Versioned policy selection identity shared by contexts and registries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty


POLICY_SELECTION_SCHEMA_VERSION = "wafer_frontend.policy_selection/v1alpha1"


class RegistryKind(str, Enum):
    FUSION_PARTITION = "fusion_partition"
    INTER_DIE = "inter_die"
    STANDALONE_COLLECTIVE = "standalone_collective"
    INTRA_DIE = "intra_die"
    COARSE_LOWERING = "coarse_lowering"
    STANDALONE_LOWERING = "standalone_lowering"
    FUSED_LOWERING = "fused_lowering"
    MANIFEST_LINKER = "manifest_linker"


_INTERFACE_VERSIONS = MappingProxyType(
    {
        RegistryKind.FUSION_PARTITION: (
            "wafer_frontend.policy_interface/fusion_partition/v1"
        ),
        RegistryKind.INTER_DIE: "wafer_frontend.policy_interface/inter_die/v1",
        RegistryKind.STANDALONE_COLLECTIVE: (
            "wafer_frontend.policy_interface/standalone_collective/v1"
        ),
        RegistryKind.INTRA_DIE: "wafer_frontend.policy_interface/intra_die/v1",
        RegistryKind.COARSE_LOWERING: (
            "wafer_frontend.policy_interface/coarse_lowering/v1"
        ),
        RegistryKind.STANDALONE_LOWERING: (
            "wafer_frontend.policy_interface/standalone_lowering/v1"
        ),
        RegistryKind.FUSED_LOWERING: (
            "wafer_frontend.policy_interface/fused_lowering/v1"
        ),
        RegistryKind.MANIFEST_LINKER: (
            "wafer_frontend.policy_interface/manifest_linker/v1"
        ),
    }
)


def policy_interface_version(kind: RegistryKind) -> str:
    if type(kind) is not RegistryKind:
        raise SchemaError("must be a RegistryKind", path="policy.kind")
    return _INTERFACE_VERSIONS[kind]


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _validate_capability_ids(values: tuple[str, ...], path: str) -> None:
    if type(values) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    previous: str | None = None
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        validate_nonempty(value, item_path)
        if previous is not None and value <= previous:
            raise SchemaError(
                "must be unique and strictly increasing", path=item_path
            )
        previous = value
    if not values:
        raise SchemaError("must contain at least one capability", path=path)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    schema_version: str
    id: str
    kind: RegistryKind
    name: str
    interface_version: str
    implementation_id: str
    implementation_schema_version: str
    configuration_digest: str
    capability_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        kind: RegistryKind,
        name: str,
        implementation_id: str,
        implementation_schema_version: str,
        configuration_digest: str,
        capability_ids: tuple[str, ...],
    ) -> "PolicySelection":
        semantic_key = {
            "kind": kind,
            "name": name,
            "interface_version": policy_interface_version(kind),
            "implementation_id": implementation_id,
            "implementation_schema_version": implementation_schema_version,
            "configuration_digest": configuration_digest,
            "capability_ids": capability_ids,
        }
        result = cls(
            schema_version=POLICY_SELECTION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "policy_selection",
                semantic_key,
                schema_version=POLICY_SELECTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def validate(self, path: str = "policy_selection") -> None:
        if self.schema_version != POLICY_SELECTION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if type(self.kind) is not RegistryKind:
            raise SchemaError("must be a RegistryKind", path=f"{path}.kind")
        validate_nonempty(self.name, f"{path}.name")
        expected_interface = policy_interface_version(self.kind)
        if self.interface_version != expected_interface:
            raise SchemaError(
                f"must be exactly {expected_interface!r}",
                path=f"{path}.interface_version",
            )
        validate_nonempty(self.implementation_id, f"{path}.implementation_id")
        validate_nonempty(
            self.implementation_schema_version,
            f"{path}.implementation_schema_version",
        )
        _validate_digest(
            self.configuration_digest, f"{path}.configuration_digest"
        )
        _validate_capability_ids(self.capability_ids, f"{path}.capability_ids")
        expected_id = stable_artifact_id(
            "policy_selection",
            {
                "kind": self.kind,
                "name": self.name,
                "interface_version": self.interface_version,
                "implementation_id": self.implementation_id,
                "implementation_schema_version": (
                    self.implementation_schema_version
                ),
                "configuration_digest": self.configuration_digest,
                "capability_ids": self.capability_ids,
            },
            schema_version=POLICY_SELECTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable selection id; expected {expected_id!r}",
                path=f"{path}.id",
            )
