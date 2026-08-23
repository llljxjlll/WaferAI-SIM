"""Explicit policy/lowering registration without aliases or fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
import inspect
from types import MappingProxyType

from ..errors import RegistryError, StageNotImplementedError, UnsupportedFeatureError
from ..schema.policy import (
    POLICY_SELECTION_SCHEMA_VERSION,
    PolicySelection,
    RegistryKind,
    policy_interface_version,
)
from ..schema.serde import canonical_digest


class RegistrationState(str, Enum):
    DECLARED = "declared"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class _RegistryContract:
    interface: str
    interface_version: str
    required_method: str


_KIND_CONTRACTS = MappingProxyType(
    {
        RegistryKind.FUSION_PARTITION: _RegistryContract(
            "FusionPartition",
            policy_interface_version(RegistryKind.FUSION_PARTITION),
            "run",
        ),
        RegistryKind.INTER_DIE: _RegistryContract(
            "InterDiePolicy",
            policy_interface_version(RegistryKind.INTER_DIE),
            "plan",
        ),
        RegistryKind.STANDALONE_COLLECTIVE: _RegistryContract(
            "StandaloneCollectivePolicy",
            policy_interface_version(RegistryKind.STANDALONE_COLLECTIVE),
            "plan",
        ),
        RegistryKind.INTRA_DIE: _RegistryContract(
            "IntraDiePolicy",
            policy_interface_version(RegistryKind.INTRA_DIE),
            "schedule",
        ),
        RegistryKind.COARSE_LOWERING: _RegistryContract(
            "CoarseLowering",
            policy_interface_version(RegistryKind.COARSE_LOWERING),
            "lower",
        ),
        RegistryKind.STANDALONE_LOWERING: _RegistryContract(
            "StandaloneCollectiveLowering",
            policy_interface_version(RegistryKind.STANDALONE_LOWERING),
            "lower",
        ),
        RegistryKind.FUSED_LOWERING: _RegistryContract(
            "IsaRegionLowering",
            policy_interface_version(RegistryKind.FUSED_LOWERING),
            "lower",
        ),
        RegistryKind.MANIFEST_LINKER: _RegistryContract(
            "ManifestLinker",
            policy_interface_version(RegistryKind.MANIFEST_LINKER),
            "link",
        ),
    }
)


def _require_nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError("must be a non-empty string", path=path)
    return value


def _require_digest(value: object, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegistryError("must be a lowercase SHA-256 digest", path=path)
    return value


def _require_sorted_capability_ids(
    values: object, path: str
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise RegistryError("must be an immutable tuple", path=path)
    previous: str | None = None
    for index, value in enumerate(values):
        _require_nonempty(value, f"{path}[{index}]")
        if previous is not None and value <= previous:
            raise RegistryError(
                "must be unique and strictly increasing", path=f"{path}[{index}]"
            )
        previous = value
    return values


@dataclass(frozen=True, slots=True)
class Registration:
    kind: RegistryKind
    name: str
    interface: str
    interface_version: str
    available_stage: str
    state: RegistrationState
    implementation_id: str | None
    implementation_schema_version: str | None
    configuration_digest: str | None
    capability_ids: tuple[str, ...]

    def validate(self, path: str = "registration") -> None:
        if type(self.kind) is not RegistryKind:
            raise RegistryError("kind must be a RegistryKind", path=f"{path}.kind")
        contract = _KIND_CONTRACTS[self.kind]
        if self.interface != contract.interface:
            raise RegistryError(
                f"interface must be exactly {contract.interface!r}",
                path=f"{path}.interface",
            )
        if self.interface_version != contract.interface_version:
            raise RegistryError(
                (
                    "interface_version must be exactly "
                    f"{contract.interface_version!r}"
                ),
                path=f"{path}.interface_version",
            )
        for field_name in ("name", "available_stage"):
            _require_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.state) is not RegistrationState:
            raise RegistryError(
                "state must be a RegistrationState", path=f"{path}.state"
            )
        _require_sorted_capability_ids(
            self.capability_ids, f"{path}.capability_ids"
        )
        metadata = (
            self.implementation_id,
            self.implementation_schema_version,
            self.configuration_digest,
        )
        if self.state is RegistrationState.DECLARED:
            if any(value is not None for value in metadata) or self.capability_ids:
                raise RegistryError(
                    "declared implementations cannot claim active metadata",
                    path=path,
                )
            return
        for field_name in (
            "implementation_id",
            "implementation_schema_version",
        ):
            _require_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _require_digest(
            self.configuration_digest, f"{path}.configuration_digest"
        )
        if not self.capability_ids:
            raise RegistryError(
                "active implementations require capability_ids",
                path=f"{path}.capability_ids",
            )


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    selection: PolicySelection
    implementation: object


Factory = Callable[[], object]


class PolicyRegistry:
    def __init__(self) -> None:
        self._registrations: dict[tuple[RegistryKind, str], Registration] = {}
        self._factories: dict[tuple[RegistryKind, str], Factory] = {}

    def declare(
        self,
        kind: RegistryKind,
        name: str,
        *,
        interface: str,
        available_stage: str,
    ) -> Registration:
        kind = self._require_kind(kind)
        path = f"registry.{kind.value}"
        for field_name, value in (
            ("name", name),
            ("interface", interface),
            ("available_stage", available_stage),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RegistryError(
                    f"{field_name} must be a non-empty string",
                    path=path,
                )
        contract = _KIND_CONTRACTS[kind]
        if interface != contract.interface:
            raise RegistryError(
                f"interface must be exactly {contract.interface!r} for {kind.value}",
                path=path,
            )
        key = (kind, name)
        if key in self._registrations:
            raise RegistryError(
                f"duplicate registration {name!r}", path=path
            )
        registration = Registration(
            kind=kind,
            name=name,
            interface=interface,
            interface_version=contract.interface_version,
            available_stage=available_stage,
            state=RegistrationState.DECLARED,
            implementation_id=None,
            implementation_schema_version=None,
            configuration_digest=None,
            capability_ids=(),
        )
        registration.validate(path)
        self._registrations[key] = registration
        return registration

    def activate(
        self,
        kind: RegistryKind,
        name: str,
        factory: Factory,
        *,
        implementation_id: str,
        implementation_schema_version: str,
        capability_ids: tuple[str, ...],
        configuration: object = (),
    ) -> Registration:
        kind = self._require_kind(kind)
        name = self._require_name(name, kind)
        path = f"registry.{kind.value}"
        key = (kind, name)
        registration = self._registrations.get(key)
        if registration is None:
            raise RegistryError(
                f"cannot activate undeclared implementation {name!r}",
                path=path,
            )
        if registration.state is RegistrationState.ACTIVE:
            raise RegistryError(
                f"implementation {name!r} is already active",
                path=path,
            )
        if not callable(factory):
            raise RegistryError("factory must be callable", path=path)
        try:
            inspect.signature(factory).bind()
        except (TypeError, ValueError) as exc:
            raise RegistryError(
                "factory must be callable with zero arguments",
                path=path,
            ) from exc
        _require_nonempty(implementation_id, f"{path}.implementation_id")
        _require_nonempty(
            implementation_schema_version,
            f"{path}.implementation_schema_version",
        )
        _require_sorted_capability_ids(
            capability_ids, f"{path}.capability_ids"
        )
        if not capability_ids:
            raise RegistryError(
                "active implementations require capability_ids",
                path=f"{path}.capability_ids",
            )
        try:
            configuration_digest = canonical_digest(configuration)
        except Exception as exc:
            raise RegistryError(
                f"configuration is not canonical-JSON serializable: {exc}",
                path=f"{path}.configuration",
            ) from exc
        active = replace(
            registration,
            state=RegistrationState.ACTIVE,
            implementation_id=implementation_id,
            implementation_schema_version=implementation_schema_version,
            configuration_digest=configuration_digest,
            capability_ids=capability_ids,
        )
        active.validate(path)
        self._registrations[key] = active
        self._factories[key] = factory
        return active

    def registration(self, kind: RegistryKind, name: str) -> Registration:
        kind = self._require_kind(kind)
        name = self._require_name(name, kind)
        registration = self._registrations.get((kind, name))
        if registration is None:
            raise UnsupportedFeatureError(
                f"implementation {name!r} is not registered and will not fall back",
                path=f"registry.{kind.value}",
            )
        return registration

    def resolve(self, kind: RegistryKind, name: str) -> Factory:
        registration = self.registration(kind, name)
        if registration.state is RegistrationState.DECLARED:
            raise StageNotImplementedError(
                f"implementation {name!r} is declared for {registration.available_stage} "
                "but is not active",
                path=f"registry.{kind.value}",
            )
        return self._factories[(kind, name)]

    def create(self, kind: RegistryKind, name: str) -> object:
        kind = self._require_kind(kind)
        name = self._require_name(name, kind)
        path = f"registry.{kind.value}"
        factory = self.resolve(kind, name)
        try:
            implementation = factory()
        except Exception as exc:
            raise RegistryError(
                f"factory failed for implementation {name!r}: {exc}",
                path=path,
            ) from exc
        required_method = _KIND_CONTRACTS[kind].required_method
        try:
            method = getattr(implementation, required_method)
        except Exception as exc:
            raise RegistryError(
                f"implementation {name!r} must provide callable {required_method}",
                path=path,
            ) from exc
        if not callable(method):
            raise RegistryError(
                f"implementation {name!r} must provide callable {required_method}",
                path=path,
            )
        return implementation

    def instantiate(self, kind: RegistryKind, name: str) -> ResolvedPolicy:
        kind = self._require_kind(kind)
        name = self._require_name(name, kind)
        implementation = self.create(kind, name)
        registration = self.registration(kind, name)
        assert registration.implementation_id is not None
        assert registration.implementation_schema_version is not None
        assert registration.configuration_digest is not None
        selection = PolicySelection.create(
            kind=registration.kind,
            name=registration.name,
            implementation_id=registration.implementation_id,
            implementation_schema_version=(
                registration.implementation_schema_version
            ),
            configuration_digest=registration.configuration_digest,
            capability_ids=registration.capability_ids,
        )
        return ResolvedPolicy(
            selection=selection,
            implementation=implementation,
        )

    def entries(self) -> tuple[Registration, ...]:
        return tuple(
            sorted(
                self._registrations.values(),
                key=lambda item: (item.kind.value, item.name),
            )
        )

    @staticmethod
    def _require_kind(kind: object) -> RegistryKind:
        if not isinstance(kind, RegistryKind):
            raise RegistryError("kind must be a RegistryKind", path="registry")
        return kind

    @staticmethod
    def _require_name(name: object, kind: RegistryKind) -> str:
        if not isinstance(name, str) or not name.strip():
            raise RegistryError(
                "name must be a non-empty string",
                path=f"registry.{kind.value}",
            )
        return name


def production_registry() -> PolicyRegistry:
    from .naive_inter_die import DirectAllGatherPolicy, NaiveInterDiePolicy
    from .naive_intra_die import (
        NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
        NaiveIntraDiePolicy,
    )
    from .swizzle_defaults import SWIZZLE_POLICY_SCHEMA_VERSION, production_swizzle_policy

    registry = PolicyRegistry()
    for kind, name, interface, stage in (
        (RegistryKind.FUSION_PARTITION, "gemm_coll", "FusionPartition", "N4"),
        (RegistryKind.INTER_DIE, "naive", "InterDiePolicy", "N4"),
        (RegistryKind.INTER_DIE, "swizzle_topo", "InterDiePolicy", "O1"),
        (
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
            "StandaloneCollectivePolicy",
            "N4",
        ),
        (RegistryKind.INTRA_DIE, "naive", "IntraDiePolicy", "N5"),
        (RegistryKind.INTRA_DIE, "optimized", "IntraDiePolicy", "O2"),
        (RegistryKind.COARSE_LOWERING, "json_coarse", "CoarseLowering", "N6"),
        (
            RegistryKind.STANDALONE_LOWERING,
            "strict_actions",
            "StandaloneCollectiveLowering",
            "N6",
        ),
        (RegistryKind.FUSED_LOWERING, "isa_region", "IsaRegionLowering", "N6"),
        (RegistryKind.MANIFEST_LINKER, "manifest_v1", "ManifestLinker", "N6"),
    ):
        registry.declare(
            kind,
            name,
            interface=interface,
            available_stage=stage,
        )
    shared_capabilities = ("s1.gemm_collective.naive",)
    registry.activate(
        RegistryKind.INTER_DIE,
        "naive",
        NaiveInterDiePolicy,
        implementation_id="wafer_frontend.policy.inter_die.naive",
        implementation_schema_version="wafer_frontend.naive_inter_die_policy/v1",
        capability_ids=shared_capabilities,
    )
    registry.activate(
        RegistryKind.INTER_DIE,
        "swizzle_topo",
        production_swizzle_policy,
        implementation_id="wafer_frontend.policy.inter_die.swizzle_topo",
        implementation_schema_version=SWIZZLE_POLICY_SCHEMA_VERSION,
        capability_ids=(
            "o1.swizzle.ag_gemm",
            "o1.swizzle.gemm_ar",
            "o1.swizzle.gemm_rs",
        ),
    )
    registry.activate(
        RegistryKind.STANDALONE_COLLECTIVE,
        "direct_all_gather",
        DirectAllGatherPolicy,
        implementation_id="wafer_frontend.policy.standalone.direct_all_gather",
        implementation_schema_version=(
            "wafer_frontend.direct_all_gather_policy/v1"
        ),
        capability_ids=shared_capabilities,
    )
    registry.activate(
        RegistryKind.INTRA_DIE,
        "naive",
        NaiveIntraDiePolicy,
        implementation_id="wafer_frontend.policy.intra_die.naive",
        implementation_schema_version=NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
        capability_ids=shared_capabilities,
    )
    return registry


def default_registry() -> PolicyRegistry:
    """Return a fresh production registry; retained as the public default name."""

    return production_registry()
