"""Immutable context consumed by the N3 placement pass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import PlacementSpec
from .ir1 import PhysicalFabric
from .persistent_state import HbmAddressSpace, StateKind


PLACEMENT_CONTEXT_SCHEMA_VERSION = "wafer_frontend.placement_context/v1alpha3"
PERSISTENT_STATE_RESERVATION_POLICY_SCHEMA_VERSION = (
    "wafer_frontend.persistent_state_reservation_policy/v1alpha1"
)
PERSISTENT_STATE_SLOT_RESERVATION_SCHEMA_VERSION = (
    "wafer_frontend.persistent_state_slot_reservation/v1alpha1"
)


class TrafficTemplate(str, Enum):
    """Canonical traffic templates understood by the N3 placement model."""

    DIRECT_A2A_UNIT_CHUNK_V1 = "direct_a2a_unit_chunk/v1"


@dataclass(frozen=True, slots=True)
class PersistentStateSlotReservation:
    """Fixed per-state HBM slot size for one persistent-state kind."""

    schema_version: str
    producer_pass: str
    id: str
    kind: StateKind
    slot_bytes: int

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        kind: StateKind,
        slot_bytes: int,
    ) -> "PersistentStateSlotReservation":
        semantic = {"kind": kind, "slot_bytes": slot_bytes}
        result = cls(
            schema_version=PERSISTENT_STATE_SLOT_RESERVATION_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "persistent_state_slot_reservation",
                semantic,
                schema_version=PERSISTENT_STATE_SLOT_RESERVATION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "persistent_state_slot_reservation") -> None:
        if self.schema_version != PERSISTENT_STATE_SLOT_RESERVATION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.kind) is not StateKind:
            raise SchemaError("must be a StateKind", path=f"{path}.kind")
        validate_uint64(self.slot_bytes, f"{path}.slot_bytes")
        if self.slot_bytes == 0 or self.slot_bytes % 64:
            raise SchemaError(
                "must be a positive multiple of 64 bytes",
                path=f"{path}.slot_bytes",
            )
        expected = stable_artifact_id(
            "persistent_state_slot_reservation",
            {"kind": self.kind, "slot_bytes": self.slot_bytes},
            schema_version=PERSISTENT_STATE_SLOT_RESERVATION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable artifact id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class PersistentStateReservationPolicy:
    """Versioned fixed-slot policy preserving state addresses across programs."""

    schema_version: str
    producer_pass: str
    id: str
    slots: tuple[PersistentStateSlotReservation, ...]

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        slots: tuple[PersistentStateSlotReservation, ...],
    ) -> "PersistentStateReservationPolicy":
        semantic = {"slots": tuple(sorted(slots, key=lambda item: item.kind.value))}
        result = cls(
            schema_version=PERSISTENT_STATE_RESERVATION_POLICY_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "persistent_state_reservation_policy",
                semantic,
                schema_version=PERSISTENT_STATE_RESERVATION_POLICY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "persistent_state_reservation_policy") -> None:
        if self.schema_version != PERSISTENT_STATE_RESERVATION_POLICY_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.slots) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.slots")
        if not self.slots:
            raise SchemaError("must not be empty", path=f"{path}.slots")
        if self.slots != tuple(sorted(self.slots, key=lambda item: item.kind.value)):
            raise SchemaError("must use canonical kind order", path=f"{path}.slots")
        kinds: set[StateKind] = set()
        for index, slot in enumerate(self.slots):
            if type(slot) is not PersistentStateSlotReservation:
                raise SchemaError(
                    "must be a PersistentStateSlotReservation",
                    path=f"{path}.slots[{index}]",
                )
            slot.validate(f"{path}.slots[{index}]")
            if slot.kind in kinds:
                raise SchemaError("duplicate state kind", path=f"{path}.slots[{index}].kind")
            kinds.add(slot.kind)
        expected = stable_artifact_id(
            "persistent_state_reservation_policy",
            {"slots": self.slots},
            schema_version=PERSISTENT_STATE_RESERVATION_POLICY_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable artifact id", path=f"{path}.id")

    def slot_bytes(self, kind: StateKind) -> int | None:
        return next(
            (slot.slot_bytes for slot in self.slots if slot.kind is kind),
            None,
        )


@dataclass(frozen=True, slots=True)
class PlacementContext:
    """Self-contained, canonical context for ``PassManager.run_pass``.

    N3 supports one traffic contract only: every ordered pair in a physical
    group sends one unit chunk.  Keeping the version in the enum value makes a
    future change to that contract an explicit schema migration rather than a
    silent reinterpretation of an existing context digest.
    """

    schema_version: str
    producer_pass: str
    id: str
    fabric: PhysicalFabric
    placement: PlacementSpec
    traffic_templates: tuple[TrafficTemplate, ...]
    hbm_address_spaces: tuple[HbmAddressSpace, ...]
    persistent_state_reservation_policy: PersistentStateReservationPolicy | None

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        fabric: PhysicalFabric,
        placement: PlacementSpec,
        traffic_templates: tuple[TrafficTemplate, ...] = (
            TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1,
        ),
        hbm_address_spaces: tuple[HbmAddressSpace, ...] = (),
        persistent_state_reservation_policy: PersistentStateReservationPolicy | None = None,
    ) -> "PlacementContext":
        semantic_key = {
            "fabric": fabric,
            "placement": placement,
            "traffic_templates": traffic_templates,
            "hbm_address_spaces": hbm_address_spaces,
            "persistent_state_reservation_policy": persistent_state_reservation_policy,
        }
        return cls(
            schema_version=PLACEMENT_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "placement_context",
                semantic_key,
                schema_version=PLACEMENT_CONTEXT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "fabric": self.fabric,
            "placement": self.placement,
            "traffic_templates": self.traffic_templates,
            "hbm_address_spaces": self.hbm_address_spaces,
            "persistent_state_reservation_policy": self.persistent_state_reservation_policy,
        }

    def validate(self, path: str = "placement_context") -> None:
        if self.schema_version != PLACEMENT_CONTEXT_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.fabric) is not PhysicalFabric:
            raise SchemaError("must be a PhysicalFabric", path=f"{path}.fabric")
        if type(self.placement) is not PlacementSpec:
            raise SchemaError("must be a PlacementSpec", path=f"{path}.placement")
        if type(self.traffic_templates) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.traffic_templates",
            )
        for index, template in enumerate(self.traffic_templates):
            if type(template) is not TrafficTemplate:
                raise SchemaError(
                    "must be a TrafficTemplate",
                    path=f"{path}.traffic_templates[{index}]",
                )

        expected_templates = (TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1,)
        if self.traffic_templates != expected_templates:
            raise SchemaError(
                "must contain exactly 'direct_a2a_unit_chunk/v1' once",
                path=f"{path}.traffic_templates",
            )

        self.fabric.validate(f"{path}.fabric")
        self.placement.validate(f"{path}.placement")
        if self.persistent_state_reservation_policy is not None:
            if type(self.persistent_state_reservation_policy) is not PersistentStateReservationPolicy:
                raise SchemaError(
                    "must be a PersistentStateReservationPolicy",
                    path=f"{path}.persistent_state_reservation_policy",
                )
            self.persistent_state_reservation_policy.validate(
                f"{path}.persistent_state_reservation_policy"
            )
        known_die_ids = {die.id for die in self.fabric.dies}
        if type(self.hbm_address_spaces) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.hbm_address_spaces",
            )
        if self.hbm_address_spaces != tuple(
            sorted(self.hbm_address_spaces, key=lambda item: (item.die_id, item.id))
        ):
            raise SchemaError(
                "must use canonical die/id order",
                path=f"{path}.hbm_address_spaces",
            )
        hbm_die_ids: set[int] = set()
        hbm_ranges: list[tuple[int, int]] = []
        for index, space in enumerate(self.hbm_address_spaces):
            space_path = f"{path}.hbm_address_spaces[{index}]"
            if type(space) is not HbmAddressSpace:
                raise SchemaError("must be an HbmAddressSpace", path=space_path)
            space.validate(space_path)
            if space.die_id not in known_die_ids:
                raise SchemaError(
                    "references a die outside the physical fabric",
                    path=f"{space_path}.die_id",
                )
            if space.die_id in hbm_die_ids:
                raise SchemaError("duplicate die HBM address space", path=space_path)
            if space.alignment_bytes != 64:
                raise SchemaError(
                    "Stage-1a requires exactly 64-byte HBM alignment",
                    path=f"{space_path}.alignment_bytes",
                )
            start = space.base_address
            end = start + space.size_bytes
            if any(
                start < old_end and old_start < end
                for old_start, old_end in hbm_ranges
            ):
                raise SchemaError(
                    "HBM address spaces must not overlap", path=space_path
                )
            hbm_die_ids.add(space.die_id)
            hbm_ranges.append((start, end))
        if hbm_die_ids and hbm_die_ids != known_die_ids:
            raise SchemaError(
                "must cover every physical die when HBM backing is enabled",
                path=f"{path}.hbm_address_spaces",
            )
        for group_index, group in enumerate(self.placement.groups):
            for die_index, die_id in enumerate(group.die_ids):
                if die_id not in known_die_ids:
                    raise SchemaError(
                        "references a die outside the physical fabric",
                        path=(
                            f"{path}.placement.groups[{group_index}]"
                            f".die_ids[{die_index}]"
                        ),
                    )

        expected_id = stable_artifact_id(
            "placement_context",
            self._semantic_key(),
            schema_version=PLACEMENT_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


__all__ = [
    "PLACEMENT_CONTEXT_SCHEMA_VERSION",
    "PlacementContext",
    "TrafficTemplate",
]
