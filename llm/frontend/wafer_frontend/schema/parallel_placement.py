"""Explicit logical-parallel placement on a physical rectangular Die Mesh.

This module deliberately does not replace ``flexible_mesh_groups``.  It is the
P1 contract used by new workload entry points: ``RectMeshSpec`` describes every
physical Die (including routing-only/idle Dies), while this artifact describes
only workload ranks and their logical parallel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty
from .rect_mesh import RectMeshSpec
from .serde import canonical_digest


PARALLEL_PLACEMENT_SCHEMA_VERSION = "wafer_frontend.parallel_placement/v1alpha1"
PARALLEL_GROUP_SCHEMA_VERSION = "wafer_frontend.parallel_group/v1alpha1"
PARAMETER_OWNERSHIP_SCHEMA_VERSION = (
    "wafer_frontend.parameter_ownership_domain/v1alpha1"
)


class ParallelWorkloadKind(str, Enum):
    DENSE = "dense"
    MOE = "moe"


class ParallelGroupKind(str, Enum):
    MODEL = "model"
    TP = "tp"
    DP = "dp"
    EP = "ep"
    SHARED_PARAMETER_SYNC = "shared_parameter_sync"
    EXPERT_GRADIENT_SYNC = "expert_gradient_sync"


class ParameterOwnershipKind(str, Enum):
    SHARED = "shared"
    EXPERT = "expert"


@dataclass(frozen=True, slots=True)
class LogicalRankCoordinate:
    """Logical coordinate; PP is present in the ABI but fixed to zero in v1."""

    dp: int
    ep: int
    tp: int
    pp: int = 0

    def validate(self, path: str = "logical_coordinate") -> None:
        for name in ("dp", "ep", "tp", "pp"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SchemaError("must be a non-negative int", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class ParallelRankPlacement:
    logical_rank: int
    coordinate: LogicalRankCoordinate
    die_id: int
    local_core: int

    def validate(self, path: str = "rank_placement") -> None:
        for name in ("logical_rank", "die_id", "local_core"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SchemaError("must be a non-negative int", path=f"{path}.{name}")
        if type(self.coordinate) is not LogicalRankCoordinate:
            raise SchemaError(
                "must be a LogicalRankCoordinate", path=f"{path}.coordinate"
            )
        self.coordinate.validate(f"{path}.coordinate")


@dataclass(frozen=True, slots=True)
class ParallelGroup:
    schema_version: str
    id: str
    kind: ParallelGroupKind
    index: int
    ranks: tuple[int, ...]

    @classmethod
    def create(
        cls,
        *,
        kind: ParallelGroupKind,
        index: int,
        ranks: tuple[int, ...],
    ) -> "ParallelGroup":
        semantic = {"kind": kind, "index": index, "ranks": ranks}
        result = cls(
            schema_version=PARALLEL_GROUP_SCHEMA_VERSION,
            id=stable_artifact_id(
                "parallel_group",
                semantic,
                schema_version=PARALLEL_GROUP_SCHEMA_VERSION,
            ),
            kind=kind,
            index=index,
            ranks=ranks,
        )
        result.validate()
        return result

    def validate(self, path: str = "parallel_group") -> None:
        if self.schema_version != PARALLEL_GROUP_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.id, f"{path}.id")
        if type(self.kind) is not ParallelGroupKind:
            raise SchemaError("must be a ParallelGroupKind", path=f"{path}.kind")
        if type(self.index) is not int or self.index < 0:
            raise SchemaError("must be a non-negative int", path=f"{path}.index")
        if (
            type(self.ranks) is not tuple
            or not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
        ):
            raise SchemaError(
                "must contain unique non-negative logical ranks",
                path=f"{path}.ranks",
            )
        expected = stable_artifact_id(
            "parallel_group",
            {"kind": self.kind, "index": self.index, "ranks": self.ranks},
            schema_version=PARALLEL_GROUP_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable group id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class ParameterOwnershipDomain:
    """Ranks storing replicas of one TP shard of shared or expert parameters."""

    schema_version: str
    id: str
    kind: ParameterOwnershipKind
    tp_shard: int
    expert_id: int | None
    primary_rank: int
    replica_ranks: tuple[int, ...]
    synchronization_group_id: str

    @classmethod
    def create(
        cls,
        *,
        kind: ParameterOwnershipKind,
        tp_shard: int,
        expert_id: int | None,
        replica_ranks: tuple[int, ...],
        synchronization_group_id: str,
    ) -> "ParameterOwnershipDomain":
        primary_rank = min(replica_ranks) if replica_ranks else -1
        semantic = {
            "kind": kind,
            "tp_shard": tp_shard,
            "expert_id": expert_id,
            "primary_rank": primary_rank,
            "replica_ranks": replica_ranks,
            "synchronization_group_id": synchronization_group_id,
        }
        result = cls(
            schema_version=PARAMETER_OWNERSHIP_SCHEMA_VERSION,
            id=stable_artifact_id(
                "parameter_ownership_domain",
                semantic,
                schema_version=PARAMETER_OWNERSHIP_SCHEMA_VERSION,
            ),
            kind=kind,
            tp_shard=tp_shard,
            expert_id=expert_id,
            primary_rank=primary_rank,
            replica_ranks=replica_ranks,
            synchronization_group_id=synchronization_group_id,
        )
        result.validate()
        return result

    def validate(self, path: str = "parameter_ownership") -> None:
        if self.schema_version != PARAMETER_OWNERSHIP_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.id, f"{path}.id")
        if type(self.kind) is not ParameterOwnershipKind:
            raise SchemaError("must be a ParameterOwnershipKind", path=f"{path}.kind")
        if type(self.tp_shard) is not int or self.tp_shard < 0:
            raise SchemaError("must be a non-negative int", path=f"{path}.tp_shard")
        if self.kind is ParameterOwnershipKind.SHARED:
            if self.expert_id is not None:
                raise SchemaError(
                    "shared ownership must not name an expert",
                    path=f"{path}.expert_id",
                )
        elif type(self.expert_id) is not int or self.expert_id < 0:
            raise SchemaError(
                "expert ownership requires a non-negative expert id",
                path=f"{path}.expert_id",
            )
        if (
            type(self.replica_ranks) is not tuple
            or not self.replica_ranks
            or len(set(self.replica_ranks)) != len(self.replica_ranks)
            or any(type(rank) is not int or rank < 0 for rank in self.replica_ranks)
        ):
            raise SchemaError(
                "must contain unique non-negative logical ranks",
                path=f"{path}.replica_ranks",
            )
        if self.primary_rank != min(self.replica_ranks):
            raise SchemaError(
                "must be the lowest replica rank", path=f"{path}.primary_rank"
            )
        validate_nonempty(
            self.synchronization_group_id,
            f"{path}.synchronization_group_id",
        )
        semantic = {
            "kind": self.kind,
            "tp_shard": self.tp_shard,
            "expert_id": self.expert_id,
            "primary_rank": self.primary_rank,
            "replica_ranks": self.replica_ranks,
            "synchronization_group_id": self.synchronization_group_id,
        }
        expected = stable_artifact_id(
            "parameter_ownership_domain",
            semantic,
            schema_version=PARAMETER_OWNERSHIP_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable ownership id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class ParallelPlacement:
    schema_version: str
    workload_kind: ParallelWorkloadKind
    mesh: RectMeshSpec
    tp_degree: int
    dp_degree: int
    ep_degree: int
    pp_degree: int
    num_experts: int
    rank_placements: tuple[ParallelRankPlacement, ...]
    groups: tuple[ParallelGroup, ...]
    ownership_domains: tuple[ParameterOwnershipDomain, ...]

    @property
    def logical_rank_count(self) -> int:
        return self.tp_degree * self.dp_degree * self.ep_degree * self.pp_degree

    @property
    def active_die_ids(self) -> tuple[int, ...]:
        return tuple(placement.die_id for placement in self.rank_placements)

    @property
    def idle_die_ids(self) -> tuple[int, ...]:
        active = set(self.active_die_ids)
        return tuple(die_id for die_id in range(self.mesh.rank_count) if die_id not in active)

    @property
    def routing_die_ids(self) -> tuple[int, ...]:
        """All physical Dies remain routing participants, including idle Dies."""

        return tuple(range(self.mesh.rank_count))

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def select_groups(self, kind: ParallelGroupKind) -> tuple[ParallelGroup, ...]:
        if type(kind) is not ParallelGroupKind:
            raise SchemaError("must be a ParallelGroupKind", path="group_kind")
        return tuple(group for group in self.groups if group.kind is kind)

    def placement_for_rank(self, logical_rank: int) -> ParallelRankPlacement:
        if type(logical_rank) is not int or not 0 <= logical_rank < len(self.rank_placements):
            raise SchemaError("logical rank is not mapped", path="logical_rank")
        by_rank = {item.logical_rank: item for item in self.rank_placements}
        if logical_rank not in by_rank:
            raise SchemaError("logical rank is not mapped", path="logical_rank")
        return by_rank[logical_rank]

    def validate(self, path: str = "parallel_placement") -> None:
        if self.schema_version != PARALLEL_PLACEMENT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.workload_kind) is not ParallelWorkloadKind:
            raise SchemaError(
                "must be a ParallelWorkloadKind", path=f"{path}.workload_kind"
            )
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        for name in ("tp_degree", "dp_degree", "ep_degree", "pp_degree"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SchemaError("must be a positive int", path=f"{path}.{name}")
        if self.pp_degree != 1:
            raise SchemaError("v1 requires PP=1", path=f"{path}.pp_degree")
        if self.workload_kind is ParallelWorkloadKind.DENSE:
            if self.ep_degree != 1:
                raise SchemaError("Dense requires EP=1", path=f"{path}.ep_degree")
            if type(self.num_experts) is not int or self.num_experts != 0:
                raise SchemaError("Dense requires zero experts", path=f"{path}.num_experts")
        else:
            if type(self.num_experts) is not int or self.num_experts <= 0:
                raise SchemaError(
                    "MoE requires a positive expert count", path=f"{path}.num_experts"
                )
            if self.num_experts % self.ep_degree != 0:
                raise SchemaError(
                    "expert count must be divisible by EP degree",
                    path=f"{path}.num_experts",
                )
        if self.logical_rank_count > self.mesh.rank_count:
            raise SchemaError(
                "logical rank count exceeds physical Die count",
                path=f"{path}.rank_placements",
            )
        self._validate_rank_placements(path)
        group_by_id = self._validate_groups(path)
        self._validate_ownership(group_by_id, path)

    def _validate_rank_placements(self, path: str) -> None:
        if type(self.rank_placements) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.rank_placements")
        if len(self.rank_placements) != self.logical_rank_count:
            raise SchemaError(
                "must map every logical rank exactly once",
                path=f"{path}.rank_placements",
            )
        for index, item in enumerate(self.rank_placements):
            if type(item) is not ParallelRankPlacement:
                raise SchemaError(
                    "must be a ParallelRankPlacement",
                    path=f"{path}.rank_placements[{index}]",
                )
            item.validate(f"{path}.rank_placements[{index}]")
        ranks = tuple(item.logical_rank for item in self.rank_placements)
        if tuple(sorted(ranks)) != tuple(range(self.logical_rank_count)):
            raise SchemaError(
                "logical ranks must be complete and unique",
                path=f"{path}.rank_placements",
            )
        die_ids = tuple(item.die_id for item in self.rank_placements)
        if len(set(die_ids)) != len(die_ids):
            raise SchemaError(
                "v1 permits at most one workload rank per Die",
                path=f"{path}.rank_placements",
            )
        if any(die_id >= self.mesh.rank_count for die_id in die_ids):
            raise SchemaError(
                "Die lies outside the physical Mesh",
                path=f"{path}.rank_placements",
            )
        expected_coordinates = {
            LogicalRankCoordinate(dp=dp, ep=ep, tp=tp, pp=pp)
            for pp in range(self.pp_degree)
            for dp in range(self.dp_degree)
            for ep in range(self.ep_degree)
            for tp in range(self.tp_degree)
        }
        coordinates = {item.coordinate for item in self.rank_placements}
        if coordinates != expected_coordinates or len(coordinates) != len(self.rank_placements):
            raise SchemaError(
                "logical coordinates must cover the TP/DP/EP/PP product exactly",
                path=f"{path}.rank_placements",
            )

    def _validate_groups(self, path: str) -> dict[str, ParallelGroup]:
        if type(self.groups) is not tuple or not self.groups:
            raise SchemaError("must contain canonical groups", path=f"{path}.groups")
        group_by_id: dict[str, ParallelGroup] = {}
        group_keys: set[tuple[ParallelGroupKind, int]] = set()
        for index, group in enumerate(self.groups):
            if type(group) is not ParallelGroup:
                raise SchemaError("must be a ParallelGroup", path=f"{path}.groups[{index}]")
            group.validate(f"{path}.groups[{index}]")
            if group.id in group_by_id or (group.kind, group.index) in group_keys:
                raise SchemaError("group ids and keys must be unique", path=f"{path}.groups")
            if any(rank >= self.logical_rank_count for rank in group.ranks):
                raise SchemaError(
                    "group contains an unmapped logical rank",
                    path=f"{path}.groups[{index}].ranks",
                )
            group_by_id[group.id] = group
            group_keys.add((group.kind, group.index))
        expected = _build_groups(self.rank_placements, self.workload_kind)
        actual_semantics = tuple((g.kind, g.index, g.ranks) for g in self.groups)
        expected_semantics = tuple((g.kind, g.index, g.ranks) for g in expected)
        if actual_semantics != expected_semantics:
            raise SchemaError(
                "groups do not match the canonical logical mapping",
                path=f"{path}.groups",
            )
        return group_by_id

    def _validate_ownership(
        self, group_by_id: dict[str, ParallelGroup], path: str
    ) -> None:
        if type(self.ownership_domains) is not tuple or not self.ownership_domains:
            raise SchemaError(
                "must contain parameter ownership domains",
                path=f"{path}.ownership_domains",
            )
        seen: set[tuple[ParameterOwnershipKind, int, int | None]] = set()
        for index, owner in enumerate(self.ownership_domains):
            owner_path = f"{path}.ownership_domains[{index}]"
            if type(owner) is not ParameterOwnershipDomain:
                raise SchemaError("must be a ParameterOwnershipDomain", path=owner_path)
            owner.validate(owner_path)
            key = (owner.kind, owner.tp_shard, owner.expert_id)
            if key in seen:
                raise SchemaError("ownership domain is duplicated", path=owner_path)
            seen.add(key)
            if owner.tp_shard >= self.tp_degree:
                raise SchemaError("TP shard is out of range", path=f"{owner_path}.tp_shard")
            if owner.kind is ParameterOwnershipKind.EXPERT:
                if self.workload_kind is not ParallelWorkloadKind.MOE:
                    raise SchemaError(
                        "Dense placement cannot own expert parameters", path=owner_path
                    )
                assert owner.expert_id is not None
                if owner.expert_id >= self.num_experts:
                    raise SchemaError(
                        "expert id is out of range", path=f"{owner_path}.expert_id"
                    )
            group = group_by_id.get(owner.synchronization_group_id)
            if group is None:
                raise SchemaError(
                    "synchronization group is not present",
                    path=f"{owner_path}.synchronization_group_id",
                )
            expected_kind = (
                ParallelGroupKind.SHARED_PARAMETER_SYNC
                if owner.kind is ParameterOwnershipKind.SHARED
                else ParallelGroupKind.EXPERT_GRADIENT_SYNC
            )
            if group.kind is not expected_kind or group.ranks != owner.replica_ranks:
                raise SchemaError(
                    "owner replicas must exactly match its synchronization group",
                    path=owner_path,
                )
        expected = _build_ownership_domains(
            self.rank_placements,
            self.groups,
            self.workload_kind,
            self.tp_degree,
            self.ep_degree,
            self.num_experts,
        )
        actual_semantics = tuple(
            (o.kind, o.tp_shard, o.expert_id, o.replica_ranks, o.synchronization_group_id)
            for o in self.ownership_domains
        )
        expected_semantics = tuple(
            (o.kind, o.tp_shard, o.expert_id, o.replica_ranks, o.synchronization_group_id)
            for o in expected
        )
        if actual_semantics != expected_semantics:
            raise SchemaError(
                "ownership domains do not match the canonical logical mapping",
                path=f"{path}.ownership_domains",
            )


def _rank_by_coordinate(
    placements: tuple[ParallelRankPlacement, ...],
) -> dict[LogicalRankCoordinate, int]:
    return {item.coordinate: item.logical_rank for item in placements}


def _build_groups(
    placements: tuple[ParallelRankPlacement, ...],
    workload_kind: ParallelWorkloadKind,
) -> tuple[ParallelGroup, ...]:
    by_coordinate = _rank_by_coordinate(placements)
    dp_degree = 1 + max(coord.dp for coord in by_coordinate)
    ep_degree = 1 + max(coord.ep for coord in by_coordinate)
    tp_degree = 1 + max(coord.tp for coord in by_coordinate)
    pp_degree = 1 + max(coord.pp for coord in by_coordinate)
    groups: list[ParallelGroup] = []

    def add(kind: ParallelGroupKind, ranks: tuple[int, ...]) -> None:
        kind_index = sum(group.kind is kind for group in groups)
        groups.append(ParallelGroup.create(kind=kind, index=kind_index, ranks=ranks))

    add(ParallelGroupKind.MODEL, tuple(range(len(placements))))
    for pp in range(pp_degree):
        for dp in range(dp_degree):
            for ep in range(ep_degree):
                add(
                    ParallelGroupKind.TP,
                    tuple(
                        by_coordinate[LogicalRankCoordinate(dp, ep, tp, pp)]
                        for tp in range(tp_degree)
                    ),
                )
    for pp in range(pp_degree):
        for ep in range(ep_degree):
            for tp in range(tp_degree):
                add(
                    ParallelGroupKind.DP,
                    tuple(
                        by_coordinate[LogicalRankCoordinate(dp, ep, tp, pp)]
                        for dp in range(dp_degree)
                    ),
                )
    if workload_kind is ParallelWorkloadKind.MOE:
        for pp in range(pp_degree):
            for dp in range(dp_degree):
                for tp in range(tp_degree):
                    add(
                        ParallelGroupKind.EP,
                        tuple(
                            by_coordinate[LogicalRankCoordinate(dp, ep, tp, pp)]
                            for ep in range(ep_degree)
                        ),
                    )
    for tp in range(tp_degree):
        add(
            ParallelGroupKind.SHARED_PARAMETER_SYNC,
            tuple(
                by_coordinate[LogicalRankCoordinate(dp, ep, tp, pp)]
                for pp in range(pp_degree)
                for dp in range(dp_degree)
                for ep in range(ep_degree)
            ),
        )
    if workload_kind is ParallelWorkloadKind.MOE:
        # One group per (EP partition, TP shard); experts assigned to that EP
        # partition reference the same replica group but retain distinct owner IDs.
        for ep in range(ep_degree):
            for tp in range(tp_degree):
                add(
                    ParallelGroupKind.EXPERT_GRADIENT_SYNC,
                    tuple(
                        by_coordinate[LogicalRankCoordinate(dp, ep, tp, pp)]
                        for pp in range(pp_degree)
                        for dp in range(dp_degree)
                    ),
                )
    return tuple(groups)


def _build_ownership_domains(
    placements: tuple[ParallelRankPlacement, ...],
    groups: tuple[ParallelGroup, ...],
    workload_kind: ParallelWorkloadKind,
    tp_degree: int,
    ep_degree: int,
    num_experts: int,
) -> tuple[ParameterOwnershipDomain, ...]:
    shared_groups = tuple(
        group for group in groups if group.kind is ParallelGroupKind.SHARED_PARAMETER_SYNC
    )
    owners = [
        ParameterOwnershipDomain.create(
            kind=ParameterOwnershipKind.SHARED,
            tp_shard=tp,
            expert_id=None,
            replica_ranks=shared_groups[tp].ranks,
            synchronization_group_id=shared_groups[tp].id,
        )
        for tp in range(tp_degree)
    ]
    if workload_kind is ParallelWorkloadKind.MOE:
        expert_groups = tuple(
            group
            for group in groups
            if group.kind is ParallelGroupKind.EXPERT_GRADIENT_SYNC
        )
        experts_per_partition = num_experts // ep_degree
        for expert_id in range(num_experts):
            ep = expert_id // experts_per_partition
            for tp in range(tp_degree):
                group = expert_groups[ep * tp_degree + tp]
                owners.append(
                    ParameterOwnershipDomain.create(
                        kind=ParameterOwnershipKind.EXPERT,
                        tp_shard=tp,
                        expert_id=expert_id,
                        replica_ranks=group.ranks,
                        synchronization_group_id=group.id,
                    )
                )
    return tuple(owners)


def _build_parallel_placement(
    *,
    mesh: RectMeshSpec,
    workload_kind: ParallelWorkloadKind,
    tp_degree: int,
    dp_degree: int,
    ep_degree: int,
    pp_degree: int,
    num_experts: int,
    active_die_ids: tuple[int, ...] | None,
    local_core_ids: tuple[int, ...] | None,
) -> ParallelPlacement:
    mesh.validate("parallel_placement.mesh")
    for name, value in (
        ("tp_degree", tp_degree),
        ("dp_degree", dp_degree),
        ("ep_degree", ep_degree),
        ("pp_degree", pp_degree),
    ):
        if type(value) is not int or value <= 0:
            raise SchemaError("must be a positive int", path=name)
    if pp_degree != 1:
        raise SchemaError("v1 requires PP=1", path="pp_degree")
    if workload_kind is ParallelWorkloadKind.DENSE:
        if ep_degree != 1 or num_experts != 0:
            raise SchemaError(
                "Dense requires EP=1 and zero experts", path="num_experts"
            )
    elif (
        type(num_experts) is not int
        or num_experts <= 0
        or num_experts % ep_degree != 0
    ):
        raise SchemaError(
            "MoE expert count must be positive and divisible by EP degree",
            path="num_experts",
        )
    rank_count = tp_degree * dp_degree * ep_degree * pp_degree
    if rank_count > mesh.rank_count:
        raise SchemaError(
            "logical rank count exceeds physical Die count",
            path="active_die_ids",
        )
    if active_die_ids is None:
        active_die_ids = tuple(range(rank_count))
    if type(active_die_ids) is not tuple or len(active_die_ids) != rank_count:
        raise SchemaError(
            "must contain one Die id per logical rank", path="active_die_ids"
        )
    if local_core_ids is None:
        local_core_ids = (0,) * rank_count
    if type(local_core_ids) is not tuple or len(local_core_ids) != rank_count:
        raise SchemaError(
            "must contain one local core id per logical rank", path="local_core_ids"
        )
    placements: list[ParallelRankPlacement] = []
    logical_rank = 0
    for pp in range(pp_degree):
        for dp in range(dp_degree):
            for ep in range(ep_degree):
                for tp in range(tp_degree):
                    placements.append(
                        ParallelRankPlacement(
                            logical_rank=logical_rank,
                            coordinate=LogicalRankCoordinate(dp=dp, ep=ep, tp=tp, pp=pp),
                            die_id=active_die_ids[logical_rank],
                            local_core=local_core_ids[logical_rank],
                        )
                    )
                    logical_rank += 1
    placement_tuple = tuple(placements)
    groups = _build_groups(placement_tuple, workload_kind)
    owners = _build_ownership_domains(
        placement_tuple,
        groups,
        workload_kind,
        tp_degree,
        ep_degree,
        num_experts,
    )
    result = ParallelPlacement(
        schema_version=PARALLEL_PLACEMENT_SCHEMA_VERSION,
        workload_kind=workload_kind,
        mesh=mesh,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        ep_degree=ep_degree,
        pp_degree=pp_degree,
        num_experts=num_experts,
        rank_placements=placement_tuple,
        groups=groups,
        ownership_domains=owners,
    )
    result.validate()
    return result


def build_dense_parallel_placement(
    mesh: RectMeshSpec,
    *,
    tp_degree: int,
    dp_degree: int,
    pp_degree: int = 1,
    active_die_ids: tuple[int, ...] | None = None,
    local_core_ids: tuple[int, ...] | None = None,
) -> ParallelPlacement:
    """Build a canonical Dense TP x DP placement on selected physical Dies."""

    return _build_parallel_placement(
        mesh=mesh,
        workload_kind=ParallelWorkloadKind.DENSE,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        ep_degree=1,
        pp_degree=pp_degree,
        num_experts=0,
        active_die_ids=active_die_ids,
        local_core_ids=local_core_ids,
    )


def build_moe_parallel_placement(
    mesh: RectMeshSpec,
    *,
    tp_degree: int,
    ep_degree: int,
    dp_degree: int,
    num_experts: int,
    pp_degree: int = 1,
    active_die_ids: tuple[int, ...] | None = None,
    local_core_ids: tuple[int, ...] | None = None,
) -> ParallelPlacement:
    """Build canonical MoE TP x EP x DP placement and ownership domains."""

    return _build_parallel_placement(
        mesh=mesh,
        workload_kind=ParallelWorkloadKind.MOE,
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        ep_degree=ep_degree,
        pp_degree=pp_degree,
        num_experts=num_experts,
        active_die_ids=active_die_ids,
        local_core_ids=local_core_ids,
    )


__all__ = [
    "PARALLEL_GROUP_SCHEMA_VERSION",
    "PARALLEL_PLACEMENT_SCHEMA_VERSION",
    "PARAMETER_OWNERSHIP_SCHEMA_VERSION",
    "LogicalRankCoordinate",
    "ParallelGroup",
    "ParallelGroupKind",
    "ParallelPlacement",
    "ParallelRankPlacement",
    "ParallelWorkloadKind",
    "ParameterOwnershipDomain",
    "ParameterOwnershipKind",
    "build_dense_parallel_placement",
    "build_moe_parallel_placement",
]
