"""Strict timing workloads for dynamic signed-score top1 MoE routing.

Public 0x27/0x28 records are not available until a native bridge exists.
These source workloads specify five disjoint backward SRAM addresses, not
functional gradient numerics or a successful full-model training timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, stable_artifact_id


_VERSION = "wafer_frontend.moe_router_signed_weight_workload/v1alpha1"


@dataclass(frozen=True, slots=True)
class MoeRouterSignedRoute:
    token_index: int
    source_rank: int
    selected_expert: int
    expert_home_rank: int
    expert_slot_index: int


@dataclass(frozen=True, slots=True)
class MoeRouterSignedExpertGroup:
    expert_home_rank: int
    expert_index: int
    offset_bytes: int
    size_bytes: int


def _layout(
    rank_rows: int, hidden_size: int, expert_count: int, source_rank: int,
    routes: tuple[MoeRouterSignedRoute, ...],
    groups: tuple[MoeRouterSignedExpertGroup, ...],
    path: str,
) -> tuple[tuple[int, int, int], ...]:
    if (any(type(value) is not int or value <= 0
            for value in (rank_rows, hidden_size, expert_count))
            or type(source_rank) is not int or source_rank < 0
            or type(routes) is not tuple or len(routes) != rank_rows
            or type(groups) is not tuple or not groups):
        raise SchemaError("one positive rank-local source/hidden/expert route tile is required",
                          path=path)
    row_bytes = 2 * hidden_size
    groups_by_id = {}
    next_byte = 0
    for item in groups:
        if (type(item) is not MoeRouterSignedExpertGroup
                or type(item.expert_home_rank) is not int
                or item.expert_home_rank < 0
                or type(item.expert_index) is not int
                or item.expert_index < 0 or item.expert_index >= expert_count
                or type(item.offset_bytes) is not int
                or type(item.size_bytes) is not int
                or item.offset_bytes != next_byte
                or item.size_bytes <= 0
                or item.size_bytes % row_bytes):
            raise SchemaError("return expert groups must occupy contiguous distinct FP16 spans",
                              path=path)
        key = (item.expert_home_rank, item.expert_index)
        if key in groups_by_id:
            raise SchemaError("expert home/identity may appear in one return group",
                              path=path)
        groups_by_id[key] = item
        next_byte += item.size_bytes
    if next_byte != rank_rows * row_bytes:
        raise SchemaError("all source token expert outputs must occupy full H-sized SRAM tape",
                          path=path)
    indices, byte_lanes = set(), []
    slots = {key: set() for key in groups_by_id}
    for index, item in enumerate(routes):
        if (type(item) is not MoeRouterSignedRoute
                or type(item.token_index) is not int or item.token_index < 0
                or item.token_index in indices
                or (index > 0 and item.token_index <= routes[index - 1].token_index)
                or item.source_rank != source_rank
                or type(item.expert_slot_index) is not int
                or item.expert_slot_index < 0
                or type(item.selected_expert) is not int
                or item.selected_expert < 0 or item.selected_expert >= expert_count):
            raise SchemaError("route table must derive one ordered frozen selected expert per source token",
                              path=path)
        key = (item.expert_home_rank, item.selected_expert)
        group = groups_by_id.get(key)
        if (group is None
                or item.expert_slot_index in slots[key]
                or item.expert_slot_index * row_bytes >= group.size_bytes):
            raise SchemaError("route slot must select one actual expert RETURN row, with no overlap",
                              path=path)
        slots[key].add(item.expert_slot_index)
        indices.add(item.token_index)
        byte_lanes.append((
            (index * expert_count + item.selected_expert) * 2,
            group.offset_bytes + item.expert_slot_index * row_bytes,
            index * row_bytes,
        ))
    if any(slots[key] != set(range(group.size_bytes // row_bytes))
           for key, group in groups_by_id.items()):
        raise SchemaError("no selected expert slot may be dropped or invented",
                          path=path)
    return tuple(byte_lanes)


@dataclass(frozen=True, slots=True)
class MoeRouterSignedWeightedForwardWorkload:
    id: str
    source_dynamic_case_ref: str
    source_gate_action_ref: str
    source_weighted_combine_action_ref: str
    rank_rows: int
    hidden_size: int
    expert_count: int
    source_rank: int
    routes: tuple[MoeRouterSignedRoute, ...]
    expert_groups: tuple[MoeRouterSignedExpertGroup, ...]
    score_dtype: DType
    expert_dtype: DType
    combined_dtype: DType

    @classmethod
    def create(cls, **semantic):
        value = cls(
            stable_artifact_id("moe_router_signed_weight_forward", semantic,
                               schema_version=_VERSION), **semantic)
        value.validate()
        return value

    def validate(self, path="moe_router_signed_weight_forward") -> None:
        for name in ("source_dynamic_case_ref", "source_gate_action_ref",
                     "source_weighted_combine_action_ref"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise SchemaError("forward source case/GATE/WEIGHTED_COMBINE must be explicit",
                                  path=f"{path}.{name}")
        if (self.score_dtype, self.expert_dtype, self.combined_dtype) != (
                DType.FP16, DType.FP16, DType.FP16):
            raise SchemaError("signed router forward must read and write actual FP16 SRAM bytes",
                              path=path)
        self.byte_lanes(path)
        semantic = {name: getattr(self, name)
                    for name in self.__dataclass_fields__ if name != "id"}
        if self.id != stable_artifact_id(
                "moe_router_signed_weight_forward", semantic,
                schema_version=_VERSION):
            raise SchemaError("forward workload source signature changed",
                              path=f"{path}.id")

    def byte_lanes(self, path="moe_router_signed_weight_forward"):
        return _layout(self.rank_rows, self.hidden_size, self.expert_count,
                       self.source_rank, self.routes, self.expert_groups, path)

    @property
    def score_bytes(self):
        return 2 * self.rank_rows * self.expert_count

    @property
    def expert_return_bytes(self):
        return 2 * self.rank_rows * self.hidden_size

    @property
    def combined_bytes(self):
        return self.expert_return_bytes

    @property
    def logical_flops(self):
        return 2 * self.rank_rows * self.hidden_size


@dataclass(frozen=True, slots=True)
class MoeRouterSignedScoreBackwardWorkload:
    id: str
    source_weighted_forward_ref: str
    source_combine_backward_action_ref: str
    rank_rows: int
    hidden_size: int
    expert_count: int
    source_rank: int
    routes: tuple[MoeRouterSignedRoute, ...]
    expert_groups: tuple[MoeRouterSignedExpertGroup, ...]
    score_dtype: DType
    expert_dtype: DType
    upstream_dtype: DType
    dscore_dtype: DType
    dexpert_dtype: DType

    @classmethod
    def create(cls, **semantic):
        value = cls(
            stable_artifact_id("moe_router_signed_score_backward", semantic,
                               schema_version=_VERSION), **semantic)
        value.validate()
        return value

    def validate(self, path="moe_router_signed_score_backward") -> None:
        if (type(self.source_weighted_forward_ref) is not str
                or not self.source_weighted_forward_ref
                or type(self.source_combine_backward_action_ref) is not str
                or not self.source_combine_backward_action_ref):
            raise SchemaError("dScore/dExpert must depend on one real weighted forward and source backward action",
                              path=path)
        if (self.score_dtype, self.expert_dtype, self.upstream_dtype,
                self.dscore_dtype, self.dexpert_dtype) != (DType.FP16,) * 5:
            raise SchemaError("backward has three FP16 input and two different FP16 output SRAM operands",
                              path=path)
        self.byte_lanes(path)
        semantic = {name: getattr(self, name)
                    for name in self.__dataclass_fields__ if name != "id"}
        if self.id != stable_artifact_id(
                "moe_router_signed_score_backward", semantic,
                schema_version=_VERSION):
            raise SchemaError("backward workload source signature changed",
                              path=f"{path}.id")

    def byte_lanes(self, path="moe_router_signed_score_backward"):
        return _layout(self.rank_rows, self.hidden_size, self.expert_count,
                       self.source_rank, self.routes, self.expert_groups, path)

    @property
    def operand_bytes(self):
        score_bytes = 2 * self.rank_rows * self.expert_count
        hidden_bytes = 2 * self.rank_rows * self.hidden_size
        return (score_bytes, hidden_bytes, hidden_bytes,
                score_bytes, hidden_bytes)

    @property
    def logical_flops(self):
        # 2H for the selected expert dot-product, H to scale dExpert.
        return 3 * self.rank_rows * self.hidden_size


__all__ = ["MoeRouterSignedRoute", "MoeRouterSignedExpertGroup",
           "MoeRouterSignedWeightedForwardWorkload",
           "MoeRouterSignedScoreBackwardWorkload"]
