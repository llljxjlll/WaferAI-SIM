"""Source-bound expert RETURN layout for signed top1 weighted combine.

Group the real P2 expert slots physically; the frozen per-token route table
selects a rank-major return segment and slot without inventing extra flows.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..lowering.flexible_moe_multi_production import (
    expert_projection_action_ids,
)
from .moe_full_train_named_wgrad_operands import (
    _buffer_slice, _one_record, _operand_slice,
)
from .moe_full_train_router_score_source import (
    MoeTrainableRouterScorePath, MoeTrainableSignedRouterRequirements,
)


_VERSION = "wafer_frontend.moe_signed_router_return_protocol/v1alpha1"


@dataclass(frozen=True, slots=True)
class MoeRouterExpertReturnSegment:
    expert_home_rank: int
    expert_index: int
    offset_bytes: int
    size_bytes: int
    assignment_refs: tuple[str, ...]
    combine_flow_ref: str | None


@dataclass(frozen=True, slots=True)
class MoeRouterReturnLane:
    token_index: int
    expert_home_rank: int
    expert_index: int
    expert_slot_index: int
    signed_score_offset_bytes: int
    expert_output_offset_bytes: int
    combined_output_offset_bytes: int


@dataclass(frozen=True, slots=True)
class MoeRouterSourceReturnPlacement:
    step: int
    layer: int
    source_rank: int
    score_tape_bytes: int
    expert_return_bytes: int
    combined_output_bytes: int
    segments: tuple[MoeRouterExpertReturnSegment, ...]
    lanes: tuple[MoeRouterReturnLane, ...]


@dataclass(frozen=True, slots=True)
class MoeRouterSignedReturnProtocol:
    id: str
    source_router_score_ref: str
    source_moe_sequence_ref: str
    version: str
    placements: tuple[MoeRouterSourceReturnPlacement, ...]

    def validate_against(self, score: MoeTrainableSignedRouterRequirements,
                         sequence: MoeCompileSequence) -> None:
        if self != build_moe_router_signed_return_protocol(score, sequence):
            raise SchemaError("route table, EP return bytes and P2 COMBINE flow must be rebuilt from real source",
                              path="moe_router_signed_return")

    def require_physical_return_transport(
        self, score: MoeTrainableSignedRouterRequirements,
        sequence: MoeCompileSequence,
    ) -> None:
        """Reject RETURN flows that overlap local expert bytes or alias tape."""
        self.validate_against(score, sequence)
        units = {(unit.step, unit.layer): unit for unit in sequence.units}
        for placement in self.placements:
            unit = units[placement.step, placement.layer]
            manifest = unit.linked_manifest
            try:
                tape = _buffer_slice(
                    manifest, placement.source_rank,
                    ".expert_return_tape", DType.FP16, 0,
                    placement.expert_return_bytes)
            except SchemaError as exc:
                raise SchemaError("expert RETURN lacks one source-owned rank-major physical SRAM tape",
                                  path=f"router_return.step{placement.step}.layer{placement.layer}") from exc
            for segment in placement.segments:
                root = _buffer_slice(
                    manifest, placement.source_rank, ".expert_return_tape",
                    DType.FP16, segment.offset_bytes, segment.size_bytes)
                if segment.combine_flow_ref is None:
                    expert = next((action for action in unit.plan.actions
                                   if action.kind is MoeRectActionKind.EXPERT_FORWARD
                                   and action.rank == segment.expert_home_rank
                                   and action.assignment_refs == segment.assignment_refs),
                                  None)
                    if expert is None:
                        raise SchemaError("local source expert has no exact P2 forward owner",
                                          path=f"router_return.step{placement.step}")
                    down_id = expert_projection_action_ids(
                        unit.plan.id, expert.id)[2]
                    _, fragment, stream, index = _one_record(
                        manifest, placement.source_rank, down_id,
                        (RecordOpcode.MATMUL,))
                    actual = _operand_slice(
                        manifest, fragment, stream, index,
                        "output_address", segment.size_bytes,
                        require_exact_view=True)
                else:
                    receiver = next((action for action in unit.plan.actions
                                     if action.kind is MoeRectActionKind.RECV
                                     and action.flow_ref == segment.combine_flow_ref),
                                    None)
                    if receiver is None or receiver.rank != placement.source_rank:
                        raise SchemaError("remote expert RETURN has no exact source P2 receiving action",
                                          path=f"router_return.step{placement.step}")
                    _, fragment, stream, index = _one_record(
                        manifest, placement.source_rank, receiver.id,
                        (RecordOpcode.DTE_RECV,))
                    actual = _operand_slice(
                        manifest, fragment, stream, index,
                        "destination_address", segment.size_bytes,
                        require_exact_view=True)
                if actual != root or root.buffer_abi_ref != tape.buffer_abi_ref:
                    raise SchemaError("real DTE/local expert RETURN must land in disjoint slot/home SRAM tape spans",
                                      path=f"router_return.step{placement.step}.layer{placement.layer}")


def _placement(path: MoeTrainableRouterScorePath, unit) -> MoeRouterSourceReturnPlacement:
    if not path.routes:
        raise SchemaError("zero source tokens cannot create fake router return work",
                          path="moe_router_return.source")
    spec, plan = unit.spec, unit.plan
    if path.hidden_size != spec.hidden_size or path.expert_count != spec.expert_count:
        raise SchemaError("router return dimensions differ from source P2",
                          path="moe_router_return.source")
    row_bytes = 2 * path.hidden_size
    groups = sorted({(row.expert_home_rank, row.selected_expert)
                     for row in path.routes})
    segments = []
    group_base = {}
    offset = 0
    for home, expert_index in groups:
        chosen = tuple(row for row in path.routes
                       if (row.expert_home_rank, row.selected_expert)
                       == (home, expert_index))
        slots = tuple(row.expert_slot_index for row in chosen)
        if slots != tuple(range(len(chosen))):
            raise SchemaError("actual expert slot indices must be canonical contiguous within home/expert",
                              path=f"router_return.step{path.step}.layer{path.layer}")
        refs = tuple(f"assignment.{row.token_index}" for row in chosen)
        flows = [flow for flow in plan.flows
                 if flow.stage is MoeRectFlowStage.COMBINE
                 and flow.source_rank == home
                 and flow.destination_rank == path.source_rank
                 and flow.assignment_refs == refs]
        if home == path.source_rank:
            if flows:
                raise SchemaError("local expert RETURN cannot masquerade as DTE transport",
                                  path=f"router_return.step{path.step}")
            flow_ref = None
        elif len(flows) != 1 or flows[0].logical_bytes != len(chosen) * row_bytes:
            raise SchemaError("remote expert RETURN must have one exact grouped P2 flow and full bytes",
                              path=f"router_return.step{path.step}")
        else:
            flow_ref = flows[0].id
        segments.append(MoeRouterExpertReturnSegment(
            home, expert_index, offset, len(chosen) * row_bytes, refs, flow_ref))
        group_base[(home, expert_index)] = offset
        offset += len(chosen) * row_bytes
    lanes = tuple(MoeRouterReturnLane(
        row.token_index, row.expert_home_rank, row.selected_expert,
        row.expert_slot_index,
        (index * path.expert_count + row.selected_expert) * 2,
        group_base[(row.expert_home_rank, row.selected_expert)]
        + row.expert_slot_index * row_bytes,
        index * row_bytes,
    ) for index, row in enumerate(path.routes))
    if (offset != path.forward_expert_bytes
            or len(lanes) != len(path.routes)
            or len(set(lane.expert_output_offset_bytes for lane in lanes))
               != len(lanes)
            or any(lane.expert_output_offset_bytes < 0
                   or lane.expert_output_offset_bytes + row_bytes > offset
                   for lane in lanes)):
        raise SchemaError("source expert RETURN staging omits or overlaps a selected token",
                          path=f"router_return.step{path.step}.layer{path.layer}")
    return MoeRouterSourceReturnPlacement(
        path.step, path.layer, path.source_rank, path.score_tape_bytes,
        offset, len(lanes) * row_bytes, tuple(segments), lanes)


def build_moe_router_signed_return_protocol(
    score: MoeTrainableSignedRouterRequirements,
    sequence: MoeCompileSequence,
) -> MoeRouterSignedReturnProtocol:
    score.validate_against(sequence)
    units = {(unit.step, unit.layer): unit for unit in sequence.units}
    placements = tuple(_placement(path, units[path.step, path.layer])
                       for path in score.paths if path.routes)
    semantic = {
        "source_router_score_ref": score.id,
        "source_moe_sequence_ref": sequence.id,
        "version": _VERSION,
        "placements": placements,
    }
    return MoeRouterSignedReturnProtocol(
        stable_artifact_id("moe_router_signed_return", semantic,
                           schema_version=_VERSION),
        **semantic,
    )


__all__ = ["MoeRouterExpertReturnSegment", "MoeRouterReturnLane",
           "MoeRouterSourceReturnPlacement", "MoeRouterSignedReturnProtocol",
           "build_moe_router_signed_return_protocol"]
