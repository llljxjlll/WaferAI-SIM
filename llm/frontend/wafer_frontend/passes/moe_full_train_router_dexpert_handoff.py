"""Source-backed dExpert ownership and physical router→expert backward handoff.

This contract only accepts a native score-backward record with two owned
outputs.  The current static production leaf fails its physical gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_named_wgrad_operands import (
    _buffer_slice, _one_record, _operand_slice,
)
from .moe_full_train_router_native_protocol import (
    MoeRouterSourceNativeProtocol, build_moe_router_native_protocol,
)
from .moe_full_train_router_return_protocol import (
    MoeRouterSignedReturnProtocol,
)
from .moe_full_train_router_score_source import (
    MoeTrainableSignedRouterRequirements, _router_record, _route_table,
)

_VERSION = "wafer_frontend.moe_router_dexpert_handoff/v1alpha1"


@dataclass(frozen=True, slots=True)
class MoeRouterExpertGradientSegment:
    step: int
    layer: int
    source_rank: int
    expert_home_rank: int
    expert_index: int
    source_offset_bytes: int
    size_bytes: int
    assignment_refs: tuple[str, ...]
    backward_flow_ref: str | None
    dgrad_action_ref: str
    send_action_ref: str | None
    recv_action_ref: str | None


@dataclass(frozen=True, slots=True)
class MoeRouterDexpertHandoff:
    id: str
    source_router_ref: str
    source_return_ref: str
    source_native_ref: str
    source_sequence_ref: str
    version: str
    segments: tuple[MoeRouterExpertGradientSegment, ...]

    def validate_against(self, score: MoeTrainableSignedRouterRequirements,
                         returned: MoeRouterSignedReturnProtocol,
                         native: MoeRouterSourceNativeProtocol,
                         sequence: MoeCompileSequence) -> None:
        if self != build_moe_router_dexpert_handoff(score, returned, native, sequence):
            raise SchemaError("expert dY route, P2 gradient flow or native dual-output source changed",
                              path="moe_router_dexpert_handoff")

    def require_physical_dexpert_consumption(
        self, score: MoeTrainableSignedRouterRequirements,
        returned: MoeRouterSignedReturnProtocol,
        native: MoeRouterSourceNativeProtocol,
        sequence: MoeCompileSequence,
    ) -> None:
        """Match true 0x28 output, transport bytes and first expert DGRAD dY.

        Both DTE endpoints must close onto independently allocated real SRAM
        BufferABI and exact relocation/tensor slices; no source-only proof can
        substitute for this physical gate.
        """
        self.validate_against(score, returned, native, sequence)
        backward_opcode = RecordOpcode._value2member_map_.get(0x28)
        if backward_opcode is None:
            raise SchemaError("native router score-backward public opcode 0x28 is not implemented",
                              path="moe_router_dexpert_handoff.physical")
        units = {(unit.step, unit.layer): unit for unit in sequence.units}
        path_by_key = {(path.step, path.layer, path.source_rank): path
                       for path in score.paths if path.routes}
        for pair in native.pairs:
            unit = units[pair.step, pair.layer]
            path = path_by_key[(pair.step, pair.layer, pair.source_rank)]
            manifest = unit.linked_manifest
            found = _router_record(manifest, pair.source_rank,
                                   path.combine_backward_action_ref)
            exact = [item for item in found if item[0].opcode is backward_opcode]
            if len(exact) != 1 or len(found) != 1:
                raise SchemaError("one real router backward dual-output record required",
                                  path=f"router_dexpert.step{pair.step}.layer{pair.layer}")
            record, fragment, stream, index = exact[0]
            _route_table(record, path)
            for name, suffix, size in (
                ("score_address", ".router_score_tape",
                 pair.score_backward.operand_bytes[0]),
                ("expert_return_address", ".expert_return_tape",
                 pair.score_backward.operand_bytes[1]),
                ("upstream_address", ".dcombined_gradient",
                 pair.score_backward.operand_bytes[2]),
                ("dscore_address", ".router_score_gradient",
                 pair.score_backward.operand_bytes[3]),
                ("dexpert_address", ".router_dexpert_gradient",
                 pair.score_backward.operand_bytes[4]),
            ):
                expected = _buffer_slice(manifest, pair.source_rank, suffix,
                                         DType.FP16, 0, size)
                if (_operand_slice(manifest, fragment, stream, index, name,
                                   size, require_exact_view=True) != expected):
                    raise SchemaError("dual-gradient native record has an unbound input/output or wrong physical tape",
                                      path=f"router_dexpert.step{pair.step}.layer{pair.layer}")
        for segment in self.segments:
            unit = units[(segment.step, segment.layer)]
            manifest = unit.linked_manifest
            source = _buffer_slice(manifest, segment.source_rank,
                                   ".router_dexpert_gradient", DType.FP16,
                                   segment.source_offset_bytes,
                                   segment.size_bytes)
            if segment.backward_flow_ref is None:
                if segment.expert_home_rank != segment.source_rank:
                    raise SchemaError("remote expert cannot consume local dExpert bytes",
                                      path="moe_router_dexpert_handoff.physical")
                target = source
            else:
                _, fsend, ssend, isend = _one_record(
                    manifest, segment.source_rank, segment.send_action_ref,
                    (RecordOpcode.DTE_SEND,))
                if (_operand_slice(manifest, fsend, ssend, isend,
                                   "source_address", segment.size_bytes,
                                   require_exact_view=True) != source):
                    raise SchemaError("BACKWARD_GRADIENT SEND must read the selected dExpert group, not raw dCombined",
                                      path=f"router_dexpert.step{segment.step}.layer{segment.layer}")
                target = _buffer_slice(manifest, segment.expert_home_rank,
                                       ".expert_weighted_upstream", DType.FP16,
                                       0, segment.size_bytes)
                _, frecv, srecv, irecv = _one_record(
                    manifest, segment.expert_home_rank,
                    segment.recv_action_ref, (RecordOpcode.DTE_RECV,))
                if (_operand_slice(manifest, frecv, srecv, irecv,
                                   "destination_address", segment.size_bytes,
                                   require_exact_view=True) != target):
                    raise SchemaError("BACKWARD_GRADIENT RECV must write real expert dY SRAM",
                                      path=f"router_dexpert.step{segment.step}.layer{segment.layer}")
            _, fdgrad, sdgrad, idgrad = _one_record(
                manifest, segment.expert_home_rank,
                segment.dgrad_action_ref, (RecordOpcode.MATMUL,))
            if (_operand_slice(manifest, fdgrad, sdgrad, idgrad,
                               "input_address", segment.size_bytes,
                               require_exact_view=True) != target):
                raise SchemaError("expert three-projection DGRAD down derivative consumes wrong dY",
                                  path=f"router_dexpert.step{segment.step}.layer{segment.layer}")


def build_moe_router_dexpert_handoff(
    score: MoeTrainableSignedRouterRequirements,
    returned: MoeRouterSignedReturnProtocol,
    native: MoeRouterSourceNativeProtocol,
    sequence: MoeCompileSequence,
) -> MoeRouterDexpertHandoff:
    score.validate_against(sequence)
    returned.validate_against(score, sequence)
    native.validate_against(score, returned, sequence)
    units = {(unit.step, unit.layer): unit for unit in sequence.units}
    paths = {(path.step, path.layer, path.source_rank): path
             for path in score.paths if path.routes}
    pairs = {(pair.step, pair.layer, pair.source_rank): pair
             for pair in native.pairs}
    segments = []
    for placement in returned.placements:
        key = (placement.step, placement.layer, placement.source_rank)
        path, pair = paths[key], pairs[key]
        if pair.score_backward.operand_bytes[4] != placement.expert_return_bytes:
            raise SchemaError("dual output must cover every grouped P2 expert dY row",
                              path="moe_router_dexpert_handoff.source")
        unit = units[(placement.step, placement.layer)]
        for group in placement.segments:
            dgrad = [action for action in unit.plan.actions
                     if action.kind is MoeRectActionKind.EXPERT_DGRAD
                     and action.rank == group.expert_home_rank
                     and action.assignment_refs == group.assignment_refs]
            flow = [item for item in unit.plan.flows
                    if item.stage is MoeRectFlowStage.BACKWARD_GRADIENT
                    and item.source_rank == placement.source_rank
                    and item.destination_rank == group.expert_home_rank
                    and item.assignment_refs == group.assignment_refs]
            if len(dgrad) != 1 or (group.expert_home_rank !=
                                   placement.source_rank and len(flow) != 1):
                raise SchemaError("one exact P2 expert DGRAD and remote source gradient flow required",
                                  path=f"router_dexpert.step{placement.step}.layer{placement.layer}")
            if group.expert_home_rank == placement.source_rank:
                if flow:
                    raise SchemaError("local expert must not fabricate gradient DTE flow",
                                      path="moe_router_dexpert_handoff.source")
                source_flow, send, recv = None, None, None
            else:
                source_flow = flow[0].id
                if flow[0].logical_bytes != group.size_bytes:
                    raise SchemaError("remote expert DGRAD gradient flow dropped real token bytes",
                                      path="moe_router_dexpert_handoff.source")
                send = [a for a in unit.plan.actions
                        if a.kind is MoeRectActionKind.SEND
                        and a.flow_ref == source_flow]
                recv = [a for a in unit.plan.actions
                        if a.kind is MoeRectActionKind.RECV
                        and a.flow_ref == source_flow]
                if len(send) != 1 or len(recv) != 1:
                    raise SchemaError("one exact P2 SEND→RECV action pair required",
                                      path="moe_router_dexpert_handoff.source")
                send, recv = send[0].id, recv[0].id
            segments.append(MoeRouterExpertGradientSegment(
                placement.step, placement.layer, placement.source_rank,
                group.expert_home_rank, group.expert_index,
                group.offset_bytes, group.size_bytes, group.assignment_refs,
                source_flow, dgrad[0].id, send, recv))
        if (sum(item.size_bytes for item in segments
                if (item.step, item.layer, item.source_rank) == key)
                != pair.score_backward.operand_bytes[4]
                or len(path.routes) != sum(len(group.assignment_refs)
                                          for group in placement.segments)):
            raise SchemaError("local+remote expert dY groups omit a signed route row",
                              path="moe_router_dexpert_handoff.source")
    semantic = {"source_router_ref": score.id,
                "source_return_ref": returned.id,
                "source_native_ref": native.id,
                "source_sequence_ref": sequence.id,
                "version": _VERSION,
                "segments": tuple(segments)}
    return MoeRouterDexpertHandoff(
        stable_artifact_id("moe_router_dexpert_handoff", semantic,
                           schema_version=_VERSION), **semantic)


__all__ = ["MoeRouterExpertGradientSegment", "MoeRouterDexpertHandoff",
           "build_moe_router_dexpert_handoff"]
