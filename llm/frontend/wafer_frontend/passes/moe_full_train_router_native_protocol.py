"""Bind source-signed score and expert RETURN groups to router native work."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.moe_router_signed_weight_workload import (
    MoeRouterSignedExpertGroup, MoeRouterSignedRoute,
    MoeRouterSignedScoreBackwardWorkload,
    MoeRouterSignedWeightedForwardWorkload,
)
from .moe_full_train_router_return_protocol import (
    MoeRouterSignedReturnProtocol,
    build_moe_router_signed_return_protocol,
)
from .moe_full_train_router_score_source import (
    MoeTrainableSignedRouterRequirements,
)


_VERSION = "wafer_frontend.moe_router_native_protocol/v1alpha1"


@dataclass(frozen=True, slots=True)
class MoeRouterNativeSourcePair:
    step: int
    layer: int
    source_rank: int
    weighted_forward: MoeRouterSignedWeightedForwardWorkload
    score_backward: MoeRouterSignedScoreBackwardWorkload


@dataclass(frozen=True, slots=True)
class MoeRouterSourceNativeProtocol:
    id: str
    source_dynamic_score_ref: str
    source_return_protocol_ref: str
    source_moe_sequence_ref: str
    version: str
    pairs: tuple[MoeRouterNativeSourcePair, ...]

    def validate_against(self, score: MoeTrainableSignedRouterRequirements,
                         returned: MoeRouterSignedReturnProtocol,
                         sequence: MoeCompileSequence) -> None:
        if self != build_moe_router_native_protocol(score, returned, sequence):
            raise SchemaError("router FWD/BWD op geometry/dScore+dExpert ownership changed from source P2",
                              path="moe_router_native_protocol")


def build_moe_router_native_protocol(
    score: MoeTrainableSignedRouterRequirements,
    returned: MoeRouterSignedReturnProtocol,
    sequence: MoeCompileSequence,
) -> MoeRouterSourceNativeProtocol:
    score.validate_against(sequence)
    returned.validate_against(score, sequence)
    if returned != build_moe_router_signed_return_protocol(score, sequence):
        raise SchemaError("router native routes must match real expert RETURN source",
                          path="moe_router_native_protocol")
    paths = {(path.step, path.layer, path.source_rank): path
             for path in score.paths if path.routes}
    pairs = []
    for placement in returned.placements:
        source = paths.get((placement.step, placement.layer,
                            placement.source_rank))
        if source is None:
            raise SchemaError("zero-work rank cannot invent score/expert work",
                              path="moe_router_native_protocol")
        routes = tuple(MoeRouterSignedRoute(
            route.token_index, route.source_rank,
            route.selected_expert, route.expert_home_rank,
            route.expert_slot_index) for route in source.routes)
        groups = tuple(MoeRouterSignedExpertGroup(
            segment.expert_home_rank, segment.expert_index,
            segment.offset_bytes, segment.size_bytes)
            for segment in placement.segments)
        common = {
            "rank_rows": len(routes), "hidden_size": source.hidden_size,
            "expert_count": source.expert_count,
            "source_rank": source.source_rank,
            "routes": routes, "expert_groups": groups,
        }
        forward = MoeRouterSignedWeightedForwardWorkload.create(
            source_dynamic_case_ref=score.dynamic_score_case_ref,
            source_gate_action_ref=source.gate_action_ref,
            source_weighted_combine_action_ref=
                source.weighted_combine_action_ref,
            score_dtype=DType.FP16, expert_dtype=DType.FP16,
            combined_dtype=DType.FP16, **common,
        )
        backward = MoeRouterSignedScoreBackwardWorkload.create(
            source_weighted_forward_ref=forward.id,
            source_combine_backward_action_ref=
                source.combine_backward_action_ref,
            score_dtype=DType.FP16, expert_dtype=DType.FP16,
            upstream_dtype=DType.FP16, dscore_dtype=DType.FP16,
            dexpert_dtype=DType.FP16, **common,
        )
        if (forward.score_bytes != placement.score_tape_bytes
                or forward.expert_return_bytes != placement.expert_return_bytes
                or forward.combined_bytes != placement.combined_output_bytes
                or backward.operand_bytes != (
                    forward.score_bytes, forward.expert_return_bytes,
                    forward.combined_bytes, forward.score_bytes,
                    forward.expert_return_bytes)):
            raise SchemaError("router native physical input/output bytes must cover every real token",
                              path="moe_router_native_protocol")
        pairs.append(MoeRouterNativeSourcePair(
            placement.step, placement.layer, placement.source_rank,
            forward, backward))
    semantic = {
        "source_dynamic_score_ref": score.id,
        "source_return_protocol_ref": returned.id,
        "source_moe_sequence_ref": sequence.id,
        "version": _VERSION,
        "pairs": tuple(pairs),
    }
    return MoeRouterSourceNativeProtocol(
        stable_artifact_id("moe_router_native_protocol", semantic,
                           schema_version=_VERSION),
        **semantic,
    )


__all__ = ["MoeRouterNativeSourcePair", "MoeRouterSourceNativeProtocol",
           "build_moe_router_native_protocol"]
