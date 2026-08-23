from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import build_ir0, logical_expand, place_ir0
from llm.frontend.wafer_frontend.policies.naive_fusion_partition import (
    NaiveFusionPartition,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern, GemmPartition, OpPhase
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec


def _graph(*, tp: int = 2, layers: int = 1) -> IR1:
    raw = valid_spec()
    raw["parallel"]["instances"][0].update(tp=tp, sp=tp > 1)  # type: ignore[index]
    raw["model"]["L"] = layers  # type: ignore[index]
    spec = from_data(ExperimentSpec, raw, path="spec")
    logical = logical_expand(build_ir0(spec)).entries[0].graph
    fabric = valid_ir1().fabric
    context = PlacementContext.create(
        producer_pass="test_naive_fusion_partition",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    return place_ir0(logical, context)


def _rebuild(graph: IR1, **updates: object) -> IR1:
    fields: dict[str, object] = {
        "producer_pass": graph.producer_pass,
        "source_ir0_id": graph.source_ir0_id,
        "profile": graph.profile,
        "fabric": graph.fabric,
        "instances": graph.instances,
        "groups": graph.groups,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "fused_op_skeletons": graph.fused_op_skeletons,
        "cross_routes": graph.cross_routes,
    }
    fields.update(updates)
    return IR1.create(**fields)  # type: ignore[arg-type]


def _replace_node(graph: IR1, node_id: str, replacement: object) -> IR1:
    return _rebuild(
        graph,
        nodes=tuple(
            replacement if node.id == node_id else node for node in graph.nodes
        ),
    )


def _boundaries(graph: IR1, members: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    member_set = set(members)
    inputs = tuple(
        value.id
        for value in graph.values
        if member_set.intersection(value.consumers)
        and value.producer not in member_set
    )
    outputs = tuple(
        value.id
        for value in graph.values
        if value.producer in member_set
        and (
            not value.consumers
            or any(consumer not in member_set for consumer in value.consumers)
        )
    )
    return inputs, outputs


def _gemm_rs_candidates(graph: IR1):
    return tuple(
        candidate
        for candidate in graph.fusion_candidates
        if candidate.semantic_contract.pattern is FusionPattern.GEMM_RS
    )


class NaiveFusionPartitionTest(unittest.TestCase):
    def test_tp1_tp2_and_l2_select_all_candidates_in_source_order(self) -> None:
        policy = NaiveFusionPartition()
        for tp, layers, expected_count in ((1, 1, 0), (2, 1, 2), (2, 2, 4)):
            with self.subTest(tp=tp, layers=layers):
                graph = _graph(tp=tp, layers=layers)
                source_digest = canonical_digest(graph)
                result = policy.run(graph)
                self.assertEqual(len(result), expected_count)
                selected = _gemm_rs_candidates(graph)
                self.assertEqual(
                    tuple(skeleton.fusion_ref for skeleton in result),
                    tuple(candidate.id for candidate in selected),
                )
                for candidate, skeleton in zip(selected, result, strict=True):
                    self.assertEqual(skeleton.member_node_ids, candidate.members)
                    self.assertEqual(skeleton.boundary_inputs, candidate.boundary_inputs)
                    self.assertEqual(skeleton.boundary_outputs, candidate.boundary_outputs)
                    self.assertEqual(skeleton.semantic_contract, candidate.semantic_contract)
                    self.assertEqual(skeleton.impl.value, "none")
                self.assertEqual(policy.run(graph), result)
                self.assertEqual(canonical_digest(graph), source_digest)

    def test_skeleton_identity_is_profile_independent(self) -> None:
        graph = _graph()
        other_profile = replace(graph.profile, kv_pages=graph.profile.kv_pages + 1)
        other = _rebuild(graph, profile=other_profile)
        self.assertNotEqual(graph.id, other.id)
        self.assertEqual(
            tuple(item.id for item in NaiveFusionPartition().run(graph)),
            tuple(item.id for item in NaiveFusionPartition().run(other)),
        )

    def test_overlapping_candidates_fail_closed(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        duplicate = replace(
            candidate,
            id=f"{candidate.id}.overlap",
        )
        forged = _rebuild(
            graph,
            fusion_candidates=(candidate, duplicate),
        )
        with self.assertRaisesRegex(SchemaError, "share member"):
            NaiveFusionPartition().run(forged)

    def test_member_stage_and_phase_scope_fail_closed(self) -> None:
        graph = _graph()
        rs_id = _gemm_rs_candidates(graph)[0].members[1]
        rs = next(node for node in graph.nodes if node.id == rs_id)
        for replacement in (
            replace(rs, stage=rs.stage + 1),
            replace(rs, phase=OpPhase.DGRAD),
        ):
            with self.subTest(stage=replacement.stage, phase=replacement.phase):
                with self.assertRaisesRegex(SchemaError, "share instance, stage, phase"):
                    NaiveFusionPartition().run(
                        _replace_node(graph, rs.id, replacement)
                    )

    def test_non_direct_members_fail_closed(self) -> None:
        graph = _graph()
        first, second = _gemm_rs_candidates(graph)
        members = (first.members[0], second.members[1])
        boundary_inputs, boundary_outputs = _boundaries(graph, members)
        disconnected = replace(
            first,
            id=f"{first.id}.non_direct",
            members=members,
            boundary_inputs=boundary_inputs,
            boundary_outputs=boundary_outputs,
        )
        forged = _rebuild(graph, fusion_candidates=(disconnected,))
        with self.assertRaisesRegex(SchemaError, "direct GEMM-to-ReduceScatter DATA edge"):
            NaiveFusionPartition().run(forged)

    def test_sequence_parallel_gemm_cannot_enter_fusion_candidate(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        gemm = next(node for node in graph.nodes if node.id == candidate.members[0])
        logical_m, logical_n, logical_k = gemm.workload.logical_shape
        replacement = replace(
            gemm,
            workload=replace(
                gemm.workload,
                partition=GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT,
                rank_shape=(logical_m // 2, logical_n, logical_k),
            ),
        )
        forged = _replace_node(graph, gemm.id, replacement)
        with self.assertRaisesRegex(SchemaError, "cannot enter a fusion candidate"):
            NaiveFusionPartition().run(forged)

    def test_boundary_and_contract_must_remain_exact(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        cases = (
            (
                replace(
                    candidate,
                    boundary_inputs=tuple(reversed(candidate.boundary_inputs)),
                ),
                "boundary_inputs",
            ),
            (
                replace(
                    candidate,
                    semantic_contract=replace(
                        candidate.semantic_contract,
                        tile_domain=("N", "M"),
                    ),
                ),
                "tile_domain",
            ),
        )
        for changed, message in cases:
            with self.subTest(message=message):
                forged = _rebuild(
                    graph,
                    fusion_candidates=(changed,),
                )
                with self.assertRaisesRegex(SchemaError, message):
                    NaiveFusionPartition().run(forged)


if __name__ == "__main__":
    unittest.main()
