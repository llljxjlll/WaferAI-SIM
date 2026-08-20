from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import build_ir0, logical_expand, place_ir0
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.policies.naive_fusion_partition import (
    NaiveFusionPartition,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.schema.action import (
    BarrierScope,
    ChunkDim,
    FusionActionKind,
    FusionPlan,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, GemmPartition
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.persistent_state import (
    canonical_state_staging_value_id,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec
from test_fabric_loader import minimal_hardware


def _partitioned_graph(*, tp: int) -> IR1:
    raw = valid_spec()
    raw["parallel"]["instances"][0].update(tp=tp, sp=tp > 1)  # type: ignore[index]
    if tp == 4:
        raw["model"]["KVH"] = 4  # type: ignore[index]
    spec = from_data(ExperimentSpec, raw, path="spec")
    logical = logical_expand(build_ir0(spec)).entries[0].graph
    fabric = (
        valid_ir1().fabric
        if tp <= 2
        else physical_fabric_from_data(minimal_hardware(2, 2))
    )
    context = PlacementContext.create(
        producer_pass="test_naive_inter_die",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = place_ir0(logical, context)
    skeletons = NaiveFusionPartition().run(placed)
    fields = placed._semantic_key()
    fields["fused_op_skeletons"] = skeletons
    result = IR1.create(producer_pass="fusion_partition", **fields)
    result.validate()
    return result


def _actions(plan: object) -> tuple[object, ...]:
    return tuple(
        action
        for program in getattr(plan, "rank_programs")
        for action in program.actions
    )


def _standalone_all_gather(graph: IR1):
    fused = {
        member
        for skeleton in graph.fused_op_skeletons
        for member in skeleton.member_node_ids
    }
    return next(
        node
        for node in graph.nodes
        if node.id not in fused
        and getattr(node.workload, "collective", None) is CollectiveKind.ALL_GATHER
    )


class NaiveInterDiePolicyTest(unittest.TestCase):
    def test_sequence_parallel_gemm_cannot_enter_inter_die_fusion(self) -> None:
        graph = _partitioned_graph(tp=2)
        fused_op = graph.fused_op_skeletons[0]
        gemm = next(
            node for node in graph.nodes if node.id == fused_op.member_node_ids[0]
        )
        logical_m, logical_n, logical_k = gemm.workload.logical_shape
        replacement = replace(
            gemm,
            workload=replace(
                gemm.workload,
                partition=GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT,
                rank_shape=(logical_m // 2, logical_n, logical_k),
            ),
        )
        fields = graph._semantic_key()
        fields["nodes"] = tuple(
            replacement if node.id == gemm.id else node for node in graph.nodes
        )
        forged = IR1.create(producer_pass=graph.producer_pass, **fields)
        with self.assertRaisesRegex(SchemaError, "cannot enter inter-die"):
            NaiveInterDiePolicy().plan(forged, fused_op, forged.profile)

    def test_tp2_tp4_fused_counts_flops_routes_bytes_and_determinism(self) -> None:
        policy = NaiveInterDiePolicy()
        for tp, expected_actions in ((2, 12), (4, 56)):
            with self.subTest(tp=tp):
                graph = _partitioned_graph(tp=tp)
                fused_op = graph.fused_op_skeletons[0]
                source_digest = canonical_digest(graph)
                plan = policy.plan(graph, fused_op, graph.profile)
                self.assertEqual(policy.plan(graph, fused_op, graph.profile), plan)
                self.assertEqual(plan.producer_pass, "inter_die_plan")
                self.assertEqual(len(plan.chunk_slices), tp)
                self.assertEqual(
                    tuple(chunk.owner_rank for chunk in plan.chunk_slices),
                    tuple(range(tp)),
                )
                self.assertEqual(sum(len(item.actions) for item in plan.rank_programs), expected_actions)
                self.assertTrue(all(len(item.actions) == 4 * tp - 2 for item in plan.rank_programs))

                actions = _actions(plan)
                kinds = Counter(action.kind for action in actions)
                self.assertEqual(kinds[FusionActionKind.COMP], tp * tp)
                self.assertEqual(kinds[FusionActionKind.SEND], tp * (tp - 1))
                self.assertEqual(kinds[FusionActionKind.RECV], tp * (tp - 1))
                self.assertEqual(kinds[FusionActionKind.WAIT], tp * (tp - 1))
                self.assertEqual(kinds[FusionActionKind.REDUCE], tp)
                self.assertEqual(len({action.id for action in actions}), len(actions))
                self.assertEqual(
                    len({action.sync.completion_event for action in actions}),
                    len(actions),
                )

                gemm = next(
                    node for node in graph.nodes if node.id == fused_op.member_node_ids[0]
                )
                rank_local_flops = sum(
                    2 * math.prod(action.compute.workload.rank_shape)
                    for action in actions
                    if action.kind is FusionActionKind.COMP
                    and action.compute is not None
                )
                self.assertEqual(
                    rank_local_flops,
                    2 * math.prod(gemm.workload.logical_shape),
                )

                group = next(item for item in graph.groups if item.id == plan.group_ref)
                manifest = graph.persistent_state_manifest
                self.assertIsNotNone(manifest)
                assert manifest is not None
                declarations = {item.id: item for item in manifest.declarations}
                _, full_n, _ = gemm.workload.logical_shape
                rank_k = gemm.workload.rank_shape[2]
                for program in plan.rank_programs:
                    matching_accesses = tuple(
                        access
                        for access in graph.state_accesses
                        if access.node_ref == gemm.id
                        and access.rank == program.rank
                        and declarations[access.state_ref].identity.tensor_ref
                        == gemm.inputs[1]
                    )
                    self.assertEqual(len(matching_accesses), 1)
                    expected_rhs = canonical_state_staging_value_id(
                        matching_accesses[0].id
                    )
                    comps = tuple(
                        action
                        for action in program.actions
                        if action.kind is FusionActionKind.COMP
                    )
                    self.assertEqual(len(comps), tp)
                    self.assertEqual(
                        {action.reads[1] for action in comps},
                        {expected_rhs},
                    )
                    self.assertEqual(len({action.reads[0] for action in comps}), tp)
                    self.assertEqual(len({action.writes[0] for action in comps}), tp)
                    placement = next(
                        item
                        for item in group.placements
                        if item.rank == program.rank
                    )
                    for action in comps:
                        assert action.compute is not None
                        assert action.compute.tile is not None
                        rhs_slice = action.compute.tile.input_slices[1]
                        self.assertEqual(rhs_slice.operand_id, expected_rhs)
                        self.assertEqual(rhs_slice.source_value_id, gemm.inputs[1])
                        self.assertEqual(
                            rhs_slice.logical_offset,
                            (placement.logical_coord[0] * rank_k, 0),
                        )
                        self.assertEqual(rhs_slice.logical_shape, (rank_k, full_n))
                routes = {
                    (route.source_rank, route.destination_rank): route.die_path
                    for route in group.embedding.routes
                }
                channels = {
                    action.logical_channel
                    for action in actions
                    if action.kind is FusionActionKind.SEND
                }
                self.assertEqual(len(channels), tp * (tp - 1))
                for action in actions:
                    if action.kind not in (FusionActionKind.SEND, FusionActionKind.RECV):
                        continue
                    program_rank = next(
                        program.rank
                        for program in plan.rank_programs
                        if action in program.actions
                    )
                    source, destination = (
                        (program_rank, action.peer_rank)
                        if action.kind is FusionActionKind.SEND
                        else (action.peer_rank, program_rank)
                    )
                    self.assertEqual(action.expected_route, routes[(source, destination)])
                    endpoint = f".rank.{source}.to.rank.{destination}"
                    self.assertEqual(action.logical_channel.count(endpoint), 1)
                self.assertEqual(
                    sum(
                        action.bytes
                        for action in actions
                        if action.kind is FusionActionKind.SEND
                    ),
                    (tp - 1) * sum(chunk.bytes for chunk in plan.chunk_slices),
                )
                plan.validate_against(graph)
                self.assertEqual(canonical_digest(graph), source_digest)

    def test_stateful_fused_rhs_identity_and_slice_fail_closed(self) -> None:
        graph = _partitioned_graph(tp=2)
        plan = NaiveInterDiePolicy().plan(
            graph, graph.fused_op_skeletons[0], graph.profile
        )
        program = plan.rank_programs[0]
        comp_index = next(
            index
            for index, action in enumerate(program.actions)
            if action.kind is FusionActionKind.COMP
        )
        comp = program.actions[comp_index]
        assert comp.compute is not None and comp.compute.tile is not None

        def rebuild(*, rhs_id: str, rhs_offset: tuple[int, ...]) -> FusionPlan:
            rhs_operand = replace(comp.compute.inputs[1], value_id=rhs_id)
            rhs_slice = replace(
                comp.compute.tile.input_slices[1],
                operand_id=rhs_id,
                logical_offset=rhs_offset,
            )
            compute = replace(
                comp.compute,
                inputs=(comp.compute.inputs[0], rhs_operand),
                tile=replace(
                    comp.compute.tile,
                    input_slices=(
                        comp.compute.tile.input_slices[0],
                        rhs_slice,
                    ),
                ),
            )
            forged_action = replace(
                comp,
                reads=(comp.reads[0], rhs_id),
                compute=compute,
            )
            actions = (
                program.actions[:comp_index]
                + (forged_action,)
                + program.actions[comp_index + 1 :]
            )
            changed = replace(
                plan,
                rank_programs=(
                    replace(program, actions=actions),
                    *plan.rank_programs[1:],
                ),
            )
            return FusionPlan.create(
                producer_pass=plan.producer_pass,
                **changed._semantic_key(),
            )

        original_rhs = comp.reads[1]
        original_offset = comp.compute.tile.input_slices[1].logical_offset
        forged_id = rebuild(rhs_id="forged.rhs", rhs_offset=original_offset)
        with self.assertRaisesRegex(SchemaError, "canonical state staging"):
            forged_id.validate_against(graph)

        forged_slice = rebuild(rhs_id=original_rhs, rhs_offset=(1, 0))
        with self.assertRaisesRegex(SchemaError, "exact chunk-local GEMM"):
            forged_slice.validate_against(graph)

    def test_fused_reduction_rank_order_and_lineage_are_functional_oracle(self) -> None:
        graph = _partitioned_graph(tp=4)
        plan = NaiveInterDiePolicy().plan(
            graph,
            graph.fused_op_skeletons[0],
            graph.profile,
        )
        ranks = tuple(range(4))
        for chunk in plan.chunk_slices:
            owner_program = plan.rank_programs[chunk.owner_rank]
            reduce = next(
                action
                for action in owner_program.actions
                if action.kind is FusionActionKind.REDUCE
                and action.chunk_id == chunk.chunk_id
            )
            self.assertEqual(reduce.reduction.input_ranks, ranks)
            expected_reads = []
            expected_deps = []
            for rank in ranks:
                if rank == chunk.owner_rank:
                    comp = next(
                        action
                        for action in owner_program.actions
                        if action.kind is FusionActionKind.COMP
                        and action.chunk_id == chunk.chunk_id
                    )
                    expected_reads.append(comp.writes[0])
                    expected_deps.append(comp.id)
                else:
                    recv = next(
                        action
                        for action in owner_program.actions
                        if action.kind is FusionActionKind.RECV
                        and action.chunk_id == chunk.chunk_id
                        and action.peer_rank == rank
                    )
                    wait = next(
                        action
                        for action in owner_program.actions
                        if action.kind is FusionActionKind.WAIT
                        and action.chunk_id == chunk.chunk_id
                        and action.sync.wait_event == recv.sync.completion_event
                    )
                    self.assertEqual(wait.deps, (recv.id,))
                    expected_reads.append(recv.writes[0])
                    expected_deps.append(wait.id)
            self.assertEqual(reduce.reads, tuple(expected_reads))
            self.assertEqual(reduce.deps, tuple(expected_deps))

    def test_tp2_tp4_all_gather_counts_routes_barrier_and_rank_coverage(self) -> None:
        policy = DirectAllGatherPolicy()
        for tp, expected_actions in ((2, 8), (4, 32)):
            with self.subTest(tp=tp):
                graph = _partitioned_graph(tp=tp)
                op = _standalone_all_gather(graph)
                plan = policy.plan(graph, op, graph.profile)
                self.assertEqual(policy.plan(graph, op, graph.profile), plan)
                self.assertEqual(plan.producer_pass, "inter_die_plan")
                self.assertEqual(len(plan.chunk_slices), tp)
                self.assertEqual(sum(len(item.actions) for item in plan.rank_programs), expected_actions)
                self.assertTrue(all(len(item.actions) == 2 * tp for item in plan.rank_programs))

                actions = _actions(plan)
                kinds = Counter(action.kind for action in actions)
                self.assertEqual(kinds[FusionActionKind.LOCAL_COPY], tp)
                self.assertEqual(kinds[FusionActionKind.SEND], tp * (tp - 1))
                self.assertEqual(kinds[FusionActionKind.RECV], tp * (tp - 1))
                self.assertEqual(kinds[FusionActionKind.BARRIER], tp)
                barriers = [
                    action for action in actions if action.kind is FusionActionKind.BARRIER
                ]
                self.assertEqual(
                    {action.sync.barrier for action in barriers},
                    {barriers[0].sync.barrier},
                )
                self.assertIs(barriers[0].sync.barrier.scope, BarrierScope.PLAN)

                group = next(item for item in graph.groups if item.id == plan.group_ref)
                routes = {
                    (route.source_rank, route.destination_rank): route.die_path
                    for route in group.embedding.routes
                }
                for program in plan.rank_programs:
                    placed = tuple(
                        action
                        for action in program.actions
                        if action.kind
                        in (FusionActionKind.LOCAL_COPY, FusionActionKind.RECV)
                    )
                    self.assertEqual(
                        tuple(action.chunk_id for action in placed),
                        tuple(range(tp)),
                    )
                    barrier = program.actions[-1]
                    self.assertEqual(barrier.deps, tuple(action.id for action in placed))
                    for action in program.actions:
                        if action.kind not in (FusionActionKind.SEND, FusionActionKind.RECV):
                            continue
                        source, destination = (
                            (program.rank, action.peer_rank)
                            if action.kind is FusionActionKind.SEND
                            else (action.peer_rank, program.rank)
                        )
                        self.assertEqual(action.expected_route, routes[(source, destination)])
                self.assertEqual(
                    sum(
                        action.bytes
                        for action in actions
                        if action.kind is FusionActionKind.SEND
                    ),
                    (tp - 1) * sum(chunk.bytes for chunk in plan.chunk_slices),
                )
                self.assertIn(plan.chunk_dim, (ChunkDim.M, ChunkDim.N))
                plan.validate_against(graph)

    def test_tp1_and_unsupported_inputs_fail_closed(self) -> None:
        tp1 = _partitioned_graph(tp=1)
        self.assertEqual(tp1.fusion_candidates, ())
        self.assertEqual(tp1.fused_op_skeletons, ())

        graph = _partitioned_graph(tp=2)
        fused_op = graph.fused_op_skeletons[0]
        with self.assertRaisesRegex(SchemaError, "skeleton in IR1"):
            NaiveInterDiePolicy().plan(
                graph,
                replace(fused_op, id=f"{fused_op.id}.unknown"),
                graph.profile,
            )
        with self.assertRaisesRegex(SchemaError, "exactly match IR1 profile"):
            NaiveInterDiePolicy().plan(
                graph,
                fused_op,
                replace(graph.profile, kv_pages=graph.profile.kv_pages + 1),
            )
        reduce_scatter = next(
            node
            for node in graph.nodes
            if getattr(node.workload, "collective", None)
            is CollectiveKind.REDUCE_SCATTER
        )
        with self.assertRaisesRegex(SchemaError, "only AllGather"):
            DirectAllGatherPolicy().plan(graph, reduce_scatter, graph.profile)


if __name__ == "__main__":
    unittest.main()
