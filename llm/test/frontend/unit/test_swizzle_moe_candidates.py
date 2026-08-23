from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_comet_mesh import (
    build_comet_mesh_moe_candidate_grid,
    build_comet_mesh_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidate,
    build_direct_xy_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    build_moe_action_owner_map,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe import (
    MoeActionWitness,
    MoePacketWitness,
    MoeRankProgram,
    MoeSwizzleCandidate,
    MoeTileWitness,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


def _semantic(value: object) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _rebuild_with_packet(
    candidate: MoeSwizzleCandidate,
    packet_index: int,
    replacement: MoePacketWitness,
) -> MoeSwizzleCandidate:
    old_packet = candidate.packetization[packet_index]
    packet_map = {old_packet.id: replacement.id}
    packets = list(candidate.packetization)
    packets[packet_index] = replacement
    new_packet_index = {item.id: item for item in packets}
    old_actions = {
        item.id: item
        for program in candidate.rank_programs
        for item in program.actions
    }
    action_map = {}
    rebuilt = {}
    while len(rebuilt) != len(old_actions):
        progressed = False
        for old_id, action in old_actions.items():
            if old_id in rebuilt or any(ref not in action_map for ref in action.deps):
                continue
            semantic = _semantic(action)
            semantic["deps"] = tuple(action_map[ref] for ref in action.deps)
            if action.packet_ref in packet_map:
                packet = new_packet_index[packet_map[action.packet_ref]]
                semantic.update(
                    packet_ref=packet.id,
                    stage=packet.stage,
                    pivot_rank=packet.pivot_rank,
                    assignment_refs=tuple(item.assignment_ref for item in packet.slices),
                )
                if action.kind is SwizzleActionKind.SEND:
                    semantic.update(
                        rank=packet.source_rank,
                        route_ref=packet.route_ref,
                        peer_rank=packet.destination_rank,
                        logical_bytes=packet.logical_bytes,
                    )
                elif action.kind is SwizzleActionKind.RECV:
                    semantic.update(
                        rank=packet.destination_rank,
                        route_ref=packet.route_ref,
                        peer_rank=packet.source_rank,
                        logical_bytes=packet.logical_bytes,
                    )
                else:
                    semantic.update(rank=packet.destination_rank)
            new_action = MoeActionWitness.create(**semantic)
            rebuilt[old_id] = new_action
            action_map[old_id] = new_action.id
            progressed = True
        if not progressed:
            raise AssertionError("test helper could not rebuild action DAG")
    rank_programs = tuple(
        MoeRankProgram(
            rank=rank,
            actions=tuple(
                rebuilt[item.id]
                for program in candidate.rank_programs
                for item in program.actions
                if rebuilt[item.id].rank == rank
            ),
        )
        for rank in range(len(candidate.rank_programs))
    )
    tiles = tuple(
        MoeTileWitness.create(
            **{
                **_semantic(tile),
                "required_packet_refs": tuple(
                    packet_map.get(ref, ref) for ref in tile.required_packet_refs
                ),
            }
        )
        for tile in candidate.tile_schedule
    )
    return MoeSwizzleCandidate.create(
        **{
            **_semantic(candidate),
            "packetization": tuple(packets),
            "rank_programs": rank_programs,
            "tile_schedule": tiles,
        }
    )


def _candidate_with_actions(candidate, actions):
    programs = tuple(
        MoeRankProgram(
            rank=rank,
            actions=tuple(item for item in actions if item.rank == rank),
        )
        for rank in range(len(candidate.rank_programs))
    )
    return MoeSwizzleCandidate.create(
        **{**_semantic(candidate), "rank_programs": programs}
    )


def _candidate_without_dependency(candidate, target_ref, dependency_ref):
    old_actions = {
        item.id: item
        for program in candidate.rank_programs
        for item in program.actions
    }
    action_map = {}
    rebuilt = {}
    while len(rebuilt) != len(old_actions):
        progressed = False
        for old_id, action in old_actions.items():
            if old_id in rebuilt or any(ref not in action_map for ref in action.deps):
                continue
            semantic = _semantic(action)
            semantic["deps"] = tuple(
                action_map[ref]
                for ref in action.deps
                if not (old_id == target_ref and ref == dependency_ref)
            )
            new_action = MoeActionWitness.create(**semantic)
            rebuilt[old_id] = new_action
            action_map[old_id] = new_action.id
            progressed = True
        if not progressed:
            raise AssertionError("test helper could not rebuild dependency DAG")
    programs = tuple(
        MoeRankProgram(
            rank=rank,
            actions=tuple(
                rebuilt[item.id]
                for program in candidate.rank_programs
                for item in program.actions
                if rebuilt[item.id].rank == rank
            ),
        )
        for rank in range(len(candidate.rank_programs))
    )
    return MoeSwizzleCandidate.create(
        **{**_semantic(candidate), "rank_programs": programs}
    )


class MoePersonalizedCandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = []
        for case in build_moe_swizzle_scale_cases()[:2]:
            execution = build_moe_swizzle_execution(
                case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
            )
            for region in discover_moe_swizzle_regions(case.spec, case.oracle, execution):
                problem = build_moe_swizzle_problem(
                    region, case.spec, case.oracle, execution,
                    hardware_facts=case.hardware_facts,
                    endpoint_session_contract=case.endpoint_session_contract,
                )
                direct = build_direct_xy_moe_candidate(
                    problem, case.spec, case.oracle, execution
                )
                comets = build_comet_mesh_moe_candidates(
                    problem, case.spec, case.oracle, execution
                )
                cls.rows.append((case, problem, direct, comets))

    def test_direct_xy_dispatch_arrival_and_combine_n_block(self) -> None:
        for case, problem, direct, _ in self.rows:
            tokens = case.spec.tokens
            remote = sum(
                item.source_rank != item.expert_rank
                for item in problem.region.semantic_witness.traffic.assignments
            )
            self.assertIs(
                direct.algorithm, SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A
            )
            self.assertEqual(
                len(direct.packetization),
                6
                * (1 if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 2),
            )
            self.assertEqual(
                {item.assignment_ref for packet in direct.packetization for item in packet.slices},
                set(problem.region.assignment_refs) - {
                    item.id
                    for item in problem.region.semantic_witness.traffic.assignments
                    if item.source_rank == item.expert_rank
                },
            )
            if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM:
                self.assertEqual(
                    Counter({
                        arrival: sum(len(item.assignment_refs) for item in direct.tile_schedule if item.arrival_class == arrival)
                        for arrival in (0, 1, 2)
                    }),
                    Counter({0: tokens // 4, 1: tokens // 2, 2: tokens // 4}),
                )
                self.assertEqual({item.n_block_index for item in direct.tile_schedule}, {None})
            else:
                self.assertEqual({item.n_block_index for item in direct.tile_schedule}, {0})
                self.assertEqual(
                    {ref for item in direct.tile_schedule for ref in item.assignment_refs},
                    set(problem.region.assignment_refs),
                )
                self.assertEqual(
                    {(item.output_column_offset, item.output_column_extent) for item in direct.tile_schedule},
                    {(0, 16)},
                )
            direct.validate_against(problem)

    def test_c4_skewed_capacity_is_candidate_discoverable(self) -> None:
        case = build_moe_swizzle_scale_cases()[4]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
        )
        self.assertTrue(execution.execution_ready)
        self.assertTrue(execution.capacity.admitted)
        for region in discover_moe_swizzle_regions(
            case.spec, case.oracle, execution
        ):
            problem = build_moe_swizzle_problem(
                region, case.spec, case.oracle, execution,
                hardware_facts=case.hardware_facts,
                endpoint_session_contract=case.endpoint_session_contract,
            )
            direct = build_direct_xy_moe_candidate(
                problem, case.spec, case.oracle, execution
            )
            direct.validate_against(problem)
            comets = build_comet_mesh_moe_candidates(
                problem, case.spec, case.oracle, execution
            )
            self.assertTrue(comets)
            for candidate in comets:
                candidate.validate_against(problem)

    def test_comet_local_one_stage_two_stage_and_action_dag(self) -> None:
        for case, problem, _, comets in self.rows:
            self.assertEqual(
                tuple((item.unroll_degree, item.double_buffer) for item in comets),
                ((1, False), (2, True)),
            )
            comet = comets[0]
            tokens = case.spec.tokens
            remote = sum(
                item.source_rank != item.expert_rank
                for item in problem.region.semantic_witness.traffic.assignments
            )
            two_stage = tokens // 4
            self.assertIs(
                comet.algorithm, SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A
            )
            self.assertEqual(
                len(comet.packetization),
                8
                * (1 if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 2),
            )
            self.assertEqual(
                Counter(item.stage for item in comet.packetization),
                Counter({
                    0: 4 * (1 if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 2),
                    1: 4 * (1 if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 2),
                }),
            )
            self.assertEqual(
                Counter({
                    arrival: sum(len(item.assignment_refs) for item in comet.tile_schedule if item.arrival_class == arrival)
                    for arrival in (0, 1, 2)
                }),
                Counter({
                    0: tokens // 4,
                    1: tokens // 2,
                    2: tokens // 4,
                }),
            )
            actions = tuple(
                item for program in comet.rank_programs for item in program.actions
            )
            self.assertEqual(
                Counter(item.kind for item in actions),
                Counter(
                    {
                        SwizzleActionKind.COMP: (2 * len(comet.tile_schedule) if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else len(comet.tile_schedule)),
                        SwizzleActionKind.SWIGLU: (len(comet.tile_schedule) if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 0),
                        SwizzleActionKind.SEND: len(comet.packetization),
                        SwizzleActionKind.RECV: len(comet.packetization),
                        SwizzleActionKind.WAIT: len(comet.packetization),
                    }
                ),
            )
            if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM:
                self.assertEqual({item.n_block_index for item in comet.tile_schedule}, {None})
            else:
                self.assertEqual({item.n_block_index for item in comet.tile_schedule}, {0})
            comet.validate_against(problem)

    def test_route_pivot_slice_and_stage_tamper_fail_closed(self) -> None:
        _, direct_problem, direct, _ = self.rows[0]
        packet = direct.packetization[0]
        wrong_route = next(
            item.id
            for item in direct_problem.topology.group.routes
            if item.id != packet.route_ref
        )
        forged = MoePacketWitness.create(
            **{**_semantic(packet), "route_ref": wrong_route}
        )
        with self.assertRaises(SchemaError):
            _rebuild_with_packet(direct, 0, forged).validate_against(direct_problem)

        _, comet_problem, _, comets = self.rows[0]
        comet = comets[0]
        slice_counts = Counter(
            slice_.assignment_ref
            for item in comet.packetization
            for slice_ in item.slices
        )
        diagonal_index = next(
            index
            for index, item in enumerate(comet.packetization)
            if item.stage == 0
            and any(slice_counts[slice_.assignment_ref] == 2 for slice_ in item.slices)
        )
        diagonal = comet.packetization[diagonal_index]
        for replacement in (
            MoePacketWitness.create(
                **{**_semantic(diagonal), "pivot_rank": diagonal.source_rank}
            ),
            MoePacketWitness.create(
                **{
                    **_semantic(diagonal),
                    "slices": (
                        replace(
                            diagonal.slices[0],
                            destination_offset_bytes=diagonal.slices[0].destination_offset_bytes + 2,
                        ),
                    ) + diagonal.slices[1:],
                }
            ),
            MoePacketWitness.create(
                **{**_semantic(diagonal), "stage": 2}
            ),
        ):
            with self.assertRaises(SchemaError):
                _rebuild_with_packet(comet, diagonal_index, replacement).validate_against(comet_problem)

    def test_missing_or_third_dispatch_comp_fails_exact_multiplicity(self) -> None:
        _, problem, direct, _ = self.rows[0]
        actions = list(
            item for program in direct.rank_programs for item in program.actions
        )
        comps = [item for item in actions if item.kind is SwizzleActionKind.COMP]
        keeper = comps[0]
        victim = next(
            item
            for item in comps
            if item.rank == keeper.rank
            and item.assignment_refs != keeper.assignment_refs
        )
        merged = MoeActionWitness.create(
            **{
                **_semantic(keeper),
                "original_action_refs": keeper.original_action_refs
                + victim.original_action_refs,
                "flops": keeper.flops + victim.flops,
            }
        )
        missing_actions = tuple(
            merged if item.id == keeper.id else item
            for item in actions
            if item.id != victim.id
        )
        with self.assertRaises(SchemaError):
            _candidate_with_actions(direct, missing_actions).validate_against(problem)

        split_target = comps[0]
        first = MoeActionWitness.create(
            **{**_semantic(split_target), "flops": split_target.flops // 2}
        )
        second = MoeActionWitness.create(
            **{
                **_semantic(split_target),
                "original_action_refs": (),
                "flops": split_target.flops - first.flops,
            }
        )
        third_actions = tuple(
            action
            for item in actions
            for action in (
                (first, second) if item.id == split_target.id else (item,)
            )
        )
        with self.assertRaises(SchemaError):
            _candidate_with_actions(direct, third_actions).validate_against(problem)

    def test_dispatch_grouped_swiglu_is_exact_and_fail_closed(self) -> None:
        _, problem, direct, _ = next(
            row for row in self.rows
            if row[1].region.pattern is FusionPattern.MOE_DISPATCH_GEMM
        )
        actions = tuple(
            item for program in direct.rank_programs for item in program.actions
        )
        swiglus = tuple(
            item for item in actions
            if item.kind is SwizzleActionKind.SWIGLU
        )
        assignments = {
            item.id: item
            for item in problem.region.semantic_witness.traffic.assignments
        }
        self.assertEqual(len(swiglus), len(direct.tile_schedule))
        self.assertEqual(
            Counter(ref for item in swiglus for ref in item.assignment_refs),
            Counter({ref: 1 for ref in problem.region.assignment_refs}),
        )
        by_id = {item.id: item for item in actions}
        for item in swiglus:
            self.assertEqual(
                {by_id[ref].work_role for ref in item.deps}, {"gate", "up"}
            )
            self.assertEqual(
                item.original_action_refs,
                tuple(assignments[ref].swiglu_action_ref for ref in item.assignment_refs),
            )
            self.assertEqual(
                item.logical_bytes,
                len(item.assignment_refs)
                * problem.region.semantic_witness.traffic.expert_gemms[item.expert_index].n
                * 2,
            )
            self.assertEqual(item.packed_value_ref, f"moe.swiglu.{item.tile_index}")
        with self.assertRaisesRegex(SchemaError, "SWIGLU|provenance"):
            _candidate_with_actions(
                direct, tuple(item for item in actions if item.id != swiglus[0].id)
            ).validate_against(problem)
        forged = MoeActionWitness.create(
            **{**_semantic(swiglus[0]), "packed_value_ref": "moe.swiglu.forged"}
        )
        with self.assertRaisesRegex(SchemaError, "SWIGLU|provenance"):
            _candidate_with_actions(
                direct, tuple(forged if item.id == swiglus[0].id else item for item in actions)
            ).validate_against(problem)

    def test_combine_transport_closes_one_full_n_producer(self) -> None:
        case, problem, _, _ = next(
            row
            for row in self.rows
            if row[0].spec.name == "C1"
            and row[1].region.pattern is FusionPattern.MOE_GEMM_COMBINE
        )
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
        )
        candidates = build_direct_xy_moe_candidates(
            problem, case.spec, case.oracle, execution
        )
        t1 = next(
            item for item in candidates
            if item.token_block_size == 4
            and item.transport_output_block_count == 1
        )
        t2 = next(
            item for item in candidates
            if item.token_block_size == 4
            and item.transport_output_block_count == 2
        )
        self.assertEqual(t1.compute_output_block_count, 1)
        self.assertEqual(t2.compute_output_block_count, 1)
        self.assertEqual(len(t1.packetization) * 2, len(t2.packetization))
        self.assertEqual(t1.cost.packet_count * 2, t2.cost.packet_count)
        self.assertEqual(t1.cost.descriptor_count * 2, t2.cost.descriptor_count)
        self.assertEqual(t1.cost.event_count * 2, t2.cost.event_count)
        t1_actions = tuple(
            item for program in t1.rank_programs for item in program.actions
        )
        t2_actions = tuple(
            item for program in t2.rank_programs for item in program.actions
        )
        self.assertEqual(
            sum(item.flops for item in t1_actions if item.kind is SwizzleActionKind.COMP),
            sum(item.flops for item in t2_actions if item.kind is SwizzleActionKind.COMP),
        )
        self.assertEqual(
            sum(item.kind is SwizzleActionKind.COMP for item in t1_actions),
            sum(item.kind is SwizzleActionKind.COMP for item in t2_actions),
        )
        by_id = {item.id: item for item in t1_actions}
        send = next(
            item for item in t1_actions
            if item.kind is SwizzleActionKind.SEND
            and any(by_id[ref].kind is SwizzleActionKind.COMP for ref in item.deps)
        )
        comp_deps = tuple(
            ref for ref in send.deps
            if by_id[ref].kind is SwizzleActionKind.COMP
        )
        self.assertEqual(len(comp_deps), 1)
        self.assertEqual({by_id[ref].n_block_index for ref in comp_deps}, {0})
        with self.assertRaisesRegex(SchemaError, "exact full-N producer"):
            _candidate_without_dependency(
                t1, send.id, comp_deps[0]
            ).validate_against(problem)

        packet = t1.packetization[0]
        partial = MoePacketWitness.create(
            **{
                **_semantic(packet),
                "slices": (
                    replace(packet.slices[0], bytes=packet.slices[0].bytes - 2),
                ) + packet.slices[1:],
                "logical_bytes": packet.logical_bytes - 2,
            }
        )
        with self.assertRaises(SchemaError):
            _rebuild_with_packet(t1, 0, partial).validate_against(problem)
        wrong_mode = MoeSwizzleCandidate.create(
            **{**_semantic(t1), "transport_output_block_count": 2}
        )
        with self.assertRaises(SchemaError):
            wrong_mode.validate_against(problem)

    def test_m_block_grid_shapes_work_and_session_width(self) -> None:
        observed = set()
        reverse_edge_witnessed = False
        for case in build_moe_swizzle_scale_cases()[1:3]:
            execution = build_moe_swizzle_execution(
                case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
            )
            for region in discover_moe_swizzle_regions(
                case.spec, case.oracle, execution
            ):
                problem = build_moe_swizzle_problem(
                    region, case.spec, case.oracle, execution,
                    hardware_facts=case.hardware_facts,
                    endpoint_session_contract=case.endpoint_session_contract,
                )
                candidates = build_direct_xy_moe_candidates(
                    problem, case.spec, case.oracle, execution
                )
                expected_transport_counts = (
                    (1,)
                    if region.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else (2, 1)
                )
                self.assertEqual(
                    tuple(
                        (item.transport_output_block_count, item.token_block_size)
                        for item in candidates
                    ),
                    tuple(
                        (transport_count, m_block)
                        for transport_count in expected_transport_counts
                        for m_block in (8, 4, 2, 1)
                    ),
                )
                self.assertEqual(
                    len({item.id for item in candidates}),
                    4 * len(expected_transport_counts),
                )
                for candidate in candidates:
                    candidate.validate_against(problem)
                    actions = tuple(
                        item
                        for program in candidate.rank_programs
                        for item in program.actions
                    )
                    gemms = problem.region.semantic_witness.traffic.expert_gemms
                    for action in actions:
                        if action.kind is not SwizzleActionKind.COMP:
                            continue
                        n = gemms[action.expert_index].n
                        shape = (len(action.assignment_refs), n, gemms[action.expert_index].k)
                        observed.add(shape)
                        self.assertEqual(
                            action.flops,
                            2 * shape[0] * shape[1] * shape[2],
                        )
                    self.assertEqual(
                        sum(
                            item.flops
                            for item in actions
                            if item.kind is SwizzleActionKind.COMP
                        ),
                        problem.region.semantic_witness.traffic.expert_gemm_flops,
                    )
                    self.assertEqual(
                        candidate.cost.logical_payload_bytes,
                        problem.region.semantic_witness.traffic.logical_payload_bytes,
                    )

                    if candidate.token_block_size not in (2, 4, 8):
                        continue
                    by_id = {item.id: item for item in actions}
                    ancestor_cache = {}

                    def ancestors(ref):
                        if ref not in ancestor_cache:
                            result = set(by_id[ref].deps)
                            for dependency in by_id[ref].deps:
                                result.update(ancestors(dependency))
                            ancestor_cache[ref] = result
                        return ancestor_cache[ref]

                    owners = build_moe_action_owner_map(problem, actions)
                    triples = {}
                    for action in actions:
                        if action.packet_ref is not None:
                            triples.setdefault(
                                (action.packet_ref, action.stage), {}
                            )[action.kind] = action
                    endpoints_by_core = {}
                    for triple in triples.values():
                        wait = triple[SwizzleActionKind.WAIT]
                        for kind in (
                            SwizzleActionKind.SEND,
                            SwizzleActionKind.RECV,
                        ):
                            endpoint = triple[kind]
                            endpoints_by_core.setdefault(
                                owners[endpoint.id].runtime_core_id, []
                            ).append((endpoint, wait))
                    for endpoints in endpoints_by_core.values():
                        adjacency = {
                            left: tuple(
                                right
                                for right, (endpoint, _) in enumerate(endpoints)
                                if left != right
                                and endpoints[left][1].id in ancestors(endpoint.id)
                            )
                            for left in range(len(endpoints))
                        }
                        matched = {}

                        def augment(left, seen):
                            for right in adjacency[left]:
                                if right in seen:
                                    continue
                                seen.add(right)
                                if right not in matched or augment(matched[right], seen):
                                    matched[right] = left
                                    return True
                            return False

                        matching = sum(
                            augment(left, set()) for left in range(len(endpoints))
                        )
                        self.assertLessEqual(
                            len(endpoints) - matching,
                            problem.endpoint_session_capacity,
                        )
                    if (
                        case.spec.name == "C2"
                        and region.pattern is FusionPattern.MOE_DISPATCH_GEMM
                        and candidate.token_block_size == 2
                    ):
                        for triple in triples.values():
                            recv = triple[SwizzleActionKind.RECV]
                            if recv.pipeline_index != 0:
                                continue
                            core = owners[recv.id].runtime_core_id
                            later_waits = tuple(
                                other[SwizzleActionKind.WAIT]
                                for other in triples.values()
                                if other[SwizzleActionKind.WAIT].pipeline_index == 4
                                and owners[
                                    other[SwizzleActionKind.RECV].id
                                ].runtime_core_id == core
                            )
                            if any(recv.id in ancestors(wait.id) for wait in later_waits):
                                self.assertTrue(
                                    all(wait.id not in recv.deps for wait in later_waits)
                                )
                                reverse_edge_witnessed = True
                                break
        self.assertEqual(
            observed,
            {
                (1, 32, 16), (2, 32, 16),
                (4, 32, 16), (8, 32, 16),
                (1, 16, 32), (2, 16, 32),
                (4, 16, 32), (8, 16, 32),
            },
        )
        self.assertTrue(reverse_edge_witnessed)

    def test_comet_m_block_and_unroll_are_orthogonal(self) -> None:
        case, problem, _, _ = next(
            row for row in self.rows if row[0].spec.name == "C1"
        )
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
        )
        grid = build_comet_mesh_moe_candidate_grid(
            problem, case.spec, case.oracle, execution
        )
        self.assertEqual(
            tuple(
                (item.token_block_size, item.unroll_degree, item.double_buffer)
                for item in grid
            ),
            tuple(
                (m_block, unroll, unroll == 2)
                for m_block in (8, 4, 2, 1)
                for unroll in (1, 2)
            ),
        )
        self.assertEqual(len({item.id for item in grid}), 8)
        m2_u1, m2_u2 = tuple(
            item for item in grid if item.token_block_size == 2
        )
        self.assertFalse(m2_u1.double_buffer)
        self.assertTrue(m2_u2.double_buffer)
        self.assertEqual(
            tuple((item.m_block_index, item.m_block_size) for item in m2_u1.tile_schedule),
            tuple((item.m_block_index, item.m_block_size) for item in m2_u2.tile_schedule),
        )
        tile = m2_u2.tile_schedule[0]
        with self.assertRaises(SchemaError):
            MoeTileWitness.create(
                **{**_semantic(tile), "m_block_size": tile.m_block_size + 1}
            )

    def test_double_buffer_has_two_finite_slots_and_real_overlap(self) -> None:
        for case, problem, _, comets in self.rows:
            u1, u2 = comets
            self.assertNotEqual(
                tuple(tuple(item.id for item in program.actions) for program in u1.rank_programs),
                tuple(tuple(item.id for item in program.actions) for program in u2.rank_programs),
            )
            for program in u2.rank_programs:
                self.assertEqual({item.buffer_slot for item in program.actions}, {0, 1})
                self.assertEqual({item.buffer_family for item in program.actions}, {
                    "dispatch_operand"
                    if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else "combine_output"
                })
            actions = tuple(
                item for program in u2.rank_programs for item in program.actions
            )
            by_id = {item.id: item for item in actions}
            ancestors = {}
            def deps(item):
                if item.id not in ancestors:
                    ancestors[item.id] = set(item.deps)
                    for ref in item.deps:
                        ancestors[item.id].update(deps(by_id[ref]))
                return ancestors[item.id]
            independent = any(
                left.rank == right.rank
                and left.buffer_slot != right.buffer_slot
                and left.id not in deps(right)
                and right.id not in deps(left)
                for left in actions
                for right in actions
            )
            self.assertTrue(independent)

            if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM:
                comps = {}
                packets = {}
                for item in actions:
                    if item.kind is SwizzleActionKind.COMP:
                        comps.setdefault(item.tile_index, []).append(item)
                    if item.packet_ref is not None:
                        packets[(item.packet_ref, item.kind)] = item
                domains = {}
                for tile in u2.tile_schedule:
                    comp = comps[tile.tile_index][0]
                    domains.setdefault((comp.rank, comp.buffer_slot), []).append(tile)
                    pair = comps[tile.tile_index]
                    self.assertEqual(len(pair), 2)
                    self.assertNotIn(pair[0].id, pair[1].deps)
                    self.assertNotIn(pair[1].id, pair[0].deps)
                witnessed_reuse = False
                for tiles in domains.values():
                    tiles.sort(key=lambda item: comps[item.tile_index][0].pipeline_index)
                    for previous, current in zip(tiles, tiles[1:]):
                        readers = {item.id for item in comps[previous.tile_index]}
                        targets = [
                            packets[(ref, SwizzleActionKind.RECV)]
                            for ref in current.required_packet_refs
                        ] or comps[current.tile_index]
                        self.assertTrue(all(readers.issubset(set(item.deps)) for item in targets))
                        witnessed_reuse = True

                possible_reuse = any(len(items) > 1 for items in domains.values())
                if possible_reuse:
                    self.assertTrue(witnessed_reuse)

            referenced = {ref for item in actions for ref in item.deps}
            target = next(item for item in reversed(actions) if item.id not in referenced)
            forged = MoeActionWitness.create(
                **{**_semantic(target), "buffer_slot": 1 - target.buffer_slot}
            )
            tampered_actions = tuple(
                forged if item.id == target.id else item for item in actions
            )
            with self.assertRaisesRegex(SchemaError, "pipeline/family/physical-slot"):
                _candidate_with_actions(u2, tampered_actions).validate_against(problem)


if __name__ == "__main__":
    unittest.main()
