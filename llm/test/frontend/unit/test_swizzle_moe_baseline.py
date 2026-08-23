from __future__ import annotations

from collections import Counter
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import build_moe_action_owner_map
from llm.frontend.wafer_frontend.schema.swizzle_moe_placement import (
    build_moe_action_dynamic_root_bindings,
    build_moe_action_dynamic_root_keys,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe import (
    MoeActionWitness,
    MoeRankProgram,
    MoeSwizzleCandidate,
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


class MoeExecutableUnfusedBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = []
        for case in build_moe_swizzle_scale_cases()[:4]:
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
                baseline = build_executable_moe_unfused_baseline(
                    problem, case.spec, case.oracle, execution
                )
                cls.rows.append((case, execution, problem, baseline))

    def test_c0_c3_exact_action_and_work_closure(self) -> None:
        for case, _, problem, baseline in self.rows:
            traffic = problem.region.semantic_witness.traffic
            tokens = case.spec.tokens
            remote = len(case.oracle.remote_token_indices)
            actions = tuple(
                action
                for program in baseline.rank_programs
                for action in program.actions
            )
            comp_count = (
                2 * tokens
                if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
                else tokens
            )
            self.assertIs(baseline.algorithm, SwizzleAlgorithm.UNFUSED)
            self.assertEqual(tuple(item.rank for item in baseline.rank_programs), (0, 1, 2, 3))
            self.assertTrue(all(item.actions for item in baseline.rank_programs))
            self.assertEqual(
                len(actions),
                comp_count + 3 * remote
                + (tokens if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 0),
            )
            self.assertEqual(
                Counter(item.kind for item in actions),
                Counter(
                    {
                        SwizzleActionKind.COMP: comp_count,
                        SwizzleActionKind.SWIGLU: (
                            tokens
                            if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
                            else 0
                        ),
                        SwizzleActionKind.SEND: remote,
                        SwizzleActionKind.RECV: remote,
                        SwizzleActionKind.WAIT: remote,
                    }
                ),
            )
            self.assertEqual(
                sum(item.logical_bytes for item in actions if item.kind is SwizzleActionKind.SEND),
                traffic.logical_payload_bytes,
            )
            self.assertEqual(
                sum(item.flops for item in actions if item.kind is SwizzleActionKind.COMP),
                traffic.expert_gemm_flops,
            )
            self.assertEqual(
                baseline.cost.region_boundary_output_bytes,
                traffic.region_boundary_output_bytes,
            )
            self.assertEqual(set(baseline.original_action_refs), set(problem.region.member_refs))
            self.assertEqual(len(baseline.original_action_refs), len(actions))

    def test_transport_triples_and_peer_waves_are_exact(self) -> None:
        for _, _, problem, baseline in self.rows:
            actions = tuple(
                action
                for program in baseline.rank_programs
                for action in program.actions
            )
            grouped = {}
            for action in actions:
                if action.kind in (
                    SwizzleActionKind.SEND,
                    SwizzleActionKind.RECV,
                    SwizzleActionKind.WAIT,
                ):
                    grouped.setdefault(action.packet_ref, []).append(action)
            lane_count = problem.endpoint_session_capacity
            owners = build_moe_action_owner_map(problem, actions)
            for packet_ref, triple in grouped.items():
                by_kind = {item.kind: item for item in triple}
                send, recv, wait = (
                    by_kind[SwizzleActionKind.SEND],
                    by_kind[SwizzleActionKind.RECV],
                    by_kind[SwizzleActionKind.WAIT],
                )
                self.assertEqual(len(triple), 3)
                self.assertEqual(set(wait.deps), {send.id, recv.id})
                self.assertEqual(
                    (send.packet_ref, recv.packet_ref, wait.packet_ref),
                    (packet_ref, packet_ref, packet_ref),
                )
            by_id = {item.id: item for item in actions}
            ancestors_by_id = {}
            def ancestors(item):
                if item.id not in ancestors_by_id:
                    result = set(item.deps)
                    for ref in item.deps:
                        result.update(ancestors(by_id[ref]))
                    ancestors_by_id[item.id] = result
                return ancestors_by_id[item.id]
            endpoints_by_core = {}
            wait_endpoint_cores = {
                next(
                    item for item in triple
                    if item.kind is SwizzleActionKind.WAIT
                ).id: {
                    owners[item.id].runtime_core_id
                    for item in triple
                    if item.kind in (
                        SwizzleActionKind.SEND, SwizzleActionKind.RECV,
                    )
                }
                for triple in grouped.values()
            }
            for triple in grouped.values():
                by_kind = {item.kind: item for item in triple}
                send = by_kind[SwizzleActionKind.SEND]
                recv = by_kind[SwizzleActionKind.RECV]
                wait = by_kind[SwizzleActionKind.WAIT]
                for endpoint in (send, recv):
                    core = owners[endpoint.id].runtime_core_id
                    endpoints_by_core.setdefault(core, []).append((endpoint, wait))
                    for ref in endpoint.deps:
                        if by_id[ref].kind is SwizzleActionKind.WAIT:
                            self.assertIn(core, wait_endpoint_cores[ref])
            for endpoints in endpoints_by_core.values():
                adjacency = {
                    left: tuple(
                        right
                        for right, (endpoint, _) in enumerate(endpoints)
                        if left != right
                        and endpoints[left][1].id in ancestors(endpoint)
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
                self.assertLessEqual(len(endpoints) - matching, lane_count)
            cores_by_rank = {
                rank: {owners[item.id].runtime_core_id for item in actions if item.rank == rank and item.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)}
                for rank in range(len(problem.topology.group.placements))
            }
            self.assertTrue(any(len(cores) >= 2 for cores in cores_by_rank.values()))
            baseline.validate_against(problem)

    def test_dynamic_root_family_slot_witness_is_exact(self) -> None:
        expected_counts = (8, 6, 8, 6, 8, 6, 8, 6)
        for (_, _, problem, baseline), expected_count in zip(
            self.rows, expected_counts, strict=True,
        ):
            actions = tuple(
                action
                for program in baseline.rank_programs
                for action in program.actions
            )
            bindings = build_moe_action_dynamic_root_bindings(
                problem, baseline.pattern, baseline.algorithm, actions,
            )
            roots = build_moe_action_dynamic_root_keys(
                problem, baseline.pattern, baseline.algorithm, actions,
            )
            self.assertEqual(len(roots), expected_count)
            self.assertEqual(
                set(roots), {root for _, root in bindings},
            )
            self.assertEqual(
                {root[1] for root in roots},
                {
                    "dispatch_operand"
                    if baseline.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else "combine_output"
                },
            )
            self.assertEqual({root[2] for root in roots}, {0})

        _, _, problem, baseline = self.rows[0]
        actions = tuple(
            action for program in baseline.rank_programs for action in program.actions
        )
        owner_ref = build_moe_action_dynamic_root_bindings(
            problem, baseline.pattern, baseline.algorithm, actions,
        )[0][0]
        owner = next(action for action in actions if action.id == owner_ref)
        missing = MoeActionWitness.create(
            **{
                **_semantic(owner),
                "pipeline_index": None,
                "buffer_slot": None,
                "buffer_family": None,
            }
        )
        with self.assertRaisesRegex(SchemaError, "dynamic owner lacks"):
            build_moe_action_dynamic_root_bindings(
                problem, baseline.pattern, baseline.algorithm,
                tuple(missing if action.id == owner.id else action for action in actions),
            )
        non_owner = next(
            action for action in actions
            if action.buffer_family is None and action.tile_index is not None
        )
        forged = MoeActionWitness.create(
            **{
                **_semantic(non_owner),
                "pipeline_index": non_owner.tile_index,
                "buffer_slot": 0,
                "buffer_family": "dispatch_operand",
            }
        )
        with self.assertRaisesRegex(SchemaError, "non-owner claims"):
            build_moe_action_dynamic_root_bindings(
                problem, baseline.pattern, baseline.algorithm,
                tuple(forged if action.id == non_owner.id else action for action in actions),
            )

    def test_wrong_wait_dependency_fails_candidate_level_rebuild(self) -> None:
        _, _, problem, baseline = self.rows[1]
        programs = list(baseline.rank_programs)
        referenced = {ref for program in programs for item in program.actions for ref in item.deps}
        program_index, wait_index = next(
            (program_index, action_index)
            for program_index, program in enumerate(programs)
            for action_index, item in enumerate(program.actions)
            if item.kind is SwizzleActionKind.WAIT and item.id not in referenced
        )
        actions = list(programs[program_index].actions)
        wait = actions[wait_index]
        actions[wait_index] = MoeActionWitness.create(
            **{**_semantic(wait), "deps": (wait.deps[1],)}
        )
        programs[program_index] = MoeRankProgram(
            rank=programs[program_index].rank,
            actions=tuple(actions),
        )
        tampered = MoeSwizzleCandidate.create(
            **{**_semantic(baseline), "rank_programs": tuple(programs)}
        )
        with self.assertRaisesRegex(SchemaError, "transport triple"):
            tampered.validate_against(problem)


if __name__ == "__main__":
    unittest.main()
