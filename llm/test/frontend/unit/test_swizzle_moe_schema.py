from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.problem import build_swizzle_problem
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleGroupView,
    SwizzleRankPlacement,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe import (
    MoeTrafficScenarioKind,
    MoeFusionRegion,
    MoeSwizzleProblem,
    MoeTopologyWitness,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_scale import MoeSwizzleScaleRole
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)

from _fixtures import valid_ir1
from test_swizzle_schema import _profile


class MoeSwizzleSchemaDiscoveryProblemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_moe_swizzle_scale_cases()
        cls.executions = tuple(
            build_moe_swizzle_execution(
                case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
            )
            for case in cls.cases
        )
        cls.regions = tuple(
            discover_moe_swizzle_regions(case.spec, case.oracle, execution)
            for case, execution in zip(
                cls.cases, cls.executions, strict=True
            )
        )
        cls.problems = tuple(
            tuple(
                build_moe_swizzle_problem(
                    region, case.spec, case.oracle, execution,
                    hardware_facts=case.hardware_facts,
                    endpoint_session_contract=case.endpoint_session_contract,
                )
                for region in regions
            )
            for case, execution, regions in zip(
                cls.cases, cls.executions, cls.regions, strict=True
            )
        )

    def test_c0_c4_regions_scale_from_execution_truth(self) -> None:
        for case, regions in zip(self.cases, self.regions, strict=True):
            dispatch, combine = regions
            tokens = case.spec.tokens
            remote = len(case.oracle.remote_token_indices)
            self.assertEqual(
                (dispatch.pattern, combine.pattern),
                (
                    FusionPattern.MOE_DISPATCH_GEMM,
                    FusionPattern.MOE_GEMM_COMBINE,
                ),
            )
            self.assertEqual(
                tuple(len(item.member_refs) for item in regions),
                (3 * tokens + 3 * remote, tokens + 3 * remote),
            )
            self.assertEqual(
                tuple(
                    (len(item.boundary_input_refs), len(item.boundary_output_refs))
                    for item in regions
                ),
                ((3 * tokens, tokens), (2 * tokens, tokens)),
            )
            for region, flops, boundary_bytes in zip(
                regions,
                (4 * tokens * 16 * 32, 2 * tokens * 16 * 32),
                (tokens * 32 * 2, tokens * 16 * 2),
                strict=True,
            ):
                traffic = region.semantic_witness.traffic
                self.assertEqual(
                    (
                        len(traffic.assignments),
                        len(traffic.expert_gemms),
                        len(traffic.pair_routes),
                        traffic.logical_payload_bytes,
                        traffic.expert_gemm_flops,
                        traffic.region_boundary_output_bytes,
                    ),
                    (tokens, 4, 12, remote * 32, flops, boundary_bytes),
                )
                self.assertEqual(region.assignment_refs, tuple(item.id for item in traffic.assignments))
                self.assertEqual(region.trace_digest, case.oracle.trace_digest)

    def test_calibration_validation_and_capacity_roles_are_exact(self) -> None:
        self.assertEqual(
            tuple(case.spec.role for case in self.cases),
            (MoeSwizzleScaleRole.CONTROL, MoeSwizzleScaleRole.CALIBRATION,
             MoeSwizzleScaleRole.VALIDATION, MoeSwizzleScaleRole.VALIDATION,
             MoeSwizzleScaleRole.CAPACITY),
        )
        self.assertEqual(
            tuple(pair[0].scale_role for pair in self.problems),
            tuple(case.spec.role for case in self.cases),
        )

    def test_c0_serde_rebuild_and_c4_capacity_probe_is_validation_plannable(self) -> None:
        case = self.cases[0]
        execution = self.executions[0]
        rebuilt = discover_moe_swizzle_regions(case.spec, case.oracle, execution)
        self.assertEqual(rebuilt, self.regions[0])
        self.assertEqual(
            loads_dataclass(
                MoeFusionRegion,
                canonical_json(rebuilt[0]),
                path="region",
            ),
            rebuilt[0],
        )
        c4_regions = discover_moe_swizzle_regions(
            self.cases[4].spec,
            self.cases[4].oracle,
            self.executions[4],
        )
        self.assertEqual(len(c4_regions), 2)
        self.assertTrue(self.executions[4].execution_ready)
        self.assertTrue(self.executions[4].capacity.admitted)
        with self.assertRaises(SchemaError):
            replace(rebuilt[0], trace_digest="0" * 64).validate()

    def test_problem_2x2_and_nonrectangle_direct_only(self) -> None:
        for problem_pair in self.problems:
            for problem in problem_pair:
                problem.validate()
                self.assertTrue(problem.topology.complete_rectangle)
                self.assertIn(
                    SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A,
                    problem.allowed_algorithms,
                )
                scenarios = {item.kind: item for item in problem.traffic_scenarios}
                actual = scenarios[MoeTrafficScenarioKind.ACTUAL]
                capacity = scenarios[MoeTrafficScenarioKind.CAPACITY]
                p95 = scenarios[MoeTrafficScenarioKind.P95]
                self.assertTrue(actual.executable_binding)
                self.assertFalse(capacity.executable_binding)
                self.assertGreaterEqual(capacity.logical_payload_bytes, actual.logical_payload_bytes)
                self.assertGreaterEqual(p95.logical_payload_bytes, actual.logical_payload_bytes)
                self.assertLessEqual(p95.logical_payload_bytes, capacity.logical_payload_bytes)
        problem = self.problems[0][0]
        group = problem.topology.group
        line_group = SwizzleGroupView(
            group_ref=group.group_ref,
            logical_shape=(1, 4),
            placements=tuple(
                SwizzleRankPlacement(rank=rank, x=rank, y=0) for rank in range(4)
            ),
            routes=group.routes,
        )
        line_topology = MoeTopologyWitness(
            group=line_group,
            row_orders=((0, 1, 2, 3),),
            column_orders=((0,), (1,), (2,), (3,)),
            pivot_by_pair=tuple(
                (source, destination, destination)
                for source in range(4)
                for destination in range(4)
                if source != destination
            ),
            complete_rectangle=False,
        )
        direct_only = MoeSwizzleProblem.create(
            source_execution_id=problem.source_execution_id,
            scale_name=problem.scale_name,
            scale_role=problem.scale_role,
            region=problem.region,
            topology=line_topology,
            traffic_scenarios=problem.traffic_scenarios,
            allowed_algorithms=(
                SwizzleAlgorithm.UNFUSED,
                SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A,
            ),
            hardware_facts=problem.hardware_facts,
            endpoint_session_contract=problem.endpoint_session_contract,
            max_candidates=problem.max_candidates,
        )
        direct_only.validate()
        with self.assertRaises(SchemaError):
            replace(
                problem.topology,
                group=replace(group, routes=group.routes[:-1]),
            ).validate()

    def test_hardware_session_and_p95_tamper_fail_closed(self) -> None:
        case = self.cases[0]
        facts = case.hardware_facts
        cores = [list(items) for items in facts.ordered_cores_by_die]
        cores[0][0] = replace(cores[0][0], runtime_core_id=cores[0][1].runtime_core_id)
        with self.assertRaisesRegex(SchemaError, "runtime core ids"):
            replace(facts, ordered_cores_by_die=tuple(tuple(items) for items in cores)).validate()
        resources = list(facts.route_resources)
        resources[0] = replace(resources[0], bytes_per_cycle=0)
        with self.assertRaisesRegex(SchemaError, "bandwidth"):
            replace(facts, route_resources=tuple(resources)).validate()
        with self.assertRaisesRegex(SchemaError, "endpoint session contract"):
            replace(case.endpoint_session_contract, capacity_per_core=4).validate()
        problem = self.problems[0][0]
        scenarios = list(problem.traffic_scenarios)
        p95_index = next(index for index, item in enumerate(scenarios) if item.kind is MoeTrafficScenarioKind.P95)
        scenarios[p95_index] = replace(scenarios[p95_index], logical_payload_bytes=scenarios[p95_index].logical_payload_bytes + 1)
        with self.assertRaisesRegex(SchemaError, "P95 must equal ACTUAL"):
            replace(problem, traffic_scenarios=tuple(scenarios)).validate()

    def test_dense_and_moe_patterns_are_bidirectionally_closed(self) -> None:
        with self.assertRaises(SchemaError):
            replace(self.regions[0][0], pattern=FusionPattern.GEMM_RS).validate()
        ir1 = valid_ir1()
        dense = build_swizzle_problem(
            ir1,
            ir1.fused_op_skeletons[0],
            _profile(),
            SwizzleConstraints(
                allowed_algorithms=(SwizzleAlgorithm.UNFUSED,),
                max_candidates=32,
                max_actions=512,
                max_buffers=64,
                max_chunk_count=32,
                allow_unroll_two=True,
            ),
        )
        with self.assertRaisesRegex(SchemaError, "MoeSwizzleProblem"):
            replace(dense, pattern=FusionPattern.MOE_GEMM_COMBINE).validate()


if __name__ == "__main__":
    unittest.main()
