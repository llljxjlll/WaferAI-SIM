from __future__ import annotations

from dataclasses import replace

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
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
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    build_moe_swizzle_pair_cost_context,
    decide_moe_swizzle,
    estimate_moe_swizzle_materialized_pair_cycles,
    select_moe_swizzle_workload_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidate,
    build_direct_xy_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleDecisionReason,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe import MoeSwizzleCost
from llm.frontend.wafer_frontend.schema.swizzle_moe_placement import (
    MoeEndpointCoreWidth,
    MoeStorageIntervalColorDepth,
    MoeWholeCoreLifecycleCount,
    MoeWholePairFeasibility,
    MoeWholePairPlacementFeasibility,
    MoeWholePairPlacementReason,
    MoeWorkloadEndpointFeasibility,
    build_moe_candidate_core_lifecycle_floor,
    build_moe_candidate_dynamic_root_keys,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
    MoeCalibrationKind, MoeCalibrationSample, MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


def _measured_profile():
    digests = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    samples = []
    for shape in MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES:
        cycles = 5 + (2 * shape[0] * shape[1] * shape[2]) // 256
        for sample in range(3):
            for repeat in range(2):
                samples.append(MoeCalibrationSample(
                    MoeCalibrationKind.GROUP_GEMM, sample, repeat, cycles,
                    shape, DType.FP16, *digests,
                ))
    for shape in MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES:
        cycles = 3 + shape[2] // 32
        for sample in range(3):
            for repeat in range(2):
                samples.append(MoeCalibrationSample(
                    MoeCalibrationKind.SWIGLU_GROUP, sample, repeat, cycles,
                    shape, DType.FP16, *digests,
                ))
    fixed = tuple(
        item for item in MoeCalibrationKind
        if item not in (
            MoeCalibrationKind.GROUP_GEMM, MoeCalibrationKind.SWIGLU_GROUP,
        )
    )
    for index, kind in enumerate(fixed):
        for sample in range(3):
            for repeat in range(2):
                samples.append(MoeCalibrationSample(
                    kind, sample, repeat, index + 1, None, None, *digests,
                ))
    profile = MoeSwizzleCalibrationProfile.create(
        samples=tuple(samples), tool_sha256=digests[0],
        hardware_sha256=digests[1], simulation_sha256=digests[2],
        mapping_sha256=digests[3],
    )
    if profile.status is not MoeCalibrationStatus.MEASURED or len(profile.samples) != 168:
        raise AssertionError("synthetic profile must preserve exact 168-sample quotient")
    return profile


def _whole_pair_witnesses(
    dispatch, combine, *, blocked_pairs=(), placement_blocked_pairs=(),
):
    blocked = set(blocked_pairs)
    placement_blocked = set(placement_blocked_pairs)
    first_core = dispatch.problem.hardware_facts.ordered_cores_by_die[0][0]
    floor_by_candidate = {
        candidate.id: build_moe_candidate_core_lifecycle_floor(
            decision.problem, candidate,
        )
        for decision in (dispatch, combine)
        for candidate in decision.ranked_candidates
    }
    witnesses = []
    for dispatch_candidate in dispatch.ranked_candidates:
        for combine_candidate in combine.ranked_candidates:
            pair = (dispatch_candidate.id, combine_candidate.id)
            if pair in placement_blocked:
                witnesses.append(MoeWholePairFeasibility.create(
                    candidate_refs=pair,
                    placement=MoeWholePairPlacementFeasibility.create(
                        workload_projection_id="unit.workload",
                        replacement_projection_id="unit.replacement",
                        placement_digest="e" * 64,
                        reason=MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE,
                        failure_path="projection.values",
                        failure_message="value crosses cores without an explicit LOCAL_COPY",
                        feasible=False,
                    ),
                    endpoint=None,
                    workload_abi_id=None,
                    dynamic_root_keys=(),
                    dynamic_sram_high_water_bytes=0,
                    dynamic_sram_capacity_bytes=dispatch.problem.sram_capacity_bytes,
                    storage_color_depths=(),
                    core_lifecycle_counts=(),
                    whole_physical_root_count=0,
                    whole_alloc_count=0,
                    whole_free_count=0,
                    feasible=False,
                ))
                continue
            floor_by_core = {}
            for candidate in (dispatch_candidate, combine_candidate):
                for item in floor_by_candidate[candidate.id]:
                    counts = floor_by_core.setdefault(
                        item.runtime_core_id, [0, 0, 0],
                    )
                    counts[0] += item.alloc_count
                    counts[1] += item.bind_count
                    counts[2] += item.free_count
            core_lifecycle_counts = tuple(
                MoeWholeCoreLifecycleCount(core, *counts)
                for core, counts in sorted(floor_by_core.items())
            )
            whole_alloc_count = sum(
                item.alloc_count for item in core_lifecycle_counts
            )
            whole_free_count = sum(
                item.free_count for item in core_lifecycle_counts
            )
            peak = 4 if pair in blocked else 1
            endpoint = MoeWorkloadEndpointFeasibility.create(
                workload_projection_id="unit.workload",
                replacement_projection_id="unit.replacement",
                placement_digest="e" * 64,
                widths=(MoeEndpointCoreWidth(
                    logical_core=first_core.logical_core,
                    runtime_core_id=first_core.runtime_core_id,
                    session_action_refs=tuple(f"session.{index}" for index in range(peak)),
                    max_inflight=peak,
                ),),
                capacity_per_core=dispatch.problem.endpoint_session_capacity,
                peak_width=peak,
                feasible=peak <= dispatch.problem.endpoint_session_capacity,
            )
            witnesses.append(MoeWholePairFeasibility.create(
                candidate_refs=pair,
                placement=MoeWholePairPlacementFeasibility.create(
                    workload_projection_id="unit.workload",
                    replacement_projection_id="unit.replacement",
                    placement_digest="e" * 64,
                    reason=MoeWholePairPlacementReason.ADMITTED,
                    failure_path=None, failure_message=None, feasible=True,
                ),
                endpoint=endpoint,
                workload_abi_id="unit.workload_abi",
                dynamic_root_keys=((first_core.runtime_core_id, "unit.dynamic", 0),),
                dynamic_sram_high_water_bytes=1,
                dynamic_sram_capacity_bytes=dispatch.problem.sram_capacity_bytes,
                storage_color_depths=(MoeStorageIntervalColorDepth(
                    first_core.runtime_core_id, "swiglu_output", 1,
                ),),
                core_lifecycle_counts=core_lifecycle_counts,
                whole_physical_root_count=whole_alloc_count,
                whole_alloc_count=whole_alloc_count,
                whole_free_count=whole_free_count,
                feasible=endpoint.feasible,
            ))
    return tuple(witnesses)


class MoePersonalizedCostDecisionTest(unittest.TestCase):
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
                baseline = build_executable_moe_unfused_baseline(
                    problem, case.spec, case.oracle, execution
                )
                direct = build_direct_xy_moe_candidate(
                    problem, case.spec, case.oracle, execution
                )
                comets = build_comet_mesh_moe_candidates(
                    problem, case.spec, case.oracle, execution
                )
                cls.rows.append((problem, baseline, direct, comets))

    def test_all_algorithms_use_exact_same_work(self) -> None:
        for problem, baseline, direct, comets in self.rows:
            traffic = problem.region.semantic_witness.traffic
            for candidate in (baseline, direct) + comets:
                cost = candidate.cost
                self.assertEqual(
                    (
                        cost.logical_payload_bytes,
                        cost.expert_gemm_flops,
                        cost.region_boundary_output_bytes,
                    ),
                    (
                        traffic.logical_payload_bytes,
                        traffic.expert_gemm_flops,
                        traffic.region_boundary_output_bytes,
                    ),
                )
                self.assertLessEqual(cost.lower_cycles, cost.estimated_cycles)
                self.assertLessEqual(cost.estimated_cycles, cost.upper_cycles)
                self.assertLessEqual(cost.max_inflight, problem.endpoint_session_capacity)
                self.assertFalse(cost.calibrated)
                roots = build_moe_candidate_dynamic_root_keys(problem, candidate)
                self.assertEqual(len(roots), cost.physical_root_count)
                runtime_cores = {
                    core.runtime_core_id
                    for cores in problem.hardware_facts.ordered_cores_by_die
                    for core in cores
                }
                self.assertTrue(roots)
                self.assertTrue(all(root[0] in runtime_cores for root in roots))
                self.assertEqual(
                    {root[1] for root in roots},
                    {
                        "dispatch_operand"
                        if candidate.pattern.value == "moe_dispatch_gemm"
                        else "combine_output"
                    },
                )
            self.assertIs(baseline.algorithm, SwizzleAlgorithm.UNFUSED)
            self.assertGreaterEqual(
                comets[0].cost.packet_count, direct.cost.packet_count
            )

    def test_provisional_cost_always_falls_back_and_is_input_order_stable(self) -> None:
        for problem, baseline, direct, comets in self.rows:
            forward = decide_moe_swizzle(
                problem, baseline, (direct,) + comets
            )
            reverse = decide_moe_swizzle(
                problem, baseline, tuple(reversed((direct,) + comets))
            )
            self.assertEqual(
                tuple(item.id for item in forward.ranked_candidates),
                tuple(item.id for item in reverse.ranked_candidates),
            )
            self.assertEqual(forward.selected_candidate_ref, baseline.id)
            self.assertIs(
                forward.decision_reason,
                SwizzleDecisionReason.NO_PROFITABLE_FUSION,
            )
            self.assertFalse(forward.performance_complete)
            self.assertEqual(forward, reverse)

    def test_measured_profile_ranking_provenance_and_failclosed_inputs(self) -> None:
        profile = _measured_profile()
        case = build_moe_swizzle_scale_cases()[1]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
        )
        expected_winners = (
            (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 2, 1, 1, 115.0),
            (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 2, 1, 1, 84.0),
        )
        for region, expected in zip(
            discover_moe_swizzle_regions(case.spec, case.oracle, execution),
            expected_winners, strict=True,
        ):
            problem = build_moe_swizzle_problem(
                region, case.spec, case.oracle, execution,
                hardware_facts=case.hardware_facts,
                endpoint_session_contract=case.endpoint_session_contract,
            )
            baseline = build_executable_moe_unfused_baseline(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            directs = build_direct_xy_moe_candidates(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            comets = build_comet_mesh_moe_candidate_grid(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            fused = directs + comets
            candidates = (baseline,) + fused
            forward = decide_moe_swizzle(problem, baseline, fused)
            reverse = decide_moe_swizzle(
                problem, baseline, tuple(reversed(fused))
            )
            self.assertTrue(forward.performance_complete)
            self.assertEqual(forward, reverse)
            winner = forward.ranked_candidates[0]
            self.assertEqual(
                (
                    winner.algorithm, winner.token_block_size,
                    winner.unroll_degree, winner.transport_output_block_count,
                    winner.cost.estimated_cycles,
                ),
                expected,
            )
            self.assertIs(
                forward.decision_reason,
                SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES,
            )
            self.assertLess(
                winner.cost.estimated_cycles, baseline.cost.estimated_cycles
            )
            self.assertEqual(
                (winner.cost.prologue_cycles, winner.cost.epilogue_cycles),
                (baseline.cost.prologue_cycles, baseline.cost.epilogue_cycles),
            )
            self.assertLess(
                winner.cost.steady_state_cycles,
                baseline.cost.steady_state_cycles,
            )
            if problem.region.pattern.value == "moe_dispatch_gemm":
                self.assertGreater(baseline.cost.swiglu_group_cycles, 0.0)
                self.assertLess(
                    winner.cost.swiglu_group_cycles,
                    baseline.cost.swiglu_group_cycles,
                )
            else:
                self.assertEqual(baseline.cost.swiglu_group_cycles, 0.0)
                self.assertEqual(winner.cost.swiglu_group_cycles, 0.0)
            for candidate in candidates:
                self.assertTrue(candidate.cost.calibrated)
                self.assertEqual(candidate.cost.calibration_profile_ref, profile.id)
                actual = next(item for item in candidate.cost.scenario_estimates if item.scenario.value == "actual")
                p95 = next(item for item in candidate.cost.scenario_estimates if item.scenario.value == "p95")
                self.assertEqual(actual, replace(p95, scenario=actual.scenario))
            incomplete = MoeSwizzleCalibrationProfile.create(
                samples=profile.samples[:-1], tool_sha256=profile.tool_sha256,
                hardware_sha256=profile.hardware_sha256, simulation_sha256=profile.simulation_sha256,
                mapping_sha256=profile.mapping_sha256,
            )
            with self.assertRaisesRegex(SchemaError, "must be MEASURED"):
                build_executable_moe_unfused_baseline(
                    problem, case.spec, case.oracle, execution,
                    calibration_profile=incomplete,
                )
            with self.assertRaisesRegex(SchemaError, "sample/config SHA drifted"):
                build_direct_xy_moe_candidate(
                    problem, case.spec, case.oracle, execution,
                    calibration_profile=replace(profile, tool_sha256="f" * 64),
                )

    def test_c2_measured_full_grid_selects_both_patterns(self) -> None:
        profile = _measured_profile()
        case = build_moe_swizzle_scale_cases()[2]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
        )
        expected_winners = (
            (1, 2, 1, 146.0),
            (1, 4, 1, 104.0),
        )
        for region, expected in zip(
            discover_moe_swizzle_regions(case.spec, case.oracle, execution),
            expected_winners, strict=True,
        ):
            problem = build_moe_swizzle_problem(
                region, case.spec, case.oracle, execution,
                hardware_facts=case.hardware_facts,
                endpoint_session_contract=case.endpoint_session_contract,
            )
            baseline = build_executable_moe_unfused_baseline(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            fused = build_direct_xy_moe_candidates(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            ) + build_comet_mesh_moe_candidate_grid(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            decision = decide_moe_swizzle(problem, baseline, fused)
            winner = decision.ranked_candidates[0]
            self.assertEqual(
                (
                    winner.transport_output_block_count,
                    winner.token_block_size, winner.unroll_degree,
                    winner.cost.estimated_cycles,
                ),
                expected,
            )
            self.assertIs(
                winner.algorithm, SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A
            )
            self.assertIs(
                decision.decision_reason,
                SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES,
            )
            self.assertLess(
                winner.cost.estimated_cycles, baseline.cost.estimated_cycles
            )
            self.assertEqual(
                (winner.cost.prologue_cycles, winner.cost.epilogue_cycles),
                (baseline.cost.prologue_cycles, baseline.cost.epilogue_cycles),
            )
            self.assertLess(
                winner.cost.steady_state_cycles,
                baseline.cost.steady_state_cycles,
            )

    def test_joint_selection_consumes_complete_whole_witnesses(self) -> None:
        profile = _measured_profile()
        case = build_moe_swizzle_scale_cases()[1]
        execution = build_moe_swizzle_execution(
            case.spec, case.oracle, MoeScaleExecutionMode.INFER_FORWARD
        )
        decisions = []
        for region in discover_moe_swizzle_regions(
            case.spec, case.oracle, execution
        ):
            problem = build_moe_swizzle_problem(
                region, case.spec, case.oracle, execution,
                hardware_facts=case.hardware_facts,
                endpoint_session_contract=case.endpoint_session_contract,
            )
            baseline = build_executable_moe_unfused_baseline(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            fused = build_direct_xy_moe_candidates(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            ) + build_comet_mesh_moe_candidate_grid(
                problem, case.spec, case.oracle, execution,
                calibration_profile=profile,
            )
            decisions.append(decide_moe_swizzle(problem, baseline, fused))
        dispatch, combine = decisions
        independent_pair = (
            dispatch.selected_candidate_ref, combine.selected_candidate_ref,
        )
        all_witnesses = _whole_pair_witnesses(
            dispatch, combine,
            placement_blocked_pairs=(independent_pair,),
        )
        pair_context = build_moe_swizzle_pair_cost_context(
            dispatch, combine,
        )
        pair_bounds = pair_context.bounds
        pair_order = tuple(pair for pair, _ in pair_bounds)
        with self.assertRaisesRegex(SchemaError, "source decision lineage"):
            replace(
                pair_context, source_dispatch_decision_id="forged-decision",
            ).validate_against(dispatch, combine)
        with self.assertRaisesRegex(SchemaError, "bounds disagree"):
            replace(
                pair_context,
                bounds=((pair_context.bounds[0][0], pair_context.bounds[0][1] + 1.0),)
                + pair_context.bounds[1:],
            ).validate_against(dispatch, combine)
        with self.assertRaisesRegex(SchemaError, "lifecycle floors disagree"):
            replace(
                pair_context,
                lifecycle_floor_cycles=(
                    (pair_context.lifecycle_floor_cycles[0][0],
                     pair_context.lifecycle_floor_cycles[0][1] + 1.0),
                ) + pair_context.lifecycle_floor_cycles[1:],
            ).validate_against(dispatch, combine)
        with self.assertRaisesRegex(SchemaError, "nonnegative fixed cycles"):
            replace(
                pair_context,
                lifecycle_fixed_cycles=(-1.0,) + pair_context.lifecycle_fixed_cycles[1:],
            ).validate_against(dispatch, combine)
        witness_by_pair = {
            item.candidate_refs: item for item in all_witnesses
        }
        frontier = []
        best_corrected = float("inf")
        for pair, lower_bound in pair_bounds:
            if lower_bound >= best_corrected:
                break
            frontier.append(pair)
            if witness_by_pair[pair].feasible:
                best_corrected = min(
                    best_corrected,
                    estimate_moe_swizzle_materialized_pair_cycles(
                        dispatch, combine, witness_by_pair[pair],
                        context=pair_context,
                    ),
                )
        # Synthetic feasible witnesses use the exact typed candidate
        # lifecycle floor as their whole lifecycle.  Therefore actual must be
        # exactly the lower bound; lower+whole would expose the old double count.
        exact_floor_pair = next(
            pair for pair in pair_order if witness_by_pair[pair].feasible
        )
        self.assertEqual(
            estimate_moe_swizzle_materialized_pair_cycles(
                dispatch, combine, witness_by_pair[exact_floor_pair],
                context=pair_context,
            ),
            dict(pair_bounds)[exact_floor_pair],
        )
        baseline_pair = (dispatch.baseline.id, combine.baseline.id)
        mode_pairs = {
            (dispatch_ref, combine_ref)
            for dispatch_ref in (baseline_pair[0], independent_pair[0])
            for combine_ref in (baseline_pair[1], independent_pair[1])
        }
        required_pairs = mode_pairs | set(frontier)
        witnesses = tuple(
            item for item in all_witnesses
            if item.candidate_refs in required_pairs
        )
        forward = select_moe_swizzle_workload_deployment(
            dispatch, combine, witnesses
        )
        reverse = select_moe_swizzle_workload_deployment(
            dispatch, combine, tuple(reversed(witnesses))
        )
        self.assertEqual(forward, reverse)
        self.assertNotEqual(
            (forward.selected_dispatch_candidate_ref,
             forward.selected_combine_candidate_ref),
            independent_pair,
        )
        self.assertTrue(forward.performance_complete)
        self.assertIs(
            forward.decision_reason,
            SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES,
        )
        selected = next(
            item for item in witnesses
            if item.candidate_refs == forward.ranked_pair_refs[0]
        )
        self.assertTrue(selected.feasible)
        selected_pair = selected.candidate_refs
        selected_floor = dict(pair_context.lifecycle_floor_cycles)[selected_pair]
        self.assertEqual(
            forward.selected_estimated_cycles,
            dict(pair_bounds)[selected_pair],
        )
        self.assertGreater(selected_floor, 0.0)
        with self.assertRaisesRegex(SchemaError, "cycle sums disagree"):
            replace(
                forward,
                selected_estimated_cycles=(
                    forward.selected_estimated_cycles + selected_floor
                ),
            ).validate()
        self.assertIs(
            witness_by_pair[independent_pair].placement.reason,
            MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE,
        )
        self.assertIsNone(witness_by_pair[independent_pair].endpoint)
        self.assertEqual(forward.lower_bound_pair_ref, pair_order[0])
        self.assertEqual(forward.frontier_pair_refs, tuple(frontier))
        expected_stop = (
            None
            if len(frontier) == len(pair_bounds)
            else pair_bounds[len(frontier)][1]
        )
        self.assertEqual(forward.frontier_stop_lower_bound, expected_stop)
        with self.assertRaisesRegex(SchemaError, "frontier stop lower bound"):
            replace(
                forward,
                frontier_stop_lower_bound=(
                    1.0 if expected_stop is None else expected_stop + 1.0
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "missing exact witness"):
            select_moe_swizzle_workload_deployment(
                dispatch, combine, tuple(
                    item for item in witnesses
                    if item.candidate_refs != frontier[0]
                )
            )
        with self.assertRaisesRegex(SchemaError, "unique candidate keys"):
            select_moe_swizzle_workload_deployment(
                dispatch, combine, witnesses + (witnesses[0],)
            )
        fallback_only = next(pair for pair in mode_pairs if pair not in frontier)
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            select_moe_swizzle_workload_deployment(
                dispatch, combine, tuple(
                    item for item in witnesses
                    if item.candidate_refs != fallback_only
                ),
            )
        outside = next(
            item for item in all_witnesses
            if item.candidate_refs not in required_pairs
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            select_moe_swizzle_workload_deployment(
                dispatch, combine, witnesses + (outside,)
            )
        with self.assertRaisesRegex(SchemaError, "full-grid argmin"):
            replace(
                forward,
                lower_bound_pair_ref=baseline_pair,
            ).validate()
        with self.assertRaisesRegex(SchemaError, "exact cost-ordered prefix"):
            replace(
                forward,
                frontier_pair_refs=tuple(reversed(forward.frontier_pair_refs)),
            ).validate()
        self.assertEqual(
            forward.id,
            select_moe_swizzle_workload_deployment(
                dispatch, combine, witnesses
            ).id,
        )

    def test_unbound_caller_cannot_forge_measured_calibration(self) -> None:
        cost = self.rows[0][1].cost
        semantic = {
            name: getattr(cost, name)
            for name in cost.__dataclass_fields__
            if name not in ("schema_version", "id")
        }
        semantic["calibrated"] = True
        with self.assertRaisesRegex(SchemaError, "None profile must remain PROVISIONAL"):
            MoeSwizzleCost.create(**semantic)

    def test_candidate_and_decision_rebuilds_are_stable(self) -> None:
        problem, baseline, direct, comets = self.rows[0]
        first = decide_moe_swizzle(problem, baseline, (direct,) + comets)
        second = decide_moe_swizzle(problem, baseline, (direct,) + comets)
        self.assertEqual(first.id, second.id)
        self.assertEqual(first, second)
        self.assertNotEqual(comets[0].id, comets[1].id)
        first_roots = build_moe_candidate_dynamic_root_keys(problem, comets[0])
        second_roots = build_moe_candidate_dynamic_root_keys(problem, comets[1])
        self.assertNotEqual(first_roots, second_roots)
        self.assertEqual(len(first_roots), comets[0].cost.physical_root_count)
        self.assertEqual(len(second_roots), comets[1].cost.physical_root_count)


if __name__ == "__main__":
    unittest.main()
